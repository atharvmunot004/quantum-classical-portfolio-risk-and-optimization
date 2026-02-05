"""
Portfolio Optimizer Module for QMV Optimization.

Optimizes portfolios using QUBO-based mean-variance optimization.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import time

from .qmv_qubo import QMVQUBOBuilder
from .precompute_registry import PrecomputeRegistry
from .returns import compute_expected_returns, compute_covariance_matrix


class QMVPortfolioOptimizer:
    """
    Optimizes portfolios using QMV (Quantum Mean-Variance) optimization.
    
    Uses precomputed covariance and expected returns per asset set.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        registry: PrecomputeRegistry,
        estimation_window: int = 252,
        bits_per_asset: int = 4,
        weight_step: float = 0.0625,
        max_weight_per_asset: float = 0.25,
        budget_penalty: float = 20.0,
        max_weight_penalty: float = 10.0,
        num_reads: int = 5000,
        random_seed: int = 42
    ):
        """
        Initialize QMV portfolio optimizer.
        
        Args:
            returns: DataFrame of returns
            registry: Precompute registry
            estimation_window: Window size for estimation
            bits_per_asset: Number of bits per asset
            weight_step: Weight step size
            max_weight_per_asset: Maximum weight per asset
            budget_penalty: Budget constraint penalty
            max_weight_penalty: Max weight constraint penalty
            num_reads: Number of reads for solver
            random_seed: Random seed
        """
        self.returns = returns
        self.registry = registry
        self.estimation_window = estimation_window
        self.bits_per_asset = bits_per_asset
        self.weight_step = weight_step
        self.max_weight_per_asset = max_weight_per_asset
        self.budget_penalty = budget_penalty
        self.max_weight_penalty = max_weight_penalty
        self.num_reads = num_reads
        self.random_seed = random_seed
    
    def precompute_asset_set_statistics(
        self,
        asset_set: Tuple[str, ...],
        date: pd.Timestamp
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precompute expected returns and covariance for an asset set.
        
        Args:
            asset_set: Tuple of asset names
            date: Current date
            
        Returns:
            Tuple of (expected_returns, covariance_matrix)
        """
        # Check cache
        expected_returns = self.registry.get_expected_returns(asset_set)
        covariance_matrix = self.registry.get_covariance_matrix(asset_set)
        
        if expected_returns is not None and covariance_matrix is not None:
            return expected_returns, covariance_matrix
        
        # Compute
        asset_list = list(asset_set)
        returns_subset = self.returns[asset_list]
        
        try:
            date_idx = returns_subset.index.get_loc(date)
        except KeyError:
            date_idx = returns_subset.index.searchsorted(date)
            if date_idx >= len(returns_subset):
                date_idx = len(returns_subset) - 1
        
        window_start = max(0, date_idx - self.estimation_window + 1)
        window_returns = returns_subset.iloc[window_start:date_idx + 1]
        
        expected_returns = compute_expected_returns(
            window_returns,
            self.estimation_window,
            annualize=True
        ).values
        
        covariance_matrix = compute_covariance_matrix(
            window_returns,
            self.estimation_window,
            use_shrinkage=True
        ).values
        
        # Store in cache
        self.registry.store_expected_returns(asset_set, expected_returns)
        self.registry.store_covariance_matrix(asset_set, covariance_matrix)
        
        return expected_returns, covariance_matrix
    
    def optimize_portfolio(
        self,
        asset_set: Tuple[str, ...],
        date: pd.Timestamp,
        lambda_risk: float
    ) -> Dict:
        """
        Optimize portfolio using QMV.
        
        Args:
            asset_set: Tuple of asset names
            date: Current date
            lambda_risk: Risk aversion parameter
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Precompute statistics
        expected_returns, covariance_matrix = self.precompute_asset_set_statistics(asset_set, date)
        
        # Check QUBO cache
        qubo_result = self.registry.get_qubo_matrix(asset_set, lambda_risk)
        
        if qubo_result is not None:
            Q, constant = qubo_result
        else:
            # Build QUBO
            qubo_builder = QMVQUBOBuilder(
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                lambda_risk=lambda_risk,
                bits_per_asset=self.bits_per_asset,
                weight_step=self.weight_step,
                max_weight_per_asset=self.max_weight_per_asset,
                budget_penalty=self.budget_penalty,
                max_weight_penalty=self.max_weight_penalty
            )
            
            Q, constant = qubo_builder.build_qubo()
            
            # Cache QUBO
            self.registry.store_qubo_matrix(asset_set, lambda_risk, Q, constant)
        
        # Solve QUBO (using classical fallback)
        solution, energy = self._solve_qubo_classical(Q, constant)
        
        # Decode solution
        qubo_builder = QMVQUBOBuilder(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            lambda_risk=lambda_risk,
            bits_per_asset=self.bits_per_asset,
            weight_step=self.weight_step,
            max_weight_per_asset=self.max_weight_per_asset
        )
        
        weights = qubo_builder.decode_solution(solution)
        
        # Repair solution
        repaired_solution = qubo_builder.repair_solution(solution)
        repaired_weights = qubo_builder.decode_solution(repaired_solution)
        
        runtime_ms = (time.time() - start_time) * 1000
        
        return {
            'asset_set': asset_set,
            'date': date,
            'lambda_risk': lambda_risk,
            'weights': repaired_weights,
            'solution': repaired_solution,
            'energy': energy,
            'runtime_ms': runtime_ms
        }
    
    def _solve_qubo_classical(
        self,
        Q: np.ndarray,
        constant: float
    ) -> Tuple[np.ndarray, float]:
        """
        Solve QUBO using classical optimization (fallback).
        
        Args:
            Q: QUBO matrix
            constant: Constant term
            
        Returns:
            Tuple of (solution, energy)
        """
        np.random.seed(self.random_seed)
        n = Q.shape[0]
        
        best_solution = None
        best_energy = float('inf')
        
        # Random search
        for _ in range(self.num_reads):
            solution = np.random.randint(0, 2, size=n)
            energy = solution.T @ Q @ solution + constant
            
            if energy < best_energy:
                best_energy = energy
                best_solution = solution.copy()
        
        return best_solution, best_energy
