"""
Portfolio Evaluator Module for QAOA CVaR Optimization.

Evaluates portfolios in batch using QAOA-optimized solutions.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import time

from .qubo_builder import QUBOBuilder
from .qaoa_portfolio import run_qaoa_optimization, QAOAResult
from .returns import generate_scenario_matrix
from .precompute_registry import PrecomputeRegistry
from .gpu_acceleration import is_gpu_available


class PortfolioEvaluator:
    """
    Evaluates portfolios using QAOA-based CVaR optimization.
    
    Uses precomputed quantum artifacts per asset set and date.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        registry: PrecomputeRegistry,
        estimation_window: int = 252,
        target_k: List[int] = [5, 10],
        return_weights: List[float] = [0.25, 0.75],
        risk_weights: List[float] = [1.0, 2.0],
        diversification_weights: List[float] = [0.1, 0.5],
        confidence_levels: List[float] = [0.95, 0.99],
        reps_grid: List[int] = [1, 2, 3],
        shots: int = 5000,
        maxiter: int = 250,
        use_gpu: bool = False
    ):
        """
        Initialize portfolio evaluator.
        
        Args:
            returns: DataFrame of returns
            registry: Precompute registry
            estimation_window: Window size for estimation
            target_k: List of target cardinalities
            return_weights: List of return weights
            risk_weights: List of risk weights
            diversification_weights: List of diversification weights
            confidence_levels: List of confidence levels
            reps_grid: List of QAOA reps values
            shots: Number of shots
            maxiter: Maximum optimizer iterations
        """
        self.returns = returns
        self.registry = registry
        self.estimation_window = estimation_window
        self.target_k = target_k
        self.return_weights = return_weights
        self.risk_weights = risk_weights
        self.diversification_weights = diversification_weights
        self.confidence_levels = confidence_levels
        self.reps_grid = reps_grid
        self.shots = shots
        self.maxiter = maxiter
        self.use_gpu = use_gpu and is_gpu_available()
    
    def optimize_portfolio(
        self,
        asset_set: Tuple[str, ...],
        date: pd.Timestamp,
        return_weight: float,
        risk_weight: float,
        diversification_weight: float,
        target_k: int,
        confidence_level: float,
        reps: int
    ) -> QAOAResult:
        """
        Optimize portfolio using QAOA.
        
        Args:
            asset_set: Tuple of asset names
            date: Current date
            return_weight: Return objective weight
            risk_weight: Risk objective weight
            diversification_weight: Diversification objective weight
            target_k: Target cardinality
            confidence_level: CVaR confidence level
            reps: QAOA reps
            
        Returns:
            QAOAResult with optimization results
        """
        # Get or generate scenario matrix
        scenario_matrix = self.registry.get_scenario_matrix(asset_set, date)
        if scenario_matrix is None:
            asset_list = list(asset_set)
            returns_subset = self.returns[asset_list]
            
            # Get window ending at date
            try:
                date_idx = returns_subset.index.get_loc(date)
            except KeyError:
                # Find nearest date
                date_idx = returns_subset.index.searchsorted(date)
                if date_idx >= len(returns_subset):
                    date_idx = len(returns_subset) - 1
            window_start = max(0, date_idx - self.estimation_window + 1)
            window_returns = returns_subset.iloc[window_start:date_idx + 1]
            
            scenario_matrix = generate_scenario_matrix(window_returns, self.estimation_window)
            self.registry.store_scenario_matrix(asset_set, date, scenario_matrix)
        
        # Compute expected returns and correlation
        asset_list = list(asset_set)
        returns_subset = self.returns[asset_list]
        try:
            date_idx = returns_subset.index.get_loc(date)
        except KeyError:
            # Find nearest date
            date_idx = returns_subset.index.searchsorted(date)
            if date_idx >= len(returns_subset):
                date_idx = len(returns_subset) - 1
        window_start = max(0, date_idx - self.estimation_window + 1)
        window_returns = returns_subset.iloc[window_start:date_idx + 1]
        
        expected_returns = window_returns.mean().values
        correlation_matrix = window_returns.corr().values
        
        # Build QUBO
        qubo_builder = QUBOBuilder(
            expected_returns=expected_returns,
            scenario_matrix=scenario_matrix,
            correlation_matrix=correlation_matrix,
            return_weight=return_weight,
            risk_weight=risk_weight,
            diversification_weight=diversification_weight,
            target_k=target_k,
            confidence_level=confidence_level,
            use_gpu=self.use_gpu
        )
        
        Q, constant = qubo_builder.build_qubo()
        
        # Check cache
        weight_config = (return_weight, risk_weight, diversification_weight, target_k, confidence_level)
        cached_result = self.registry.get_qaoa_result(asset_set, date, weight_config, reps)
        
        if cached_result is not None:
            return cached_result
        
        # Run QAOA
        result = run_qaoa_optimization(
            Q,
            constant,
            reps=reps,
            shots=self.shots,
            alpha=confidence_level,
            maxiter=self.maxiter
        )
        
        # Cache result
        self.registry.store_qaoa_result(asset_set, date, weight_config, reps, result)
        
        return result
