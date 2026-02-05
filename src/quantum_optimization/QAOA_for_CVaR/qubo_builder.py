"""
QUBO Construction Module for QAOA Portfolio CVaR Optimization.

Formulates portfolio optimization as a Quadratic Unconstrained Binary Optimization (QUBO) problem.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy import stats

from .gpu_acceleration import (
    get_array_module,
    to_gpu_array,
    to_cpu_array,
    is_gpu_available,
    should_use_gpu
)


class QUBOBuilder:
    """
    Builds QUBO matrices for CVaR-based portfolio optimization.
    
    Objectives:
    - Maximize expected return
    - Minimize CVaR (tail risk)
    - Maximize diversification (minimize correlation penalty)
    
    Constraints:
    - Budget constraint (fully invested)
    - Cardinality constraint (target number of assets)
    - Long-only constraint (binary variables)
    """
    
    def __init__(
        self,
        expected_returns: np.ndarray,
        scenario_matrix: np.ndarray,
        correlation_matrix: np.ndarray,
        return_weight: float = 0.5,
        risk_weight: float = 1.0,
        diversification_weight: float = 0.1,
        budget_penalty: float = 10.0,
        cardinality_penalty: float = 10.0,
        target_k: int = 5,
        confidence_level: float = 0.95,
        normalization: bool = True,
        use_gpu: bool = False
    ):
        """
        Initialize QUBO builder.
        
        Args:
            expected_returns: Expected returns for each asset
            scenario_matrix: Scenario matrix (num_scenarios x num_assets)
            correlation_matrix: Asset correlation matrix
            return_weight: Weight for return objective
            risk_weight: Weight for risk objective
            diversification_weight: Weight for diversification objective
            budget_penalty: Penalty weight for budget constraint
            cardinality_penalty: Penalty weight for cardinality constraint
            target_k: Target number of assets to select
            confidence_level: CVaR confidence level
            normalization: Whether to normalize objectives
        """
        self.expected_returns = expected_returns
        self.scenario_matrix = scenario_matrix
        self.correlation_matrix = correlation_matrix
        self.return_weight = return_weight
        self.risk_weight = risk_weight
        self.diversification_weight = diversification_weight
        self.budget_penalty = budget_penalty
        self.cardinality_penalty = cardinality_penalty
        self.target_k = target_k
        self.confidence_level = confidence_level
        self.normalization = normalization
        self.use_gpu = use_gpu and is_gpu_available()
        
        self.n = len(expected_returns)
        
        # Normalize if requested
        if normalization:
            self._normalize_objectives()
    
    def _normalize_objectives(self):
        """Normalize objectives using z-score normalization."""
        # Normalize expected returns
        if self.expected_returns.std() > 1e-10:
            self.expected_returns = (self.expected_returns - self.expected_returns.mean()) / self.expected_returns.std()
    
    def _build_return_objective(self) -> np.ndarray:
        """
        Build return maximization objective.
        
        Returns:
            Q matrix for return objective (minimize negative return)
        """
        Q = np.zeros((self.n, self.n))
        
        # Linear terms: -mu_i * x_i (negative because we minimize)
        for i in range(self.n):
            Q[i, i] = -self.return_weight * self.expected_returns[i]
        
        return Q
    
    def _build_cvar_proxy_objective(self) -> np.ndarray:
        """
        Build CVaR proxy objective using scenario-based approximation.
        
        Returns:
            Q matrix for CVaR proxy objective
        """
        # For small matrices (< 50x50), CPU is faster due to GPU overhead
        # Only use GPU for larger problems or batch operations
        use_gpu_here = self.use_gpu and self.n >= 50
        
        xp = get_array_module(use_gpu_here, matrix_size=self.n)
        
        # Compute scenario losses: L_s = -R_s @ w
        # For equal weights: w_i = x_i / sum(x)
        # Approximate CVaR using worst-case scenarios
        
        num_scenarios = self.scenario_matrix.shape[0]
        tail_size = int((1 - self.confidence_level) * num_scenarios)
        
        if tail_size == 0:
            tail_size = 1
        
        # Convert to GPU arrays only if beneficial
        scenario_matrix = to_gpu_array(self.scenario_matrix, use_gpu_here)
        correlation_matrix = to_gpu_array(self.correlation_matrix, use_gpu_here)
        
        # For each asset, compute average loss in worst scenarios
        asset_losses = -scenario_matrix  # Loss = -return
        
        # Sort scenarios by portfolio loss (using equal weights approximation)
        # Use mean loss per asset as proxy
        mean_losses = xp.mean(asset_losses, axis=0)
        
        # Initialize Q matrix on GPU/CPU
        Q = xp.zeros((self.n, self.n))
        
        # Quadratic penalty for high-loss assets
        for i in range(self.n):
            Q[i, i] = self.risk_weight * mean_losses[i]
        
        # Correlation penalty in tail scenarios
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(correlation_matrix[i, j]) > 0.1:
                    Q[i, j] = self.risk_weight * 0.5 * correlation_matrix[i, j]
        
        # Convert back to CPU if needed
        return to_cpu_array(Q)
    
    def _build_diversification_objective(self) -> np.ndarray:
        """
        Build diversification objective (pairwise correlation penalty).
        
        Returns:
            Q matrix for diversification objective
        """
        Q = np.zeros((self.n, self.n))
        
        # Quadratic terms: correlation penalty
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    Q[i, j] = self.diversification_weight * self.correlation_matrix[i, j]
        
        return Q
    
    def _build_budget_constraint(self) -> np.ndarray:
        """
        Build budget constraint penalty.
        
        Constraint: sum(x_i) = target_k (for equal weights)
        
        Returns:
            Q matrix for budget constraint penalty
        """
        Q = np.zeros((self.n, self.n))
        
        # Penalty: P * (sum(x_i) - target_k)^2
        for i in range(self.n):
            for j in range(self.n):
                Q[i, j] += self.budget_penalty
            Q[i, i] -= 2 * self.budget_penalty * self.target_k
        
        return Q
    
    def _build_cardinality_constraint(self) -> np.ndarray:
        """
        Build cardinality constraint penalty.
        
        Constraint: sum(x_i) = target_k
        
        Returns:
            Q matrix for cardinality constraint penalty
        """
        Q = np.zeros((self.n, self.n))
        
        # Penalty: P * (sum(x_i) - target_k)^2
        for i in range(self.n):
            for j in range(self.n):
                Q[i, j] += self.cardinality_penalty
            Q[i, i] -= 2 * self.cardinality_penalty * self.target_k
        
        return Q
    
    def build_qubo(self) -> Tuple[np.ndarray, float]:
        """
        Build complete QUBO matrix.
        
        QUBO form: minimize x^T Q x + c
        where x is binary vector
        
        Returns:
            Tuple of (Q matrix, constant term)
        """
        # For small matrices (< 50x50), CPU is faster due to GPU overhead
        # Only use GPU for larger problems
        use_gpu_here = self.use_gpu and self.n >= 50
        
        # Build all components on CPU (they're small matrices)
        Q = np.zeros((self.n, self.n))
        
        # Add objectives (all CPU for small matrices)
        Q += self._build_return_objective()
        Q += self._build_cvar_proxy_objective()
        Q += self._build_diversification_objective()
        
        # Add constraint penalties
        Q += self._build_budget_constraint()
        Q += self._build_cardinality_constraint()
        
        # Constant term
        constant = self.budget_penalty * self.target_k ** 2 + self.cardinality_penalty * self.target_k ** 2
        
        return Q, constant
    
    def solution_to_weights(self, solution: np.ndarray) -> np.ndarray:
        """
        Convert binary solution to portfolio weights.
        
        Args:
            solution: Binary vector indicating selected assets
            
        Returns:
            Array of portfolio weights (equal weights among selected)
        """
        selected = solution.sum()
        if selected == 0:
            return np.zeros(self.n)
        
        weights = solution / selected
        return weights
