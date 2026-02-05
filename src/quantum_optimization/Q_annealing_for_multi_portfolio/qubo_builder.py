"""
QUBO Construction Module for Multi-Objective Portfolio Optimization.

Formulates portfolio optimization as a Quadratic Unconstrained Binary Optimization (QUBO) problem.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats


class QUBOBuilder:
    """
    Builds QUBO matrices for multi-objective portfolio optimization.
    
    Objectives:
    - Maximize expected return
    - Minimize CVaR (tail risk)
    - Maximize diversification (minimize correlation penalty)
    
    Constraints:
    - Budget constraint (fully invested)
    - Cardinality constraint (min/max number of assets)
    - Long-only constraint (binary variables)
    """
    
    def __init__(
        self,
        expected_returns: pd.Series,
        cvar: pd.Series,
        correlation_matrix: pd.DataFrame,
        return_weight: float = 1.0,
        risk_weight: float = 1.0,
        diversification_weight: float = 0.1,
        budget_penalty: float = 10.0,
        cardinality_penalty: float = 8.0,
        min_assets: int = 5,
        max_assets: int = 15,
        normalization: bool = True
    ):
        """
        Initialize QUBO builder.
        
        Args:
            expected_returns: Expected returns for each asset
            cvar: CVaR values for each asset
            correlation_matrix: Asset correlation matrix
            return_weight: Weight for return objective
            risk_weight: Weight for risk objective
            diversification_weight: Weight for diversification objective
            budget_penalty: Penalty weight for budget constraint
            cardinality_penalty: Penalty weight for cardinality constraint
            min_assets: Minimum number of assets
            max_assets: Maximum number of assets
            normalization: Whether to normalize objectives
        """
        self.expected_returns = expected_returns
        self.cvar = cvar
        self.correlation_matrix = correlation_matrix
        self.return_weight = return_weight
        self.risk_weight = risk_weight
        self.diversification_weight = diversification_weight
        self.budget_penalty = budget_penalty
        self.cardinality_penalty = cardinality_penalty
        self.min_assets = min_assets
        self.max_assets = max_assets
        self.normalization = normalization
        
        # Align assets
        self.assets = expected_returns.index.intersection(
            cvar.index.intersection(correlation_matrix.index)
        )
        
        if len(self.assets) == 0:
            raise ValueError("No common assets between inputs")
        
        self.n = len(self.assets)
        
        # Extract aligned arrays
        self.mu = expected_returns[self.assets].values
        self.cvar_values = cvar[self.assets].values
        self.corr_matrix = correlation_matrix.loc[self.assets, self.assets].values
        
        # Normalize if requested
        if normalization:
            self._normalize_objectives()
    
    def _normalize_objectives(self):
        """Normalize objectives using z-score normalization."""
        # Normalize expected returns
        if self.mu.std() > 1e-10:
            self.mu = (self.mu - self.mu.mean()) / self.mu.std()
        
        # Normalize CVaR (already negative, so we want to minimize)
        if self.cvar_values.std() > 1e-10:
            self.cvar_values = (self.cvar_values - self.cvar_values.mean()) / self.cvar_values.std()
    
    def _build_return_objective(self) -> np.ndarray:
        """
        Build return maximization objective.
        
        Returns:
            Q matrix for return objective (minimize negative return)
        """
        Q = np.zeros((self.n, self.n))
        
        # Linear terms: -mu_i * x_i (negative because we minimize)
        for i in range(self.n):
            Q[i, i] = -self.return_weight * self.mu[i]
        
        return Q
    
    def _build_risk_objective(self) -> np.ndarray:
        """
        Build CVaR minimization objective.
        
        Returns:
            Q matrix for risk objective
        """
        Q = np.zeros((self.n, self.n))
        
        # Linear terms: cvar_i * x_i (positive because CVaR is negative)
        for i in range(self.n):
            Q[i, i] = self.risk_weight * self.cvar_values[i]
        
        return Q
    
    def _build_diversification_objective(self) -> np.ndarray:
        """
        Build diversification objective (pairwise correlation penalty).
        
        Returns:
            Q matrix for diversification objective
        """
        Q = np.zeros((self.n, self.n))
        
        # Quadratic terms: correlation penalty
        # Minimize sum of correlations between selected assets
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    # Penalize high correlation
                    Q[i, j] = self.diversification_weight * self.corr_matrix[i, j]
        
        return Q
    
    def _build_budget_constraint(self) -> np.ndarray:
        """
        Build budget constraint penalty.
        
        Constraint: sum(x_i) = 1 (fully invested)
        
        Returns:
            Q matrix for budget constraint penalty
        """
        Q = np.zeros((self.n, self.n))
        
        # Penalty: P * (sum(x_i) - 1)^2
        # = P * (sum(x_i)^2 - 2*sum(x_i) + 1)
        # = P * (sum_i sum_j x_i x_j - 2*sum_i x_i + 1)
        
        for i in range(self.n):
            for j in range(self.n):
                Q[i, j] += self.budget_penalty
            Q[i, i] -= 2 * self.budget_penalty
        
        return Q
    
    def _build_cardinality_constraint(self) -> np.ndarray:
        """
        Build cardinality constraint penalty.
        
        Constraint: min_assets <= sum(x_i) <= max_assets
        
        Returns:
            Q matrix for cardinality constraint penalty
        """
        Q = np.zeros((self.n, self.n))
        
        # Penalty for violating min constraint: P * max(0, min_assets - sum(x_i))^2
        # Penalty for violating max constraint: P * max(0, sum(x_i) - max_assets)^2
        
        # Approximate with quadratic penalty
        # For min: P * (min_assets - sum(x_i))^2 if sum(x_i) < min_assets
        # For max: P * (sum(x_i) - max_assets)^2 if sum(x_i) > max_assets
        
        # Use soft constraint: penalize deviation from target
        target_assets = (self.min_assets + self.max_assets) / 2
        
        for i in range(self.n):
            for j in range(self.n):
                Q[i, j] += self.cardinality_penalty
            Q[i, i] -= 2 * self.cardinality_penalty * target_assets
        
        return Q
    
    def build_qubo(self) -> Tuple[np.ndarray, float]:
        """
        Build complete QUBO matrix.
        
        QUBO form: minimize x^T Q x + c
        where x is binary vector
        
        Returns:
            Tuple of (Q matrix, constant term)
        """
        Q = np.zeros((self.n, self.n))
        
        # Add objectives
        Q += self._build_return_objective()
        Q += self._build_risk_objective()
        Q += self._build_diversification_objective()
        
        # Add constraint penalties
        Q += self._build_budget_constraint()
        Q += self._build_cardinality_constraint()
        
        # Constant term (doesn't affect optimization but useful for energy calculation)
        constant = self.budget_penalty + self.cardinality_penalty * (
            (self.min_assets + self.max_assets) / 2
        ) ** 2
        
        return Q, constant
    
    def compute_energy(self, solution: np.ndarray) -> float:
        """
        Compute QUBO energy for a given solution.
        
        Args:
            solution: Binary vector (n,)
            
        Returns:
            Energy value
        """
        Q, constant = self.build_qubo()
        energy = solution.T @ Q @ solution + constant
        return float(energy)
    
    def solution_to_weights(self, solution: np.ndarray) -> pd.Series:
        """
        Convert binary solution to portfolio weights.
        
        Args:
            solution: Binary vector indicating selected assets
            
        Returns:
            Series of portfolio weights (normalized to sum to 1)
        """
        if solution.sum() == 0:
            # No assets selected, return equal weights
            weights = pd.Series(1.0 / self.n, index=self.assets)
        else:
            # Equal weights among selected assets
            weights = pd.Series(solution / solution.sum(), index=self.assets)
        
        return weights
