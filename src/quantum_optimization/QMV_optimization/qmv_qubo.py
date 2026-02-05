"""
QMV QUBO Construction Module.

Formulates mean-variance portfolio optimization as QUBO with discretized binary weights.
"""
import numpy as np
from typing import Tuple, Optional
from sklearn.covariance import LedoitWolf


class QMVQUBOBuilder:
    """
    Builds QUBO matrices for mean-variance portfolio optimization.
    
    Uses binary discretized weight encoding where each asset's weight
    is represented by multiple binary variables.
    """
    
    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        lambda_risk: float = 1.0,
        bits_per_asset: int = 4,
        weight_step: float = 0.0625,
        max_weight_per_asset: float = 0.25,
        budget_penalty: float = 20.0,
        max_weight_penalty: float = 10.0,
        normalization: bool = True
    ):
        """
        Initialize QMV QUBO builder.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix
            lambda_risk: Risk aversion parameter
            bits_per_asset: Number of bits per asset for weight encoding
            weight_step: Step size for discretized weights
            max_weight_per_asset: Maximum weight per asset
            budget_penalty: Penalty weight for budget constraint
            max_weight_penalty: Penalty weight for max weight constraint
            normalization: Whether to normalize objectives
        """
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.lambda_risk = lambda_risk
        self.bits_per_asset = bits_per_asset
        self.weight_step = weight_step
        self.max_weight_per_asset = max_weight_per_asset
        self.budget_penalty = budget_penalty
        self.max_weight_penalty = max_weight_penalty
        self.normalization = normalization
        
        self.n_assets = len(expected_returns)
        self.n_bits_total = self.n_assets * self.bits_per_asset
        
        # Normalize if requested
        if normalization:
            self._normalize_objectives()
    
    def _normalize_objectives(self):
        """Normalize objectives using z-score normalization."""
        # Normalize expected returns
        if self.expected_returns.std() > 1e-10:
            self.expected_returns = (self.expected_returns - self.expected_returns.mean()) / self.expected_returns.std()
        
        # Normalize covariance (scale by std)
        if self.covariance_matrix.std() > 1e-10:
            self.covariance_matrix = self.covariance_matrix / self.covariance_matrix.std()
    
    def _encode_weight(self, bit_index: int) -> float:
        """
        Encode weight value from bit index.
        
        Args:
            bit_index: Bit index (0 to 2^bits_per_asset - 1)
            
        Returns:
            Weight value
        """
        return bit_index * self.weight_step
    
    def build_qubo(self) -> Tuple[np.ndarray, float]:
        """
        Build QUBO matrix for mean-variance optimization.
        
        Objective: minimize lambda * w^T Sigma w - mu^T w
        
        Returns:
            Tuple of (Q matrix, constant term)
        """
        Q = np.zeros((self.n_bits_total, self.n_bits_total))
        
        # Build weight encoding: w_i = sum_j (bit_ij * weight_value_j)
        # where weight_value_j = j * weight_step for j in [0, 2^bits_per_asset - 1]
        
        # For each asset i, we have bits_per_asset binary variables
        # representing discretized weight levels
        
        # Mean-variance objective terms:
        # 1. Risk term: lambda * w^T Sigma w
        # 2. Return term: -mu^T w
        
        # Risk term quadratic expansion
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                sigma_ij = self.covariance_matrix[i, j]
                
                # Contribution to Q from asset i and asset j
                for bit_i in range(self.bits_per_asset):
                    for bit_j in range(self.bits_per_asset):
                        idx_i = i * self.bits_per_asset + bit_i
                        idx_j = j * self.bits_per_asset + bit_j
                        
                        weight_i = self._encode_weight(bit_i)
                        weight_j = self._encode_weight(bit_j)
                        
                        if idx_i == idx_j:
                            # Diagonal term
                            Q[idx_i, idx_j] += self.lambda_risk * sigma_ij * weight_i * weight_j
                        else:
                            # Off-diagonal term (symmetric)
                            Q[idx_i, idx_j] += self.lambda_risk * sigma_ij * weight_i * weight_j
        
        # Return term linear expansion
        for i in range(self.n_assets):
            mu_i = self.expected_returns[i]
            
            for bit_i in range(self.bits_per_asset):
                idx_i = i * self.bits_per_asset + bit_i
                weight_i = self._encode_weight(bit_i)
                
                # Linear term: -mu_i * weight_i
                Q[idx_i, idx_i] -= mu_i * weight_i
        
        # Budget constraint: sum_i w_i = 1
        # Penalty: P * (sum_i w_i - 1)^2
        for i in range(self.n_assets):
            for bit_i in range(self.bits_per_asset):
                idx_i = i * self.bits_per_asset + bit_i
                weight_i = self._encode_weight(bit_i)
                
                for j in range(self.n_assets):
                    for bit_j in range(self.bits_per_asset):
                        idx_j = j * self.bits_per_asset + bit_j
                        weight_j = self._encode_weight(bit_j)
                        
                        Q[idx_i, idx_j] += self.budget_penalty * weight_i * weight_j
                
                Q[idx_i, idx_i] -= 2 * self.budget_penalty * weight_i
        
        # Max weight constraint: w_i <= max_weight_per_asset
        # Penalty: P * max(0, w_i - max_weight)^2
        # Approximate with quadratic penalty
        for i in range(self.n_assets):
            for bit_i in range(self.bits_per_asset):
                idx_i = i * self.bits_per_asset + bit_i
                weight_i = self._encode_weight(bit_i)
                
                if weight_i > self.max_weight_per_asset:
                    Q[idx_i, idx_i] += self.max_weight_penalty * (weight_i - self.max_weight_per_asset) ** 2
        
        # Constant term
        constant = self.budget_penalty
        
        return Q, constant
    
    def decode_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        Decode binary solution to continuous weights.
        
        Args:
            solution: Binary vector (n_bits_total,)
            
        Returns:
            Array of portfolio weights (n_assets,)
        """
        weights = np.zeros(self.n_assets)
        
        for i in range(self.n_assets):
            # Sum up bits for asset i
            for bit_i in range(self.bits_per_asset):
                idx = i * self.bits_per_asset + bit_i
                if solution[idx] > 0.5:  # Binary threshold
                    weights[i] += self._encode_weight(bit_i)
        
        # Normalize to ensure fully invested
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        return weights
    
    def repair_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        Repair solution to satisfy constraints.
        
        Args:
            solution: Binary solution vector
            
        Returns:
            Repaired solution vector
        """
        # Decode to weights
        weights = self.decode_solution(solution)
        
        # Repair max weight constraint
        weights = np.clip(weights, 0, self.max_weight_per_asset)
        
        # Repair budget constraint (normalize)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        # Re-encode to binary (simplified: use nearest discretized value)
        repaired_solution = np.zeros_like(solution)
        
        for i in range(self.n_assets):
            # Find nearest discretized weight
            weight_i = weights[i]
            nearest_bit = int(round(weight_i / self.weight_step))
            nearest_bit = max(0, min(nearest_bit, 2**self.bits_per_asset - 1))
            
            # Set corresponding bit
            idx = i * self.bits_per_asset + nearest_bit
            if idx < len(repaired_solution):
                repaired_solution[idx] = 1
        
        return repaired_solution
