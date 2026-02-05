"""
Precompute Registry for QMV Portfolio Optimization.

Manages caching and reuse of quantum artifacts across portfolios.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple


class PrecomputeRegistry:
    """
    Registry for caching QMV precomputed artifacts.
    
    Groups portfolios by asset set and precomputes covariance and expected returns
    once per unique asset set, then reuses them.
    """
    
    def __init__(
        self,
        registry_root: str = "cache/qmv_precompute",
        persist_to_disk: bool = True
    ):
        """
        Initialize precompute registry.
        
        Args:
            registry_root: Root directory for cache
            persist_to_disk: Whether to persist cache to disk
        """
        self.registry_root = Path(registry_root)
        self.persist_to_disk = persist_to_disk
        
        if persist_to_disk:
            self.registry_root.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._asset_set_index: Dict[Tuple[str, ...], int] = {}
        self._expected_returns: Dict[Tuple[str, ...], np.ndarray] = {}
        self._covariance_matrices: Dict[Tuple[str, ...], np.ndarray] = {}
        self._qubo_matrices: Dict[Tuple, np.ndarray] = {}
    
    def register_asset_set(self, asset_set: Tuple[str, ...]) -> int:
        """Register an asset set and return its ID."""
        if asset_set not in self._asset_set_index:
            asset_set_id = len(self._asset_set_index)
            self._asset_set_index[asset_set] = asset_set_id
        else:
            asset_set_id = self._asset_set_index[asset_set]
        
        return asset_set_id
    
    def store_expected_returns(
        self,
        asset_set: Tuple[str, ...],
        expected_returns: np.ndarray
    ):
        """Store expected returns for an asset set."""
        self._expected_returns[asset_set] = expected_returns
    
    def get_expected_returns(
        self,
        asset_set: Tuple[str, ...]
    ) -> Optional[np.ndarray]:
        """Get expected returns for an asset set."""
        return self._expected_returns.get(asset_set)
    
    def store_covariance_matrix(
        self,
        asset_set: Tuple[str, ...],
        covariance_matrix: np.ndarray
    ):
        """Store covariance matrix for an asset set."""
        self._covariance_matrices[asset_set] = covariance_matrix
    
    def get_covariance_matrix(
        self,
        asset_set: Tuple[str, ...]
    ) -> Optional[np.ndarray]:
        """Get covariance matrix for an asset set."""
        return self._covariance_matrices.get(asset_set)
    
    def store_qubo_matrix(
        self,
        asset_set: Tuple[str, ...],
        lambda_risk: float,
        Q: np.ndarray,
        constant: float
    ):
        """Store QUBO matrix for an asset set and lambda."""
        key = (asset_set, lambda_risk)
        self._qubo_matrices[key] = (Q, constant)
    
    def get_qubo_matrix(
        self,
        asset_set: Tuple[str, ...],
        lambda_risk: float
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Get QUBO matrix for an asset set and lambda."""
        key = (asset_set, lambda_risk)
        return self._qubo_matrices.get(key)
