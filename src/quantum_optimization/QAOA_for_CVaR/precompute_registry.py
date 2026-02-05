"""
Precompute Registry for QAOA Portfolio CVaR Optimization.

Manages caching and reuse of quantum artifacts across portfolios.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import hashlib


class PrecomputeRegistry:
    """
    Registry for caching QAOA precomputed artifacts.
    
    Groups portfolios by asset set and precomputes quantum artifacts
    once per unique asset set and date, then reuses them.
    """
    
    def __init__(
        self,
        registry_root: str = "cache/qaoa_cvar_precompute",
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
        self._scenario_matrices: Dict[Tuple, np.ndarray] = {}
        self._cost_hamiltonians: Dict[Tuple, object] = {}
        self._qaoa_results: Dict[Tuple, object] = {}
    
    def register_asset_set(self, asset_set: Tuple[str, ...]) -> int:
        """Register an asset set and return its ID."""
        if asset_set not in self._asset_set_index:
            asset_set_id = len(self._asset_set_index)
            self._asset_set_index[asset_set] = asset_set_id
        else:
            asset_set_id = self._asset_set_index[asset_set]
        
        return asset_set_id
    
    def store_scenario_matrix(
        self,
        asset_set: Tuple[str, ...],
        date: pd.Timestamp,
        scenario_matrix: np.ndarray
    ):
        """Store scenario matrix."""
        key = (asset_set, date)
        self._scenario_matrices[key] = scenario_matrix
    
    def get_scenario_matrix(
        self,
        asset_set: Tuple[str, ...],
        date: pd.Timestamp
    ) -> Optional[np.ndarray]:
        """Get scenario matrix."""
        key = (asset_set, date)
        return self._scenario_matrices.get(key)
    
    def store_cost_hamiltonian(
        self,
        asset_set: Tuple[str, ...],
        date: pd.Timestamp,
        weight_config: Tuple,
        hamiltonian: object
    ):
        """Store cost Hamiltonian."""
        key = (asset_set, date, weight_config)
        self._cost_hamiltonians[key] = hamiltonian
    
    def get_cost_hamiltonian(
        self,
        asset_set: Tuple[str, ...],
        date: pd.Timestamp,
        weight_config: Tuple
    ) -> Optional[object]:
        """Get cost Hamiltonian."""
        key = (asset_set, date, weight_config)
        return self._cost_hamiltonians.get(key)
    
    def store_qaoa_result(
        self,
        asset_set: Tuple[str, ...],
        date: pd.Timestamp,
        weight_config: Tuple,
        reps: int,
        result: object
    ):
        """Store QAOA result."""
        key = (asset_set, date, weight_config, reps)
        self._qaoa_results[key] = result
    
    def get_qaoa_result(
        self,
        asset_set: Tuple[str, ...],
        date: pd.Timestamp,
        weight_config: Tuple,
        reps: int
    ) -> Optional[object]:
        """Get QAOA result."""
        key = (asset_set, date, weight_config, reps)
        return self._qaoa_results.get(key)
