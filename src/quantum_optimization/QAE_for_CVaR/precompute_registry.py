"""
Precompute Registry for QAE Portfolio CVaR.

Manages caching and reuse of quantum artifacts across portfolios.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Set
import hashlib
import json


class PrecomputeRegistry:
    """
    Registry for caching QAE precomputed artifacts.
    
    Groups portfolios by asset set and precomputes quantum CVaR artifacts
    once per unique asset set, then reuses them across portfolios.
    """
    
    def __init__(
        self,
        registry_root: str = "cache/qae_precompute",
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
        self._distribution_params: Dict[Tuple[str, ...], Dict] = {}
        self._qae_tail_expectations: Dict[Tuple, float] = {}
        self._var_thresholds: Dict[Tuple, float] = {}
        
        # Load from disk if exists
        if persist_to_disk:
            self._load_from_disk()
    
    def _load_from_disk(self):
        """Load cache from disk."""
        asset_set_index_path = self.registry_root / "asset_sets.parquet"
        if asset_set_index_path.exists():
            df = pd.read_parquet(asset_set_index_path)
            for _, row in df.iterrows():
                asset_set = tuple(row['asset_set'])
                self._asset_set_index[asset_set] = row['asset_set_id']
    
    def _save_to_disk(self):
        """Save cache to disk."""
        if not self.persist_to_disk:
            return
        
        # Save asset set index
        if self._asset_set_index:
            rows = []
            for asset_set, asset_set_id in self._asset_set_index.items():
                rows.append({
                    'asset_set_id': asset_set_id,
                    'asset_set': list(asset_set)
                })
            df = pd.DataFrame(rows)
            df.to_parquet(self.registry_root / "asset_sets.parquet", index=False)
    
    def register_asset_set(self, asset_set: Tuple[str, ...]) -> int:
        """
        Register an asset set and return its ID.
        
        Args:
            asset_set: Tuple of asset names
            
        Returns:
            Asset set ID
        """
        if asset_set not in self._asset_set_index:
            asset_set_id = len(self._asset_set_index)
            self._asset_set_index[asset_set] = asset_set_id
            self._save_to_disk()
        else:
            asset_set_id = self._asset_set_index[asset_set]
        
        return asset_set_id
    
    def get_asset_set_id(self, asset_set: Tuple[str, ...]) -> Optional[int]:
        """
        Get asset set ID if registered.
        
        Args:
            asset_set: Tuple of asset names
            
        Returns:
            Asset set ID or None if not registered
        """
        return self._asset_set_index.get(asset_set)
    
    def store_distribution_params(
        self,
        asset_set: Tuple[str, ...],
        dist_params: Dict
    ):
        """
        Store distribution parameters for an asset set.
        
        Args:
            asset_set: Tuple of asset names
            dist_params: Distribution parameters dictionary
        """
        self._distribution_params[asset_set] = dist_params
    
    def get_distribution_params(
        self,
        asset_set: Tuple[str, ...]
    ) -> Optional[Dict]:
        """
        Get distribution parameters for an asset set.
        
        Args:
            asset_set: Tuple of asset names
            
        Returns:
            Distribution parameters or None
        """
        return self._distribution_params.get(asset_set)
    
    def store_qae_tail_expectation(
        self,
        asset_set: Tuple[str, ...],
        confidence_level: float,
        tail_expectation: float
    ):
        """
        Store QAE tail expectation for an asset set and confidence level.
        
        Args:
            asset_set: Tuple of asset names
            confidence_level: Confidence level
            tail_expectation: Tail expectation value
        """
        key = (asset_set, confidence_level)
        self._qae_tail_expectations[key] = tail_expectation
    
    def get_qae_tail_expectation(
        self,
        asset_set: Tuple[str, ...],
        confidence_level: float
    ) -> Optional[float]:
        """
        Get QAE tail expectation for an asset set and confidence level.
        
        Args:
            asset_set: Tuple of asset names
            confidence_level: Confidence level
            
        Returns:
            Tail expectation or None
        """
        key = (asset_set, confidence_level)
        return self._qae_tail_expectations.get(key)
    
    def store_var_threshold(
        self,
        asset_set: Tuple[str, ...],
        confidence_level: float,
        var_threshold: float
    ):
        """
        Store VaR threshold for an asset set and confidence level.
        
        Args:
            asset_set: Tuple of asset names
            confidence_level: Confidence level
            var_threshold: VaR threshold value
        """
        key = (asset_set, confidence_level)
        self._var_thresholds[key] = var_threshold
    
    def get_var_threshold(
        self,
        asset_set: Tuple[str, ...],
        confidence_level: float
    ) -> Optional[float]:
        """
        Get VaR threshold for an asset set and confidence level.
        
        Args:
            asset_set: Tuple of asset names
            confidence_level: Confidence level
            
        Returns:
            VaR threshold or None
        """
        key = (asset_set, confidence_level)
        return self._var_thresholds.get(key)
    
    def create_portfolio_to_asset_set_map(
        self,
        portfolio_weights: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create mapping from portfolios to asset sets.
        
        Args:
            portfolio_weights: DataFrame of portfolio weights (N portfolios x M assets)
            
        Returns:
            DataFrame with portfolio_id and asset_set_id
        """
        rows = []
        
        for portfolio_id in portfolio_weights.index:
            # Get non-zero weights (assets in this portfolio)
            weights_row = portfolio_weights.loc[portfolio_id]
            asset_set = tuple(sorted(weights_row[weights_row > 1e-10].index))
            
            asset_set_id = self.register_asset_set(asset_set)
            
            rows.append({
                'portfolio_id': portfolio_id,
                'asset_set': list(asset_set),
                'asset_set_id': asset_set_id
            })
        
        return pd.DataFrame(rows)
