"""
Caching module for Markowitz Mean-Variance optimization.

Implements in-memory caching for expensive computations that are reused
across portfolio optimizations with the same asset universe.
"""
import hashlib
from typing import Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np


class MarkowitzCache:
    """
    Cache for Markowitz optimization computations.
    
    Caches expensive computations that are invariant across portfolio optimizations:
    - Covariance matrix (by asset set hash and estimation window)
    - Expected returns (by asset set hash)
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize cache.
        
        Args:
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _asset_universe_hash(self, assets: Tuple[str, ...]) -> str:
        """
        Generate hash for asset universe.
        
        Args:
            assets: Tuple of asset identifiers
            
        Returns:
            Hash string
        """
        asset_str = "_".join(sorted(assets))
        return hashlib.md5(asset_str.encode()).hexdigest()
    
    def _make_key(
        self,
        cache_type: str,
        asset_universe_hash: str,
        estimation_window: Optional[int] = None,
        shrinkage_method: Optional[str] = None
    ) -> str:
        """
        Create cache key.
        
        Args:
            cache_type: Type of cached value ('covariance', 'expected_returns')
            asset_universe_hash: Hash of asset universe
            estimation_window: Optional estimation window
            shrinkage_method: Optional shrinkage method
            
        Returns:
            Cache key string
        """
        parts = [cache_type, asset_universe_hash]
        if estimation_window is not None:
            parts.append(f"w{estimation_window}")
        if shrinkage_method is not None:
            parts.append(shrinkage_method)
        return "_".join(parts)
    
    def get_covariance_matrix(
        self,
        assets: Tuple[str, ...],
        estimation_window: int,
        shrinkage_method: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get cached covariance matrix.
        
        Args:
            assets: Tuple of asset identifiers
            estimation_window: Estimation window size
            shrinkage_method: Shrinkage method used
            
        Returns:
            Cached covariance matrix or None
        """
        if not self.enabled:
            return None
        
        asset_hash = self._asset_universe_hash(assets)
        key = self._make_key("covariance", asset_hash, estimation_window, shrinkage_method)
        
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key].copy()
        
        self.cache_misses += 1
        return None
    
    def set_covariance_matrix(
        self,
        assets: Tuple[str, ...],
        estimation_window: int,
        cov_matrix: pd.DataFrame,
        shrinkage_method: Optional[str] = None
    ):
        """
        Cache covariance matrix.
        
        Args:
            assets: Tuple of asset identifiers
            estimation_window: Estimation window size
            cov_matrix: Covariance matrix to cache
            shrinkage_method: Shrinkage method used
        """
        if not self.enabled:
            return
        
        asset_hash = self._asset_universe_hash(assets)
        key = self._make_key("covariance", asset_hash, estimation_window, shrinkage_method)
        self.cache[key] = cov_matrix.copy()
    
    def get_expected_returns(
        self,
        assets: Tuple[str, ...]
    ) -> Optional[pd.Series]:
        """
        Get cached expected returns.
        
        Args:
            assets: Tuple of asset identifiers
            
        Returns:
            Cached expected returns or None
        """
        if not self.enabled:
            return None
        
        asset_hash = self._asset_universe_hash(assets)
        key = self._make_key("expected_returns", asset_hash)
        
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key].copy()
        
        self.cache_misses += 1
        return None
    
    def set_expected_returns(
        self,
        assets: Tuple[str, ...],
        expected_returns: pd.Series
    ):
        """
        Cache expected returns.
        
        Args:
            assets: Tuple of asset identifiers
            expected_returns: Expected returns to cache
        """
        if not self.enabled:
            return
        
        asset_hash = self._asset_universe_hash(assets)
        key = self._make_key("expected_returns", asset_hash)
        self.cache[key] = expected_returns.copy()
    
    def clear(self):
        """Clear all cached values."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache hits, misses, and size
        """
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_size': len(self.cache),
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
        }

