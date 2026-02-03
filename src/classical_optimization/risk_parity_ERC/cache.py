"""
Caching module for Risk Parity ERC optimization.

Implements in-memory LRU caching for expensive computations.
"""
import hashlib
from typing import Dict, Optional, Tuple, Any
from collections import OrderedDict
import pandas as pd


class RiskParityCache:
    """
    LRU cache for Risk Parity ERC computations.
    
    Caches covariance matrices by estimation window, estimator, and rebalance date.
    """
    
    def __init__(self, enabled: bool = True, max_size: int = 5000):
        """
        Initialize cache.
        
        Args:
            enabled: Whether caching is enabled
            max_size: Maximum number of cached items
        """
        self.enabled = enabled
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.max_size = max_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _make_key(
        self,
        assets_tuple: Tuple[str, ...],
        estimation_window: int,
        estimator_name: str,
        rebalance_date: Optional[pd.Timestamp] = None
    ) -> str:
        """Create cache key."""
        asset_str = "_".join(sorted(assets_tuple))
        asset_hash = hashlib.md5(asset_str.encode()).hexdigest()[:12]
        parts = [asset_hash, f"w{estimation_window}", estimator_name]
        if rebalance_date is not None:
            parts.append(str(rebalance_date.date()))
        return "_".join(parts)
    
    def get_covariance_matrix(
        self,
        assets: Tuple[str, ...],
        estimation_window: int,
        estimator_name: str,
        rebalance_date: Optional[pd.Timestamp] = None
    ) -> Optional[pd.DataFrame]:
        """Get cached covariance matrix."""
        if not self.enabled:
            return None
        
        key = self._make_key(assets, estimation_window, estimator_name, rebalance_date)
        
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.cache_hits += 1
            return self.cache[key].copy()
        
        self.cache_misses += 1
        return None
    
    def set_covariance_matrix(
        self,
        assets: Tuple[str, ...],
        estimation_window: int,
        estimator_name: str,
        cov_matrix: pd.DataFrame,
        rebalance_date: Optional[pd.Timestamp] = None
    ):
        """Set cached covariance matrix."""
        if not self.enabled:
            return
        
        key = self._make_key(assets, estimation_window, estimator_name, rebalance_date)
        
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = cov_matrix.copy()
        # Move to end (most recently used)
        self.cache.move_to_end(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0.0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.max_size
        }

