"""
Fast asset-set level caching for static ERC optimization.
"""
from typing import Dict, Optional, Tuple
from collections import OrderedDict
import numpy as np
import pandas as pd


class AssetSetCache:
    """
    Fast cache for asset-set level computations.
    Caches covariance, expected returns, correlations, and equal-weight returns.
    """
    
    def __init__(self, enabled: bool = True, max_size: int = 200000):
        self.enabled = enabled
        self.cache: OrderedDict[Tuple, Dict] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, assets: Tuple[str, ...]) -> Tuple[str, ...]:
        """Create cache key from sorted asset tuple."""
        return tuple(sorted(assets))
    
    def get(self, assets: Tuple[str, ...], cache_type: str):
        """Get cached value."""
        if not self.enabled:
            return None
        
        key = self._make_key(assets)
        if key in self.cache:
            cache_entry = self.cache[key]
            if cache_type in cache_entry:
                self.cache.move_to_end(key)  # LRU
                self.hits += 1
                return cache_entry[cache_type]
        
        self.misses += 1
        return None
    
    def set(self, assets: Tuple[str, ...], cache_type: str, value):
        """Set cached value."""
        if not self.enabled:
            return
        
        key = self._make_key(assets)
        
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        if key not in self.cache:
            self.cache[key] = {}
        
        self.cache[key][cache_type] = value
        self.cache.move_to_end(key)  # LRU
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.max_size
        }

