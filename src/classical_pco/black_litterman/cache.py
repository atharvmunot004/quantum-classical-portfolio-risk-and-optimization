"""
Caching module for Black-Litterman optimization.

Implements invariant caching for expensive computations that don't change
across portfolio optimizations.
"""
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import pandas as pd
import numpy as np
import warnings
import os


class BLCache:
    """
    Cache for Black-Litterman computations.
    
    Caches expensive computations that are invariant across portfolio optimizations:
    - Daily returns (by asset set)
    - Covariance matrix (by asset set and estimation window)
    - Market equilibrium returns (by asset set)
    - Market weights (by asset set)
    - Parsed views (by asset set and views hash)
    - Posterior returns (by asset set and views hash)
    - Posterior covariance (by asset set and views hash)
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
    
    def _make_key(
        self,
        cache_type: str,
        asset_set: Optional[Tuple[str, ...]] = None,
        estimation_window: Optional[int] = None,
        views_hash: Optional[str] = None
    ) -> str:
        """
        Create cache key.
        
        Args:
            cache_type: Type of cached value
            asset_set: Sorted tuple of asset names
            estimation_window: Estimation window size
            views_hash: Hash of views
            
        Returns:
            Cache key string
        """
        parts = [cache_type]
        
        if asset_set is not None:
            parts.append(f"assets_{hash(tuple(sorted(asset_set)))}")
        
        if estimation_window is not None:
            parts.append(f"window_{estimation_window}")
        
        if views_hash is not None:
            parts.append(f"views_{views_hash}")
        
        return "_".join(parts)
    
    def _hash_views(self, views: Dict) -> str:
        """
        Create hash of views dictionary.
        
        Args:
            views: Views dictionary
            
        Returns:
            Hash string
        """
        # Convert to JSON-serializable format and hash
        views_str = json.dumps(views, sort_keys=True, default=str)
        return hashlib.md5(views_str.encode()).hexdigest()
    
    def _get_asset_set(self, data: pd.DataFrame) -> Tuple[str, ...]:
        """Extract sorted asset set from DataFrame."""
        if isinstance(data, pd.DataFrame):
            return tuple(sorted(data.columns.tolist()))
        elif isinstance(data, pd.Series):
            return tuple(sorted(data.index.tolist()))
        else:
            return tuple()
    
    def get_daily_returns(
        self,
        asset_set: Tuple[str, ...]
    ) -> Optional[pd.DataFrame]:
        """Get cached daily returns."""
        if not self.enabled:
            return None
        
        key = self._make_key("daily_returns", asset_set=asset_set)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        return None
    
    def set_daily_returns(
        self,
        asset_set: Tuple[str, ...],
        returns: pd.DataFrame
    ):
        """Cache daily returns."""
        if not self.enabled:
            return
        
        key = self._make_key("daily_returns", asset_set=asset_set)
        self.cache[key] = returns.copy()
    
    def get_covariance_matrix(
        self,
        asset_set: Tuple[str, ...],
        estimation_window: int
    ) -> Optional[pd.DataFrame]:
        """Get cached covariance matrix."""
        if not self.enabled:
            return None
        
        key = self._make_key(
            "covariance_matrix",
            asset_set=asset_set,
            estimation_window=estimation_window
        )
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        return None
    
    def set_covariance_matrix(
        self,
        asset_set: Tuple[str, ...],
        estimation_window: int,
        cov_matrix: pd.DataFrame
    ):
        """Cache covariance matrix."""
        if not self.enabled:
            return
        
        key = self._make_key(
            "covariance_matrix",
            asset_set=asset_set,
            estimation_window=estimation_window
        )
        self.cache[key] = cov_matrix.copy()
    
    def get_market_equilibrium(
        self,
        asset_set: Tuple[str, ...]
    ) -> Optional[Tuple[pd.Series, pd.Series]]:
        """Get cached market equilibrium returns and weights."""
        if not self.enabled:
            return None
        
        key = self._make_key("market_equilibrium", asset_set=asset_set)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        return None
    
    def set_market_equilibrium(
        self,
        asset_set: Tuple[str, ...],
        prior_returns: pd.Series,
        market_weights: pd.Series
    ):
        """Cache market equilibrium returns and weights."""
        if not self.enabled:
            return
        
        key = self._make_key("market_equilibrium", asset_set=asset_set)
        self.cache[key] = (prior_returns.copy(), market_weights.copy())
    
    def get_parsed_views(
        self,
        asset_set: Tuple[str, ...],
        views_hash: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get cached parsed views (P, Q, Omega)."""
        if not self.enabled:
            return None
        
        key = self._make_key("parsed_views", asset_set=asset_set, views_hash=views_hash)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        return None
    
    def set_parsed_views(
        self,
        asset_set: Tuple[str, ...],
        views_hash: str,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray
    ):
        """Cache parsed views."""
        if not self.enabled:
            return
        
        key = self._make_key("parsed_views", asset_set=asset_set, views_hash=views_hash)
        self.cache[key] = (P.copy(), Q.copy(), Omega.copy())
    
    def get_posterior_returns(
        self,
        asset_set: Tuple[str, ...],
        views_hash: str
    ) -> Optional[pd.Series]:
        """Get cached posterior returns."""
        if not self.enabled:
            return None
        
        key = self._make_key("posterior_returns", asset_set=asset_set, views_hash=views_hash)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        return None
    
    def set_posterior_returns(
        self,
        asset_set: Tuple[str, ...],
        views_hash: str,
        posterior_returns: pd.Series
    ):
        """Cache posterior returns."""
        if not self.enabled:
            return
        
        key = self._make_key("posterior_returns", asset_set=asset_set, views_hash=views_hash)
        self.cache[key] = posterior_returns.copy()
    
    def get_posterior_covariance(
        self,
        asset_set: Tuple[str, ...],
        views_hash: str
    ) -> Optional[pd.DataFrame]:
        """Get cached posterior covariance."""
        if not self.enabled:
            return None
        
        key = self._make_key("posterior_covariance", asset_set=asset_set, views_hash=views_hash)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        return None
    
    def set_posterior_covariance(
        self,
        asset_set: Tuple[str, ...],
        views_hash: str,
        posterior_cov: pd.DataFrame
    ):
        """Cache posterior covariance."""
        if not self.enabled:
            return
        
        key = self._make_key("posterior_covariance", asset_set=asset_set, views_hash=views_hash)
        self.cache[key] = posterior_cov.copy()
    
    def get_hit_ratio(self) -> float:
        """Get cache hit ratio."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


class PrecomputeRegistry:
    """
    Disk-backed precompute registry for Black-Litterman artifacts.
    
    Persists precomputed values per asset_set_id and views_hash for reuse
    across portfolio evaluations.
    """
    
    def __init__(self, registry_root: str = "cache/bl_precompute"):
        """
        Initialize precompute registry.
        
        Args:
            registry_root: Root directory for precomputed artifacts
        """
        self.registry_root = Path(registry_root)
        self.registry_root.mkdir(parents=True, exist_ok=True)
    
    def _hash_views(self, views: Dict) -> str:
        """Create hash of views dictionary."""
        views_str = json.dumps(views, sort_keys=True, default=str)
        return hashlib.md5(views_str.encode()).hexdigest()
    
    def _asset_set_to_id(self, asset_set: Tuple[str, ...]) -> str:
        """Convert asset set tuple to deterministic ID."""
        asset_set_str = "_".join(sorted(asset_set))
        return hashlib.md5(asset_set_str.encode()).hexdigest()[:16]
    
    def save_covariance(self, asset_set: Tuple[str, ...], cov_matrix: pd.DataFrame, estimation_window: int, shrinkage_method: str):
        """Save covariance matrix."""
        asset_set_id = self._asset_set_to_id(asset_set)
        key = f"{asset_set_id}_w{estimation_window}_{shrinkage_method}"
        path = self.registry_root / f"cov_{key}.parquet"
        cov_matrix.to_parquet(path)
    
    def load_covariance(self, asset_set: Tuple[str, ...], estimation_window: int, shrinkage_method: str) -> Optional[pd.DataFrame]:
        """Load covariance matrix."""
        asset_set_id = self._asset_set_to_id(asset_set)
        key = f"{asset_set_id}_w{estimation_window}_{shrinkage_method}"
        path = self.registry_root / f"cov_{key}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None
    
    def save_market_equilibrium(self, asset_set: Tuple[str, ...], prior_returns: pd.Series, market_weights: pd.Series, risk_aversion: float):
        """Save market equilibrium returns and weights."""
        asset_set_id = self._asset_set_to_id(asset_set)
        key = f"{asset_set_id}_ra{risk_aversion}"
        prior_path = self.registry_root / f"prior_pi_{key}.parquet"
        weights_path = self.registry_root / f"market_w_{key}.parquet"
        prior_returns.to_frame().to_parquet(prior_path)
        market_weights.to_frame().to_parquet(weights_path)
    
    def load_market_equilibrium(self, asset_set: Tuple[str, ...], risk_aversion: float) -> Optional[Tuple[pd.Series, pd.Series]]:
        """Load market equilibrium returns and weights."""
        asset_set_id = self._asset_set_to_id(asset_set)
        key = f"{asset_set_id}_ra{risk_aversion}"
        prior_path = self.registry_root / f"prior_pi_{key}.parquet"
        weights_path = self.registry_root / f"market_w_{key}.parquet"
        if prior_path.exists() and weights_path.exists():
            prior = pd.read_parquet(prior_path).iloc[:, 0]
            weights = pd.read_parquet(weights_path).iloc[:, 0]
            return prior, weights
        return None
    
    def save_parsed_views(self, asset_set: Tuple[str, ...], views_hash: str, P: np.ndarray, Q: np.ndarray, Omega: np.ndarray):
        """Save parsed views (P, Q, Omega)."""
        asset_set_id = self._asset_set_to_id(asset_set)
        key = f"{asset_set_id}_{views_hash}"
        path = self.registry_root / f"views_{key}.npz"
        np.savez(path, P=P, Q=Q, Omega=Omega)
    
    def load_parsed_views(self, asset_set: Tuple[str, ...], views_hash: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load parsed views."""
        asset_set_id = self._asset_set_to_id(asset_set)
        key = f"{asset_set_id}_{views_hash}"
        path = self.registry_root / f"views_{key}.npz"
        if path.exists():
            data = np.load(path)
            return data['P'], data['Q'], data['Omega']
        return None
    
    def save_posterior(self, asset_set: Tuple[str, ...], views_hash: str, posterior_returns: pd.Series, posterior_cov: pd.DataFrame):
        """Save posterior returns and covariance."""
        asset_set_id = self._asset_set_to_id(asset_set)
        key = f"{asset_set_id}_{views_hash}"
        returns_path = self.registry_root / f"post_mu_{key}.parquet"
        cov_path = self.registry_root / f"post_cov_{key}.parquet"
        posterior_returns.to_frame().to_parquet(returns_path)
        posterior_cov.to_parquet(cov_path)
    
    def load_posterior(self, asset_set: Tuple[str, ...], views_hash: str) -> Optional[Tuple[pd.Series, pd.DataFrame]]:
        """Load posterior returns and covariance."""
        asset_set_id = self._asset_set_to_id(asset_set)
        key = f"{asset_set_id}_{views_hash}"
        returns_path = self.registry_root / f"post_mu_{key}.parquet"
        cov_path = self.registry_root / f"post_cov_{key}.parquet"
        if returns_path.exists() and cov_path.exists():
            returns = pd.read_parquet(returns_path).iloc[:, 0]
            cov = pd.read_parquet(cov_path)
            return returns, cov
        return None
    
    def save_optimal_weights(self, asset_set: Tuple[str, ...], views_hash: str, constraints_hash: str, optimal_weights: pd.Series):
        """Save optimal weights."""
        asset_set_id = self._asset_set_to_id(asset_set)
        key = f"{asset_set_id}_{views_hash}_{constraints_hash}"
        path = self.registry_root / f"optimal_w_{key}.parquet"
        optimal_weights.to_frame().to_parquet(path)
    
    def load_optimal_weights(self, asset_set: Tuple[str, ...], views_hash: str, constraints_hash: str) -> Optional[pd.Series]:
        """Load optimal weights."""
        asset_set_id = self._asset_set_to_id(asset_set)
        key = f"{asset_set_id}_{views_hash}_{constraints_hash}"
        path = self.registry_root / f"optimal_w_{key}.parquet"
        if path.exists():
            return pd.read_parquet(path).iloc[:, 0]
        return None

