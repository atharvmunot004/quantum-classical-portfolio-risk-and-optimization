"""
Parameter cache for QGAN asset-level estimation.

Caches QGAN parameters and training results per asset/window/date to avoid recomputation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import warnings


class QGANParameterCache:
    """
    Cache for QGAN parameters and training results.
    
    Key: (asset, date, estimation_window, num_qubits, ansatz_layers, shots, ...)
    """
    
    def __init__(self, cache_path: Optional[Path] = None, key_fields: Optional[list] = None):
        self.cache: Dict[Tuple, Dict[str, Any]] = {}
        self.cache_path = Path(cache_path) if cache_path else None
        self.key_fields = key_fields or [
            "asset", "date", "estimation_window", "generator_num_qubits",
            "ansatz_layers", "shots", "batch_size", "max_iterations", "return_type"
        ]
        self.cache_hits = 0
        self.cache_misses = 0
        
        if self.cache_path and self.cache_path.exists():
            try:
                self.load_cache()
            except Exception as e:
                warnings.warn(f"Failed to load QGAN cache: {e}")
    
    def _make_key(self, rec: Dict[str, Any]) -> Tuple:
        return tuple(rec.get(k) for k in self.key_fields if k in rec)
    
    def get(self, rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached value."""
        key = self._make_key(rec)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        return None
    
    def set(self, rec: Dict[str, Any], value: Dict[str, Any]):
        """Set cached value."""
        key = self._make_key(rec)
        self.cache[key] = {**rec, **value}
    
    def get_hit_ratio(self) -> float:
        """Get cache hit ratio."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def load_cache(self):
        """Load cache from parquet file."""
        if not self.cache_path or not self.cache_path.exists():
            return
        df = pd.read_parquet(self.cache_path)
        for _, row in df.iterrows():
            key = tuple(row.get(k) for k in self.key_fields if k in row)
            self.cache[key] = row.to_dict()
    
    def save_cache(self, records: list):
        """Save cache to parquet file."""
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if records:
            df = pd.DataFrame(records)
            if self.cache_path.exists():
                existing = pd.read_parquet(self.cache_path)
                df = pd.concat([existing, df]).drop_duplicates(
                    subset=self.key_fields, keep='last'
                )
            df.to_parquet(self.cache_path, index=False)
