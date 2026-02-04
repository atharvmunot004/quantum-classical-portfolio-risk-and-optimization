"""Cache for qPCA window parameters (date, estimation_window, top_k, ...)."""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings


class QPCAWindowCache:
    def __init__(
        self,
        cache_path: Optional[Path] = None,
        key_fields: Optional[List[str]] = None,
    ):
        self.cache_path = Path(cache_path) if cache_path else None
        self.key_fields = key_fields or [
            "date", "estimation_window", "top_k", "precision_bits", "shots",
            "shrinkage_method", "return_type",
        ]
        self._store: Dict[Tuple, Dict[str, Any]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        if self.cache_path and self.cache_path.exists():
            try:
                self._load()
            except Exception as e:
                warnings.warn(f"Failed to load qPCA cache: {e}")

    def _key(self, rec: Dict[str, Any]) -> Tuple:
        return tuple(rec.get(k) for k in self.key_fields if k in rec)

    def get(self, rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        k = self._key(rec)
        if k in self._store:
            self.cache_hits += 1
            return self._store[k]
        self.cache_misses += 1
        return None

    def set(self, rec: Dict[str, Any], value: Dict[str, Any]):
        self._store[self._key(rec)] = {**rec, **value}

    def get_hit_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def _load(self):
        if not self.cache_path or not self.cache_path.exists():
            return
        df = pd.read_parquet(self.cache_path)
        for _, row in df.iterrows():
            k = tuple(row.get(f) for f in self.key_fields if f in row)
            self._store[k] = row.to_dict()

    def save_records(self, records: List[Dict[str, Any]]):
        if not self.cache_path or not records:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(records)
        existing = self.cache_path.exists()
        if existing:
            old = pd.read_parquet(self.cache_path)
            key_cols = [c for c in self.key_fields if c in old.columns and c in df.columns]
            if key_cols:
                combined = pd.concat([old, df], ignore_index=True)
                combined = combined.drop_duplicates(subset=key_cols, keep="last")
                combined.to_parquet(self.cache_path, index=False)
            else:
                pd.concat([old, df], ignore_index=True).to_parquet(self.cache_path, index=False)
        else:
            df.to_parquet(self.cache_path, index=False)
