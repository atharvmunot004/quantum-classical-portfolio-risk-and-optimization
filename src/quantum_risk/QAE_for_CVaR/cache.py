"""
Parameter cache for QAE VaR/CVaR estimation.

Caches distribution parameters and risk estimates per asset/window/confidence/date
to avoid recomputation when re-running with same configuration.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import warnings


class QAEParameterCache:
    """
    Cache for QAE distribution parameters and risk estimates.

    Key: (asset, date, estimation_window, confidence_level)
    """

    def __init__(self, cache_path: Optional[Path] = None):
        self.cache: Dict[Tuple, Dict[str, Any]] = {}
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache_hits = 0
        self.cache_misses = 0

        if self.cache_path and self.cache_path.exists():
            try:
                self.load_cache()
            except Exception as e:
                warnings.warn(f"Failed to load QAE cache: {e}")

    def _make_key(
        self,
        asset: str,
        date: pd.Timestamp,
        estimation_window: int,
        confidence_level: float
    ) -> Tuple:
        return (asset, pd.Timestamp(date), estimation_window, confidence_level)

    def get(
        self,
        asset: str,
        date: pd.Timestamp,
        estimation_window: int,
        confidence_level: float
    ) -> Optional[Dict[str, Any]]:
        key = self._make_key(asset, date, estimation_window, confidence_level)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        return None

    def set(
        self,
        asset: str,
        date: pd.Timestamp,
        estimation_window: int,
        confidence_level: float,
        value: Dict[str, Any]
    ):
        key = self._make_key(asset, date, estimation_window, confidence_level)
        self.cache[key] = value

    def get_hit_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def load_cache(self):
        if not self.cache_path or not self.cache_path.exists():
            return
        df = pd.read_parquet(self.cache_path)
        for _, row in df.iterrows():
            key = (
                str(row['asset']),
                pd.Timestamp(row['date']),
                int(row['estimation_window']),
                float(row['confidence_level'])
            )
            self.cache[key] = row.to_dict()

    def save_cache(self):
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for (asset, date, window, conf), val in self.cache.items():
            rec = {'asset': asset, 'date': date, 'estimation_window': window, 'confidence_level': conf}
            for k, v in val.items():
                if k not in rec:
                    rec[k] = v
            records.append(rec)
        if records:
            pd.DataFrame(records).to_parquet(self.cache_path, index=False)
