"""Covariance and density matrix construction for qPCA (Ledoit-Wolf shrinkage, trace normalization)."""
import numpy as np
import pandas as pd
import time
from typing import Tuple, Optional
from sklearn.covariance import LedoitWolf


def build_covariance(
    returns_window: np.ndarray,
    shrinkage_method: Optional[str] = "ledoit_wolf",
) -> Tuple[np.ndarray, float]:
    """
    Build cross-asset covariance from return matrix (T x N).
    Returns (covariance, time_ms).
    """
    t0 = time.perf_counter()
    if shrinkage_method == "ledoit_wolf":
        lw = LedoitWolf()
        cov = lw.fit(returns_window).covariance_
    else:
        cov = np.cov(returns_window, rowvar=False)
    if cov.shape[0] != returns_window.shape[1]:
        cov = np.cov(returns_window.T)
    time_ms = (time.perf_counter() - t0) * 1000
    return np.asarray(cov, dtype=float), time_ms


def to_density_matrix(
    cov: np.ndarray,
    method: str = "trace_normalization",
    numerical_stability_eps: float = 1e-10,
) -> np.ndarray:
    """
    Convert covariance to density matrix (trace-normalize).
    rho = C / tr(C). Ensures rho >= 0 and tr(rho) = 1.
    """
    tr = np.trace(cov)
    if tr <= 0 or not np.isfinite(tr):
        raise ValueError("Covariance trace must be positive and finite for density matrix.")
    rho = cov / (tr + numerical_stability_eps)
    return rho
