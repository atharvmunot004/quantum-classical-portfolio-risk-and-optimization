"""
Metrics computation for QAOA CVaR asset-level evaluation.

Tail risk, distribution, quantum-specific, and runtime metrics.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict


def compute_tail_metrics(
    losses: pd.Series,
    cvar_series: pd.Series,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """Tail mean loss, tail max loss, RMSE CVaR vs losses."""
    violations = losses > cvar_series
    if violations.sum() == 0:
        return {
            'tail_mean_loss': np.nan,
            'tail_max_loss': np.nan,
            'rmse_cvar_vs_losses': np.nan,
        }
    tail_losses = losses[violations]
    return {
        'tail_mean_loss': float(tail_losses.mean()),
        'tail_max_loss': float(tail_losses.max()),
        'rmse_cvar_vs_losses': float(np.sqrt(np.mean((losses[violations] - cvar_series[violations])**2))),
    }


def compute_distribution_metrics(returns: pd.Series) -> Dict[str, float]:
    """Skewness, kurtosis, Jarque-Bera for returns."""
    r = returns.dropna()
    if len(r) < 3:
        return {
            'skewness': np.nan,
            'kurtosis': np.nan,
            'jarque_bera_statistic': np.nan,
            'jarque_bera_p_value': np.nan,
        }
    jb_stat, jb_p = stats.jarque_bera(r)
    return {
        'skewness': float(stats.skew(r)),
        'kurtosis': float(stats.kurtosis(r)),
        'jarque_bera_statistic': float(jb_stat),
        'jarque_bera_p_value': float(jb_p),
    }


def compute_runtime_metrics(runtimes: list) -> Dict[str, float]:
    """Total, mean, p95 runtime metrics."""
    if not runtimes:
        return {
            'total_runtime_ms': np.nan,
            'runtime_per_asset_ms': np.nan,
            'mean_qaoa_optimize_time_ms': np.nan,
            'p95_qaoa_optimize_time_ms': np.nan,
        }
    arr = np.array(runtimes)
    return {
        'total_runtime_ms': float(np.sum(arr)),
        'runtime_per_asset_ms': float(np.mean(arr)),
        'mean_qaoa_optimize_time_ms': float(np.mean(arr)),
        'p95_qaoa_optimize_time_ms': float(np.percentile(arr, 95)),
    }
