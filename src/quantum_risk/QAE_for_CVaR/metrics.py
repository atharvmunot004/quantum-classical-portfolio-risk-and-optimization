"""
Metrics computation for QAE VaR/CVaR evaluation.

Tail risk, distribution, quantum-specific, and runtime metrics.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict


def compute_tail_metrics(
    losses: pd.Series,
    var_series: pd.Series,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    violations = losses > var_series
    if violations.sum() == 0:
        return {
            'mean_exceedance': np.nan, 'max_exceedance': np.nan,
            'std_exceedance': np.nan, 'quantile_loss_score': np.nan,
            'rmse_var_vs_losses': np.nan
        }
    exceedances = (losses - var_series)[violations]
    alpha = 1 - confidence_level
    quantile_loss = np.mean(
        np.maximum(alpha * (losses - var_series), (alpha - 1) * (losses - var_series))
    )
    return {
        'mean_exceedance': exceedances.mean(),
        'max_exceedance': exceedances.max(),
        'std_exceedance': exceedances.std(),
        'quantile_loss_score': quantile_loss,
        'rmse_var_vs_losses': np.sqrt(np.mean((losses[violations] - var_series[violations])**2))
    }


def compute_cvar_tail_metrics(
    losses: pd.Series,
    cvar_series: pd.Series,
    var_series: pd.Series,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    cvar_violations = losses > cvar_series
    if cvar_violations.sum() == 0:
        return {
            'cvar_mean_exceedance': np.nan, 'cvar_max_exceedance': np.nan,
            'cvar_std_exceedance': np.nan, 'rmse_cvar_vs_losses': np.nan
        }
    exceedances = (losses - cvar_series)[cvar_violations]
    return {
        'cvar_mean_exceedance': exceedances.mean(),
        'cvar_max_exceedance': exceedances.max(),
        'cvar_std_exceedance': exceedances.std(),
        'rmse_cvar_vs_losses': np.sqrt(np.mean((losses[cvar_violations] - cvar_series[cvar_violations])**2))
    }


def compute_distribution_metrics(returns: pd.Series) -> Dict[str, float]:
    r = returns.dropna()
    if len(r) < 3:
        return {'skewness': np.nan, 'kurtosis': np.nan, 'jarque_bera_p_value': np.nan, 'jarque_bera_statistic': np.nan}
    jb_stat, jb_p = stats.jarque_bera(r)
    return {
        'skewness': stats.skew(r),
        'kurtosis': stats.kurtosis(r),
        'jarque_bera_p_value': jb_p,
        'jarque_bera_statistic': jb_stat
    }


def compute_runtime_metrics(runtimes: list) -> Dict[str, float]:
    if not runtimes:
        return {
            'total_runtime_ms': np.nan, 'p95_runtime_ms': np.nan,
            'mean_runtime_ms': np.nan, 'runtime_per_asset_ms': np.nan
        }
    arr = np.array(runtimes)
    return {
        'total_runtime_ms': np.sum(arr) * 1000,
        'p95_runtime_ms': np.percentile(arr, 95) * 1000,
        'mean_runtime_ms': np.mean(arr) * 1000,
        'runtime_per_asset_ms': np.mean(arr) * 1000,
    }
