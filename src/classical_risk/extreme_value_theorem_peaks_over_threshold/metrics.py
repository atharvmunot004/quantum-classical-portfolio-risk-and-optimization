"""
Metrics computation module for EVT-POT VaR and CVaR evaluation.

Computes tail risk, distribution, and runtime metrics with EVT-specific metrics
like tail index and shape-scale stability. All metrics computed in loss space.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional, List
import time


def compute_tail_metrics(
    losses: pd.Series,
    var_series: pd.Series,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Compute tail risk metrics including exceedances for VaR using losses.
    
    All exceedances computed as (loss - VaR) for violating points.
    
    Args:
        losses: Series of actual losses (loss_t = -returns_t)
        var_series: Series of VaR values (positive, represents loss quantile)
        confidence_level: Confidence level used for VaR
        
    Returns:
        Dictionary of tail metrics
    """
    violations = losses > var_series
    
    if violations.sum() == 0:
        return {
            'mean_exceedance': np.nan,
            'max_exceedance': np.nan,
            'std_exceedance': np.nan,
            'quantile_loss_score': np.nan,
            'rmse_var_vs_losses': np.nan
        }
    
    # Exceedances (actual losses beyond VaR) = (loss - VaR) for violating points
    exceedances = (losses - var_series)[violations]
    
    mean_exceedance = exceedances.mean()
    max_exceedance = exceedances.max()
    std_exceedance = exceedances.std()
    
    # Quantile loss (pinball loss) for VaR - computed using losses and VaR
    alpha = 1 - confidence_level
    quantile_loss = np.mean(
        np.maximum(alpha * (losses - var_series), (alpha - 1) * (losses - var_series))
    )
    
    # RMSE between VaR and actual losses (only for violations)
    rmse_var_vs_losses = np.sqrt(np.mean((losses[violations] - var_series[violations])**2))
    
    return {
        'mean_exceedance': mean_exceedance,
        'max_exceedance': max_exceedance,
        'std_exceedance': std_exceedance,
        'quantile_loss_score': quantile_loss,
        'rmse_var_vs_losses': rmse_var_vs_losses
    }


def compute_cvar_tail_metrics(
    losses: pd.Series,
    cvar_series: pd.Series,
    var_series: pd.Series,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Compute tail risk metrics for CVaR including exceedances using losses.
    
    Args:
        losses: Series of actual losses (loss_t = -returns_t)
        cvar_series: Series of CVaR values (positive, represents expected loss given VaR exceeded)
        var_series: Series of VaR values
        confidence_level: Confidence level used for CVaR
        
    Returns:
        Dictionary of CVaR tail metrics
    """
    # CVaR violations: losses > CVaR
    cvar_violations = losses > cvar_series
    
    if cvar_violations.sum() == 0:
        return {
            'cvar_mean_exceedance': np.nan,
            'cvar_max_exceedance': np.nan,
            'cvar_std_exceedance': np.nan,
            'rmse_cvar_vs_losses': np.nan
        }
    
    # Exceedances (actual losses beyond CVaR) = (loss - CVaR) for violating points
    cvar_exceedances = (losses - cvar_series)[cvar_violations]
    
    cvar_mean_exceedance = cvar_exceedances.mean()
    cvar_max_exceedance = cvar_exceedances.max()
    cvar_std_exceedance = cvar_exceedances.std()
    
    # RMSE between CVaR and actual losses (only for violations)
    rmse_cvar_vs_losses = np.sqrt(np.mean((losses[cvar_violations] - cvar_series[cvar_violations])**2))
    
    return {
        'cvar_mean_exceedance': cvar_mean_exceedance,
        'cvar_max_exceedance': cvar_max_exceedance,
        'cvar_std_exceedance': cvar_std_exceedance,
        'rmse_cvar_vs_losses': rmse_cvar_vs_losses
    }


def compute_evt_tail_metrics(
    losses: pd.Series,
    var_series: pd.Series,
    threshold: float,
    xi: float,
    beta: float,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Compute EVT-specific tail metrics using losses.
    
    Args:
        losses: Series of actual losses (loss_t = -returns_t)
        var_series: Series of VaR values (positive, represents loss quantile)
        threshold: Threshold used for POT
        xi: GPD shape parameter (tail index)
        beta: GPD scale parameter
        confidence_level: Confidence level
        
    Returns:
        Dictionary of EVT-specific metrics
    """
    violations = losses > var_series
    
    # Expected shortfall exceedance (mean exceedance beyond VaR)
    # Compute ES exceedance only when violations > 0; otherwise NaN
    expected_shortfall_exceedance = np.nan
    if violations.sum() > 0:
        exceedances = (losses - var_series)[violations]
        expected_shortfall_exceedance = exceedances.mean()
    
    # Tail index (xi) - already provided
    tail_index_xi = xi
    
    # Shape-scale stability: coefficient of variation of exceedances
    shape_scale_stability = np.nan
    if violations.sum() > 0:
        exceedances = (losses - var_series)[violations]
        if len(exceedances) > 1 and exceedances.std() > 0:
            shape_scale_stability = exceedances.std() / exceedances.mean()
    
    return {
        'expected_shortfall_exceedance': expected_shortfall_exceedance,
        'tail_index_xi': tail_index_xi,
        'scale_beta': beta,
        'shape_scale_stability': shape_scale_stability
    }


def compute_distribution_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Compute distribution metrics for returns.
    
    Args:
        returns: Series of returns
        
    Returns:
        Dictionary of distribution metrics
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) < 3:
        return {
            'skewness': np.nan,
            'kurtosis': np.nan,
            'jarque_bera_p_value': np.nan
        }
    
    skewness = stats.skew(returns_clean)
    kurtosis = stats.kurtosis(returns_clean)  # Excess kurtosis
    
    # Jarque-Bera test for normality
    jb_stat, jb_p_value = stats.jarque_bera(returns_clean)
    
    return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'jarque_bera_p_value': jb_p_value,
        'jarque_bera_statistic': jb_stat
    }


def compute_runtime_metrics(runtimes: list[float]) -> Dict[str, float]:
    """
    Compute runtime performance metrics for execution batches.
    
    Runtime metrics describe execution batches, not individual model configurations.
    These should be stored once per experiment run, not per row.
    
    Args:
        runtimes: List of runtime values in seconds
        
    Returns:
        Dictionary of runtime metrics
    """
    if len(runtimes) == 0:
        return {
            'total_runtime_ms': np.nan,
            'p95_runtime_ms': np.nan,
            'mean_runtime_ms': np.nan,
            'median_runtime_ms': np.nan,
            'min_runtime_ms': np.nan,
            'max_runtime_ms': np.nan
        }
    
    runtimes_array = np.array(runtimes)
    
    return {
        'total_runtime_ms': np.sum(runtimes_array) * 1000,
        'p95_runtime_ms': np.percentile(runtimes_array, 95) * 1000,
        'mean_runtime_ms': np.mean(runtimes_array) * 1000,
        'median_runtime_ms': np.median(runtimes_array) * 1000,
        'min_runtime_ms': np.min(runtimes_array) * 1000,
        'max_runtime_ms': np.max(runtimes_array) * 1000
    }

