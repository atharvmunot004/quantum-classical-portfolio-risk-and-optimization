"""
Backtesting module for QAOA CVaR asset-level evaluation.

Implements CVaR violation detection and accuracy metrics.
Uses loss space: violation when loss > CVaR (exceedance).
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict


def detect_var_violations(
    losses: pd.Series,
    var_series: pd.Series,
    confidence_level: float = 0.95
) -> pd.Series:
    """Detect VaR violations: loss > VaR."""
    return losses > var_series


def detect_cvar_violations(
    losses: pd.Series,
    cvar_series: pd.Series,
    var_series: pd.Series,
    confidence_level: float = 0.95
) -> pd.Series:
    """Detect CVaR violations: loss > CVaR (exceedance of tail threshold)."""
    return losses > cvar_series


def compute_hit_rate(violations: pd.Series) -> float:
    return float(violations.mean())


def compute_violation_ratio(violations: pd.Series, expected_violation_rate: float) -> float:
    actual_rate = violations.mean()
    if expected_violation_rate == 0:
        return np.inf if actual_rate > 0 else 1.0
    return float(actual_rate / expected_violation_rate)


def traffic_light_zone(violations: pd.Series, confidence_level: float = 0.95) -> str:
    n, x = len(violations), int(violations.sum())
    p = 1 - confidence_level
    exp_250 = 250 * p
    if exp_250 < 5:
        green_max, yellow_max = 4, 9
    else:
        green_max, yellow_max = int(exp_250 * 1.6), int(exp_250 * 2.4)
    scale = n / 250
    if x <= green_max * scale:
        return 'green'
    if x <= yellow_max * scale:
        return 'yellow'
    return 'red'


def compute_accuracy_metrics(
    losses: pd.Series,
    cvar_series: pd.Series,
    confidence_level: float = 0.95,
    compute_traffic_light: bool = True
) -> Dict[str, float]:
    """Compute hit rate, violation ratio, traffic light for CVaR exceedances."""
    violations = detect_cvar_violations(
        losses, cvar_series, cvar_series, confidence_level
    )
    exp_rate = 1 - confidence_level
    metrics = {
        'hit_rate': compute_hit_rate(violations),
        'violation_ratio': compute_violation_ratio(violations, exp_rate),
        'num_exceedances_vs_cvar': int(violations.sum()),
        'exceedance_rate_vs_cvar': float(violations.mean()),
        'total_observations': len(violations),
        'expected_violations': len(violations) * exp_rate,
    }
    if violations.sum() > 0:
        exceedances = (losses - cvar_series)[violations]
        metrics['mean_exceedance_given_cvar'] = float(exceedances.mean())
        metrics['max_exceedance_given_cvar'] = float(exceedances.max())
    else:
        metrics['mean_exceedance_given_cvar'] = np.nan
        metrics['max_exceedance_given_cvar'] = np.nan
    if compute_traffic_light:
        metrics['traffic_light_zone'] = traffic_light_zone(violations, confidence_level)
    return metrics
