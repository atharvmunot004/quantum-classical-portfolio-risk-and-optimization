"""
Backtesting module for QAE VaR and CVaR evaluation.

Implements VaR/CVaR violation detection and accuracy metrics.
Uses loss space: violation when loss > VaR (or loss > CVaR).
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
    """Detect CVaR violations: loss > CVaR."""
    return losses > cvar_series


def compute_hit_rate(violations: pd.Series) -> float:
    return violations.mean()


def compute_violation_ratio(violations: pd.Series, expected_violation_rate: float) -> float:
    actual_rate = violations.mean()
    if expected_violation_rate == 0:
        return np.inf if actual_rate > 0 else 1.0
    return actual_rate / expected_violation_rate


def kupiec_test(
    violations: pd.Series,
    confidence_level: float = 0.95,
    significance_level: float = 0.05
) -> Dict[str, float]:
    n = len(violations)
    x = violations.sum()
    p = 1 - confidence_level
    if n == 0 or (x == 0 and p > 0):
        return {'test_statistic': np.nan, 'p_value': np.nan, 'reject_null': False}
    if x == 0:
        lr_stat = -2 * n * np.log(1 - p)
    elif x == n:
        lr_stat = -2 * n * np.log(p)
    else:
        lr_stat = -2 * (
            x * np.log(p) + (n - x) * np.log(1 - p) -
            x * np.log(x / n) - (n - x) * np.log((n - x) / n)
        )
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    return {'test_statistic': lr_stat, 'p_value': p_value, 'reject_null': p_value < significance_level}


def christoffersen_test(
    violations: pd.Series,
    confidence_level: float = 0.95,
    significance_level: float = 0.05
) -> Dict[str, float]:
    n = len(violations)
    v = violations.values.astype(int)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        if v[i-1] == 0 and v[i] == 0: n00 += 1
        elif v[i-1] == 0 and v[i] == 1: n01 += 1
        elif v[i-1] == 1 and v[i] == 0: n10 += 1
        else: n11 += 1
    n0, n1 = n00 + n01, n10 + n11
    if n0 == 0 or n1 == 0:
        return {
            'independence_test_statistic': np.nan, 'independence_p_value': np.nan,
            'independence_reject_null': False, 'conditional_coverage_test_statistic': np.nan,
            'conditional_coverage_p_value': np.nan, 'conditional_coverage_reject_null': False
        }
    pi0 = n01 / n0
    pi1 = n11 / n1
    pi = (n01 + n11) / (n0 + n1)
    term1 = n1 * np.log(pi) + n0 * np.log(1 - pi) if 0 < pi < 1 else 0
    term2 = n01 * np.log(pi0) + n00 * np.log(1 - pi0) if 0 < pi0 < 1 else 0
    term3 = n11 * np.log(pi1) + n10 * np.log(1 - pi1) if 0 < pi1 < 1 else 0
    lr_ind = -2 * (term1 - term2 - term3)
    lr_ind = max(0, lr_ind) if not (np.isnan(lr_ind) or np.isinf(lr_ind)) else np.nan
    ind_p = 1 - stats.chi2.cdf(lr_ind, df=1) if not np.isnan(lr_ind) else np.nan
    p, x = 1 - confidence_level, v.sum()
    if x == 0 or x == n:
        lr_cc = lr_ind
    else:
        lr_uc = -2 * (x * np.log(p) + (n - x) * np.log(1 - p) - x * np.log(x/n) - (n-x) * np.log((n-x)/n))
        lr_cc = lr_uc + lr_ind
    cc_p = 1 - stats.chi2.cdf(lr_cc, df=2)
    return {
        'independence_test_statistic': lr_ind,
        'independence_p_value': ind_p,
        'independence_reject_null': ind_p < significance_level if not np.isnan(ind_p) else False,
        'conditional_coverage_test_statistic': lr_cc,
        'conditional_coverage_p_value': cc_p,
        'conditional_coverage_reject_null': cc_p < significance_level
    }


def traffic_light_zone(violations: pd.Series, confidence_level: float = 0.95) -> str:
    n, x = len(violations), violations.sum()
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
    var_series: pd.Series,
    confidence_level: float = 0.95,
    compute_traffic_light: bool = True
) -> Dict[str, float]:
    violations = detect_var_violations(losses, var_series, confidence_level)
    exp_rate = 1 - confidence_level
    kupiec = kupiec_test(violations, confidence_level)
    christ = christoffersen_test(violations, confidence_level)
    metrics = {
        'hit_rate': compute_hit_rate(violations),
        'violation_ratio': compute_violation_ratio(violations, exp_rate),
        'kupiec_unconditional_coverage': kupiec['p_value'],
        'kupiec_test_statistic': kupiec['test_statistic'],
        'kupiec_reject_null': kupiec['reject_null'],
        'christoffersen_independence': christ['independence_p_value'],
        'christoffersen_independence_statistic': christ['independence_test_statistic'],
        'christoffersen_independence_reject_null': christ['independence_reject_null'],
        'christoffersen_conditional_coverage': christ['conditional_coverage_p_value'],
        'christoffersen_conditional_coverage_statistic': christ['conditional_coverage_test_statistic'],
        'christoffersen_conditional_coverage_reject_null': christ['conditional_coverage_reject_null'],
        'num_violations': violations.sum(),
        'total_observations': len(violations),
        'expected_violations': len(violations) * exp_rate,
    }
    metrics['traffic_light_zone'] = traffic_light_zone(violations, confidence_level) if compute_traffic_light else None
    return metrics
