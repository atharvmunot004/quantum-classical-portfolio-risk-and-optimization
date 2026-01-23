"""Backtesting module for VaR and CVaR evaluation.

IEEE convention used:
- loss_t = -return_t
- VaR_t and CVaR_t are POSITIVE loss magnitudes
- violation_t = (loss_t > VaR_t) for VaR, and (loss_t > CVaR_t) for CVaR

Implements:
- Hit rate and violation ratio
- Kupiec unconditional coverage test
- Christoffersen independence and conditional coverage tests
- Basel traffic-light zone indicator

Important correctness fix vs earlier versions:
- compute_accuracy_metrics ALWAYS computes Kupiec even if x=0 (no early return that forces NaNs).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


def detect_var_violations(returns: pd.Series, var_series: pd.Series, confidence_level: float = 0.95) -> pd.Series:
    """Detect VaR violations (exceedances): loss_t > VaR_t."""
    losses = -returns
    return losses > var_series


def detect_cvar_violations(
    returns: pd.Series,
    cvar_series: pd.Series,
    var_series: pd.Series,
    confidence_level: float = 0.95
) -> pd.Series:
    """Detect CVaR violations: loss_t > CVaR_t."""
    losses = -returns
    return losses > cvar_series


def compute_hit_rate(violations: pd.Series) -> float:
    """Proportion of violations."""
    return float(violations.mean()) if len(violations) else float('nan')


def compute_violation_ratio(violations: pd.Series, expected_violation_rate: float) -> float:
    """Actual violation rate divided by expected violation rate."""
    actual_rate = float(violations.mean()) if len(violations) else float('nan')
    if expected_violation_rate == 0:
        return np.inf if actual_rate > 0 else 1.0
    return actual_rate / expected_violation_rate


def kupiec_test(
    violations: pd.Series,
    confidence_level: float = 0.95,
    significance_level: float = 0.05
) -> Dict[str, float]:
    """Kupiec (1995) unconditional coverage LR test."""
    n = int(len(violations))
    x = int(violations.sum())
    p = float(1.0 - confidence_level)

    if n == 0:
        return {'test_statistic': np.nan, 'p_value': np.nan, 'reject_null': False}

    # Likelihood-ratio form with stable edge handling
    if x == 0:
        lr_stat = -2.0 * n * np.log(max(1.0 - p, 1e-15))
    elif x == n:
        lr_stat = -2.0 * n * np.log(max(p, 1e-15))
    else:
        phat = x / n
        lr_stat = -2.0 * (
            x * np.log(max(p, 1e-15)) + (n - x) * np.log(max(1.0 - p, 1e-15))
            - x * np.log(max(phat, 1e-15)) - (n - x) * np.log(max(1.0 - phat, 1e-15))
        )

    lr_stat = float(max(0.0, lr_stat))
    p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=1))
    reject_null = bool(p_value < significance_level)

    return {'test_statistic': lr_stat, 'p_value': p_value, 'reject_null': reject_null}


def christoffersen_test(
    violations: pd.Series,
    confidence_level: float = 0.95,
    significance_level: float = 0.05
) -> Dict[str, float]:
    """Christoffersen (1998) independence and conditional coverage tests."""
    n = int(len(violations))
    v = violations.astype(int).to_numpy()

    if n < 2:
        return {
            'independence_test_statistic': np.nan,
            'independence_p_value': np.nan,
            'independence_reject_null': False,
            'conditional_coverage_test_statistic': np.nan,
            'conditional_coverage_p_value': np.nan,
            'conditional_coverage_reject_null': False
        }

    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        if v[i - 1] == 0 and v[i] == 0:
            n00 += 1
        elif v[i - 1] == 0 and v[i] == 1:
            n01 += 1
        elif v[i - 1] == 1 and v[i] == 0:
            n10 += 1
        else:
            n11 += 1

    n0 = n00 + n01
    n1 = n10 + n11

    # If there are no transitions from one of the states, independence is not identifiable.
    if n0 == 0 or n1 == 0:
        lr_ind = np.nan
        p_ind = np.nan
    else:
        pi0 = n01 / n0
        pi1 = n11 / n1
        pi = (n01 + n11) / (n0 + n1)

        def _ll(k: int, total: int, prob: float) -> float:
            prob = float(np.clip(prob, 1e-15, 1 - 1e-15))
            return k * np.log(prob) + (total - k) * np.log(1.0 - prob)

        ll_null = _ll(n01 + n11, n0 + n1, pi)
        ll_alt = _ll(n01, n0, pi0) + _ll(n11, n1, pi1)
        lr_ind = float(max(0.0, -2.0 * (ll_null - ll_alt)))
        p_ind = float(1.0 - stats.chi2.cdf(lr_ind, df=1))

    reject_ind = bool((not np.isnan(p_ind)) and (p_ind < significance_level))

    # Conditional coverage = UC + IND
    uc = kupiec_test(violations, confidence_level=confidence_level, significance_level=significance_level)
    lr_uc = uc['test_statistic']

    if np.isnan(lr_ind) or np.isnan(lr_uc):
        lr_cc = np.nan
        p_cc = np.nan
    else:
        lr_cc = float(max(0.0, lr_uc + lr_ind))
        p_cc = float(1.0 - stats.chi2.cdf(lr_cc, df=2))

    reject_cc = bool((not np.isnan(p_cc)) and (p_cc < significance_level))

    return {
        'independence_test_statistic': lr_ind,
        'independence_p_value': p_ind,
        'independence_reject_null': reject_ind,
        'conditional_coverage_test_statistic': lr_cc,
        'conditional_coverage_p_value': p_cc,
        'conditional_coverage_reject_null': reject_cc
    }


def traffic_light_zone(violations: pd.Series, confidence_level: float = 0.95) -> str:
    """Basel traffic-light zone (scaled from 250 obs)."""
    n = int(len(violations))
    x = int(violations.sum())
    p = float(1.0 - confidence_level)

    if n == 0:
        return 'invalid'

    expected_per_250 = 250.0 * p
    if expected_per_250 < 5:
        green_max, yellow_max = 4, 9
    else:
        green_max = int(expected_per_250 * 1.6)
        yellow_max = int(expected_per_250 * 2.4)

    scale = n / 250.0
    green_thr = green_max * scale
    yellow_thr = yellow_max * scale

    if x <= green_thr:
        return 'green'
    if x <= yellow_thr:
        return 'yellow'
    return 'red'


def compute_accuracy_metrics(returns: pd.Series, var_series: pd.Series, confidence_level: float = 0.95) -> Dict[str, float]:
    """Compute VaR backtesting metrics (IEEE conventions).

    Returns a dict suitable for direct column expansion.

    IMPORTANT:
    - We always compute Kupiec UC test (even if x=0) to avoid NaN-filled tables.
    - Christoffersen may be NaN when identifiability conditions fail.
    """
    violations = detect_var_violations(returns, var_series, confidence_level)
    expected_rate = float(1.0 - confidence_level)

    n = int(len(violations))
    x = int(violations.sum())

    hit_rate = float(x / n) if n else np.nan
    violation_ratio_val = compute_violation_ratio(violations, expected_rate) if n else np.nan

    kup = kupiec_test(violations, confidence_level=confidence_level)
    chr_ = christoffersen_test(violations, confidence_level=confidence_level)

    return {
        'hit_rate': hit_rate,
        'violation_ratio': float(violation_ratio_val),
        'kupiec_unconditional_coverage': float(kup['p_value']),
        'kupiec_test_statistic': float(kup['test_statistic']),
        'kupiec_reject_null': bool(kup['reject_null']),
        'christoffersen_independence': float(chr_['independence_p_value']) if 'independence_p_value' in chr_ else np.nan,
        'christoffersen_independence_statistic': float(chr_['independence_test_statistic']) if 'independence_test_statistic' in chr_ else np.nan,
        'christoffersen_independence_reject_null': bool(chr_.get('independence_reject_null', False)),
        'christoffersen_conditional_coverage': float(chr_['conditional_coverage_p_value']) if 'conditional_coverage_p_value' in chr_ else np.nan,
        'christoffersen_conditional_coverage_statistic': float(chr_['conditional_coverage_test_statistic']) if 'conditional_coverage_test_statistic' in chr_ else np.nan,
        'christoffersen_conditional_coverage_reject_null': bool(chr_.get('conditional_coverage_reject_null', False)),
        'traffic_light_zone': traffic_light_zone(violations, confidence_level=confidence_level),
        'num_violations': x,
        'total_observations': n,
        'expected_violations': float(n * expected_rate),
        'configuration_valid': bool(n > 0)
    }
