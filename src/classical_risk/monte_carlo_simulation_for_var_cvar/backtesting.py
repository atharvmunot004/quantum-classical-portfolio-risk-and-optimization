"""
Backtesting module for VaR and CVaR evaluation.

Implements VaR and CVaR violation detection and accuracy metrics including:
- Hit rate and violation ratio
- Kupiec unconditional coverage test
- Christoffersen independence and conditional coverage tests
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional


def detect_var_violations(
    returns: pd.Series,
    var_series: pd.Series,
    confidence_level: float = 0.95
) -> pd.Series:
    """
    Detect VaR violations (exceedances).
    
    A violation occurs when the actual return is less than -VaR.
    
    Args:
        returns: Series of actual returns
        var_series: Series of VaR values
        confidence_level: Confidence level used for VaR calculation
        
    Returns:
        Boolean Series indicating violations (True = violation)
    """
    # VaR is typically positive, so violation is when return < -VaR
    violations = returns < -var_series
    
    return violations


def compute_hit_rate(violations: pd.Series) -> float:
    """
    Compute hit rate (proportion of violations).
    
    Args:
        violations: Boolean Series indicating violations
        
    Returns:
        Hit rate (proportion of violations)
    """
    return violations.mean()


def compute_violation_ratio(
    violations: pd.Series,
    expected_violation_rate: float
) -> float:
    """
    Compute violation ratio (actual violations / expected violations).
    
    Args:
        violations: Boolean Series indicating violations
        expected_violation_rate: Expected violation rate (e.g., 0.05 for 95% VaR)
        
    Returns:
        Violation ratio
    """
    actual_rate = violations.mean()
    if expected_violation_rate == 0:
        return np.inf if actual_rate > 0 else 1.0
    return actual_rate / expected_violation_rate


def kupiec_test(
    violations: pd.Series,
    confidence_level: float = 0.95,
    significance_level: float = 0.05
) -> Dict[str, float]:
    """
    Kupiec (1995) unconditional coverage test.
    
    Tests whether the observed violation rate matches the expected rate.
    
    Args:
        violations: Boolean Series indicating violations
        confidence_level: Confidence level used for VaR
        significance_level: Significance level for the test
        
    Returns:
        Dictionary with test statistics and p-value
    """
    n = len(violations)
    x = violations.sum()  # Number of violations
    p = 1 - confidence_level  # Expected violation rate
    
    if n == 0 or x == 0:
        return {
            'test_statistic': np.nan,
            'p_value': np.nan,
            'reject_null': False
        }
    
    # Likelihood ratio test statistic
    if x == 0:
        lr_stat = -2 * n * np.log(1 - p)
    elif x == n:
        lr_stat = -2 * n * np.log(p)
    else:
        lr_stat = -2 * (
            x * np.log(p) + (n - x) * np.log(1 - p) -
            x * np.log(x / n) - (n - x) * np.log((n - x) / n)
        )
    
    # Chi-square distribution with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    reject_null = p_value < significance_level
    
    return {
        'test_statistic': lr_stat,
        'p_value': p_value,
        'reject_null': reject_null
    }


def christoffersen_test(
    violations: pd.Series,
    confidence_level: float = 0.95,
    significance_level: float = 0.05
) -> Dict[str, float]:
    """
    Christoffersen (1998) independence and conditional coverage tests.
    
    Tests whether violations are independent (no clustering).
    
    Args:
        violations: Boolean Series indicating violations
        confidence_level: Confidence level used for VaR
        significance_level: Significance level for the test
        
    Returns:
        Dictionary with test statistics and p-values
    """
    n = len(violations)
    violations_array = violations.values.astype(int)
    
    # Count transitions
    n00 = 0  # No violation -> No violation
    n01 = 0  # No violation -> Violation
    n10 = 0  # Violation -> No violation
    n11 = 0  # Violation -> Violation
    
    for i in range(1, n):
        if violations_array[i-1] == 0 and violations_array[i] == 0:
            n00 += 1
        elif violations_array[i-1] == 0 and violations_array[i] == 1:
            n01 += 1
        elif violations_array[i-1] == 1 and violations_array[i] == 0:
            n10 += 1
        elif violations_array[i-1] == 1 and violations_array[i] == 1:
            n11 += 1
    
    n0 = n00 + n01  # Number of non-violations
    n1 = n10 + n11  # Number of violations (as previous state)
    
    if n0 == 0 or n1 == 0:
        return {
            'independence_test_statistic': np.nan,
            'independence_p_value': np.nan,
            'independence_reject_null': False,
            'conditional_coverage_test_statistic': np.nan,
            'conditional_coverage_p_value': np.nan,
            'conditional_coverage_reject_null': False
        }
    
    # Transition probabilities
    pi0 = n01 / n0 if n0 > 0 else 0
    pi1 = n11 / n1 if n1 > 0 else 0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0
    
    # Independence test statistic
    term1 = 0.0
    if n0 + n1 > 0:
        if pi > 0:
            term1 += n1 * np.log(pi)
        if pi < 1:
            term1 += n0 * np.log(1 - pi)
    
    term2 = 0.0
    if n01 > 0 and pi0 > 0:
        term2 += n01 * np.log(pi0)
    if n00 > 0 and pi0 < 1:
        term2 += n00 * np.log(1 - pi0)
    
    term3 = 0.0
    if n11 > 0 and pi1 > 0:
        term3 += n11 * np.log(pi1)
    if n10 > 0 and pi1 < 1:
        term3 += n10 * np.log(1 - pi1)
    
    lr_ind = -2 * (term1 - term2 - term3)
    
    if (n01 == 0 and n11 == 0) or (n00 == 0 and n10 == 0) or (n0 == 0 or n1 == 0):
        lr_ind = 0
    elif np.isnan(lr_ind) or np.isinf(lr_ind):
        lr_ind = np.nan
    
    if np.isnan(lr_ind) or np.isinf(lr_ind):
        independence_p_value = np.nan
        independence_reject_null = False
    else:
        independence_p_value = 1 - stats.chi2.cdf(max(0, lr_ind), df=1)
        independence_reject_null = independence_p_value < significance_level
    
    # Conditional coverage test
    p = 1 - confidence_level
    x = violations.sum()
    
    if x == 0 or x == n:
        lr_cc = lr_ind
    else:
        log_p = np.log(p) if p > 0 else 0
        log_1_minus_p = np.log(1 - p) if p < 1 else 0
        log_x_n = np.log(x / n) if x > 0 else 0
        log_n_minus_x_n = np.log((n - x) / n) if (n - x) > 0 else 0
        
        lr_uc = -2 * (
            x * log_p + (n - x) * log_1_minus_p -
            x * log_x_n - (n - x) * log_n_minus_x_n
        )
        
        if np.isnan(lr_uc) or np.isinf(lr_uc):
            lr_cc = lr_ind
        else:
            lr_cc = lr_uc + lr_ind
    
    conditional_coverage_p_value = 1 - stats.chi2.cdf(lr_cc, df=2)
    conditional_coverage_reject_null = conditional_coverage_p_value < significance_level
    
    return {
        'independence_test_statistic': lr_ind,
        'independence_p_value': independence_p_value,
        'independence_reject_null': independence_reject_null,
        'conditional_coverage_test_statistic': lr_cc,
        'conditional_coverage_p_value': conditional_coverage_p_value,
        'conditional_coverage_reject_null': conditional_coverage_reject_null
    }


def traffic_light_zone(
    violations: pd.Series,
    confidence_level: float = 0.95,
    window_size_days: int = 250,
    alpha: float = 0.99
) -> str:
    """
    Basel Traffic Light approach for VaR backtesting.
    
    Implements the Basel Committee's traffic light system for VaR backtesting.
    The system classifies portfolios into green, yellow, or red zones based on
    the number of violations relative to expected violations.
    
    Args:
        violations: Boolean Series indicating violations
        confidence_level: Confidence level used for VaR (e.g., 0.95 for 95% VaR)
        window_size_days: Window size in days for Basel method (default: 250)
        alpha: Significance level for Basel method (default: 0.99)
        
    Returns:
        Zone name ('green', 'yellow', 'red')
    """
    n = len(violations)
    x = violations.sum()
    p = 1 - confidence_level
    
    # Expected violations in the reference window
    expected_per_window = window_size_days * p
    
    # Basel thresholds are based on 250-day window
    # For 99% confidence (alpha=0.99), the thresholds are:
    # Green: 0-4 violations in 250 days
    # Yellow: 5-9 violations in 250 days  
    # Red: 10+ violations in 250 days
    
    # For other confidence levels, scale accordingly
    # The Basel framework uses these thresholds for 99% VaR
    if confidence_level >= 0.99:
        # Standard Basel thresholds for 99% VaR
        green_max = 4
        yellow_max = 9
    else:
        # Scale thresholds based on expected violations
        # Green zone: violations <= 0.016 * expected (roughly)
        # Yellow zone: violations <= 0.036 * expected (roughly)
        green_max = max(1, int(expected_per_window * 0.016))
        yellow_max = max(green_max + 1, int(expected_per_window * 0.036))
    
    # Scale to actual window size
    scale_factor = n / window_size_days
    green_threshold = green_max * scale_factor
    yellow_threshold = yellow_max * scale_factor
    
    if x <= green_threshold:
        return 'green'
    elif x <= yellow_threshold:
        return 'yellow'
    else:
        return 'red'


def detect_cvar_violations(
    returns: pd.Series,
    cvar_series: pd.Series,
    var_series: pd.Series,
    confidence_level: float = 0.95
) -> pd.Series:
    """
    Detect CVaR violations (exceedances).
    
    A CVaR violation occurs when the actual return is less than -CVaR.
    
    Args:
        returns: Series of actual returns
        cvar_series: Series of CVaR values
        var_series: Series of VaR values (needed to identify tail events)
        confidence_level: Confidence level used for CVaR calculation
        
    Returns:
        Boolean Series indicating CVaR violations (True = violation)
    """
    # CVaR violation: return < -CVaR (more severe than VaR)
    cvar_violations = returns < -cvar_series
    
    return cvar_violations


def compute_accuracy_metrics(
    returns: pd.Series,
    var_series: pd.Series,
    confidence_level: float = 0.95,
    traffic_light_config: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Compute all accuracy metrics for VaR backtesting.
    
    Args:
        returns: Series of actual returns
        var_series: Series of VaR values
        confidence_level: Confidence level used for VaR
        traffic_light_config: Optional dict with 'method', 'window_size_days', 'alpha' keys
        
    Returns:
        Dictionary of accuracy metrics
    """
    violations = detect_var_violations(returns, var_series, confidence_level)
    expected_violation_rate = 1 - confidence_level
    
    hit_rate = compute_hit_rate(violations)
    violation_ratio_val = compute_violation_ratio(violations, expected_violation_rate)
    
    kupiec_results = kupiec_test(violations, confidence_level)
    christoffersen_results = christoffersen_test(violations, confidence_level)
    
    # Traffic light with config
    if traffic_light_config and traffic_light_config.get('enabled', True):
        window_size_days = traffic_light_config.get('window_size_days', 250)
        alpha = traffic_light_config.get('alpha', 0.99)
        traffic_light = traffic_light_zone(violations, confidence_level, window_size_days, alpha)
    else:
        traffic_light = traffic_light_zone(violations, confidence_level)
    
    metrics = {
        'hit_rate': hit_rate,
        'violation_ratio': violation_ratio_val,
        'kupiec_unconditional_coverage': kupiec_results['p_value'],
        'kupiec_test_statistic': kupiec_results['test_statistic'],
        'kupiec_reject_null': kupiec_results['reject_null'],
        'christoffersen_independence': christoffersen_results['independence_p_value'],
        'christoffersen_independence_statistic': christoffersen_results['independence_test_statistic'],
        'christoffersen_independence_reject_null': christoffersen_results['independence_reject_null'],
        'christoffersen_conditional_coverage': christoffersen_results['conditional_coverage_p_value'],
        'christoffersen_conditional_coverage_statistic': christoffersen_results['conditional_coverage_test_statistic'],
        'christoffersen_conditional_coverage_reject_null': christoffersen_results['conditional_coverage_reject_null'],
        'traffic_light_zone': traffic_light,
        'num_violations': violations.sum(),
        'total_observations': len(violations),
        'expected_violations': len(violations) * expected_violation_rate
    }
    
    return metrics

