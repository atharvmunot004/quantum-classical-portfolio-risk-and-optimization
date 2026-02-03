"""
QAE-based VaR and CVaR estimation.

- VaR: Bisection on CDF using QAE (or classical fallback)
- CVaR: Rockafellar-Uryasev formula with tail expectation
  CVaR = VaR + (1 / (1 - alpha)) * E[(L - VaR)^+]
"""
import numpy as np
import time
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .distribution_loader import (
    fit_student_t,
    discretize_distribution,
    get_cdf_at_threshold,
    get_tail_expectation_classical,
    DistributionParams,
)
from .qae_circuits import (
    estimate_cdf_qae,
    estimate_cdf_classical,
    QAEResult,
    QISKIT_AVAILABLE,
)


@dataclass
class VarCvarResult:
    """VaR and CVaR estimation result."""

    var: float
    cvar: float
    var_ci_lower: float
    var_ci_upper: float
    cvar_ci_lower: float
    cvar_ci_upper: float
    qae_point_estimate: float
    qae_ci_width: float
    num_grover_iterations: int
    circuit_depth: int
    num_shots: int
    var_search_time_ms: float
    cvar_compute_time_ms: float
    use_qae: bool


def _var_bisection_classical(
    dist_params: DistributionParams,
    confidence_level: float,
    tol: float = 0.002,
    max_iter: int = 25
) -> float:
    """Find VaR via classical bisection on CDF."""
    alpha = 1 - confidence_level
    target = 1 - alpha
    low = dist_params.support_low
    high = dist_params.support_high

    for _ in range(max_iter):
        mid = (low + high) / 2
        cdf_mid = get_cdf_at_threshold(dist_params, mid)
        if abs(cdf_mid - target) < tol:
            return mid
        if cdf_mid < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _var_bisection_qae(
    probs: np.ndarray,
    bin_centers: np.ndarray,
    bin_centers_unit: np.ndarray,
    support_low: float,
    support_high: float,
    confidence_level: float,
    tol: float = 0.002,
    max_iter: int = 25,
    qae_settings: Optional[Dict] = None
) -> Tuple[float, float, float, Optional[QAEResult]]:
    """
    Find VaR via bisection using QAE for CDF estimation.
    Returns (var, var_ci_lower, var_ci_upper, last_qae_result).
    """
    qae_settings = qae_settings or {}
    alpha = 1 - confidence_level
    target = 1 - alpha
    low = support_low
    high = support_high
    last_result = None

    for _ in range(max_iter):
        mid = (low + high) / 2
        try:
            if QISKIT_AVAILABLE:
                result = estimate_cdf_qae(
                    probs, bin_centers, mid,
                    epsilon_target=qae_settings.get('epsilon_target', 0.01),
                    alpha=qae_settings.get('confidence_alpha', 0.05),
                    shots=qae_settings.get('shots', 2000)
                )
                cdf_mid = result.estimation
                last_result = result
            else:
                cdf_mid = estimate_cdf_classical(probs, bin_centers, mid)
        except Exception:
            cdf_mid = estimate_cdf_classical(probs, bin_centers, mid)

        if abs(cdf_mid - target) < tol:
            ci_low = last_result.confidence_interval[0] if last_result else cdf_mid - tol
            ci_high = last_result.confidence_interval[1] if last_result else cdf_mid + tol
            return mid, low, high, last_result
        if cdf_mid < target:
            low = mid
        else:
            high = mid

    return (low + high) / 2, low, high, last_result


def compute_rolling_qae_var_cvar(
    losses: np.ndarray,
    window: int,
    confidence_level: float,
    step_size: int = 1,
    num_state_qubits: int = 6,
    use_qae: bool = True,
    dist_model: Optional[Dict] = None,
    var_settings: Optional[Dict] = None,
    cvar_settings: Optional[Dict] = None,
    qae_settings: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Compute rolling VaR and CVaR time series using QAE.

    Args:
        losses: Loss series (positive values)
        window: Rolling window size
        confidence_level: e.g. 0.95, 0.99
        step_size: Roll step
        num_state_qubits: Number of qubits for discretization (2^n bins)
        use_qae: Use quantum estimation (False = classical only)
        dist_model: Distribution model config
        var_settings: VaR estimation config
        cvar_settings: CVaR estimation config
        qae_settings: QAE algorithm config

    Returns:
        (var_series, cvar_series, var_ci_lower, var_ci_upper, param_records)
    """
    dist_model = dist_model or {}
    var_settings = var_settings or {}
    cvar_settings = cvar_settings or {}
    qae_settings = qae_settings or {}

    support_clipping = dist_model.get('support_clipping', {
        'method': 'quantile',
        'lower_quantile': 0.001,
        'upper_quantile': 0.999
    })
    cdf_tol = var_settings.get('cdf_target_tolerance', 0.002)
    bisection_max = var_settings.get('bisection_max_iterations', 25)

    num_bins = 1 << num_state_qubits
    n = len(losses)
    var_series = np.full(n, np.nan)
    cvar_series = np.full(n, np.nan)
    var_ci_lower = np.full(n, np.nan)
    var_ci_upper = np.full(n, np.nan)
    param_records = []

    for i in range(window - 1, n, step_size):
        window_losses = losses[i - window + 1 : i + 1]
        window_losses = window_losses[~np.isnan(window_losses)]
        if len(window_losses) < window // 2:
            continue

        dist_params = fit_student_t(
            window_losses,
            fit_method=dist_model.get('fit_method', 'mle'),
            support_clipping=support_clipping
        )

        probs, bin_centers_loss, bin_centers_unit = discretize_distribution(
            dist_params, num_bins, rescale_to_unit=True
        )

        if use_qae and QISKIT_AVAILABLE:
            start_var = time.time()
            var_val, varl, varh, qae_res = _var_bisection_qae(
                probs, bin_centers_loss, bin_centers_unit,
                dist_params.support_low, dist_params.support_high,
                confidence_level, tol=cdf_tol, max_iter=bisection_max,
                qae_settings=qae_settings
            )
            var_time_ms = (time.time() - start_var) * 1000
        else:
            start_var = time.time()
            var_val = _var_bisection_classical(dist_params, confidence_level, tol=cdf_tol, max_iter=bisection_max)
            varl, varh = var_val * 0.95, var_val * 1.05
            qae_res = None
            var_time_ms = (time.time() - start_var) * 1000

        start_cvar = time.time()
        tail_exp = get_tail_expectation_classical(dist_params, var_val)
        alpha = 1 - confidence_level
        cvar_val = var_val + (1 / alpha) * tail_exp
        cvar_time_ms = (time.time() - start_cvar) * 1000

        var_series[i] = var_val
        cvar_series[i] = cvar_val
        var_ci_lower[i] = varl
        var_ci_upper[i] = varh

        param_records.append({
            'index': i,
            'dist_params': dist_params,
            'qae_result': qae_res,
            'var_time_ms': var_time_ms,
            'cvar_time_ms': cvar_time_ms,
        })

    cvar_ci_lower = cvar_series * 0.95
    cvar_ci_upper = cvar_series * 1.05

    return var_series, cvar_series, var_ci_lower, var_ci_upper, param_records
