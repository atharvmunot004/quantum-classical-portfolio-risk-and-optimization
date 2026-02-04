"""
Distribution fitting and discretization for QAE state preparation.

Fits parametric (Student-t) distribution to losses with support clipping,
then produces discretized probability vectors for quantum encoding.
"""
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DistributionParams:
    """Fitted distribution parameters."""

    family: str
    params: Dict[str, float]
    support_low: float
    support_high: float
    success: bool
    n_obs: int


def fit_student_t(
    losses: np.ndarray,
    fit_method: str = 'mle',
    support_clipping: Optional[Dict] = None
) -> DistributionParams:
    """
    Fit Student-t distribution to losses.

    Args:
        losses: Array of loss values (positive)
        fit_method: 'mle' for maximum likelihood
        support_clipping: Dict with 'method', 'lower_quantile', 'upper_quantile'
                         e.g. {'method': 'quantile', 'lower_quantile': 0.001, 'upper_quantile': 0.999}

    Returns:
        DistributionParams with df, loc, scale and support bounds
    """
    losses_clean = losses[~np.isnan(losses)]
    losses_clean = losses_clean[losses_clean > 0]  # Losses should be positive

    if len(losses_clean) < 10:
        return DistributionParams(
            family='student_t',
            params={'df': 5.0, 'loc': np.mean(losses_clean), 'scale': np.std(losses_clean)},
            support_low=0.0,
            support_high=np.max(losses_clean) * 2,
            success=False,
            n_obs=len(losses_clean)
        )

    support_low = float(np.percentile(losses_clean, 0.1))
    support_high = float(np.percentile(losses_clean, 99.9))

    if support_clipping and support_clipping.get('method') == 'quantile':
        q_low = support_clipping.get('lower_quantile', 0.001)
        q_high = support_clipping.get('upper_quantile', 0.999)
        support_low = float(np.quantile(losses_clean, q_low))
        support_high = float(np.quantile(losses_clean, q_high))

    # Clip losses to support for fitting
    clipped = losses_clean[(losses_clean >= support_low) & (losses_clean <= support_high)]

    if len(clipped) < 5:
        clipped = losses_clean

    success = True
    try:
        if fit_method == 'mle':
            df, loc, scale = stats.t.fit(clipped)
        else:
            # Method of moments fallback
            df = 5.0
            loc = np.mean(clipped)
            scale = np.std(clipped)
    except Exception:
        df, loc, scale = 5.0, np.mean(clipped), max(np.std(clipped), 1e-8)
        success = False

    params = {'df': float(df), 'loc': float(loc), 'scale': float(scale)}

    return DistributionParams(
        family='student_t',
        params=params,
        support_low=support_low,
        support_high=support_high,
        success=success,
        n_obs=len(losses_clean)
    )


def discretize_distribution(
    dist_params: DistributionParams,
    num_bins: int,
    rescale_to_unit: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Discretize the fitted distribution on a uniform grid.

    Args:
        dist_params: Fitted distribution parameters
        num_bins: Number of bins (2^num_state_qubits)
        rescale_to_unit: If True, return loss values scaled to [0,1]

    Returns:
        Tuple of (probabilities, bin_centers_loss, bin_centers_unit)
        - probabilities: shape (num_bins,), sums to 1
        - bin_centers_loss: loss values at bin centers
        - bin_centers_unit: [0,1] scaled values if rescale_to_unit
    """
    low = dist_params.support_low
    high = dist_params.support_high
    df = dist_params.params['df']
    loc = dist_params.params['loc']
    scale = dist_params.params['scale']

    edges = np.linspace(low, high, num_bins + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # CDF at edges
    cdf_vals = stats.t.cdf(edges, df=df, loc=loc, scale=scale)
    probs = np.diff(cdf_vals)

    probs = np.maximum(probs, 1e-12)
    probs = probs / probs.sum()

    if rescale_to_unit:
        bin_centers_unit = (bin_centers - low) / (high - low) if high > low else np.linspace(0, 1, num_bins)
        return probs, bin_centers, bin_centers_unit

    return probs, bin_centers, bin_centers


def get_cdf_at_threshold(
    dist_params: DistributionParams,
    threshold: float
) -> float:
    """Compute CDF(L <= threshold) for the fitted distribution."""
    df = dist_params.params['df']
    loc = dist_params.params['loc']
    scale = dist_params.params['scale']
    return float(stats.t.cdf(threshold, df=df, loc=loc, scale=scale))


def get_tail_expectation_classical(
    dist_params: DistributionParams,
    var: float
) -> float:
    """
    Compute E[(L - VaR)^+] classically for the fitted Student-t.
    Uses fixed-point quadrature (fast, stable).
    """
    df = dist_params.params['df']
    loc = dist_params.params['loc']
    scale = dist_params.params['scale']

    p_exceed = 1 - stats.t.cdf(var, df=df, loc=loc, scale=scale)
    if p_exceed < 1e-12:
        return 0.0

    upper = float(stats.t.ppf(0.9999, df=df, loc=loc, scale=scale))
    upper = max(upper, var + 5 * scale)
    x = np.linspace(var, upper, 64)
    y = (x - var) * stats.t.pdf(x, df=df, loc=loc, scale=scale)
    result = np.trapz(y, x)
    return float(max(0, result))
