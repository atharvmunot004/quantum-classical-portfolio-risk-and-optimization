"""
Loss discretization for QAOA binary tail-risk encoding.

Uses quantile-grid method to discretize losses into levels for Ising cost Hamiltonian.
Each level corresponds to a basis state in the computational basis.
"""
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class DiscretizationResult:
    """Result of loss discretization."""

    loss_levels: np.ndarray  # shape (num_levels,) - loss value at each level
    level_indices: np.ndarray  # 0..num_levels-1
    support_low: float
    support_high: float
    rescale_to_unit: bool


def discretize_losses_quantile_grid(
    losses: np.ndarray,
    num_levels: int = 256,
    clip_quantiles: Tuple[float, float] = (0.001, 0.999),
    rescale_to_unit_interval: bool = True,
) -> DiscretizationResult:
    """
    Discretize loss series using quantile-grid method.

    Creates num_levels evenly spaced quantiles of the empirical loss distribution.
    Each level gets the corresponding quantile value as its loss. This produces
    a mapping from basis state index (0..num_levels-1) to loss value for the
    cost Hamiltonian.

    Args:
        losses: 1D array of loss values (positive, e.g. negative returns)
        num_levels: Number of discretization levels (typically 2^num_qubits)
        clip_quantiles: (lower, upper) quantiles for support clipping
        rescale_to_unit_interval: If True, loss_levels are in [0,1]

    Returns:
        DiscretizationResult with loss_levels, support bounds, etc.
    """
    losses_clean = losses[~np.isnan(losses)]
    losses_clean = losses_clean[np.isfinite(losses_clean)]
    if len(losses_clean) < 2:
        raise ValueError("Insufficient loss observations for discretization")

    q_low, q_high = clip_quantiles
    support_low = float(np.quantile(losses_clean, q_low))
    support_high = float(np.quantile(losses_clean, q_high))
    if support_high <= support_low:
        support_high = float(np.max(losses_clean))
        support_low = float(np.min(losses_clean))
        if support_high <= support_low:
            support_high = support_low + 1e-8

    # Quantile grid: evenly spaced quantiles in (0, 1)
    quantiles = np.linspace(0, 1, num_levels + 2)[1:-1]  # exclude 0 and 1
    loss_levels = np.quantile(losses_clean, quantiles)
    loss_levels = np.clip(loss_levels, support_low, support_high)

    if rescale_to_unit_interval:
        span = support_high - support_low
        if span > 0:
            loss_levels = (loss_levels - support_low) / span
        else:
            loss_levels = np.zeros_like(loss_levels)

    level_indices = np.arange(num_levels)
    return DiscretizationResult(
        loss_levels=loss_levels,
        level_indices=level_indices,
        support_low=support_low,
        support_high=support_high,
        rescale_to_unit=rescale_to_unit_interval,
    )
