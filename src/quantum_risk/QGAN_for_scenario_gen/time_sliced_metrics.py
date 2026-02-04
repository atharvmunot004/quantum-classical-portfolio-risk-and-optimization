"""Time-sliced metrics for QGAN scenario generation evaluation."""
import pandas as pd
import numpy as np
from typing import Dict, List
from .metrics import (
    compute_wasserstein_distance,
    compute_ks_statistic,
    compute_moment_errors,
    compute_tail_metrics
)


def compute_time_sliced_metrics(
    real_returns: pd.Series,
    generated_returns: pd.Series,
    real_losses: pd.Series,
    generated_losses: pd.Series,
    confidence_levels: List[float],
    slice_by: str = 'year',
    minimum_observations: int = 60,
) -> List[Dict]:
    """
    Compute time-sliced metrics for QGAN evaluation.
    
    Args:
        real_returns: Real return series
        generated_returns: Generated return series
        real_losses: Real loss series
        generated_losses: Generated loss series
        confidence_levels: List of confidence levels
        slice_by: 'year', 'quarter', or 'month'
        minimum_observations: Minimum observations per slice
        
    Returns:
        List of metric dicts per slice
    """
    # Align indices
    common_idx = real_returns.index.intersection(generated_returns.index)
    if len(common_idx) < minimum_observations:
        return []
    
    real_ret = real_returns.loc[common_idx]
    gen_ret = generated_returns.loc[common_idx]
    real_loss = real_losses.loc[common_idx]
    gen_loss = generated_losses.loc[common_idx]
    
    # Create groups
    if slice_by == 'year':
        groups = real_ret.index.year
        fmt = lambda x: str(x)
    elif slice_by == 'quarter':
        groups = real_ret.index.to_period('Q')
        fmt = lambda x: str(x)
    elif slice_by == 'month':
        groups = real_ret.index.to_period('M')
        fmt = lambda x: str(x)
    else:
        raise ValueError(f"Unknown slice_by: {slice_by}")
    
    result = []
    
    for period in sorted(groups.unique()):
        mask = groups == period
        pr = real_ret[mask]
        pg = gen_ret[mask]
        plr = real_loss[mask]
        pgl = gen_loss[mask]
        
        if len(pr) < minimum_observations:
            continue
        
        # Distribution metrics
        wasserstein = compute_wasserstein_distance(pr.values, pg.values)
        ks_stat = compute_ks_statistic(pr.values, pg.values)
        moment_errors = compute_moment_errors(pr.values, pg.values)
        
        # Tail metrics for each confidence level
        tail_metrics = compute_tail_metrics(
            plr.values, pgl.values, confidence_levels
        )
        
        rec = {
            'slice': fmt(period),
            'start_date': pr.index.min().strftime('%Y-%m-%d'),
            'end_date': pr.index.max().strftime('%Y-%m-%d'),
            'wasserstein_distance': wasserstein,
            'ks_statistic': ks_stat,
            **moment_errors,
            **tail_metrics
        }
        
        result.append(rec)
    
    return result
