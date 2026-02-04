"""Time-sliced metrics for QAOA CVaR backtesting."""
import pandas as pd
import numpy as np
from typing import Dict, List

from .backtesting import (
    detect_cvar_violations,
    compute_hit_rate,
    compute_violation_ratio,
)


def compute_time_sliced_metrics(
    losses: pd.Series,
    cvar_series: pd.Series,
    var_series: pd.Series = None,
    confidence_level: float = 0.95,
    slice_by: str = 'year',
    minimum_observations: int = 60,
) -> List[Dict]:
    """
    Compute time-sliced exceedance metrics: num_exceedances_vs_cvar,
    exceedance_rate_vs_cvar, mean_exceedance_given_cvar, max_exceedance_given_cvar.
    """
    common = losses.index.intersection(cvar_series.index)
    if len(common) < minimum_observations:
        return []
    aligned_losses = losses.loc[common]
    aligned_cvar = cvar_series.loc[common]
    var_for_detect = var_series.loc[common] if var_series is not None else aligned_cvar

    if slice_by == 'year':
        groups = aligned_losses.index.year
        fmt = lambda x: str(x)
    elif slice_by == 'quarter':
        groups = aligned_losses.index.to_period('Q')
        fmt = lambda x: str(x)
    elif slice_by == 'month':
        groups = aligned_losses.index.to_period('M')
        fmt = lambda x: str(x)
    else:
        raise ValueError(f"Unknown slice_by: {slice_by}")

    exp_rate = 1 - confidence_level
    result = []
    for period in sorted(groups.unique()):
        mask = groups == period
        pl = aligned_losses[mask]
        pc = aligned_cvar[mask]
        pv = var_for_detect[mask] if var_for_detect is not None else pc
        if len(pl) < minimum_observations:
            continue
        cv = detect_cvar_violations(pl, pc, pv, confidence_level)
        rec = {
            'slice': fmt(period),
            'start_date': pl.index.min().strftime('%Y-%m-%d'),
            'end_date': pl.index.max().strftime('%Y-%m-%d'),
            'num_exceedances_vs_cvar': int(cv.sum()),
            'exceedance_rate_vs_cvar': float(compute_hit_rate(cv)),
            'violation_ratio': float(compute_violation_ratio(cv, exp_rate)),
        }
        if cv.sum() > 0:
            exceedances = (pl - pc)[cv]
            rec['mean_exceedance_given_cvar'] = float(exceedances.mean())
            rec['max_exceedance_given_cvar'] = float(exceedances.max())
        else:
            rec['mean_exceedance_given_cvar'] = np.nan
            rec['max_exceedance_given_cvar'] = np.nan
        result.append(rec)
    return result
