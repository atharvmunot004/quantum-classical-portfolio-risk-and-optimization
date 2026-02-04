"""Time-sliced metrics for QAE VaR/CVaR backtesting."""
import pandas as pd
import numpy as np
from typing import Dict, List

from .backtesting import (
    detect_var_violations,
    detect_cvar_violations,
    compute_hit_rate,
    compute_violation_ratio,
)


def compute_time_sliced_metrics(
    losses: pd.Series,
    var_series: pd.Series,
    cvar_series: pd.Series = None,
    confidence_level: float = 0.95,
    slice_by: str = 'year'
) -> List[Dict]:
    common = losses.index.intersection(var_series.index)
    if len(common) == 0:
        return []
    aligned_losses = losses.loc[common]
    aligned_var = var_series.loc[common]
    aligned_cvar = cvar_series.loc[common] if cvar_series is not None else None
    if aligned_cvar is not None:
        common = common.intersection(cvar_series.index)
        aligned_losses = losses.loc[common]
        aligned_var = var_series.loc[common]
        aligned_cvar = cvar_series.loc[common]

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
        pv = aligned_var[mask]
        pc = aligned_cvar[mask] if aligned_cvar is not None else None
        if len(pl) == 0:
            continue
        var_v = detect_var_violations(pl, pv, confidence_level)
        rec = {
            'slice': fmt(period),
            'start_date': pl.index.min().strftime('%Y-%m-%d'),
            'end_date': pl.index.max().strftime('%Y-%m-%d'),
            'hit_rate': float(compute_hit_rate(var_v)),
            'num_violations': int(var_v.sum()),
            'expected_violations': float(len(var_v) * exp_rate),
            'violation_ratio': float(compute_violation_ratio(var_v, exp_rate)),
        }
        if pc is not None:
            cv = detect_cvar_violations(pl, pc, pv, confidence_level)
            rec['cvar_hit_rate'] = float(compute_hit_rate(cv))
            rec['cvar_num_violations'] = int(cv.sum())
        result.append(rec)
    return result
