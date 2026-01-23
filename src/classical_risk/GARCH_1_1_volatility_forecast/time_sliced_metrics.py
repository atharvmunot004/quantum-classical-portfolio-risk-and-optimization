"""Time-sliced metrics for VaR/CVaR backtesting.

Slices by year/quarter/month to reveal regime dependence.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .backtesting import detect_var_violations, detect_cvar_violations, compute_hit_rate, compute_violation_ratio


def compute_time_sliced_metrics(
    returns: pd.Series,
    var_series: pd.Series,
    cvar_series: Optional[pd.Series] = None,
    confidence_level: float = 0.95,
    slice_by: str = 'year',
    min_observations: int = 60
) -> List[Dict]:
    """Compute backtesting metrics for time slices."""
    common = returns.index.intersection(var_series.index)
    if len(common) == 0:
        return []

    r = returns.loc[common]
    v = var_series.loc[common]

    c = None
    if cvar_series is not None:
        common2 = common.intersection(cvar_series.index)
        if len(common2) == 0:
            c = None
        else:
            r = returns.loc[common2]
            v = var_series.loc[common2]
            c = cvar_series.loc[common2]

    if slice_by == 'year':
        groups = r.index.year
        fmt = lambda x: str(int(x))
    elif slice_by == 'quarter':
        groups = r.index.to_period('Q')
        fmt = lambda x: str(x)
    elif slice_by == 'month':
        groups = r.index.to_period('M')
        fmt = lambda x: str(x)
    else:
        raise ValueError("slice_by must be one of: year, quarter, month")

    expected_rate = float(1.0 - confidence_level)
    out: List[Dict] = []

    for g in sorted(pd.unique(groups)):
        m = (groups == g)
        rr = r[m]
        vv = v[m]
        cc = c[m] if c is not None else None

        if len(rr) < int(min_observations):
            continue

        vvios = detect_var_violations(rr, vv, confidence_level)
        hr = compute_hit_rate(vvios)
        x = int(vvios.sum())
        n = int(len(vvios))

        row = {
            'slice': fmt(g),
            'start_date': rr.index.min().strftime('%Y-%m-%d'),
            'end_date': rr.index.max().strftime('%Y-%m-%d'),
            'hit_rate': float(hr),
            'num_violations': x,
            'expected_violations': float(n * expected_rate),
            'violation_ratio': float(compute_violation_ratio(vvios, expected_rate))
        }

        if cc is not None:
            cvios = detect_cvar_violations(rr, cc, vv, confidence_level)
            row['cvar_hit_rate'] = float(compute_hit_rate(cvios))
            row['cvar_num_violations'] = int(cvios.sum())

        out.append(row)

    return out
