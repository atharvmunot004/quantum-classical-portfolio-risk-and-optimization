"""Time-sliced factor and alignment metrics (by year, quarter, month)."""
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def compute_time_sliced_metrics(
    factor_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    slice_by: str,
    min_observations: int = 30,
) -> List[Dict[str, Any]]:
    """
    Aggregate factor metrics within time slices. factor_df has date, estimation_window, top_k, etc.
    Returns list of dicts with slice_type, slice_value, estimation_window, top_k, and aggregated metrics.
    """
    result = []
    if factor_df.empty:
        return result
    df = factor_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if slice_by == "year":
        df["_slice"] = df["date"].dt.year.astype(str)
    elif slice_by == "quarter":
        df["_slice"] = df["date"].dt.to_period("Q").astype(str)
    elif slice_by == "month":
        df["_slice"] = df["date"].dt.to_period("M").astype(str)
    else:
        return result
    for slice_val, grp in df.groupby("_slice"):
        if len(grp) < min_observations:
            continue
        for (ew, tk), sub in grp.groupby(["estimation_window", "top_k"]):
            rec = {
                "slice_type": slice_by,
                "slice_value": slice_val,
                "estimation_window": ew,
                "top_k": tk,
                "n_observations": len(sub),
            }
            if "cumulative_explained_variance" in sub.columns:
                rec["cumulative_explained_variance_mean"] = sub["cumulative_explained_variance"].mean()
            if not metrics_df.empty and "principal_angle_distance_vs_classical" in metrics_df.columns:
                m = metrics_df[(metrics_df["estimation_window"] == ew) & (metrics_df["top_k"] == tk)]
                if not m.empty:
                    rec["principal_angle_distance_mean"] = m["principal_angle_distance_vs_classical"].mean()
                    if "exposure_correlation" in m.columns:
                        rec["exposure_correlation_mean"] = m["exposure_correlation"].mean()
            result.append(rec)
    return result
