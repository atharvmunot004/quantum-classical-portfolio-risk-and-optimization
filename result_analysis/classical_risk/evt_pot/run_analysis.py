"""
EVT-POT asset-level result analysis.

Runs analysis as specified in llm.json: data validation, coverage accuracy,
statistical backtests, tail risk, EVT parameter diagnostics, time-sliced analysis,
robustness checks, ranking, and report generation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root (parent of result_analysis)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).parent / "llm.json"


def load_config() -> dict:
    """Load analysis config from llm.json."""
    for path in (CONFIG_PATH, PROJECT_ROOT / "result_analysis" / "classical_risk" / "evt_pot" / "llm.json"):
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")


def load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load metrics, time_sliced, risk_series, and optional parameter_store from project paths."""
    base = PROJECT_ROOT
    metrics = pd.read_parquet(base / config["inputs"]["metrics_table"])
    time_sliced = pd.read_parquet(base / config["inputs"]["time_sliced_metrics_table"])
    risk_series = pd.read_parquet(base / config["inputs"]["risk_series_table"])

    param_path = base / config["inputs"]["parameter_store"]
    if not param_path.exists():
        alt = base / "cache" / "evt_gpd_parameters.parquet"
        param_path = alt if alt.exists() else None
    params = None
    if param_path and Path(param_path).exists():
        try:
            params = pd.read_parquet(param_path)
        except Exception as e:
            print(f"Parameter store not loaded ({e}); continuing without.")

    return metrics, time_sliced, risk_series, params


def validate_and_prepare(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply required columns check, drop_if_missing, and consistency checks."""
    cfg = config["data_validation"]
    required = [c for c in cfg["required_columns"] if c in df.columns]
    missing_req = [c for c in cfg["required_columns"] if c not in df.columns]
    if missing_req:
        print(f"Warning: missing required columns {missing_req}; proceeding with available.")
    df = df.dropna(subset=cfg["drop_if_missing"], how="any")
    for check in cfg["consistency_checks"]:
        name, rule = check["name"], check["rule"]
        if rule == "violation_ratio > 0":
            before = len(df)
            df = df.loc[df["violation_ratio"] > 0]
            if len(df) < before:
                print(f"Consistency {name}: dropped {before - len(df)} rows.")
        elif rule == "0 <= hit_rate <= 1":
            before = len(df)
            df = df.loc[(df["hit_rate"] >= 0) & (df["hit_rate"] <= 1)]
            if len(df) < before:
                print(f"Consistency {name}: dropped {before - len(df)} rows.")
    return df


def primary_evaluation(df: pd.DataFrame, config: dict) -> dict:
    """Coverage accuracy, backtest rejection rates, traffic light aggregation."""
    out = {}
    cfg = config["primary_evaluation"]
    # Coverage accuracy
    cov = cfg["coverage_accuracy"]
    expected_hr = 1 - df["confidence_level"]
    out["hit_rate_mean"] = float(df["hit_rate"].mean())
    out["violation_ratio_mean"] = float(df["violation_ratio"].mean())
    out["expected_hit_rate_mean"] = float(expected_hr.mean())
    tol = cov.get("tolerance_band", {})
    mult = tol.get("multiplier", 5)
    out["within_tolerance_pct"] = float(
        ((df["hit_rate"] - expected_hr).abs() <= mult * 0.01).mean() * 100
    )
    # Backtests
    bt = cfg["statistical_backtests"]
    sig = bt["significance_level"]
    test_to_reject = {
        "kupiec_unconditional_coverage": "kupiec_reject_null",
        "christoffersen_independence": "christoffersen_independence_reject_null",
        "christoffersen_conditional_coverage": "christoffersen_conditional_coverage_reject_null",
    }
    for test in bt["tests"]:
        rcol = test_to_reject.get(test, test.replace("_coverage", "_reject_null"))
        if rcol in df.columns:
            out[f"{test}_rejection_rate"] = float(df[rcol].mean() * 100)
        elif test in df.columns:
            out[f"{test}_rejection_rate"] = float((df[test] <= sig).mean() * 100)
    # Regulatory
    reg = cfg["regulatory_diagnostics"]
    if reg["metric"] in df.columns:
        tl = df[reg["metric"]].value_counts()
        out["traffic_light_counts"] = tl.to_dict()
        out["traffic_light_pct"] = (tl / len(df) * 100).to_dict()
    return out


def tail_risk_analysis(df: pd.DataFrame, config: dict) -> dict:
    """VaR and CVaR tail behavior metrics with mean/median/p95."""
    out = {}
    cfg = config["tail_risk_analysis"]
    for key in ("var_tail_behavior", "cvar_tail_behavior"):
        if key not in cfg:
            continue
        block = cfg[key]
        metrics = [m for m in block["metrics"] if m in df.columns]
        agg = block.get("aggregation", ["mean", "median", "p95"])
        for m in metrics:
            s = df[m].dropna()
            if len(s) == 0:
                continue
            if "mean" in agg:
                out[f"{m}_mean"] = float(s.mean())
            if "median" in agg:
                out[f"{m}_median"] = float(s.median())
            if "p95" in agg:
                out[f"{m}_p95"] = float(s.quantile(0.95))
    return out


def evt_parameter_diagnostics(df: pd.DataFrame, config: dict) -> dict:
    """EVT parameter stats, stability by window/threshold, xi bounds validation."""
    out = {}
    cfg = config["evt_parameter_diagnostics"]
    params = [p for p in cfg["parameters"] if p in df.columns]
    for p in params:
        s = df[p].dropna()
        if len(s):
            out[f"{p}_mean"] = float(s.mean())
            out[f"{p}_std"] = float(s.std())
            out[f"{p}_cv"] = float(s.std() / s.mean()) if s.mean() != 0 else np.nan
    stab = cfg.get("stability_checks") or {}
    if stab.get("by_window") and "estimation_window" in df.columns:
        for p in params:
            if p not in df.columns:
                continue
            by_w = df.groupby("estimation_window")[p].agg(["mean", "std"])
            out[f"{p}_by_window"] = by_w.to_dict()
    if stab.get("by_threshold") and "threshold_quantile" in df.columns:
        for p in params:
            if p not in df.columns:
                continue
            by_t = df.groupby("threshold_quantile")[p].agg(["mean", "std"])
            out[f"{p}_by_threshold"] = by_t.to_dict()
    constraints = cfg.get("constraints_validation") or {}
    bounds = constraints.get("xi_bounds", [-0.5, 0.5])
    if "tail_index_xi" in df.columns and constraints.get("flag_violations"):
        xi = df["tail_index_xi"].dropna()
        viol = ((xi < bounds[0]) | (xi > bounds[1])).sum()
        out["xi_violations_count"] = int(viol)
        out["xi_violations_pct"] = float(viol / len(xi) * 100) if len(xi) else 0
    return out


def time_sliced_analysis(ts: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Derive year/quarter from time slices, aggregate metrics, regime comparison."""
    cfg = config["time_sliced_analysis"]
    if not cfg.get("enabled", True):
        return pd.DataFrame()
    slice_metrics = [m for m in cfg["metrics"] if m in ts.columns]
    if "start_date" in ts.columns:
        ts = ts.copy()
        ts["start_date"] = pd.to_datetime(ts["start_date"], errors="coerce")
        ts["year"] = ts["start_date"].dt.year
        ts["quarter"] = ts["start_date"].dt.quarter
    dims = [d for d in cfg["slice_by"] if d in ts.columns]
    if not dims or not slice_metrics:
        agg = ts.groupby(["slice"] if "slice" in ts.columns else ts.index)[slice_metrics].agg(["mean", "count"]).reset_index()
        return agg
    min_obs = cfg.get("minimum_observations", 60)
    parts = []
    for dim in dims:
        g = ts.groupby(dim)
        g = g.filter(lambda x: len(x) >= min_obs) if min_obs else ts
        if g.empty:
            continue
        agg = g.groupby(dim)[slice_metrics].agg(["mean", "median", "count"]).reset_index()
        agg["slice_dimension"] = dim
        parts.append(agg)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def robustness_checks(df: pd.DataFrame, config: dict) -> dict:
    """Threshold and window sensitivity."""
    out = {}
    cfg = config["robustness_checks"]
    if "threshold_sensitivity" in cfg:
        tq = cfg["threshold_sensitivity"].get("compare_quantiles", [0.95, 0.9, 0.85, 0.8])
        mets = [m for m in cfg["threshold_sensitivity"].get("metrics", []) if m in df.columns]
        if "threshold_quantile" in df.columns and mets:
            sub = df[df["threshold_quantile"].isin(tq)]
            by_t = sub.groupby("threshold_quantile")[mets].mean()
            out["threshold_sensitivity"] = by_t.to_dict()
    if "window_sensitivity" in cfg:
        wins = cfg["window_sensitivity"].get("compare_windows", [252, 500])
        mets = [m for m in cfg["window_sensitivity"].get("metrics", []) if m in df.columns]
        if "estimation_window" in df.columns and mets:
            sub = df[df["estimation_window"].isin(wins)]
            by_w = sub.groupby("estimation_window")[mets].mean()
            out["window_sensitivity"] = by_w.to_dict()
    return out


def computational_performance(df: pd.DataFrame, config: dict) -> dict:
    """Runtime metrics if present."""
    out = {}
    cfg = config.get("computational_performance", {})
    metrics = [m for m in cfg.get("metrics", []) if m in df.columns]
    if not metrics:
        return out
    agg = cfg.get("aggregation", ["mean", "median", "p95"])
    for m in metrics:
        s = df[m].dropna()
        if len(s):
            if "mean" in agg:
                out[f"{m}_mean"] = float(s.mean())
            if "median" in agg:
                out[f"{m}_median"] = float(s.median())
            if "p95" in agg:
                out[f"{m}_p95"] = float(s.quantile(0.95))
    return out


def ranking_and_selection(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Composite ranking and top-k selection."""
    cfg = config["ranking_and_selection"]
    df = df.copy()
    df["_expected_hr"] = 1 - df["confidence_level"]
    score = 0.0
    for rm in cfg["ranking_metrics"]:
        m, w = rm["metric"], rm.get("weight", 1.0)
        if m == "abs(hit_rate - (1 - confidence_level))":
            s = (df["hit_rate"] - df["_expected_hr"]).abs()
        elif m == "violation_ratio" and rm.get("target") == 1.0:
            s = (df["violation_ratio"] - 1.0).abs()
        elif m == "rmse_var_vs_losses" and m in df.columns:
            s = df[m].fillna(np.inf)
        else:
            continue
        # Normalize to 0-1 then weight
        smin, smax = s.min(), s.max()
        if smax > smin:
            s = (s - smin) / (smax - smin)
        score = score + s * w
    df["_rank_score"] = score
    top_k = cfg.get("output_top_k", 10)
    out = df.nsmallest(top_k, "_rank_score").drop(columns=["_expected_hr", "_rank_score"], errors="ignore")
    return out


def build_summary_table(metrics: pd.DataFrame, primary: dict, tail: dict, evt: dict, robust: dict, perf: dict) -> pd.DataFrame:
    """Single-row summary table for parquet output."""
    row = {}
    row["n_configurations"] = len(metrics)
    row["hit_rate_mean"] = primary.get("hit_rate_mean")
    row["violation_ratio_mean"] = primary.get("violation_ratio_mean")
    row["within_tolerance_pct"] = primary.get("within_tolerance_pct")
    for k, v in tail.items():
        if isinstance(v, (int, float, np.floating)):
            row[k] = float(v)
    for k, v in evt.items():
        if isinstance(v, (int, float, np.floating)):
            row[k] = float(v)
    for k, v in robust.items():
        if isinstance(v, dict):
            continue
        row[k] = v
    for k, v in perf.items():
        row[k] = v
    return pd.DataFrame([row])


def generate_figures(df: pd.DataFrame, ts: pd.DataFrame, config: dict, fig_dir: Path) -> None:
    """Produce plots listed in config."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available; skipping figures.")
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    plots = config["outputs"]["figures"].get("plots", [])

    if "hit_rate_vs_confidence" in plots and "confidence_level" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        expected = 1 - df["confidence_level"]
        ax.scatter(df["confidence_level"], df["hit_rate"], alpha=0.6, label="Hit rate")
        ax.scatter(df["confidence_level"], expected, alpha=0.6, marker="x", label="Expected")
        ax.set_xlabel("Confidence level")
        ax.set_ylabel("Hit rate")
        ax.legend()
        ax.set_title("Hit rate vs confidence level")
        plt.tight_layout()
        plt.savefig(fig_dir / "hit_rate_vs_confidence.png", dpi=150, bbox_inches="tight")
        plt.close()

    if "violation_ratio_distribution" in plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["violation_ratio"].dropna(), bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(1.0, color="green", linestyle="--", label="Target 1.0")
        ax.set_xlabel("Violation ratio")
        ax.set_ylabel("Count")
        ax.set_title("Violation ratio distribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "violation_ratio_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    if "tail_index_stability" in plots and "tail_index_xi" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["tail_index_xi"].dropna(), bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="green", linestyle="--", label="Exponential (ξ=0)")
        ax.set_xlabel("Tail index ξ")
        ax.set_ylabel("Count")
        ax.set_title("Tail index stability")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "tail_index_stability.png", dpi=150, bbox_inches="tight")
        plt.close()

    if "time_sliced_hit_rate" in plots and not ts.empty and "hit_rate" in ts.columns:
        slice_col = "slice" if "slice" in ts.columns else ("year" if "year" in ts.columns else ts.columns[0])
        fig, ax = plt.subplots(figsize=(10, 5))
        by_slice = ts.groupby(slice_col)["hit_rate"].mean()
        by_slice.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7)
        ax.set_xlabel(slice_col)
        ax.set_ylabel("Mean hit rate")
        ax.set_title("Time-sliced hit rate")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "time_sliced_hit_rate.png", dpi=150, bbox_inches="tight")
        plt.close()

    if "threshold_sensitivity_heatmap" in plots and "threshold_quantile" in df.columns:
        pivot = df.pivot_table(
            values="hit_rate",
            index=df.get("estimation_window", df.index),
            columns="threshold_quantile",
            aggfunc="mean",
        )
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(pivot, annot=True, fmt=".3f", ax=ax, cmap="RdYlGn_r")
            ax.set_title("Hit rate: threshold sensitivity heatmap")
            plt.tight_layout()
            plt.savefig(fig_dir / "threshold_sensitivity_heatmap.png", dpi=150, bbox_inches="tight")
            plt.close()
    print(f"Figures saved under {fig_dir}")


def write_report(
    path: Path,
    config: dict,
    primary: dict,
    tail: dict,
    evt: dict,
    robust: dict,
    perf: dict,
    n_configs: int,
) -> None:
    """Write markdown report with requested sections."""
    sections = config["outputs"]["report"].get("sections", [])
    lines = [
        "# EVT-POT asset-level analysis report",
        "",
        f"Config: `{CONFIG_PATH.name}`. Total configurations: {n_configs}.",
        "",
    ]
    if "coverage_accuracy_results" in sections:
        lines.extend([
            "## Coverage accuracy results",
            "",
            f"- Hit rate (mean): {primary.get('hit_rate_mean', 'N/A')}",
            f"- Violation ratio (mean): {primary.get('violation_ratio_mean', 'N/A')}",
            f"- Within tolerance (%): {primary.get('within_tolerance_pct', 'N/A')}",
            "",
        ])
    if "statistical_backtesting" in sections:
        lines.append("## Statistical backtesting")
        for k, v in primary.items():
            if "rejection_rate" in k:
                lines.append(f"- {k}: {v}%")
        lines.append("")
    if "tail_risk_characteristics" in sections:
        lines.append("## Tail risk characteristics")
        for k, v in tail.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                lines.append(f"- {k}: {v}")
        lines.append("")
    if "evt_parameter_stability" in sections:
        lines.append("## EVT parameter stability")
        for k, v in evt.items():
            if k in ("xi_violations_count", "xi_violations_pct") or ("_mean" in k or "_std" in k or "_cv" in k):
                if isinstance(v, (int, float)) and not np.isnan(v):
                    lines.append(f"- {k}: {v}")
        lines.append("")
    if "regime_sensitivity" in sections:
        lines.append("## Regime sensitivity")
        lines.append("See time-sliced analysis table.")
        lines.append("")
    if "robustness_checks" in sections:
        lines.append("## Robustness checks")
        for k, v in robust.items():
            if isinstance(v, dict):
                lines.append(f"- {k}: (see summary table)")
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")
    if "computational_performance" in sections:
        lines.append("## Computational performance")
        if perf:
            for k, v in perf.items():
                lines.append(f"- {k}: {v}")
        else:
            lines.append("No runtime metrics in asset-level metrics.")
        lines.append("")
    if "discussion" in sections:
        lines.extend([
            "## Discussion",
            "",
            "Analysis performed per llm.json: coverage accuracy, backtests, tail risk, EVT diagnostics, ",
            "time-sliced and robustness checks, and ranking. See summary and time-sliced parquet outputs.",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {path}")


def main() -> int:
    config = load_config()
    base = PROJECT_ROOT

    print("Loading data...")
    metrics, time_sliced, risk_series, params = load_data(config)
    print(f"Metrics: {metrics.shape}, Time sliced: {time_sliced.shape}, Risk series: {risk_series.shape}")

    metrics = validate_and_prepare(metrics, config)
    print(f"After validation: {len(metrics)} rows")

    primary = primary_evaluation(metrics, config)
    tail = tail_risk_analysis(metrics, config)
    evt = evt_parameter_diagnostics(metrics, config)
    robust = robustness_checks(metrics, config)
    perf = computational_performance(metrics, config)

    ts_analysis = time_sliced_analysis(time_sliced, config)
    top_configs = ranking_and_selection(metrics, config)

    summary_df = build_summary_table(metrics, primary, tail, evt, robust, perf)

    # Outputs
    out = config["outputs"]
    summary_path = base / out["summary_tables"]["path"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(summary_path, index=False)
    print(f"Summary table: {summary_path}")

    ts_path = base / out["time_sliced_tables"]["path"]
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    if not ts_analysis.empty:
        ts_analysis.to_parquet(ts_path, index=False)
        print(f"Time-sliced table: {ts_path}")
    else:
        time_sliced.head(0).to_parquet(ts_path, index=False)
        print(f"Time-sliced table (empty): {ts_path}")

    fig_dir = base / out["figures"]["path"]
    generate_figures(metrics, ts_analysis if not ts_analysis.empty else time_sliced, config, fig_dir)

    report_path = base / out["report"]["path"]
    write_report(report_path, config, primary, tail, evt, robust, perf, len(metrics))

    # Optional: save top-k ranking
    rank_path = base / "result_analysis" / "classical_risk" / "evt_pot" / "evt_asset_level_top_configs.parquet"
    top_configs.to_parquet(rank_path, index=False)
    print(f"Top configs: {rank_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
