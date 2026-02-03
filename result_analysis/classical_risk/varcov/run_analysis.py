"""
Variance-Covariance (VaR-Cov) asset-level result analysis.

Runs analysis as specified in llm.json: data validation, coverage accuracy,
statistical backtests, tail risk, distributional diagnostics, rolling parameter
diagnostics, time-sliced analysis, robustness checks, ranking, and report generation.
Outputs to result_analysis/classical_risk/varcov/; tables as markdown documents.
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
OUT_DIR = Path(__file__).parent  # result_analysis/classical_risk/varcov


def load_config() -> dict:
    """Load analysis config from llm.json."""
    for path in (CONFIG_PATH, OUT_DIR / "llm.json"):
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")


def load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load metrics, time_sliced, risk_series, optional parameter_store. No volatility series for varcov."""
    base = PROJECT_ROOT
    metrics_path = base / config["inputs"]["metrics_table"]
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics table not found: {metrics_path}")
    metrics = pd.read_parquet(metrics_path)

    ts_path = base / config["inputs"]["time_sliced_metrics_table"]
    if ts_path.exists():
        time_sliced = pd.read_parquet(ts_path)
    else:
        time_sliced = pd.DataFrame()

    risk_path = base / config["inputs"]["risk_series_table"]
    if risk_path.exists():
        risk_series = pd.read_parquet(risk_path)
    else:
        risk_series = pd.DataFrame()

    param_path = base / config["inputs"]["parameter_store"]
    params = None
    if param_path.exists():
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
    df = df.copy()
    for col in cfg["drop_if_missing"]:
        if col in df.columns:
            df = df.dropna(subset=[col], how="any")
    for check in cfg["consistency_checks"]:
        name, rule = check["name"], check["rule"]
        if rule == "violation_ratio > 0" and "violation_ratio" in df.columns:
            before = len(df)
            df = df.loc[df["violation_ratio"] > 0]
            if len(df) < before:
                print(f"Consistency {name}: dropped {before - len(df)} rows.")
        elif rule == "0 <= hit_rate <= 1" and "hit_rate" in df.columns:
            before = len(df)
            df = df.loc[(df["hit_rate"] >= 0) & (df["hit_rate"] <= 1)]
            if len(df) < before:
                print(f"Consistency {name}: dropped {before - len(df)} rows.")
    return df


def primary_evaluation(df: pd.DataFrame, config: dict) -> dict:
    """Coverage accuracy, backtest rejection rates, traffic light aggregation."""
    out = {}
    cfg = config["primary_evaluation"]
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
    bt = cfg["statistical_backtests"]
    test_to_reject = {
        "kupiec_unconditional_coverage": "kupiec_reject_null",
        "christoffersen_independence": "christoffersen_independence_reject_null",
        "christoffersen_conditional_coverage": "christoffersen_conditional_coverage_reject_null",
    }
    for test in bt["tests"]:
        rcol = test_to_reject.get(test, test.replace("_coverage", "_reject_null"))
        if rcol in df.columns:
            out[f"{test}_rejection_rate"] = float(df[rcol].mean() * 100)
    reg = cfg["regulatory_diagnostics"]
    if reg["metric"] in df.columns:
        tl = df[reg["metric"]].value_counts()
        out["traffic_light_counts"] = tl.to_dict()
        out["traffic_light_pct"] = (tl / len(df) * 100).to_dict()
    return out


def tail_risk_analysis(df: pd.DataFrame, config: dict) -> dict:
    """VaR tail behavior metrics with mean/median/p95."""
    out = {}
    cfg = config.get("tail_risk_analysis", {})
    block = cfg.get("var_tail_behavior", {})
    if not block:
        return out
    metrics = [m for m in block.get("metrics", []) if m in df.columns]
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


def distributional_diagnostics(df: pd.DataFrame, config: dict) -> dict:
    """Skewness, kurtosis, Jarque-Bera."""
    out = {}
    cfg = config.get("distributional_diagnostics", {})
    for m in cfg.get("metrics", ["skewness", "kurtosis", "jarque_bera_p_value"]):
        if m in df.columns:
            s = df[m].dropna()
            if len(s):
                out[f"{m}_mean"] = float(s.mean())
                out[f"{m}_median"] = float(s.median())
    return out


def rolling_parameter_diagnostics(df: pd.DataFrame, params: pd.DataFrame | None, config: dict) -> dict:
    """Rolling mean and volatility stability by window."""
    out = {}
    cfg = config.get("rolling_parameter_diagnostics", {})
    params_list = [p for p in cfg.get("parameters", []) if p in df.columns]
    for p in params_list:
        s = df[p].dropna()
        if len(s):
            out[f"{p}_mean"] = float(s.mean())
            out[f"{p}_std"] = float(s.std())
            out[f"{p}_cv"] = float(s.std() / s.mean()) if s.mean() != 0 else np.nan
    stab = cfg.get("stability_checks", {})
    if stab.get("by_window") and "estimation_window" in df.columns:
        for p in params_list:
            if p not in df.columns:
                continue
            by_w = df.groupby("estimation_window")[p].agg(["mean", "std"]).reset_index()
            cv = by_w["mean"].replace(0, np.nan)
            by_w["coefficient_of_variation"] = by_w["std"] / cv
            out[f"{p}_by_window"] = by_w.to_dict(orient="records")
    return out


def time_sliced_analysis(ts: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Aggregate by slice (year/quarter from start_date or slice_type + slice)."""
    cfg = config.get("time_sliced_analysis", {})
    if not cfg.get("enabled", True) or ts.empty:
        return pd.DataFrame()
    slice_metrics = [m for m in cfg.get("metrics", []) if m in ts.columns]
    if not slice_metrics:
        slice_metrics = [c for c in ts.columns if c in ("hit_rate", "violation_ratio", "num_violations", "expected_violations", "traffic_light_zone")]
    ts = ts.copy()
    if "start_date" in ts.columns:
        ts["start_date"] = pd.to_datetime(ts["start_date"], errors="coerce")
        ts["year"] = ts["start_date"].dt.year
        ts["quarter"] = ts["start_date"].dt.quarter
    dims = [d for d in cfg.get("slice_by", ["year", "quarter"]) if d in ts.columns]
    if not dims:
        if "slice_type" in ts.columns and "slice" in ts.columns:
            ts["slice_dimension"] = ts["slice_type"].astype(str) + "_" + ts["slice"].astype(str)
            dims = ["slice_dimension"]
        elif "slice_value" in ts.columns and slice_metrics:
            agg = ts.groupby("slice_value")[slice_metrics].agg(["mean", "count"]).reset_index()
            agg.columns = [c if not isinstance(c, tuple) else f"{c[0]}_{c[1]}" for c in agg.columns]
            return agg
        else:
            return pd.DataFrame()
    min_obs = cfg.get("minimum_observations", 60)
    parts = []
    for dim in dims:
        g = ts.groupby(dim).filter(lambda x: len(x) >= min_obs) if min_obs else ts
        if g.empty:
            continue
        agg = g.groupby(dim)[slice_metrics].agg(["mean", "median", "count"]).reset_index()
        agg.columns = [c if not isinstance(c, tuple) else f"{c[0]}_{c[1]}" for c in agg.columns]
        agg["slice_dimension"] = dim
        parts.append(agg)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def robustness_checks(df: pd.DataFrame, config: dict) -> dict:
    """Window sensitivity and horizon scaling check."""
    out = {}
    cfg = config.get("robustness_checks", {})
    if "window_sensitivity" in cfg:
        wins = cfg["window_sensitivity"].get("compare_windows", [252, 500])
        mets = [m for m in cfg["window_sensitivity"].get("metrics", []) if m in df.columns]
        if "estimation_window" in df.columns and mets:
            sub = df[df["estimation_window"].isin(wins)]
            if not sub.empty:
                by_w = sub.groupby("estimation_window")[mets].mean()
                out["window_sensitivity"] = by_w.to_dict()
    if "horizon_scaling_check" in cfg:
        horizons = cfg["horizon_scaling_check"].get("compare_horizons", [1, 10])
        mets = [m for m in cfg["horizon_scaling_check"].get("metrics", []) if m in df.columns]
        if "horizon" in df.columns and mets:
            sub = df[df["horizon"].isin(horizons)]
            if not sub.empty:
                by_h = sub.groupby("horizon")[mets].mean()
                out["horizon_sensitivity"] = by_h.to_dict()
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
    cfg = config.get("ranking_and_selection", {})
    df = df.copy()
    df["_expected_hr"] = 1 - df["confidence_level"]
    score = np.zeros(len(df))
    for rm in cfg.get("ranking_metrics", []):
        m, w = rm.get("metric", ""), rm.get("weight", 1.0)
        if m == "abs(hit_rate - (1 - confidence_level))":
            s = (df["hit_rate"] - df["_expected_hr"]).abs().values
        elif m == "violation_ratio" and rm.get("target") == 1.0:
            s = (df["violation_ratio"] - 1.0).abs().values
        elif m == "rmse_var_vs_losses" and m in df.columns:
            s = df[m].fillna(np.inf).values
        else:
            continue
        smin, smax = s.min(), s.max()
        if smax > smin:
            s = (s - smin) / (smax - smin)
        score = score + s * w
    df["_rank_score"] = score
    top_k = cfg.get("output_top_k", 10)
    return df.nsmallest(top_k, "_rank_score").drop(columns=["_expected_hr", "_rank_score"], errors="ignore")


def df_to_markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Convert DataFrame to markdown table; cap rows if large."""
    if df.empty:
        return "_No data._"
    d = df.head(max_rows)
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in d.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                v = ""
            cells.append(str(v)[:50])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def generate_figures(df: pd.DataFrame, ts: pd.DataFrame, risk_series: pd.DataFrame, config: dict, fig_dir: Path) -> None:
    """Produce plots listed in config."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figures.")
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    plots = config.get("outputs", {}).get("figures", {}).get("plots", [])

    if "hit_rate_vs_confidence" in plots and "confidence_level" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        expected = 1 - df["confidence_level"]
        ax.scatter(df["confidence_level"], df["hit_rate"], alpha=0.6, label="Hit rate")
        ax.scatter(df["confidence_level"], expected, alpha=0.6, marker="x", label="Expected")
        ax.set_xlabel("Confidence level")
        ax.set_ylabel("Hit rate")
        ax.legend()
        ax.set_title("Hit rate vs confidence level (VaR-Cov)")
        plt.tight_layout()
        plt.savefig(fig_dir / "hit_rate_vs_confidence.png", dpi=150, bbox_inches="tight")
        plt.close()

    if "violation_ratio_distribution" in plots and "violation_ratio" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["violation_ratio"].dropna(), bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(1.0, color="green", linestyle="--", label="Target 1.0")
        ax.set_xlabel("Violation ratio")
        ax.set_ylabel("Count")
        ax.set_title("Violation ratio distribution (VaR-Cov)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "violation_ratio_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    if "rolling_volatility_stability" in plots and "rolling_volatility" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        if "estimation_window" in df.columns:
            for w in df["estimation_window"].dropna().unique()[:5]:
                sub = df[df["estimation_window"] == w]["rolling_volatility"].dropna()
                if len(sub):
                    ax.hist(sub, bins=20, alpha=0.5, label=f"window={w}")
        else:
            ax.hist(df["rolling_volatility"].dropna(), bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Rolling volatility")
        ax.set_ylabel("Count")
        ax.set_title("Rolling volatility stability (VaR-Cov)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "rolling_volatility_stability.png", dpi=150, bbox_inches="tight")
        plt.close()

    if "time_sliced_hit_rate" in plots and not ts.empty and "hit_rate" in ts.columns:
        slice_col = "year" if "year" in ts.columns else ("quarter" if "quarter" in ts.columns else ("slice_value" if "slice_value" in ts.columns else ts.columns[0]))
        if slice_col in ts.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            by_slice = ts.groupby(slice_col)["hit_rate"].mean()
            by_slice.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7)
            ax.set_xlabel(slice_col)
            ax.set_ylabel("Mean hit rate")
            ax.set_title("Time-sliced hit rate (VaR-Cov)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(fig_dir / "time_sliced_hit_rate.png", dpi=150, bbox_inches="tight")
            plt.close()

    if "var_time_series_vs_losses" in plots and not risk_series.empty and "VaR" in risk_series.columns and "date" in risk_series.columns:
        # Sample one asset/config for clarity
        sample = risk_series.head(2000)
        if "asset" in sample.columns:
            sample = sample[sample["asset"] == sample["asset"].iloc[0]]
        sample = sample.sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(sample)), -sample["VaR"].values, label="-VaR (loss side)", alpha=0.8)
        ax.set_xlabel("Time index")
        ax.set_ylabel("VaR (loss)")
        ax.set_title("VaR time series (VaR-Cov sample)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "var_time_series_vs_losses.png", dpi=150, bbox_inches="tight")
        plt.close()
    print(f"Figures saved under {fig_dir}")


def write_report(
    path: Path,
    config: dict,
    primary: dict,
    tail: dict,
    dist_diag: dict,
    rolling_diag: dict,
    robust: dict,
    perf: dict,
    n_configs: int,
    ts_analysis: pd.DataFrame,
    top_configs: pd.DataFrame,
) -> None:
    """Write markdown report with sections and summary tables."""
    sections = config.get("outputs", {}).get("report", {}).get("sections", [])
    lines = [
        "# Variance-Covariance (VaR-Cov) asset-level analysis report",
        "",
        f"Config: `{CONFIG_PATH.name}`. Total configurations: {n_configs}.",
        "",
    ]

    if "coverage_accuracy_results" in sections:
        lines.extend([
            "## Coverage accuracy results",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Hit rate (mean) | {primary.get('hit_rate_mean', 'N/A')} |",
            f"| Violation ratio (mean) | {primary.get('violation_ratio_mean', 'N/A')} |",
            f"| Expected hit rate (mean) | {primary.get('expected_hit_rate_mean', 'N/A')} |",
            f"| Within tolerance (%) | {primary.get('within_tolerance_pct', 'N/A')} |",
            "",
        ])
        if "traffic_light_counts" in primary:
            lines.append("### Traffic light zone")
            lines.append("| Zone | Count | % |")
            lines.append("|------|-------|---|")
            for z, c in primary["traffic_light_counts"].items():
                pct = primary["traffic_light_pct"].get(z, 0)
                lines.append(f"| {z} | {c} | {pct:.1f}% |")
            lines.append("")

    if "statistical_backtesting" in sections:
        lines.append("## Statistical backtesting")
        lines.append("| Test | Rejection rate (%) |")
        lines.append("|------|--------------------|")
        for k, v in primary.items():
            if "rejection_rate" in k:
                lines.append(f"| {k} | {v}% |")
        lines.append("")

    if "tail_risk_behavior" in sections and tail:
        lines.append("## Tail risk (VaR)")
        rows = [[k, v] for k, v in tail.items() if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))]
        if rows:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in rows[:40]:
                lines.append(f"| {k} | {v} |")
        lines.append("")

    if "normality_diagnostics" in sections and dist_diag:
        lines.append("## Normality diagnostics")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in dist_diag.items():
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                lines.append(f"| {k} | {v} |")
        interp = config.get("distributional_diagnostics", {}).get("interpretation_rules", {})
        if interp.get("note"):
            lines.append("")
            lines.append(f"*{interp['note']}*")
        lines.append("")

    if "rolling_parameter_stability" in sections and rolling_diag:
        lines.append("## Rolling parameter stability")
        for k, v in rolling_diag.items():
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                lines.append(f"- {k}: {v}")
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                lines.append(f"### {k}")
                lines.append(df_to_markdown_table(pd.DataFrame(v), max_rows=20))
        lines.append("")

    if "regime_sensitivity" in sections and not ts_analysis.empty:
        lines.append("## Time-sliced analysis (regime sensitivity)")
        lines.append("")
        lines.append(df_to_markdown_table(ts_analysis, max_rows=30))
        lines.append("")

    if "robustness_checks" in sections and robust:
        lines.append("## Robustness checks")
        for k, v in robust.items():
            if isinstance(v, dict):
                lines.append(f"### {k}")
                try:
                    tbl = pd.DataFrame(v)
                    if tbl.index.name is None and len(tbl.index):
                        tbl = tbl.reset_index()
                    if "index" in tbl.columns:
                        tbl = tbl.rename(columns={"index": "estimation_window" if k == "window_sensitivity" else "horizon"})
                    lines.append(df_to_markdown_table(tbl, max_rows=20))
                except Exception:
                    lines.append(str(v))
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")

    if "computational_performance" in sections:
        lines.append("## Computational performance")
        if perf:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in perf.items():
                lines.append(f"| {k} | {v} |")
        else:
            lines.append("No runtime metrics in asset-level metrics.")
        lines.append("")

    if "discussion" in sections:
        lines.extend([
            "## Discussion",
            "",
            "Analysis performed per llm.json: coverage accuracy, backtests, tail risk, ",
            "normality diagnostics, rolling parameter stability, time-sliced and robustness checks, ",
            "and ranking. Summary is in tabular form above.",
            "",
        ])

    if not top_configs.empty:
        lines.append("## Top configurations (ranking)")
        lines.append("")
        lines.append(df_to_markdown_table(top_configs, max_rows=15))
        lines.append("")

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
    dist_diag = distributional_diagnostics(metrics, config)
    rolling_diag = rolling_parameter_diagnostics(metrics, params, config)
    robust = robustness_checks(metrics, config)
    perf = computational_performance(metrics, config)

    ts_analysis = time_sliced_analysis(time_sliced, config)
    top_configs = ranking_and_selection(metrics, config)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig_dir = OUT_DIR / "figures"
    generate_figures(metrics, ts_analysis if not ts_analysis.empty else time_sliced, risk_series, config, fig_dir)

    report_path = OUT_DIR / "varcov_asset_level_analysis_report.md"
    write_report(
        report_path, config, primary, tail, dist_diag, rolling_diag,
        robust, perf, len(metrics), ts_analysis, top_configs,
    )

    # Summary table (parquet + markdown)
    summary_df = pd.DataFrame([
        {"metric": k, "value": v} for k, v in {**primary, **tail, **dist_diag, **rolling_diag}.items()
        if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))
    ])
    if not summary_df.empty:
        summary_parquet = OUT_DIR / "varcov_asset_level_analysis_summary.parquet"
        summary_df.to_parquet(summary_parquet, index=False)
        summary_md = OUT_DIR / "varcov_asset_level_analysis_summary.md"
        summary_md.write_text("# VaR-Cov asset-level analysis summary\n\n" + df_to_markdown_table(summary_df), encoding="utf-8")
        print(f"Summary: {summary_parquet}, {summary_md}")

    if not ts_analysis.empty:
        ts_parquet = OUT_DIR / "varcov_asset_level_time_sliced_analysis.parquet"
        ts_analysis.to_parquet(ts_parquet, index=False)
        ts_md = OUT_DIR / "varcov_asset_level_time_sliced_analysis.md"
        ts_md.write_text("# VaR-Cov asset-level time-sliced analysis\n\n" + df_to_markdown_table(ts_analysis), encoding="utf-8")
        print(f"Time-sliced: {ts_parquet}, {ts_md}")

    if not top_configs.empty:
        top_md = OUT_DIR / "varcov_asset_level_top_configs.md"
        top_md.write_text("## Top configurations (VaR-Cov)\n\n" + df_to_markdown_table(top_configs), encoding="utf-8")
        print(f"Top configs (markdown): {top_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
