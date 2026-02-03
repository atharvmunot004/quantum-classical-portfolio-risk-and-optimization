"""
Monte Carlo simulation (MCS) asset-level result analysis.

Runs analysis as specified in llm.json: data validation, coverage accuracy,
statistical backtests, tail risk (VaR/CVaR), simulation method diagnostics,
distributional diagnostics, time-sliced analysis, robustness checks,
ranking, and report generation. Outputs to result_analysis/classical_risk/monte_carlo/;
tables are written as parquet and as markdown documents.
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
OUT_DIR = Path(__file__).parent  # result_analysis/classical_risk/monte_carlo


def load_config() -> dict:
    """Load analysis config from llm.json."""
    for path in (CONFIG_PATH, OUT_DIR / "llm.json"):
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")


def load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load metrics, time_sliced, risk_series, optional fitted_parameter_store."""
    base = PROJECT_ROOT
    metrics = pd.read_parquet(base / config["inputs"]["metrics_table"])
    time_sliced = pd.read_parquet(base / config["inputs"]["time_sliced_metrics_table"])
    risk_series = pd.read_parquet(base / config["inputs"]["risk_series_table"])

    param_path = base / config["inputs"].get("fitted_parameter_store", "cache/mcs_asset_fitted_parameters.parquet")
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
    missing_req = [c for c in cfg["required_columns"] if c not in df.columns]
    if missing_req:
        print(f"Warning: missing required columns {missing_req}; proceeding with available.")
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
    """VaR and CVaR tail behavior metrics with mean/median/p95."""
    out = {}
    cfg = config["tail_risk_analysis"]
    for key in ("var_tail_behavior", "cvar_tail_behavior"):
        if key not in cfg:
            continue
        block = cfg[key]
        metrics_list = [m for m in block["metrics"] if m in df.columns]
        agg = block.get("aggregation", ["mean", "median", "p95"])
        for m in metrics_list:
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


def simulation_method_diagnostics(df: pd.DataFrame, params: pd.DataFrame | None, config: dict) -> dict:
    """Method-specific parameters and parameter stability by window."""
    out = {}
    cfg = config.get("simulation_method_diagnostics", {})
    method_params = cfg.get("method_specific_parameters", {})
    for method, cols in method_params.items():
        for c in cols:
            if c in df.columns:
                sub = df.loc[df["method"] == method, c].dropna()
                if len(sub):
                    out[f"{method}_{c}_mean"] = float(sub.mean())
                    out[f"{method}_{c}_std"] = float(sub.std())
    stab = cfg.get("parameter_stability", {})
    if stab.get("by_window") and "estimation_window" in df.columns:
        for method in method_params:
            cols = [c for c in method_params[method] if c in df.columns]
            if not cols:
                continue
            sub = df.loc[df["method"] == method]
            if sub.empty:
                continue
            by_w = sub.groupby("estimation_window")[cols].agg(["mean", "std"])
            for c in cols:
                if ("mean", c) in by_w.columns:
                    out[f"{method}_{c}_by_window_mean"] = by_w[("mean", c)].to_dict()
    if params is not None and not params.empty:
        for col in params.columns:
            if col in ("method", "estimation_window", "asset"):
                continue
            s = params[col].dropna()
            if len(s):
                out[f"param_{col}_mean"] = float(s.mean())
                out[f"param_{col}_std"] = float(s.std())
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


def time_sliced_analysis(ts: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Aggregate by slice (year/quarter from start_date if present)."""
    cfg = config.get("time_sliced_analysis", {})
    if not cfg.get("enabled", True):
        return pd.DataFrame()
    slice_metrics = [m for m in cfg.get("metrics", []) if m in ts.columns]
    if not slice_metrics:
        slice_metrics = [c for c in ts.columns if c in ("hit_rate", "violation_ratio", "num_violations", "expected_violations", "traffic_light_zone")]
    if "start_date" in ts.columns:
        ts = ts.copy()
        ts["start_date"] = pd.to_datetime(ts["start_date"], errors="coerce")
        ts["year"] = ts["start_date"].dt.year
        ts["quarter"] = ts["start_date"].dt.quarter
    dims = [d for d in cfg.get("slice_by", ["year", "quarter"]) if d in ts.columns]
    if not dims:
        if "slice_value" in ts.columns and slice_metrics:
            agg = ts.groupby("slice_value")[slice_metrics].agg(["mean", "count"]).reset_index()
            agg.columns = [c if not isinstance(c, tuple) else f"{c[0]}_{c[1]}" for c in agg.columns]
            return agg
        return pd.DataFrame()
    min_obs = cfg.get("minimum_observations", 60)
    parts = []
    for dim in dims:
        g = ts.groupby(dim).filter(lambda x: len(x) >= min_obs) if min_obs else ts
        if g.empty:
            continue
        agg = g.groupby(dim)[slice_metrics].agg(["mean", "median", "count"]).reset_index()
        agg.columns = [c if not isinstance(c, tuple) else (f"{c[0]}_{c[1]}" if c[1] else str(c[0])) for c in agg.columns]
        agg["slice_dimension"] = dim
        parts.append(agg)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def robustness_checks(df: pd.DataFrame, config: dict) -> dict:
    """Method comparison, window sensitivity, horizon handling."""
    out = {}
    cfg = config.get("robustness_checks", {})
    if "method_comparison" in cfg:
        methods = cfg["method_comparison"].get("methods", [])
        mets = [m for m in cfg["method_comparison"].get("metrics", []) if m in df.columns]
        if "method" in df.columns and mets and methods:
            sub = df[df["method"].isin(methods)]
            if not sub.empty:
                by_m = sub.groupby("method")[mets].agg(["mean", "median"]).reset_index()
                out["method_comparison"] = by_m
    if "window_sensitivity" in cfg:
        wins = cfg["window_sensitivity"].get("compare_windows", [252, 500])
        mets = [m for m in cfg["window_sensitivity"].get("metrics", []) if m in df.columns]
        if "estimation_window" in df.columns and mets:
            sub = df[df["estimation_window"].isin(wins)]
            if not sub.empty:
                by_w = sub.groupby("estimation_window")[mets].mean()
                out["window_sensitivity"] = by_w.to_dict()
    if "horizon_handling_check" in cfg:
        horizons = cfg["horizon_handling_check"].get("compare_horizons", [1, 10])
        mets = [m for m in cfg["horizon_handling_check"].get("metrics", []) if m in df.columns]
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
            cells.append(str(v)[:40])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_summary_table(metrics: pd.DataFrame, primary: dict, tail: dict, sim_diag: dict,
                        dist_diag: dict, robust: dict, perf: dict) -> pd.DataFrame:
    """Single-row summary table for parquet and markdown."""
    row = {"n_configurations": len(metrics)}
    row["hit_rate_mean"] = primary.get("hit_rate_mean")
    row["violation_ratio_mean"] = primary.get("violation_ratio_mean")
    row["within_tolerance_pct"] = primary.get("within_tolerance_pct")
    for k, v in tail.items():
        if isinstance(v, (int, float, np.floating)) and not (isinstance(v, float) and np.isnan(v)):
            row[k] = float(v)
    for k, v in sim_diag.items():
        if isinstance(v, (int, float, np.floating)) and not (isinstance(v, float) and np.isnan(v)):
            row[k] = float(v)
    for k, v in dist_diag.items():
        if isinstance(v, (int, float, np.floating)) and not (isinstance(v, float) and np.isnan(v)):
            row[k] = float(v)
    for k, v in perf.items():
        row[k] = v
    return pd.DataFrame([row])


def build_method_comparison_tables(df: pd.DataFrame, config: dict) -> pd.DataFrame | None:
    """Build method comparison table (by method and confidence_level if requested)."""
    cfg = config.get("robustness_checks", {}).get("method_comparison", {})
    methods = cfg.get("methods", [])
    mets = [m for m in cfg.get("metrics", []) if m in df.columns]
    if not methods or not mets or "method" not in df.columns:
        return None
    sub = df[df["method"].isin(methods)]
    if sub.empty:
        return None
    rank_by = config.get("ranking_and_selection", {}).get("rank_by", ["method", "confidence_level"])
    grp = [c for c in rank_by if c in sub.columns]
    if not grp:
        out = sub.groupby("method")[mets].agg(["mean", "median", "count"]).reset_index()
    else:
        out = sub.groupby(grp)[mets].agg(["mean", "median"]).reset_index()
    # Flatten multi-level column names for readable markdown
    out.columns = [c if not isinstance(c, tuple) else f"{c[0]}_{c[1]}" if c[1] else str(c[0]) for c in out.columns]
    return out


def generate_figures(df: pd.DataFrame, ts: pd.DataFrame, time_sliced_raw: pd.DataFrame,
                    risk_series: pd.DataFrame, config: dict, fig_dir: Path) -> None:
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
    plots = config.get("outputs", {}).get("figures", {}).get("plots", [])

    if "hit_rate_by_method" in plots and "method" in df.columns and "hit_rate" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        by_m = df.groupby("method")["hit_rate"].agg(["mean", "std"]).reset_index()
        by_m.plot(x="method", y="mean", kind="bar", yerr="std" if "std" in by_m.columns else None, ax=ax, legend=False, capsize=3)
        ax.axhline((1 - df["confidence_level"]).mean(), color="green", linestyle="--", label="Expected")
        ax.set_xlabel("Method")
        ax.set_ylabel("Hit rate")
        ax.set_title("Hit rate by simulation method")
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "hit_rate_by_method.png", dpi=150, bbox_inches="tight")
        plt.close()

    if "violation_ratio_by_method" in plots and "method" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        by_m = df.groupby("method")["violation_ratio"].mean()
        by_m.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7)
        ax.axhline(1.0, color="green", linestyle="--", label="Target 1.0")
        ax.set_xlabel("Method")
        ax.set_ylabel("Violation ratio")
        ax.set_title("Violation ratio by simulation method")
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "violation_ratio_by_method.png", dpi=150, bbox_inches="tight")
        plt.close()

    if "tail_loss_comparison" in plots and "method" in df.columns:
        tail_cols = [c for c in ("mean_exceedance", "max_exceedance", "rmse_var_vs_losses") if c in df.columns]
        if tail_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            sub = df.groupby("method")[tail_cols].mean()
            sub.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Method")
            ax.set_ylabel("Mean value")
            ax.set_title("Tail loss metrics by method")
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="Metric")
            plt.tight_layout()
            plt.savefig(fig_dir / "tail_loss_comparison.png", dpi=150, bbox_inches="tight")
            plt.close()

    if "time_sliced_hit_rate_by_method" in plots and not time_sliced_raw.empty and "hit_rate" in time_sliced_raw.columns and "method" in time_sliced_raw.columns:
        tsr = time_sliced_raw.copy()
        if "start_date" in tsr.columns:
            tsr["year"] = pd.to_datetime(tsr["start_date"], errors="coerce").dt.year
            slice_col = "year"
        else:
            slice_col = "slice_value" if "slice_value" in tsr.columns else "slice"
        if slice_col in tsr.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            by_slice = tsr.groupby([slice_col, "method"])["hit_rate"].mean().unstack(fill_value=np.nan)
            by_slice.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7)
            ax.set_xlabel(slice_col)
            ax.set_ylabel("Mean hit rate")
            ax.set_title("Time-sliced hit rate by method")
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="Method")
            plt.tight_layout()
            plt.savefig(fig_dir / "time_sliced_hit_rate_by_method.png", dpi=150, bbox_inches="tight")
            plt.close()

    if "var_cvar_time_series_comparison" in plots and not risk_series.empty:
        plot_cols = [c for c in ("VaR", "CVaR", "var", "cvar", "loss", "returns") if c in risk_series.columns]
        if not plot_cols:
            plot_cols = [c for c in risk_series.columns if c not in ("asset", "method", "date")][:4]
        if plot_cols:
            sample = risk_series.head(500)
            if "date" in sample.columns:
                sample = sample.set_index("date")
            numeric = sample[plot_cols].select_dtypes(include=[np.number])
            if not numeric.empty:
                numeric.plot(figsize=(10, 4), alpha=0.8)
                plt.title("VaR / CVaR time series (sample)")
                plt.tight_layout()
                plt.savefig(fig_dir / "var_cvar_time_series_comparison.png", dpi=150, bbox_inches="tight")
                plt.close()
    print(f"Figures saved under {fig_dir}")


def write_report(
    path: Path,
    config: dict,
    primary: dict,
    tail: dict,
    sim_diag: dict,
    dist_diag: dict,
    robust: dict,
    perf: dict,
    n_configs: int,
    ts_analysis: pd.DataFrame,
    top_configs: pd.DataFrame,
    method_comparison_df: pd.DataFrame | None,
) -> None:
    """Write markdown report with sections and markdown tables."""
    sections = config.get("outputs", {}).get("report", {}).get("sections", [])
    lines = [
        "# Monte Carlo simulation (MCS) asset-level analysis report",
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

    if "comparison_of_simulation_methods" in sections and method_comparison_df is not None and not method_comparison_df.empty:
        lines.append("## Comparison of simulation methods")
        lines.append("")
        lines.append(df_to_markdown_table(method_comparison_df, max_rows=30))
        lines.append("")

    if "tail_risk_characteristics" in sections and tail:
        lines.append("## Tail risk (VaR/CVaR)")
        rows = [[k, v] for k, v in tail.items() if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))]
        if rows:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in rows[:50]:
                lines.append(f"| {k} | {v} |")
        lines.append("")

    if "distributional_diagnostics" in sections and dist_diag:
        lines.append("## Distributional diagnostics")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in dist_diag.items():
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                lines.append(f"| {k} | {v} |")
        lines.append("")

    if "regime_sensitivity" in sections and not ts_analysis.empty:
        lines.append("## Time-sliced analysis (regime sensitivity)")
        lines.append("")
        lines.append(df_to_markdown_table(ts_analysis, max_rows=30))
        lines.append("")

    if "robustness_checks" in sections and robust:
        lines.append("## Robustness checks")
        for k, v in robust.items():
            if isinstance(v, pd.DataFrame) and not v.empty:
                # Flatten tuple column names for markdown
                v_flat = v.copy()
                v_flat.columns = [c if not isinstance(c, tuple) else f"{c[0]}_{c[1]}" if c[1] else str(c[0]) for c in v_flat.columns]
                lines.append(f"### {k}")
                lines.append("")
                lines.append(df_to_markdown_table(v_flat, max_rows=20))
                lines.append("")
            elif isinstance(v, dict) and v:
                try:
                    # Dict of scalars or dict of dicts (e.g. window_sensitivity: {metric: {window: value}})
                    first_val = next(iter(v.values()))
                    if isinstance(first_val, dict):
                        tbl = pd.DataFrame(v).T  # rows = metric, columns = window/horizon
                        tbl = tbl.reset_index().rename(columns={"index": "metric"})
                    else:
                        tbl = pd.DataFrame(list(v.items()), columns=["key", "value"])
                    if not tbl.empty:
                        lines.append(f"### {k}")
                        lines.append("")
                        lines.append(df_to_markdown_table(tbl, max_rows=20))
                        lines.append("")
                except Exception:
                    lines.append(f"- {k}: (see data)")
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
            "simulation method and distributional diagnostics, time-sliced and robustness checks, ",
            "and ranking. Summary and time-sliced tables are available as parquet and markdown.",
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
    sim_diag = simulation_method_diagnostics(metrics, params, config)
    dist_diag = distributional_diagnostics(metrics, config)
    robust = robustness_checks(metrics, config)
    perf = computational_performance(metrics, config)

    ts_analysis = time_sliced_analysis(time_sliced, config)
    top_configs = ranking_and_selection(metrics, config)
    method_comparison_df = build_method_comparison_tables(metrics, config)

    summary_df = build_summary_table(metrics, primary, tail, sim_diag, dist_diag, robust, perf)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config["outputs"]

    # Parquet outputs
    summary_path = base / out["summary_tables"]["path"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(summary_path, index=False)
    print(f"Summary table: {summary_path}")

    ts_path = base / out["time_sliced_tables"]["path"]
    if not ts_analysis.empty:
        ts_analysis.to_parquet(ts_path, index=False)
        print(f"Time-sliced table: {ts_path}")
    else:
        time_sliced.head(0).to_parquet(ts_path, index=False)
        print(f"Time-sliced table (empty): {ts_path}")

    mc_path = base / out["method_comparison_tables"]["path"]
    if method_comparison_df is not None and not method_comparison_df.empty:
        method_comparison_df.to_parquet(mc_path, index=False)
        print(f"Method comparison table: {mc_path}")
    else:
        pd.DataFrame().to_parquet(mc_path, index=False)
        print(f"Method comparison table (empty): {mc_path}")

    # Markdown table documents
    summary_md = OUT_DIR / "mcs_asset_level_analysis_summary.md"
    summary_md.write_text("# MCS asset-level analysis summary\n\n" + df_to_markdown_table(summary_df), encoding="utf-8")
    print(f"Summary markdown: {summary_md}")

    ts_md = OUT_DIR / "mcs_asset_level_time_sliced_analysis.md"
    ts_md.write_text("# MCS asset-level time-sliced analysis\n\n" + df_to_markdown_table(ts_analysis, max_rows=100), encoding="utf-8")
    print(f"Time-sliced markdown: {ts_md}")

    if not top_configs.empty:
        top_md = OUT_DIR / "mcs_asset_level_top_configs.md"
        top_md.write_text("# MCS asset-level top configurations\n\n" + df_to_markdown_table(top_configs), encoding="utf-8")
        top_configs.to_parquet(OUT_DIR / "mcs_asset_level_top_configs.parquet", index=False)
        print(f"Top configs: {top_md}")

    if method_comparison_df is not None and not method_comparison_df.empty:
        mc_md = OUT_DIR / "mcs_method_comparison_tables.md"
        mc_md.write_text("# MCS method comparison tables\n\n" + df_to_markdown_table(method_comparison_df), encoding="utf-8")
        print(f"Method comparison markdown: {mc_md}")

    fig_dir = base / out["figures"]["path"]
    generate_figures(metrics, ts_analysis if not ts_analysis.empty else time_sliced, time_sliced, risk_series, config, fig_dir)

    report_path = base / out["report"]["path"]
    write_report(
        report_path, config, primary, tail, sim_diag, dist_diag,
        robust, perf, len(metrics), ts_analysis, top_configs, method_comparison_df,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
