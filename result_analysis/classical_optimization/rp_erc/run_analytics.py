"""
Risk Parity (ERC) classical optimization result analysis.

Based on llm.json in the same folder: loads config and metrics, runs data validation,
risk contribution balance, portfolio quality, baseline comparison, downside/tail risk,
portfolio structure, distributional diagnostics, robustness/sensitivity, ranking,
and computational performance. Writes markdown tables and report sections to this folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ============================================================
# Paths – config is llm.json in the same folder
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
CONFIG_PATH = THIS_DIR / "llm.json"


def find_project_root(start: Path) -> Path:
    """Walk upward until we find the project root containing 'results/'."""
    current = start
    for _ in range(10):
        if (current / "results").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not locate project root containing 'results/' directory")


PROJECT_ROOT = find_project_root(THIS_DIR)
OUT_ROOT = THIS_DIR

# ============================================================
# Config & input loading
# ============================================================


def load_config() -> dict:
    """Load analysis config from llm.json (same folder as this script)."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"llm.json not found at: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(p: str) -> Path:
    """Resolve path: absolute, relative to config dir, or project root."""
    path = Path(p)
    if path.is_absolute():
        return path
    for base in (CONFIG_PATH.parent, PROJECT_ROOT):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (PROJECT_ROOT / path).resolve()


def load_inputs(config: dict) -> pd.DataFrame:
    """Load metrics parquet (ERC has no efficient frontier)."""
    inputs = config["inputs"]
    metrics_path = resolve_path(inputs["metrics_table"])
    return pd.read_parquet(metrics_path)


# ============================================================
# Data validation
# ============================================================


def validate_metrics(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply required metrics and consistency checks from llm.json."""
    dv = config.get("data_validation", {})
    required = dv.get("required_metrics", [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required metrics: {missing}")

    df = df.dropna(subset=[c for c in required if c in df.columns])

    for check in dv.get("consistency_checks", []):
        name, rule = check.get("name"), check.get("rule", "")
        if name == "non_negative_risk_contributions" and "min_risk_contribution" in df.columns:
            df = df[df["min_risk_contribution"] >= 0]
        elif name == "finite_deviation_score" and "risk_parity_deviation_score" in df.columns:
            df = df[np.isfinite(df["risk_parity_deviation_score"])]

    return df.reset_index(drop=True)


# ============================================================
# Aggregation and markdown helpers
# ============================================================


def agg_stats(
    df: pd.DataFrame,
    metrics: list[str],
    aggregations: list[str] | None = None,
) -> pd.DataFrame:
    """Compute mean, median, p95 (or custom) for given metrics."""
    if aggregations is None:
        aggregations = ["mean", "median", "p95"]
    stats: dict[str, Any] = {}
    for m in metrics:
        if m not in df.columns:
            continue
        s = df[m].dropna()
        if len(s) == 0:
            continue
        if "mean" in aggregations:
            stats[f"{m}_mean"] = float(s.mean())
        if "median" in aggregations:
            stats[f"{m}_median"] = float(s.median())
        if "p95" in aggregations:
            stats[f"{m}_p95"] = float(s.quantile(0.95))
    return pd.DataFrame([stats]) if stats else pd.DataFrame()


def markdown_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    """Convert DataFrame to markdown table."""
    if df.empty:
        return "_No data available._"
    d = df.head(max_rows)
    cols = list(d.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in d.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                v = ""
            elif isinstance(v, float):
                v = round(v, 6)
            cells.append(str(v)[:50])
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def write_md(path: Path, title: str, df: pd.DataFrame, max_rows: int = 100) -> None:
    """Write a markdown file with title and table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"# {title}\n\n{markdown_table(df, max_rows=max_rows)}\n"
    path.write_text(text, encoding="utf-8")


# ============================================================
# Primary risk parity analysis
# ============================================================


def risk_contribution_balance(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Risk contribution balance metrics (mean, median, p95)."""
    cfg = config.get("primary_risk_parity_analysis", {}).get("risk_contribution_balance", {})
    mets = [m for m in cfg.get("metrics", []) if m in metrics.columns]
    agg = cfg.get("aggregation", ["mean", "median", "p95"])
    return agg_stats(metrics, mets, agg)


def portfolio_quality_metrics(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Portfolio quality (return, volatility, Sharpe, Sortino, max_drawdown, Calmar)."""
    cfg = config.get("primary_risk_parity_analysis", {}).get("portfolio_quality_metrics", {})
    mets = [m for m in cfg.get("metrics", []) if m in metrics.columns]
    agg = cfg.get("aggregation", ["mean", "median", "p95"])
    return agg_stats(metrics, mets, agg)


# ============================================================
# Baseline comparison
# ============================================================


def baseline_comparison(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Baseline comparison metrics aggregation."""
    cfg = config.get("baseline_comparison_analysis", {})
    if not cfg.get("enabled", True):
        return pd.DataFrame()
    mets = [m for m in cfg.get("comparison_metrics", []) if m in metrics.columns]
    agg = cfg.get("aggregation", ["mean", "median"])
    return agg_stats(metrics, mets, agg)


# ============================================================
# Downside and tail risk
# ============================================================


def downside_tail_risk(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Downside and tail risk metrics."""
    cfg = config.get("downside_and_tail_risk_analysis", {})
    mets = [m for m in cfg.get("metrics", []) if m in metrics.columns]
    return agg_stats(metrics, mets, ["mean", "median", "p95"])


# ============================================================
# Portfolio structure
# ============================================================


def portfolio_structure_analysis(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Structure metrics (num_assets, HHI, effective N, weight_entropy, correlation)."""
    cfg = config.get("portfolio_structure_analysis", {})
    mets = [m for m in cfg.get("metrics", []) if m in metrics.columns]
    agg = cfg.get("aggregation", ["mean", "median"])
    return agg_stats(metrics, mets, agg)


# ============================================================
# Distributional diagnostics
# ============================================================


def distributional_diagnostics(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Skewness, kurtosis, Jarque-Bera."""
    cfg = config.get("distributional_diagnostics", {})
    mets = [m for m in cfg.get("metrics", []) if m in metrics.columns]
    return agg_stats(metrics, mets, ["mean", "median"])


# ============================================================
# Robustness and sensitivity
# ============================================================


def robustness_sensitivity(metrics: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """By covariance_estimator and estimation_window."""
    out: dict[str, pd.DataFrame] = {}
    cfg = config.get("robustness_and_sensitivity", {})

    est_cfg = cfg.get("covariance_estimator_sensitivity", {})
    if est_cfg and "covariance_estimator" in metrics.columns:
        compare = est_cfg.get("compare_estimators", [])
        mets = [m for m in est_cfg.get("metrics", []) if m in metrics.columns]
        sub = metrics[metrics["covariance_estimator"].isin(compare)] if compare else metrics
        if not sub.empty and mets:
            by_est = sub.groupby("covariance_estimator")[mets].agg(["mean", "median"]).reset_index()
            out["covariance_estimator_sensitivity"] = by_est

    win_cfg = cfg.get("window_sensitivity", {})
    if win_cfg and "estimation_window" in metrics.columns:
        mets = [m for m in win_cfg.get("metrics", []) if m in metrics.columns]
        if mets:
            by_win = metrics.groupby("estimation_window")[mets].agg(["mean", "median"]).reset_index()
            out["window_sensitivity"] = by_win

    return out


# ============================================================
# Ranking analysis
# ============================================================


def ranking_analysis(metrics: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Composite ranking by weighted metrics; return top_k and percentiles."""
    cfg = config.get("ranking_analysis", {})
    ranking_metrics = cfg.get("ranking_metrics", [])
    selection = cfg.get("selection", {})
    top_k = selection.get("top_k", 25)
    percentiles = selection.get("also_report_percentiles", [1, 5, 10])

    score = np.zeros(len(metrics))
    for r in ranking_metrics:
        metric = r.get("metric")
        if metric not in metrics.columns:
            continue
        w = r.get("weight", 1.0)
        higher = r.get("higher_is_better", True)
        values = metrics[metric].astype(float).fillna(0)
        vmin, vmax = values.min(), values.max()
        if vmax > vmin:
            norm = (values - vmin) / (vmax - vmin)
        else:
            norm = values * 0
        if not higher:
            norm = 1.0 - norm
        score += norm * w

    ranked = metrics.copy()
    ranked["_composite_score"] = score
    ranked = ranked.sort_values("_composite_score", ascending=False).reset_index(drop=True)
    top_df = ranked.head(top_k).drop(columns=["_composite_score"], errors="ignore")

    pct_stats: dict[str, float] = {}
    for p in percentiles:
        idx = max(0, int(len(ranked) * p / 100) - 1)
        if idx < len(ranked):
            pct_stats[f"percentile_{p}"] = float(ranked.iloc[idx]["_composite_score"])

    return top_df, pct_stats


# ============================================================
# Computational performance
# ============================================================


def computational_performance(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Runtime metrics aggregation."""
    cfg = config.get("computational_performance_analysis", {})
    mets = [m for m in cfg.get("metrics", []) if m in metrics.columns]
    agg = cfg.get("aggregation", ["mean", "median", "p95"])
    return agg_stats(metrics, mets, agg)


# ============================================================
# Report sections and markdown tables
# ============================================================


def write_report_sections(
    out_dir: Path,
    config: dict,
    n_metrics: int,
    risk_balance_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    downside_df: pd.DataFrame,
    structure_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    robustness: dict[str, pd.DataFrame],
    top_ranked: pd.DataFrame,
    runtime_df: pd.DataFrame,
    percentile_stats: dict,
) -> list[str]:
    """Write report section .md files per outputs.report.include_sections."""
    report_cfg = config.get("outputs", {}).get("report", {})
    sections = report_cfg.get("include_sections", [])
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def save(name: str, content: list[str]) -> None:
        if content:
            path = out_dir / name
            path.write_text("\n".join(content), encoding="utf-8")
            written.append(name)
            print(f"  {name}")

    if "experimental_setup" in sections:
        save("erc_experimental_setup.md", [
            "# Experimental setup",
            "",
            f"Config: `{CONFIG_PATH.name}`. Total portfolios (after validation): {n_metrics}.",
            "",
            "Analysis uses Risk Parity (Equal Risk Contribution) classical optimization. "
            "Inputs: metrics table from `results/classical_optimization/rp_erc_metrics.parquet`.",
            "",
        ])

    if "risk_parity_and_erc_theory" in sections:
        save("erc_risk_parity_theory.md", [
            "# Risk parity and ERC theory",
            "",
            "Equal Risk Contribution (ERC) portfolios balance risk contributions across assets rather than "
            "weights. Lower risk_parity_deviation_score and equal_risk_gap indicate better adherence to "
            "equal risk contribution; diversification is by risk, not by weight.",
            "",
        ])

    if "risk_contribution_balance_results" in sections and not risk_balance_df.empty:
        lines = ["# Risk contribution balance results", "", markdown_table(risk_balance_df, max_rows=50), ""]
        save("erc_risk_contribution_balance_results.md", lines)

    if "portfolio_quality_and_downside_risk" in sections:
        lines = ["# Portfolio quality and downside risk", ""]
        if not quality_df.empty:
            lines.append(markdown_table(quality_df))
            lines.append("")
        if not downside_df.empty:
            lines.append("## Downside and tail risk")
            lines.append("")
            lines.append(markdown_table(downside_df))
            lines.append("")
        if len(lines) > 2:
            save("erc_portfolio_quality_and_downside_risk.md", lines)

    if "comparison_with_baselines" in sections and not baseline_df.empty:
        lines = ["# Comparison with baselines", "", markdown_table(baseline_df), ""]
        save("erc_comparison_with_baselines.md", lines)

    if "robustness_and_sensitivity" in sections and robustness:
        lines = ["# Robustness and sensitivity", ""]
        for name, df in robustness.items():
            if not df.empty:
                lines.append(f"## {name}")
                lines.append("")
                lines.append(markdown_table(df))
                lines.append("")
        save("erc_robustness_and_sensitivity.md", lines)

    if "statistical_significance" in sections:
        save("erc_statistical_significance.md", [
            "# Statistical significance",
            "",
            "Statistical significance analysis (e.g. Jobson–Korkie, Memmel Sharpe correction, "
            "Diebold–Mariano) can be added when corresponding metrics or test outputs are available.",
            "",
        ])

    if "computational_performance" in sections and not runtime_df.empty:
        lines = ["# Computational performance", "", markdown_table(runtime_df), ""]
        save("erc_computational_performance.md", lines)

    if not top_ranked.empty or percentile_stats:
        lines = ["# Top portfolios (ranking)", ""]
        if not top_ranked.empty:
            lines.append(markdown_table(top_ranked, max_rows=30))
            lines.append("")
        if percentile_stats:
            lines.append("## Composite score percentiles")
            lines.append("")
            lines.append("| Percentile | Score |")
            lines.append("|------------|-------|")
            for k, v in percentile_stats.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")
        save("erc_ranking.md", lines)

    if "discussion" in sections:
        save("erc_discussion.md", [
            "# Discussion",
            "",
            "Analysis performed per llm.json: risk contribution balance, portfolio quality, baseline "
            "comparison, downside/tail risk, portfolio structure, distributional diagnostics, "
            "robustness/sensitivity, ranking, and computational performance. ERC encourages diversification "
            "by equalizing risk contributions; tail risk and diagnostics are reported for context.",
            "",
        ])

    return written


def write_index_md(out_dir: Path, table_names: list[str], section_names: list[str]) -> None:
    """Write an index file listing all generated .md files."""
    lines = [
        "# Risk Parity (ERC) analysis – index",
        "",
        "Generated report and table files in this folder:",
        "",
        "## Summary tables",
        "",
    ]
    for name in table_names:
        lines.append(f"- [{name}]({name})")
    lines.append("")
    lines.append("## Report sections")
    lines.append("")
    for name in section_names:
        lines.append(f"- [{name}]({name})")
    lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "erc_analysis_index.md").write_text("\n".join(lines), encoding="utf-8")
    print("  erc_analysis_index.md")


# ============================================================
# Main pipeline
# ============================================================


def main() -> int:
    config = load_config()

    print("Loading data (based on llm.json in this folder)...")
    metrics = load_inputs(config)
    print(f"Metrics: {metrics.shape}")

    metrics = validate_metrics(metrics, config)
    print(f"After validation: {len(metrics)} rows")

    risk_balance_df = risk_contribution_balance(metrics, config)
    quality_df = portfolio_quality_metrics(metrics, config)
    baseline_df = baseline_comparison(metrics, config)
    downside_df = downside_tail_risk(metrics, config)
    structure_df = portfolio_structure_analysis(metrics, config)
    dist_df = distributional_diagnostics(metrics, config)
    robustness = robustness_sensitivity(metrics, config)
    top_ranked, percentile_stats = ranking_analysis(metrics, config)
    runtime_df = computational_performance(metrics, config)

    # Output markdown tables per llm.json outputs.markdown_tables
    tables_cfg = config.get("outputs", {}).get("markdown_tables", {})
    root_path = OUT_ROOT
    root_path.mkdir(parents=True, exist_ok=True)

    table_specs = [
        ("erc_risk_contribution_balance.md", risk_balance_df, "ERC Risk Contribution Balance"),
        ("erc_portfolio_quality_summary.md", quality_df, "ERC Portfolio Quality Summary"),
        ("erc_baseline_comparison.md", baseline_df, "ERC Baseline Comparison"),
        ("erc_structure_metrics.md", structure_df, "ERC Structure Metrics"),
        ("erc_runtime_profile.md", runtime_df, "ERC Runtime Profile"),
    ]

    table_names_written: list[str] = []
    tables_by_name = {t["name"]: t for t in tables_cfg.get("tables", [])}
    for name, df, title in table_specs:
        if df is not None and not df.empty:
            out_path = root_path / name
            contents = tables_by_name.get(name, {}).get("contents", [])
            cols = [c for c in contents if c in df.columns] if contents else list(df.columns)
            write_md(out_path, title, df[cols] if cols else df)
            table_names_written.append(name)
            print(f"  {out_path.name}")

    print("Report sections:")
    section_names = write_report_sections(
        OUT_ROOT,
        config,
        len(metrics),
        risk_balance_df,
        quality_df,
        baseline_df,
        downside_df,
        structure_df,
        dist_df,
        robustness,
        top_ranked,
        runtime_df,
        percentile_stats,
    )

    print("Index:")
    write_index_md(OUT_ROOT, table_names_written, section_names)

    print("ERC result analysis completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
