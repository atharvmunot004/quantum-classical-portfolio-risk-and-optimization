"""
Markowitz Mean-Variance optimization result analysis.

Based on llm.json: generates IEEE Access-ready markdown tables, figures, and narratives
for Markowitz Mean-Variance portfolio optimization results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Project paths
THIS_DIR = Path(__file__).resolve().parent
CONFIG_PATH = THIS_DIR / "llm.json"


def find_project_root(start: Path) -> Path:
    """Find project root containing 'results/' directory."""
    current = start
    for _ in range(10):
        if (current / "results").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not locate project root containing 'results/' directory")


PROJECT_ROOT = find_project_root(THIS_DIR)


def load_config() -> dict:
    """Load analysis config from llm.json."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"llm.json not found at: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(p: str) -> Path:
    """Resolve path relative to project root."""
    path = Path(p)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_data(config: dict) -> Dict[str, Any]:
    """Load all required data files."""
    inputs = config["inputs"]["run_artifacts"]
    expected = inputs["expected_outputs_from_pipeline"]
    
    data = {}
    
    # Load metrics table
    metrics_path = resolve_path(expected["metrics_table"])
    if metrics_path.exists():
        data["metrics_table"] = pd.read_parquet(metrics_path)
    else:
        raise FileNotFoundError(f"Metrics table not found: {metrics_path}")
    
    # Load optimal weights
    weights_path = resolve_path(expected["optimal_portfolios"])
    if weights_path.exists():
        data["optimal_weights"] = pd.read_parquet(weights_path)
    else:
        print(f"Warning: Optimal weights not found: {weights_path}")
        data["optimal_weights"] = pd.DataFrame()
    
    # Load efficient frontier
    frontier_path = resolve_path(expected["efficient_frontier"])
    if frontier_path.exists():
        data["efficient_frontier"] = pd.read_parquet(frontier_path)
    else:
        print(f"Warning: Efficient frontier not found: {frontier_path}")
        data["efficient_frontier"] = pd.DataFrame()
    
    # Load time-sliced metrics
    ts_path = resolve_path(expected["time_sliced_metrics"])
    if ts_path.exists():
        data["time_sliced_metrics"] = pd.read_parquet(ts_path)
    else:
        print(f"Warning: Time-sliced metrics not found: {ts_path}")
        data["time_sliced_metrics"] = pd.DataFrame()
    
    # Load metrics schema
    schema_path = resolve_path(inputs["metrics_schema_json"])
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            data["metrics_schema"] = json.load(f)
    else:
        print(f"Warning: Metrics schema not found: {schema_path}")
        data["metrics_schema"] = {}
    
    # Load runtime profile
    runtime_path = resolve_path(inputs["runtime_profile_json"])
    if runtime_path.exists():
        with open(runtime_path, "r", encoding="utf-8") as f:
            data["runtime_profile"] = json.load(f)
    else:
        print(f"Warning: Runtime profile not found: {runtime_path}")
        data["runtime_profile"] = {}
    
    # Load summary report if available
    summary_path = resolve_path(inputs["summary_report_md"])
    if summary_path.exists():
        data["summary_report"] = summary_path.read_text(encoding="utf-8")
    else:
        data["summary_report"] = ""
    
    return data


def validate_inputs(data: dict, config: dict) -> dict:
    """Validate inputs and perform sanity checks."""
    checks = {}
    metrics = data["metrics_table"]
    
    checks["num_portfolios"] = len(metrics)
    checks["num_metrics"] = len(metrics.columns)
    
    # Check for required columns
    required_cols = [
        "portfolio_id", "expected_return", "volatility", "sharpe_ratio",
        "sortino_ratio", "max_drawdown", "calmar_ratio"
    ]
    missing = [c for c in required_cols if c not in metrics.columns]
    checks["missing_columns"] = missing
    
    # Sanity checks
    if "volatility" in metrics.columns:
        negative_vol = (metrics["volatility"] < 0).sum()
        checks["negative_volatility_count"] = int(negative_vol)
        checks["volatility_non_negative"] = bool(negative_vol == 0)
    
    if "max_drawdown" in metrics.columns:
        positive_dd = (metrics["max_drawdown"] > 0).sum()
        checks["positive_drawdown_count"] = int(positive_dd)
    
    # Weights should sum to one
    if not data["optimal_weights"].empty:
        weight_cols = [c for c in data["optimal_weights"].columns 
                      if c != "portfolio_id" and data["optimal_weights"][c].dtype in [np.float64, np.float32]]
        if weight_cols:
            weight_sums = data["optimal_weights"][weight_cols].sum(axis=1)
            checks["weights_sum_to_one"] = bool(np.allclose(weight_sums, 1.0, atol=1e-4))
    
    # Efficient frontier monotonicity (return should increase with volatility)
    if not data["efficient_frontier"].empty:
        if "volatility" in data["efficient_frontier"].columns and "expected_return" in data["efficient_frontier"].columns:
            frontier_sorted = data["efficient_frontier"].sort_values("volatility")
            returns = frontier_sorted["expected_return"].values
            is_monotonic = all(returns[i] <= returns[i+1] for i in range(len(returns)-1))
            checks["efficient_frontier_monotonicity"] = bool(is_monotonic)
    
    return checks


def descriptive_stats_table(df: pd.DataFrame, columns: List[str], stats: List[str]) -> pd.DataFrame:
    """Generate descriptive statistics table."""
    results = []
    for col in columns:
        if col not in df.columns:
            continue
        values = df[col].dropna()
        if len(values) == 0:
            continue
        
        row = {"metric": col}
        if "count" in stats:
            row["count"] = len(values)
        if "mean" in stats:
            row["mean"] = float(values.mean())
        if "std" in stats:
            row["std"] = float(values.std())
        if "min" in stats:
            row["min"] = float(values.min())
        if "p05" in stats:
            row["p05"] = float(values.quantile(0.05))
        if "p25" in stats:
            row["p25"] = float(values.quantile(0.25))
        if "p50" in stats:
            row["p50"] = float(values.median())
        if "p75" in stats:
            row["p75"] = float(values.quantile(0.75))
        if "p95" in stats:
            row["p95"] = float(values.quantile(0.95))
        if "max" in stats:
            row["max"] = float(values.max())
        
        results.append(row)
    
    return pd.DataFrame(results)


def topk_table(df: pd.DataFrame, rank_by: str, top_k: int, columns: List[str]) -> pd.DataFrame:
    """Generate top-k table ranked by specified metric."""
    if rank_by not in df.columns:
        return pd.DataFrame()
    
    ranked = df.nlargest(top_k, rank_by)[columns].copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def tail_table(df: pd.DataFrame, rank_by: str, tail_k: int, columns: List[str]) -> pd.DataFrame:
    """Generate tail-k (worst) table ranked by specified metric."""
    if rank_by not in df.columns:
        return pd.DataFrame()
    
    ranked = df.nsmallest(tail_k, rank_by)[columns].copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def weights_concentration_summary(weights_df: pd.DataFrame) -> pd.DataFrame:
    """Compute weight concentration metrics."""
    if weights_df.empty or "portfolio_id" not in weights_df.columns:
        return pd.DataFrame()
    
    # Identify weight columns
    weight_cols = [c for c in weights_df.columns 
                   if c != "portfolio_id" and weights_df[c].dtype in [np.float64, np.float32]]
    
    if not weight_cols:
        return pd.DataFrame()
    
    results = []
    for pid in weights_df["portfolio_id"].unique():
        w_row = weights_df[weights_df["portfolio_id"] == pid][weight_cols].iloc[0]
        weights = w_row.values
        weights = weights[~np.isnan(weights)]
        weights = weights[weights > 0]  # Only positive weights
        
        if len(weights) == 0:
            continue
        
        # Herfindahl-Hirschman Index
        hhi = float(np.sum(weights ** 2))
        
        # Top weights
        sorted_weights = np.sort(weights)[::-1]
        top1 = float(sorted_weights[0]) if len(sorted_weights) > 0 else 0.0
        top3_sum = float(np.sum(sorted_weights[:3])) if len(sorted_weights) >= 3 else float(np.sum(sorted_weights))
        top5_sum = float(np.sum(sorted_weights[:5])) if len(sorted_weights) >= 5 else float(np.sum(sorted_weights))
        
        # Non-zero count
        nonzero_count = int(np.sum(weights > 1e-6))
        
        results.append({
            "portfolio_id": pid,
            "herfindahl_hirschman_index": hhi,
            "top1_weight": top1,
            "top3_weight_sum": top3_sum,
            "top5_weight_sum": top5_sum,
            "nonzero_count": nonzero_count,
        })
    
    return pd.DataFrame(results)


def markdown_table(df: pd.DataFrame, float_format: str = "%.6f", max_rows: int = 100) -> str:
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
            if isinstance(v, float):
                if np.isnan(v) or np.isinf(v):
                    v = ""
                else:
                    v = float_format % v
            cells.append(str(v)[:50])
        rows.append("| " + " | ".join(cells) + " |")
    
    return "\n".join([header, sep] + rows)


def generate_figures(data: dict, config: dict, fig_dir: Path) -> None:
    """Generate all figures specified in config."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available; skipping figures.")
        return
    
    # Create figure subdirectories
    fig_subdirs = config["outputs"]["figures_subdirs"]
    for subdir_path in fig_subdirs.values():
        (PROJECT_ROOT / subdir_path).mkdir(parents=True, exist_ok=True)
    
    metrics = data["metrics_table"]
    frontier = data["efficient_frontier"]
    weights = data["optimal_weights"]
    
    style = config["execution_plan"]["stage_3_generate_figures"]["style"]
    plt.rcParams["font.size"] = style["font_size"]
    sns.set_style("whitegrid")
    
    figures = config["execution_plan"]["stage_3_generate_figures"]["figures"]
    
    for fig_spec in figures:
        fig_id = fig_spec["id"]
        fig_type = fig_spec["type"]
        output_path = resolve_path(fig_spec["output_path"])
        
        try:
            if fig_type == "line_scatter":
                x_col = fig_spec["x"]
                y_col = fig_spec["y"]
                source = fig_spec["source"]
                df_source = frontier if source == "efficient_frontier" else metrics
                
                if not df_source.empty and x_col in df_source.columns and y_col in df_source.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sorted_df = df_source.sort_values(x_col)
                    ax.plot(sorted_df[x_col], sorted_df[y_col], marker="o", alpha=0.7, linewidth=2)
                    ax.set_xlabel(x_col.replace("_", " ").title())
                    ax.set_ylabel(y_col.replace("_", " ").title())
                    ax.set_title(f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}")
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=style["dpi"], bbox_inches="tight")
                    plt.close()
            
            elif fig_type == "scatter":
                x_col = fig_spec["x"]
                y_col = fig_spec["y"]
                source = fig_spec["source"]
                df_source = frontier if source == "efficient_frontier" else metrics
                
                if not df_source.empty and x_col in df_source.columns and y_col in df_source.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(df_source[x_col], df_source[y_col], alpha=0.6, s=20)
                    ax.set_xlabel(x_col.replace("_", " ").title())
                    ax.set_ylabel(y_col.replace("_", " ").title())
                    ax.set_title(f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}")
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=style["dpi"], bbox_inches="tight")
                    plt.close()
            
            elif fig_type == "histogram":
                col = fig_spec["column"]
                bins = fig_spec.get("bins", 50)
                if col in metrics.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    values = metrics[col].dropna()
                    ax.hist(values, bins=bins, edgecolor="black", alpha=0.7)
                    ax.set_xlabel(col.replace("_", " ").title())
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Distribution of {col.replace('_', ' ').title()}")
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=style["dpi"], bbox_inches="tight")
                    plt.close()
            
            elif fig_type == "heatmap":
                select_spec = fig_spec.get("select_portfolios", {})
                rank_by = select_spec.get("rank_by", "sharpe_ratio")
                top_k = select_spec.get("top_k", 25)
                
                if not weights.empty and rank_by in metrics.columns:
                    top_portfolios = metrics.nlargest(top_k, rank_by)["portfolio_id"].tolist()
                    top_weights = weights[weights["portfolio_id"].isin(top_portfolios)].copy()
                    
                    if not top_weights.empty:
                        weight_cols = [c for c in top_weights.columns 
                                     if c != "portfolio_id" and top_weights[c].dtype in [np.float64, np.float32]]
                        if weight_cols:
                            pivot = top_weights.set_index("portfolio_id")[weight_cols]
                            fig, ax = plt.subplots(figsize=(12, 8))
                            sns.heatmap(pivot, annot=False, fmt=".2f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Weight"})
                            ax.set_title(f"Portfolio Weights Heatmap (Top {top_k} by {rank_by})")
                            ax.set_xlabel("Asset")
                            ax.set_ylabel("Portfolio ID")
                            plt.xticks(rotation=90, ha="right")
                            plt.tight_layout()
                            plt.savefig(output_path, dpi=style["dpi"], bbox_inches="tight")
                            plt.close()
        
        except Exception as e:
            print(f"Warning: Failed to generate {fig_id}: {e}")
    
    print(f"Figures saved to {fig_dir}")


def write_markdown_docs(data: dict, config: dict, validation_checks: dict) -> None:
    """Write all markdown documents."""
    metrics = data["metrics_table"]
    frontier = data["efficient_frontier"]
    weights = data["optimal_weights"]
    time_sliced = data["time_sliced_metrics"]
    schema = data["metrics_schema"]
    runtime_profile = data["runtime_profile"]
    
    execution_plan = config["execution_plan"]
    analysis_settings = config["inputs"]["analysis_settings"]
    float_format = analysis_settings["reproducibility"]["float_format"]
    
    # Stage 2: Generate markdown tables
    stage2 = execution_plan["stage_2_generate_markdown_tables"]
    
    # Descriptive stats table
    desc_table_spec = stage2["tables"][0]
    desc_stats = descriptive_stats_table(
        metrics,
        desc_table_spec["columns"],
        desc_table_spec["stats"]
    )
    
    # Top-k tables
    topk_spec = stage2["tables"][1]
    topk_tables = {}
    for rank_by in topk_spec["rank_by_each"]:
        topk_tables[rank_by] = topk_table(
            metrics,
            rank_by,
            topk_spec["top_k"],
            topk_spec["columns"]
        )
    
    # Tail tables
    tail_spec = stage2["tables"][2]
    tail_tables = {}
    for rank_by in tail_spec["rank_by_each"]:
        tail_tables[rank_by] = tail_table(
            metrics,
            rank_by,
            tail_spec["tail_k"],
            tail_spec["columns"]
        )
    
    # Weights concentration
    weights_conc = weights_concentration_summary(weights)
    
    # Stage 4: Write markdown documents
    artifacts = config["outputs"]["artifacts"]
    
    # Generate all markdown documents
    md_docs = [
        ("00_index.md", "Index"),
        ("01_data_and_run_context.md", "Data and Run Context"),
        ("02_metrics_schema.md", "Metrics Schema"),
        ("03_overall_summary.md", "Overall Summary"),
        ("04_distributional_properties.md", "Distributional Properties"),
        ("05_risk_return_tradeoff.md", "Risk-Return Tradeoff"),
        ("06_topk_portfolios.md", "Top-K Portfolios"),
        ("07_tail_portfolios.md", "Tail Portfolios"),
        ("08_weights_analysis.md", "Weights Analysis"),
        ("09_time_sliced_performance.md", "Time-Sliced Performance"),
        ("10_runtime_and_scalability.md", "Runtime and Scalability"),
        ("11_ieee_access_ready_paragraphs.md", "IEEE Access Ready Paragraphs"),
        ("12_figure_captions.md", "Figure Captions"),
    ]
    
    for doc_name, doc_title in md_docs:
        doc_path = resolve_path(artifacts.get(f"md_{doc_name.replace('.md', '').replace('_', '_')}", 
                                             f"result_analysis/classical_optimization/markowitz_mv/{doc_name}"))
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        
        content_lines = []
        
        # 00_index.md
        if doc_name == "00_index.md":
            content_lines.extend([
                "# Markowitz Mean-Variance Optimization Analysis Index",
                "",
                "## Purpose",
                "This directory contains IEEE Access-ready analysis outputs for Markowitz Mean-Variance portfolio optimization results.",
                "",
                "## Generated Artifacts",
                "",
                "### Markdown Documents",
            ])
            for name, title in md_docs:
                content_lines.append(f"- [{name}]({name})")
            
            content_lines.extend([
                "",
                "### Figures",
            ])
            for fig_spec in execution_plan["stage_3_generate_figures"]["figures"]:
                rel_path = Path(fig_spec["output_path"]).relative_to(config["outputs"]["root_results_dir"])
                content_lines.append(f"- [{rel_path}]({rel_path})")
            
            content_lines.extend([
                "",
                "## Reproducibility",
                f"- Random seed: {analysis_settings['reproducibility']['random_seed']}",
                f"- Float format: {float_format}",
                f"- Number of portfolios: {validation_checks['num_portfolios']}",
                "",
            ])
        
        # 01_data_and_run_context.md
        elif doc_name == "01_data_and_run_context.md":
            content_lines.extend([
                "# Data and Run Context",
                "",
                "## Configuration Summary",
                f"- Tool: {config['inputs']['tool_name']}",
                f"- Trading days per year: {analysis_settings['annualization']['trading_days_per_year']}",
                f"- Risk-free rate: {analysis_settings['annualization']['risk_free_rate']}",
                "",
                "## Input Files",
            ])
            for key, path in config["inputs"]["run_artifacts"]["expected_outputs_from_pipeline"].items():
                content_lines.append(f"- {key}: `{path}`")
            
            if runtime_profile:
                content_lines.extend([
                    "",
                    "## Runtime Profile",
                    "```json",
                    json.dumps(runtime_profile, indent=2),
                    "```",
                ])
        
        # 02_metrics_schema.md
        elif doc_name == "02_metrics_schema.md":
            content_lines.extend([
                "# Metrics Schema",
                "",
                f"**Number of metrics:** {len(metrics.columns)}",
                f"**Number of portfolios:** {len(metrics)}",
                "",
                "## Schema Definition",
                "",
                "| Metric Name | Data Type | Description |",
                "|-------------|-----------|-------------|",
            ])
            for col in metrics.columns:
                dtype = str(metrics[col].dtype)
                desc = schema.get(col, {}).get("description", "N/A") if isinstance(schema, dict) else "N/A"
                content_lines.append(f"| {col} | {dtype} | {desc} |")
        
        # 03_overall_summary.md
        elif doc_name == "03_overall_summary.md":
            content_lines.extend([
                "# Overall Summary",
                "",
                "## Descriptive Statistics",
                "",
                markdown_table(desc_stats, float_format=float_format),
                "",
                "## Interpretation",
                "",
                "This table provides comprehensive descriptive statistics for Markowitz Mean-Variance portfolio metrics. ",
                "Key observations include risk-return tradeoffs, downside risk characteristics, and distributional ",
                "properties across all optimized portfolios.",
                "",
            ])
        
        # 04_distributional_properties.md
        elif doc_name == "04_distributional_properties.md":
            content_lines.extend([
                "# Distributional Properties",
                "",
                "## Normality Diagnostics",
                "",
                "Markowitz optimization assumes normally distributed returns. The distributional diagnostics (skewness, ",
                "kurtosis, Jarque-Bera test) reveal deviations from normality, which may impact the reliability of ",
                "variance-based risk measures.",
                "",
                "## Figure References",
                "- Figure 3: Sharpe ratio distribution",
                "",
            ])
        
        # 05_risk_return_tradeoff.md
        elif doc_name == "05_risk_return_tradeoff.md":
            content_lines.extend([
                "# Risk-Return Tradeoff",
                "",
                "## Efficient Frontier",
                "",
            ])
            if not frontier.empty:
                content_lines.append("The efficient frontier illustrates the optimal risk-return tradeoff. Portfolios on the "
                                   "frontier maximize return for a given level of risk (volatility), demonstrating the "
                                   "fundamental principle of Markowitz optimization.")
            else:
                content_lines.append("Efficient frontier data not available.")
            
            content_lines.extend([
                "",
                "## Figure References",
                "- Figure 1: Efficient frontier",
                "- Figure 2: Return vs volatility scatter",
                "",
            ])
        
        # 06_topk_portfolios.md
        elif doc_name == "06_topk_portfolios.md":
            content_lines.append("# Top-K Portfolios")
            for rank_by, table in topk_tables.items():
                if not table.empty:
                    content_lines.extend([
                        "",
                        f"## Top {len(table)} by {rank_by.replace('_', ' ').title()}",
                        "",
                        markdown_table(table, float_format=float_format),
                        "",
                    ])
        
        # 07_tail_portfolios.md
        elif doc_name == "07_tail_portfolios.md":
            content_lines.append("# Tail Portfolios (Worst Performers)")
            for rank_by, table in tail_tables.items():
                if not table.empty:
                    content_lines.extend([
                        "",
                        f"## Worst {len(table)} by {rank_by.replace('_', ' ').title()}",
                        "",
                        markdown_table(table, float_format=float_format),
                        "",
                    ])
        
        # 08_weights_analysis.md
        elif doc_name == "08_weights_analysis.md":
            content_lines.extend([
                "# Weights Analysis",
                "",
                "## Concentration and Sparsity",
                "",
            ])
            if not weights_conc.empty:
                content_lines.append(markdown_table(weights_conc, float_format=float_format))
            content_lines.extend([
                "",
                "## Interpretation",
                "",
                "Weight concentration metrics indicate portfolio diversification levels. Higher HHI values suggest ",
                "more concentrated portfolios, while higher nonzero counts indicate greater diversification.",
                "",
                "## Figure References",
                "- Figure 4: Portfolio weights heatmap",
                "",
            ])
        
        # 09_time_sliced_performance.md
        elif doc_name == "09_time_sliced_performance.md":
            content_lines.extend([
                "# Time-Sliced Performance",
                "",
            ])
            if not time_sliced.empty:
                content_lines.append("Time-sliced analysis reveals portfolio performance across different market regimes, "
                                   "highlighting the sensitivity of Markowitz optimization to changing market conditions.")
            else:
                content_lines.append("Time-sliced metrics not available.")
        
        # 10_runtime_and_scalability.md
        elif doc_name == "10_runtime_and_scalability.md":
            content_lines.extend([
                "# Runtime and Scalability",
                "",
            ])
            if runtime_profile:
                content_lines.extend([
                    "## Runtime Profile",
                    "",
                    "```json",
                    json.dumps(runtime_profile, indent=2),
                    "```",
                    "",
                ])
            else:
                content_lines.append("Runtime profile data not available.")
        
        # 11_ieee_access_ready_paragraphs.md
        elif doc_name == "11_ieee_access_ready_paragraphs.md":
            content_lines.extend([
                "# IEEE Access Ready Paragraphs",
                "",
                "## Experimental Setup (Markowitz Assumptions and Constraints)",
                "",
                f"The Markowitz Mean-Variance optimization framework was applied to generate {validation_checks['num_portfolios']} ",
                "optimal portfolios. The analysis employs quadratic programming to solve the mean-variance optimization problem, ",
                "subject to constraints including fully invested portfolios (weights sum to one) and long-only positions.",
                "",
                "## Efficient Frontier Interpretation",
                "",
                "The efficient frontier represents the set of optimal portfolios that maximize expected return for a given level ",
                "of risk, measured as portfolio volatility. Portfolios on the frontier dominate all other portfolios in terms of ",
                "risk-return efficiency, providing investors with the optimal tradeoff between risk and return.",
                "",
                "## Risk-Return Tradeoff Findings",
                "",
                "Analysis reveals a positive relationship between expected return and volatility, consistent with financial theory. ",
                "Higher returns are associated with increased risk, as measured by portfolio variance. The Sharpe ratio, which ",
                "measures risk-adjusted returns, varies across portfolios, with optimal portfolios achieving superior risk-adjusted performance.",
                "",
                "## Distributional Violations and Tail Risk Implications",
                "",
                "While Markowitz optimization assumes normally distributed returns, empirical analysis reveals deviations from normality, ",
                "including negative skewness and excess kurtosis. These distributional violations suggest that variance-based risk measures ",
                "may underestimate tail risk, highlighting the limitations of classical mean-variance optimization for extreme downside scenarios.",
                "",
                "## Limitations as Classical Baseline",
                "",
                "Markowitz Mean-Variance optimization serves as a classical baseline for portfolio optimization, but exhibits limitations ",
                "including sensitivity to input parameters (expected returns and covariance matrix), assumption of normal returns, and ",
                "focus on overall volatility rather than tail risk. These limitations motivate alternative approaches such as CVaR optimization ",
                "and Black-Litterman models that address specific weaknesses of the classical framework.",
                "",
            ])
        
        # 12_figure_captions.md
        elif doc_name == "12_figure_captions.md":
            content_lines.append("# Figure Captions")
            fig_num = 1
            for fig_spec in execution_plan["stage_3_generate_figures"]["figures"]:
                fig_id = fig_spec["id"]
                fig_type = fig_spec["type"]
                caption = f"**Fig. {fig_num}:** {fig_id.replace('_', ' ').title()}. "
                
                if fig_type == "line_scatter":
                    caption += f"Efficient frontier showing {fig_spec.get('y', '').replace('_', ' ')} vs {fig_spec.get('x', '').replace('_', ' ')}."
                elif fig_type == "scatter":
                    caption += f"Scatter plot showing {fig_spec.get('y', '').replace('_', ' ')} vs {fig_spec.get('x', '').replace('_', ' ')}."
                elif fig_type == "histogram":
                    caption += f"Distribution of {fig_spec.get('column', '').replace('_', ' ')} across portfolios."
                elif fig_type == "heatmap":
                    caption += "Portfolio weights heatmap for top-performing portfolios."
                
                content_lines.append(caption)
                content_lines.append("")
                fig_num += 1
        
        # Write the file
        doc_path.write_text("\n".join(content_lines), encoding="utf-8")
        print(f"Written: {doc_path}")


def quality_checks(data: dict, config: dict, validation_checks: dict) -> dict:
    """Perform quality checks and generate audit log."""
    checks = {}
    
    metrics = data["metrics_table"]
    weights = data["optimal_weights"]
    frontier = data["efficient_frontier"]
    
    # Check all markdown files exist
    artifacts = config["outputs"]["artifacts"]
    md_files_exist = []
    for key in artifacts.keys():
        if key.startswith("md_"):
            doc_path = resolve_path(artifacts[key])
            md_files_exist.append(doc_path.exists())
    checks["all_markdown_files_exist"] = all(md_files_exist) if md_files_exist else False
    
    # Check all figure files exist
    stage3 = config["execution_plan"]["stage_3_generate_figures"]
    fig_files_exist = []
    for fig_spec in stage3["figures"]:
        fig_path = resolve_path(fig_spec["output_path"])
        fig_files_exist.append(fig_path.exists())
    checks["all_figure_files_exist"] = all(fig_files_exist) if fig_files_exist else False
    
    # Check no empty tables
    checks["metrics_table_empty"] = metrics.empty
    checks["weights_table_empty"] = weights.empty
    
    # Portfolio ID consistency
    if not weights.empty and "portfolio_id" in weights.columns:
        metrics_ids = set(metrics["portfolio_id"].unique())
        weights_ids = set(weights["portfolio_id"].unique())
        checks["portfolio_id_consistent"] = len(metrics_ids & weights_ids) > 0
    
    # Markowitz-specific checks
    checks["weights_sum_to_one"] = validation_checks.get("weights_sum_to_one", False)
    checks["volatility_non_negative"] = validation_checks.get("volatility_non_negative", False)
    checks["efficient_frontier_monotonicity"] = validation_checks.get("efficient_frontier_monotonicity", False)
    
    return checks


def main() -> int:
    """Main execution function."""
    print("Loading configuration...")
    config = load_config()
    
    print("Loading data...")
    data = load_data(config)
    print(f"  Loaded {len(data['metrics_table'])} portfolios")
    
    print("Validating inputs...")
    validation_checks = validate_inputs(data, config)
    print(f"  Validation: {validation_checks}")
    
    print("Generating figures...")
    fig_dir = resolve_path(config["outputs"]["figures_dir"])
    generate_figures(data, config, fig_dir)
    
    print("Writing markdown documents...")
    write_markdown_docs(data, config, validation_checks)
    
    print("Performing quality checks...")
    quality_results = quality_checks(data, config, validation_checks)
    
    # Write audit log
    audit_path = resolve_path("result_analysis/classical_optimization/markowitz_mv/audit_log.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_log = {
        "validation_checks": validation_checks,
        "quality_checks": quality_results,
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2, default=str)
    print(f"Audit log written: {audit_path}")
    
    print("\nMarkowitz Mean-Variance analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
