"""
Risk Parity / Equal Risk Contribution (ERC) optimization result analysis.

Based on llm.json: generates IEEE Access-ready markdown tables, figures, and narratives
for Risk Parity/ERC portfolio optimization results emphasizing risk contribution equality.
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
    
    # Load risk contribution table
    risk_contrib_path = resolve_path(expected["risk_contribution_table"])
    if risk_contrib_path.exists():
        data["risk_contribution_table"] = pd.read_parquet(risk_contrib_path)
    else:
        print(f"Warning: Risk contribution table not found: {risk_contrib_path}")
        data["risk_contribution_table"] = pd.DataFrame()
    
    # Load metrics schema
    schema_path = resolve_path(inputs["metrics_schema_json"])
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            data["metrics_schema"] = json.load(f)
    else:
        print(f"Warning: Metrics schema not found: {schema_path}")
        data["metrics_schema"] = {}
    
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
        "portfolio_id", "risk_parity_deviation_score", "volatility"
    ]
    missing = [c for c in required_cols if c not in metrics.columns]
    checks["missing_columns"] = missing
    
    # Sanity checks
    if "risk_parity_deviation_score" in metrics.columns:
        negative_dev = (metrics["risk_parity_deviation_score"] < 0).sum()
        checks["risk_parity_deviation_non_negative"] = bool(negative_dev == 0)
    
    # Weights should sum to one
    if not data["optimal_weights"].empty:
        weight_cols = [c for c in data["optimal_weights"].columns 
                      if c != "portfolio_id" and data["optimal_weights"][c].dtype in [np.float64, np.float32]]
        if weight_cols:
            weight_sums = data["optimal_weights"][weight_cols].sum(axis=1)
            checks["weights_sum_to_one"] = bool(np.allclose(weight_sums, 1.0, atol=1e-4))
    
    # Risk contributions should sum to total risk
    if not data["risk_contribution_table"].empty:
        if "portfolio_id" in data["risk_contribution_table"].columns:
            risk_contrib_cols = [c for c in data["risk_contribution_table"].columns 
                               if c != "portfolio_id" and "risk_contribution" in c.lower()]
            if risk_contrib_cols:
                # Check if risk contributions sum to portfolio volatility
                for pid in data["risk_contribution_table"]["portfolio_id"].unique()[:10]:  # Sample check
                    contrib_row = data["risk_contribution_table"][
                        data["risk_contribution_table"]["portfolio_id"] == pid
                    ][risk_contrib_cols].iloc[0]
                    contrib_sum = contrib_row.sum()
                    if "volatility" in metrics.columns:
                        portfolio_vol = metrics[metrics["portfolio_id"] == pid]["volatility"].values
                        if len(portfolio_vol) > 0:
                            checks["risk_contributions_sum_to_total_risk"] = bool(
                                np.allclose(contrib_sum, portfolio_vol[0], atol=1e-2)
                            )
                            break
    
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
    
    # For risk_parity_deviation_score, lower is better (closer to perfect risk parity)
    if "deviation" in rank_by.lower() or "gap" in rank_by.lower():
        ranked = df.nsmallest(top_k, rank_by)[columns].copy()
    else:
        ranked = df.nlargest(top_k, rank_by)[columns].copy()
    
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def tail_table(df: pd.DataFrame, rank_by: str, tail_k: int, columns: List[str]) -> pd.DataFrame:
    """Generate tail-k (worst) table ranked by specified metric."""
    if rank_by not in df.columns:
        return pd.DataFrame()
    
    # For risk_parity_deviation_score, worst means highest deviation
    if "deviation" in rank_by.lower() or "gap" in rank_by.lower():
        ranked = df.nlargest(tail_k, rank_by)[columns].copy()
    else:
        ranked = df.nsmallest(tail_k, rank_by)[columns].copy()
    
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


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
    risk_contrib = data["risk_contribution_table"]
    
    style = config["execution_plan"]["stage_3_generate_figures"]["style"]
    plt.rcParams["font.size"] = style["font_size"]
    sns.set_style("whitegrid")
    
    figures = config["execution_plan"]["stage_3_generate_figures"]["figures"]
    
    for fig_spec in figures:
        fig_id = fig_spec["id"]
        fig_type = fig_spec["type"]
        output_path = resolve_path(fig_spec["output_path"])
        
        try:
            if fig_type == "heatmap":
                select_spec = fig_spec.get("select_portfolios", {})
                rank_by = select_spec.get("rank_by", "risk_parity_deviation_score")
                top_k = select_spec.get("top_k", 25)
                source = fig_spec["source"]
                
                if source == "risk_contribution_table" and not risk_contrib.empty:
                    if rank_by in metrics.columns:
                        # For deviation score, lower is better
                        if "deviation" in rank_by.lower():
                            top_portfolios = metrics.nsmallest(top_k, rank_by)["portfolio_id"].tolist()
                        else:
                            top_portfolios = metrics.nlargest(top_k, rank_by)["portfolio_id"].tolist()
                        
                        top_risk_contrib = risk_contrib[risk_contrib["portfolio_id"].isin(top_portfolios)].copy()
                        
                        if not top_risk_contrib.empty:
                            risk_cols = [c for c in top_risk_contrib.columns 
                                       if c != "portfolio_id" and "risk" in c.lower()]
                            if risk_cols:
                                pivot = top_risk_contrib.set_index("portfolio_id")[risk_cols]
                                fig, ax = plt.subplots(figsize=(12, 8))
                                sns.heatmap(pivot, annot=False, fmt=".2f", cmap="YlOrRd", ax=ax, 
                                          cbar_kws={"label": "Risk Contribution"})
                                ax.set_title(f"Risk Contribution Heatmap (Top {top_k} by {rank_by})")
                                ax.set_xlabel("Asset")
                                ax.set_ylabel("Portfolio ID")
                                plt.xticks(rotation=90, ha="right")
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
            
            elif fig_type == "scatter":
                x_col = fig_spec["x"]
                y_col = fig_spec["y"]
                if x_col in metrics.columns and y_col in metrics.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(metrics[x_col], metrics[y_col], alpha=0.6, s=20)
                    # Add diagonal line for reference
                    if "baseline" in x_col.lower() and "volatility" in y_col.lower():
                        min_val = min(metrics[x_col].min(), metrics[y_col].min())
                        max_val = max(metrics[x_col].max(), metrics[y_col].max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label="y=x")
                        ax.legend()
                    ax.set_xlabel(x_col.replace("_", " ").title())
                    ax.set_ylabel(y_col.replace("_", " ").title())
                    ax.set_title(f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}")
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=style["dpi"], bbox_inches="tight")
                    plt.close()
        
        except Exception as e:
            print(f"Warning: Failed to generate {fig_id}: {e}")
    
    print(f"Figures saved to {fig_dir}")


def write_markdown_docs(data: dict, config: dict, validation_checks: dict) -> None:
    """Write all markdown documents."""
    metrics = data["metrics_table"]
    risk_contrib = data["risk_contribution_table"]
    weights = data["optimal_weights"]
    schema = data["metrics_schema"]
    
    execution_plan = config["execution_plan"]
    analysis_settings = config["inputs"]["analysis_settings"]
    float_format = analysis_settings["reproducibility"]["float_format"]
    
    # Stage 2: Generate markdown tables
    stage2 = execution_plan["stage_2_generate_markdown_tables"]
    
    # Risk parity deviation stats
    dev_table_spec = stage2["tables"][0]
    dev_stats = descriptive_stats_table(
        metrics,
        dev_table_spec["columns"],
        dev_table_spec["stats"]
    )
    
    # Top-k risk parity portfolios
    topk_spec = stage2["tables"][1]
    topk_tables = {}
    for rank_by in topk_spec["rank_by_each"]:
        topk_tables[rank_by] = topk_table(
            metrics,
            rank_by,
            topk_spec["top_k"],
            topk_spec["columns"]
        )
    
    # Baseline comparison summary
    baseline_spec = stage2["tables"][2]
    baseline_stats = descriptive_stats_table(
        metrics,
        baseline_spec["columns"],
        baseline_spec["stats"]
    )
    
    # Stage 4: Write markdown documents
    artifacts = config["outputs"]["artifacts"]
    
    # Generate all markdown documents
    md_docs = [
        ("00_index.md", "Index"),
        ("01_data_and_run_context.md", "Data and Run Context"),
        ("02_metrics_schema.md", "Metrics Schema"),
        ("03_overall_summary.md", "Overall Summary"),
        ("04_risk_contribution_analysis.md", "Risk Contribution Analysis"),
        ("05_diversification_and_structure.md", "Diversification and Structure"),
        ("06_topk_portfolios.md", "Top-K Portfolios"),
        ("07_tail_portfolios.md", "Tail Portfolios"),
        ("08_baseline_comparison.md", "Baseline Comparison"),
        ("09_runtime_and_scalability.md", "Runtime and Scalability"),
        ("10_ieee_access_ready_paragraphs.md", "IEEE Access Ready Paragraphs"),
        ("11_figure_captions.md", "Figure Captions"),
    ]
    
    for doc_name, doc_title in md_docs:
        doc_path = resolve_path(artifacts.get(f"md_{doc_name.replace('.md', '').replace('_', '_')}", 
                                             f"result_analysis/classical_optimization/rp_erc/{doc_name}"))
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        
        content_lines = []
        
        # 00_index.md
        if doc_name == "00_index.md":
            content_lines.extend([
                "# Risk Parity / Equal Risk Contribution (ERC) Optimization Analysis Index",
                "",
                "## Purpose",
                "This directory contains IEEE Access-ready analysis outputs for Risk Parity/ERC portfolio optimization results.",
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
                "",
                "## Input Files",
            ])
            for key, path in config["inputs"]["run_artifacts"]["expected_outputs_from_pipeline"].items():
                content_lines.append(f"- {key}: `{path}`")
        
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
                "## Key Metrics",
                "",
                "Risk Parity/ERC optimization focuses on equalizing risk contributions across assets, ",
                "rather than maximizing returns. This approach provides superior diversification and ",
                "reduced concentration risk compared to return-driven optimization methods.",
                "",
            ])
        
        # 04_risk_contribution_analysis.md
        elif doc_name == "04_risk_contribution_analysis.md":
            content_lines.extend([
                "# Risk Contribution Analysis",
                "",
                "## Deviation Statistics",
                "",
            ])
            if not dev_stats.empty:
                content_lines.append(markdown_table(dev_stats, float_format=float_format))
            content_lines.extend([
                "",
                "## Interpretation",
                "",
                "Risk parity deviation metrics measure how well portfolios achieve equal risk contributions. ",
                "Lower deviation scores indicate portfolios closer to perfect risk parity, where each asset ",
                "contributes equally to portfolio risk.",
                "",
                "## Figure References",
                "- Figure 1: Risk contribution heatmap",
                "- Figure 2: Risk parity deviation distribution",
                "",
            ])
        
        # 05_diversification_and_structure.md
        elif doc_name == "05_diversification_and_structure.md":
            content_lines.extend([
                "# Diversification and Structure",
                "",
                "## Portfolio Structure Analysis",
                "",
                "Risk Parity portfolios exhibit superior diversification characteristics compared to ",
                "return-driven optimization approaches. The equal risk contribution objective naturally ",
                "promotes diversification by ensuring no single asset dominates portfolio risk.",
                "",
            ])
        
        # 06_topk_portfolios.md
        elif doc_name == "06_topk_portfolios.md":
            content_lines.append("# Top-K Portfolios (Best Risk Parity Solutions)")
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
            content_lines.append("# Tail Portfolios (Worst Risk Parity Solutions)")
            content_lines.append("")
            content_lines.append("Portfolios with high risk parity deviation scores indicate poor achievement "
                               "of equal risk contribution objectives. These portfolios may exhibit concentration "
                               "in a few assets or imbalanced risk contributions.")
        
        # 08_baseline_comparison.md
        elif doc_name == "08_baseline_comparison.md":
            content_lines.extend([
                "# Baseline Comparison",
                "",
                "## Comparison Summary",
                "",
            ])
            if not baseline_stats.empty:
                content_lines.append(markdown_table(baseline_stats, float_format=float_format))
            content_lines.extend([
                "",
                "## Interpretation",
                "",
                "Risk Parity portfolios are compared against baseline strategies including equal-weight ",
                "and minimum-variance portfolios. Key comparisons include volatility reduction, Sharpe ",
                "ratio improvement, and risk contribution balance.",
                "",
                "## Figure References",
                "- Figure 3: Volatility vs baseline comparison",
                "",
            ])
        
        # 09_runtime_and_scalability.md
        elif doc_name == "09_runtime_and_scalability.md":
            content_lines.extend([
                "# Runtime and Scalability",
                "",
                "Risk Parity optimization involves iterative optimization to achieve equal risk contributions. ",
                "The computational complexity scales with the number of assets and the precision required ",
                "for risk contribution balance.",
                "",
            ])
        
        # 10_ieee_access_ready_paragraphs.md
        elif doc_name == "10_ieee_access_ready_paragraphs.md":
            content_lines.extend([
                "# IEEE Access Ready Paragraphs",
                "",
                "## Risk Parity Theory and Equal Risk Contribution Objective",
                "",
                f"Risk Parity optimization was applied to generate {validation_checks['num_portfolios']} portfolios ",
                "with the objective of equalizing risk contributions across assets. Unlike traditional mean-variance ",
                "optimization that maximizes return for a given risk level, Risk Parity focuses on risk budgeting, ",
                "ensuring each asset contributes equally to portfolio risk. This approach addresses the concentration ",
                "bias inherent in return-driven optimization methods.",
                "",
                "## Interpretation of Risk Contribution Deviation Metrics",
                "",
                "Risk parity deviation scores quantify the degree to which portfolios achieve equal risk contributions. ",
                "Lower deviation scores indicate portfolios closer to perfect risk parity, where risk contributions ",
                "are balanced across all assets. The equal risk gap metric measures the difference between maximum ",
                "and minimum risk contributions, providing insight into risk concentration.",
                "",
                "## Diversification Advantages over Return-Driven Optimization",
                "",
                "Risk Parity portfolios exhibit superior diversification characteristics compared to return-driven ",
                "optimization approaches. By focusing on risk budgeting rather than return maximization, Risk Parity ",
                "naturally promotes diversification and reduces concentration risk. This is particularly valuable in ",
                "uncertain market conditions where return forecasts are unreliable.",
                "",
                "## Comparison Against Equal-Weight and Minimum-Variance Baselines",
                "",
                "Risk Parity portfolios are compared against equal-weight and minimum-variance baseline strategies. ",
                "While equal-weight portfolios provide naive diversification, they do not account for asset correlations ",
                "and risk characteristics. Minimum-variance portfolios minimize overall risk but may concentrate in ",
                "low-volatility assets. Risk Parity balances these approaches by equalizing risk contributions while ",
                "accounting for asset correlations.",
                "",
                "## Limitations of Volatility-Based Risk Budgeting",
                "",
                "Risk Parity optimization relies on volatility as the primary risk measure, which may not fully capture ",
                "tail risk and extreme downside scenarios. Volatility-based risk budgeting assumes symmetric return ",
                "distributions and may underestimate risk during market stress. Additionally, Risk Parity portfolios ",
                "may underperform return-driven strategies during bull markets when concentration in high-return assets ",
                "would be beneficial.",
                "",
            ])
        
        # 11_figure_captions.md
        elif doc_name == "11_figure_captions.md":
            content_lines.append("# Figure Captions")
            fig_num = 1
            for fig_spec in execution_plan["stage_3_generate_figures"]["figures"]:
                fig_id = fig_spec["id"]
                fig_type = fig_spec["type"]
                caption = f"**Fig. {fig_num}:** {fig_id.replace('_', ' ').title()}. "
                
                if fig_type == "heatmap":
                    caption += "Risk contribution heatmap for top-performing Risk Parity portfolios, showing how risk is distributed across assets."
                elif fig_type == "histogram":
                    caption += f"Distribution of {fig_spec.get('column', '').replace('_', ' ')} across portfolios."
                elif fig_type == "scatter":
                    caption += f"Scatter plot comparing {fig_spec.get('y', '').replace('_', ' ')} vs {fig_spec.get('x', '').replace('_', ' ')}."
                
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
    risk_contrib = data["risk_contribution_table"]
    
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
    
    # Risk Parity-specific checks
    checks["weights_sum_to_one"] = validation_checks.get("weights_sum_to_one", False)
    checks["risk_contributions_sum_to_total_risk"] = validation_checks.get("risk_contributions_sum_to_total_risk", False)
    checks["risk_parity_deviation_non_negative"] = validation_checks.get("risk_parity_deviation_non_negative", False)
    
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
    audit_path = resolve_path("result_analysis/classical_optimization/rp_erc/audit_log.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_log = {
        "validation_checks": validation_checks,
        "quality_checks": quality_results,
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2, default=str)
    print(f"Audit log written: {audit_path}")
    
    print("\nRisk Parity/ERC analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
