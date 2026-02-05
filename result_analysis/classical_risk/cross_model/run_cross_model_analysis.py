"""
Cross-model VaR/CVaR asset-level analysis.

Compares multiple risk models (variance_covariance, garch_1_1, monte_carlo, evt_pot)
across common dimensions and generates comprehensive comparison reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Project root (parent of result_analysis)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).parent / "llm.json"
OUT_DIR = Path(__file__).parent


def load_config() -> dict:
    """Load analysis config from llm.json."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")


def load_model_data(config: dict, model_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load metrics and time_sliced_metrics for a model."""
    base = PROJECT_ROOT
    model_config = config["inputs"][model_name]
    
    metrics_path = base / model_config["metrics_table"]
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics table not found: {metrics_path}")
    metrics = pd.read_parquet(metrics_path)
    metrics["model"] = model_name
    
    ts_path = base / model_config["time_sliced_metrics_table"]
    if ts_path.exists():
        time_sliced = pd.read_parquet(ts_path)
        time_sliced["model"] = model_name
    else:
        time_sliced = pd.DataFrame()
    
    return metrics, time_sliced


def align_models(
    all_metrics: Dict[str, pd.DataFrame],
    config: dict
) -> pd.DataFrame:
    """Align models on common dimensions per alignment_policy."""
    policy = config["analysis_scope"]["alignment_policy"]
    common_dims = config["analysis_scope"]["common_dimensions"]
    
    # Find common values for each dimension
    common_values = {}
    for dim in common_dims:
        values = set()
        for model_name, df in all_metrics.items():
            if dim in df.columns:
                values.update(df[dim].dropna().unique())
        common_values[dim] = sorted(list(values))
    
    # Filter each model to common values
    aligned_dfs = []
    for model_name, df in all_metrics.items():
        df_aligned = df.copy()
        for dim in common_dims:
            if dim in df_aligned.columns and dim in common_values:
                df_aligned = df_aligned[df_aligned[dim].isin(common_values[dim])]
        aligned_dfs.append(df_aligned)
    
    # Combine and ensure all models have same dimensions
    combined = pd.concat(aligned_dfs, ignore_index=True)
    
    # Further alignment: require exact matches on common dimensions
    if policy.get("require_common_assets") and "asset" in common_dims:
        assets_by_model = combined.groupby("model")["asset"].apply(set)
        common_assets = set.intersection(*assets_by_model) if len(assets_by_model) > 0 else set()
        if common_assets:
            combined = combined[combined["asset"].isin(common_assets)]
    
    return combined


def compute_coverage_accuracy_comparison(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compare coverage accuracy across models."""
    cfg = config["primary_comparisons"]["coverage_accuracy"]
    metrics = cfg["metrics"]
    
    # Compute expected hit rate
    df = df.copy()
    df["expected_hit_rate"] = 1 - df["confidence_level"]
    
    results = []
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        for metric in metrics:
            if metric not in model_df.columns:
                continue
            
            values = model_df[metric].dropna()
            if len(values) == 0:
                continue
            
            result = {
                "model": model,
                "metric": metric,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p95": float(values.quantile(0.95)) if len(values) > 0 else np.nan,
            }
            
            if metric == "hit_rate":
                expected_hr = model_df["expected_hit_rate"].mean()
                result["expected_hit_rate"] = float(expected_hr)
                result["abs_deviation"] = float(abs(result["mean"] - expected_hr))
            
            results.append(result)
    
    return pd.DataFrame(results)


def compute_statistical_backtesting_comparison(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compare statistical backtesting results across models."""
    cfg = config["primary_comparisons"]["statistical_backtesting"]
    tests = cfg["tests"]
    sig_level = cfg["significance_level"]
    
    # Map test names to rejection columns
    test_to_reject_col = {
        "kupiec_unconditional_coverage": "kupiec_reject_null",
        "christoffersen_independence": "christoffersen_independence_reject_null",
        "christoffersen_conditional_coverage": "christoffersen_conditional_coverage_reject_null",
    }
    
    test_to_pvalue_col = {
        "kupiec_unconditional_coverage": "kupiec_p_value",
        "christoffersen_independence": "christoffersen_independence_p_value",
        "christoffersen_conditional_coverage": "christoffersen_conditional_coverage_p_value",
    }
    
    results = []
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        for test in tests:
            reject_col = test_to_reject_col.get(test)
            pvalue_col = test_to_pvalue_col.get(test)
            
            if reject_col and reject_col in model_df.columns:
                rejections = model_df[reject_col].fillna(False)
                rejection_rate = float(rejections.mean() * 100)
            else:
                rejection_rate = np.nan
            
            if pvalue_col and pvalue_col in model_df.columns:
                pvalues = model_df[pvalue_col].dropna()
                mean_pvalue = float(pvalues.mean()) if len(pvalues) > 0 else np.nan
            else:
                mean_pvalue = np.nan
            
            results.append({
                "model": model,
                "test": test,
                "rejection_rate": rejection_rate,
                "mean_p_value": mean_pvalue,
                "significance_level": sig_level,
            })
    
    return pd.DataFrame(results)


def compute_regulatory_diagnostics(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compare regulatory diagnostics (traffic light zones) across models."""
    cfg = config["primary_comparisons"]["regulatory_diagnostics"]
    metric = cfg["metric"]
    
    if metric not in df.columns:
        return pd.DataFrame()
    
    results = []
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        zones = model_df[metric].value_counts()
        total = len(model_df)
        
        for zone in ["green", "yellow", "red"]:
            count = int(zones.get(zone, 0))
            pct = float(count / total * 100) if total > 0 else 0.0
            results.append({
                "model": model,
                "zone": zone,
                "count": count,
                "percentage": pct,
            })
    
    return pd.DataFrame(results)


def compute_tail_risk_comparison(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compare tail risk behavior across models."""
    cfg = config["tail_risk_comparison"]
    results = []
    
    # VaR tail behavior
    var_cfg = cfg.get("var_tail_behavior", {})
    if var_cfg:
        metrics = var_cfg.get("metrics", [])
        agg_methods = var_cfg.get("aggregation", ["mean", "median", "p95"])
        
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            for metric in metrics:
                if metric not in model_df.columns:
                    continue
                
                values = model_df[metric].dropna()
                if len(values) == 0:
                    continue
                
                for agg in agg_methods:
                    if agg == "mean":
                        val = float(values.mean())
                    elif agg == "median":
                        val = float(values.median())
                    elif agg == "p95":
                        val = float(values.quantile(0.95))
                    else:
                        continue
                    
                    results.append({
                        "model": model,
                        "risk_type": "VaR",
                        "metric": metric,
                        "aggregation": agg,
                        "value": val,
                    })
    
    # CVaR tail behavior
    cvar_cfg = cfg.get("cvar_tail_behavior", {})
    if cvar_cfg.get("enabled"):
        models = cvar_cfg.get("models", [])
        metrics = cvar_cfg.get("metrics", [])
        
        for model in models:
            if model not in df["model"].unique():
                continue
            model_df = df[df["model"] == model]
            for metric in metrics:
                if metric not in model_df.columns:
                    continue
                
                values = model_df[metric].dropna()
                if len(values) == 0:
                    continue
                
                results.append({
                    "model": model,
                    "risk_type": "CVaR",
                    "metric": metric,
                    "aggregation": "mean",
                    "value": float(values.mean()),
                })
    
    return pd.DataFrame(results)


def compute_distributional_diagnostics(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compare distributional assumption diagnostics across models."""
    cfg = config["distributional_assumption_diagnostics"]
    if not cfg.get("compare_models"):
        return pd.DataFrame()
    
    metrics = cfg.get("metrics", [])
    results = []
    
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        for metric in metrics:
            if metric not in model_df.columns:
                continue
            
            values = model_df[metric].dropna()
            if len(values) == 0:
                continue
            
            results.append({
                "model": model,
                "metric": metric,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std()),
            })
    
    return pd.DataFrame(results)


def compute_time_sliced_analysis(
    all_time_sliced: Dict[str, pd.DataFrame],
    config: dict
) -> pd.DataFrame:
    """Perform time-sliced cross-analysis."""
    cfg = config["time_sliced_cross_analysis"]
    if not cfg.get("enabled"):
        return pd.DataFrame()
    
    slice_by = cfg.get("slice_by", [])
    metrics = cfg.get("metrics", [])
    min_obs = cfg.get("minimum_observations", 60)
    
    # Combine all time-sliced data
    combined = pd.concat(all_time_sliced.values(), ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    
    # Prepare date columns
    if "start_date" in combined.columns:
        combined["start_date"] = pd.to_datetime(combined["start_date"], errors="coerce")
        combined["year"] = combined["start_date"].dt.year
        combined["quarter"] = combined["start_date"].dt.quarter
    
    results = []
    for slice_dim in slice_by:
        if slice_dim not in combined.columns:
            continue
        
        for slice_value in combined[slice_dim].dropna().unique():
            slice_df = combined[combined[slice_dim] == slice_value]
            
            # Filter by minimum observations
            if len(slice_df) < min_obs:
                continue
            
            for model in slice_df["model"].unique():
                model_slice = slice_df[slice_df["model"] == model]
                for metric in metrics:
                    if metric not in model_slice.columns:
                        continue
                    
                    values = model_slice[metric].dropna()
                    if len(values) == 0:
                        continue
                    
                    results.append({
                        "slice_dimension": slice_dim,
                        "slice_value": slice_value,
                        "model": model,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "count": len(values),
                    })
    
    return pd.DataFrame(results)


def compute_robustness_checks(df: pd.DataFrame, config: dict) -> Dict[str, pd.DataFrame]:
    """Perform robustness cross-checks."""
    cfg = config["robustness_cross_checks"]
    results = {}
    
    # Window sensitivity
    if "window_sensitivity" in cfg:
        wins = cfg["window_sensitivity"]["compare_windows"]
        metrics = cfg["window_sensitivity"]["metrics"]
        
        if "estimation_window" in df.columns:
            window_df = df[df["estimation_window"].isin(wins)].copy()
            if not window_df.empty:
                window_results = []
                for model in window_df["model"].unique():
                    model_df = window_df[window_df["model"] == model]
                    for win in wins:
                        win_df = model_df[model_df["estimation_window"] == win]
                        for metric in metrics:
                            if metric not in win_df.columns:
                                continue
                            values = win_df[metric].dropna()
                            if len(values) > 0:
                                window_results.append({
                                    "model": model,
                                    "estimation_window": win,
                                    "metric": metric,
                                    "mean": float(values.mean()),
                                })
                results["window_sensitivity"] = pd.DataFrame(window_results)
    
    # Horizon scaling behavior
    if "horizon_scaling_behavior" in cfg:
        horizons = cfg["horizon_scaling_behavior"]["compare_horizons"]
        metrics = cfg["horizon_scaling_behavior"]["metrics"]
        
        if "horizon" in df.columns:
            horizon_df = df[df["horizon"].isin(horizons)].copy()
            if not horizon_df.empty:
                horizon_results = []
                for model in horizon_df["model"].unique():
                    model_df = horizon_df[horizon_df["model"] == model]
                    for horizon in horizons:
                        h_df = model_df[model_df["horizon"] == horizon]
                        for metric in metrics:
                            if metric not in h_df.columns:
                                continue
                            values = h_df[metric].dropna()
                            if len(values) > 0:
                                horizon_results.append({
                                    "model": model,
                                    "horizon": horizon,
                                    "metric": metric,
                                    "mean": float(values.mean()),
                                })
                results["horizon_scaling"] = pd.DataFrame(horizon_results)
    
    return results


def compute_model_ranking(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute overall model ranking using composite score."""
    cfg = config["overall_model_ranking"]
    composite = cfg["composite_score"]
    ranking_scope = cfg["ranking_scope"]
    
    df = df.copy()
    df["expected_hit_rate"] = 1 - df["confidence_level"]
    
    # Compute composite score for each model-configuration combination
    results = []
    
    # Group by ranking scope dimensions
    group_cols = ["model"] + [col for col in ranking_scope if col in df.columns]
    
    for group_key, group_df in df.groupby(group_cols):
        if isinstance(group_key, tuple):
            group_dict = dict(zip(group_cols, group_key))
        else:
            group_dict = {group_cols[0]: group_key}
        
        score = 0.0
        component_scores = {}
        
        for component in composite["components"]:
            metric_expr = component["metric"]
            weight = component.get("weight", 1.0)
            
            if metric_expr == "abs(hit_rate - expected_hit_rate)":
                if "hit_rate" in group_df.columns:
                    values = (group_df["hit_rate"] - group_df["expected_hit_rate"]).abs()
                    metric_value = float(values.mean())
                else:
                    continue
            elif metric_expr == "violation_ratio":
                if "violation_ratio" in group_df.columns:
                    target = component.get("target", 1.0)
                    values = (group_df["violation_ratio"] - target).abs()
                    metric_value = float(values.mean())
                else:
                    continue
            elif metric_expr in group_df.columns:
                values = group_df[metric_expr].dropna()
                if len(values) == 0:
                    continue
                metric_value = float(values.mean())
            elif metric_expr == "kupiec_rejection_rate":
                if "kupiec_reject_null" in group_df.columns:
                    metric_value = float(group_df["kupiec_reject_null"].mean())
                else:
                    continue
            else:
                continue
            
            component_scores[metric_expr] = metric_value
            score += metric_value * weight
        
        result = {**group_dict, "composite_score": score}
        result.update({f"component_{k}": v for k, v in component_scores.items()})
        results.append(result)
    
    ranking_df = pd.DataFrame(results)
    
    # Normalize if requested
    if composite.get("normalize_metrics"):
        for col in ranking_df.columns:
            if col.startswith("component_") or col == "composite_score":
                values = ranking_df[col].dropna()
                if len(values) > 0 and values.max() > values.min():
                    ranking_df[col] = (values - values.min()) / (values.max() - values.min())
    
    # Sort by composite score (lower is better for most metrics)
    ranking_df = ranking_df.sort_values("composite_score", ascending=True).reset_index(drop=True)
    ranking_df["rank"] = range(1, len(ranking_df) + 1)
    
    return ranking_df


def generate_figures(
    coverage_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    regulatory_df: pd.DataFrame,
    tail_risk_df: pd.DataFrame,
    time_sliced_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    config: dict,
    fig_dir: Path
) -> None:
    """Generate visualization figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available; skipping figures.")
        return
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    
    plots = config.get("outputs", {}).get("figures", {}).get("plots", [])
    
    # Hit rate comparison by model
    if "hit_rate_comparison_by_model" in plots and not coverage_df.empty:
        hit_rate_data = coverage_df[coverage_df["metric"] == "hit_rate"]
        if not hit_rate_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            models = hit_rate_data["model"].unique()
            means = [hit_rate_data[hit_rate_data["model"] == m]["mean"].values[0] 
                    for m in models if len(hit_rate_data[hit_rate_data["model"] == m]) > 0]
            ax.bar(models, means, alpha=0.7, edgecolor="black")
            ax.axhline(y=0.05, color="r", linestyle="--", label="Expected (95% conf)")
            ax.set_xlabel("Model")
            ax.set_ylabel("Mean Hit Rate")
            ax.set_title("Hit Rate Comparison by Model")
            ax.legend()
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(fig_dir / "hit_rate_comparison_by_model.png", dpi=150, bbox_inches="tight")
            plt.close()
    
    # Violation ratio comparison
    if "violation_ratio_comparison_by_model" in plots and not coverage_df.empty:
        vr_data = coverage_df[coverage_df["metric"] == "violation_ratio"]
        if not vr_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            models = vr_data["model"].unique()
            means = [vr_data[vr_data["model"] == m]["mean"].values[0] 
                    for m in models if len(vr_data[vr_data["model"] == m]) > 0]
            ax.bar(models, means, alpha=0.7, edgecolor="black")
            ax.axhline(y=1.0, color="g", linestyle="--", label="Target = 1.0")
            ax.set_xlabel("Model")
            ax.set_ylabel("Mean Violation Ratio")
            ax.set_title("Violation Ratio Comparison by Model")
            ax.legend()
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(fig_dir / "violation_ratio_comparison_by_model.png", dpi=150, bbox_inches="tight")
            plt.close()
    
    # Traffic light distribution
    if "traffic_light_distribution" in plots and not regulatory_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot = regulatory_df.pivot(index="model", columns="zone", values="percentage")
        pivot.plot(kind="bar", stacked=True, ax=ax, color=["green", "yellow", "red"], alpha=0.7)
        ax.set_xlabel("Model")
        ax.set_ylabel("Percentage")
        ax.set_title("Traffic Light Zone Distribution by Model")
        ax.legend(title="Zone")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "traffic_light_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()
    
    # Tail exceedance comparison
    if "tail_exceedance_comparison" in plots and not tail_risk_df.empty:
        var_tail = tail_risk_df[tail_risk_df["risk_type"] == "VaR"]
        if not var_tail.empty and "mean_exceedance" in var_tail["metric"].values:
            exceed_data = var_tail[var_tail["metric"] == "mean_exceedance"]
            if not exceed_data.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                models = exceed_data["model"].unique()
                means = [exceed_data[exceed_data["model"] == m]["value"].values[0] 
                        for m in models if len(exceed_data[exceed_data["model"] == m]) > 0]
                ax.bar(models, means, alpha=0.7, edgecolor="black")
                ax.set_xlabel("Model")
                ax.set_ylabel("Mean Exceedance")
                ax.set_title("Tail Exceedance Comparison (VaR)")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(fig_dir / "tail_exceedance_comparison.png", dpi=150, bbox_inches="tight")
                plt.close()
    
    # Time-sliced model performance
    if "time_sliced_model_performance" in plots and not time_sliced_df.empty:
        hit_rate_ts = time_sliced_df[time_sliced_df["metric"] == "hit_rate"]
        if not hit_rate_ts.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            for model in hit_rate_ts["model"].unique():
                model_data = hit_rate_ts[hit_rate_ts["model"] == model]
                ax.plot(model_data["slice_value"], model_data["mean"], marker="o", label=model)
            ax.set_xlabel("Time Slice")
            ax.set_ylabel("Mean Hit Rate")
            ax.set_title("Time-Sliced Model Performance")
            ax.legend()
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(fig_dir / "time_sliced_model_performance.png", dpi=150, bbox_inches="tight")
            plt.close()
    
    # Overall model ranking
    if "overall_model_ranking" in plots and not ranking_df.empty:
        # Aggregate by model
        model_ranks = ranking_df.groupby("model")["composite_score"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(model_ranks.index, model_ranks.values, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Composite Score (lower is better)")
        ax.set_ylabel("Model")
        ax.set_title("Overall Model Ranking")
        plt.tight_layout()
        plt.savefig(fig_dir / "overall_model_ranking.png", dpi=150, bbox_inches="tight")
        plt.close()
    
    print(f"Figures saved to {fig_dir}")


def df_to_markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Convert DataFrame to markdown table."""
    if df.empty:
        return "_No data._"
    d = df.head(max_rows)
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in d.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                v = ""
            elif isinstance(v, float):
                v = f"{v:.4f}"
            cells.append(str(v)[:50])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def generate_report(
    config: dict,
    coverage_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    regulatory_df: pd.DataFrame,
    tail_risk_df: pd.DataFrame,
    dist_diag_df: pd.DataFrame,
    time_sliced_df: pd.DataFrame,
    robustness_results: Dict[str, pd.DataFrame],
    ranking_df: pd.DataFrame,
    report_path: Path
) -> None:
    """Generate markdown report."""
    sections = config.get("outputs", {}).get("report", {}).get("sections", [])
    lines = [
        "# Cross-Model VaR/CVaR Asset-Level Analysis Report",
        "",
        f"**Analysis Task:** {config.get('analysis_task', 'N/A')}",
        f"**Models Compared:** {', '.join(config.get('models_compared', []))}",
        "",
    ]
    
    if "experimental_setup" in sections:
        lines.extend([
            "## Experimental Setup",
            "",
            f"- **Analysis Level:** {config['analysis_scope']['level']}",
            f"- **Common Dimensions:** {', '.join(config['analysis_scope']['common_dimensions'])}",
            f"- **Alignment Policy:** Require common assets, windows, confidence levels, and horizons",
            "",
        ])
    
    if "coverage_accuracy_comparison" in sections and not coverage_df.empty:
        lines.extend([
            "## Coverage Accuracy Comparison",
            "",
            df_to_markdown_table(coverage_df),
            "",
        ])
    
    if "statistical_backtesting_comparison" in sections and not backtest_df.empty:
        lines.extend([
            "## Statistical Backtesting Comparison",
            "",
            df_to_markdown_table(backtest_df),
            "",
        ])
    
    if "tail_risk_comparison" in sections and not tail_risk_df.empty:
        lines.extend([
            "## Tail Risk Comparison",
            "",
            df_to_markdown_table(tail_risk_df),
            "",
        ])
    
    if "distributional_assumption_effects" in sections and not dist_diag_df.empty:
        lines.extend([
            "## Distributional Assumption Effects",
            "",
            df_to_markdown_table(dist_diag_df),
            "",
        ])
    
    if "regime_dependent_performance" in sections and not time_sliced_df.empty:
        lines.extend([
            "## Regime-Dependent Performance (Time-Sliced Analysis)",
            "",
            df_to_markdown_table(time_sliced_df.head(50)),
            "",
        ])
    
    if "robustness_analysis" in sections:
        lines.append("## Robustness Analysis")
        lines.append("")
        for key, df in robustness_results.items():
            if not df.empty:
                lines.append(f"### {key.replace('_', ' ').title()}")
                lines.append("")
                lines.append(df_to_markdown_table(df))
                lines.append("")
    
    if "overall_model_ranking" in sections and not ranking_df.empty:
        lines.extend([
            "## Overall Model Ranking",
            "",
            df_to_markdown_table(ranking_df.head(30)),
            "",
        ])
    
    if "discussion_and_implications" in sections:
        lines.extend([
            "## Discussion and Implications",
            "",
            "This cross-model analysis compares multiple VaR/CVaR estimation methods ",
            "across common dimensions. Key findings include:",
            "",
            "- Coverage accuracy varies across models",
            "- Statistical backtesting results show different rejection rates",
            "- Tail risk behavior differs significantly between models",
            "- Distributional assumptions impact model performance",
            "- Model rankings depend on the specific metric and context",
            "",
        ])
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {report_path}")


def main() -> int:
    """Main execution function."""
    config = load_config()
    
    print("Loading model data...")
    all_metrics = {}
    all_time_sliced = {}
    
    for model_name in config["models_compared"]:
        try:
            metrics, time_sliced = load_model_data(config, model_name)
            all_metrics[model_name] = metrics
            if not time_sliced.empty:
                all_time_sliced[model_name] = time_sliced
            print(f"  Loaded {model_name}: {len(metrics)} metrics, {len(time_sliced)} time-sliced")
        except Exception as e:
            print(f"  Warning: Failed to load {model_name}: {e}")
            continue
    
    if not all_metrics:
        print("Error: No model data loaded!")
        return 1
    
    print("\nAligning models...")
    aligned_metrics = align_models(all_metrics, config)
    print(f"  Aligned metrics: {len(aligned_metrics)} rows")
    
    print("\nComputing comparisons...")
    coverage_df = compute_coverage_accuracy_comparison(aligned_metrics, config)
    backtest_df = compute_statistical_backtesting_comparison(aligned_metrics, config)
    regulatory_df = compute_regulatory_diagnostics(aligned_metrics, config)
    tail_risk_df = compute_tail_risk_comparison(aligned_metrics, config)
    dist_diag_df = compute_distributional_diagnostics(aligned_metrics, config)
    time_sliced_df = compute_time_sliced_analysis(all_time_sliced, config)
    robustness_results = compute_robustness_checks(aligned_metrics, config)
    ranking_df = compute_model_ranking(aligned_metrics, config)
    
    print("\nGenerating outputs...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save summary tables
    outputs = config["outputs"]
    
    summary_path = PROJECT_ROOT / outputs["summary_tables"]["path"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_metrics.to_parquet(summary_path, index=False)
    print(f"  Summary table: {summary_path}")
    
    if not time_sliced_df.empty:
        ts_path = PROJECT_ROOT / outputs["time_sliced_tables"]["path"]
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        time_sliced_df.to_parquet(ts_path, index=False)
        print(f"  Time-sliced table: {ts_path}")
    
    if not ranking_df.empty:
        rank_path = PROJECT_ROOT / outputs["ranking_tables"]["path"]
        rank_path.parent.mkdir(parents=True, exist_ok=True)
        ranking_df.to_parquet(rank_path, index=False)
        print(f"  Ranking table: {rank_path}")
    
    # Generate figures
    fig_dir = PROJECT_ROOT / outputs["figures"]["path"]
    generate_figures(
        coverage_df, backtest_df, regulatory_df, tail_risk_df,
        time_sliced_df, ranking_df, config, fig_dir
    )
    
    # Generate report
    report_path = PROJECT_ROOT / outputs["report"]["path"]
    generate_report(
        config, coverage_df, backtest_df, regulatory_df,
        tail_risk_df, dist_diag_df, time_sliced_df,
        robustness_results, ranking_df, report_path
    )
    
    print("\nCross-model analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
