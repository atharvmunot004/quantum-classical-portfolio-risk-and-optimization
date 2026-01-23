"""Markdown report generation for GARCH VaR/CVaR evaluation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def generate_report(
    metrics_df: pd.DataFrame,
    time_sliced_metrics_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Union[str, Path]] = None,
    garch_settings: Optional[Dict] = None,
    report_sections: Optional[List[str]] = None
) -> str:
    """Generate a concise IEEE-friendly markdown report."""
    if report_sections is None:
        report_sections = [
            "model_configuration",
            "batch_execution_summary",
            "aggregate_backtesting_results",
            "tail_behavior_summary",
            "garch_specification_and_estimation",
            "robustness_and_normality_checks",
            "runtime_statistics",
            "time_sliced_backtesting"
        ]

    lines: List[str] = []
    lines.append("# GARCH(1,1) VaR/CVaR Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if "model_configuration" in report_sections:
        lines.append("## Model Configuration")
        lines.append("")
        lines.append("GARCH(1,1):  σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁")
        lines.append("")
        if garch_settings:
            lines.append(f"- Distribution: {garch_settings.get('distribution', 't')}")
            lines.append(f"- Estimation windows: {garch_settings.get('estimation_windows', [])}")
            lines.append(f"- Confidence levels: {garch_settings.get('confidence_levels', [])}")
            lines.append(f"- Horizons: {garch_settings.get('horizons', {})}")
        lines.append("")

    if "batch_execution_summary" in report_sections:
        lines.append("## Evaluation Summary")
        lines.append("")
        lines.append(f"- Total configurations: {len(metrics_df)}")
        if 'asset' in metrics_df.columns:
            lines.append(f"- Unique assets: {metrics_df['asset'].nunique()}")
        lines.append("")

    if "aggregate_backtesting_results" in report_sections and len(metrics_df) > 0:
        lines.append("## Backtesting Results")
        lines.append("")
        if 'hit_rate' in metrics_df.columns and 'confidence_level' in metrics_df.columns:
            lines.append("### Hit Rate vs Expected")
            lines.append("")
            lines.append("| Confidence | Avg Hit Rate | Expected | Diff |")
            lines.append("|---:|---:|---:|---:|")
            for cl in sorted(metrics_df['confidence_level'].dropna().unique()):
                sub = metrics_df[metrics_df['confidence_level'] == cl]
                avg_hr = float(sub['hit_rate'].mean())
                exp = float(1.0 - cl)
                lines.append(f"| {cl:.0%} | {avg_hr:.4f} | {exp:.4f} | {avg_hr-exp:+.4f} |")
            lines.append("")

        if 'kupiec_unconditional_coverage' in metrics_df.columns:
            passed = int((metrics_df['kupiec_unconditional_coverage'] > 0.05).sum())
            lines.append(f"- Kupiec UC passed: {passed}/{len(metrics_df)} ({passed/len(metrics_df)*100:.1f}%)")

        if 'christoffersen_conditional_coverage' in metrics_df.columns:
            passed = int((metrics_df['christoffersen_conditional_coverage'] > 0.05).sum())
            lines.append(f"- Christoffersen CC passed: {passed}/{len(metrics_df)} ({passed/len(metrics_df)*100:.1f}%)")

        if 'traffic_light_zone' in metrics_df.columns:
            lines.append("")
            lines.append("### Traffic Light Zones")
            vc = metrics_df['traffic_light_zone'].value_counts(dropna=False)
            for k, v in vc.items():
                lines.append(f"- {k}: {int(v)}")
            lines.append("")

    if "tail_behavior_summary" in report_sections and len(metrics_df) > 0:
        lines.append("## Tail Risk Summary")
        lines.append("")
        for col in ['quantile_loss_score', 'rmse_var_vs_losses', 'rmse_cvar_vs_losses']:
            if col in metrics_df.columns:
                lines.append(f"- {col}: mean={metrics_df[col].mean():.6f}, median={metrics_df[col].median():.6f}")
        lines.append("")

    if "garch_specification_and_estimation" in report_sections and len(metrics_df) > 0:
        lines.append("## GARCH Parameter Diagnostics")
        lines.append("")
        if 'alpha_plus_beta' in metrics_df.columns:
            apb = metrics_df['alpha_plus_beta']
            lines.append(f"- α+β: mean={apb.mean():.4f}, median={apb.median():.4f}, min={apb.min():.4f}, max={apb.max():.4f}")
        if 'persistence_half_life' in metrics_df.columns:
            lines.append(f"- Half-life (days): mean={metrics_df['persistence_half_life'].mean():.2f}")
        if 'long_run_volatility' in metrics_df.columns:
            lines.append(f"- Long-run volatility: mean={metrics_df['long_run_volatility'].mean():.6f}")
        lines.append("")

    if "robustness_and_normality_checks" in report_sections and len(metrics_df) > 0:
        lines.append("## Normality Diagnostics")
        lines.append("")
        for col in ['skewness', 'kurtosis', 'jarque_bera_p_value']:
            if col in metrics_df.columns:
                lines.append(f"- {col}: mean={metrics_df[col].mean():.4f}")
        lines.append("")

    if "time_sliced_backtesting" in report_sections and time_sliced_metrics_df is not None and len(time_sliced_metrics_df) > 0:
        lines.append("## Time-Sliced Backtesting")
        lines.append("")
        if 'slice_type' in time_sliced_metrics_df.columns:
            for st in sorted(time_sliced_metrics_df['slice_type'].unique()):
                lines.append(f"### {st}")
                sub = time_sliced_metrics_df[time_sliced_metrics_df['slice_type'] == st]
                if 'hit_rate' in sub.columns:
                    lines.append(f"- Avg hit rate: {sub['hit_rate'].mean():.4f}")
                if 'violation_ratio' in sub.columns:
                    lines.append(f"- Avg violation ratio: {sub['violation_ratio'].mean():.4f}")
                lines.append("")

    # Always include a numeric summary block
    lines.append("## Detailed Numeric Summary")
    lines.append("")
    num = metrics_df.select_dtypes(include=[np.number])
    if len(num.columns) > 0 and len(num) > 0:
        lines.append("```")
        lines.append(num.describe().to_string())
        lines.append("```")
    else:
        lines.append("No numeric data available.")
    lines.append("")

    report = "\n".join(lines)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report, encoding='utf-8')

    return report
