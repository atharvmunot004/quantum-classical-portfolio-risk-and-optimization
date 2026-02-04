"""
Report generation for QAOA CVaR asset-level evaluation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


def generate_report(
    metrics_df: pd.DataFrame,
    output_path: Path,
    quantum_settings: Optional[Dict] = None,
    report_sections: Optional[List[str]] = None,
    runtime_metrics: Optional[Dict] = None,
):
    sections = report_sections or [
        "quantum_methodology_overview",
        "qaoa_cost_and_mixer_construction",
        "cvar_based_sampling_and_objective",
        "optimization_convergence_and_stability",
        "backtesting_and_tail_diagnostics",
        "time_sliced_tail_behavior",
        "quantum_vs_classical_runtime_comparison",
        "key_insights",
    ]
    quantum_settings = quantum_settings or {}
    runtime_metrics = runtime_metrics or {}
    lines = []

    lines.append("# QAOA CVaR Asset-Level Risk Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if "quantum_methodology_overview" in sections:
        lines.append("## Quantum Methodology Overview")
        lines.append("")
        lines.append("QAOA is used as a quantum optimization routine for asset-level tail-risk scoring.")
        lines.append("A discretized tail-loss objective is encoded as a cost Hamiltonian and optimized")
        lines.append("using CVaR-based sampling of measurement outcomes (mean of worst alpha fraction).")
        lines.append("")

    if "qaoa_cost_and_mixer_construction" in sections:
        lines.append("## QAOA Cost and Mixer Construction")
        lines.append("")
        lines.append(f"- **Cost Hamiltonian:** Ising form, built from discretized loss objective (quantile grid)")
        lines.append(f"- **State qubits:** {quantum_settings.get('problem_encoding', {}).get('num_state_qubits', 8)}")
        lines.append("- **Mixer:** Standard transverse-field X mixer for unconstrained binary search")
        lines.append("")

    if "cvar_based_sampling_and_objective" in sections:
        lines.append("## CVaR-Based Sampling and Objective")
        lines.append("")
        lines.append("Objective = mean of worst (1-alpha) fraction of sampled loss outcomes from QAOA measurements.")
        lines.append("This CVaR objective is evaluated during optimization; the final CVaR estimate")
        lines.append("is the tail-average of the optimized state's measurement distribution.")
        lines.append("")

    if "backtesting_and_tail_diagnostics" in sections:
        lines.append("## Backtesting and Tail Diagnostics")
        lines.append("")
        if 'exceedance_rate_vs_cvar' in metrics_df.columns:
            lines.append(f"- **Mean exceedance rate vs CVaR:** {metrics_df['exceedance_rate_vs_cvar'].mean():.4f}")
        if 'num_exceedances_vs_cvar' in metrics_df.columns:
            lines.append(f"- **Total exceedances:** {int(metrics_df['num_exceedances_vs_cvar'].sum())}")
        if 'traffic_light_zone' in metrics_df.columns:
            for z in ['green', 'yellow', 'red']:
                c = (metrics_df['traffic_light_zone'] == z).sum()
                if c > 0:
                    lines.append(f"- **{z.capitalize()} zone:** {c} configurations")
        lines.append("")

    if "quantum_vs_classical_runtime_comparison" in sections and runtime_metrics:
        lines.append("## Runtime")
        lines.append("")
        for k, v in runtime_metrics.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                lines.append(f"- **{k}:** {v:.2f}")
        lines.append("")

    if "key_insights" in sections:
        lines.append("## Key Insights")
        lines.append("")
        if 'violation_ratio' in metrics_df.columns:
            vr = metrics_df['violation_ratio'].mean()
            if vr > 1.2:
                lines.append("- Risk underestimation: excessive CVaR breaches")
            elif vr < 0.8:
                lines.append("- Risk overestimation")
            else:
                lines.append("- Adequate risk calibration")
        lines.append("")

    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(f"- **Asset-config combinations:** {len(metrics_df)}")
    if 'asset' in metrics_df.columns:
        lines.append(f"- **Assets:** {metrics_df['asset'].nunique()}")
    lines.append("")
    numeric = metrics_df.select_dtypes(include=[np.number]).columns
    if len(numeric) > 0:
        lines.append("```")
        lines.append(metrics_df[numeric].describe().to_string())
        lines.append("```")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
