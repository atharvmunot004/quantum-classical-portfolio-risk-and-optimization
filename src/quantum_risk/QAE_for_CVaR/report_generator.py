"""
Report generation for QAE VaR/CVaR evaluation.
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
        "distribution_encoding_and_state_preparation",
        "qae_var_and_cvar_construction",
        "confidence_interval_behavior",
        "backtesting_results",
        "time_sliced_backtesting",
        "quantum_vs_classical_runtime_comparison",
        "key_insights",
    ]
    quantum_settings = quantum_settings or {}
    runtime_metrics = runtime_metrics or {}
    lines = []

    lines.append("# QAE (Quantum Amplitude Estimation) VaR/CVaR Asset-Level Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if "quantum_methodology_overview" in sections:
        lines.append("## Quantum Methodology Overview")
        lines.append("")
        lines.append("Quantum Amplitude Estimation (QAE) is applied to individual asset loss distributions.")
        lines.append("VaR is estimated via CDF bisection using QAE; CVaR via Rockafellar-Uryasev with tail expectation.")
        lines.append("")

    if "distribution_encoding_and_state_preparation" in sections:
        lines.append("## Distribution Encoding and State Preparation")
        lines.append("")
        lines.append(f"- **Distribution family:** {quantum_settings.get('distribution_model', {}).get('family', 'student_t')}")
        lines.append(f"- **State qubits:** {quantum_settings.get('quantum_encoding', {}).get('num_state_qubits', 6)}")
        lines.append("- Discretization: uniform grid in loss space, rescaled to unit interval")
        lines.append("")

    if "qae_var_and_cvar_construction" in sections:
        lines.append("## QAE VaR and CVaR Construction")
        lines.append("")
        lines.append("- **VaR:** Bisection on CDF(L <= x) to find x such that CDF(x) ≈ 1-α")
        lines.append("- **CVaR:** VaR + (1/(1-α)) × E[(L - VaR)^+]")
        lines.append("")

    if "backtesting_results" in sections:
        lines.append("## Backtesting Results")
        lines.append("")
        if 'hit_rate' in metrics_df.columns and 'confidence_level' in metrics_df.columns:
            for cl in sorted(metrics_df['confidence_level'].unique()):
                sub = metrics_df[metrics_df['confidence_level'] == cl]
                exp = 1 - cl
                obs = sub['hit_rate'].mean()
                lines.append(f"**Confidence {cl:.0%}:** expected violation rate {exp:.4f}, observed {obs:.4f}")
            lines.append("")
        if 'violation_ratio' in metrics_df.columns:
            lines.append(f"- **Average violation ratio:** {metrics_df['violation_ratio'].mean():.4f}")
            lines.append("")
        if 'traffic_light_zone' in metrics_df.columns:
            for z in ['green', 'yellow', 'red']:
                c = (metrics_df['traffic_light_zone'] == z).sum()
                if c > 0:
                    lines.append(f"- **{z.capitalize()}:** {c} configurations")
            lines.append("")

    if "quantum_vs_classical_runtime_comparison" in sections and runtime_metrics:
        lines.append("## Runtime")
        lines.append("")
        for k, v in runtime_metrics.items():
            if isinstance(v, (int, float)):
                lines.append(f"- **{k}:** {v:.2f}")
        lines.append("")

    if "key_insights" in sections:
        lines.append("## Key Insights")
        lines.append("")
        if 'violation_ratio' in metrics_df.columns:
            vr = metrics_df['violation_ratio'].mean()
            if vr > 1.2:
                lines.append("- Risk underestimation: excessive VaR breaches")
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
