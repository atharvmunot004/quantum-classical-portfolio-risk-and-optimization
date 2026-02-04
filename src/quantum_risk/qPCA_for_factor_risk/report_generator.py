"""Markdown report for qPCA factor risk asset-level evaluation."""
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np


def generate_report(
    metrics_df: pd.DataFrame,
    output_path: Path,
    factor_df: Optional[pd.DataFrame] = None,
    runtime_metrics: Optional[Dict] = None,
    report_sections: Optional[List[str]] = None,
    qpca_settings: Optional[Dict] = None,
):
    sections = report_sections or [
        "quantum_methodology_overview",
        "density_matrix_construction_and_shrinkage",
        "qpca_eigenspectrum_estimation",
        "factor_exposure_construction",
        "factor_risk_proxy_methodology",
        "factor_stability_and_regime_shifts",
        "classical_pca_alignment_results",
        "runtime_and_state_preparation_overhead",
        "key_insights",
    ]
    runtime_metrics = runtime_metrics or {}
    qpca_settings = qpca_settings or {}
    lines = []

    lines.append("# qPCA Factor Risk Asset-Level Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if "quantum_methodology_overview" in sections:
        lines.append("## Quantum Methodology Overview")
        lines.append("")
        lines.append("Quantum PCA (qPCA) extracts dominant latent factors from rolling-window return")
        lines.append("covariance via density-matrix eigenstructure. Factors and explained variance are")
        lines.append("computed per window; asset-level factor exposures and factor-implied risk proxies")
        lines.append("(VaR/CVaR) support comparison with classical PCA-based factor risk.")
        lines.append("")

    if "density_matrix_construction_and_shrinkage" in sections:
        lines.append("## Density Matrix Construction and Shrinkage")
        lines.append("")
        lines.append("Cross-asset covariance is estimated with Ledoit-Wolf shrinkage per rolling window,")
        lines.append("then trace-normalized to a density matrix for qPCA. This ensures PSD and unit trace.")
        lines.append("")

    if "qpca_eigenspectrum_estimation" in sections:
        lines.append("## qPCA Eigenspectrum Estimation")
        lines.append("")
        lines.append("Eigenvalues and eigenvectors are obtained by classical eigensolver (simulating")
        lines.append("quantum phase estimation). Circuit depth/width and precision bits are reported")
        lines.append("for hardware feasibility discussion.")
        lines.append("")

    if "factor_exposure_construction" in sections:
        lines.append("## Factor Exposure Construction")
        lines.append("")
        lines.append("Per-asset factor exposures are the loadings from projecting each asset onto the")
        lines.append("top-k qPCA factors (eigenvectors) in each window.")
        lines.append("")

    if "factor_risk_proxy_methodology" in sections:
        lines.append("## Factor Risk Proxy Methodology")
        lines.append("")
        lines.append("Factor-implied VaR and CVaR use a Gaussian factor model: total variance =")
        lines.append("factor-explained variance + idiosyncratic variance; tail quantiles from normal.")
        lines.append("")

    if "factor_stability_and_regime_shifts" in sections and factor_df is not None and not factor_df.empty:
        lines.append("## Factor Stability and Regime Shifts")
        lines.append("")
        if "cumulative_explained_variance" in factor_df.columns:
            lines.append(f"- **Mean cumulative explained variance (top factors):** {factor_df['cumulative_explained_variance'].mean():.4f}")
        lines.append("")

    if "classical_pca_alignment_results" in sections and not metrics_df.empty:
        lines.append("## Classical PCA Alignment Results")
        lines.append("")
        for col in ["principal_angle_distance_vs_classical", "explained_variance_gap", "exposure_correlation"]:
            if col in metrics_df.columns:
                lines.append(f"- **{col}:** mean = {metrics_df[col].mean():.4f}")
        lines.append("")

    if "runtime_and_state_preparation_overhead" in sections and runtime_metrics:
        lines.append("## Runtime and State Preparation Overhead")
        lines.append("")
        for k, v in runtime_metrics.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                lines.append(f"- **{k}:** {v:.2f}")
        lines.append("")

    if "key_insights" in sections:
        lines.append("## Key Insights")
        lines.append("")
        if "exposure_correlation" in metrics_df.columns:
            ec = metrics_df["exposure_correlation"].mean()
            if ec > 0.9:
                lines.append("- High alignment between qPCA and classical PCA factor exposures.")
            elif ec < 0.7:
                lines.append("- Notable differences between qPCA and classical PCA exposures.")
            else:
                lines.append("- Moderate alignment between qPCA and classical PCA.")
        lines.append("")

    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(f"- **Metric rows:** {len(metrics_df)}")
    if "asset" in metrics_df.columns:
        lines.append(f"- **Assets:** {metrics_df['asset'].nunique()}")
    numeric = metrics_df.select_dtypes(include=[np.number]).columns
    if len(numeric) > 0:
        lines.append("```")
        lines.append(metrics_df[numeric].describe().to_string())
        lines.append("```")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
