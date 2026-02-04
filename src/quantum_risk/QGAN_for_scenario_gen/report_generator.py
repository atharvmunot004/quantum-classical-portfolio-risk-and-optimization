"""
Report generation for QGAN scenario generation asset-level evaluation.
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
    """
    Generate markdown report for QGAN evaluation.
    
    Args:
        metrics_df: Metrics DataFrame
        output_path: Output file path
        quantum_settings: QGAN settings dict
        report_sections: List of sections to include
        runtime_metrics: Runtime metrics dict
    """
    sections = report_sections or [
        "quantum_methodology_overview",
        "data_preprocessing_and_discretization",
        "qgan_architecture_and_training",
        "scenario_generation_protocol",
        "distribution_and_tail_fidelity_results",
        "stylized_facts_validation",
        "time_sliced_regime_analysis",
        "runtime_and_scalability",
        "key_insights",
    ]
    quantum_settings = quantum_settings or {}
    runtime_metrics = runtime_metrics or {}
    lines = []
    
    lines.append("# QGAN Scenario Generation Asset-Level Risk Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    if "quantum_methodology_overview" in sections:
        lines.append("## Quantum Methodology Overview")
        lines.append("")
        lines.append("Quantum Generative Adversarial Networks (QGAN) are trained per asset")
        lines.append("to learn the empirical distribution of returns/losses within rolling windows")
        lines.append("and to generate realistic synthetic scenarios. Generated scenarios are used")
        lines.append("for asset-level risk assessment (VaR/CVaR estimation and tail diagnostics).")
        lines.append("")
    
    if "data_preprocessing_and_discretization" in sections:
        lines.append("## Data Preprocessing and Discretization")
        lines.append("")
        gen_cfg = quantum_settings.get('generator', {})
        data_cfg = quantum_settings.get('data_representation', {})
        lines.append(f"- **Number of qubits:** {gen_cfg.get('num_qubits', 6)}")
        lines.append(f"- **Discretization bins:** {data_cfg.get('discretization', {}).get('num_bins', 64)}")
        lines.append(f"- **Standardization:** {quantum_settings.get('standardization', {}).get('method', 'robust_zscore')}")
        lines.append("")
    
    if "qgan_architecture_and_training" in sections:
        lines.append("## QGAN Architecture and Training")
        lines.append("")
        gen_cfg = quantum_settings.get('generator', {})
        disc_cfg = quantum_settings.get('discriminator', {})
        opt_cfg = quantum_settings.get('optimization', {})
        lines.append(f"- **Generator:** Hardware-efficient ansatz with {gen_cfg.get('ansatz', {}).get('layers', 3)} layers")
        lines.append(f"- **Discriminator:** MLP with hidden layers {disc_cfg.get('hidden_layers', [64, 32])}")
        lines.append(f"- **Max iterations:** {opt_cfg.get('max_iterations', 300)}")
        lines.append(f"- **Batch size:** {opt_cfg.get('batch_size', 256)}")
        lines.append("")
    
    if "scenario_generation_protocol" in sections:
        lines.append("## Scenario Generation Protocol")
        lines.append("")
        scen_cfg = quantum_settings.get('scenario_generation', {})
        lines.append(f"- **Scenarios per timestamp:** {scen_cfg.get('num_scenarios_per_timestamp', 10000)}")
        lines.append(f"- **Horizons:** {scen_cfg.get('horizons', {}).get('scaled_horizons', [1])}")
        lines.append("")
    
    if "distribution_and_tail_fidelity_results" in sections:
        lines.append("## Distribution and Tail Fidelity Results")
        lines.append("")
        if 'wasserstein_distance' in metrics_df.columns:
            lines.append(f"- **Mean Wasserstein distance:** {metrics_df['wasserstein_distance'].mean():.6f}")
        if 'ks_statistic' in metrics_df.columns:
            lines.append(f"- **Mean KS statistic:** {metrics_df['ks_statistic'].mean():.6f}")
        if 'var_error_95' in metrics_df.columns:
            lines.append(f"- **Mean VaR error (95%):** {metrics_df['var_error_95'].mean():.6f}")
        if 'cvar_error_95' in metrics_df.columns:
            lines.append(f"- **Mean CVaR error (95%):** {metrics_df['cvar_error_95'].mean():.6f}")
        lines.append("")
    
    if "stylized_facts_validation" in sections:
        lines.append("## Stylized Facts Validation")
        lines.append("")
        if 'volatility_clustering_proxy' in metrics_df.columns:
            lines.append(f"- **Volatility clustering proxy:** {metrics_df['volatility_clustering_proxy'].mean():.6f}")
        if 'leptokurtosis_gap' in metrics_df.columns:
            lines.append(f"- **Leptokurtosis gap:** {metrics_df['leptokurtosis_gap'].mean():.6f}")
        lines.append("")
    
    if "runtime_and_scalability" in sections and runtime_metrics:
        lines.append("## Runtime and Scalability")
        lines.append("")
        for k, v in runtime_metrics.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                lines.append(f"- **{k}:** {v:.2f}")
        lines.append("")
    
    if "key_insights" in sections:
        lines.append("## Key Insights")
        lines.append("")
        if 'wasserstein_distance' in metrics_df.columns:
            wd_mean = metrics_df['wasserstein_distance'].mean()
            if wd_mean < 0.1:
                lines.append("- Excellent distribution fidelity: low Wasserstein distance")
            elif wd_mean < 0.5:
                lines.append("- Good distribution fidelity")
            else:
                lines.append("- Distribution fidelity needs improvement")
        
        if 'mode_collapse_score' in metrics_df.columns:
            mc_mean = metrics_df['mode_collapse_score'].mean()
            if mc_mean > 0.5:
                lines.append("- Good diversity: low mode collapse")
            else:
                lines.append("- Mode collapse detected: generated samples lack diversity")
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
