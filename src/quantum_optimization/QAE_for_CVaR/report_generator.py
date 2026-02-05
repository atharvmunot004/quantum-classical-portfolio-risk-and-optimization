"""
Report Generator Module for QAE Portfolio CVaR Evaluation.

Generates comprehensive reports on evaluation results.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime


def generate_report(
    results: Dict,
    config: Dict,
    output_path: Path
) -> str:
    """
    Generate comprehensive evaluation report.
    
    Args:
        results: Dictionary with evaluation results
        config: Configuration dictionary
        output_path: Path to save report
        
    Returns:
        Report content as string
    """
    report_lines = []
    
    # Header
    report_lines.append("# QAE Portfolio CVaR Evaluation Report")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Summary
    report_lines.append("## Summary")
    report_lines.append("")
    
    if 'evaluation_results' in results and not results['evaluation_results'].empty:
        eval_df = results['evaluation_results']
        report_lines.append(f"- Total portfolio evaluations: {len(eval_df):,}")
        report_lines.append(f"- Unique portfolios: {eval_df.index.nunique():,}")
        
        if 'confidence_level' in eval_df.columns:
            report_lines.append(f"- Confidence levels evaluated: {eval_df['confidence_level'].unique().tolist()}")
        
        if 'cvar' in eval_df.columns:
            report_lines.append(f"- Mean CVaR: {eval_df['cvar'].mean():.6f}")
            report_lines.append(f"- Median CVaR: {eval_df['cvar'].median():.6f}")
            report_lines.append(f"- Std CVaR: {eval_df['cvar'].std():.6f}")
    
    if 'portfolio_asset_set_map' in results:
        map_df = results['portfolio_asset_set_map']
        report_lines.append(f"- Unique asset sets: {map_df['asset_set_id'].nunique():,}")
    
    report_lines.append("")
    
    # Quantum Metrics
    report_lines.append("## Quantum-Specific Metrics")
    report_lines.append("")
    
    if 'evaluation_results' in results and not results['evaluation_results'].empty:
        eval_df = results['evaluation_results']
        
        if 'qae_point_estimate' in eval_df.columns:
            report_lines.append(f"- Mean QAE point estimate: {eval_df['qae_point_estimate'].mean():.6f}")
        
        if 'qae_ci_width' in eval_df.columns:
            report_lines.append(f"- Mean QAE CI width: {eval_df['qae_ci_width'].mean():.6f}")
        
        if 'num_grover_iterations' in eval_df.columns:
            report_lines.append(f"- Mean Grover iterations: {eval_df['num_grover_iterations'].mean():.1f}")
        
        if 'circuit_depth' in eval_df.columns:
            report_lines.append(f"- Mean circuit depth: {eval_df['circuit_depth'].mean():.1f}")
        
        if 'circuit_width' in eval_df.columns:
            report_lines.append(f"- Mean circuit width: {eval_df['circuit_width'].mean():.1f}")
    
    report_lines.append("")
    
    # CVaR Statistics
    report_lines.append("## CVaR Statistics")
    report_lines.append("")
    
    if 'evaluation_results' in results and not results['evaluation_results'].empty:
        eval_df = results['evaluation_results']
        
        if 'confidence_level' in eval_df.columns and 'cvar' in eval_df.columns:
            for conf_level in sorted(eval_df['confidence_level'].unique()):
                conf_data = eval_df[eval_df['confidence_level'] == conf_level]
                report_lines.append(f"### Confidence Level: {conf_level}")
                report_lines.append("")
                report_lines.append(f"- Mean CVaR: {conf_data['cvar'].mean():.6f}")
                report_lines.append(f"- Median CVaR: {conf_data['cvar'].median():.6f}")
                report_lines.append(f"- Min CVaR: {conf_data['cvar'].min():.6f}")
                report_lines.append(f"- Max CVaR: {conf_data['cvar'].max():.6f}")
                report_lines.append("")
    
    # Key Insights
    report_lines.append("## Key Insights")
    report_lines.append("")
    report_lines.append("1. QAE-based CVaR estimation enables efficient portfolio risk evaluation")
    report_lines.append("2. Precomputation per asset set allows reuse across portfolios")
    report_lines.append("3. Quantum artifacts are cached and reused for computational efficiency")
    report_lines.append("")
    
    report_content = "\n".join(report_lines)
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content
