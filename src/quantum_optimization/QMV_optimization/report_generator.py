"""
Report Generator Module for QMV Portfolio Optimization.

Generates comprehensive reports on optimization results.
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
    Generate comprehensive optimization report.
    
    Args:
        results: Dictionary with optimization results
        config: Configuration dictionary
        output_path: Path to save report
        
    Returns:
        Report content as string
    """
    report_lines = []
    
    # Header
    report_lines.append("# QMV Portfolio Optimization Report")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Summary
    report_lines.append("## Summary")
    report_lines.append("")
    
    if 'results' in results and not results['results'].empty:
        results_df = results['results']
        report_lines.append(f"- Total optimizations: {len(results_df):,}")
        report_lines.append(f"- Unique dates: {results_df['date'].nunique():,}")
        report_lines.append(f"- Unique lambda values: {results_df['lambda_risk'].nunique():,}")
        
        if 'energy' in results_df.columns:
            report_lines.append(f"- Mean energy: {results_df['energy'].mean():.6f}")
            report_lines.append(f"- Best energy: {results_df['energy'].min():.6f}")
        
        if 'runtime_ms' in results_df.columns:
            report_lines.append(f"- Mean runtime: {results_df['runtime_ms'].mean():.1f} ms")
    
    if 'performance' in results and not results['performance'].empty:
        perf_df = results['performance']
        report_lines.append(f"- Portfolios evaluated: {len(perf_df):,}")
        
        if 'realized_return' in perf_df.columns:
            report_lines.append(f"- Mean realized return: {perf_df['realized_return'].mean():.6f}")
        if 'realized_volatility' in perf_df.columns:
            report_lines.append(f"- Mean realized volatility: {perf_df['realized_volatility'].mean():.6f}")
        if 'sharpe_ratio' in perf_df.columns:
            report_lines.append(f"- Mean Sharpe ratio: {perf_df['sharpe_ratio'].mean():.6f}")
    
    report_lines.append("")
    
    # Optimization Quality
    report_lines.append("## Optimization Quality")
    report_lines.append("")
    
    if 'results' in results and not results['results'].empty:
        results_df = results['results']
        
        if 'energy' in results_df.columns:
            sorted_energies = results_df['energy'].sort_values()
            if len(sorted_energies) > 1:
                energy_gap = sorted_energies.iloc[1] - sorted_energies.iloc[0]
                report_lines.append(f"- Energy gap: {energy_gap:.6f}")
    
    report_lines.append("")
    
    # Performance Statistics
    report_lines.append("## Performance Statistics")
    report_lines.append("")
    
    if 'performance' in results and not results['performance'].empty:
        perf_df = results['performance']
        
        if 'lambda_risk' in perf_df.columns:
            for lambda_val in sorted(perf_df['lambda_risk'].unique()):
                lambda_data = perf_df[perf_df['lambda_risk'] == lambda_val]
                report_lines.append(f"### Lambda Risk: {lambda_val}")
                report_lines.append("")
                
                if 'realized_return' in lambda_data.columns:
                    report_lines.append(f"- Mean return: {lambda_data['realized_return'].mean():.6f}")
                if 'realized_volatility' in lambda_data.columns:
                    report_lines.append(f"- Mean volatility: {lambda_data['realized_volatility'].mean():.6f}")
                if 'sharpe_ratio' in lambda_data.columns:
                    report_lines.append(f"- Mean Sharpe: {lambda_data['sharpe_ratio'].mean():.6f}")
                report_lines.append("")
    
    # Key Insights
    report_lines.append("## Key Insights")
    report_lines.append("")
    report_lines.append("1. QMV optimization enables quantum-enhanced mean-variance portfolio construction")
    report_lines.append("2. Binary discretized weight encoding allows QUBO formulation")
    report_lines.append("3. Precomputation per asset set reduces repeated covariance estimation")
    report_lines.append("4. Lambda risk sweep explores efficient frontier")
    report_lines.append("")
    
    report_content = "\n".join(report_lines)
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content
