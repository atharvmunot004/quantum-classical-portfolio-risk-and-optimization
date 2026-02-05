"""
Report Generator Module for QAOA Portfolio CVaR Optimization.

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
    report_lines.append("# QAOA Portfolio CVaR Optimization Report")
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
        report_lines.append(f"- Unique configurations: {len(results_df):,}")
        
        if 'best_energy' in results_df.columns:
            report_lines.append(f"- Mean best energy: {results_df['best_energy'].mean():.6f}")
            report_lines.append(f"- Best energy achieved: {results_df['best_energy'].min():.6f}")
    
    if 'samples' in results and not results['samples'].empty:
        samples_df = results['samples']
        report_lines.append(f"- Total samples: {len(samples_df):,}")
    
    if 'performance' in results and not results['performance'].empty:
        perf_df = results['performance']
        report_lines.append(f"- Portfolios evaluated: {len(perf_df):,}")
        
        if 'realized_return' in perf_df.columns:
            report_lines.append(f"- Mean realized return: {perf_df['realized_return'].mean():.6f}")
        if 'realized_cvar' in perf_df.columns:
            report_lines.append(f"- Mean realized CVaR: {perf_df['realized_cvar'].mean():.6f}")
        if 'sharpe_ratio' in perf_df.columns:
            report_lines.append(f"- Mean Sharpe ratio: {perf_df['sharpe_ratio'].mean():.6f}")
    
    report_lines.append("")
    
    # Quantum Metrics
    report_lines.append("## Quantum-Specific Metrics")
    report_lines.append("")
    
    if 'results' in results and not results['results'].empty:
        results_df = results['results']
        
        if 'circuit_depth' in results_df.columns:
            report_lines.append(f"- Mean circuit depth: {results_df['circuit_depth'].mean():.1f}")
        if 'circuit_width' in results_df.columns:
            report_lines.append(f"- Mean circuit width: {results_df['circuit_width'].mean():.1f}")
        if 'shots' in results_df.columns:
            report_lines.append(f"- Mean shots: {results_df['shots'].mean():.0f}")
        if 'total_time_ms' in results_df.columns:
            report_lines.append(f"- Mean optimization time: {results_df['total_time_ms'].mean():.1f} ms")
    
    report_lines.append("")
    
    # Performance Statistics
    report_lines.append("## Performance Statistics")
    report_lines.append("")
    
    if 'performance' in results and not results['performance'].empty:
        perf_df = results['performance']
        
        if 'realized_return' in perf_df.columns:
            report_lines.append("### Realized Return")
            report_lines.append(f"- Mean: {perf_df['realized_return'].mean():.6f}")
            report_lines.append(f"- Median: {perf_df['realized_return'].median():.6f}")
            report_lines.append(f"- Std: {perf_df['realized_return'].std():.6f}")
            report_lines.append("")
        
        if 'realized_cvar' in perf_df.columns:
            report_lines.append("### Realized CVaR")
            report_lines.append(f"- Mean: {perf_df['realized_cvar'].mean():.6f}")
            report_lines.append(f"- Median: {perf_df['realized_cvar'].median():.6f}")
            report_lines.append("")
    
    # Key Insights
    report_lines.append("## Key Insights")
    report_lines.append("")
    report_lines.append("1. QAOA-based portfolio optimization enables quantum-enhanced asset selection")
    report_lines.append("2. Precomputation per asset set and date reduces repeated Hamiltonian construction")
    report_lines.append("3. CVaR-of-energy objective focuses optimization on worst-case scenarios")
    report_lines.append("4. Multi-objective weight sweeps explore Pareto-efficient solutions")
    report_lines.append("")
    
    report_content = "\n".join(report_lines)
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content
