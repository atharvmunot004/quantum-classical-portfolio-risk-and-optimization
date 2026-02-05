"""
Report Generator Module for Quantum Portfolio Optimization.

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
    report_sections = config.get('outputs', {}).get('report', {}).get('include_sections', [])
    
    report_lines = []
    
    # Header
    report_lines.append("# Quantum Annealing Multi-Objective Portfolio Optimization Report")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Problem Formulation
    if 'problem_formulation' in report_sections:
        report_lines.append("## Problem Formulation")
        report_lines.append("")
        report_lines.append("### Objectives")
        report_lines.append("")
        report_lines.append("1. **Expected Return Maximization**: Maximize portfolio expected return")
        report_lines.append("2. **Tail Risk Minimization**: Minimize Conditional Value at Risk (CVaR)")
        report_lines.append("3. **Diversification**: Minimize pairwise correlation penalty")
        report_lines.append("")
        report_lines.append("### Constraints")
        report_lines.append("")
        constraints = config.get('constraints', {})
        report_lines.append(f"- **Budget Constraint**: Fully invested (sum of weights = 1.0)")
        report_lines.append(f"- **Cardinality Constraint**: {constraints.get('cardinality', {}).get('min_assets', 5)}-{constraints.get('cardinality', {}).get('max_assets', 15)} assets")
        report_lines.append(f"- **Long-Only**: No short selling allowed")
        report_lines.append("")
    
    # QUBO Construction
    if 'qubo_construction' in report_sections:
        report_lines.append("## QUBO Construction")
        report_lines.append("")
        qubo_config = config.get('qubo_construction', {})
        report_lines.append("The portfolio optimization problem is formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem:")
        report_lines.append("")
        report_lines.append("```")
        report_lines.append("minimize: x^T Q x + c")
        report_lines.append("subject to: x ∈ {0, 1}^n")
        report_lines.append("```")
        report_lines.append("")
        report_lines.append("### Objective Weights")
        obj_weights = qubo_config.get('objective_weights', {})
        report_lines.append(f"- Return weight: {obj_weights.get('return_weight', [])}")
        report_lines.append(f"- Risk weight: {obj_weights.get('risk_weight', [])}")
        report_lines.append(f"- Diversification weight: {obj_weights.get('diversification_weight', [])}")
        report_lines.append("")
        report_lines.append("### Penalty Weights")
        penalty_weights = qubo_config.get('penalty_weights', {})
        report_lines.append(f"- Budget penalty: {penalty_weights.get('budget_penalty', 10.0)}")
        report_lines.append(f"- Cardinality penalty: {penalty_weights.get('cardinality_penalty', 8.0)}")
        report_lines.append("")
    
    # Quantum Annealing and Embedding
    if 'quantum_annealing_and_embedding' in report_sections:
        report_lines.append("## Quantum Annealing and Embedding")
        report_lines.append("")
        qa_settings = config.get('quantum_annealing_settings', {})
        report_lines.append(f"### Solver Configuration")
        report_lines.append(f"- Backend: {qa_settings.get('backend', 'dwave_or_simulated_annealer')}")
        report_lines.append(f"- Number of reads: {qa_settings.get('annealing_parameters', {}).get('num_reads', 5000)}")
        report_lines.append(f"- Annealing time: {qa_settings.get('annealing_parameters', {}).get('annealing_time_us', [20, 50])} μs")
        report_lines.append("")
        
        # Embedding info
        if 'optimization_metrics' in results and not results['optimization_metrics'].empty:
            metrics_df = results['optimization_metrics']
            if 'embedding_size' in metrics_df.columns:
                avg_embedding_size = metrics_df['embedding_size'].mean()
                report_lines.append(f"- Average embedding size: {avg_embedding_size:.1f} qubits")
            if 'logical_to_physical_qubit_ratio' in metrics_df.columns:
                avg_ratio = metrics_df['logical_to_physical_qubit_ratio'].mean()
                report_lines.append(f"- Average logical to physical qubit ratio: {avg_ratio:.3f}")
            if 'chain_break_fraction' in metrics_df.columns:
                avg_chain_break = metrics_df['chain_break_fraction'].mean()
                report_lines.append(f"- Average chain break fraction: {avg_chain_break:.3f}")
        report_lines.append("")
    
    # Multi-Objective Tradeoffs
    if 'multi_objective_tradeoffs' in report_sections:
        report_lines.append("## Multi-Objective Tradeoffs")
        report_lines.append("")
        if 'portfolio_performance' in results and not results['portfolio_performance'].empty:
            perf_df = results['portfolio_performance']
            
            # Group by weight configuration
            if 'return_weight' in perf_df.columns:
                report_lines.append("### Performance by Weight Configuration")
                report_lines.append("")
                
                summary = perf_df.groupby(['return_weight', 'risk_weight', 'diversification_weight']).agg({
                    'realized_return': 'mean',
                    'realized_volatility': 'mean',
                    'realized_cvar': 'mean',
                    'sharpe_ratio': 'mean'
                }).reset_index()
                
                report_lines.append(summary.to_string(index=False))
                report_lines.append("")
        report_lines.append("")
    
    # Out-of-Sample Performance
    if 'out_of_sample_performance' in report_sections:
        report_lines.append("## Out-of-Sample Performance")
        report_lines.append("")
        if 'portfolio_performance' in results and not results['portfolio_performance'].empty:
            perf_df = results['portfolio_performance']
            
            if 'realized_return' in perf_df.columns:
                report_lines.append("### Overall Performance Metrics")
                report_lines.append("")
                report_lines.append(f"- Mean Realized Return: {perf_df['realized_return'].mean():.4f}")
                report_lines.append(f"- Mean Realized Volatility: {perf_df['realized_volatility'].mean():.4f}")
                report_lines.append(f"- Mean Realized CVaR: {perf_df['realized_cvar'].mean():.4f}")
                report_lines.append(f"- Mean Sharpe Ratio: {perf_df['sharpe_ratio'].mean():.4f}")
                report_lines.append("")
        report_lines.append("")
    
    # Benchmark Comparison
    if 'benchmark_comparison' in report_sections:
        report_lines.append("## Benchmark Comparison")
        report_lines.append("")
        report_lines.append("Benchmark comparison would require running classical optimization methods.")
        report_lines.append("This section can be populated after running benchmark optimizations.")
        report_lines.append("")
    
    # Limitations and Scalability
    if 'limitations_and_scalability' in report_sections:
        report_lines.append("## Limitations and Scalability")
        report_lines.append("")
        report_lines.append("### Current Limitations")
        report_lines.append("")
        report_lines.append("1. **Asset Universe Size**: Limited by quantum annealer connectivity (currently ~40 assets)")
        report_lines.append("2. **Embedding Overhead**: Logical to physical qubit ratio affects solution quality")
        report_lines.append("3. **Chain Breaks**: May occur in D-Wave systems, affecting solution quality")
        report_lines.append("4. **Computation Time**: Quantum annealing may be slower than classical methods for small problems")
        report_lines.append("")
        report_lines.append("### Scalability Considerations")
        report_lines.append("")
        report_lines.append("- Larger asset universes require more sophisticated embedding strategies")
        report_lines.append("- Multi-objective weight sweeps increase computation time linearly")
        report_lines.append("- Rolling window optimization scales with number of rebalancing dates")
        report_lines.append("")
    
    # Key Insights
    if 'key_insights' in report_sections:
        report_lines.append("## Key Insights")
        report_lines.append("")
        if 'optimization_metrics' in results and not results['optimization_metrics'].empty:
            metrics_df = results['optimization_metrics']
            
            if 'best_energy' in metrics_df.columns:
                report_lines.append(f"- Best energy achieved: {metrics_df['best_energy'].min():.4f}")
            if 'energy_gap' in metrics_df.columns:
                report_lines.append(f"- Average energy gap: {metrics_df['energy_gap'].mean():.4f}")
            if 'pareto_front_size' in metrics_df.columns:
                report_lines.append(f"- Average Pareto front size: {metrics_df['pareto_front_size'].mean():.1f}")
        
        report_lines.append("")
        report_lines.append("### Recommendations")
        report_lines.append("")
        report_lines.append("1. Experiment with different weight configurations to explore Pareto frontier")
        report_lines.append("2. Monitor chain break fractions to assess solution quality")
        report_lines.append("3. Compare with classical optimization methods for validation")
        report_lines.append("4. Consider hybrid quantum-classical approaches for larger problems")
        report_lines.append("")
    
    # Summary
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append(f"Total optimizations performed: {results.get('num_optimizations', 0)}")
    if 'portfolio_weights' in results and not results['portfolio_weights'].empty:
        report_lines.append(f"Total portfolio weight records: {len(results['portfolio_weights'])}")
    report_lines.append("")
    
    report_content = "\n".join(report_lines)
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content
