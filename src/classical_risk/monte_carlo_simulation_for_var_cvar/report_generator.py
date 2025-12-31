"""
Report generation module for Monte Carlo VaR/CVaR evaluation results.

Generates comprehensive markdown reports with all evaluation metrics and insights.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime


def generate_report(
    metrics_df: pd.DataFrame,
    output_path: Union[str, Path],
    monte_carlo_settings: Optional[Dict] = None,
    report_sections: Optional[List[str]] = None
) -> str:
    """
    Generate comprehensive markdown report from metrics DataFrame.
    
    Args:
        metrics_df: DataFrame with all computed metrics
        output_path: Path to save the report
        monte_carlo_settings: Dictionary with Monte Carlo settings
        report_sections: List of sections to include (default matches llm.json spec)
        
    Returns:
        Report content as string
    """
    if report_sections is None:
        report_sections = [
            "model_configuration",
            "simulation_design",
            "batch_execution_summary",
            "aggregate_backtesting_results",
            "tail_behavior_summary",
            "runtime_statistics"
        ]
    
    report_lines = []
    
    # Header
    report_lines.append("# Monte Carlo Simulation for VaR/CVaR Evaluation Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Model Configuration
    if "model_configuration" in report_sections:
        report_lines.append("## Model Configuration")
        report_lines.append("")
        if monte_carlo_settings:
            report_lines.append("### Distribution Estimation")
            report_lines.append("")
            mean_model = monte_carlo_settings.get('mean_model', {})
            covariance_model = monte_carlo_settings.get('covariance_model', {})
            report_lines.append(f"- **Mean Model:** {mean_model.get('estimator', 'sample_mean')} (enabled: {mean_model.get('enabled', True)})")
            report_lines.append(f"- **Covariance Model:** {covariance_model.get('estimator', 'sample_covariance')}")
            shrinkage = covariance_model.get('shrinkage', {})
            if shrinkage.get('enabled', False):
                report_lines.append(f"  - **Shrinkage:** {shrinkage.get('method', 'ledoit_wolf')}")
            report_lines.append("")
    
    # Kernel Loop Order Validation
    if "kernel_loop_order_validation" in report_sections:
        report_lines.append("## Kernel Loop Order Validation")
        report_lines.append("")
        report_lines.append("This implementation follows the time-first batch execution design:")
        report_lines.append("")
        report_lines.append("**Loop Order:** `time -> batch -> simulations`")
        report_lines.append("")
        report_lines.append("- **Outer loop:** time_index (rolling window end)")
        report_lines.append("- **Inner loop:** portfolio_batch")
        report_lines.append("- **Forbidden in batched path:**")
        report_lines.append("  - `process_single_portfolio`")
        report_lines.append("  - `compute_rolling_var`")
        report_lines.append("  - `compute_rolling_cvar`")
        report_lines.append("")
        report_lines.append("**Design Rationale:**")
        report_lines.append("Rolling Monte Carlo risk must be computed time-first: at each rolling window end index,")
        report_lines.append("scenarios are generated/loaded once and reused to compute VaR/CVaR for the entire")
        report_lines.append("portfolio batch via a single BLAS matmul, then ALL metrics are updated via streaming accumulators.")
        report_lines.append("This eliminates per-portfolio rolling loops while preserving all metrics.")
        report_lines.append("")
    
    # Simulation Design
    if "simulation_design" in report_sections:
        report_lines.append("## Methodology Overview")
        report_lines.append("")
        report_lines.append("### Monte Carlo Simulation for VaR and CVaR")
        report_lines.append("")
        report_lines.append("Monte Carlo simulation generates multiple scenarios of portfolio returns based on historical data.")
        report_lines.append("VaR and CVaR are then calculated from the distribution of simulated returns.")
        report_lines.append("")
        report_lines.append("**Advantages:**")
        report_lines.append("- Can capture non-normal distributions")
        report_lines.append("- Flexible with different distribution assumptions")
        report_lines.append("- Provides both VaR and CVaR estimates")
        report_lines.append("")
        report_lines.append("**Method:**")
        report_lines.append("1. Estimate mean and covariance from historical returns")
        report_lines.append("2. Simulate multiple scenarios of future returns")
        report_lines.append("3. Calculate VaR as the quantile of simulated losses")
        report_lines.append("4. Calculate CVaR as the expected loss beyond VaR")
        report_lines.append("")
    
        report_lines.append("## Simulation Design")
        report_lines.append("")
        
        if monte_carlo_settings:
            report_lines.append("### Simulation Parameters")
            report_lines.append("")
            report_lines.append(f"- **Number of Simulations:** {monte_carlo_settings.get('num_simulations', 'N/A'):,}")
            report_lines.append(f"- **Distribution Type:** {monte_carlo_settings.get('distribution_type', 'N/A')}")
            report_lines.append(f"- **Random Seed:** {monte_carlo_settings.get('random_seed', 'N/A')}")
            report_lines.append(f"- **Confidence Levels:** {monte_carlo_settings.get('confidence_levels', [])}")
            horizons_config = monte_carlo_settings.get('horizons', {})
            if isinstance(horizons_config, dict):
                base_horizon = horizons_config.get('base_horizon', 1)
                scaled_horizons = horizons_config.get('scaled_horizons', [])
                report_lines.append(f"- **Base Horizon:** {base_horizon} days")
                report_lines.append(f"- **Scaled Horizons:** {scaled_horizons} days")
                report_lines.append(f"- **Scaling Rule:** {horizons_config.get('scaling_rule', 'sqrt_time')}")
            else:
                report_lines.append(f"- **Horizons:** {horizons_config} days")
            report_lines.append(f"- **Estimation Windows:** {monte_carlo_settings.get('estimation_windows', [])} days")
            report_lines.append("")
            report_lines.append("### Design Principle")
            report_lines.append("")
            report_lines.append("- Asset-level simulation with portfolio projection")
            report_lines.append("- Scenarios estimated once per estimation window and reused across portfolios")
            report_lines.append("- Batched execution for scalability")
            report_lines.append("")
    
    # Batch Execution Summary
    if "batch_execution_summary" in report_sections:
        report_lines.append("## Batch Execution Summary")
        report_lines.append("")
        report_lines.append(f"- **Total Portfolios Evaluated:** {len(metrics_df['portfolio_id'].unique()) if 'portfolio_id' in metrics_df.columns else 'N/A'}")
        report_lines.append(f"- **Total Configurations:** {len(metrics_df)}")
        report_lines.append("")
    
    # Aggregate Backtesting Results
    if "aggregate_backtesting_results" in report_sections:
        report_lines.append("## Aggregate Backtesting Results")
        report_lines.append("")
        
        if 'hit_rate' in metrics_df.columns:
            avg_hit_rate = metrics_df['hit_rate'].mean()
            report_lines.append(f"- **Average Hit Rate:** {avg_hit_rate:.4f}")
            report_lines.append("")
        
        if 'violation_ratio' in metrics_df.columns:
            avg_violation_ratio = metrics_df['violation_ratio'].mean()
            report_lines.append(f"- **Average Violation Ratio:** {avg_violation_ratio:.4f}")
            report_lines.append("  - Ratio > 1 indicates overestimation of risk")
            report_lines.append("  - Ratio < 1 indicates underestimation of risk")
            report_lines.append("")
        
        if 'traffic_light_zone' in metrics_df.columns:
            zone_counts = metrics_df['traffic_light_zone'].value_counts()
            report_lines.append("### Traffic Light Zones")
            report_lines.append("")
            for zone, count in zone_counts.items():
                report_lines.append(f"- **{zone.capitalize()}:** {count} portfolios ({count/len(metrics_df)*100:.1f}%)")
            report_lines.append("")
        
        if 'kupiec_unconditional_coverage' in metrics_df.columns:
            kupiec_passed = (metrics_df['kupiec_unconditional_coverage'] > 0.05).sum()
            report_lines.append(f"- **Kupiec Test Passed:** {kupiec_passed}/{len(metrics_df)} portfolios ({kupiec_passed/len(metrics_df)*100:.1f}%)")
            report_lines.append("")
    
    # VaR/CVaR Comparison
    if "var_cvar_comparison" in report_sections:
        report_lines.append("## VaR vs CVaR Comparison")
        report_lines.append("")
        report_lines.append("CVaR (Conditional Value at Risk) provides additional insight into tail risk beyond VaR.")
        report_lines.append("")
        
        if 'cvar_mean_exceedance' in metrics_df.columns:
            avg_cvar_mean = metrics_df['cvar_mean_exceedance'].mean()
            avg_var_mean = metrics_df['mean_exceedance'].mean() if 'mean_exceedance' in metrics_df.columns else np.nan
            report_lines.append(f"- **Average CVaR Mean Exceedance:** {avg_cvar_mean:.6f}")
            if not np.isnan(avg_var_mean):
                report_lines.append(f"- **Average VaR Mean Exceedance:** {avg_var_mean:.6f}")
                report_lines.append(f"- **Difference:** {avg_cvar_mean - avg_var_mean:.6f}")
            report_lines.append("")
        
        if 'cvar_max_exceedance' in metrics_df.columns:
            avg_cvar_max = metrics_df['cvar_max_exceedance'].mean()
            avg_var_max = metrics_df['max_exceedance'].mean() if 'max_exceedance' in metrics_df.columns else np.nan
            report_lines.append(f"- **Average CVaR Max Exceedance:** {avg_cvar_max:.6f}")
            if not np.isnan(avg_var_max):
                report_lines.append(f"- **Average VaR Max Exceedance:** {avg_var_max:.6f}")
            report_lines.append("")
    
    # Tail Behavior Summary
    if "tail_behavior_summary" in report_sections:
        report_lines.append("## Tail Behavior Summary")
        report_lines.append("")
        
        if 'mean_exceedance' in metrics_df.columns:
            avg_mean_exceedance = metrics_df['mean_exceedance'].mean()
            report_lines.append(f"- **Average Mean Exceedance (VaR):** {avg_mean_exceedance:.6f}")
            report_lines.append("")
        
        if 'max_exceedance' in metrics_df.columns:
            avg_max_exceedance = metrics_df['max_exceedance'].mean()
            report_lines.append(f"- **Average Max Exceedance (VaR):** {avg_max_exceedance:.6f}")
            report_lines.append("")
        
        if 'cvar_mean_exceedance' in metrics_df.columns:
            avg_cvar_mean_exceedance = metrics_df['cvar_mean_exceedance'].mean()
            report_lines.append(f"- **Average CVaR Mean Exceedance:** {avg_cvar_mean_exceedance:.6f}")
            report_lines.append("")
        
        if 'cvar_max_exceedance' in metrics_df.columns:
            avg_cvar_max_exceedance = metrics_df['cvar_max_exceedance'].mean()
            report_lines.append(f"- **Average CVaR Max Exceedance:** {avg_cvar_max_exceedance:.6f}")
            report_lines.append("")
    
    # Tail Behavior Summary
    if "tail_behavior_summary" in report_sections:
        report_lines.append("## Tail Behavior Summary")
        report_lines.append("")
        
        if 'mean_exceedance' in metrics_df.columns:
            avg_mean_exceedance = metrics_df['mean_exceedance'].mean()
            report_lines.append(f"- **Average Mean Exceedance (VaR):** {avg_mean_exceedance:.6f}")
            report_lines.append("")
        
        if 'max_exceedance' in metrics_df.columns:
            avg_max_exceedance = metrics_df['max_exceedance'].mean()
            report_lines.append(f"- **Average Max Exceedance (VaR):** {avg_max_exceedance:.6f}")
            report_lines.append("")
        
        if 'cvar_mean_exceedance' in metrics_df.columns:
            avg_cvar_mean_exceedance = metrics_df['cvar_mean_exceedance'].mean()
            report_lines.append(f"- **Average CVaR Mean Exceedance:** {avg_cvar_mean_exceedance:.6f}")
            report_lines.append("")
        
        if 'cvar_max_exceedance' in metrics_df.columns:
            avg_cvar_max_exceedance = metrics_df['cvar_max_exceedance'].mean()
            report_lines.append(f"- **Average CVaR Max Exceedance:** {avg_cvar_max_exceedance:.6f}")
            report_lines.append("")
        
        if 'num_active_assets' in metrics_df.columns:
            avg_active_assets = metrics_df['num_active_assets'].mean()
            report_lines.append(f"- **Average Number of Active Assets:** {avg_active_assets:.2f}")
            report_lines.append("")
        
        if 'hhi_concentration' in metrics_df.columns:
            avg_hhi = metrics_df['hhi_concentration'].mean()
            report_lines.append(f"- **Average HHI Concentration:** {avg_hhi:.4f}")
            report_lines.append("  - Higher values indicate more concentrated portfolios")
            report_lines.append("")
        
        if 'skewness' in metrics_df.columns:
            avg_skewness = metrics_df['skewness'].mean()
            report_lines.append(f"- **Average Skewness:** {avg_skewness:.4f}")
            report_lines.append("  - Negative values indicate left-skewed (tail risk)")
            report_lines.append("")
        
        if 'kurtosis' in metrics_df.columns:
            avg_kurtosis = metrics_df['kurtosis'].mean()
            report_lines.append(f"- **Average Excess Kurtosis:** {avg_kurtosis:.4f}")
            report_lines.append("  - Positive values indicate fat tails")
            report_lines.append("")
    
    # Runtime Statistics
    if "runtime_statistics" in report_sections:
        report_lines.append("## Runtime Statistics")
        report_lines.append("")
        
        if 'simulation_time_ms' in metrics_df.columns:
            avg_sim_time = metrics_df['simulation_time_ms'].mean()
            report_lines.append(f"- **Average Simulation Time:** {avg_sim_time:.2f} ms")
            report_lines.append("")
        
        if 'runtime_per_portfolio_ms' in metrics_df.columns:
            avg_runtime = metrics_df['runtime_per_portfolio_ms'].mean()
            report_lines.append(f"- **Average Runtime per Portfolio:** {avg_runtime:.2f} ms")
            report_lines.append("")
        
        if 'p95_runtime_ms' in metrics_df.columns:
            p95_runtime = metrics_df['p95_runtime_ms'].mean()
            report_lines.append(f"- **95th Percentile Runtime:** {p95_runtime:.2f} ms")
            report_lines.append("")
    
    # Memory Statistics
    if "memory_statistics" in report_sections:
        report_lines.append("## Memory Statistics")
        report_lines.append("")
        report_lines.append("Memory usage statistics are tracked during batch execution.")
        report_lines.append("Check batch_progress.json in the cache directory for detailed memory metrics.")
        report_lines.append("")
        report_lines.append("**Key Metrics:**")
        report_lines.append("- Peak RSS (Resident Set Size): Peak memory usage during execution")
        report_lines.append("- Swap Usage: Should be 0 MB (swap usage indicates memory pressure)")
        report_lines.append("")
        report_lines.append("**Memory Optimization Features:**")
        report_lines.append("- Float32 used for simulations and projections (halves memory footprint)")
        report_lines.append("- Thread-based parallelism (avoids process memory duplication)")
        report_lines.append("- Shard-based I/O (reduces memory spikes from large DataFrames)")
        report_lines.append("")
    
    # Summary Statistics Table
    report_lines.append("## Summary Statistics")
    report_lines.append("")
    report_lines.append("### Aggregate Metrics")
    report_lines.append("")
    
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary_stats = metrics_df[numeric_cols].describe()
        report_lines.append("```")
        report_lines.append(summary_stats.to_string())
        report_lines.append("```")
        report_lines.append("")
    
    report_content = "\n".join(report_lines)
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content

