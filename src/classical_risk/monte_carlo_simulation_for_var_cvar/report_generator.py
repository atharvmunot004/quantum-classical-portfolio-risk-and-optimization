"""
Report generation module for Monte Carlo VaR/CVaR evaluation results.
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
    risk_series_df: Optional[pd.DataFrame] = None,
    time_sliced_metrics_df: Optional[pd.DataFrame] = None,
    report_sections: Optional[List[str]] = None
) -> str:
    """Generate comprehensive markdown report from metrics DataFrame."""
    if report_sections is None:
        report_sections = [
            'methodology_overview',
            'monte_carlo_methods_and_assumptions',
            'rolling_forecast_construction',
            'backtesting_results',
            'time_sliced_backtesting',
            'tail_risk_behavior',
            'distributional_characteristics',
            'key_insights'
        ]
    
    report_lines = []
    
    # Header
    report_lines.append("# Monte Carlo Simulation for VaR/CVaR Evaluation Report (Asset Level)")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("**Scope:** Asset-level evaluation only (no portfolio aggregation)")
    report_lines.append("")
    
    # Methodology Overview
    if "methodology_overview" in report_sections:
        report_lines.append("## Methodology Overview")
        report_lines.append("")
        report_lines.append("Monte Carlo simulation generates multiple scenarios of asset returns based on historical data.")
        report_lines.append("VaR and CVaR are then calculated from the distribution of simulated returns per asset.")
        report_lines.append("")
        report_lines.append("**Design Principle:**")
        report_lines.append("- Asset-level only: evaluation is performed strictly on individual asset return series")
        report_lines.append("- No portfolio aggregation: all metrics computed per asset")
        report_lines.append("- Rolling window parameter estimation per asset")
        report_lines.append("- Forward return paths simulated per asset")
        report_lines.append("")
    
    # Monte Carlo Methods and Assumptions
    if "monte_carlo_methods_and_assumptions" in report_sections:
        report_lines.append("## Monte Carlo Methods and Assumptions")
        report_lines.append("")
        
        if monte_carlo_settings:
            methods = monte_carlo_settings.get('methods', [])
            for method in methods:
                if not method.get('enabled', True):
                    continue
                
                method_name = method.get('name', 'unknown')
                report_lines.append(f"### {method_name.replace('_', ' ').title()}")
                report_lines.append("")
                
                if method_name == 'historical_bootstrap':
                    bootstrap_type = method.get('bootstrap_type', 'iid')
                    block_bootstrap = method.get('block_bootstrap', {})
                    report_lines.append(f"- **Bootstrap Type:** {bootstrap_type}")
                    if block_bootstrap.get('enabled'):
                        report_lines.append(f"- **Block Length:** {block_bootstrap.get('block_length', 'N/A')}")
                
                elif method_name == 'parametric_normal':
                    fit_config = method.get('fit', {})
                    report_lines.append(f"- **Mean:** {fit_config.get('mu', 'sample_mean')}")
                    report_lines.append(f"- **Volatility:** {fit_config.get('sigma', 'sample_std')}")
                
                elif method_name == 'parametric_student_t':
                    fit_config = method.get('fit', {})
                    df_config = fit_config.get('df', {})
                    report_lines.append(f"- **Mean:** {fit_config.get('mu', 'sample_mean')}")
                    report_lines.append(f"- **Volatility:** {fit_config.get('sigma', 'sample_std')}")
                    report_lines.append(f"- **DF Mode:** {df_config.get('mode', 'mle_or_fixed')}")
                    report_lines.append(f"- **Fixed DF:** {df_config.get('fixed_df', 'N/A')}")
                
                elif method_name == 'filtered_ewma_bootstrap':
                    filter_config = method.get('filter', {})
                    report_lines.append(f"- **Lambda:** {filter_config.get('lambda', 'N/A')}")
                
                report_lines.append("")
            
            horizons_config = monte_carlo_settings.get('horizons', {})
            if isinstance(horizons_config, dict):
                report_lines.append("### Horizon Handling")
                report_lines.append("")
                report_lines.append(f"- **Method:** {horizons_config.get('horizon_handling', 'path_simulation')}")
                report_lines.append(f"- **Base Horizon:** {horizons_config.get('base_horizon', 1)} days")
                report_lines.append(f"- **Scaled Horizons:** {horizons_config.get('scaled_horizons', [])} days")
                report_lines.append(f"- **Path Aggregation:** {monte_carlo_settings.get('path_aggregation_rule', 'sum')}")
                report_lines.append("")
    
    # Rolling Forecast Construction
    if "rolling_forecast_construction" in report_sections:
        report_lines.append("## Rolling Forecast Construction")
        report_lines.append("")
        
        if monte_carlo_settings:
            rolling_config = monte_carlo_settings.get('rolling', {})
            report_lines.append(f"- **Step Size:** {rolling_config.get('step_size', 1)}")
            report_lines.append(f"- **Forecast Method:** {rolling_config.get('forecast_method', 'rolling')}")
            report_lines.append(f"- **Forecast Target:** {rolling_config.get('forecast_target', 'one_step_ahead')}")
            report_lines.append(f"- **Estimation Windows:** {monte_carlo_settings.get('estimation_windows', [])} days")
            report_lines.append(f"- **Number of Simulations:** {monte_carlo_settings.get('num_simulations', 'N/A'):,}")
            report_lines.append("")
    
    # Backtesting Results
    if "backtesting_results" in report_sections:
        report_lines.append("## Backtesting Results")
        report_lines.append("")
        
        if 'hit_rate' in metrics_df.columns:
            avg_hit_rate = metrics_df['hit_rate'].mean()
            report_lines.append(f"- **Average Hit Rate:** {avg_hit_rate:.4f}")
            report_lines.append("")
        
        if 'violation_ratio' in metrics_df.columns:
            avg_violation_ratio = metrics_df['violation_ratio'].mean()
            report_lines.append(f"- **Average Violation Ratio:** {avg_violation_ratio:.4f}")
            report_lines.append("")
        
        if 'traffic_light_zone' in metrics_df.columns:
            zone_counts = metrics_df['traffic_light_zone'].value_counts()
            report_lines.append("### Traffic Light Zones")
            report_lines.append("")
            for zone, count in zone_counts.items():
                report_lines.append(f"- **{zone.capitalize()}:** {count} ({count/len(metrics_df)*100:.1f}%)")
            report_lines.append("")
        
        if 'kupiec_unconditional_coverage' in metrics_df.columns:
            kupiec_passed = (metrics_df['kupiec_unconditional_coverage'] > 0.05).sum()
            report_lines.append(f"- **Kupiec Test Passed:** {kupiec_passed}/{len(metrics_df)} ({kupiec_passed/len(metrics_df)*100:.1f}%)")
            report_lines.append("")
        
        if 'christoffersen_conditional_coverage' in metrics_df.columns:
            cc_passed = (metrics_df['christoffersen_conditional_coverage'] > 0.05).sum()
            report_lines.append(f"- **Christoffersen Conditional Coverage Passed:** {cc_passed}/{len(metrics_df)} ({cc_passed/len(metrics_df)*100:.1f}%)")
            report_lines.append("")
    
    # Time-Sliced Backtesting
    if "time_sliced_backtesting" in report_sections and time_sliced_metrics_df is not None and len(time_sliced_metrics_df) > 0:
        report_lines.append("## Time-Sliced Backtesting")
        report_lines.append("")
        
        if 'slice_type' in time_sliced_metrics_df.columns:
            for slice_type in time_sliced_metrics_df['slice_type'].unique():
                report_lines.append(f"### By {slice_type.title()}")
                report_lines.append("")
                
                slice_data = time_sliced_metrics_df[time_sliced_metrics_df['slice_type'] == slice_type]
                if 'hit_rate' in slice_data.columns:
                    avg_hit_rate = slice_data['hit_rate'].mean()
                    report_lines.append(f"- **Average Hit Rate:** {avg_hit_rate:.4f}")
                    report_lines.append("")
                report_lines.append("")
    
    # Tail Risk Behavior
    if "tail_risk_behavior" in report_sections:
        report_lines.append("## Tail Risk Behavior")
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
    
    # Distributional Characteristics
    if "distributional_characteristics" in report_sections:
        report_lines.append("## Distributional Characteristics")
        report_lines.append("")
        
        if 'skewness' in metrics_df.columns:
            avg_skewness = metrics_df['skewness'].mean()
            report_lines.append(f"- **Average Skewness:** {avg_skewness:.4f}")
            report_lines.append("")
        
        if 'kurtosis' in metrics_df.columns:
            avg_kurtosis = metrics_df['kurtosis'].mean()
            report_lines.append(f"- **Average Excess Kurtosis:** {avg_kurtosis:.4f}")
            report_lines.append("")
        
        if 'jarque_bera_p_value' in metrics_df.columns:
            jb_passed = (metrics_df['jarque_bera_p_value'] > 0.05).sum()
            report_lines.append(f"- **Jarque-Bera Normality Test Passed:** {jb_passed}/{len(metrics_df)} ({jb_passed/len(metrics_df)*100:.1f}%)")
            report_lines.append("")
    
    # Key Insights
    if "key_insights" in report_sections:
        report_lines.append("## Key Insights")
        report_lines.append("")
        
        if 'method' in metrics_df.columns:
            report_lines.append("### Method Comparison")
            report_lines.append("")
            method_summary = metrics_df.groupby('method').agg({
                'hit_rate': 'mean',
                'violation_ratio': 'mean',
                'kupiec_unconditional_coverage': lambda x: (x > 0.05).mean()
            })
            report_lines.append("```")
            report_lines.append(method_summary.to_string())
            report_lines.append("```")
            report_lines.append("")
        
        if 'confidence_level' in metrics_df.columns:
            report_lines.append("### Confidence Level Comparison")
            report_lines.append("")
            conf_summary = metrics_df.groupby('confidence_level').agg({
                'hit_rate': 'mean',
                'violation_ratio': 'mean'
            })
            report_lines.append("```")
            report_lines.append(conf_summary.to_string())
            report_lines.append("```")
            report_lines.append("")
    
    report_content = "\n".join(report_lines)
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content
