"""
Report generation module for Variance-Covariance VaR evaluation results.

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
    varcov_settings: Optional[Dict] = None,
    report_sections: Optional[List[str]] = None,
    runtime_metrics: Optional[Dict] = None
) -> str:
    """
    Generate comprehensive markdown report from metrics DataFrame.
    
    Args:
        metrics_df: DataFrame with all computed metrics
        output_path: Path to save the report
        varcov_settings: Dictionary with Variance-Covariance VaR settings
        report_sections: List of sections to include
        runtime_metrics: Optional dictionary with runtime metrics
        
    Returns:
        Report content as string
    """
    if report_sections is None:
        report_sections = [
            "methodology_overview",
            "normality_assumption",
            "rolling_mean_volatility_estimation",
            "variance_covariance_var_construction",
            "backtesting_results",
            "time_sliced_backtesting",
            "distributional_characteristics",
            "computational_performance",
            "key_insights"
        ]
    
    report_lines = []
    
    # Header
    report_lines.append("# Variance-Covariance Value-at-Risk Evaluation Report")
    report_lines.append("Asset-Level Evaluation")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Methodology Overview
    if "methodology_overview" in report_sections:
        report_lines.append("## Methodology Overview")
        report_lines.append("")
        report_lines.append("### Variance-Covariance (Parametric) VaR")
        report_lines.append("")
        report_lines.append("The Variance-Covariance method assumes returns follow a normal distribution")
        report_lines.append("and calculates VaR using parametric estimation:")
        report_lines.append("")
        report_lines.append("```")
        report_lines.append("VaR_h = -μ + z_{1-α} * σ * √(h)")
        report_lines.append("```")
        report_lines.append("")
        report_lines.append("Where:")
        report_lines.append("- μ = mean return (estimated from rolling window)")
        report_lines.append("- σ = standard deviation of returns (estimated from rolling window)")
        report_lines.append("- z_{1-α} = (1-α) quantile of standard normal distribution")
        report_lines.append("- α = 1 - confidence_level (e.g., α = 0.05 for 95% VaR)")
        report_lines.append("- h = time horizon in days")
        report_lines.append("- √(h) = square root scaling for multi-day horizons")
        report_lines.append("")
        report_lines.append("This method is evaluated strictly per asset using rolling windows")
        report_lines.append("under the assumption of conditional normality.")
        report_lines.append("")
        
        if varcov_settings:
            report_lines.append("### VaR Settings")
            report_lines.append("")
            report_lines.append(f"- **Distributional Assumption:** {varcov_settings.get('distributional_assumption', 'normal')}")
            report_lines.append(f"- **Mean Estimator:** {varcov_settings.get('mean_estimator', 'sample_mean')}")
            report_lines.append(f"- **Volatility Estimator:** {varcov_settings.get('volatility_estimator', 'sample_std')}")
            report_lines.append(f"- **Confidence Levels:** {varcov_settings.get('confidence_levels', [])}")
            horizons_config = varcov_settings.get('horizons', {})
            base_horizon = horizons_config.get('base_horizon', 1)
            scaled_horizons = horizons_config.get('scaled_horizons', [])
            horizons = [base_horizon] + scaled_horizons
            report_lines.append(f"- **Horizons:** {horizons} days")
            report_lines.append(f"- **Scaling Rule:** {horizons_config.get('scaling_rule', 'sqrt_time')}")
            report_lines.append(f"- **Estimation Windows:** {varcov_settings.get('estimation_windows', [])} days")
            report_lines.append("")
    
    # Normality Assumption
    if "normality_assumption" in report_sections:
        report_lines.append("## Normality Assumption")
        report_lines.append("")
        report_lines.append("The Variance-Covariance method assumes returns follow a normal distribution.")
        report_lines.append("This assumption is tested using the Jarque-Bera test for normality.")
        report_lines.append("")
        
        if 'jarque_bera_p_value' in metrics_df.columns:
            normality_passed = (metrics_df['jarque_bera_p_value'] > 0.05).sum()
            total_assets = len(metrics_df)
            report_lines.append(f"- **Normality Tests Passed (Jarque-Bera, p > 0.05):** {normality_passed}/{total_assets} assets ({normality_passed/total_assets*100:.1f}%)")
            report_lines.append("")
            
            if normality_passed / total_assets < 0.5:
                report_lines.append("**Warning:** Less than 50% of assets pass the normality test.")
                report_lines.append("This suggests the normal distribution assumption may not hold for many assets.")
                report_lines.append("")
    
    # Rolling Mean and Volatility Estimation
    if "rolling_mean_volatility_estimation" in report_sections:
        report_lines.append("## Rolling Mean and Volatility Estimation")
        report_lines.append("")
        report_lines.append("Mean and volatility are estimated using rolling windows:")
        report_lines.append("")
        
        if varcov_settings:
            mean_estimator = varcov_settings.get('mean_estimator', 'sample_mean')
            vol_estimator = varcov_settings.get('volatility_estimator', 'sample_std')
            report_lines.append(f"- **Mean Estimator:** {mean_estimator}")
            report_lines.append(f"- **Volatility Estimator:** {vol_estimator}")
            report_lines.append("")
        
        if 'rolling_mean' in metrics_df.columns:
            avg_mean = metrics_df['rolling_mean'].mean()
            report_lines.append(f"- **Average Rolling Mean:** {avg_mean:.6f}")
            report_lines.append("")
        
        if 'rolling_volatility' in metrics_df.columns:
            avg_vol = metrics_df['rolling_volatility'].mean()
            report_lines.append(f"- **Average Rolling Volatility:** {avg_vol:.6f}")
            report_lines.append("")
    
    # Variance-Covariance VaR Construction
    if "variance_covariance_var_construction" in report_sections:
        report_lines.append("## Variance-Covariance VaR Construction")
        report_lines.append("")
        report_lines.append("VaR is constructed using the parametric formula with estimated mean and volatility.")
        report_lines.append("For each rolling window position:")
        report_lines.append("1. Estimate mean (μ) and volatility (σ) from the window")
        report_lines.append("2. Compute VaR = -μ + z_{1-α} * σ * √(h)")
        report_lines.append("3. Advance window by step size and repeat")
        report_lines.append("")
    
    # Summary Statistics
    report_lines.append("## Summary Statistics")
    report_lines.append("")
    num_unique_assets = metrics_df['asset'].nunique() if 'asset' in metrics_df.columns else len(metrics_df)
    report_lines.append(f"- **Total Assets Evaluated:** {num_unique_assets}")
    report_lines.append(f"- **Total Configuration Combinations:** {len(metrics_df)}")
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
            report_lines.append("  - Ratio > 1 indicates overestimation of risk")
            report_lines.append("  - Ratio < 1 indicates underestimation of risk")
            report_lines.append("  - Ratio ≈ 1 indicates accurate risk estimation")
            report_lines.append("")
        
        if 'traffic_light_zone' in metrics_df.columns:
            zone_counts = metrics_df['traffic_light_zone'].value_counts()
            report_lines.append("### Traffic Light Zones")
            report_lines.append("")
            for zone, count in zone_counts.items():
                if pd.notna(zone):
                    report_lines.append(f"- **{zone.capitalize()}:** {count} asset-configurations ({count/len(metrics_df)*100:.1f}%)")
            report_lines.append("")
        
        if 'kupiec_unconditional_coverage' in metrics_df.columns:
            kupiec_passed = (metrics_df['kupiec_unconditional_coverage'] > 0.05).sum()
            report_lines.append(f"- **Kupiec Test Passed (p > 0.05):** {kupiec_passed}/{len(metrics_df)} configurations ({kupiec_passed/len(metrics_df)*100:.1f}%)")
            report_lines.append("")
        
        if 'christoffersen_conditional_coverage' in metrics_df.columns:
            christoffersen_passed = (metrics_df['christoffersen_conditional_coverage'] > 0.05).sum()
            report_lines.append(f"- **Christoffersen Conditional Coverage Test Passed (p > 0.05):** {christoffersen_passed}/{len(metrics_df)} configurations ({christoffersen_passed/len(metrics_df)*100:.1f}%)")
            report_lines.append("")
    
    # Time-Sliced Backtesting
    if "time_sliced_backtesting" in report_sections:
        report_lines.append("## Time-Sliced Backtesting")
        report_lines.append("")
        report_lines.append("Backtesting metrics are computed for specific time periods (year, quarter, month)")
        report_lines.append("to analyze temporal patterns in VaR performance.")
        report_lines.append("")
        report_lines.append("See time-sliced metrics table for detailed results by time period.")
        report_lines.append("")
    
    # Distributional Characteristics
    if "distributional_characteristics" in report_sections:
        report_lines.append("## Distributional Characteristics")
        report_lines.append("")
        
        if 'skewness' in metrics_df.columns:
            avg_skewness = metrics_df['skewness'].mean()
            report_lines.append(f"- **Average Skewness:** {avg_skewness:.4f}")
            report_lines.append("  - Negative values indicate left-skewed distributions (tail risk)")
            report_lines.append("  - Positive values indicate right-skewed distributions")
            report_lines.append("")
        
        if 'kurtosis' in metrics_df.columns:
            avg_kurtosis = metrics_df['kurtosis'].mean()
            report_lines.append(f"- **Average Excess Kurtosis:** {avg_kurtosis:.4f}")
            report_lines.append("  - Positive values indicate fat tails (excess kurtosis > 0)")
            report_lines.append("  - Normal distribution has excess kurtosis = 0")
            report_lines.append("")
            if avg_kurtosis > 3:
                report_lines.append("**Warning:** Average excess kurtosis > 3 indicates significant fat tails.")
                report_lines.append("This violates the normality assumption of Variance-Covariance VaR.")
                report_lines.append("")
        
        if 'jarque_bera_p_value' in metrics_df.columns:
            avg_jb_p = metrics_df['jarque_bera_p_value'].mean()
            report_lines.append(f"- **Average Jarque-Bera p-value:** {avg_jb_p:.4f}")
            report_lines.append("  - p > 0.05 suggests normality cannot be rejected")
            report_lines.append("  - p < 0.05 suggests non-normality")
            report_lines.append("")
    
    # Computational Performance
    if "computational_performance" in report_sections:
        report_lines.append("## Computational Performance")
        report_lines.append("")
        
        if runtime_metrics:
            if 'total_runtime_ms' in runtime_metrics:
                total_runtime = runtime_metrics['total_runtime_ms'] / 1000
                report_lines.append(f"- **Total Runtime:** {total_runtime/60:.2f} minutes ({total_runtime:.2f} seconds)")
                report_lines.append("")
            
            if 'runtime_per_asset_ms' in runtime_metrics:
                avg_runtime = runtime_metrics['runtime_per_asset_ms']
                report_lines.append(f"- **Average Runtime per Asset:** {avg_runtime:.2f} ms")
                report_lines.append("")
            
            if 'p95_runtime_ms' in runtime_metrics:
                p95_runtime = runtime_metrics['p95_runtime_ms']
                report_lines.append(f"- **95th Percentile Runtime:** {p95_runtime:.2f} ms")
                report_lines.append("")
            
            if 'mean_estimation_time_ms' in runtime_metrics:
                mean_time = runtime_metrics['mean_estimation_time_ms']
                report_lines.append(f"- **Mean Estimation Time:** {mean_time:.2f} ms")
                report_lines.append("")
            
            if 'volatility_estimation_time_ms' in runtime_metrics:
                vol_time = runtime_metrics['volatility_estimation_time_ms']
                report_lines.append(f"- **Volatility Estimation Time:** {vol_time:.2f} ms")
                report_lines.append("")
            
            if 'var_compute_time_ms' in runtime_metrics:
                var_time = runtime_metrics['var_compute_time_ms']
                report_lines.append(f"- **VaR Computation Time:** {var_time:.2f} ms")
                report_lines.append("")
            
            if 'cache_hit_ratio' in runtime_metrics:
                cache_ratio = runtime_metrics['cache_hit_ratio']
                report_lines.append(f"- **Cache Hit Ratio:** {cache_ratio:.2%}")
                report_lines.append("")
    
    # Key Insights
    if "key_insights" in report_sections:
        report_lines.append("## Key Insights")
        report_lines.append("")
        report_lines.append("### Findings")
        report_lines.append("")
        
        if 'violation_ratio' in metrics_df.columns:
            avg_vr = metrics_df['violation_ratio'].mean()
            if avg_vr > 1.2:
                report_lines.append("- **Risk Overestimation:** VaR tends to overestimate risk (violation ratio > 1.2)")
                report_lines.append("  This may indicate the normal distribution assumption is too conservative.")
            elif avg_vr < 0.8:
                report_lines.append("- **Risk Underestimation:** VaR tends to underestimate risk (violation ratio < 0.8)")
                report_lines.append("  This may indicate the normal distribution assumption fails to capture tail risk.")
            else:
                report_lines.append("- **Adequate Risk Estimation:** VaR provides reasonable risk estimates")
            report_lines.append("")
        
        if 'kurtosis' in metrics_df.columns:
            avg_kurt = metrics_df['kurtosis'].mean()
            if avg_kurt > 3:
                report_lines.append("- **Fat Tails Detected:** Returns exhibit significant fat tails (excess kurtosis > 3)")
                report_lines.append("  This violates the normality assumption and may limit VaR accuracy.")
            elif avg_kurt > 0:
                report_lines.append("- **Moderate Fat Tails:** Returns exhibit some fat tails (excess kurtosis > 0)")
                report_lines.append("  The normality assumption may be approximately valid.")
            else:
                report_lines.append("- **Normal-Like Tails:** Returns exhibit approximately normal tail behavior.")
            report_lines.append("")
        
        if 'jarque_bera_p_value' in metrics_df.columns:
            normality_rate = (metrics_df['jarque_bera_p_value'] > 0.05).mean()
            if normality_rate < 0.5:
                report_lines.append("- **Normality Assumption Violated:** Less than 50% of assets pass normality tests.")
                report_lines.append("  Consider alternative methods (e.g., EVT, GARCH) for assets with non-normal returns.")
            report_lines.append("")
        
        report_lines.append("### Recommendations")
        report_lines.append("")
        report_lines.append("- Use Variance-Covariance VaR as a baseline for comparison with other methods")
        report_lines.append("- For assets with fat tails or non-normal returns, consider EVT or GARCH methods")
        report_lines.append("- Monitor assets in 'red' traffic light zone more closely")
        report_lines.append("- Adjust confidence levels or horizons based on backtesting results")
        report_lines.append("- Consider longer estimation windows for more stable parameter estimates")
        report_lines.append("")
    
    report_content = "\n".join(report_lines)
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content
