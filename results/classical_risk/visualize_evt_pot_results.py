"""
Visualize EVT-POT VaR/CVaR evaluation results.

Generates comprehensive visualizations showing:
- Backtesting accuracy metrics
- Tail risk behavior
- Portfolio structure effects
- Distribution characteristics
- Performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Color palette
COLORS = sns.color_palette("husl", 10)


def load_evt_results(data_dir=None):
    """
    Load EVT-POT results from parquet or JSON file.
    
    Parameters:
    -----------
    data_dir : Path or str, optional
        Directory containing results files. Defaults to script directory.
    
    Returns:
    --------
    pd.DataFrame
        Results DataFrame
    """
    if data_dir is None:
        data_dir = Path(__file__).parent
    else:
        data_dir = Path(data_dir)
    
    # Try to load Parquet first (more efficient)
    parquet_path = data_dir / "evt_pot_var_cvar_metrics.parquet"
    
    if parquet_path.exists():
        print(f"Loading EVT-POT results from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        raise FileNotFoundError(f"Results file not found in {data_dir}")
    
    print(f"Loaded {len(df):,} portfolio-configuration combinations")
    print(f"Columns: {len(df.columns)}")
    return df


def plot_backtesting_accuracy(df, output_dir):
    """Plot backtesting accuracy metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Hit rate distribution
    ax1 = axes[0, 0]
    hit_rates = df['hit_rate'].dropna()
    ax1.hist(hit_rates, bins=50, color=COLORS[0], alpha=0.7, edgecolor='black')
    ax1.axvline(hit_rates.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {hit_rates.mean():.4f}')
    expected_rate = 1 - df['confidence_level'].iloc[0]
    ax1.axvline(expected_rate, color='green', linestyle='--', linewidth=2,
                label=f'Expected: {expected_rate:.4f}')
    ax1.set_xlabel('Hit Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Hit Rate Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Violation ratio distribution
    ax2 = axes[0, 1]
    violation_ratios = df['violation_ratio'].dropna()
    ax2.hist(violation_ratios, bins=50, color=COLORS[1], alpha=0.7, edgecolor='black')
    ax2.axvline(violation_ratios.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {violation_ratios.mean():.4f}')
    ax2.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Ideal: 1.0')
    ax2.set_xlabel('Violation Ratio', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Violation Ratio Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Traffic light zones
    ax3 = axes[1, 0]
    if 'traffic_light_zone' in df.columns:
        zone_counts = df['traffic_light_zone'].value_counts()
        colors_map = {'green': 'green', 'yellow': 'orange', 'red': 'red'}
        zone_colors = [colors_map.get(zone, 'gray') for zone in zone_counts.index]
        bars = ax3.bar(zone_counts.index, zone_counts.values, color=zone_colors, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Traffic Light Zone', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax3.set_title('Traffic Light Zone Distribution', fontsize=13, fontweight='bold')
        for bar, val in zip(bars, zone_counts.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:,}\n({val/len(df)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
    
    # Kupiec test p-values
    ax4 = axes[1, 1]
    kupiec_pvals = df['kupiec_unconditional_coverage'].dropna()
    ax4.hist(kupiec_pvals, bins=50, color=COLORS[2], alpha=0.7, edgecolor='black')
    ax4.axvline(0.05, color='red', linestyle='--', linewidth=2, label='Significance: 0.05')
    ax4.set_xlabel('Kupiec Test P-Value', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Kupiec Unconditional Coverage Test', fontsize=13, fontweight='bold')
    passed = (kupiec_pvals > 0.05).sum()
    ax4.text(0.7, 0.9, f'Passed: {passed:,}/{len(kupiec_pvals):,}\n({passed/len(kupiec_pvals)*100:.1f}%)',
             transform=ax4.transAxes, fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.suptitle('Backtesting Accuracy Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = output_dir / "1_backtesting_accuracy.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_tail_risk_metrics(df, output_dir):
    """Plot tail risk metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Mean exceedance distribution
    ax1 = axes[0, 0]
    mean_exceedances = df['mean_exceedance'].dropna()
    if len(mean_exceedances) > 0:
        ax1.hist(mean_exceedances, bins=50, color=COLORS[0], alpha=0.7, edgecolor='black')
        ax1.axvline(mean_exceedances.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_exceedances.mean():.6f}')
        ax1.set_xlabel('Mean Exceedance', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Mean Exceedance Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
    
    # Max exceedance distribution
    ax2 = axes[0, 1]
    max_exceedances = df['max_exceedance'].dropna()
    if len(max_exceedances) > 0:
        ax2.hist(max_exceedances, bins=50, color=COLORS[1], alpha=0.7, edgecolor='black')
        ax2.axvline(max_exceedances.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {max_exceedances.mean():.6f}')
        ax2.set_xlabel('Max Exceedance', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Max Exceedance Distribution', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    # CVaR mean exceedance
    ax3 = axes[0, 2]
    cvar_exceedances = df['cvar_mean_exceedance'].dropna()
    if len(cvar_exceedances) > 0:
        ax3.hist(cvar_exceedances, bins=50, color=COLORS[2], alpha=0.7, edgecolor='black')
        ax3.axvline(cvar_exceedances.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {cvar_exceedances.mean():.6f}')
        ax3.set_xlabel('CVaR Mean Exceedance', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('CVaR Mean Exceedance Distribution', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
    
    # Tail index (xi) distribution
    ax4 = axes[1, 0]
    tail_indices = df['tail_index_xi'].dropna()
    if len(tail_indices) > 0:
        ax4.hist(tail_indices, bins=50, color=COLORS[3], alpha=0.7, edgecolor='black')
        ax4.axvline(tail_indices.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {tail_indices.mean():.4f}')
        ax4.axvline(0, color='green', linestyle='--', linewidth=2, label='Exponential (xi=0)')
        ax4.set_xlabel('Tail Index (ξ)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.set_title('Tail Index Distribution', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
    
    # Scale parameter (beta) distribution
    ax5 = axes[1, 1]
    scale_betas = df['scale_beta'].dropna()
    if len(scale_betas) > 0:
        ax5.hist(scale_betas, bins=50, color=COLORS[4], alpha=0.7, edgecolor='black')
        ax5.axvline(scale_betas.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {scale_betas.mean():.6f}')
        ax5.set_xlabel('Scale Parameter (β)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax5.set_title('Scale Parameter Distribution', fontsize=13, fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
    
    # Expected shortfall exceedance
    ax6 = axes[1, 2]
    es_exceedances = df['expected_shortfall_exceedance'].dropna()
    if len(es_exceedances) > 0:
        ax6.hist(es_exceedances, bins=50, color=COLORS[5], alpha=0.7, edgecolor='black')
        ax6.axvline(es_exceedances.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {es_exceedances.mean():.6f}')
        ax6.set_xlabel('Expected Shortfall Exceedance', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax6.set_title('Expected Shortfall Exceedance Distribution', fontsize=13, fontweight='bold')
        ax6.legend()
        ax6.grid(alpha=0.3)
    
    plt.suptitle('Tail Risk Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = output_dir / "2_tail_risk_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_portfolio_structure_effects(df, output_dir):
    """Plot portfolio structure effects on risk metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Violation ratio vs HHI
    ax1 = axes[0, 0]
    if 'hhi_concentration' in df.columns and 'violation_ratio' in df.columns:
        hhi = df['hhi_concentration'].dropna()
        vr = df['violation_ratio'].dropna()
        common_idx = hhi.index.intersection(vr.index)
        if len(common_idx) > 0:
            ax1.scatter(hhi[common_idx], vr[common_idx], alpha=0.3, s=10, color=COLORS[0])
            ax1.set_xlabel('HHI Concentration', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Violation Ratio', fontsize=12, fontweight='bold')
            ax1.set_title('Violation Ratio vs Portfolio Concentration', fontsize=13, fontweight='bold')
            ax1.axhline(1.0, color='green', linestyle='--', linewidth=2, label='Ideal: 1.0')
            ax1.legend()
            ax1.grid(alpha=0.3)
    
    # Violation ratio vs number of assets
    ax2 = axes[0, 1]
    if 'num_active_assets' in df.columns and 'violation_ratio' in df.columns:
        num_assets = df['num_active_assets'].dropna()
        vr = df['violation_ratio'].dropna()
        common_idx = num_assets.index.intersection(vr.index)
        if len(common_idx) > 0:
            ax2.scatter(num_assets[common_idx], vr[common_idx], alpha=0.3, s=10, color=COLORS[1])
            ax2.set_xlabel('Number of Active Assets', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Violation Ratio', fontsize=12, fontweight='bold')
            ax2.set_title('Violation Ratio vs Portfolio Size', fontsize=13, fontweight='bold')
            ax2.axhline(1.0, color='green', linestyle='--', linewidth=2, label='Ideal: 1.0')
            ax2.legend()
            ax2.grid(alpha=0.3)
    
    # Tail index vs HHI
    ax3 = axes[1, 0]
    if 'hhi_concentration' in df.columns and 'tail_index_xi' in df.columns:
        hhi = df['hhi_concentration'].dropna()
        xi = df['tail_index_xi'].dropna()
        common_idx = hhi.index.intersection(xi.index)
        if len(common_idx) > 0:
            ax3.scatter(hhi[common_idx], xi[common_idx], alpha=0.3, s=10, color=COLORS[2])
            ax3.set_xlabel('HHI Concentration', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Tail Index (ξ)', fontsize=12, fontweight='bold')
            ax3.set_title('Tail Index vs Portfolio Concentration', fontsize=13, fontweight='bold')
            ax3.axhline(0, color='green', linestyle='--', linewidth=2, label='Exponential (xi=0)')
            ax3.legend()
            ax3.grid(alpha=0.3)
    
    # Portfolio size distribution
    ax4 = axes[1, 1]
    if 'num_active_assets' in df.columns:
        num_assets = df['num_active_assets'].dropna()
        ax4.hist(num_assets, bins=30, color=COLORS[3], alpha=0.7, edgecolor='black')
        ax4.axvline(num_assets.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {num_assets.mean():.1f}')
        ax4.set_xlabel('Number of Active Assets', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.set_title('Portfolio Size Distribution', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
    
    plt.suptitle('Portfolio Structure Effects', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = output_dir / "3_portfolio_structure_effects.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_distribution_metrics(df, output_dir):
    """Plot distribution characteristics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Skewness distribution
    ax1 = axes[0, 0]
    if 'skewness' in df.columns:
        skewness = df['skewness'].dropna()
        ax1.hist(skewness, bins=50, color=COLORS[0], alpha=0.7, edgecolor='black')
        ax1.axvline(skewness.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {skewness.mean():.4f}')
        ax1.axvline(0, color='green', linestyle='--', linewidth=2, label='Symmetric (skew=0)')
        ax1.set_xlabel('Skewness', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Return Skewness Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
    
    # Kurtosis distribution
    ax2 = axes[0, 1]
    if 'kurtosis' in df.columns:
        kurtosis = df['kurtosis'].dropna()
        ax2.hist(kurtosis, bins=50, color=COLORS[1], alpha=0.7, edgecolor='black')
        ax2.axvline(kurtosis.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {kurtosis.mean():.4f}')
        ax2.axvline(0, color='green', linestyle='--', linewidth=2, label='Normal (kurt=0)')
        ax2.set_xlabel('Excess Kurtosis', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Return Kurtosis Distribution', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    # Jarque-Bera test results
    ax3 = axes[1, 0]
    if 'jarque_bera_p_value' in df.columns:
        jb_pvals = df['jarque_bera_p_value'].dropna()
        ax3.hist(jb_pvals, bins=50, color=COLORS[2], alpha=0.7, edgecolor='black')
        ax3.axvline(0.05, color='red', linestyle='--', linewidth=2, label='Significance: 0.05')
        ax3.set_xlabel('Jarque-Bera Test P-Value', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Jarque-Bera Normality Test', fontsize=13, fontweight='bold')
        passed = (jb_pvals > 0.05).sum()
        ax3.text(0.7, 0.9, f'Normal: {passed:,}/{len(jb_pvals):,}\n({passed/len(jb_pvals)*100:.1f}%)',
                 transform=ax3.transAxes, fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3.legend()
        ax3.grid(alpha=0.3)
    
    # Skewness vs Kurtosis
    ax4 = axes[1, 1]
    if 'skewness' in df.columns and 'kurtosis' in df.columns:
        skew = df['skewness'].dropna()
        kurt = df['kurtosis'].dropna()
        common_idx = skew.index.intersection(kurt.index)
        if len(common_idx) > 0:
            ax4.scatter(skew[common_idx], kurt[common_idx], alpha=0.3, s=10, color=COLORS[3])
            ax4.set_xlabel('Skewness', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Excess Kurtosis', fontsize=12, fontweight='bold')
            ax4.set_title('Skewness vs Kurtosis', fontsize=13, fontweight='bold')
            ax4.axhline(0, color='green', linestyle='--', linewidth=1, alpha=0.5)
            ax4.axvline(0, color='green', linestyle='--', linewidth=1, alpha=0.5)
            ax4.grid(alpha=0.3)
    
    plt.suptitle('Distribution Characteristics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = output_dir / "4_distribution_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_performance_metrics(df, output_dir):
    """Plot performance and runtime metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Runtime distribution
    ax1 = axes[0, 0]
    if 'runtime_per_portfolio_ms' in df.columns:
        runtime = df['runtime_per_portfolio_ms'].dropna()
        ax1.hist(runtime, bins=50, color=COLORS[0], alpha=0.7, edgecolor='black')
        ax1.axvline(runtime.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {runtime.mean():.2f} ms')
        ax1.axvline(runtime.median(), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {runtime.median():.2f} ms')
        ax1.set_xlabel('Runtime per Portfolio (ms)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Runtime Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
    
    # Cache hit ratio
    ax2 = axes[0, 1]
    if 'cache_hit_ratio' in df.columns:
        cache_ratio = df['cache_hit_ratio'].dropna()
        unique_ratios = cache_ratio.unique()
        if len(unique_ratios) > 0:
            ratio_counts = cache_ratio.value_counts()
            bars = ax2.bar([f'{r:.1%}' for r in ratio_counts.index], ratio_counts.values,
                          color=COLORS[1], alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Cache Hit Ratio', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax2.set_title('Cache Hit Ratio Distribution', fontsize=13, fontweight='bold')
            for bar, val in zip(bars, ratio_counts.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
    
    # Runtime vs portfolio size
    ax3 = axes[1, 0]
    if 'num_active_assets' in df.columns and 'runtime_per_portfolio_ms' in df.columns:
        num_assets = df['num_active_assets'].dropna()
        runtime = df['runtime_per_portfolio_ms'].dropna()
        common_idx = num_assets.index.intersection(runtime.index)
        if len(common_idx) > 0:
            ax3.scatter(num_assets[common_idx], runtime[common_idx], alpha=0.3, s=10, color=COLORS[2])
            ax3.set_xlabel('Number of Active Assets', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Runtime per Portfolio (ms)', fontsize=12, fontweight='bold')
            ax3.set_title('Runtime vs Portfolio Size', fontsize=13, fontweight='bold')
            ax3.grid(alpha=0.3)
    
    # Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = "Performance Summary\n" + "="*30 + "\n\n"
    
    if 'runtime_per_portfolio_ms' in df.columns:
        runtime = df['runtime_per_portfolio_ms'].dropna()
        summary_text += f"Runtime:\n"
        summary_text += f"  Mean: {runtime.mean():.2f} ms\n"
        summary_text += f"  Median: {runtime.median():.2f} ms\n"
        summary_text += f"  P95: {runtime.quantile(0.95):.2f} ms\n\n"
    
    if 'cache_hit_ratio' in df.columns:
        cache_ratio = df['cache_hit_ratio'].dropna()
        summary_text += f"Cache Performance:\n"
        summary_text += f"  Hit Ratio: {cache_ratio.mean():.1%}\n\n"
    
    if 'hit_rate' in df.columns:
        hit_rate = df['hit_rate'].dropna()
        summary_text += f"Backtesting:\n"
        summary_text += f"  Mean Hit Rate: {hit_rate.mean():.4f}\n"
        summary_text += f"  Expected: {1 - df['confidence_level'].iloc[0]:.4f}\n"
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=12, fontfamily='monospace', verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Performance Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = output_dir / "5_performance_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_correlation_heatmap(df, output_dir):
    """Plot correlation heatmap of key metrics."""
    # Select key numeric columns for correlation
    key_metrics = [
        'hit_rate', 'violation_ratio', 'mean_exceedance', 'max_exceedance',
        'cvar_mean_exceedance', 'tail_index_xi', 'scale_beta',
        'num_active_assets', 'hhi_concentration', 'skewness', 'kurtosis',
        'runtime_per_portfolio_ms'
    ]
    
    available_metrics = [m for m in key_metrics if m in df.columns]
    corr_df = df[available_metrics].corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Heatmap of Key Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    output_path = output_dir / "6_correlation_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main function to generate all visualizations."""
    print("=" * 60)
    print("EVT-POT Results Visualization Generator")
    print("=" * 60)
    
    # Load data
    df = load_evt_results()
    
    # Create output directory
    output_dir = Path(__file__).parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")
    
    # Generate all visualizations
    print("Generating visualizations...")
    print("-" * 60)
    
    plot_backtesting_accuracy(df, output_dir)
    plot_tail_risk_metrics(df, output_dir)
    plot_portfolio_structure_effects(df, output_dir)
    plot_distribution_metrics(df, output_dir)
    plot_performance_metrics(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    
    print("-" * 60)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated visualizations:")
    print("  1. Backtesting Accuracy Metrics")
    print("  2. Tail Risk Metrics")
    print("  3. Portfolio Structure Effects")
    print("  4. Distribution Metrics")
    print("  5. Performance Metrics")
    print("  6. Correlation Heatmap")
    print("=" * 60)


if __name__ == "__main__":
    main()











