"""
Visualize portfolio data with comprehensive graphs and charts.
Generates multiple visualizations showing portfolio composition, distributions, and statistics.
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

# Color palette for stocks
STOCK_COLORS = sns.color_palette("husl", 10)


def load_portfolio_data(data_dir=None):
    """
    Load portfolio data from CSV or Parquet file.
    
    Parameters:
    -----------
    data_dir : Path or str, optional
        Directory containing portfolio files. Defaults to script directory.
    
    Returns:
    --------
    pd.DataFrame
        Portfolio weights DataFrame
    dict
        Portfolio metadata from llm.json
    """
    if data_dir is None:
        data_dir = Path(__file__).parent
    else:
        data_dir = Path(data_dir)
    
    # Try to load Parquet first (more efficient), fall back to CSV
    parquet_path = data_dir / "portfolios.parquet"
    csv_path = data_dir / "portfolios.csv"
    
    if parquet_path.exists():
        print(f"Loading portfolios from: {parquet_path}")
        portfolios_df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        print(f"Loading portfolios from: {csv_path}")
        portfolios_df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Portfolio data not found in {data_dir}")
    
    # Load metadata
    metadata_path = data_dir / "llm.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    print(f"Loaded {len(portfolios_df):,} portfolios with {len(portfolios_df.columns)} stocks")
    return portfolios_df, metadata


def plot_portfolio_size_distribution(portfolios_df, metadata, output_dir):
    """Plot distribution of portfolio sizes (number of active stocks)."""
    portfolio_sizes = portfolios_df.astype(bool).sum(axis=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    size_counts = portfolio_sizes.value_counts().sort_index()
    bars = ax1.bar(size_counts.index, size_counts.values, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Portfolio Size (Number of Active Stocks)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Portfolios', fontsize=12, fontweight='bold')
    ax1.set_title('Portfolio Size Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9)
    
    # Pie chart
    if metadata and 'portfolio_size_distribution' in metadata:
        sizes = []
        counts = []
        for size, info in sorted(metadata['portfolio_size_distribution'].items(), key=lambda x: int(x[0])):
            sizes.append(f"{size} stocks")
            counts.append(info['percentage'])
        
        colors = sns.color_palette("Set3", len(sizes))
        wedges, texts, autotexts = ax2.pie(counts, labels=sizes, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        ax2.set_title('Portfolio Size Distribution (Percentage)', fontsize=14, fontweight='bold')
    else:
        size_counts_pct = (size_counts / len(portfolios_df) * 100).sort_index()
        sizes = [f"{s} stocks" for s in size_counts_pct.index]
        colors = sns.color_palette("Set3", len(sizes))
        wedges, texts, autotexts = ax2.pie(size_counts_pct.values, labels=sizes, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        ax2.set_title('Portfolio Size Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "1_portfolio_size_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_weight_distribution_per_stock(portfolios_df, output_dir):
    """Plot weight distribution for each stock."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    stocks = portfolios_df.columns.tolist()
    
    for idx, stock in enumerate(stocks):
        ax = axes[idx]
        weights = portfolios_df[stock]
        non_zero_weights = weights[weights > 0]
        
        # Histogram of non-zero weights
        ax.hist(non_zero_weights, bins=50, color=STOCK_COLORS[idx], alpha=0.7, edgecolor='black')
        ax.axvline(non_zero_weights.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {non_zero_weights.mean():.3f}')
        ax.axvline(non_zero_weights.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {non_zero_weights.median():.3f}')
        
        ax.set_xlabel('Weight', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{stock}\n(Non-zero: {len(non_zero_weights):,} portfolios)', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Weight Distribution per Stock (Non-Zero Weights Only)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / "2_weight_distribution_per_stock.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_weight_statistics_comparison(portfolios_df, output_dir):
    """Plot comparison of weight statistics across all stocks."""
    stats_data = []
    for stock in portfolios_df.columns:
        weights = portfolios_df[stock]
        non_zero = weights[weights > 0]
        stats_data.append({
            'Stock': stock,
            'Mean Weight': weights.mean(),
            'Non-Zero Mean': non_zero.mean() if len(non_zero) > 0 else 0,
            'Std Dev': weights.std(),
            'Non-Zero %': (weights > 0).sum() / len(weights) * 100
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Mean weights
    ax1 = axes[0, 0]
    bars1 = ax1.barh(stats_df['Stock'], stats_df['Mean Weight'], color=STOCK_COLORS, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Mean Weight', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Weight Across All Portfolios', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars1, stats_df['Mean Weight'])):
        ax1.text(val, i, f' {val:.3f}', va='center', fontsize=9)
    
    # Non-zero mean weights
    ax2 = axes[0, 1]
    bars2 = ax2.barh(stats_df['Stock'], stats_df['Non-Zero Mean'], color=STOCK_COLORS, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Mean Weight (Non-Zero Only)', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Weight for Portfolios Including Each Stock', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars2, stats_df['Non-Zero Mean'])):
        ax2.text(val, i, f' {val:.3f}', va='center', fontsize=9)
    
    # Standard deviation
    ax3 = axes[1, 0]
    bars3 = ax3.barh(stats_df['Stock'], stats_df['Std Dev'], color=STOCK_COLORS, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax3.set_title('Weight Standard Deviation', fontsize=13, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars3, stats_df['Std Dev'])):
        ax3.text(val, i, f' {val:.3f}', va='center', fontsize=9)
    
    # Non-zero percentage
    ax4 = axes[1, 1]
    bars4 = ax4.barh(stats_df['Stock'], stats_df['Non-Zero %'], color=STOCK_COLORS, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Percentage of Portfolios Including Each Stock', fontsize=13, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars4, stats_df['Non-Zero %'])):
        ax4.text(val, i, f' {val:.1f}%', va='center', fontsize=9)
    
    plt.suptitle('Weight Statistics Comparison Across All Stocks', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = output_dir / "3_weight_statistics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_sample_portfolios(portfolios_df, output_dir, num_samples=6):
    """Plot pie charts for sample portfolios."""
    # Select diverse sample portfolios
    portfolio_sizes = portfolios_df.astype(bool).sum(axis=1)
    
    # Get samples of different sizes with their indices
    samples = []
    sample_indices = []
    for size in range(2, 11):
        size_portfolios = portfolios_df[portfolio_sizes == size]
        if len(size_portfolios) > 0:
            idx = size_portfolios.index[0]
            samples.append(size_portfolios.iloc[0])
            sample_indices.append(idx)
            if len(samples) >= num_samples:
                break
    
    # Fill remaining with random samples if needed
    while len(samples) < num_samples:
        rand_idx = np.random.randint(0, len(portfolios_df))
        samples.append(portfolios_df.iloc[rand_idx])
        sample_indices.append(portfolios_df.index[rand_idx])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (portfolio, orig_idx) in enumerate(zip(samples[:num_samples], sample_indices[:num_samples])):
        ax = axes[idx]
        
        # Get non-zero weights
        non_zero = portfolio[portfolio > 0]
        labels = [f"{stock}\n({weight:.1%})" for stock, weight in non_zero.items()]
        
        # Create pie chart
        colors = [STOCK_COLORS[list(portfolios_df.columns).index(stock)] for stock in non_zero.index]
        wedges, texts, autotexts = ax.pie(non_zero.values, labels=labels, autopct='',
                                         colors=colors, startangle=90)
        
        # Make labels more readable
        for text in texts:
            text.set_fontsize(9)
            text.set_fontweight('bold')
        
        portfolio_size = (portfolio > 0).sum()
        ax.set_title(f'Portfolio #{orig_idx}\n'
                    f'Size: {int(portfolio_size)} stocks', 
                    fontsize=12, fontweight='bold')
    
    plt.suptitle('Sample Portfolio Compositions', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = output_dir / "4_sample_portfolio_compositions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_weight_heatmap(portfolios_df, output_dir, sample_size=1000):
    """Plot heatmap showing weight patterns across portfolios."""
    # Sample portfolios for visualization (too many to show all)
    if len(portfolios_df) > sample_size:
        sample_indices = np.random.choice(len(portfolios_df), sample_size, replace=False)
        sample_df = portfolios_df.iloc[sample_indices].copy()
        print(f"Sampling {sample_size} portfolios for heatmap visualization")
    else:
        sample_df = portfolios_df.copy()
    
    # Sort by portfolio size for better visualization
    portfolio_sizes = sample_df.astype(bool).sum(axis=1)
    # Reset index to ensure proper sorting
    sample_df = sample_df.reset_index(drop=True)
    portfolio_sizes = sample_df.astype(bool).sum(axis=1)
    sample_df = sample_df.iloc[portfolio_sizes.sort_values().index].reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(sample_df) / 50)))
    
    # Create heatmap
    sns.heatmap(sample_df.T, cmap='YlOrRd', cbar_kws={'label': 'Weight'}, 
                ax=ax, vmin=0, vmax=1, linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Portfolio Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stock', fontsize=12, fontweight='bold')
    ax.set_title(f'Portfolio Weight Heatmap (Sample of {len(sample_df):,} portfolios)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "5_portfolio_weight_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_weight_boxplot(portfolios_df, output_dir):
    """Plot boxplots showing weight distribution for each stock."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for boxplot (only non-zero weights)
    plot_data = []
    plot_labels = []
    for stock in portfolios_df.columns:
        non_zero = portfolios_df[portfolios_df[stock] > 0][stock]
        if len(non_zero) > 0:
            plot_data.append(non_zero.values)
            plot_labels.append(stock)
    
    # Create boxplot
    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True, 
                   showmeans=True, meanline=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], STOCK_COLORS[:len(plot_labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Stock', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight (Non-Zero Only)', fontsize=12, fontweight='bold')
    ax.set_title('Weight Distribution Boxplot per Stock', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = output_dir / "6_weight_boxplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_overall_weight_distribution(portfolios_df, output_dir):
    """Plot overall weight distribution across all portfolios."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Flatten all weights
    all_weights = portfolios_df.values.flatten()
    non_zero_weights = all_weights[all_weights > 0]
    
    # Histogram of all weights (including zeros)
    ax1 = axes[0]
    ax1.hist(all_weights, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(all_weights.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {all_weights.mean():.3f}')
    ax1.axvline(np.median(all_weights), color='green', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(all_weights):.3f}')
    ax1.set_xlabel('Weight', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Weight Distribution (Including Zeros)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Histogram of non-zero weights only
    ax2 = axes[1]
    ax2.hist(non_zero_weights, bins=100, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(non_zero_weights.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {non_zero_weights.mean():.3f}')
    ax2.axvline(np.median(non_zero_weights), color='green', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(non_zero_weights):.3f}')
    ax2.set_xlabel('Weight', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Non-Zero Weight Distribution\n({len(non_zero_weights):,} non-zero weights)', 
                 fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle('Overall Weight Distribution Across All Portfolios', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / "7_overall_weight_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_stock_cooccurrence(portfolios_df, output_dir):
    """Plot co-occurrence matrix showing which stocks appear together in portfolios."""
    # Create binary matrix (1 if stock is in portfolio, 0 otherwise)
    binary_df = (portfolios_df > 0).astype(int)
    
    # Calculate co-occurrence matrix
    cooccurrence = binary_df.T @ binary_df
    
    # Normalize by number of portfolios
    cooccurrence_pct = (cooccurrence / len(portfolios_df)) * 100
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(cooccurrence_pct, annot=True, fmt='.1f', cmap='Blues', 
               cbar_kws={'label': 'Co-occurrence Percentage (%)'}, 
               square=True, ax=ax, linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Stock', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stock', fontsize=12, fontweight='bold')
    ax.set_title('Stock Co-occurrence Matrix\n(Percentage of portfolios containing both stocks)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "8_stock_cooccurrence_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main function to generate all visualizations."""
    print("=" * 60)
    print("Portfolio Visualization Generator")
    print("=" * 60)
    
    # Load data
    portfolios_df, metadata = load_portfolio_data()
    
    # Create output directory
    output_dir = Path(__file__).parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")
    
    # Generate all visualizations
    print("Generating visualizations...")
    print("-" * 60)
    
    plot_portfolio_size_distribution(portfolios_df, metadata, output_dir)
    plot_weight_distribution_per_stock(portfolios_df, output_dir)
    plot_weight_statistics_comparison(portfolios_df, output_dir)
    plot_sample_portfolios(portfolios_df, output_dir)
    plot_weight_heatmap(portfolios_df, output_dir)
    plot_weight_boxplot(portfolios_df, output_dir)
    plot_overall_weight_distribution(portfolios_df, output_dir)
    plot_stock_cooccurrence(portfolios_df, output_dir)
    
    print("-" * 60)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated visualizations:")
    print("  1. Portfolio Size Distribution")
    print("  2. Weight Distribution per Stock")
    print("  3. Weight Statistics Comparison")
    print("  4. Sample Portfolio Compositions")
    print("  5. Portfolio Weight Heatmap")
    print("  6. Weight Boxplot")
    print("  7. Overall Weight Distribution")
    print("  8. Stock Co-occurrence Matrix")
    print("=" * 60)


if __name__ == "__main__":
    main()

