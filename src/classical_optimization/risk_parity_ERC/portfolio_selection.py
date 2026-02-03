"""
Portfolio selection protocol for statistically representative sampling.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging

try:
    from scipy.stats import spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


def compute_equal_weight_volatility(returns: pd.DataFrame) -> float:
    """Compute equal-weight portfolio volatility."""
    n = len(returns.columns)
    if n == 0:
        return np.nan
    equal_weights = np.ones(n) / n
    portfolio_returns = (returns * equal_weights).sum(axis=1)
    return portfolio_returns.std() * np.sqrt(252)


def compute_equal_weight_hhi(assets: List[str]) -> float:
    """Compute HHI for equal-weight portfolio."""
    n = len(assets)
    if n == 0:
        return np.nan
    weights = np.ones(n) / n
    return np.sum(weights**2)


def stratify_portfolios(
    baseline_portfolios: pd.DataFrame,
    returns: pd.DataFrame,
    stratification_config: Dict
) -> pd.DataFrame:
    """
    Add stratification dimensions to portfolios DataFrame.
    
    Args:
        baseline_portfolios: DataFrame with portfolio information
        returns: Returns DataFrame for volatility computation
        stratification_config: Configuration for stratification dimensions
        
    Returns:
        DataFrame with added stratification columns
    """
    portfolios = baseline_portfolios.copy()
    
    # Num assets dimension
    if 'num_assets' in stratification_config:
        if isinstance(portfolios.iloc[0], pd.Series):
            portfolios['num_assets'] = portfolios.apply(lambda row: len([a for a in row.index if pd.notna(row[a]) and row[a] != 0]), axis=1)
        else:
            portfolios['num_assets'] = portfolios.apply(lambda row: len(row) if isinstance(row, (list, tuple)) else 1, axis=1)
    
    # Ex-ante volatility dimension
    if 'ex_ante_volatility' in stratification_config:
        logger.info("Computing ex-ante volatility for stratification...")
        volatilities = []
        for idx in portfolios.index:
            portfolio_assets = portfolios.loc[idx]
            if isinstance(portfolio_assets, pd.Series):
                assets = [a for a in portfolio_assets.index if a in returns.columns and pd.notna(portfolio_assets[a]) and portfolio_assets[a] != 0]
            else:
                assets = [a for a in portfolio_assets if a in returns.columns]
            
            if len(assets) > 0:
                returns_subset = returns[assets]
                vol = compute_equal_weight_volatility(returns_subset)
                volatilities.append(vol)
            else:
                volatilities.append(np.nan)
        
        portfolios['ex_ante_volatility'] = volatilities
        
        # Create bins
        vol_config = stratification_config['ex_ante_volatility']
        quantiles = vol_config.get('quantiles', [0.33, 0.66])
        valid_vols = portfolios['ex_ante_volatility'].dropna()
        if len(valid_vols) > 0:
            q33 = valid_vols.quantile(quantiles[0])
            q66 = valid_vols.quantile(quantiles[1])
            
            def assign_vol_bin(vol):
                if pd.isna(vol):
                    return 'unknown'
                elif vol <= q33:
                    return 'low'
                elif vol <= q66:
                    return 'medium'
                else:
                    return 'high'
            
            portfolios['volatility_bin'] = portfolios['ex_ante_volatility'].apply(assign_vol_bin)
    
    # Concentration proxy dimension
    if 'concentration_proxy' in stratification_config:
        logger.info("Computing concentration proxy for stratification...")
        hhis = []
        for idx in portfolios.index:
            portfolio_assets = portfolios.loc[idx]
            if isinstance(portfolio_assets, pd.Series):
                assets = [a for a in portfolio_assets.index if a in returns.columns and pd.notna(portfolio_assets[a]) and portfolio_assets[a] != 0]
            else:
                assets = [a for a in portfolio_assets if a in returns.columns]
            
            hhi = compute_equal_weight_hhi(assets)
            hhis.append(hhi)
        
        portfolios['concentration_hhi'] = hhis
        
        # Create bins
        conc_config = stratification_config['concentration_proxy']
        quantiles = conc_config.get('quantiles', [0.33, 0.66])
        valid_hhis = portfolios['concentration_hhi'].dropna()
        if len(valid_hhis) > 0:
            q33 = valid_hhis.quantile(quantiles[0])
            q66 = valid_hhis.quantile(quantiles[1])
            
            def assign_conc_bin(hhi):
                if pd.isna(hhi):
                    return 'unknown'
                elif hhi <= q33:
                    return 'diversified'
                elif hhi <= q66:
                    return 'moderate'
                else:
                    return 'concentrated'
            
            portfolios['concentration_bin'] = portfolios['concentration_hhi'].apply(assign_conc_bin)
    
    return portfolios


def stratified_random_sample(
    portfolios: pd.DataFrame,
    sample_size: int,
    stratification_config: Dict,
    random_seed: int = 42,
    without_replacement: bool = True
) -> pd.DataFrame:
    """
    Perform stratified random sampling.
    
    Args:
        portfolios: DataFrame with stratification columns
        sample_size: Target sample size
        stratification_config: Stratification configuration
        random_seed: Random seed
        without_replacement: Whether to sample without replacement
        
    Returns:
        Sampled portfolios DataFrame
    """
    np.random.seed(random_seed)
    
    # Get stratification dimensions
    num_assets_config = stratification_config.get('num_assets', {})
    volatility_config = stratification_config.get('ex_ante_volatility', {})
    concentration_config = stratification_config.get('concentration_proxy', {})
    
    # Create stratification groups
    portfolios['stratum'] = portfolios.index.astype(str)
    
    if 'num_assets' in portfolios.columns and num_assets_config.get('bins'):
        bins = num_assets_config['bins']
        min_per_bin = num_assets_config.get('ensure_min_per_bin', 0)
        
        def assign_num_assets_bin(num):
            for i, bin_max in enumerate(bins):
                if num <= bin_max:
                    return f"assets_{bin_max}"
            return f"assets_{bins[-1]}_plus"
        
        portfolios['num_assets_bin'] = portfolios['num_assets'].apply(assign_num_assets_bin)
        portfolios['stratum'] = portfolios['stratum'] + '_' + portfolios['num_assets_bin']
    
    if 'volatility_bin' in portfolios.columns:
        portfolios['stratum'] = portfolios['stratum'] + '_' + portfolios['volatility_bin']
    
    if 'concentration_bin' in portfolios.columns:
        portfolios['stratum'] = portfolios['stratum'] + '_' + portfolios['concentration_bin']
    
    # Sample from each stratum
    sampled = []
    stratum_counts = portfolios['stratum'].value_counts()
    
    # Proportional allocation
    total_available = len(portfolios)
    for stratum, count in stratum_counts.items():
        stratum_size = int(sample_size * count / total_available)
        stratum_data = portfolios[portfolios['stratum'] == stratum]
        
        if len(stratum_data) > 0:
            n_sample = min(stratum_size, len(stratum_data))
            if n_sample > 0:
                sampled_stratum = stratum_data.sample(n=n_sample, random_state=random_seed, replace=not without_replacement)
                sampled.append(sampled_stratum)
    
    # If we don't have enough, fill randomly
    sampled_df = pd.concat(sampled) if sampled else pd.DataFrame()
    if len(sampled_df) < sample_size:
        remaining = sample_size - len(sampled_df)
        remaining_portfolios = portfolios[~portfolios.index.isin(sampled_df.index)]
        if len(remaining_portfolios) > 0:
            additional = remaining_portfolios.sample(n=min(remaining, len(remaining_portfolios)), random_state=random_seed)
            sampled_df = pd.concat([sampled_df, additional])
    
    # Ensure minimum per bin for num_assets
    if 'num_assets_bin' in portfolios.columns and min_per_bin > 0:
        for bin_name in portfolios['num_assets_bin'].unique():
            bin_in_sample = sampled_df[sampled_df['num_assets_bin'] == bin_name]
            if len(bin_in_sample) < min_per_bin:
                needed = min_per_bin - len(bin_in_sample)
                bin_available = portfolios[(portfolios['num_assets_bin'] == bin_name) & (~portfolios.index.isin(sampled_df.index))]
                if len(bin_available) > 0:
                    additional = bin_available.sample(n=min(needed, len(bin_available)), random_state=random_seed)
                    sampled_df = pd.concat([sampled_df, additional])
    
    return sampled_df.head(sample_size)


def select_portfolios(
    baseline_portfolios: pd.DataFrame,
    returns: pd.DataFrame,
    selection_config: Dict
) -> Dict[str, pd.DataFrame]:
    """
    Select portfolios according to the selection protocol.
    
    Args:
        baseline_portfolios: All available portfolios
        returns: Returns DataFrame for stratification
        selection_config: Portfolio selection configuration
        
    Returns:
        Dictionary with selected portfolio DataFrames for each sample
    """
    logger.info("Starting portfolio selection protocol...")
    
    # Stratify portfolios
    main_sample_config = selection_config.get('main_experiment_sample', {})
    stratification = main_sample_config.get('stratification_dimensions', {})
    
    portfolios_stratified = stratify_portfolios(baseline_portfolios, returns, stratification)
    
    results = {}
    
    # Main experiment sample
    if main_sample_config.get('enable', False):
        sample_size = main_sample_config.get('sample_size', 3000)
        random_seed = main_sample_config.get('sampling_constraints', {}).get('random_seed', 42)
        without_replacement = main_sample_config.get('sampling_constraints', {}).get('without_replacement', True)
        
        logger.info(f"Selecting main experiment sample: {sample_size} portfolios")
        main_sample = stratified_random_sample(
            portfolios_stratified,
            sample_size,
            stratification,
            random_seed=random_seed,
            without_replacement=without_replacement
        )
        results['main_sample'] = main_sample
        logger.info(f"Selected {len(main_sample)} portfolios for main experiment")
    
    # Robustness sample (subset of main)
    robustness_config = selection_config.get('robustness_and_sensitivity_sample', {})
    if robustness_config.get('enable', False) and 'main_sample' in results:
        sample_size = robustness_config.get('sample_size', 500)
        random_seed = robustness_config.get('random_seed', 123)
        
        logger.info(f"Selecting robustness sample: {sample_size} portfolios from main sample")
        main_sample = results['main_sample']
        
        # Additional coverage constraints
        additional_constraints = robustness_config.get('additional_coverage_constraints', {}).get('ensure_extreme_cases', {})
        
        robustness_sample = main_sample.sample(n=min(sample_size, len(main_sample)), random_state=random_seed)
        
        # Ensure extreme cases if specified
        if additional_constraints:
            if 'volatility_bin' in main_sample.columns:
                low_vol = main_sample[main_sample['volatility_bin'] == 'low']
                high_vol = main_sample[main_sample['volatility_bin'] == 'high']
                high_conc = main_sample[main_sample.get('concentration_bin', '') == 'concentrated'] if 'concentration_bin' in main_sample.columns else pd.DataFrame()
                
                min_low_vol = additional_constraints.get('min_low_volatility', 0)
                min_high_vol = additional_constraints.get('min_high_volatility', 0)
                min_high_conc = additional_constraints.get('min_high_concentration', 0)
                
                # Add extreme cases
                if len(low_vol) > 0 and min_low_vol > 0:
                    needed = min_low_vol - len(robustness_sample[robustness_sample['volatility_bin'] == 'low'])
                    if needed > 0:
                        additional = low_vol[~low_vol.index.isin(robustness_sample.index)].head(needed)
                        robustness_sample = pd.concat([robustness_sample, additional])
                
                if len(high_vol) > 0 and min_high_vol > 0:
                    needed = min_high_vol - len(robustness_sample[robustness_sample['volatility_bin'] == 'high'])
                    if needed > 0:
                        additional = high_vol[~high_vol.index.isin(robustness_sample.index)].head(needed)
                        robustness_sample = pd.concat([robustness_sample, additional])
                
                if len(high_conc) > 0 and min_high_conc > 0:
                    needed = min_high_conc - len(robustness_sample[robustness_sample.get('concentration_bin', '') == 'concentrated'])
                    if needed > 0:
                        additional = high_conc[~high_conc.index.isin(robustness_sample.index)].head(needed)
                        robustness_sample = pd.concat([robustness_sample, additional])
        
        results['robustness_sample'] = robustness_sample.head(sample_size)
        logger.info(f"Selected {len(results['robustness_sample'])} portfolios for robustness sample")
    
    # Convergence validation sample (simple random)
    convergence_config = selection_config.get('convergence_validation_sample', {})
    if convergence_config.get('enable', False):
        sample_size = convergence_config.get('sample_size', 10000)
        random_seed = convergence_config.get('random_seed', 2026)
        
        logger.info(f"Selecting convergence validation sample: {sample_size} portfolios (simple random)")
        convergence_sample = portfolios_stratified.sample(n=min(sample_size, len(portfolios_stratified)), random_state=random_seed)
        results['convergence_sample'] = convergence_sample
        logger.info(f"Selected {len(convergence_sample)} portfolios for convergence validation")
    
    return results


def check_convergence(
    metrics_df: pd.DataFrame,
    convergence_config: Dict,
    metrics_tracked: List[str]
) -> Dict:
    """
    Check convergence of metrics using stopping rule validation.
    
    Args:
        metrics_df: DataFrame with computed metrics
        convergence_config: Convergence validation configuration
        metrics_tracked: List of metrics to track
        
    Returns:
        Dictionary with convergence status for each metric
    """
    stopping_rule = convergence_config.get('stopping_rule_validation', {})
    if not stopping_rule.get('enable', False):
        return {}
    
    convergence_criteria = stopping_rule.get('convergence_criteria', {})
    results = {}
    
    # Sort by portfolio_id to ensure consistent ordering
    metrics_df = metrics_df.sort_index() if 'portfolio_id' in metrics_df.columns else metrics_df
    
    for metric in metrics_tracked:
        if metric not in metrics_df.columns:
            continue
        
        metric_values = metrics_df[metric].dropna().values
        
        if len(metric_values) == 0:
            continue
        
        converged = True
        reasons = []
        
        # Mean stability check
        mean_stability = convergence_criteria.get('mean_stability', {})
        if mean_stability:
            window_size = mean_stability.get('window_size', 500)
            relative_tol = mean_stability.get('relative_tolerance', 0.02)
            
            if len(metric_values) >= window_size * 2:
                # Compare means of consecutive windows
                window1_mean = np.mean(metric_values[-window_size*2:-window_size])
                window2_mean = np.mean(metric_values[-window_size:])
                
                if window1_mean != 0:
                    relative_change = abs((window2_mean - window1_mean) / window1_mean)
                    if relative_change > relative_tol:
                        converged = False
                        reasons.append(f"mean_stability: {relative_change:.4f} > {relative_tol}")
        
        # Confidence interval width check
        ci_width = convergence_criteria.get('confidence_interval_width', {})
        if ci_width and converged:
            max_width_frac = ci_width.get('max_width_fraction', 0.1)
            ci_level = ci_width.get('confidence_level', 0.95)
            
            if len(metric_values) >= 100:
                # Bootstrap CI
                n_bootstrap = 1000
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    sample = np.random.choice(metric_values, size=len(metric_values), replace=True)
                    bootstrap_means.append(np.mean(sample))
                
                alpha = 1 - ci_level
                lower = np.percentile(bootstrap_means, 100 * alpha / 2)
                upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
                width = upper - lower
                mean_val = np.mean(metric_values)
                
                if mean_val != 0:
                    width_frac = width / abs(mean_val)
                    if width_frac > max_width_frac:
                        converged = False
                        reasons.append(f"ci_width: {width_frac:.4f} > {max_width_frac}")
        
        # Rank stability check
        rank_stability = convergence_criteria.get('rank_stability', {})
        if rank_stability and converged:
            threshold = rank_stability.get('threshold', 0.95)
            window_size = rank_stability.get('window_size', 500)
            
            if len(metric_values) >= window_size * 2:
                # Compare ranks of consecutive windows
                window1_values = metric_values[-window_size*2:-window_size]
                window2_values = metric_values[-window_size:]
                
                # Get ranks
                window1_ranks = pd.Series(window1_values).rank()
                window2_ranks = pd.Series(window2_values).rank()
                
                # Spearman correlation
                if len(window1_ranks) == len(window2_ranks) and SCIPY_AVAILABLE:
                    corr, _ = spearmanr(window1_ranks, window2_ranks)
                    if corr < threshold:
                        converged = False
                        reasons.append(f"rank_stability: {corr:.4f} < {threshold}")
                elif not SCIPY_AVAILABLE:
                    # Fallback: use Pearson correlation on ranks
                    corr = np.corrcoef(window1_ranks, window2_ranks)[0, 1]
                    if not np.isnan(corr) and corr < threshold:
                        converged = False
                        reasons.append(f"rank_stability: {corr:.4f} < {threshold}")
        
        results[metric] = {
            'converged': converged,
            'reasons': reasons,
            'final_mean': float(np.mean(metric_values)),
            'final_std': float(np.std(metric_values)),
            'n_samples': len(metric_values)
        }
    
    return results


def save_selection_metadata(
    selected_portfolios: Dict[str, pd.DataFrame],
    selection_config: Dict,
    output_config: Dict
):
    """Save selection metadata and portfolio IDs."""
    output_base = Path(output_config.get('output_files', {}).get('main_sample_ids', 'results/selection')).parent
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Save portfolio IDs
    if 'main_sample' in selected_portfolios:
        main_ids = pd.DataFrame({'portfolio_id': selected_portfolios['main_sample'].index})
        main_ids.to_parquet(output_config['output_files']['main_sample_ids'])
        logger.info(f"Saved main sample IDs to {output_config['output_files']['main_sample_ids']}")
    
    if 'robustness_sample' in selected_portfolios:
        robustness_ids = pd.DataFrame({'portfolio_id': selected_portfolios['robustness_sample'].index})
        robustness_ids.to_parquet(output_config['output_files']['robustness_sample_ids'])
        logger.info(f"Saved robustness sample IDs to {output_config['output_files']['robustness_sample_ids']}")
    
    if 'convergence_sample' in selected_portfolios:
        convergence_ids = pd.DataFrame({'portfolio_id': selected_portfolios['convergence_sample'].index})
        convergence_ids.to_parquet(output_config['output_files']['convergence_sample_ids'])
        logger.info(f"Saved convergence sample IDs to {output_config['output_files']['convergence_sample_ids']}")
    
    # Save summary JSON
    summary = {
        'main_sample_size': len(selected_portfolios.get('main_sample', pd.DataFrame())),
        'robustness_sample_size': len(selected_portfolios.get('robustness_sample', pd.DataFrame())),
        'convergence_sample_size': len(selected_portfolios.get('convergence_sample', pd.DataFrame())),
        'selection_config': selection_config
    }
    
    summary_path = output_config['output_files']['selection_summary_json']
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved selection summary to {summary_path}")

