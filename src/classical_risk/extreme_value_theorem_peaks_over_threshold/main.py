"""
Main evaluation script for EVT-POT VaR and CVaR.

Orchestrates the entire VaR/CVaR evaluation pipeline with asset-level EVT fitting:
- Data loading
- Asset-level EVT parameter estimation (with caching)
- Portfolio tail projection
- Backtesting
- Metrics computation
- Report generation
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

# Suppress RuntimeWarning about module import in multiprocessing workers
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*found in sys.modules.*')

from .returns import (
    load_panel_prices,
    load_portfolio_weights,
    compute_daily_returns,
    compute_portfolio_returns
)
from .evt_calculator import (
    compute_all_asset_evt_parameters,
    EVTParameterCache,
    compute_var_from_evt,
    compute_cvar_from_evt
)
from .portfolio_projection import (
    project_portfolio_var_cvar,
    compute_rolling_portfolio_var_cvar
)
from .backtesting import compute_accuracy_metrics, detect_cvar_violations
from .metrics import (
    compute_tail_metrics,
    compute_cvar_tail_metrics,
    compute_evt_tail_metrics,
    compute_structure_metrics,
    compute_distribution_metrics,
    compute_runtime_metrics
)
from .report_generator import generate_report
from .time_sliced_metrics import compute_time_sliced_metrics


def _restructure_results_by_portfolio(
    all_results: List[Dict],
    aligned_data_dict: Dict[int, Dict],
    prices: pd.DataFrame
) -> List[Dict]:
    """
    Restructure results grouped by portfolio.
    
    Args:
        all_results: List of flat result dictionaries
        aligned_data_dict: Dictionary mapping portfolio_id to aligned data
        prices: Original price DataFrame for date range
        
    Returns:
        List of portfolio result dictionaries with nested structure
    """
    # Group results by portfolio_id
    portfolios_dict = {}
    
    for result in all_results:
        portfolio_id = result['portfolio_id']
        
        if portfolio_id not in portfolios_dict:
            portfolios_dict[portfolio_id] = {
                'portfolio_id': portfolio_id,
                'structure': {
                    'portfolio_size': result.get('portfolio_size', result.get('num_active_assets', 0)),
                    'num_active_assets': result.get('num_active_assets', result.get('portfolio_size', 0)),
                    'hhi': result.get('hhi_concentration', np.nan),
                    'effective_assets': result.get('effective_number_of_assets', np.nan),
                    'covariance_condition_number': result.get('covariance_condition_number', np.nan)
                },
                'distribution': {
                    'skewness': result.get('skewness', np.nan),
                    'kurtosis': result.get('kurtosis', np.nan),
                    'jarque_bera_p_value': result.get('jarque_bera_p_value', np.nan),
                    'jarque_bera_statistic': result.get('jarque_bera_statistic', np.nan)
                },
                'var_evaluations': []
            }
        
        # Create VaR/CVaR evaluation entry
        var_eval = {
            'confidence_level': result['confidence_level'],
            'horizon': result['horizon'],
            'estimation_window': result['estimation_window'],
            'global_metrics': {
                'hit_rate': result.get('hit_rate', np.nan),
                'num_violations': result.get('num_violations', 0),
                'expected_violations': result.get('expected_violations', np.nan),
                'violation_ratio': result.get('violation_ratio', np.nan),
                'accuracy_tests': {
                    'kupiec_p_value': result.get('kupiec_unconditional_coverage', np.nan),
                    'kupiec_statistic': result.get('kupiec_test_statistic', np.nan),
                    'kupiec_reject_null': result.get('kupiec_reject_null', False),
                    'christoffersen_independence_p': result.get('christoffersen_independence', np.nan),
                    'christoffersen_independence_statistic': result.get('christoffersen_independence_statistic', np.nan),
                    'christoffersen_independence_reject': result.get('christoffersen_independence_reject_null', False),
                    'christoffersen_cc_p': result.get('christoffersen_conditional_coverage', np.nan),
                    'christoffersen_cc_statistic': result.get('christoffersen_conditional_coverage_statistic', np.nan),
                    'christoffersen_cc_reject': result.get('christoffersen_conditional_coverage_reject_null', False),
                    'traffic_light_zone': result.get('traffic_light_zone', 'unknown')
                },
                'tail_metrics': {
                    'mean_exceedance': result.get('mean_exceedance', np.nan),
                    'max_exceedance': result.get('max_exceedance', np.nan),
                    'std_exceedance': result.get('std_exceedance', np.nan),
                    'quantile_loss_score': result.get('quantile_loss_score', np.nan),
                    'rmse_var_vs_losses': result.get('rmse_var_vs_losses', np.nan),
                    'rmse_cvar_vs_losses': result.get('rmse_cvar_vs_losses', np.nan),
                    'cvar_mean_exceedance': result.get('cvar_mean_exceedance', np.nan),
                    'cvar_max_exceedance': result.get('cvar_max_exceedance', np.nan),
                    'expected_shortfall_exceedance': result.get('expected_shortfall_exceedance', np.nan),
                    'tail_index_xi': result.get('tail_index_xi', np.nan),
                    'scale_beta': result.get('scale_beta', np.nan),
                    'shape_scale_stability': result.get('shape_scale_stability', np.nan)
                },
                'runtime': {
                    'runtime_ms': result.get('var_runtime_ms', np.nan),
                    'evt_fitting_time_ms': result.get('evt_fitting_time_ms', np.nan),
                    'threshold_selection_time_ms': result.get('threshold_selection_time_ms', np.nan),
                    'p95_runtime_ms': result.get('p95_runtime_ms', np.nan),
                    'median_runtime_ms': result.get('median_runtime_ms', np.nan),
                    'cache_hit_ratio': result.get('cache_hit_ratio', np.nan)
                }
            },
            'time_sliced_metrics': []
        }
        
        # Add time-sliced metrics if available
        if portfolio_id in aligned_data_dict:
            aligned_data = aligned_data_dict[portfolio_id]
            key = f"{result['confidence_level']}_{result['horizon']}_{result['estimation_window']}"
            if key in aligned_data:
                time_slices = compute_time_sliced_metrics(
                    aligned_data[key]['returns'],
                    aligned_data[key]['var'],
                    cvar_series=aligned_data[key].get('cvar'),
                    confidence_level=result['confidence_level'],
                    slice_by='year'
                )
                var_eval['time_sliced_metrics'] = time_slices
        
        portfolios_dict[portfolio_id]['var_evaluations'].append(var_eval)
    
    return list(portfolios_dict.values())


def _save_restructured_json(
    json_path: Path,
    portfolio_results: List[Dict],
    summary_stats: Dict,
    data_period: str,
    num_portfolios: int,
    confidence_levels: List[float],
    horizons: List[int],
    estimation_windows: List[int],
    evt_settings: Optional[Dict] = None
):
    """
    Save results in the restructured JSON format.
    
    Args:
        json_path: Path to save JSON file
        portfolio_results: List of restructured portfolio results
        summary_stats: Summary statistics dictionary
        data_period: Data period string
        num_portfolios: Number of portfolios evaluated
        confidence_levels: List of confidence levels
        horizons: List of horizons
        estimation_windows: List of estimation windows
        evt_settings: EVT model settings
    """
    # Convert NaN to None for JSON serialization
    def clean_nan(obj):
        if isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(item) for item in obj]
        elif isinstance(obj, (np.floating, float)) and np.isnan(obj):
            return None
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    output_data = {
        'metadata': {
            'task': 'evt_pot_var_cvar_evaluation_optimized',
            'data_period': data_period,
            'portfolios_evaluated': num_portfolios,
            'confidence_levels': confidence_levels,
            'horizons': horizons,
            'estimation_windows': estimation_windows,
            'evt_settings': evt_settings or {},
            'generated_at': datetime.now().isoformat()
        },
        'portfolio_results': clean_nan(portfolio_results),
        'summary': clean_nan(summary_stats)
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)


def _compute_summary_statistics(
    all_results: List[Dict],
    runtimes: List[float],
    cache_hit_ratio: float = 0.0
) -> Dict:
    """
    Compute summary statistics across all portfolios.
    
    Args:
        all_results: List of flat result dictionaries
        runtimes: List of runtime values
        cache_hit_ratio: Cache hit ratio
        
    Returns:
        Dictionary with summary statistics
    """
    if len(all_results) == 0:
        return {}
    
    results_df = pd.DataFrame(all_results)
    
    # Portfolio-level insights
    portfolio_insights = {}
    
    # Average violation ratios by confidence level
    for cl in [0.95, 0.99, 0.995]:
        cl_results = results_df[results_df['confidence_level'] == cl]
        if len(cl_results) > 0:
            portfolio_insights[f'avg_violation_ratio_{int(cl*100)}'] = float(
                cl_results['violation_ratio'].mean()
            )
    
    # Traffic light zones
    if 'traffic_light_zone' in results_df.columns:
        zone_counts = results_df['traffic_light_zone'].value_counts()
        total = len(results_df)
        portfolio_insights['percent_red_zone'] = float(zone_counts.get('red', 0) / total) if total > 0 else 0.0
        portfolio_insights['percent_yellow_zone'] = float(zone_counts.get('yellow', 0) / total) if total > 0 else 0.0
        portfolio_insights['percent_green_zone'] = float(zone_counts.get('green', 0) / total) if total > 0 else 0.0
    
    # Distribution effects
    distribution_effects = {}
    if 'skewness' in results_df.columns:
        distribution_effects['avg_skew'] = float(results_df['skewness'].mean())
    if 'kurtosis' in results_df.columns:
        distribution_effects['avg_kurtosis'] = float(results_df['kurtosis'].mean())
    if 'jarque_bera_p_value' in results_df.columns:
        rejection_rate = (results_df['jarque_bera_p_value'] < 0.05).mean()
        distribution_effects['normality_rejection_rate'] = float(rejection_rate)
    
    # Structure effects
    structure_effects = {}
    if 'hhi_concentration' in results_df.columns and 'violation_ratio' in results_df.columns:
        corr = results_df['hhi_concentration'].corr(results_df['violation_ratio'])
        if not np.isnan(corr):
            structure_effects['correlation_hhi_vs_violation_ratio'] = float(corr)
    
    # Runtime stats
    runtime_stats = {}
    if len(runtimes) > 0:
        runtime_array = np.array(runtimes) * 1000  # Convert to ms
        runtime_stats['mean_runtime_ms'] = float(np.mean(runtime_array))
        runtime_stats['p95_runtime_ms'] = float(np.percentile(runtime_array, 95))
        runtime_stats['median_runtime_ms'] = float(np.median(runtime_array))
    
    runtime_stats['cache_hit_ratio'] = float(cache_hit_ratio)
    
    # CVaR metrics summary
    cvar_insights = {}
    if 'cvar_mean_exceedance' in results_df.columns:
        cvar_insights['avg_cvar_mean_exceedance'] = float(results_df['cvar_mean_exceedance'].mean())
    if 'cvar_max_exceedance' in results_df.columns:
        cvar_insights['avg_cvar_max_exceedance'] = float(results_df['cvar_max_exceedance'].mean())
    
    # EVT-specific metrics
    evt_insights = {}
    if 'tail_index_xi' in results_df.columns:
        evt_insights['avg_tail_index_xi'] = float(results_df['tail_index_xi'].mean())
        evt_insights['std_tail_index_xi'] = float(results_df['tail_index_xi'].std())
    if 'expected_shortfall_exceedance' in results_df.columns:
        evt_insights['avg_expected_shortfall_exceedance'] = float(results_df['expected_shortfall_exceedance'].mean())
    
    return {
        'portfolio_level_insights': portfolio_insights,
        'distribution_effects': distribution_effects,
        'structure_effects': structure_effects,
        'runtime_stats': runtime_stats,
        'cvar_insights': cvar_insights,
        'evt_insights': evt_insights
    }


def _process_single_portfolio(
    portfolio_data: Tuple[int, Tuple, pd.Series],
    daily_returns: pd.DataFrame,
    asset_evt_parameters: Dict[Tuple[str, int, float], Dict[str, Any]],
    confidence_levels: List[float],
    horizons: List[int],
    estimation_windows: List[int],
    threshold_quantiles: list,
    scaling_rule: str,
    aggregation_method: str,
    compute_time_slices: bool = True
) -> Tuple[List[Dict], float, Dict]:
    """
    Process a single portfolio using asset-level EVT parameters.
    
    Args:
        portfolio_data: Tuple of (portfolio_idx, portfolio_id, portfolio_weights)
        daily_returns: DataFrame of daily returns
        asset_evt_parameters: Dictionary of asset-level EVT parameters
        confidence_levels: List of confidence levels
        horizons: List of horizons
        estimation_windows: List of estimation windows
        threshold_quantiles: List of threshold quantiles
        scaling_rule: Scaling rule for horizon
        aggregation_method: Aggregation method for portfolio projection
        compute_time_slices: Whether to compute time-sliced metrics
        
    Returns:
        Tuple of (list of result dictionaries, total runtime in seconds, 
                 dict with aligned returns/VaR/CVaR for time slicing)
    """
    portfolio_idx, portfolio_id, portfolio_weights = portfolio_data
    
    results = []
    runtimes = []
    aligned_data = {}
    
    try:
        # Compute portfolio returns for backtesting
        portfolio_returns = compute_portfolio_returns(
            daily_returns,
            portfolio_weights,
            align_assets=True
        )
    except Exception as e:
        return results, 0.0, {}
    
    # Compute covariance matrix for structure metrics (once per portfolio)
    try:
        common_assets = daily_returns.columns.intersection(portfolio_weights.index)
        returns_aligned = daily_returns[common_assets]
        covariance_matrix = returns_aligned.cov()
    except:
        covariance_matrix = None
    
    # Evaluate for each combination of settings
    for confidence_level in confidence_levels:
        for horizon in horizons:
            for window in estimation_windows:
                for quantile in threshold_quantiles:
                    try:
                        start_time = time.time()
                        
                        # Project portfolio VaR/CVaR from asset-level EVT parameters
                        portfolio_var, portfolio_cvar, proj_diagnostics = project_portfolio_var_cvar(
                            portfolio_weights,
                            asset_evt_parameters,
                            daily_returns,
                            window,
                            quantile,
                            confidence_level,
                            horizon,
                            scaling_rule,
                            aggregation_method
                        )
                        
                        runtime = time.time() - start_time
                        runtimes.append(runtime)
                        
                        if np.isnan(portfolio_var) or np.isnan(portfolio_cvar):
                            continue
                        
                        # Create VaR/CVaR series for backtesting
                        # For simplicity, use constant values (can be extended to rolling)
                        var_series = pd.Series(
                            [portfolio_var] * len(portfolio_returns),
                            index=portfolio_returns.index
                        )
                        cvar_series = pd.Series(
                            [portfolio_cvar] * len(portfolio_returns),
                            index=portfolio_returns.index
                        )
                        
                        # Align returns and VaR/CVaR
                        aligned_returns = portfolio_returns
                        aligned_var = var_series
                        aligned_cvar = cvar_series
                        
                        if len(aligned_returns) == 0:
                            continue
                        
                        # Store aligned data for time slicing
                        if compute_time_slices:
                            key = f"{confidence_level}_{horizon}_{window}_{quantile}"
                            aligned_data[key] = {
                                'returns': aligned_returns,
                                'var': aligned_var,
                                'cvar': aligned_cvar,
                                'confidence_level': confidence_level
                            }
                        
                        # Compute accuracy metrics
                        accuracy_metrics = compute_accuracy_metrics(
                            aligned_returns,
                            aligned_var,
                            confidence_level=confidence_level
                        )
                        
                        # Compute tail metrics for VaR
                        tail_metrics = compute_tail_metrics(
                            aligned_returns,
                            aligned_var,
                            confidence_level=confidence_level
                        )
                        
                        # Compute CVaR tail metrics
                        cvar_tail_metrics = compute_cvar_tail_metrics(
                            aligned_returns,
                            aligned_cvar,
                            aligned_var,
                            confidence_level=confidence_level
                        )
                        
                        # Compute EVT-specific metrics
                        # Get representative asset parameters for metrics
                        evt_tail_metrics = {
                            'expected_shortfall_exceedance': np.nan,
                            'tail_index_xi': np.nan,
                            'scale_beta': np.nan,
                            'shape_scale_stability': np.nan
                        }
                        
                        # Try to get average tail index from assets in portfolio
                        common_assets = portfolio_weights.index.intersection(daily_returns.columns)
                        xi_values = []
                        beta_values = []
                        for asset in common_assets:
                            key = (asset, window, quantile)
                            if key in asset_evt_parameters:
                                params = asset_evt_parameters[key]
                                if params.get('success', False):
                                    xi_values.append(params['xi'])
                                    beta_values.append(params['beta'])
                        
                        if len(xi_values) > 0:
                            evt_tail_metrics['tail_index_xi'] = np.mean(xi_values)
                            evt_tail_metrics['scale_beta'] = np.mean(beta_values)
                        
                        # Compute structure metrics
                        structure_metrics = compute_structure_metrics(
                            portfolio_weights,
                            covariance_matrix
                        )
                        
                        # Compute distribution metrics
                        distribution_metrics = compute_distribution_metrics(
                            aligned_returns
                        )
                        
                        # Combine all metrics
                        result = {
                            'portfolio_id': portfolio_id,
                            'confidence_level': confidence_level,
                            'horizon': horizon,
                            'estimation_window': window,
                            'threshold_quantile': quantile,
                            'var_runtime_ms': runtime * 1000,
                            'evt_fitting_time_ms': 0.0,  # Asset-level fitting done separately
                            'threshold_selection_time_ms': 0.0,  # Done during asset-level fitting
                            **accuracy_metrics,
                            **tail_metrics,
                            **cvar_tail_metrics,
                            **evt_tail_metrics,
                            **structure_metrics,
                            **distribution_metrics
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        continue
    
    # Return total runtime per portfolio
    total_runtime = sum(runtimes) if len(runtimes) > 0 else 0.0
    
    return results, total_runtime, aligned_data


def evaluate_evt_pot_var_cvar(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    n_jobs: Optional[int] = None,
    max_portfolios: Optional[int] = 100
) -> pd.DataFrame:
    """
    Main function to evaluate VaR and CVaR using EVT-POT methodology with asset-level fitting.
    
    This function implements the optimized workflow:
    1. Fit EVT parameters at asset level (once per asset/window/quantile)
    2. Reuse parameters across all portfolios
    3. Project portfolio VaR/CVaR from asset-level parameters
    
    Args:
        config_path: Path to JSON configuration file
        config_dict: Configuration dictionary (if not loading from file)
        n_jobs: Number of parallel workers (default: number of CPU cores)
        max_portfolios: Maximum number of portfolios to process (default: 100, None for all)
        
    Returns:
        DataFrame with all computed metrics
    """
    # Load configuration
    if config_dict is None:
        if config_path is None:
            config_path = Path(__file__).parent / "llm.json"
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = config_dict
    
    # Get paths (resolve relative to project root)
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    
    panel_price_path = project_root / config['inputs']['panel_price_path']
    portfolio_weights_path = project_root / config['inputs']['portfolio_weights_path']
    
    # Adjust path if it says "preprocessed" but file is in "processed"
    if not panel_price_path.exists() and 'preprocessed' in str(panel_price_path):
        panel_price_path = project_root / str(config['inputs']['panel_price_path']).replace('preprocessed', 'processed')
    
    print("=" * 80)
    print("EVT-POT (EXTREME VALUE THEORY - PEAKS OVER THRESHOLD) VAR/CVAR EVALUATION")
    print("Asset-Level EVT Fitting with Portfolio Projection")
    print("=" * 80)
    print(f"\nLoading data...")
    print(f"  Panel prices: {panel_price_path}")
    print(f"  Portfolio weights: {portfolio_weights_path}")
    
    # Load data
    prices = load_panel_prices(panel_price_path)
    portfolio_weights_df = load_portfolio_weights(portfolio_weights_path)
    
    # Get data period
    data_period = f"{prices.index.min().strftime('%Y-%m-%d')} to {prices.index.max().strftime('%Y-%m-%d')}"
    
    print(f"\nLoaded:")
    print(f"  Prices: {len(prices)} dates, {len(prices.columns)} assets")
    print(f"  Portfolios: {len(portfolio_weights_df)} portfolios")
    print(f"  Data period: {data_period}")
    
    # Compute daily returns
    evt_settings = config.get('evt_settings', {})
    return_type = evt_settings.get('return_type', 'log')
    print(f"\nComputing daily returns (method: {return_type})...")
    daily_returns = compute_daily_returns(prices, method=return_type)
    print(f"  Daily returns: {len(daily_returns)} dates")
    
    # Get EVT settings
    confidence_levels = evt_settings.get('confidence_levels', [0.99])
    horizons_config = evt_settings.get('horizons', {})
    base_horizon = horizons_config.get('base_horizon', 1)
    scaled_horizons = horizons_config.get('scaled_horizons', [10])
    horizons = [base_horizon] + scaled_horizons
    scaling_rule = horizons_config.get('scaling_rule', 'sqrt_time')
    estimation_windows = evt_settings.get('estimation_windows', [500])
    threshold_settings = evt_settings.get('threshold_selection', {})
    threshold_quantiles = threshold_settings.get('quantiles', [0.95])
    min_exceedances = threshold_settings.get('min_exceedances', 50)
    shape_constraints = evt_settings.get('shape_constraints', {})
    xi_lower = shape_constraints.get('xi_lower_bound', -0.5)
    xi_upper = shape_constraints.get('xi_upper_bound', 0.5)
    gpd_fitting_method = evt_settings.get('gpd_fitting_method', 'pwm')
    
    # Get computation strategy
    computation_strategy = config.get('computation_strategy', {})
    enable_cache = computation_strategy.get('enable_parameter_cache', True)
    portfolio_projection = config.get('design_principle', {}).get('portfolio_projection', True)
    aggregation_method = config.get('modules', {}).get('portfolio_tail_projection', {}).get('aggregation_method', 'weighted_tail_expectation')
    
    print(f"\nEVT-POT Settings:")
    print(f"  Method: {evt_settings.get('method', 'peaks_over_threshold')}")
    print(f"  Distribution: {evt_settings.get('distribution', 'generalized_pareto')}")
    print(f"  GPD Fitting Method: {gpd_fitting_method}")
    print(f"  Threshold quantiles: {threshold_quantiles}")
    print(f"  Minimum exceedances: {min_exceedances}")
    print(f"  Shape parameter bounds: [{xi_lower}, {xi_upper}]")
    print(f"  Confidence levels: {confidence_levels}")
    print(f"  Horizons: {horizons} days (scaling: {scaling_rule})")
    print(f"  Estimation windows: {estimation_windows} days")
    print(f"  Asset-level EVT fitting: Enabled")
    print(f"  Parameter caching: {'Enabled' if enable_cache else 'Disabled'}")
    print(f"  Portfolio projection: {'Enabled' if portfolio_projection else 'Disabled'}")
    
    # Initialize parameter cache
    cache = None
    if enable_cache:
        outputs = config.get('outputs', {})
        cache_path = outputs.get('evt_parameter_store')
        if cache_path:
            cache_path = project_root / cache_path
            cache = EVTParameterCache(cache_path)
            print(f"  Cache path: {cache_path}")
    
    # Step 1: Compute asset-level EVT parameters (once for all portfolios)
    print(f"\n{'='*80}")
    print("STEP 1: Computing Asset-Level EVT Parameters")
    print(f"{'='*80}")
    print(f"  This step fits EVT parameters for each asset/window/quantile combination")
    print(f"  Parameters will be reused across all portfolios for efficiency")
    
    start_evt_time = time.time()
    asset_evt_parameters = compute_all_asset_evt_parameters(
        daily_returns,
        estimation_windows,
        threshold_quantiles,
        min_exceedances,
        xi_lower,
        xi_upper,
        cache=cache
    )
    evt_fitting_time = time.time() - start_evt_time
    
    # Count successful fits
    successful_fits = sum(1 for p in asset_evt_parameters.values() if p.get('success', False))
    total_fits = len(asset_evt_parameters)
    
    print(f"\n  Asset-level EVT fitting completed:")
    print(f"    Total parameter sets: {total_fits}")
    print(f"    Successful fits: {successful_fits}")
    print(f"    Fitting time: {evt_fitting_time:.2f} seconds")
    if cache:
        print(f"    Cache hit ratio: {cache.get_hit_ratio():.2%}")
    
    # Save cache
    if cache:
        cache.save_cache()
    
    # Step 2: Process portfolios using asset-level parameters
    print(f"\n{'='*80}")
    print("STEP 2: Processing Portfolios with Portfolio Projection")
    print(f"{'='*80}")
    
    # Limit number of portfolios if specified
    num_portfolios_total = len(portfolio_weights_df)
    if max_portfolios is not None and max_portfolios > 0:
        num_portfolios = min(num_portfolios_total, max_portfolios)
        portfolio_weights_df = portfolio_weights_df.iloc[:num_portfolios]
        print(f"  Limiting to first {num_portfolios:,} portfolios (out of {num_portfolios_total:,} total)")
    else:
        num_portfolios = num_portfolios_total
        print(f"  Processing all {num_portfolios:,} portfolios")
    
    # Calculate total combinations for progress tracking
    total_combinations = num_portfolios * len(confidence_levels) * len(horizons) * len(estimation_windows) * len(threshold_quantiles)
    print(f"  Total portfolio-configuration combinations: {total_combinations:,}")
    
    # Initialize results
    all_results = []
    runtimes = []
    aligned_data_dict = {}
    
    # Process portfolios
    print(f"\n  Processing portfolios...")
    start_time_total = time.time()
    
    for portfolio_idx, (portfolio_id, portfolio_weights) in enumerate(portfolio_weights_df.iterrows()):
        if (portfolio_idx + 1) % 100 == 0 or (portfolio_idx + 1) in [1, 10, 50, 500, 1000]:
            print(f"  Processing portfolio {portfolio_idx + 1:,}/{num_portfolios:,} ({100*(portfolio_idx+1)/num_portfolios:.1f}%)...", flush=True)
        
        portfolio_data = (portfolio_idx, portfolio_id, portfolio_weights)
        results, runtime, aligned_data = _process_single_portfolio(
            portfolio_data,
            daily_returns,
            asset_evt_parameters,
            confidence_levels,
            horizons,
            estimation_windows,
            threshold_quantiles,
            scaling_rule,
            aggregation_method,
            compute_time_slices=True
        )
        
        all_results.extend(results)
        runtimes.append(runtime)
        if aligned_data:
            aligned_data_dict[portfolio_id] = aligned_data
    
    total_runtime = time.time() - start_time_total
    avg_runtime_per_portfolio = total_runtime / num_portfolios if num_portfolios > 0 else 0
    
    # Create results DataFrame
    if len(all_results) == 0:
        raise ValueError("No results computed. Check data and configuration.")
    
    results_df = pd.DataFrame(all_results)
    
    # Add runtime metrics
    runtime_metrics = compute_runtime_metrics(runtimes)
    for key, value in runtime_metrics.items():
        results_df[key] = value
    
    # Add cache hit ratio if available
    if cache:
        results_df['cache_hit_ratio'] = cache.get_hit_ratio()
    
    # Restructure results by portfolio
    portfolio_results = _restructure_results_by_portfolio(
        all_results,
        aligned_data_dict,
        prices
    )
    
    # Compute summary statistics
    cache_hit_ratio = cache.get_hit_ratio() if cache else 0.0
    summary_stats = _compute_summary_statistics(all_results, runtimes, cache_hit_ratio)
    
    print(f"\nCompleted evaluation of {len(results_df)} portfolio-configuration combinations")
    print(f"  Total runtime: {total_runtime/60:.2f} minutes ({total_runtime:.2f} seconds)")
    print(f"  EVT fitting time: {evt_fitting_time:.2f} seconds")
    print(f"  Portfolio processing time: {total_runtime:.2f} seconds")
    print(f"  Average runtime per portfolio: {avg_runtime_per_portfolio*1000:.2f} ms")
    if cache:
        print(f"  Cache hit ratio: {cache_hit_ratio:.2%}")
    
    # Save results
    outputs = config.get('outputs', {})
    
    if 'metrics_table' in outputs:
        metrics_path = project_root / outputs['metrics_table']
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving metrics table...")
        print(f"  Path: {metrics_path}")
        
        if metrics_path.suffix == '.parquet':
            results_df.to_parquet(metrics_path, index=False)
        elif metrics_path.suffix == '.csv':
            results_df.to_csv(metrics_path, index=False)
        else:
            metrics_path = metrics_path.with_suffix('.parquet')
            results_df.to_parquet(metrics_path, index=False)
        
        print(f"  Saved: {metrics_path}")
    
    # Also save JSON if specified separately
    if 'metrics_json' in outputs:
        json_path = project_root / outputs['metrics_json']
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving metrics JSON...")
        print(f"  Path: {json_path}")
        
        _save_restructured_json(
            json_path,
            portfolio_results,
            summary_stats,
            data_period,
            num_portfolios,
            confidence_levels,
            horizons,
            estimation_windows,
            evt_settings
        )
        
        print(f"  Saved: {json_path}")
    
    # Generate report
    if 'summary_report' in outputs:
        report_path = project_root / outputs['summary_report']
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating report...")
        print(f"  Path: {report_path}")
        
        generate_report(
            results_df,
            report_path,
            evt_settings=evt_settings,
            report_sections=config.get('report_sections')
        )
        
        print(f"  Saved: {report_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return results_df


def main():
    """Main entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate EVT-POT VaR/CVaR for portfolios (asset-level EVT fitting)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration JSON file (default: llm.json in same directory)'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=None,
        help='Number of parallel workers (not used in current implementation)'
    )
    parser.add_argument(
        '--max-portfolios',
        type=int,
        default=100,
        help='Maximum number of portfolios to process (default: 100, use 0 to process all)'
    )
    
    args = parser.parse_args()
    
    max_portfolios = args.max_portfolios
    if max_portfolios == 0:
        max_portfolios = None
    
    results_df = evaluate_evt_pot_var_cvar(
        config_path=args.config,
        n_jobs=args.n_jobs,
        max_portfolios=max_portfolios
    )
    
    print(f"\nResults summary:")
    print(f"  Total rows: {len(results_df)}")
    print(f"  Columns: {len(results_df.columns)}")
    print(f"\nFirst few rows:")
    print(results_df.head())


if __name__ == "__main__":
    main()
