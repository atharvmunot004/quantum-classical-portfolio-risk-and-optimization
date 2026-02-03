"""
Main evaluation script for Black-Litterman Portfolio Optimization.

Orchestrates the entire optimization pipeline including:
- Data loading (prices, baseline portfolios)
- Returns computation
- Market equilibrium returns (prior) from portfolios
- Synthetic view generation
- Posterior returns and covariance computation
- Portfolio optimization
- Efficient frontier generation
- Metrics computation
- Report generation
"""
import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import time
from datetime import datetime
import warnings
from multiprocessing import Pool, cpu_count
import os

# Suppress RuntimeWarning about sys.modules at module level
# This warning occurs when using python -m flag and is harmless
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*found in sys.modules.*')
# Suppress numpy correlation warnings for zero variance cases
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered in divide.*')

from .returns import (
    load_panel_prices,
    load_baseline_portfolios,
    load_investor_views,
    compute_daily_returns
)
from .black_litterman_optimizer import (
    compute_covariance_matrix,
    derive_market_equilibrium_returns,
    generate_synthetic_views,
    parse_and_scale_views,
    compute_posterior_bl_returns,
    compute_posterior_covariance,
    optimize_portfolio,
    generate_efficient_frontier
)
from .metrics import (
    compute_portfolio_statistics,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_information_ratio,
    compute_tracking_error,
    compute_alpha_vs_market,
    compute_bl_specific_metrics,
    compute_structure_metrics,
    compute_risk_metrics,
    compute_distribution_metrics,
    compute_comparison_metrics,
    compute_runtime_metrics
)
from .time_sliced_metrics import (
    compute_time_sliced_returns,
    compute_time_sliced_risk_metrics,
    compute_time_sliced_tail_metrics,
    compare_time_sliced_prior_vs_posterior,
    analyze_temporal_performance_stability
)
from .report_generator import generate_report, generate_metrics_schema
from .cache import BLCache, PrecomputeRegistry
from .metrics import compute_batch_metrics_vectorized
from .time_sliced_metrics import compute_batch_time_sliced_metrics

# Try to import GPU acceleration utilities
try:
    from .gpu_acceleration import (
        is_gpu_available, get_gpu_info, clear_gpu_cache,
        batch_portfolio_returns_matrix_multiply
    )
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    is_gpu_available = lambda: False
    get_gpu_info = lambda: {'available': False}
    clear_gpu_cache = lambda: None
    batch_portfolio_returns_matrix_multiply = None


def _save_restructured_json(
    json_path: Path,
    portfolio_results: List[Dict],
    summary_stats: Dict,
    data_period: str,
    num_portfolios: int,
    bl_settings: Dict
):
    """
    Save results in the restructured JSON format.
    
    Args:
        json_path: Path to save JSON file
        portfolio_results: List of restructured portfolio results
        summary_stats: Summary statistics dictionary
        data_period: Data period string
        num_portfolios: Number of portfolios optimized
        bl_settings: Black-Litterman optimization settings
    """
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
            'task': 'black_litterman_optimization',
            'data_period': data_period,
            'portfolios_optimized': num_portfolios,
            'black_litterman_settings': bl_settings,
            'generated_at': datetime.now().isoformat()
        },
        'portfolio_results': clean_nan(portfolio_results),
        'summary': clean_nan(summary_stats)
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)


def _compute_summary_statistics(
    all_results: List[Dict],
    runtimes: List[float]
) -> Dict:
    """
    Compute summary statistics across all portfolios.
    
    Args:
        all_results: List of flat result dictionaries
        runtimes: List of runtime values
        
    Returns:
        Dictionary with summary statistics
    """
    if len(all_results) == 0:
        return {}
    
    results_df = pd.DataFrame(all_results)
    
    portfolio_insights = {}
    
    if 'expected_return' in results_df.columns:
        portfolio_insights['avg_expected_return'] = float(results_df['expected_return'].mean())
        portfolio_insights['max_expected_return'] = float(results_df['expected_return'].max())
    
    if 'volatility' in results_df.columns:
        portfolio_insights['avg_volatility'] = float(results_df['volatility'].mean())
        portfolio_insights['min_volatility'] = float(results_df['volatility'].min())
    
    if 'sharpe_ratio' in results_df.columns:
        portfolio_insights['avg_sharpe_ratio'] = float(results_df['sharpe_ratio'].mean())
        portfolio_insights['max_sharpe_ratio'] = float(results_df['sharpe_ratio'].max())
    
    # Black-Litterman specific insights
    bl_insights = {}
    if 'posterior_sharpe_vs_prior_sharpe' in results_df.columns:
        bl_insights['avg_sharpe_improvement'] = float(results_df['posterior_sharpe_vs_prior_sharpe'].mean())
    if 'information_gain_from_views' in results_df.columns:
        bl_insights['avg_information_gain'] = float(results_df['information_gain_from_views'].mean())
    if 'view_impact_magnitude' in results_df.columns:
        bl_insights['avg_view_impact'] = float(results_df['view_impact_magnitude'].mean())
    
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
    if 'hhi_concentration' in results_df.columns and 'sharpe_ratio' in results_df.columns:
        corr = results_df['hhi_concentration'].corr(results_df['sharpe_ratio'])
        if not np.isnan(corr):
            structure_effects['correlation_hhi_vs_sharpe'] = float(corr)
    
    # Runtime stats
    runtime_stats = {}
    if len(runtimes) > 0:
        runtime_array = np.array(runtimes) * 1000  # Convert to ms
        runtime_stats['mean_runtime_ms'] = float(np.mean(runtime_array))
        runtime_stats['p95_runtime_ms'] = float(np.percentile(runtime_array, 95))
        runtime_stats['median_runtime_ms'] = float(np.median(runtime_array))
    
    return {
        'portfolio_level_insights': portfolio_insights,
        'black_litterman_insights': bl_insights,
        'distribution_effects': distribution_effects,
        'structure_effects': structure_effects,
        'runtime_stats': runtime_stats
    }


def _process_single_optimization_worker(args):
    """
    Worker function for parallel processing.
    
    Args:
        args: Tuple of (returns, baseline_portfolios, portfolio_id, bl_settings, 
              risk_free_rate, random_seed, investor_views_loaded, cache)
    
    Returns:
        Tuple of (result dictionary, runtime in seconds)
    """
    # Suppress warnings in worker processes - do this FIRST before any other imports
    import warnings
    import sys
    
    # Suppress all RuntimeWarnings (including sys.modules warnings from runpy)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning, message='.*cvxpy.*')
    
    # Also redirect stderr to suppress the warnings if they still appear
    # (Some warnings bypass the filter and go directly to stderr)
    original_stderr = sys.stderr
    try:
        # Create a filter that suppresses RuntimeWarning messages
        class WarningFilter:
            def write(self, text):
                if 'RuntimeWarning' in text and 'sys.modules' in text:
                    return  # Suppress these warnings
                original_stderr.write(text)
            def flush(self):
                original_stderr.flush()
        
        sys.stderr = WarningFilter()
        
        (returns, baseline_portfolios, portfolio_id, bl_settings, 
         risk_free_rate, random_seed, investor_views_loaded, cache) = args
        result = _process_single_optimization(
            returns,
            baseline_portfolios,
            portfolio_id,
            bl_settings,
            risk_free_rate,
            random_seed=random_seed,
            investor_views_loaded=investor_views_loaded,
            cache=cache
        )
    finally:
        sys.stderr = original_stderr
    
    return result


def _process_single_optimization(
    returns: pd.DataFrame,
    baseline_portfolios: pd.DataFrame,
    portfolio_id: Union[int, str],
    bl_settings: Dict,
    risk_free_rate: float,
    random_seed: Optional[int] = None,
    investor_views_loaded: Optional[Dict] = None,
    cache: Optional[BLCache] = None
) -> Tuple[Dict, float]:
    """
    Process a single Black-Litterman optimization for a portfolio.
    
    Args:
        returns: DataFrame of daily returns
        baseline_portfolios: DataFrame with baseline portfolio information
        portfolio_id: Portfolio identifier
        bl_settings: Black-Litterman optimization settings
        risk_free_rate: Risk-free rate
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (result dictionary, runtime in seconds)
    """
    start_time = time.time()
    
    try:
        # Get assets for this portfolio
        if portfolio_id in baseline_portfolios.index:
            portfolio_assets = baseline_portfolios.loc[portfolio_id]
            if isinstance(portfolio_assets, pd.Series):
                assets = portfolio_assets.index.tolist()
            else:
                assets = portfolio_assets
        else:
            assets = returns.columns.tolist()
        
        # Filter to assets available in returns
        assets = [a for a in assets if a in returns.columns]
        
        if len(assets) == 0:
            return {}, 0.0
        
        returns_subset = returns[assets]
        asset_set = tuple(sorted(assets))
        
        # Get settings
        cov_settings = bl_settings.get('covariance_estimation', {})
        market_eq_settings = bl_settings.get('market_equilibrium', {})
        views_settings = bl_settings.get('views', {})
        posterior_settings = bl_settings.get('posterior_settings', {})
        optimization_settings = bl_settings.get('optimization', {})
        constraints = optimization_settings.get('constraints', {})
        solver_settings = optimization_settings.get('solver', {})
        
        # Get cache settings from config (passed via bl_settings for worker processes)
        global_caching = bl_settings.get('_global_caching', {})
        cache_once = global_caching.get('cache_once', {})
        
        # Get GPU settings
        gpu_config = bl_settings.get('_gpu_config', {})
        use_gpu_for_cov = gpu_config.get('enabled', False) and 'covariance_computation' in gpu_config.get('use_for', [])
        use_gpu_for_posterior = gpu_config.get('enabled', False) and 'posterior_computation' in gpu_config.get('use_for', [])
        
        estimation_window = cov_settings.get('estimation_window', 252)
        
        # Check cache for covariance matrix
        cov_time = 0.0
        if cache is not None and cache_once.get('covariance_matrix', False):
            cached_cov = cache.get_covariance_matrix(asset_set, estimation_window)
            if cached_cov is not None:
                cov_matrix = cached_cov
            else:
                cov_matrix, cov_time = compute_covariance_matrix(
                    returns_subset,
                    method=cov_settings.get('method', 'sample'),
                    window=estimation_window,
                    use_shrinkage=cov_settings.get('shrinkage', {}).get('use_shrinkage', False),
                    shrinkage_method=cov_settings.get('shrinkage', {}).get('method', 'ledoit_wolf'),
                    estimation_windows=[estimation_window],
                    use_gpu=use_gpu_for_cov
                )
                cache.set_covariance_matrix(asset_set, estimation_window, cov_matrix)
        else:
            cov_matrix, cov_time = compute_covariance_matrix(
                returns_subset,
                method=cov_settings.get('method', 'sample'),
                window=estimation_window,
                use_shrinkage=cov_settings.get('shrinkage', {}).get('use_shrinkage', False),
                shrinkage_method=cov_settings.get('shrinkage', {}).get('method', 'ledoit_wolf'),
                estimation_windows=[estimation_window],
                use_gpu=use_gpu_for_cov
            )
        
        # Check cache for market equilibrium
        eq_time = 0.0
        if cache is not None and cache_once.get('market_equilibrium_returns', False):
            cached_eq = cache.get_market_equilibrium(asset_set)
            if cached_eq is not None:
                prior_returns, market_weights = cached_eq
                eq_time = 0.0
            else:
                prior_returns, market_weights, eq_time = derive_market_equilibrium_returns(
                    returns_subset,
                    baseline_portfolios,
                    cov_matrix,
                    risk_aversion=bl_settings.get('risk_aversion', 2.5),
                    derive_from_portfolios=market_eq_settings.get('derive_pi_from_portfolios', True),
                    market_weight_source=market_eq_settings.get('market_weight_source', 'baseline_portfolios'),
                    use_cap_weighted=market_eq_settings.get('use_cap_weighted_benchmark', False)
                )
                eq_time = eq_time * 1000  # Convert to ms
                cache.set_market_equilibrium(asset_set, prior_returns, market_weights)
        else:
            prior_returns, market_weights, eq_time = derive_market_equilibrium_returns(
                returns_subset,
                baseline_portfolios,
                cov_matrix,
                risk_aversion=bl_settings.get('risk_aversion', 2.5),
                derive_from_portfolios=market_eq_settings.get('derive_pi_from_portfolios', True),
                market_weight_source=market_eq_settings.get('market_weight_source', 'baseline_portfolios'),
                use_cap_weighted=market_eq_settings.get('use_cap_weighted_benchmark', False)
            )
            eq_time = eq_time * 1000  # Convert to ms
        
        # Load or generate views
        views_start = time.time()
        if investor_views_loaded is not None:
            # Use loaded views, but filter to available assets
            investor_views = {
                'absolute_views': [
                    v for v in investor_views_loaded.get('absolute_views', [])
                    if v.get('asset') in assets
                ],
                'relative_views': [
                    v for v in investor_views_loaded.get('relative_views', [])
                    if len(v.get('assets', [])) >= 2 and
                    all(a in assets for a in v.get('assets', [])[:2])
                ]
            }
        elif views_settings.get('generate_synthetic_views', True):
            investor_views = generate_synthetic_views(
                returns_subset,
                num_views=views_settings.get('num_views', 5),
                view_generation_method=views_settings.get('view_generation_method', 'return_differentials'),
                random_seed=random_seed
            )
        else:
            investor_views = {"absolute_views": [], "relative_views": []}
        
        # Hash views for caching
        if cache is not None:
            views_hash = cache._hash_views(investor_views)
        else:
            views_hash = None
        
        # Check cache for parsed views
        if cache is not None and cache_once.get('parsed_views', False) and views_hash is not None:
            cached_views = cache.get_parsed_views(asset_set, views_hash)
            if cached_views is not None:
                P, Q, Omega = cached_views
                view_time = 0.0
            else:
                P, Q, Omega, view_time = parse_and_scale_views(
                    investor_views,
                    assets,
                    cov_matrix,
                    tau=bl_settings.get('tau', 0.025),
                    view_type=views_settings.get('view_type', 'absolute_relative_mixed'),
                    uncertainty_matrix=views_settings.get('uncertainty_matrix', 'diagonal'),
                    confidence_levels=views_settings.get('confidence_levels', [0.50, 0.75, 0.90])
                )
                view_time = (time.time() - views_start) * 1000
                cache.set_parsed_views(asset_set, views_hash, P, Q, Omega)
        else:
            P, Q, Omega, view_time = parse_and_scale_views(
                investor_views,
                assets,
                cov_matrix,
                tau=bl_settings.get('tau', 0.025),
                view_type=views_settings.get('view_type', 'absolute_relative_mixed'),
                uncertainty_matrix=views_settings.get('uncertainty_matrix', 'diagonal'),
                confidence_levels=views_settings.get('confidence_levels', [0.50, 0.75, 0.90])
            )
            view_time = (time.time() - views_start) * 1000
        
        # Check cache for posterior returns
        post_time = 0.0
        if cache is not None and cache_once.get('posterior_returns', False) and views_hash is not None:
            cached_post = cache.get_posterior_returns(asset_set, views_hash)
            if cached_post is not None:
                posterior_returns = cached_post
            else:
                posterior_returns, post_time = compute_posterior_bl_returns(
                    prior_returns,
                    cov_matrix,
                    P,
                    Q,
                    Omega,
                    tau=bl_settings.get('tau', 0.025),
                    use_gpu=use_gpu_for_posterior
                )
                post_time = post_time * 1000  # Convert to ms
                cache.set_posterior_returns(asset_set, views_hash, posterior_returns)
        else:
            posterior_returns, post_time = compute_posterior_bl_returns(
                prior_returns,
                cov_matrix,
                P,
                Q,
                Omega,
                tau=bl_settings.get('tau', 0.025),
                use_gpu=use_gpu_for_posterior
            )
            post_time = post_time * 1000  # Convert to ms
        
        # Check cache for posterior covariance
        post_cov_time = 0.0
        if cache is not None and cache_once.get('posterior_covariance', False) and views_hash is not None:
            cached_post_cov = cache.get_posterior_covariance(asset_set, views_hash)
            if cached_post_cov is not None:
                posterior_cov = cached_post_cov
            else:
                posterior_cov, post_cov_time = compute_posterior_covariance(
                    cov_matrix,
                    P,
                    Omega,
                    tau=bl_settings.get('tau', 0.025),
                    method=posterior_settings.get('posterior_covariance_method', 'black_litterman'),
                    use_gpu=use_gpu_for_posterior
                )
                post_cov_time = post_cov_time * 1000  # Convert to ms
                cache.set_posterior_covariance(asset_set, views_hash, posterior_cov)
        else:
            posterior_cov, post_cov_time = compute_posterior_covariance(
                cov_matrix,
                P,
                Omega,
                tau=bl_settings.get('tau', 0.025),
                method=posterior_settings.get('posterior_covariance_method', 'black_litterman'),
                use_gpu=use_gpu_for_posterior
            )
            post_cov_time = post_cov_time * 1000  # Convert to ms
        
        # Determine objective
        objective = optimization_settings.get('objective', 'maximize_posterior_sharpe')
        lambda_values = optimization_settings.get('risk_return_tradeoff', {}).get('lambda_values', [1.0])
        risk_aversion = lambda_values[0]  # Use first lambda value for this optimization
        
        # Optimize portfolio
        solver_method = solver_settings.get('method', 'osqp')
        fallback_solver = solver_settings.get('fallback', 'osqp')
        optimal_weights, opt_info, solver_time = optimize_portfolio(
            posterior_returns,
            posterior_cov,
            objective=objective,
            risk_aversion=risk_aversion,
            risk_free_rate=risk_free_rate,
            constraints=constraints,
            solver=solver_method,
            tolerance=solver_settings.get('tolerance', 1e-6),
            fallback_solver=fallback_solver
        )
        
        # Store optimal weights for time-sliced evaluation
        optimal_weights_series = optimal_weights
        
        # Compute portfolio returns
        portfolio_returns = (returns_subset * optimal_weights).sum(axis=1)
        
        # Compute all metrics
        portfolio_stats = compute_portfolio_statistics(
            portfolio_returns,
            optimal_weights,
            posterior_returns,
            posterior_cov,
            risk_free_rate
        )
        
        sharpe = compute_sharpe_ratio(portfolio_returns, risk_free_rate)
        sortino = compute_sortino_ratio(portfolio_returns, risk_free_rate)
        max_dd = compute_max_drawdown(portfolio_returns)
        calmar = compute_calmar_ratio(portfolio_returns)
        
        # Market comparison metrics
        market_portfolio_returns = (returns_subset * market_weights).sum(axis=1)
        info_ratio = compute_information_ratio(portfolio_returns, market_portfolio_returns)
        tracking_error = compute_tracking_error(portfolio_returns, market_portfolio_returns)
        alpha = compute_alpha_vs_market(portfolio_returns, market_portfolio_returns, risk_free_rate)
        
        # Black-Litterman specific metrics
        bl_metrics = compute_bl_specific_metrics(
            prior_returns,
            posterior_returns,
            market_weights,
            optimal_weights,
            cov_matrix,
            posterior_cov,
            P,
            Q,
            Omega,
            portfolio_returns,
            market_portfolio_returns,
            risk_free_rate,
            tau=bl_settings.get('tau', 0.025),
            risk_aversion=bl_settings.get('risk_aversion', 2.5)
        )
        
        structure_metrics = compute_structure_metrics(
            optimal_weights,
            posterior_cov,
            returns_subset,
            market_weights
        )
        
        risk_metrics = compute_risk_metrics(portfolio_returns)
        
        distribution_metrics = compute_distribution_metrics(portfolio_returns)
        
        # Comparison metrics
        prior_portfolio_returns = (returns_subset * market_weights).sum(axis=1)
        comparison_metrics = compute_comparison_metrics(
            prior_portfolio_returns,
            portfolio_returns,
            market_portfolio_returns,
            prior_returns,
            posterior_returns,
            market_weights,
            optimal_weights,
            risk_free_rate
        )
        
        runtime_total = time.time() - start_time
        
        # Combine all results
        result = {
            'portfolio_id': portfolio_id,
            'objective': objective,
            'risk_aversion': risk_aversion,
            'tau': bl_settings.get('tau', 0.025),
            **portfolio_stats,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'information_ratio': info_ratio,
            'tracking_error_vs_market': tracking_error,
            'alpha_vs_market_portfolio': alpha,
            **bl_metrics,
            **structure_metrics,
            **risk_metrics,
            **distribution_metrics,
            **comparison_metrics,
            'runtime_per_optimization_ms': runtime_total * 1000,
            'covariance_estimation_time_ms': cov_time,
            'equilibrium_returns_calculation_time_ms': eq_time,
            'view_processing_time_ms': view_time,
            'posterior_calculation_time_ms': post_time + post_cov_time,
            'solver_time_ms': solver_time,
            'optimization_status': opt_info.get('status', 'unknown')
        }
        
        # Store additional data for time-sliced evaluation
        result['_portfolio_returns'] = portfolio_returns
        result['_prior_portfolio_returns'] = prior_portfolio_returns
        result['_market_portfolio_returns'] = market_portfolio_returns
        result['_optimal_weights'] = optimal_weights_series
        result['_posterior_returns'] = posterior_returns
        
        return result, runtime_total
        
    except Exception as e:
        warnings.warn(f"Error optimizing portfolio {portfolio_id}: {e}")
        import traceback
        traceback.print_exc()
        return {}, 0.0


def _hash_constraints(constraints: Dict) -> str:
    """Create hash of constraints dictionary."""
    constraints_str = json.dumps(constraints, sort_keys=True, default=str)
    return hashlib.md5(constraints_str.encode()).hexdigest()[:16]


def run_black_litterman_optimization(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    max_portfolios: Optional[int] = 100,
    n_jobs: Optional[int] = None
) -> Dict:
    """
    Run Black-Litterman optimization pipeline.
    
    Args:
        config_path: Path to llm.json configuration file
        config_dict: Optional configuration dictionary (overrides config_path)
        max_portfolios: Maximum number of portfolios to process (default: 100, None for all)
        
    Returns:
        Dictionary with optimization results
    """
    # Load configuration
    if config_dict is None:
        if config_path is None:
            current_file = Path(__file__)
            config_path = current_file.parent / 'llm.json'
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = config_dict
    
    # Extract settings
    inputs = config['inputs']
    bl_settings = config['black_litterman_settings'].copy()
    precompute_config = config.get('precompute_registry', {})
    gpu_config = config.get('gpu_acceleration', {})
    batch_config = config.get('batch_evaluation', {})
    optimization_config = config.get('optimization', {})
    outputs = config['outputs']
    execution_plan = config.get('execution_plan', {})
    
    # Check GPU availability
    gpu_enabled = gpu_config.get('enabled', False)
    use_gpu = gpu_enabled and GPU_UTILS_AVAILABLE and is_gpu_available()
    
    if gpu_enabled:
        if use_gpu:
            gpu_info = get_gpu_info()
            print(f"GPU acceleration enabled: {gpu_info.get('library', 'Unknown')} on {gpu_info.get('device', 'Unknown')}")
        else:
            if gpu_config.get('fallback_to_cpu', True):
                print("GPU acceleration requested but not available. Falling back to CPU.")
            else:
                raise RuntimeError("GPU acceleration required but not available. Install CuPy and ensure GPU is accessible.")
    
    # Initialize precompute registry
    registry = PrecomputeRegistry(precompute_config.get('registry_root', 'cache/bl_precompute'))
    
    print("Loading data...")
    # Load data
    prices = load_panel_prices(inputs['panel_price_path'])
    baseline_portfolios = load_baseline_portfolios(inputs['baseline_portfolios_path'])
    risk_free_rate = inputs.get('risk_free_rate', 0.0)
    
    # Load investor views if path is provided
    investor_views_loaded = None
    if 'investor_views_path' in inputs:
        try:
            investor_views_loaded = load_investor_views(inputs['investor_views_path'])
            print(f"Loaded investor views from {inputs['investor_views_path']}")
        except Exception as e:
            warnings.warn(f"Could not load investor views: {e}")
    
    print("Computing returns...")
    # Compute returns
    daily_returns = compute_daily_returns(prices, method=inputs.get('return_type', 'log'))
    
    # Save daily returns if configured
    if precompute_config.get('enabled', False):
        returns_path = Path(precompute_config['artifacts']['daily_returns_full_universe'])
        returns_path.parent.mkdir(parents=True, exist_ok=True)
        daily_returns.to_parquet(returns_path)
    
    # Get portfolio IDs
    if isinstance(baseline_portfolios, pd.DataFrame):
        portfolio_ids = baseline_portfolios.index.tolist()
    else:
        portfolio_ids = [0]
    
    num_portfolios_total = len(portfolio_ids)
    
    # Limit number of portfolios if specified
    if max_portfolios is not None and max_portfolios > 0:
        num_portfolios = min(num_portfolios_total, max_portfolios)
        portfolio_ids = portfolio_ids[:num_portfolios]
        print(f"Optimizing {num_portfolios:,} portfolios (out of {num_portfolios_total:,} total)...")
    else:
        num_portfolios = num_portfolios_total
        print(f"Optimizing {num_portfolios:,} portfolios...")
    
    runtime_profile = {'stages': {}}
    stage_start = time.time()
    
    # ========== STAGE A: Precompute per unique asset set ==========
    if execution_plan.get('stage_A_precompute_per_unique_asset_set', True):
        print("\n=== Stage A: Precomputing per unique asset set ===")
        stage_a_start = time.time()
        
        # Build unique asset sets from baseline portfolios
        asset_set_to_portfolios = {}
        portfolio_to_asset_set = {}
        
        all_assets = set(daily_returns.columns)
        
        for portfolio_id in portfolio_ids:
            if portfolio_id in baseline_portfolios.index:
                portfolio_data = baseline_portfolios.loc[portfolio_id]
                if isinstance(portfolio_data, pd.Series):
                    assets = [a for a in portfolio_data.index if a in all_assets and portfolio_data[a] > 0]
                else:
                    assets = [a for a in all_assets if a in portfolio_data]
            else:
                assets = list(all_assets)
            
            asset_set = tuple(sorted(assets))
            portfolio_to_asset_set[portfolio_id] = asset_set
            
            if asset_set not in asset_set_to_portfolios:
                asset_set_to_portfolios[asset_set] = []
            asset_set_to_portfolios[asset_set].append(portfolio_id)
        
        print(f"Found {len(asset_set_to_portfolios)} unique asset sets")
        
        # Precompute per asset_set
        asset_set_results = {}
        cov_settings = bl_settings.get('covariance_estimation', {})
        market_eq_settings = bl_settings.get('market_equilibrium', {})
        views_settings = bl_settings.get('views', {})
        posterior_settings = bl_settings.get('posterior_settings', {})
        optimization_settings = optimization_config
        constraints = optimization_settings.get('constraints', {})
        solver_settings = optimization_settings.get('solver', {})
        estimation_window = cov_settings.get('estimation_window', 252)
        shrinkage_method = cov_settings.get('shrinkage', {}).get('method', 'ledoit_wolf')
        
        for asset_set_idx, (asset_set, portfolio_list) in enumerate(asset_set_to_portfolios.items()):
            if len(asset_set) == 0:
                continue
            
            print(f"  Processing asset set {asset_set_idx + 1}/{len(asset_set_to_portfolios)} ({len(asset_set)} assets, {len(portfolio_list)} portfolios)...")
            
            returns_subset = daily_returns[list(asset_set)]
            
            # Check cache for covariance
            cov_matrix = registry.load_covariance(asset_set, estimation_window, shrinkage_method)
            if cov_matrix is None:
                cov_matrix, _ = compute_covariance_matrix(
                    returns_subset,
                    method=cov_settings.get('method', 'sample'),
                    window=estimation_window,
                    use_shrinkage=cov_settings.get('shrinkage', {}).get('use_shrinkage', False),
                    shrinkage_method=shrinkage_method
                )
                if precompute_config.get('enabled', False):
                    registry.save_covariance(asset_set, cov_matrix, estimation_window, shrinkage_method)
            
            # Market equilibrium
            risk_aversion = bl_settings.get('risk_aversion', 2.5)
            eq_result = registry.load_market_equilibrium(asset_set, risk_aversion)
            if eq_result is None:
                prior_returns, market_weights, _ = derive_market_equilibrium_returns(
                    returns_subset,
                    baseline_portfolios,
                    cov_matrix,
                    risk_aversion=risk_aversion,
                    derive_from_portfolios=market_eq_settings.get('derive_pi_from_portfolios', True),
                    market_weight_source=market_eq_settings.get('market_weight_source', 'baseline_portfolios')
                )
                if precompute_config.get('enabled', False):
                    registry.save_market_equilibrium(asset_set, prior_returns, market_weights, risk_aversion)
            else:
                prior_returns, market_weights = eq_result
            
            # Views
            if investor_views_loaded is not None:
                investor_views = {
                    'absolute_views': [v for v in investor_views_loaded.get('absolute_views', []) if v.get('asset') in asset_set],
                    'relative_views': [v for v in investor_views_loaded.get('relative_views', []) 
                                     if len(v.get('assets', [])) >= 2 and all(a in asset_set for a in v.get('assets', [])[:2])]
                }
            elif views_settings.get('use_external_views', True):
                investor_views = investor_views_loaded if investor_views_loaded else {"absolute_views": [], "relative_views": []}
            else:
                investor_views = generate_synthetic_views(
                    returns_subset,
                    num_views=views_settings.get('num_views', 5),
                    view_generation_method=views_settings.get('view_generation_method', 'return_differentials')
                )
            
            # Hash views
            views_hash = registry._hash_views(investor_views)
            
            # Parsed views
            parsed_views = registry.load_parsed_views(asset_set, views_hash)
            if parsed_views is None:
                P, Q, Omega, _ = parse_and_scale_views(
                    investor_views,
                    list(asset_set),
                    cov_matrix,
                    tau=bl_settings.get('tau', 0.025),
                    view_type=views_settings.get('view_type', 'absolute_relative_mixed'),
                    uncertainty_matrix=views_settings.get('uncertainty_matrix', 'idzorek'),
                    confidence_levels=views_settings.get('confidence_levels', [0.5, 0.75, 0.9])
                )
                if precompute_config.get('enabled', False):
                    registry.save_parsed_views(asset_set, views_hash, P, Q, Omega)
            else:
                P, Q, Omega = parsed_views
            
            # Posterior
            post_result = registry.load_posterior(asset_set, views_hash)
            if post_result is None:
                posterior_returns, _ = compute_posterior_bl_returns(
                    prior_returns,
                    cov_matrix,
                    P,
                    Q,
                    Omega,
                    tau=bl_settings.get('tau', 0.025)
                )
                posterior_cov, _ = compute_posterior_covariance(
                    cov_matrix,
                    P,
                    Omega,
                    tau=bl_settings.get('tau', 0.025),
                    method=posterior_settings.get('posterior_covariance_method', 'black_litterman')
                )
                if precompute_config.get('enabled', False):
                    registry.save_posterior(asset_set, views_hash, posterior_returns, posterior_cov)
            else:
                posterior_returns, posterior_cov = post_result
            
            # Optimal weights
            constraints_hash = _hash_constraints(constraints)
            optimal_weights = registry.load_optimal_weights(asset_set, views_hash, constraints_hash)
            if optimal_weights is None:
                optimal_weights, _, _ = optimize_portfolio(
                    posterior_returns,
                    posterior_cov,
                    objective=optimization_settings.get('objective', 'mean_variance'),
                    risk_aversion=optimization_settings.get('risk_aversion', 1.0),
                    risk_free_rate=risk_free_rate,
                    constraints=constraints,
                    solver=solver_settings.get('method', 'closed_form_preferred'),
                    tolerance=solver_settings.get('tolerance', 1e-6),
                    fallback_solver=solver_settings.get('fallback', 'osqp'),
                    use_closed_form_when_possible=solver_settings.get('use_closed_form_when_possible', True)
                )
                if precompute_config.get('enabled', False):
                    registry.save_optimal_weights(asset_set, views_hash, constraints_hash, optimal_weights)
            
            asset_set_results[asset_set] = {
                'optimal_weights': optimal_weights,
                'posterior_returns': posterior_returns,
                'posterior_cov': posterior_cov,
                'prior_returns': prior_returns,
                'market_weights': market_weights
            }
        
        runtime_profile['stages']['stage_A_precompute'] = time.time() - stage_a_start
        print(f"Stage A completed in {runtime_profile['stages']['stage_A_precompute']:.2f}s")
    
    # Get all assets (needed for Stage B and later)
    all_assets = sorted(daily_returns.columns)
    N_assets = len(all_assets)
    N_portfolios = len(portfolio_ids)
    
    # ========== STAGE B: Expand weights to full universe ==========
    if execution_plan.get('stage_B_expand_weights_to_full_universe', True):
        print("\n=== Stage B: Expanding weights to full universe ===")
        stage_b_start = time.time()
        
        # Build full-universe weight matrix W (N_assets x N_portfolios)
        W_full = np.zeros((N_assets, N_portfolios))
        
        for portfolio_idx, portfolio_id in enumerate(portfolio_ids):
            asset_set = portfolio_to_asset_set[portfolio_id]
            optimal_weights = asset_set_results[asset_set]['optimal_weights']
            
            # Place subset weights in correct asset columns
            for asset in asset_set:
                if asset in all_assets and asset in optimal_weights.index:
                    asset_idx = all_assets.index(asset)
                    W_full[asset_idx, portfolio_idx] = optimal_weights[asset]
        
        # Save weight matrix
        if precompute_config.get('enabled', False):
            W_path = Path(precompute_config['artifacts']['full_universe_weight_matrix_100k'])
            W_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(W_path, W_full)
        
        runtime_profile['stages']['stage_B_expand_weights'] = time.time() - stage_b_start
        print(f"Stage B completed in {runtime_profile['stages']['stage_B_expand_weights']:.2f}s")
    
    # ========== STAGE C: Batch evaluate all portfolios ==========
    if execution_plan.get('stage_C_batch_evaluate_all_portfolios', True):
        print("\n=== Stage C: Batch evaluating all portfolios ===")
        stage_c_start = time.time()
        
        # Compute portfolio returns for all portfolios in batch: R(T x N_assets) @ W(N_assets x N_portfolios)
        returns_array = daily_returns[all_assets].values  # T x N_assets
        
        # Clear GPU cache before large batch operation if configured
        if use_gpu and gpu_config.get('memory_management', {}).get('clear_cache_periodically', False):
            clear_gpu_cache()
        
        if batch_config.get('enabled', True) and batch_portfolio_returns_matrix_multiply is not None and use_gpu:
            print("  Computing portfolio returns on GPU...")
            portfolio_returns_matrix = batch_portfolio_returns_matrix_multiply(
                returns_array,
                W_full,
                use_gpu=True
            )
        else:
            print("  Computing portfolio returns on CPU...")
            portfolio_returns_matrix = returns_array @ W_full  # T x N_portfolios
        
        # Save portfolio returns matrix if configured
        if 'portfolio_daily_returns_matrix' in outputs:
            np.save(outputs['portfolio_daily_returns_matrix'], portfolio_returns_matrix)
        
        # Compute metrics in batch
        print("  Computing metrics in batch...")
        metrics_use_gpu = (
            use_gpu and 
            batch_config.get('metrics_batching', {}).get('compute_metrics_vectorized', True) and
            batch_config.get('metrics_batching', {}).get('accurate_metrics_no_sampling', True)
        )
        batch_metrics = compute_batch_metrics_vectorized(
            portfolio_returns_matrix,
            risk_free_rate,
            use_gpu=metrics_use_gpu
        )
        
        # Clear GPU cache periodically if configured
        if use_gpu and gpu_config.get('memory_management', {}).get('clear_cache_periodically', False):
            clear_every_n = gpu_config.get('memory_management', {}).get('clear_every_n_portfolios', 20000)
            if num_portfolios > clear_every_n:
                clear_gpu_cache()
        
        # Build metrics DataFrame
        metrics_data = []
        for portfolio_idx, portfolio_id in enumerate(portfolio_ids):
            metrics_data.append({
                'portfolio_id': portfolio_id,
                'mean_return': batch_metrics['mean_return'][portfolio_idx],
                'volatility': batch_metrics['volatility'][portfolio_idx],
                'sharpe_ratio': batch_metrics['sharpe_ratio'][portfolio_idx],
                'sortino_ratio': batch_metrics['sortino_ratio'][portfolio_idx],
                'max_drawdown': batch_metrics['max_drawdown'][portfolio_idx],
                'calmar_ratio': batch_metrics['calmar_ratio'][portfolio_idx],
                'value_at_risk': batch_metrics['value_at_risk'][portfolio_idx],
                'conditional_value_at_risk': batch_metrics['conditional_value_at_risk'][portfolio_idx],
                'skewness': batch_metrics['skewness'][portfolio_idx],
                'kurtosis': batch_metrics['kurtosis'][portfolio_idx],
                'jarque_bera_p_value': batch_metrics['jarque_bera_p_value'][portfolio_idx]
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Time-sliced metrics
        if batch_config.get('metrics_batching', {}).get('time_sliced_metrics', {}).get('enabled', False):
            print("  Computing time-sliced metrics...")
            time_unit = batch_config['metrics_batching']['time_sliced_metrics'].get('time_unit', 'calendar_year')
            time_sliced_df = compute_batch_time_sliced_metrics(
                portfolio_returns_matrix,
                daily_returns.index,
                np.array(portfolio_ids),
                risk_free_rate,
                time_unit
            )
            
            if 'time_sliced_metrics' in outputs:
                output_path = Path(outputs['time_sliced_metrics'])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                time_sliced_df.to_parquet(output_path)
        
        runtime_profile['stages']['stage_C_batch_evaluate'] = time.time() - stage_c_start
        print(f"Stage C completed in {runtime_profile['stages']['stage_C_batch_evaluate']:.2f}s")
    
    # Build optimal weights and posterior returns DataFrames
    optimal_weights_list = []
    posterior_returns_list = []
    posterior_cov_list = []
    
    for portfolio_id in portfolio_ids:
        asset_set = portfolio_to_asset_set[portfolio_id]
        optimal_weights = asset_set_results[asset_set]['optimal_weights']
        posterior_returns = asset_set_results[asset_set]['posterior_returns']
        posterior_cov = asset_set_results[asset_set]['posterior_cov']
        
        # Expand to full universe
        weights_dict = {asset: 0.0 for asset in all_assets}
        for asset in optimal_weights.index:
            if asset in weights_dict:
                weights_dict[asset] = optimal_weights[asset]
        weights_dict['portfolio_id'] = portfolio_id
        optimal_weights_list.append(weights_dict)
        
        posterior_dict = {asset: 0.0 for asset in all_assets}
        for asset in posterior_returns.index:
            if asset in posterior_dict:
                posterior_dict[asset] = posterior_returns[asset]
        posterior_dict['portfolio_id'] = portfolio_id
        posterior_returns_list.append(posterior_dict)
        
        # Store posterior covariance (subset only, as it's asset_set-specific)
        cov_dict = posterior_cov.to_dict()
        cov_dict['portfolio_id'] = portfolio_id
        posterior_cov_list.append(cov_dict)
    
    optimal_weights_df = pd.DataFrame(optimal_weights_list).set_index('portfolio_id')
    posterior_returns_df = pd.DataFrame(posterior_returns_list).set_index('portfolio_id')
    
    # Save posterior covariance if configured
    if 'posterior_covariance' in outputs:
        # Note: posterior covariance is asset_set-specific, so we save a simplified version
        # For full analysis, use the precomputed values per asset_set
        pass
    
    # Save outputs
    print("\nSaving results...")
    output_base = Path(outputs['metrics_table']).parent
    output_base.mkdir(parents=True, exist_ok=True)
    
    metrics_df.to_parquet(outputs['metrics_table'])
    
    if 'optimal_portfolios' in outputs:
        optimal_weights_df.to_parquet(outputs['optimal_portfolios'])
    
    if 'posterior_returns' in outputs:
        posterior_returns_df.to_parquet(outputs['posterior_returns'])
    
    if 'posterior_covariance' in outputs:
        # Save a representative posterior covariance (from first asset_set)
        # Full covariance matrices are asset_set-specific and stored in precompute registry
        if len(asset_set_results) > 0:
            first_asset_set = list(asset_set_results.keys())[0]
            first_posterior_cov = asset_set_results[first_asset_set]['posterior_cov']
            first_posterior_cov.to_parquet(outputs['posterior_covariance'])
    
    # Runtime profile
    runtime_profile['total_time'] = time.time() - stage_start
    runtime_profile['gpu_info'] = get_gpu_info() if GPU_UTILS_AVAILABLE else {'available': False}
    
    if 'runtime_profile' in outputs:
        with open(outputs['runtime_profile'], 'w') as f:
            json.dump(runtime_profile, f, indent=2, default=str)
    
    # Summary statistics
    summary_stats = _compute_summary_statistics(metrics_data, [])
    
    # Generate report
    if 'summary_report' in outputs:
        print("Generating report...")
        from .report_generator import generate_report
        generate_report(metrics_df, outputs['summary_report'], bl_settings, [])
    
    # Generate metrics schema
    if 'metrics_schema_json' in outputs:
        print("Generating metrics schema...")
        from .report_generator import generate_metrics_schema
        generate_metrics_schema(metrics_df, outputs['metrics_schema_json'])
    
    # Save asset set index and portfolio mapping if configured
    if precompute_config.get('enabled', False):
        # Save asset set index
        asset_set_index_data = []
        for asset_set_idx, (asset_set, portfolio_list) in enumerate(asset_set_to_portfolios.items()):
            asset_set_index_data.append({
                'asset_set_id': asset_set_idx,
                'assets': ','.join(sorted(asset_set)),
                'num_portfolios': len(portfolio_list)
            })
        if asset_set_index_data:
            asset_set_index_df = pd.DataFrame(asset_set_index_data)
            asset_set_index_path = Path(precompute_config['artifacts']['asset_set_index'])
            asset_set_index_path.parent.mkdir(parents=True, exist_ok=True)
            asset_set_index_df.to_parquet(asset_set_index_path)
        
        # Save portfolio to asset set mapping
        portfolio_map_data = []
        for portfolio_id, asset_set in portfolio_to_asset_set.items():
            asset_set_id = list(asset_set_to_portfolios.keys()).index(asset_set) if asset_set in asset_set_to_portfolios else -1
            portfolio_map_data.append({
                'portfolio_id': portfolio_id,
                'asset_set_id': asset_set_id,
                'assets': ','.join(sorted(asset_set))
            })
        if portfolio_map_data:
            portfolio_map_df = pd.DataFrame(portfolio_map_data)
            portfolio_map_path = Path(precompute_config['artifacts']['portfolio_to_asset_set_map'])
            portfolio_map_path.parent.mkdir(parents=True, exist_ok=True)
            portfolio_map_df.to_parquet(portfolio_map_path)
    
    print(f"\nOptimization complete! Total time: {runtime_profile['total_time']:.2f}s")
    
    return {
        'metrics_df': metrics_df,
        'summary_stats': summary_stats,
        'num_portfolios': num_portfolios,
        'num_portfolios_total': num_portfolios_total,
        'runtime_profile': runtime_profile
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Black-Litterman Portfolio Optimization'
    )
    parser.add_argument(
        '--max-portfolios',
        type=int,
        default=100,
        help='Maximum number of portfolios to process (default: 100, use 0 for all)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration JSON file (default: llm.json in module directory)'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto, uses all CPU cores)'
    )
    
    args = parser.parse_args()
    
    # Convert 0 to None (process all portfolios)
    max_portfolios = None if args.max_portfolios == 0 else args.max_portfolios
    
    # Run optimization
    results = run_black_litterman_optimization(
        config_path=args.config,
        max_portfolios=max_portfolios,
        n_jobs=args.n_jobs
    )
    print(f"Optimized {results['num_portfolios']:,} portfolios (out of {results['num_portfolios_total']:,} total)")
