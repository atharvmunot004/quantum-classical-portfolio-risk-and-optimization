"""
Main evaluation script for Markowitz Mean-Variance Portfolio Optimization.

Orchestrates the entire optimization pipeline including:
- Data loading
- Returns computation
- Covariance and expected returns estimation
- Portfolio optimization
- Efficient frontier generation
- Metrics computation
- Report generation
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

# Suppress RuntimeWarning about module import in multiprocessing workers
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*found in sys.modules.*')

from .returns import (
    load_panel_prices,
    load_portfolio_universe,
    compute_daily_returns
)
from .markowitz_optimizer import (
    compute_covariance_matrix,
    compute_expected_returns,
    optimize_portfolio,
    generate_efficient_frontier
)
from .metrics import (
    compute_portfolio_statistics,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_structure_metrics,
    compute_risk_metrics,
    compute_distribution_metrics,
    compute_runtime_metrics
)
from .report_generator import generate_report, generate_metrics_schema
from .time_sliced_metrics import compute_time_sliced_metrics
from .cache import MarkowitzCache


def _save_restructured_json(
    json_path: Path,
    portfolio_results: List[Dict],
    summary_stats: Dict,
    data_period: str,
    num_portfolios: int,
    markowitz_settings: Dict
):
    """
    Save results in the restructured JSON format.
    
    Args:
        json_path: Path to save JSON file
        portfolio_results: List of restructured portfolio results
        summary_stats: Summary statistics dictionary
        data_period: Data period string
        num_portfolios: Number of portfolios optimized
        markowitz_settings: Markowitz optimization settings
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
            'task': 'markowitz_mean_variance_optimization',
            'data_period': data_period,
            'portfolios_optimized': num_portfolios,
            'markowitz_settings': markowitz_settings,
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
    
    # Portfolio-level insights
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
    
    if 'effective_number_of_assets' in results_df.columns and 'volatility' in results_df.columns:
        corr = results_df['effective_number_of_assets'].corr(results_df['volatility'])
        if not np.isnan(corr):
            structure_effects['correlation_enc_vs_volatility'] = float(corr)
    
    # Runtime stats
    runtime_stats = {}
    if len(runtimes) > 0:
        runtime_array = np.array(runtimes) * 1000  # Convert to ms
        runtime_stats['mean_runtime_ms'] = float(np.mean(runtime_array))
        runtime_stats['p95_runtime_ms'] = float(np.percentile(runtime_array, 95))
        runtime_stats['median_runtime_ms'] = float(np.median(runtime_array))
    
    return {
        'portfolio_level_insights': portfolio_insights,
        'distribution_effects': distribution_effects,
        'structure_effects': structure_effects,
        'runtime_stats': runtime_stats
    }


def _process_single_optimization(
    returns: pd.DataFrame,
    portfolio_universe: pd.DataFrame,
    portfolio_id: Union[int, str],
    markowitz_settings: Dict,
    risk_free_rate: float,
    modules: Dict,
    random_seed: Optional[int] = None,
    cache: Optional[MarkowitzCache] = None,
    portfolio_universe_semantics: str = 'asset_set_only'
) -> Tuple[Dict, pd.Series, float]:
    """
    Process a single portfolio optimization.
    
    Args:
        returns: DataFrame of daily returns
        portfolio_universe: DataFrame with portfolio universe information
        portfolio_id: Portfolio identifier
        markowitz_settings: Markowitz optimization settings
        risk_free_rate: Risk-free rate
        random_seed: Random seed for reproducibility
        cache: Optional cache for covariance and expected returns
        portfolio_universe_semantics: How to interpret portfolio universe ('asset_set_only')
        
    Returns:
        Tuple of (result dictionary, optimal weights Series, runtime in seconds)
    """
    start_time = time.time()
    
    try:
        # Get assets for this portfolio based on semantics
        if portfolio_universe_semantics == 'asset_set_only':
            # Only use asset sets, ignore any weights
            if portfolio_id in portfolio_universe.index:
                portfolio_row = portfolio_universe.loc[portfolio_id]
                if isinstance(portfolio_row, pd.Series):
                    # Get column names that are assets (not metadata columns)
                    # Assume asset columns are those that exist in returns
                    assets = [col for col in portfolio_row.index if col in returns.columns]
                else:
                    assets = []
            else:
                assets = []
        else:
            # Legacy behavior: try to extract assets from various formats
            if portfolio_id in portfolio_universe.index:
                portfolio_assets = portfolio_universe.loc[portfolio_id]
                if isinstance(portfolio_assets, pd.Series):
                    assets = [col for col in portfolio_assets.index if col in returns.columns]
                else:
                    assets = portfolio_assets if isinstance(portfolio_assets, list) else []
            else:
                assets = []
        
        # If no assets found, use all available assets
        if len(assets) == 0:
            assets = returns.columns.tolist()
        
        # Filter to assets available in returns
        assets = [a for a in assets if a in returns.columns]
        
        if len(assets) == 0:
            return {}, pd.Series(dtype=float), 0.0
        
        returns_subset = returns[assets]
        assets_tuple = tuple(sorted(assets))
        
        # Get settings
        cov_settings = markowitz_settings.get('covariance_estimation', {})
        exp_return_settings = markowitz_settings.get('expected_return_estimation', {})
        constraints = markowitz_settings.get('constraints', {})
        solver_settings = markowitz_settings.get('optimization_solver', {})
        
        estimation_window = cov_settings.get('estimation_windows', [252])[0] if cov_settings.get('estimation_windows') else None
        use_shrinkage = cov_settings.get('shrinkage', {}).get('use_shrinkage', False)
        shrinkage_method = cov_settings.get('shrinkage', {}).get('method', 'ledoit_wolf') if use_shrinkage else None
        
        # Check cache for covariance matrix
        cov_time = 0.0
        if cache is not None and cov_settings.get('caching', {}).get('enabled', False):
            cached_cov = cache.get_covariance_matrix(assets_tuple, estimation_window or len(returns_subset), shrinkage_method)
            if cached_cov is not None:
                cov_matrix = cached_cov
            else:
                cov_matrix, cov_time = compute_covariance_matrix(
                    returns_subset,
                    method=cov_settings.get('method', 'sample'),
                    window=estimation_window,
                    use_shrinkage=use_shrinkage,
                    shrinkage_method=shrinkage_method
                )
                cache.set_covariance_matrix(assets_tuple, estimation_window or len(returns_subset), cov_matrix, shrinkage_method)
        else:
            cov_matrix, cov_time = compute_covariance_matrix(
                returns_subset,
                method=cov_settings.get('method', 'sample'),
                window=estimation_window,
                use_shrinkage=use_shrinkage,
                shrinkage_method=shrinkage_method
            )
        
        # Check cache for expected returns
        if cache is not None and exp_return_settings.get('caching', {}).get('enabled', False):
            cached_returns = cache.get_expected_returns(assets_tuple)
            if cached_returns is not None:
                expected_returns = cached_returns
            else:
                expected_returns = compute_expected_returns(
                    returns_subset,
                    method=exp_return_settings.get('method', 'historical_mean'),
                    window=estimation_window,
                    use_annualization=exp_return_settings.get('use_annualization', True)
                )
                cache.set_expected_returns(assets_tuple, expected_returns)
        else:
            expected_returns = compute_expected_returns(
                returns_subset,
                method=exp_return_settings.get('method', 'historical_mean'),
                window=estimation_window,
                use_annualization=exp_return_settings.get('use_annualization', True)
            )
        
        # Determine objective and risk aversion
        objective = markowitz_settings.get('objective', 'min_variance')
        risk_aversion = None
        
        if markowitz_settings.get('risk_return_tradeoff', {}).get('use_risk_aversion', False):
            lambda_values = markowitz_settings['risk_return_tradeoff'].get('lambda_values', [1.0])
            risk_aversion = lambda_values[0]  # Use first lambda value
        
        # Optimize portfolio
        optimal_weights, opt_info, solver_time = optimize_portfolio(
            expected_returns,
            cov_matrix,
            objective=objective,
            risk_aversion=risk_aversion,
            risk_free_rate=risk_free_rate,
            constraints=constraints,
            solver=solver_settings.get('qp_backend', 'osqp'),
            tolerance=solver_settings.get('tolerance', 1e-6),
            warm_start=solver_settings.get('warm_start', False)
        )
        
        # Compute portfolio returns
        portfolio_returns = (returns_subset * optimal_weights).sum(axis=1)
        
        # Compute all metrics
        portfolio_stats = compute_portfolio_statistics(
            portfolio_returns,
            optimal_weights,
            expected_returns,
            cov_matrix,
            risk_free_rate
        )
        
        sharpe = compute_sharpe_ratio(portfolio_returns, risk_free_rate)
        sortino = compute_sortino_ratio(portfolio_returns, risk_free_rate)
        max_dd = compute_max_drawdown(portfolio_returns)
        calmar = compute_calmar_ratio(portfolio_returns)
        
        structure_metrics = compute_structure_metrics(
            optimal_weights,
            cov_matrix,
            returns_subset
        )
        
        # Compute risk metrics with multiple confidence levels if specified
        risk_metrics_config = modules.get('metrics', {}).get('risk', {})
        risk_metrics = {}
        if isinstance(risk_metrics_config, dict):
            var_config = risk_metrics_config.get('value_at_risk', {})
            confidence_levels = var_config.get('confidence_levels', [0.95])
            for conf_level in confidence_levels:
                var_metrics = compute_risk_metrics(portfolio_returns, confidence_level=conf_level)
                if conf_level == 0.95:
                    # Default VaR and CVaR use 0.95
                    risk_metrics['value_at_risk'] = var_metrics['value_at_risk']
                    risk_metrics['conditional_value_at_risk'] = var_metrics['conditional_value_at_risk']
                # Also store for other confidence levels
                risk_metrics[f'value_at_risk_{int(conf_level*100)}'] = var_metrics['value_at_risk']
                risk_metrics[f'conditional_value_at_risk_{int(conf_level*100)}'] = var_metrics['conditional_value_at_risk']
            risk_metrics['downside_deviation'] = var_metrics['downside_deviation']
        else:
            risk_metrics = compute_risk_metrics(portfolio_returns)
        
        distribution_metrics = compute_distribution_metrics(portfolio_returns)
        
        # Compute time-sliced metrics if enabled
        time_sliced_metrics_list = []
        if modules.get('time_sliced_metrics', {}).get('enabled', False):
            slice_by_list = modules['time_sliced_metrics'].get('slice_by', ['year'])
            for slice_by in slice_by_list:
                time_slices = compute_time_sliced_metrics(
                    portfolio_returns,
                    optimal_weights,
                    expected_returns,
                    cov_matrix,
                    slice_by=slice_by,
                    risk_free_rate=risk_free_rate
                )
                time_sliced_metrics_list.extend(time_slices)
        
        runtime_total = time.time() - start_time
        
        # Combine all results
        result = {
            'portfolio_id': portfolio_id,
            'objective': objective,
            'risk_aversion': risk_aversion,
            **portfolio_stats,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            **structure_metrics,
            **risk_metrics,
            **distribution_metrics,
            'runtime_per_optimization_ms': runtime_total * 1000,
            'covariance_estimation_time_ms': cov_time,
            'solver_time_ms': solver_time,
            'optimization_status': opt_info.get('status', 'unknown')
        }
        
        # Log optimization status for first few portfolios
        if isinstance(portfolio_id, (int, str)) and str(portfolio_id) in ['0', '1', '2']:
            status = opt_info.get('status', 'unknown')
            if status in ['optimal', 'optimal_inaccurate', 'success']:
                exp_ret = portfolio_stats.get('expected_return', np.nan)
                vol = portfolio_stats.get('volatility', np.nan)
                print(f"  Portfolio {portfolio_id}: Optimization {status} | "
                      f"Return: {exp_ret:.6f} | Vol: {vol:.6f} | "
                      f"Sharpe: {sharpe:.4f} | Time: {solver_time:.2f}ms")
            else:
                print(f"  Portfolio {portfolio_id}: Optimization {status} | "
                      f"Message: {opt_info.get('message', 'N/A')}")
        
        return result, optimal_weights, runtime_total
        
    except Exception as e:
        # Only log errors for first few portfolios to avoid spam
        if isinstance(portfolio_id, (int, str)) and str(portfolio_id) in ['0', '1', '2']:
            print(f"  Portfolio {portfolio_id}: Error - {e}")
        warnings.warn(f"Error optimizing portfolio {portfolio_id}: {e}")
        return {}, pd.Series(dtype=float), 0.0


def run_markowitz_optimization(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    max_portfolios: Optional[int] = 100
) -> Dict:
    """
    Run Markowitz mean-variance optimization pipeline.
    
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
            # Default path
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
    markowitz_settings = config['markowitz_settings']
    modules = config.get('modules', {})
    outputs = config['outputs']
    report_sections = config.get('report_sections', [])
    portfolio_universe_semantics = inputs.get('portfolio_universe_semantics', 'asset_set_only')
    
    print("Loading data...")
    # Load data
    prices = load_panel_prices(inputs['panel_price_path'])
    print(f"  Loaded prices: {len(prices):,} dates, {len(prices.columns):,} assets")
    portfolio_universe = load_portfolio_universe(inputs['portfolio_universe_path'])
    print(f"  Loaded {len(portfolio_universe):,} portfolios from universe")
    risk_free_rate = inputs.get('risk_free_rate', 0.0)
    print(f"  Risk-free rate: {risk_free_rate:.6f}")
    
    print("\nComputing returns...")
    # Compute returns
    daily_returns = compute_daily_returns(prices, method=inputs.get('return_type', 'log'))
    print(f"  Computed {len(daily_returns):,} daily returns using {inputs.get('return_type', 'log')} method")
    
    # Get data period
    data_period = f"{prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}"
    print(f"  Data period: {data_period}")
    
    # Initialize cache if enabled
    cache = None
    cov_settings = markowitz_settings.get('covariance_estimation', {})
    exp_return_settings = markowitz_settings.get('expected_return_estimation', {})
    if (cov_settings.get('caching', {}).get('enabled', False) or 
        exp_return_settings.get('caching', {}).get('enabled', False)):
        cache = MarkowitzCache(enabled=True)
        print("\nCaching enabled for covariance and expected returns")
    
    # Precompute shared statistics if enabled
    precompute_config = modules.get('precompute_shared_statistics', {})
    if precompute_config.get('enabled', False):
        print("\nPrecomputing shared statistics...")
        precompute_stats = precompute_config.get('statistics', [])
        print(f"  Statistics to precompute: {', '.join(precompute_stats)}")
        
        # Get all unique asset sets from portfolio universe
        all_asset_sets = set()
        if isinstance(portfolio_universe, pd.DataFrame):
            for portfolio_id in portfolio_universe.index:
                portfolio_row = portfolio_universe.loc[portfolio_id]
                if isinstance(portfolio_row, pd.Series):
                    assets = [col for col in portfolio_row.index if col in daily_returns.columns]
                else:
                    assets = []
                if len(assets) > 0:
                    all_asset_sets.add(tuple(sorted(assets)))
        
        # If no asset sets found, use full universe
        if len(all_asset_sets) == 0:
            all_asset_sets.add(tuple(sorted(daily_returns.columns.tolist())))
        
        estimation_window = cov_settings.get('estimation_windows', [252])[0] if cov_settings.get('estimation_windows') else None
        use_shrinkage = cov_settings.get('shrinkage', {}).get('use_shrinkage', False)
        shrinkage_method = cov_settings.get('shrinkage', {}).get('method', 'ledoit_wolf') if use_shrinkage else None
        
        precompute_start = time.time()
        for asset_set_tuple in all_asset_sets:
            assets_list = list(asset_set_tuple)
            returns_subset = daily_returns[assets_list]
            
            if 'covariance_matrix' in precompute_stats and cache is not None:
                # Precompute and cache covariance
                if cache.get_covariance_matrix(asset_set_tuple, estimation_window or len(returns_subset), shrinkage_method) is None:
                    cov_matrix, _ = compute_covariance_matrix(
                        returns_subset,
                        method=cov_settings.get('method', 'sample'),
                        window=estimation_window,
                        use_shrinkage=use_shrinkage,
                        shrinkage_method=shrinkage_method
                    )
                    cache.set_covariance_matrix(asset_set_tuple, estimation_window or len(returns_subset), cov_matrix, shrinkage_method)
            
            if 'expected_returns' in precompute_stats and cache is not None:
                # Precompute and cache expected returns
                if cache.get_expected_returns(asset_set_tuple) is None:
                    expected_returns = compute_expected_returns(
                        returns_subset,
                        method=exp_return_settings.get('method', 'historical_mean'),
                        window=estimation_window,
                        use_annualization=exp_return_settings.get('use_annualization', True)
                    )
                    cache.set_expected_returns(asset_set_tuple, expected_returns)
        
        precompute_time = time.time() - precompute_start
        print(f"  Precomputed statistics for {len(all_asset_sets)} unique asset sets in {precompute_time:.2f}s")
    
    # Get portfolio IDs
    if isinstance(portfolio_universe, pd.DataFrame):
        portfolio_ids = portfolio_universe.index.tolist()
    else:
        portfolio_ids = [0]  # Single portfolio case
    
    num_portfolios_total = len(portfolio_ids)
    
    # Limit number of portfolios if specified
    if max_portfolios is not None and max_portfolios > 0:
        num_portfolios = min(num_portfolios_total, max_portfolios)
        portfolio_ids = portfolio_ids[:num_portfolios]
        print(f"\nOptimizing {num_portfolios:,} portfolios (out of {num_portfolios_total:,} total)...")
    else:
        num_portfolios = num_portfolios_total
        print(f"\nOptimizing {num_portfolios:,} portfolios...")
    
    # Log optimization settings
    objective = markowitz_settings.get('objective', 'min_variance')
    solver_settings = markowitz_settings.get('optimization_solver', {})
    solver_backend = solver_settings.get('qp_backend', 'osqp')
    print(f"  Objective: {objective}")
    print(f"  Solver: {solver_backend}")
    if markowitz_settings.get('risk_return_tradeoff', {}).get('use_risk_aversion', False):
        lambda_values = markowitz_settings['risk_return_tradeoff'].get('lambda_values', [])
        print(f"  Risk aversion parameters (λ): {lambda_values}")
    constraints = markowitz_settings.get('constraints', {})
    if constraints.get('max_weight_per_asset'):
        print(f"  Max weight per asset: {constraints['max_weight_per_asset']}")
    
    # Process portfolios
    all_results = []
    all_runtimes = []
    all_weights_dict = {}  # Store weights by portfolio_id
    efficient_frontiers = []
    successful_count = 0
    failed_count = 0
    total_start_time = time.time()
    
    print("=" * 80)
    
    for idx, portfolio_id in enumerate(portfolio_ids, 1):
        # Progress logging every 100 portfolios or at milestones
        if idx % 100 == 0 or idx == 1 or idx == num_portfolios:
            elapsed = time.time() - total_start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (num_portfolios - idx) / rate if rate > 0 else 0
            print(f"[{idx:,}/{num_portfolios:,}] Processing portfolio {portfolio_id} | "
                  f"Success: {successful_count:,} | Failed: {failed_count:,} | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")
        
        result, optimal_weights, runtime = _process_single_optimization(
            daily_returns,
            portfolio_universe,
            portfolio_id,
            markowitz_settings,
            risk_free_rate,
            modules,
            random_seed=markowitz_settings.get('random_seed'),
            cache=cache,
            portfolio_universe_semantics=portfolio_universe_semantics
        )
        
        if result:
            all_results.append(result)
            all_runtimes.append(runtime)
            all_weights_dict[portfolio_id] = optimal_weights
            successful_count += 1
        else:
            failed_count += 1
            if idx % 1000 == 0:  # Log failures periodically
                print(f"  Warning: Portfolio {portfolio_id} optimization failed")
    
    total_time = time.time() - total_start_time
    print("=" * 80)
    print(f"Optimization complete: {successful_count:,} successful, {failed_count:,} failed")
    print(f"Total time: {total_time:.2f}s | Avg time per portfolio: {total_time/num_portfolios:.3f}s")
    
    # Print cache statistics if cache was used
    if cache is not None:
        cache_stats = cache.get_stats()
        print(f"Cache statistics: {cache_stats['cache_hits']:,} hits, {cache_stats['cache_misses']:,} misses, "
              f"hit rate: {cache_stats['hit_rate']:.1%}")
    
    print("=" * 80)
    
    if len(all_results) == 0:
        raise RuntimeError("No portfolios were successfully optimized")
    
    print("\nComputing summary statistics...")
    # Compute summary statistics
    summary_stats = _compute_summary_statistics(all_results, all_runtimes)
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(all_results)
    print(f"  Computed metrics for {len(metrics_df):,} portfolios with {len(metrics_df.columns)} metrics")
    
    # Generate efficient frontier if requested
    if modules.get('generate_efficient_frontier', False):
        print("\nGenerating efficient frontier...")
        try:
            frontier_settings = markowitz_settings.get('frontier_settings', {})
            frontier_method = frontier_settings.get('method', 'target_return_constraint')
            
            # Use first portfolio's assets for frontier, or all assets
            if len(all_weights_dict) > 0:
                first_portfolio_id = portfolio_ids[0]
                if first_portfolio_id in all_weights_dict:
                    assets = all_weights_dict[first_portfolio_id].index.tolist()
                else:
                    assets = daily_returns.columns.tolist()
            else:
                assets = daily_returns.columns.tolist()
            
            returns_subset = daily_returns[assets]
            
            # Check cache for covariance and expected returns
            assets_tuple = tuple(sorted(assets))
            if cache is not None:
                estimation_window = cov_settings.get('estimation_windows', [252])[0] if cov_settings.get('estimation_windows') else None
                use_shrinkage = cov_settings.get('shrinkage', {}).get('use_shrinkage', False)
                shrinkage_method = cov_settings.get('shrinkage', {}).get('method', 'ledoit_wolf') if use_shrinkage else None
                
                cov_matrix = cache.get_covariance_matrix(assets_tuple, estimation_window or len(returns_subset), shrinkage_method)
                expected_returns = cache.get_expected_returns(assets_tuple)
                
                if cov_matrix is None:
                    cov_matrix, _ = compute_covariance_matrix(
                        returns_subset,
                        method=cov_settings.get('method', 'sample'),
                        window=estimation_window,
                        use_shrinkage=use_shrinkage,
                        shrinkage_method=shrinkage_method
                    )
                
                if expected_returns is None:
                    expected_returns = compute_expected_returns(
                        returns_subset,
                        method=exp_return_settings.get('method', 'historical_mean'),
                        window=estimation_window,
                        use_annualization=exp_return_settings.get('use_annualization', True)
                    )
            else:
                cov_matrix, _ = compute_covariance_matrix(
                    returns_subset,
                    method=cov_settings.get('method', 'sample'),
                    use_shrinkage=cov_settings.get('shrinkage', {}).get('use_shrinkage', False),
                    shrinkage_method=cov_settings.get('shrinkage', {}).get('method', 'ledoit_wolf')
                )
                
                expected_returns = compute_expected_returns(
                    returns_subset,
                    method=exp_return_settings.get('method', 'historical_mean'),
                    use_annualization=exp_return_settings.get('use_annualization', True)
                )
            
            frontier = generate_efficient_frontier(
                expected_returns,
                cov_matrix,
                num_portfolios=frontier_settings.get('num_portfolios', 100),
                risk_levels=frontier_settings.get('risk_levels', 'auto'),
                method=frontier_method,
                target_return_grid=frontier_settings.get('target_return_grid', 'linspace'),
                constraints=markowitz_settings.get('constraints', {}),
                risk_free_rate=risk_free_rate,
                solver=markowitz_settings.get('optimization_solver', {}).get('qp_backend', 'osqp'),
                tolerance=markowitz_settings.get('optimization_solver', {}).get('tolerance', 1e-6)
            )
            efficient_frontiers.append(frontier)
            print(f"  Generated {len(frontier):,} frontier points")
        except Exception as e:
            warnings.warn(f"Failed to generate efficient frontier: {e}")
            print(f"  Warning: Failed to generate efficient frontier: {e}")
    
    print("\nSaving results...")
    # Save outputs
    output_base = Path(outputs['metrics_table']).parent
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Save metrics table
    print(f"  Saving metrics to: {outputs['metrics_table']}")
    metrics_df.to_parquet(outputs['metrics_table'])
    print(f"    Saved {len(metrics_df):,} rows, {len(metrics_df.columns)} columns")
    
    # Save optimal portfolios (weights)
    if 'optimal_portfolios' in outputs and len(all_weights_dict) > 0:
        # Create weights DataFrame from stored weights
        # Get all unique assets across all portfolios
        all_assets = set()
        for weights in all_weights_dict.values():
            all_assets.update(weights.index)
        all_assets = sorted(list(all_assets))
        
        # Create DataFrame with portfolio_id as index and assets as columns
        weights_data = {}
        for portfolio_id in metrics_df['portfolio_id']:
            if portfolio_id in all_weights_dict:
                weights = all_weights_dict[portfolio_id]
                weights_data[portfolio_id] = {asset: weights.get(asset, 0.0) for asset in all_assets}
            else:
                weights_data[portfolio_id] = {asset: 0.0 for asset in all_assets}
        
        weights_df = pd.DataFrame(weights_data).T
        weights_df.index.name = 'portfolio_id'
        print(f"  Saving optimal portfolios to: {outputs['optimal_portfolios']}")
        weights_df.to_parquet(outputs['optimal_portfolios'])
        print(f"    Saved weights for {len(weights_df)} portfolios, {len(weights_df.columns)} assets")
    
    # Save efficient frontier
    if 'efficient_frontier' in outputs and len(efficient_frontiers) > 0:
        print(f"  Saving efficient frontier to: {outputs['efficient_frontier']}")
        efficient_frontiers[0].to_parquet(outputs['efficient_frontier'])
        print(f"    Saved {len(efficient_frontiers[0]):,} frontier points")
    
    # Save metrics schema JSON (not full metrics - metrics are in parquet)
    if 'metrics_schema_json' in outputs:
        print(f"  Generating metrics schema...")
        generate_metrics_schema(
            metrics_df,
            outputs['metrics_schema_json']
        )
        print(f"    Saved schema to: {outputs['metrics_schema_json']}")
    
    print("\nGenerating report...")
    # Generate report
    generate_report(
        metrics_df,
        outputs['summary_report'],
        markowitz_settings,
        report_sections
    )
    print(f"  Report saved to: {outputs['summary_report']}")
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION PIPELINE COMPLETE!")
    print("=" * 80)
    
    return {
        'metrics_df': metrics_df,
        'summary_stats': summary_stats,
        'efficient_frontiers': efficient_frontiers,
        'num_portfolios': num_portfolios,
        'num_portfolios_total': num_portfolios_total
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Markowitz Mean-Variance Portfolio Optimization'
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
    
    args = parser.parse_args()
    
    # Convert 0 to None (process all portfolios)
    max_portfolios = None if args.max_portfolios == 0 else args.max_portfolios
    
    # Run optimization
    results = run_markowitz_optimization(
        config_path=args.config,
        max_portfolios=max_portfolios
    )
    print(f"Optimized {results['num_portfolios']:,} portfolios (out of {results['num_portfolios_total']:,} total)")

