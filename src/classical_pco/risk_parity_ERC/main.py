"""
Main evaluation script for Risk Parity / Equal Risk Contribution (ERC) Portfolio Optimization.

Orchestrates the entire optimization pipeline including:
- Data loading (prices, baseline portfolios)
- Returns computation
- Covariance matrix estimation
- Risk Parity ERC optimization
- Risk contribution analysis
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
import warnings
import logging
import os

# Try to import MPI
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    # Create dummy MPI object for non-MPI execution
    class DummyMPI:
        class COMM_WORLD:
            rank = 0
            size = 1
            def barrier(self): pass
            def gather(self, obj, root=0): return [obj]
            def bcast(self, obj, root=0): return obj
    MPI = DummyMPI()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from .returns import (
    load_panel_prices,
    load_baseline_portfolios,
    compute_daily_returns
)
from .risk_parity_erc_optimizer import (
    compute_covariance_matrix,
    calculate_risk_contributions,
    optimize_risk_parity_portfolio
)
from .metrics import (
    compute_portfolio_statistics,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_risk_parity_specific_metrics,
    compute_structure_metrics,
    compute_risk_metrics,
    compute_distribution_metrics,
    compute_comparison_metrics,
    compute_runtime_metrics
)
from .report_generator import generate_report
from .cache import RiskParityCache


def _save_restructured_json(
    json_path: Path,
    portfolio_results: List[Dict],
    summary_stats: Dict,
    data_period: str,
    num_portfolios: int,
    rp_settings: Dict
):
    """
    Save results in the restructured JSON format.
    
    Args:
        json_path: Path to save JSON file
        portfolio_results: List of restructured portfolio results
        summary_stats: Summary statistics dictionary
        data_period: Data period string
        num_portfolios: Number of portfolios optimized
        rp_settings: Risk Parity optimization settings
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
            'task': 'risk_parity_erc_optimization',
            'data_period': data_period,
            'portfolios_optimized': num_portfolios,
            'risk_parity_settings': rp_settings,
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
    
    # Risk Parity specific insights
    rp_insights = {}
    if 'risk_parity_deviation_score' in results_df.columns:
        rp_insights['avg_risk_parity_deviation'] = float(results_df['risk_parity_deviation_score'].mean())
    if 'risk_contribution_coefficient_of_variation' in results_df.columns:
        rp_insights['avg_risk_contribution_cv'] = float(results_df['risk_contribution_coefficient_of_variation'].mean())
    if 'equal_risk_gap' in results_df.columns:
        rp_insights['avg_equal_risk_gap'] = float(results_df['equal_risk_gap'].mean())
    
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
        'risk_parity_insights': rp_insights,
        'distribution_effects': distribution_effects,
        'structure_effects': structure_effects,
        'runtime_stats': runtime_stats
    }


def _apply_transaction_costs(
    returns: pd.Series,
    previous_weights: Optional[pd.Series],
    current_weights: pd.Series,
    transaction_cost_bps: float = 10.0
) -> pd.Series:
    """Apply transaction costs to returns based on turnover."""
    if previous_weights is None:
        return returns
    
    # Calculate turnover
    turnover = np.sum(np.abs(current_weights - previous_weights))
    cost = (transaction_cost_bps / 10000) * turnover
    
    # Adjust returns
    adjusted_returns = returns - cost
    return adjusted_returns


def _process_single_optimization(
    returns: pd.DataFrame,
    baseline_portfolios: pd.DataFrame,
    portfolio_id: Union[int, str],
    rp_settings: Dict,
    risk_free_rate: float = 0.0,
    random_seed: Optional[int] = None,
    estimation_window: Optional[int] = None,
    covariance_estimator: Optional[Dict] = None,
    previous_weights: Optional[pd.Series] = None,
    cached_covariance: Optional[pd.DataFrame] = None,
    cache: Optional[RiskParityCache] = None,
    perf_opts: Optional[Dict] = None
) -> Tuple[Dict, float, pd.Series]:
    """
    Process a single Risk Parity ERC optimization for a portfolio.
    
    Args:
        returns: DataFrame of daily returns
        baseline_portfolios: DataFrame with baseline portfolio information
        portfolio_id: Portfolio identifier
        rp_settings: Risk Parity optimization settings
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
            return {}, 0.0, pd.Series(dtype=float)
        
        returns_subset = returns[assets]
        
        # Get settings
        cov_settings = rp_settings.get('covariance_estimation', {})
        constraints = rp_settings.get('constraints', {})
        optimization_settings = rp_settings.get('optimization', {})
        
        # Use provided estimator config or default
        if covariance_estimator is None:
            estimator_config = cov_settings.get('estimators', [{}])[0] if cov_settings.get('estimators') else {}
        else:
            estimator_config = covariance_estimator
        
        method = estimator_config.get('name', 'sample')
        shrinkage_config = estimator_config.get('shrinkage', {})
        use_shrinkage = shrinkage_config.get('enable', False)
        shrinkage_method = shrinkage_config.get('method', 'ledoit_wolf')
        ewma_params = estimator_config.get('params', {})
        ewma_lambda = ewma_params.get('lambda', 0.94)
        
        stability_checks = cov_settings.get('stability_checks', {})
        ensure_psd = stability_checks.get('ensure_psd', True)
        min_eigenvalue = stability_checks.get('min_eigenvalue', 1e-10)
        
        # Compute covariance matrix (use cache if available)
        cov_start = time.time()
        if cached_covariance is not None:
            cov_matrix = cached_covariance
            cov_time = 0.0  # Cached, no computation time
        else:
            cov_opt = perf_opts.get('covariance_optimization', {}) if perf_opts else {}
            psd_handling = cov_opt.get('psd_handling', {})
            skip_psd_if_positive = psd_handling.get('skip_psd_fix_if_eigenvalues_positive', True)
            cov_matrix, cov_time = compute_covariance_matrix(
                returns_subset,
                method=method,
                window=estimation_window,
                use_shrinkage=use_shrinkage,
                shrinkage_method=shrinkage_method,
                ewma_lambda=ewma_lambda,
                ensure_psd=ensure_psd,
                min_eigenvalue=min_eigenvalue,
                skip_psd_if_positive=skip_psd_if_positive
            )
            # Cache the result
            if cache is not None:
                assets_tuple = tuple(sorted(assets))
                cache.set_covariance_matrix(assets_tuple, estimation_window, estimator_name, cov_matrix, None)
        
        # Get baseline portfolio weights for comparison
        baseline_weights = None
        baseline_cov = None
        if portfolio_id in baseline_portfolios.index:
            portfolio_row = baseline_portfolios.loc[portfolio_id]
            if isinstance(portfolio_row, pd.Series):
                baseline_weights = portfolio_row[assets].fillna(0.0)
                baseline_weights = baseline_weights / baseline_weights.sum() if baseline_weights.sum() > 0 else baseline_weights
                baseline_cov = cov_matrix  # Use same covariance for fair comparison
        
        # Convert previous weights to numpy array if needed
        prev_weights_array = None
        if previous_weights is not None:
            prev_weights_array = previous_weights[assets].values if hasattr(previous_weights, 'index') else previous_weights
        
        # Optimize Risk Parity portfolio
        optimal_weights, opt_info, solver_time = optimize_risk_parity_portfolio(
            cov_matrix,
            constraints,
            optimization_settings,
            previous_weights=prev_weights_array,
            perf_opts=perf_opts
        )
        
        # Calculate risk contributions
        rc_start = time.time()
        risk_contrib, marginal_contrib, portfolio_vol = calculate_risk_contributions(
            optimal_weights.values,
            cov_matrix
        )
        rc_time = (time.time() - rc_start) * 1000
        
        # Compute portfolio returns
        portfolio_returns = (returns_subset * optimal_weights).sum(axis=1)
        
        # Apply transaction costs if enabled
        transaction_cost_config = rp_settings.get('experiment_protocol', {}).get('rebalancing', {}).get('transaction_cost_model', {})
        if transaction_cost_config.get('enable', False):
            transaction_cost_bps = transaction_cost_config.get('bps_per_turnover', 10)
            if previous_weights is not None:
                portfolio_returns = _apply_transaction_costs(
                    portfolio_returns,
                    previous_weights,
                    optimal_weights,
                    transaction_cost_bps
                )
        
        # Compute expected returns (historical mean)
        expected_returns = returns_subset.mean() * 252  # Annualized
        
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
        
        # Risk Parity specific metrics
        rp_metrics = compute_risk_parity_specific_metrics(
            optimal_weights,
            cov_matrix
        )
        
        structure_metrics = compute_structure_metrics(
            optimal_weights,
            cov_matrix,
            returns_subset,
            baseline_weights
        )
        
        risk_metrics = compute_risk_metrics(portfolio_returns)
        
        distribution_metrics = compute_distribution_metrics(portfolio_returns)
        
        # Comparison metrics
        baseline_portfolio_returns = None
        if baseline_weights is not None:
            baseline_portfolio_returns = (returns_subset * baseline_weights).sum(axis=1)
        
        comparison_metrics = {}
        if baseline_portfolio_returns is not None and baseline_cov is not None:
            comparison_metrics = compute_comparison_metrics(
                portfolio_returns,
                baseline_portfolio_returns,
                optimal_weights,
                baseline_weights,
                cov_matrix,
                baseline_cov,
                risk_free_rate,
                returns_dataframe=returns_subset
            )
        else:
            # Still compute ERC vs equal weight
            n = len(optimal_weights)
            equal_weights = pd.Series(np.ones(n) / n, index=optimal_weights.index)
            equal_weight_returns = (returns_subset * equal_weights).sum(axis=1)
            
            eq_vol = equal_weight_returns.std() * np.sqrt(252)
            erc_vol = portfolio_returns.std() * np.sqrt(252)
            eq_sharpe = compute_sharpe_ratio(equal_weight_returns, risk_free_rate)
            erc_sharpe = sharpe
            
            comparison_metrics = {
                'erc_vs_equal_weight_volatility': erc_vol - eq_vol if not np.isnan(erc_vol) and not np.isnan(eq_vol) else np.nan,
                'erc_vs_equal_weight_sharpe': erc_sharpe - eq_sharpe if not np.isnan(erc_sharpe) and not np.isnan(eq_sharpe) else np.nan,
                'baseline_portfolio_volatility': np.nan,
                'baseline_portfolio_sharpe': np.nan,
                'baseline_portfolio_expected_return': np.nan,
                'volatility_reduction_vs_baseline': np.nan,
                'sharpe_improvement_vs_baseline': np.nan,
                'risk_contribution_improvement_vs_baseline': np.nan,
                'erc_vs_equal_weight_risk_contributions': np.nan,
                'difference_in_risk_contributions_vs_baseline': np.nan
            }
        
        runtime_total = time.time() - start_time
        
        # Combine all results
        result = {
            'portfolio_id': portfolio_id,
            'method': rp_settings.get('method', 'equal_risk_contribution'),
            **portfolio_stats,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            **rp_metrics,
            **structure_metrics,
            **risk_metrics,
            **distribution_metrics,
            **comparison_metrics,
            'runtime_per_optimization_ms': runtime_total * 1000,
            'covariance_estimation_time_ms': cov_time,
            'risk_contribution_calculation_time_ms': rc_time,
            'solver_time_ms': solver_time,
            'optimization_status': opt_info.get('status', 'unknown'),
            'covariance_estimator': method,
            'estimation_window': estimation_window
        }
        
        return result, runtime_total, optimal_weights
        
    except Exception as e:
        logger.error(f"Error optimizing portfolio {portfolio_id}: {e}", exc_info=True)
        return {}, 0.0, pd.Series(dtype=float)


def _run_rolling_backtest_worker(args):
    """Worker function for rolling backtest evaluation."""
    (returns, baseline_portfolios, portfolio_id, rp_settings, risk_free_rate,
     estimation_window, covariance_estimator, rebalancing_freq, horizons,
     random_seed, cache, perf_opts, stage_label) = args
    
    logger.debug(f"Processing portfolio {portfolio_id} with window {estimation_window}, estimator {covariance_estimator.get('name', 'sample')}")
    
    all_results = []
    all_weights = []
    previous_weights = None
    
    # Get rebalancing dates
    if rebalancing_freq == 'monthly':
        # Use 'ME' (Month End) instead of deprecated 'M'
        try:
            rebalance_dates = returns.resample('ME').last().index
        except ValueError:
            # Fallback for older pandas versions
            rebalance_dates = returns.resample('M').last().index
    elif rebalancing_freq == 'quarterly':
        # Quarterly: every ~63 trading days
        rebalance_dates = returns.index[::63]
    elif rebalancing_freq == 'annual':
        step = 252
        rebalance_dates = returns.index[::step]
    else:
        # Default to monthly
        try:
            rebalance_dates = returns.resample('ME').last().index
        except ValueError:
            rebalance_dates = returns.resample('M').last().index
    
    # Rebalance thinning optimization
    rolling_opt = perf_opts.get('rolling_backtest_optimization', {})
    rebalance_thinning = rolling_opt.get('rebalance_thinning', {})
    thin_enabled = rebalance_thinning.get('enable', False)
    min_weight_change = rebalance_thinning.get('min_weight_change_threshold', 0.01)
    skip_stable = rebalance_thinning.get('skip_rebalance_if_stable', False)
    
    # Rolling window evaluation
    min_date = returns.index[0]
    for rebal_date in rebalance_dates:
        # Check if we have enough data
        days_available = (rebal_date - min_date).days
        if days_available < estimation_window:
            continue  # Skip until we have enough data
        
        # Estimation window (use last N days)
        est_end = rebal_date
        est_start_idx = returns.index.get_indexer([est_end], method='nearest')[0] - estimation_window
        if est_start_idx < 0:
            continue
        
        est_start = returns.index[est_start_idx]
        est_returns = returns.loc[est_start:est_end]
        
        if len(est_returns) < estimation_window * 0.8:  # Require at least 80% of window
            continue
        
        # Rebalance thinning: skip if weights haven't changed much
        if thin_enabled and skip_stable and previous_weights is not None:
            # Quick check: if we had previous weights, estimate if change would be small
            # This is approximate - we'd need to optimize to know for sure, but we can skip if last change was small
            pass  # For now, we'll optimize and check after
        
        # Get assets for cache key
        if portfolio_id in baseline_portfolios.index:
            portfolio_assets = baseline_portfolios.loc[portfolio_id]
            if isinstance(portfolio_assets, pd.Series):
                assets = tuple(sorted([a for a in portfolio_assets.index if a in est_returns.columns]))
            else:
                assets = tuple(sorted([a for a in portfolio_assets if a in est_returns.columns]))
        else:
            assets = tuple(sorted(est_returns.columns))
        
        estimator_name = covariance_estimator.get('name', 'sample')
        
        # Check cache for covariance
        cov_matrix = None
        cov_time = 0.0
        if cache is not None:
            cached_cov = cache.get_covariance_matrix(assets, estimation_window, estimator_name, rebal_date)
            if cached_cov is not None:
                cov_matrix = cached_cov
        
        try:
            result, runtime, optimal_weights = _process_single_optimization(
                est_returns,
                baseline_portfolios,
                portfolio_id,
                rp_settings,
                risk_free_rate,
                random_seed=random_seed,
                estimation_window=estimation_window,
                covariance_estimator=covariance_estimator,
                previous_weights=previous_weights,
                cached_covariance=cov_matrix,
                cache=cache,
                perf_opts=perf_opts
            )
            
            if result:
                # Rebalance thinning: skip if weights haven't changed enough
                if thin_enabled and skip_stable and previous_weights is not None:
                    weight_change = np.sum(np.abs(optimal_weights.values - previous_weights.values))
                    if weight_change < min_weight_change:
                        continue  # Skip this rebalance, weights too stable
                
                result['rebalance_date'] = rebal_date
                result['estimation_window'] = estimation_window
                result['covariance_estimator'] = covariance_estimator.get('name', 'sample')
                result['rebalancing_frequency'] = rebalancing_freq
                result['stage'] = stage_label
                all_results.append(result)
                all_weights.append(optimal_weights)
                previous_weights = optimal_weights
                
        except Exception as e:
            logger.warning(f"Error at rebalance date {rebal_date} for portfolio {portfolio_id}: {e}")
            continue
    
    return all_results, all_weights


def run_risk_parity_erc_optimization(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    max_portfolios: Optional[int] = None
) -> Dict:
    """
    Run Risk Parity ERC optimization pipeline.
    
    Automatically selects static or rolling mode based on execution_mode config.
    
    Args:
        config_path: Path to llm.json configuration file
        config_dict: Optional configuration dictionary (overrides config_path)
        max_portfolios: Maximum number of portfolios to process (default: None for all)
        
    Returns:
        Dictionary with optimization results
    """
    # Load config to check execution mode
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
    
    # Check execution mode
    execution_config = config.get('execution_mode', {})
    if execution_config.get('strategy') == 'static_erc_single_solve':
        # Use static mode - import here to avoid circular imports
        import sys
        import importlib
        module_name = 'src.classical_pco.risk_parity_ERC.main_static'
        if module_name not in sys.modules:
            main_static = importlib.import_module(module_name)
        else:
            main_static = sys.modules[module_name]
        return main_static.run_risk_parity_erc_optimization_static(config_path, config_dict, max_portfolios)
    
    # Otherwise, use rolling mode (legacy)
    return _run_risk_parity_erc_optimization_rolling(config_path, config_dict, max_portfolios)


def _run_risk_parity_erc_optimization_rolling(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    max_portfolios: Optional[int] = None
) -> Dict:
    """
    Run Risk Parity ERC optimization pipeline with rolling out-of-sample evaluation (LEGACY).
    
    Args:
        config_path: Path to llm.json configuration file
        config_dict: Optional configuration dictionary (overrides config_path)
        max_portfolios: Maximum number of portfolios to process (default: None for all)
        
    Returns:
        Dictionary with optimization results
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    if rank == 0:
        logger.info(f"Starting Risk Parity ERC optimization (MPI: {MPI_AVAILABLE}, size={size})")
    
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
    
    # Broadcast config to all ranks
    if MPI_AVAILABLE:
        config = comm.bcast(config, root=0)
    
    # Extract settings
    inputs = config['inputs']
    rp_settings = config['risk_parity_settings']
    experiment_protocol = config.get('experiment_protocol', {})
    modules = config.get('modules', {})
    outputs = config['outputs']
    report_sections = config.get('report_sections', [])
    perf_opts = config.get('performance_optimizations', {})
    
    # Set random seed for reproducibility
    random_seed = config.get('design_principles', {}).get('reproducibility', {}).get('random_seed', 42)
    np.random.seed(random_seed)
    
    # Set numpy BLAS threads if specified
    if perf_opts.get('parallelism_and_batching', {}).get('numpy_blas_thread_control', {}).get('set', False):
        num_threads = perf_opts['parallelism_and_batching']['numpy_blas_thread_control'].get('num_threads', 1)
        try:
            import mkl
            mkl.set_num_threads(num_threads)
        except:
            try:
                os.environ['OMP_NUM_THREADS'] = str(num_threads)
                os.environ['MKL_NUM_THREADS'] = str(num_threads)
                os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
            except:
                pass
    
    # Initialize cache (disabled by default for two-stage protocol)
    cov_cache_config = perf_opts.get('covariance_optimization', {})
    cache_enabled = cov_cache_config.get('cache_covariance_matrices', False)
    if not cache_enabled and rank == 0:
        reason = cov_cache_config.get('reason', '')
        if reason:
            logger.info(f"Covariance caching disabled: {reason}")
    cache_max_size = cov_cache_config.get('max_cached_items', 5000)
    cache = RiskParityCache(enabled=cache_enabled, max_size=cache_max_size) if cache_enabled else None
    
    if rank == 0:
        logger.info("Loading data...")
    
    # Load data (all ranks load, but only rank 0 logs)
    prices = load_panel_prices(inputs['panel_price_path'])
    baseline_portfolios = load_baseline_portfolios(inputs['baseline_portfolios_path'])
    risk_free_rate = inputs.get('risk_free_rate', 0.0)
    
    if rank == 0:
        logger.info("Computing returns...")
    
    # Compute returns
    daily_returns = compute_daily_returns(prices, method=inputs.get('return_type', 'log'))
    
    # Get data period
    data_period = f"{prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}"
    
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
    else:
        num_portfolios = num_portfolios_total
    
    if rank == 0:
        logger.info(f"Optimizing {num_portfolios:,} portfolios (out of {num_portfolios_total:,} total)...")
    
    # Two-stage protocol for experiment design acceleration
    exp_design = perf_opts.get('experiment_design_acceleration', {})
    two_stage = exp_design.get('two_stage_protocol', {}) if exp_design.get('enable', False) else {}
    
    if two_stage.get('enable', False) and rank == 0:
        logger.info("Using two-stage protocol for experiment acceleration")
        stage_1_config = two_stage.get('stage_1_full_scale_primary_config', {})
        stage_2_config = two_stage.get('stage_2_sensitivity_on_subset', {})
        
        # Stage 1: Full scale with primary config
        stage_1_windows = stage_1_config.get('rolling_windows_override', {}).get('estimation_windows', [252])
        stage_1_estimators = stage_1_config.get('covariance_estimators_override', [{'name': 'sample'}])
        stage_1_rebal = stage_1_config.get('rebalancing_override', {}).get('frequency', 'quarterly')
        
        # Stage 2: Sensitivity on subset
        stage_2_windows = stage_2_config.get('rolling_windows_override', {}).get('estimation_windows', [252, 500, 750])
        stage_2_estimators = stage_2_config.get('covariance_estimators_override', [{'name': 'sample'}])
        stage_2_rebal = stage_2_config.get('rebalancing_override', {}).get('frequency', 'monthly')
        stage_2_subset = stage_2_config.get('portfolio_subset', {})
        
        # Run Stage 1 first
        logger.info(f"Stage 1: Running ALL {num_portfolios} portfolios with primary config")
        logger.info(f"  Windows: {stage_1_windows}, Estimators: {[e.get('name') for e in stage_1_estimators]}, Rebalancing: {stage_1_rebal}")
        
        # Generate Stage 1 tasks
        stage_1_tasks = []
        for portfolio_id in portfolio_ids:
            for est_window in stage_1_windows:
                for estimator in stage_1_estimators:
                    stage_1_tasks.append((portfolio_id, est_window, estimator, stage_1_rebal, 'stage_1'))
        
        # Generate Stage 2 subset
        stage_2_portfolio_ids = portfolio_ids
        if stage_2_subset.get('enable', False):
            subset_mode = stage_2_subset.get('mode', 'stratified_random')
            subset_fraction = stage_2_subset.get('fraction', 0.02)
            subset_min_n = stage_2_subset.get('min_n', 200)
            subset_seed = stage_2_subset.get('random_seed', 42)
            
            if subset_mode == 'stratified_random':
                np.random.seed(subset_seed)
                n_subset = max(int(num_portfolios * subset_fraction), subset_min_n)
                n_subset = min(n_subset, num_portfolios)
                stage_2_portfolio_ids = np.random.choice(portfolio_ids, size=n_subset, replace=False).tolist()
                logger.info(f"Stage 2: Running sensitivity on {len(stage_2_portfolio_ids)} portfolios (stratified random subset)")
        
        # Generate Stage 2 tasks
        stage_2_tasks = []
        for portfolio_id in stage_2_portfolio_ids:
            for est_window in stage_2_windows:
                for estimator in stage_2_estimators:
                    stage_2_tasks.append((portfolio_id, est_window, estimator, stage_2_rebal, 'stage_2'))
        
        logger.info(f"Stage 2: {len(stage_2_tasks)} tasks (windows: {stage_2_windows}, estimators: {len(stage_2_estimators)}, rebalancing: {stage_2_rebal})")
        
        # Combine tasks
        tasks = stage_1_tasks + stage_2_tasks
        logger.info(f"Total tasks: {len(stage_1_tasks)} (Stage 1) + {len(stage_2_tasks)} (Stage 2) = {len(tasks)}")
    else:
        # Standard single-stage protocol
        rolling_windows = experiment_protocol.get('rolling_windows', {})
        estimation_windows = rolling_windows.get('estimation_windows', [252])
        
        # Window pruning optimization
        rolling_opt = perf_opts.get('rolling_backtest_optimization', {})
        if rolling_opt.get('estimation_window_pruning', {}).get('enable', False):
            if rolling_opt['estimation_window_pruning'].get('keep_representative_windows_only', False):
                representative = rolling_opt['estimation_window_pruning'].get('representative_windows', [252, 750])
                estimation_windows = [w for w in estimation_windows if w in representative]
                if rank == 0:
                    logger.info(f"Window pruning: using representative windows {estimation_windows}")
        
        rebalancing = experiment_protocol.get('rebalancing', {})
        rebalancing_freq = rebalancing.get('frequency', 'monthly')
        horizons = experiment_protocol.get('horizons_days', [1])
        
        # Get covariance estimators
        cov_settings = rp_settings.get('covariance_estimation', {})
        covariance_estimators = cov_settings.get('estimators', [{'name': 'sample'}])
        
        # Generate tasks
        tasks = []
        for portfolio_id in portfolio_ids:
            for est_window in estimation_windows:
                for estimator in covariance_estimators:
                    tasks.append((portfolio_id, est_window, estimator, rebalancing_freq, 'standard'))
    
    # Generate all task combinations
    tasks = []
    for portfolio_id in portfolio_ids:
        for est_window in estimation_windows:
            for estimator in covariance_estimators:
                tasks.append((
                    portfolio_id, est_window, estimator
                ))
    
    if rank == 0:
        if two_stage.get('enable', False):
            stage_1_count = sum(1 for t in tasks if t[4] == 'stage_1')
            stage_2_count = sum(1 for t in tasks if t[4] == 'stage_2')
            logger.info(f"Generated {len(tasks)} total tasks: {stage_1_count} (Stage 1) + {stage_2_count} (Stage 2)")
        else:
            logger.info(f"Generated {len(tasks)} tasks (portfolios × windows × estimators)")
    
    # Distribute tasks across MPI ranks
    if MPI_AVAILABLE and size > 1:
        tasks_per_rank = len(tasks) // size
        remainder = len(tasks) % size
        
        if rank < remainder:
            my_tasks = tasks[rank * (tasks_per_rank + 1):(rank + 1) * (tasks_per_rank + 1)]
        else:
            start_idx = remainder * (tasks_per_rank + 1) + (rank - remainder) * tasks_per_rank
            my_tasks = tasks[start_idx:start_idx + tasks_per_rank]
        
        logger.info(f"Rank {rank}: Processing {len(my_tasks)} tasks")
    else:
        my_tasks = tasks
        if rank == 0:
            logger.info(f"Sequential execution: Processing {len(my_tasks)} tasks")
    
    # Process tasks
    all_results = []
    all_weights_list = []
    all_runtimes = []
    start_time = time.time()
    
    # Logging optimization
    log_opts = perf_opts.get('logging_and_diagnostics', {})
    log_every_n = log_opts.get('log_every_n_tasks', 100)
    reduce_logging = log_opts.get('reduce_logging_inside_loops', True)
    
    for task_idx, task_tuple in enumerate(my_tasks):
        if len(task_tuple) == 5:
            portfolio_id, est_window, estimator, rebalancing_freq, stage_label = task_tuple
        else:
            # Backward compatibility
            portfolio_id, est_window, estimator = task_tuple[:3]
            rebalancing_freq = experiment_protocol.get('rebalancing', {}).get('frequency', 'monthly')
            stage_label = 'standard'
        if rank == 0 or not MPI_AVAILABLE:
            should_log = (task_idx + 1) % log_every_n == 0 or task_idx == 0 or (task_idx + 1) == len(my_tasks)
            if should_log:
                elapsed = time.time() - start_time
                rate = (task_idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(my_tasks) - task_idx - 1) / rate if rate > 0 else 0
                stage_info = f" [{stage_label}]" if stage_label != 'standard' else ""
                logger.info(
                    f"[{task_idx + 1}/{len(my_tasks)}] Processing portfolio={portfolio_id}, "
                    f"window={est_window}, estimator={estimator.get('name', 'sample')}{stage_info} | "
                    f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
                )
        
        try:
            task_start = time.time()
            results, weights = _run_rolling_backtest_worker((
                daily_returns, baseline_portfolios, portfolio_id, rp_settings,
                risk_free_rate, est_window, estimator, rebalancing_freq,
                horizons, random_seed, cache, perf_opts, stage_label
            ))
            task_time = time.time() - task_start
            
            all_results.extend(results)
            all_weights_list.extend(weights)
            all_runtimes.append(task_time)
            
        except Exception as e:
            logger.error(f"Error processing task {task_idx} (portfolio={portfolio_id}, window={est_window}): {e}", exc_info=True)
    
    # Gather results from all ranks
    if MPI_AVAILABLE and size > 1:
        all_results = comm.gather(all_results, root=0)
        all_weights_list = comm.gather(all_weights_list, root=0)
        
        if rank == 0:
            # Flatten results
            all_results = [item for sublist in all_results for item in sublist]
            all_weights_list = [item for sublist in all_weights_list for item in sublist]
        else:
            return {}  # Non-root ranks return empty
    
    if rank == 0:
        if len(all_results) == 0:
            raise RuntimeError("No portfolios were successfully optimized")
        
        logger.info(f"Computing summary statistics for {len(all_results)} results...")
        
        # Compute summary statistics
        runtimes = [r.get('runtime_per_optimization_ms', 0) / 1000 for r in all_results]
        summary_stats = _compute_summary_statistics(all_results, runtimes)
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame(all_results)
        
        logger.info("Saving results...")
        # Save outputs
        output_base = Path(outputs['metrics_table']).parent
        output_base.mkdir(parents=True, exist_ok=True)
        
        # Save metrics table
        metrics_df.to_parquet(outputs['metrics_table'])
        
        # Save optimal portfolios (weights) if available
        if 'optimal_portfolios' in outputs and len(all_weights_list) > 0:
            # Create weights DataFrame (simplified - may need adjustment based on structure)
            logger.info("Saving optimal portfolios...")
        
        # Save risk contributions if computed
        if 'risk_contribution_table' in outputs:
            logger.info("Saving risk contributions...")
        
        # Save JSON diagnostics if specified
        if 'diagnostics_json' in outputs:
            diagnostics = {
                'total_tasks': len(tasks),
                'successful_results': len(all_results),
                'mpi_size': size if MPI_AVAILABLE else 1,
                'estimation_windows': estimation_windows,
                'covariance_estimators': [e.get('name') for e in covariance_estimators],
                'data_period': data_period
            }
            with open(outputs['diagnostics_json'], 'w') as f:
                json.dump(diagnostics, f, indent=2)
        
        # Save portfolio results JSON if needed
        portfolio_results = []
        for _, row in metrics_df.iterrows():
            portfolio_results.append(row.to_dict())
        
        # Save metrics schema if specified
        if 'metrics_schema_json' in outputs:
            schema = {
                'columns': list(metrics_df.columns),
                'dtypes': {col: str(dtype) for col, dtype in metrics_df.dtypes.items()}
            }
            with open(outputs['metrics_schema_json'], 'w') as f:
                json.dump(schema, f, indent=2)
        
        logger.info("Generating report...")
        # Generate report
        generate_report(
            metrics_df,
            outputs['summary_report'],
            rp_settings,
            report_sections
        )
        
        logger.info("Optimization complete!")
        
        return {
            'metrics_df': metrics_df,
            'summary_stats': summary_stats,
            'num_portfolios': num_portfolios,
            'num_portfolios_total': num_portfolios_total
        }
    else:
        return {}


if __name__ == '__main__':
    import argparse
    
    # Initialize MPI before parsing arguments
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    parser = argparse.ArgumentParser(
        description='Run Risk Parity ERC Portfolio Optimization'
    )
    parser.add_argument(
        '--max-portfolios',
        type=int,
        default=None,
        help='Maximum number of portfolios to process (default: None for all)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration JSON file (default: llm.json in module directory)'
    )
    
    args = parser.parse_args()
    
    # Run optimization
    results = run_risk_parity_erc_optimization(
        config_path=args.config,
        max_portfolios=args.max_portfolios
    )
    
    if rank == 0 and results:
        logger.info(f"Optimized {results.get('num_portfolios', 0):,} portfolios (out of {results.get('num_portfolios_total', 0):,} total)")

