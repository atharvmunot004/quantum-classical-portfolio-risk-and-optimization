"""
Static ERC single-solve execution mode - simplified and fast.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import time
import logging
import os

# Try to import MPI
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    class DummyMPI:
        class COMM_WORLD:
            rank = 0
            size = 1
            def barrier(self): pass
            def gather(self, obj, root=0): return [obj]
            def bcast(self, obj, root=0): return obj
    MPI = DummyMPI()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from .returns import load_panel_prices, load_baseline_portfolios, compute_daily_returns
from .risk_parity_erc_optimizer import (
    compute_covariance_matrix,
    calculate_risk_contributions,
    optimize_risk_parity_portfolio_fixed_point
)
from .fast_cache import AssetSetCache
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
    compute_comparison_metrics
)
from .report_generator import generate_report
from .portfolio_selection import select_portfolios, save_selection_metadata, check_convergence


def _process_single_portfolio_static(
    returns_matrix: np.ndarray,
    returns_columns: np.ndarray,
    column_index_map: Dict[str, int],
    baseline_portfolios: pd.DataFrame,
    portfolio_id: Union[int, str],
    rp_settings: Dict,
    risk_free_rate: float,
    execution_config: Dict,
    asset_cache: Optional[AssetSetCache] = None
) -> Tuple[Dict, pd.Series]:
    """
    Process single portfolio in static mode: one optimization, apply to full series.
    
    Args:
        returns_matrix: Precomputed numpy array of returns (days × assets)
        returns_columns: Column names array
        column_index_map: Mapping from asset name to column index
        baseline_portfolios: Baseline portfolios DataFrame
        portfolio_id: Portfolio identifier
        rp_settings: Risk parity settings
        risk_free_rate: Risk-free rate
        execution_config: Execution mode configuration
        asset_cache: Optional asset set cache for reusing computations
        
    Returns:
        Tuple of (result dictionary, optimal weights Series)
    """
    start_time = time.time()
    max_runtime_ms = execution_config.get('performance_guardrails', {}).get('max_runtime_per_portfolio_ms', 5)
    fail_fast = execution_config.get('performance_guardrails', {}).get('fail_fast_if_exceeded', False)
    
    fast_engine = execution_config.get('fast_engine', {})
    use_numpy = fast_engine.get('data_representation', {}).get('use_numpy_backend', True)
    
    try:
        # Get assets for this portfolio
        if portfolio_id in baseline_portfolios.index:
            portfolio_assets = baseline_portfolios.loc[portfolio_id]
            if isinstance(portfolio_assets, pd.Series):
                assets = portfolio_assets.index.tolist()
            else:
                assets = portfolio_assets
        else:
            assets = returns_columns.tolist()
        
        # Filter to assets available in returns
        assets = [a for a in assets if a in column_index_map]
        if len(assets) == 0:
            return {}, pd.Series(dtype=float)
        
        assets_tuple = tuple(sorted(assets))
        
        # Get asset indices for numpy operations
        asset_indices = np.array([column_index_map[a] for a in assets])
        n_assets = len(assets)
        
        # Get covariance from cache or compute
        cov_config = fast_engine.get('covariance_estimation', {})
        cache_config = fast_engine.get('asset_set_level_caching', {})
        cache_policy = cache_config.get('covariance_cache_policy', {})
        estimation_window = cache_policy.get('estimation_window', 252)
        estimator_name = cov_config.get('estimator', 'ledoit_wolf')
        
        # Check cache
        cov_matrix = None
        if asset_cache and cache_config.get('enable', False):
            cov_matrix = asset_cache.get(assets_tuple, 'covariance_matrix')
        
        if cov_matrix is None:
            # Extract returns for estimation window (last N days)
            n_days = returns_matrix.shape[0]
            start_idx = max(0, n_days - estimation_window)
            est_returns_matrix = returns_matrix[start_idx:, asset_indices]
            
            # Compute covariance using numpy (fast path)
            if estimator_name == 'ledoit_wolf' or estimator_name == 'sample_shrinkage_ledoit_wolf':
                # Use sklearn for Ledoit-Wolf (fast closed-form)
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf()
                cov_array = lw.fit(est_returns_matrix).covariance_
            else:
                # Sample covariance
                cov_array = np.cov(est_returns_matrix.T)
            
            # Ensure PSD
            psd_config = cov_config.get('psd_handling', {})
            if psd_config.get('ensure_psd', True):
                skip_if_positive = psd_config.get('skip_psd_if_eigenvalues_positive', True)
                min_eig = psd_config.get('min_eigenvalue', 1e-10)
                
                if skip_if_positive:
                    if np.all(np.diag(cov_array) > 0):
                        min_eigval = np.linalg.eigvalsh(cov_array)[0]
                        if min_eigval >= min_eig:
                            pass  # Already PSD
                        else:
                            # Fix PSD
                            eigenvals, eigenvecs = np.linalg.eigh(cov_array)
                            eigenvals = np.maximum(eigenvals, min_eig)
                            cov_array = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                else:
                    # Always fix
                    eigenvals, eigenvecs = np.linalg.eigh(cov_array)
                    eigenvals = np.maximum(eigenvals, min_eig)
                    cov_array = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Convert to DataFrame for compatibility
            cov_matrix = pd.DataFrame(cov_array, index=assets, columns=assets)
            
            # Cache it
            if asset_cache and cache_config.get('enable', False):
                asset_cache.set(assets_tuple, 'covariance_matrix', cov_matrix)
        
        # Get constraints
        constraints = rp_settings.get('constraints', {})
        
        # Get solver config (from fast_engine if available)
        erc_solver = fast_engine.get('erc_solver', {})
        max_iter = erc_solver.get('max_iterations', 50)
        conv_tol = erc_solver.get('convergence_tol', 1e-6)
        early_stop = erc_solver.get('early_stop_on_risk_parity_deviation', {})
        early_stop_enabled = early_stop.get('enable', True)
        early_stop_threshold = early_stop.get('threshold', 5e-4)
        
        # Optimize using fixed-point algorithm
        optimal_weights, opt_info, solver_time = optimize_risk_parity_portfolio_fixed_point(
            cov_matrix,
            constraints,
            max_iterations=max_iter,
            convergence_tol=conv_tol,
            early_stop_threshold=early_stop_threshold if early_stop_enabled else None
        )
        
        # Check performance guardrail
        total_time_ms = (time.time() - start_time) * 1000
        if total_time_ms > max_runtime_ms and fail_fast:
            logger.warning(f"Portfolio {portfolio_id} exceeded max runtime ({total_time_ms:.2f}ms > {max_runtime_ms}ms)")
            return {}, pd.Series(dtype=float)
        
        # Get cached intermediates or compute
        metrics_exec = fast_engine.get('metrics_execution', {})
        reuse = metrics_exec.get('reuse_intermediates', {})
        
        # Portfolio returns (reuse if cached)
        portfolio_returns_array = None
        if asset_cache and reuse.get('reuse_portfolio_returns', True):
            portfolio_returns_array = asset_cache.get(assets_tuple, 'portfolio_returns')
        
        if portfolio_returns_array is None:
            # Apply static weights to FULL return series using numpy
            weights_array = optimal_weights.values
            portfolio_returns_array = returns_matrix[:, asset_indices] @ weights_array
            
            # Cache it
            if asset_cache and reuse.get('reuse_portfolio_returns', True):
                asset_cache.set(assets_tuple, 'portfolio_returns', portfolio_returns_array)
        
        portfolio_returns = pd.Series(portfolio_returns_array, index=pd.RangeIndex(len(portfolio_returns_array)))
        
        # Calculate risk contributions (reuse if cached)
        risk_contrib = None
        if asset_cache and reuse.get('reuse_risk_contributions', True):
            cached_rc = asset_cache.get(assets_tuple, 'risk_contributions')
            if cached_rc is not None:
                risk_contrib, marginal_contrib, portfolio_vol = cached_rc
        
        if risk_contrib is None:
            risk_contrib, marginal_contrib, portfolio_vol = calculate_risk_contributions(
                optimal_weights.values,
                cov_matrix
            )
            # Cache it
            if asset_cache and reuse.get('reuse_risk_contributions', True):
                asset_cache.set(assets_tuple, 'risk_contributions', (risk_contrib, marginal_contrib, portfolio_vol))
        
        # Expected returns (reuse if cached)
        expected_returns = None
        if asset_cache and cache_config.get('cache_objects', {}).get('expected_returns_vector', True):
            expected_returns = asset_cache.get(assets_tuple, 'expected_returns')
        
        if expected_returns is None:
            # Compute from full series using numpy
            expected_returns_array = np.mean(returns_matrix[:, asset_indices], axis=0) * 252
            expected_returns = pd.Series(expected_returns_array, index=assets)
            
            # Cache it
            if asset_cache:
                asset_cache.set(assets_tuple, 'expected_returns', expected_returns)
        
        # Compute all metrics (once, after full series)
        metrics_config = execution_config.get('metrics_protocol', {})
        
        portfolio_stats = {}
        if metrics_config.get('portfolio_quality', True):
            portfolio_stats = compute_portfolio_statistics(
                portfolio_returns,
                optimal_weights,
                expected_returns,
                cov_matrix,
                risk_free_rate
            )
            portfolio_stats['sharpe_ratio'] = compute_sharpe_ratio(portfolio_returns, risk_free_rate)
            portfolio_stats['sortino_ratio'] = compute_sortino_ratio(portfolio_returns, risk_free_rate)
            portfolio_stats['max_drawdown'] = compute_max_drawdown(portfolio_returns)
            portfolio_stats['calmar_ratio'] = compute_calmar_ratio(portfolio_returns)
        
        rp_metrics = {}
        if metrics_config.get('risk_metrics', True):
            rp_metrics = compute_risk_parity_specific_metrics(optimal_weights, cov_matrix)
        
        structure_metrics = {}
        if metrics_config.get('structure_metrics', True):
            # Get pairwise correlation (cached if available)
            pairwise_corr = None
            if asset_cache and cache_config.get('cache_objects', {}).get('pairwise_correlation_mean', True):
                pairwise_corr = asset_cache.get(assets_tuple, 'pairwise_correlation_mean')
            
            if pairwise_corr is None and n_assets > 1:
                # Compute correlation from covariance
                cov_array = cov_matrix.values
                std_array = np.sqrt(np.diag(cov_array))
                corr_array = cov_array / np.outer(std_array, std_array)
                # Get upper triangle (excluding diagonal)
                upper_triangle = corr_array[np.triu_indices_from(corr_array, k=1)]
                pairwise_corr = float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else np.nan
                
                # Cache it
                if asset_cache:
                    asset_cache.set(assets_tuple, 'pairwise_correlation_mean', pairwise_corr)
            
            # Compute structure metrics
            structure_metrics = compute_structure_metrics(
                optimal_weights,
                cov_matrix,
                pd.DataFrame(returns_matrix[:, asset_indices], columns=assets),  # For compatibility
                None
            )
            
            if pairwise_corr is not None:
                structure_metrics['pairwise_correlation_mean'] = pairwise_corr
        
        risk_metrics = {}
        if metrics_config.get('risk_metrics', True):
            risk_metrics = compute_risk_metrics(portfolio_returns)
        
        distribution_metrics = {}
        if metrics_config.get('distribution_metrics', True):
            dist_impl = metrics_exec.get('distribution_metrics', {}).get('implementation', 'numpy_fast_path')
            if dist_impl == 'numpy_fast_path':
                # Fast numpy implementation
                returns_clean = portfolio_returns_array[~np.isnan(portfolio_returns_array)]
                if len(returns_clean) >= 3:
                    from scipy import stats
                    distribution_metrics = {
                        'skewness': float(stats.skew(returns_clean)),
                        'kurtosis': float(stats.kurtosis(returns_clean))
                    }
                    if metrics_exec.get('distribution_metrics', {}).get('keep_jarque_bera', True):
                        jb_stat, jb_pvalue = stats.jarque_bera(returns_clean)
                        distribution_metrics['jarque_bera_p_value'] = float(jb_pvalue)
                else:
                    distribution_metrics = {'skewness': np.nan, 'kurtosis': np.nan, 'jarque_bera_p_value': np.nan}
            else:
                distribution_metrics = compute_distribution_metrics(portfolio_returns)
        
        comparison_metrics = {}
        if metrics_config.get('comparison_metrics', True):
            # Equal weight returns (reuse if cached)
            equal_weight_returns_array = None
            if asset_cache and reuse.get('reuse_equal_weight_returns', True) and metrics_exec.get('comparison_metrics', {}).get('avoid_recomputing_equal_weight', True):
                equal_weight_returns_array = asset_cache.get(assets_tuple, 'equal_weight_returns')
            
            if equal_weight_returns_array is None:
                # Compute equal weight returns using numpy
                n = len(optimal_weights)
                equal_weights_array = np.ones(n) / n
                equal_weight_returns_array = returns_matrix[:, asset_indices] @ equal_weights_array
                
                # Cache it
                if asset_cache and reuse.get('reuse_equal_weight_returns', True):
                    asset_cache.set(assets_tuple, 'equal_weight_returns', equal_weight_returns_array)
            
            equal_weight_returns = pd.Series(equal_weight_returns_array, index=portfolio_returns.index)
            
            eq_vol = np.std(equal_weight_returns_array) * np.sqrt(252)
            erc_vol = np.std(portfolio_returns_array) * np.sqrt(252)
            eq_sharpe = compute_sharpe_ratio(equal_weight_returns, risk_free_rate)
            erc_sharpe = portfolio_stats.get('sharpe_ratio', np.nan)
            
            comparison_metrics = {
                'erc_vs_equal_weight_volatility': erc_vol - eq_vol if not np.isnan(erc_vol) and not np.isnan(eq_vol) else np.nan,
                'erc_vs_equal_weight_sharpe': erc_sharpe - eq_sharpe if not np.isnan(erc_sharpe) and not np.isnan(eq_sharpe) else np.nan
            }
        
        # Combine all results
        result = {
            'portfolio_id': portfolio_id,
            'method': 'equal_risk_contribution',
            **portfolio_stats,
            **rp_metrics,
            **structure_metrics,
            **risk_metrics,
            **distribution_metrics,
            **comparison_metrics,
            'runtime_per_optimization_ms': total_time_ms,
            'solver_time_ms': solver_time,
            'optimization_status': opt_info.get('status', 'unknown'),
            'optimization_iterations': opt_info.get('iterations', 0),
            'risk_parity_deviation': opt_info.get('risk_parity_deviation', np.nan),
            'covariance_estimator': estimator_name,
            'estimation_window': estimation_window
        }
        
        return result, optimal_weights
        
    except Exception as e:
        logger.error(f"Error optimizing portfolio {portfolio_id}: {e}", exc_info=True)
        return {}, pd.Series(dtype=float)


def run_risk_parity_erc_optimization_static(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    max_portfolios: Optional[int] = None
) -> Dict:
    """
    Run Risk Parity ERC optimization in static single-solve mode.
    
    Args:
        config_path: Path to llm.json configuration file
        config_dict: Optional configuration dictionary
        max_portfolios: Maximum number of portfolios to process
        
    Returns:
        Dictionary with optimization results
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    if rank == 0:
        logger.info(f"Starting Risk Parity ERC optimization (Static Mode, MPI: {MPI_AVAILABLE}, size={size})")
    
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
    
    if MPI_AVAILABLE:
        config = comm.bcast(config, root=0)
    
    # Extract settings
    inputs = config['inputs']
    rp_settings = config['risk_parity_settings']
    outputs = config['outputs']
    report_sections = config.get('report_sections', [])
    execution_config = config.get('execution_mode', {})
    
    # Check execution mode
    if execution_config.get('strategy') != 'static_erc_single_solve':
        logger.warning("Execution mode is not 'static_erc_single_solve', but using static mode anyway")
    
    # Set random seed
    random_seed = config.get('design_principles', {}).get('reproducibility', {}).get('random_seed', 42)
    np.random.seed(random_seed)
    
    if rank == 0:
        logger.info("Loading data...")
    
    # Load data
    prices = load_panel_prices(inputs['panel_price_path'])
    baseline_portfolios = load_baseline_portfolios(inputs['baseline_portfolios_path'])
    risk_free_rate = inputs.get('risk_free_rate', 0.0)
    
    if rank == 0:
        logger.info("Computing returns...")
    
    # Compute returns
    daily_returns = compute_daily_returns(prices, method=inputs.get('return_type', 'log'))
    
    # Fast engine: precompute numpy matrix and column index map
    fast_engine = execution_config.get('fast_engine', {})
    data_repr = fast_engine.get('data_representation', {})
    use_numpy = data_repr.get('use_numpy_backend', True)
    
    if use_numpy:
        # Precompute returns matrix once
        returns_matrix = daily_returns.values.astype(np.float32)  # Use float32 for speed
        returns_columns = daily_returns.columns.values
        column_index_map = {col: idx for idx, col in enumerate(returns_columns)}
        
        if rank == 0:
            logger.info(f"Precomputed returns matrix: {returns_matrix.shape[0]} days × {returns_matrix.shape[1]} assets")
    else:
        returns_matrix = None
        returns_columns = None
        column_index_map = None
    
    # Initialize asset set cache
    cache_config = fast_engine.get('asset_set_level_caching', {})
    asset_cache = AssetSetCache(
        enabled=cache_config.get('enable', True),
        max_size=cache_config.get('covariance_cache_policy', {}).get('max_cached_asset_sets', 200000)
    ) if cache_config.get('enable', True) else None
    
    # Portfolio selection protocol (on rank 0, then broadcast)
    selection_config = config.get('portfolio_selection_protocol', {})
    use_selection = selection_config.get('main_experiment_sample', {}).get('enable', False)
    
    if use_selection:
        if rank == 0:
            logger.info("Applying portfolio selection protocol...")
            selected_portfolios = select_portfolios(baseline_portfolios, daily_returns, selection_config)
            
            # Use main sample for optimization
            if 'main_sample' in selected_portfolios:
                baseline_portfolios = selected_portfolios['main_sample']
                
                # Save selection metadata
                if selection_config.get('reporting_and_reproducibility', {}).get('save_selection_metadata', True):
                    save_selection_metadata(
                        selected_portfolios,
                        selection_config,
                        selection_config.get('reporting_and_reproducibility', {})
                    )
            else:
                logger.warning("Portfolio selection enabled but no main sample generated, using all portfolios")
        
        # Broadcast selected portfolio IDs to all ranks
        if MPI_AVAILABLE and size > 1:
            if rank == 0:
                portfolio_ids_to_broadcast = baseline_portfolios.index.tolist()
            else:
                portfolio_ids_to_broadcast = None
            
            portfolio_ids_to_broadcast = comm.bcast(portfolio_ids_to_broadcast, root=0)
            
            if rank != 0:
                # Filter baseline_portfolios to selected IDs
                baseline_portfolios = baseline_portfolios.loc[baseline_portfolios.index.isin(portfolio_ids_to_broadcast)]
    
    # Get portfolio IDs
    if isinstance(baseline_portfolios, pd.DataFrame):
        portfolio_ids = baseline_portfolios.index.tolist()
    else:
        portfolio_ids = [0]
    
    num_portfolios_total = len(portfolio_ids)
    
    if max_portfolios is not None and max_portfolios > 0:
        num_portfolios = min(num_portfolios_total, max_portfolios)
        portfolio_ids = portfolio_ids[:num_portfolios]
    else:
        num_portfolios = num_portfolios_total
    
    if rank == 0:
        logger.info(f"Processing {num_portfolios:,} portfolios in static mode...")
    
    # Distribute portfolios across MPI ranks
    if MPI_AVAILABLE and size > 1:
        portfolios_per_rank = num_portfolios // size
        remainder = num_portfolios % size
        
        if rank < remainder:
            my_portfolios = portfolio_ids[rank * (portfolios_per_rank + 1):(rank + 1) * (portfolios_per_rank + 1)]
        else:
            start_idx = remainder * (portfolios_per_rank + 1) + (rank - remainder) * portfolios_per_rank
            my_portfolios = portfolio_ids[start_idx:start_idx + portfolios_per_rank]
        
        logger.info(f"Rank {rank}: Processing {len(my_portfolios)} portfolios")
    else:
        my_portfolios = portfolio_ids
    
    # Process portfolios
    all_results = []
    all_weights = []
    start_time = time.time()
    
    for idx, portfolio_id in enumerate(my_portfolios):
        if rank == 0 or not MPI_AVAILABLE:
            if (idx + 1) % 100 == 0 or idx == 0 or (idx + 1) == len(my_portfolios):
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(my_portfolios) - idx - 1) / rate if rate > 0 else 0
                logger.info(f"[{idx + 1}/{len(my_portfolios)}] Processing portfolio={portfolio_id} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
        
        result, weights = _process_single_portfolio_static(
            returns_matrix,
            returns_columns,
            column_index_map,
            baseline_portfolios,
            portfolio_id,
            rp_settings,
            risk_free_rate,
            execution_config,
            asset_cache
        )
        
        if result:
            all_results.append(result)
            all_weights.append(weights)
    
    # Gather results
    if MPI_AVAILABLE and size > 1:
        all_results = comm.gather(all_results, root=0)
        all_weights = comm.gather(all_weights, root=0)
        
        if rank == 0:
            all_results = [item for sublist in all_results for item in sublist]
            all_weights = [item for sublist in all_weights for item in sublist]
        else:
            return {}
    
    if rank == 0:
        if len(all_results) == 0:
            raise RuntimeError("No portfolios were successfully optimized")
        
        logger.info(f"Successfully optimized {len(all_results)} portfolios")
        
        if asset_cache:
            cache_stats = asset_cache.get_stats()
            logger.info(f"Asset cache: {cache_stats['hits']} hits, {cache_stats['misses']} misses, hit rate: {cache_stats['hit_rate']:.1f}%")
        
        logger.info("Computing summary statistics...")
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame(all_results)
        
        # Check convergence if convergence sample was used
        convergence_config = selection_config.get('convergence_validation_sample', {})
        if convergence_config.get('enable', False) and len(metrics_df) > 0:
            metrics_tracked = convergence_config.get('metrics_tracked', [])
            convergence_results = check_convergence(metrics_df, convergence_config, metrics_tracked)
            
            if convergence_results:
                logger.info("Convergence validation results:")
                for metric, result in convergence_results.items():
                    status = "CONVERGED" if result['converged'] else "NOT CONVERGED"
                    logger.info(f"  {metric}: {status} (mean={result['final_mean']:.4f}, n={result['n_samples']})")
                    if result['reasons']:
                        for reason in result['reasons']:
                            logger.info(f"    - {reason}")
        
        # Save outputs
        output_base = Path(outputs['metrics_table']).parent
        output_base.mkdir(parents=True, exist_ok=True)
        
        metrics_df.to_parquet(outputs['metrics_table'])
        
        if 'optimal_portfolios' in outputs and len(all_weights) > 0:
            weights_df = pd.DataFrame(all_weights, index=metrics_df['portfolio_id'][:len(all_weights)])
            weights_df.to_parquet(outputs['optimal_portfolios'])
        
        # IO optimization: generate report posthoc if enabled
        io_opt = fast_engine.get('io_optimization', {})
        if not io_opt.get('disable_report_generation_during_run', True):
            logger.info("Generating report...")
            generate_report(metrics_df, outputs['summary_report'], rp_settings, report_sections)
        elif io_opt.get('generate_report_posthoc', True):
            logger.info("Report generation disabled during run (will generate posthoc)")
        
        logger.info("Optimization complete!")
        
        return {
            'metrics_df': metrics_df,
            'num_portfolios': num_portfolios,
            'num_portfolios_total': num_portfolios_total
        }
    else:
        return {}


if __name__ == '__main__':
    import argparse
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    parser = argparse.ArgumentParser(description='Run Risk Parity ERC Optimization (Static Mode)')
    parser.add_argument('--max-portfolios', type=int, default=None, help='Maximum number of portfolios')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    
    args = parser.parse_args()
    
    results = run_risk_parity_erc_optimization_static(
        config_path=args.config,
        max_portfolios=args.max_portfolios
    )
    
    if rank == 0 and results:
        logger.info(f"Optimized {results.get('num_portfolios', 0):,} portfolios")

