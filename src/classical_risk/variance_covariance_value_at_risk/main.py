"""
Main evaluation script for Historical VaR.

Orchestrates the entire VaR evaluation pipeline including:
- Data loading
- Asset-level returns computation (once)
- Portfolio return projection via linear combination
- Rolling Historical VaR calculation
- Backtesting
- Metrics computation
- Report generation

Implements optimized computation strategy:
- Asset returns computed once at asset level
- Portfolio returns via linear projection: R_p(t) = W^T R_assets(t)
- Batching of portfolios for memory efficiency
- Parallelization on estimation_window axis
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
import gc

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .returns import (
    load_panel_prices,
    load_portfolio_weights,
    construct_asset_return_matrix,
    compute_portfolio_returns_linear_projection
)
from .var_calculator import compute_rolling_historical_var, align_returns_and_var
from .backtesting import compute_accuracy_metrics
from .metrics import (
    compute_tail_metrics,
    compute_structure_metrics,
    compute_distribution_metrics,
    compute_runtime_metrics
)
from .report_generator import generate_report
from .time_sliced_metrics import compute_time_sliced_metrics


def _process_portfolio_batch(
    batch_portfolio_data: List[Tuple[int, Union[int, str], pd.Series]],
    asset_return_matrix: pd.DataFrame,
    confidence_levels: List[float],
    horizons: List[int],
    estimation_window: int,
    scaling_rule: str,
    quantile_method: str,
    interpolation: str,
    active_weight_threshold: float = 1e-6
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Process a batch of portfolios for a given estimation window.
    
    This function processes multiple portfolios in a batch using the
    optimized linear projection approach.
    
    Args:
        batch_portfolio_data: List of (portfolio_idx, portfolio_id, portfolio_weights) tuples
        asset_return_matrix: DataFrame of asset returns [num_days, num_assets]
        confidence_levels: List of confidence levels
        horizons: List of horizons
        estimation_window: Estimation window size
        scaling_rule: Horizon scaling rule
        quantile_method: Quantile method for historical VaR
        interpolation: Interpolation method
        active_weight_threshold: Threshold for active weights
        
    Returns:
        Tuple of (list of result dictionaries, dict of runtimes by portfolio)
    """
    results = []
    runtimes_by_portfolio = {}
    
    for portfolio_idx, portfolio_id, portfolio_weights in batch_portfolio_data:
        portfolio_start_time = time.time()
        
        try:
            # Compute portfolio returns using linear projection
            portfolio_returns = compute_portfolio_returns_linear_projection(
                asset_return_matrix,
                portfolio_weights,
                align_assets=True
            )
            
            # Compute covariance matrix for structure metrics (once per portfolio)
            try:
                common_assets = asset_return_matrix.columns.intersection(portfolio_weights.index)
                returns_aligned = asset_return_matrix[common_assets]
                covariance_matrix = returns_aligned.cov()
            except:
                covariance_matrix = None
            
            # Evaluate for each combination of confidence level and horizon
            for confidence_level in confidence_levels:
                for horizon in horizons:
                    try:
                        # Compute rolling Historical VaR
                        var_start_time = time.time()
                        rolling_var = compute_rolling_historical_var(
                            portfolio_returns,
                            window=estimation_window,
                            confidence_level=confidence_level,
                            horizon=horizon,
                            scaling_rule=scaling_rule,
                            quantile_method=quantile_method,
                            interpolation=interpolation
                        )
                        var_runtime = time.time() - var_start_time
                        
                        # Align returns and VaR
                        aligned_returns, aligned_var = align_returns_and_var(
                            portfolio_returns,
                            rolling_var
                        )
                        
                        if len(aligned_returns) == 0:
                            continue
                        
                        # Compute accuracy metrics
                        accuracy_metrics = compute_accuracy_metrics(
                            aligned_returns,
                            aligned_var,
                            confidence_level=confidence_level
                        )
                        
                        # Compute tail metrics (with confidence level)
                        tail_metrics = compute_tail_metrics(
                            aligned_returns,
                            aligned_var,
                            confidence_level=confidence_level
                        )
                        
                        # Compute structure metrics
                        structure_metrics = compute_structure_metrics(
                            portfolio_weights,
                            covariance_matrix
                        )
                        
                        # Apply active weight threshold
                        if 'num_active_assets' in structure_metrics:
                            active_mask = portfolio_weights.abs() >= active_weight_threshold
                            structure_metrics['num_active_assets'] = active_mask.sum()
                        
                        # Compute distribution metrics
                        distribution_metrics = compute_distribution_metrics(
                            aligned_returns
                        )
                        
                        # Combine all metrics
                        result = {
                            'portfolio_id': portfolio_id,
                            'confidence_level': confidence_level,
                            'horizon': horizon,
                            'estimation_window': estimation_window,
                            'var_runtime_ms': var_runtime * 1000,
                            **accuracy_metrics,
                            **tail_metrics,
                            **structure_metrics,
                            **distribution_metrics
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        # Continue to next configuration on error
                        warnings.warn(f"Error processing portfolio {portfolio_id}, cl={confidence_level}, h={horizon}: {e}")
                        continue
            
            portfolio_runtime = time.time() - portfolio_start_time
            runtimes_by_portfolio[portfolio_id] = portfolio_runtime
            
        except Exception as e:
            warnings.warn(f"Error processing portfolio {portfolio_id}: {e}")
            continue
    
    return results, runtimes_by_portfolio


def _process_estimation_window(
    estimation_window: int,
    asset_return_matrix: pd.DataFrame,
    portfolio_weights_df: pd.DataFrame,
    confidence_levels: List[float],
    horizons: List[int],
    scaling_rule: str,
    quantile_method: str,
    interpolation: str,
    batch_size: int,
    active_weight_threshold: float
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Process all portfolios for a given estimation window.
    
    This function handles batching of portfolios and processes them
    sequentially within each batch.
    
    Args:
        estimation_window: Estimation window size
        asset_return_matrix: DataFrame of asset returns
        portfolio_weights_df: DataFrame of portfolio weights
        confidence_levels: List of confidence levels
        horizons: List of horizons
        scaling_rule: Horizon scaling rule
        quantile_method: Quantile method
        interpolation: Interpolation method
        batch_size: Batch size for portfolios
        active_weight_threshold: Threshold for active weights
        
    Returns:
        Tuple of (list of result dictionaries, dict of runtimes)
    """
    all_results = []
    all_runtimes = {}
    
    # Process portfolios in batches
    num_portfolios = len(portfolio_weights_df)
    num_batches = (num_portfolios + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_portfolios)
        
        batch_portfolios = portfolio_weights_df.iloc[start_idx:end_idx]
        
        # Prepare batch data
        batch_portfolio_data = [
            (start_idx + i, portfolio_id, portfolio_weights)
            for i, (portfolio_id, portfolio_weights) in enumerate(batch_portfolios.iterrows())
        ]
        
        # Process batch
        batch_results, batch_runtimes = _process_portfolio_batch(
            batch_portfolio_data,
            asset_return_matrix,
            confidence_levels,
            horizons,
            estimation_window,
            scaling_rule,
            quantile_method,
            interpolation,
            active_weight_threshold
        )
        
        all_results.extend(batch_results)
        all_runtimes.update(batch_runtimes)
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            print(f"  Window {estimation_window}: Completed batch {batch_idx + 1}/{num_batches}")
    
    return all_results, all_runtimes


def evaluate_var(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    n_jobs: Optional[int] = None,
    max_portfolios: Optional[int] = None
) -> pd.DataFrame:
    """
    Main function to evaluate Historical VaR for multiple portfolios.
    
    Implements optimized computation strategy:
    - Asset returns computed once
    - Portfolio returns via linear projection
    - Batching of portfolios
    - Parallelization on estimation_window axis
    
    Args:
        config_path: Path to JSON configuration file
        config_dict: Configuration dictionary (if not loading from file)
        n_jobs: Number of parallel workers (default: number of CPU cores)
        max_portfolios: Maximum number of portfolios to process (None for all)
        
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
    
    # Get paths
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    
    panel_price_path = project_root / config['inputs']['panel_price_path']
    portfolio_weights_path = project_root / config['inputs']['portfolio_weights_path']
    
    # Adjust path if needed
    if not panel_price_path.exists() and 'preprocessed' in str(panel_price_path):
        panel_price_path = project_root / str(config['inputs']['panel_price_path']).replace('preprocessed', 'processed')
    
    print("=" * 80)
    print("HISTORICAL VALUE-AT-RISK EVALUATION (OPTIMIZED)")
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
    
    # Compute asset return matrix once (core optimization)
    var_settings = config.get('var_settings', {})
    return_type = var_settings.get('return_type', 'log')
    
    print(f"\nComputing asset-level returns (once)...")
    print(f"  Method: {return_type}")
    asset_return_matrix = construct_asset_return_matrix(prices, method=return_type)
    print(f"  Asset returns: {len(asset_return_matrix)} dates, {len(asset_return_matrix.columns)} assets")
    
    # Optionally save asset return matrix
    outputs = config.get('outputs', {})
    if 'asset_return_store' in outputs:
        asset_store_path = project_root / outputs['asset_return_store']
        asset_store_path.parent.mkdir(parents=True, exist_ok=True)
        asset_return_matrix.to_parquet(asset_store_path)
        print(f"  Saved asset returns to: {asset_store_path}")
    
    # Get VaR settings
    confidence_levels = var_settings.get('confidence_levels', [0.95, 0.99])
    horizons_config = var_settings.get('horizons', {})
    base_horizon = horizons_config.get('base_horizon', 1)
    scaled_horizons = horizons_config.get('scaled_horizons', [])
    scaling_rule = horizons_config.get('scaling_rule', 'sqrt_time')
    estimation_windows = var_settings.get('estimation_windows', [252])
    
    # Construct full horizons list
    horizons = [base_horizon] + scaled_horizons
    
    # Get computation strategy settings
    comp_strategy = config.get('computation_strategy', {})
    batch_size = comp_strategy.get('batching', {}).get('portfolios_batch_size', 5000)
    quantile_method = config.get('modules', {}).get('historical_var', {}).get('quantile_method', 'empirical')
    interpolation = config.get('modules', {}).get('historical_var', {}).get('interpolation', 'linear')
    active_weight_threshold = config.get('modules', {}).get('portfolio_structure_metrics', {}).get('active_weight_threshold', 1e-6)
    
    # Limit portfolios if specified
    num_portfolios_total = len(portfolio_weights_df)
    if max_portfolios is not None and max_portfolios > 0:
        num_portfolios = min(num_portfolios_total, max_portfolios)
        portfolio_weights_df = portfolio_weights_df.iloc[:num_portfolios]
        print(f"\n  Limiting to first {num_portfolios:,} portfolios (out of {num_portfolios_total:,} total)")
    else:
        num_portfolios = num_portfolios_total
        print(f"\n  Processing all {num_portfolios:,} portfolios")
    
    print(f"\nVaR Settings:")
    print(f"  Confidence levels: {confidence_levels}")
    print(f"  Horizons: {horizons} days (scaling: {scaling_rule})")
    print(f"  Estimation windows: {estimation_windows} days")
    print(f"  Quantile method: {quantile_method}, Interpolation: {interpolation}")
    print(f"  Batch size: {batch_size} portfolios")
    
    # Calculate total combinations
    total_combinations = num_portfolios * len(confidence_levels) * len(horizons) * len(estimation_windows)
    print(f"  Total portfolio-configuration combinations: {total_combinations:,}")
    
    # Determine number of workers for estimation window parallelization
    if n_jobs is None:
        n_jobs = cpu_count()
    elif n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        n_jobs = 1
    
    # Memory optimization: reduce workers if memory is constrained
    if PSUTIL_AVAILABLE:
        try:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            # Estimate memory per worker (rough estimate: 2-4GB per worker)
            estimated_memory_per_worker_gb = 3.0
            max_workers_by_memory = int(available_memory_gb / estimated_memory_per_worker_gb)
            if max_workers_by_memory < n_jobs:
                print(f"  Memory optimization: Reducing workers from {n_jobs} to {max_workers_by_memory} "
                      f"(available: {available_memory_gb:.1f}GB, ~{estimated_memory_per_worker_gb:.1f}GB per worker)")
                n_jobs = max(1, max_workers_by_memory)
        except Exception:
            pass  # If psutil fails, use original n_jobs
    
    # Memory optimization: batch processing
    print(f"  Batch processing: {batch_size:,} portfolios per batch (memory optimization)")
    
    # Process estimation windows
    all_results = []
    all_runtimes = {}
    
    start_time_total = time.time()
    
    if len(estimation_windows) > 1 and n_jobs > 1:
        # Parallelize across estimation windows
        print(f"\nProcessing {len(estimation_windows)} estimation windows in parallel ({n_jobs} workers)...")
        
        worker_func = partial(
            _process_estimation_window,
            asset_return_matrix=asset_return_matrix,
            portfolio_weights_df=portfolio_weights_df,
            confidence_levels=confidence_levels,
            horizons=horizons,
            scaling_rule=scaling_rule,
            quantile_method=quantile_method,
            interpolation=interpolation,
            batch_size=batch_size,
            active_weight_threshold=active_weight_threshold
        )
        
        with Pool(processes=min(n_jobs, len(estimation_windows))) as pool:
            window_results = pool.map(worker_func, estimation_windows)
        
        for window_results_list, window_runtimes in window_results:
            all_results.extend(window_results_list)
            all_runtimes.update(window_runtimes)
            # Memory optimization: force garbage collection after each window
            gc.collect()
    else:
        # Process sequentially (or single window)
        print(f"\nProcessing {len(estimation_windows)} estimation window(s)...")
        for estimation_window in estimation_windows:
            print(f"\nProcessing estimation window: {estimation_window} days")
            window_results, window_runtimes = _process_estimation_window(
                estimation_window,
                asset_return_matrix,
                portfolio_weights_df,
                confidence_levels,
                horizons,
                scaling_rule,
                quantile_method,
                interpolation,
                batch_size,
                active_weight_threshold
            )
            all_results.extend(window_results)
            all_runtimes.update(window_runtimes)
            # Memory optimization: force garbage collection after each window
            gc.collect()
    
    total_runtime = time.time() - start_time_total
    
    # Create results DataFrame
    if len(all_results) == 0:
        raise ValueError("No results computed. Check data and configuration.")
    
    results_df = pd.DataFrame(all_results)
    
    # Add runtime metrics
    runtime_values = list(all_runtimes.values())
    runtime_metrics = compute_runtime_metrics(runtime_values)
    for key, value in runtime_metrics.items():
        results_df[key] = value
    
    print(f"\nCompleted evaluation of {len(results_df)} portfolio-configuration combinations")
    print(f"  Total runtime: {total_runtime/60:.2f} minutes ({total_runtime:.2f} seconds)")
    print(f"  Average runtime per portfolio: {np.mean(runtime_values)*1000:.2f} ms")
    if len(runtime_values) > 0:
        print(f"  95th percentile runtime: {np.percentile(runtime_values, 95)*1000:.2f} ms")
    
    # Save results
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
        
        # Convert DataFrame to dict format
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
                'task': config.get('task', 'classical_var_evaluation_optimized'),
                'data_period': data_period,
                'portfolios_evaluated': num_portfolios,
                'confidence_levels': confidence_levels,
                'horizons': horizons,
                'estimation_windows': estimation_windows,
                'generated_at': datetime.now().isoformat()
            },
            'results': clean_nan(results_df.to_dict(orient='records'))
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
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
            var_settings=var_settings,
            report_sections=None
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
        description='Evaluate Historical VaR for portfolios'
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
        help='Number of parallel workers (default: number of CPU cores, use -1 for all cores, 1 for sequential)'
    )
    parser.add_argument(
        '--max-portfolios',
        type=int,
        default=None,
        help='Maximum number of portfolios to process (default: None for all)'
    )
    
    args = parser.parse_args()
    
    # Handle -1 for all cores
    n_jobs = args.n_jobs
    if n_jobs == -1:
        n_jobs = None
    
    results_df = evaluate_var(
        config_path=args.config,
        n_jobs=n_jobs,
        max_portfolios=args.max_portfolios
    )
    
    print(f"\nResults summary:")
    print(f"  Total rows: {len(results_df)}")
    print(f"  Columns: {len(results_df.columns)}")
    print(f"\nFirst few rows:")
    print(results_df.head())


if __name__ == "__main__":
    main()
