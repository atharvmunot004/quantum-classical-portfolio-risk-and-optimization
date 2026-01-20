"""
Main evaluation script for GARCH(1,1) Volatility Forecasting VaR and CVaR at Asset Level.

Orchestrates the entire asset-level VaR/CVaR evaluation pipeline:
- Asset-level GARCH fitting with rolling estimation windows
- Conditional volatility computation (cached and reused)
- Rolling one-step-ahead volatility forecasts
- Per-asset VaR/CVaR computation
- Per-asset backtesting and metrics
- Per-asset time-sliced evaluation
- Parquet-only output with schema JSON
- Markdown summary report
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

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*found in sys.modules.*')
warnings.filterwarnings('ignore', message='.*optimizer returned code.*')
warnings.filterwarnings('ignore', message='.*Inequality constraints incompatible.*')
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
warnings.filterwarnings('ignore', category=UserWarning, module='arch')

from .returns import (
    load_panel_prices,
    compute_daily_returns
)
from .garch_calculator import (
    GARCHParameterCache,
    compute_rolling_garch_asset_level,
    var_from_volatility,
    cvar_from_volatility,
    compute_horizons
)
from .backtesting import compute_accuracy_metrics
from .metrics import (
    compute_tail_metrics,
    compute_cvar_tail_metrics,
    compute_garch_metrics,
    compute_distribution_metrics,
    compute_runtime_metrics
)
from .time_sliced_metrics import compute_time_sliced_metrics
from .report_generator import generate_report


def _get_worker_count(config: Dict, default: Optional[int] = None) -> int:
    """
    Get number of workers based on configuration.
    
    Args:
        config: Configuration dictionary
        default: Default number of workers (None for auto)
        
    Returns:
        Number of workers to use
    """
    rolling_config = config.get('computation_strategy', {}).get('rolling_engine', {})
    max_workers = rolling_config.get('max_workers', 'auto')
    
    if max_workers == 'auto' or max_workers is None:
        cpu_count_val = cpu_count()
        return max(1, int(cpu_count_val * 0.8))
    
    if isinstance(max_workers, int):
        return max(1, max_workers)
    
    if default is not None:
        return max(1, default)
    
    return max(1, cpu_count())


def _process_single_asset(
    asset_data: Tuple[str, pd.Series],
    estimation_windows: List[int],
    confidence_levels: List[float],
    horizons: List[int],
    garch_config: Dict,
    evaluation_config: Dict,
    cache: Optional[GARCHParameterCache],
    runtime_instrumentation: bool
) -> Dict:
    """
    Process a single asset through the entire GARCH evaluation pipeline.
    
    Args:
        asset_data: Tuple of (asset_name, asset_returns_series)
        estimation_windows: List of estimation window sizes
        confidence_levels: List of confidence levels
        horizons: List of horizons
        garch_config: GARCH configuration dictionary
        cache: Optional parameter cache
        runtime_instrumentation: Whether to measure runtime
        
    Returns:
        Dictionary with all results for this asset
    """
    asset_name, asset_returns = asset_data
    
    if runtime_instrumentation:
        asset_start_time = time.time()
    
    results = {
        'asset': asset_name,
        'volatility_series': [],
        'risk_series': [],
        'metrics': [],
        'time_sliced_metrics': [],
        'runtime_ms': 0.0
    }
    
    try:
        # Get GARCH parameters
        p = garch_config.get('p', 1)
        q = garch_config.get('q', 1)
        dist = garch_config.get('distribution', 't')  # Default to Student-t distribution
        mean_model_config = garch_config.get('mean_model', {})
        use_mean_model = mean_model_config.get('enabled', False)
        mean = 'AR' if use_mean_model else 'Zero'
        vol = garch_config.get('model_type', 'GARCH')
        fallback_long_run_variance = garch_config.get('fallback_long_run_variance', True)
        scale_factor = garch_config.get('scale_factor', 100.0)
        
        # Process each estimation window
        for window in estimation_windows:
            # Compute rolling GARCH conditional volatility and forecasts
            garch_results = compute_rolling_garch_asset_level(
                asset_returns,
                window=window,
                p=p,
                q=q,
                dist=dist,
                mean=mean,
                vol=vol,
                fallback_long_run_variance=fallback_long_run_variance,
                cache=cache,
                asset_name=asset_name,
                scale_factor=scale_factor
            )
            
            if garch_results is None or len(garch_results) == 0:
                continue
            
            # Extract volatility series
            conditional_vol = garch_results.get('conditional_volatility')
            forecast_vol = garch_results.get('forecast_volatility')
            garch_params = garch_results.get('parameters', {})
            
            if conditional_vol is None or len(conditional_vol) == 0:
                continue
            
            # Store volatility series
            for date in conditional_vol.index:
                vol_entry = {
                    'asset': asset_name,
                    'date': date,
                    'estimation_window': window,
                    'sigma_t': conditional_vol.loc[date] if date in conditional_vol.index else np.nan,
                    'sigma_t2': conditional_vol.loc[date]**2 if date in conditional_vol.index else np.nan,
                    'one_step_forecast_sigma': forecast_vol.loc[date] if forecast_vol is not None and date in forecast_vol.index else np.nan,
                    'one_step_forecast_sigma2': forecast_vol.loc[date]**2 if forecast_vol is not None and date in forecast_vol.index else np.nan
                }
                results['volatility_series'].append(vol_entry)
            
            # Process each confidence level and horizon
            for confidence_level in confidence_levels:
                for horizon in horizons:
                    # Compute VaR and CVaR series
                    # Use forecast volatility if available, otherwise conditional volatility
                    vol_for_risk = forecast_vol if forecast_vol is not None and len(forecast_vol) > 0 else conditional_vol
                    
                    # Get degrees of freedom for t-distribution if applicable
                    df = garch_config.get('student_t_df', None)
                    if dist == 't' and df is None:
                        df = 5  # Default df >= 5
                    
                    # Compute VaR and CVaR (horizon scaling applied inside functions)
                    var_series = var_from_volatility(
                        vol_for_risk,
                        confidence_level=confidence_level,
                        horizon=horizon,  # Scaling applied inside function
                        dist=dist,
                        df=df,
                        use_gpu=False
                    )
                    
                    cvar_series = cvar_from_volatility(
                        vol_for_risk,
                        confidence_level=confidence_level,
                        horizon=horizon,  # Scaling applied inside function
                        dist=dist,
                        df=df,
                        use_gpu=False
                    )
                    
                    # Align returns with VaR/CVaR
                    common_dates = asset_returns.index.intersection(var_series.index)
                    if len(common_dates) == 0:
                        continue
                    
                    aligned_returns = asset_returns.loc[common_dates]
                    aligned_var = var_series.loc[common_dates]
                    aligned_cvar = cvar_series.loc[common_dates]
                    
                    # Store risk series
                    for date in common_dates:
                        risk_entry = {
                            'asset': asset_name,
                            'date': date,
                            'confidence_level': confidence_level,
                            'horizon': horizon,
                            'estimation_window': window,
                            'VaR': aligned_var.loc[date] if date in aligned_var.index else np.nan,
                            'CVaR': aligned_cvar.loc[date] if date in aligned_cvar.index else np.nan
                        }
                        results['risk_series'].append(risk_entry)
                    
                    # Compute backtesting metrics
                    accuracy_metrics = compute_accuracy_metrics(
                        aligned_returns,
                        aligned_var,
                        confidence_level=confidence_level
                    )
                    
                    tail_metrics = compute_tail_metrics(
                        aligned_returns,
                        aligned_var,
                        confidence_level=confidence_level
                    )
                    
                    cvar_tail_metrics = compute_cvar_tail_metrics(
                        aligned_returns,
                        aligned_cvar,
                        aligned_var,
                        confidence_level=confidence_level
                    )
                    
                    garch_specific_metrics = compute_garch_metrics(
                        garch_params,
                        conditional_vol
                    )
                    
                    distribution_metrics = compute_distribution_metrics(
                        aligned_returns
                    )
                    
                    # Combine all metrics
                    metrics_entry = {
                        'asset': asset_name,
                        'confidence_level': confidence_level,
                        'horizon': horizon,
                        'estimation_window': window,
                        **accuracy_metrics,
                        **tail_metrics,
                        **cvar_tail_metrics,
                        **garch_specific_metrics,
                        **distribution_metrics
                    }
                    results['metrics'].append(metrics_entry)
                    
                    # Compute time-sliced metrics
                    time_sliced_config = evaluation_config.get('time_sliced_metrics', {})
                    if time_sliced_config.get('enabled', True):
                        slice_by = time_sliced_config.get('slice_by', ['year', 'quarter', 'month'])
                        min_obs = time_sliced_config.get('minimum_observations_per_slice', 60)
                        
                        for slice_type in slice_by:
                            time_slices = compute_time_sliced_metrics(
                                aligned_returns,
                                aligned_var,
                                aligned_cvar,
                                confidence_level=confidence_level,
                                slice_by=slice_type,
                                min_observations=min_obs
                            )
                            
                            for time_slice in time_slices:
                                time_slice_entry = {
                                    'asset': asset_name,
                                    'slice_type': slice_type,
                                    'slice_value': time_slice['slice'],
                                    'confidence_level': confidence_level,
                                    'horizon': horizon,
                                    'estimation_window': window,
                                    **{k: v for k, v in time_slice.items() if k != 'slice'}
                                }
                                results['time_sliced_metrics'].append(time_slice_entry)
        
        if runtime_instrumentation:
            results['runtime_ms'] = (time.time() - asset_start_time) * 1000
        
    except Exception as e:
        warnings.warn(f"Error processing asset {asset_name}: {e}")
        results['error'] = str(e)
    
    return results


def evaluate_garch_var_cvar_asset_level(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    n_jobs: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Main function to evaluate VaR and CVaR using GARCH(1,1) volatility forecasting at asset level.
    
    Implements asset-level evaluation with rolling estimation windows as specified in llm.json.
    
    Args:
        config_path: Path to JSON configuration file
        config_dict: Configuration dictionary (if not loading from file)
        n_jobs: Number of parallel workers (None for auto-detection from config)
        
    Returns:
        Dictionary with DataFrames for volatility_series, risk_series, metrics_table, time_sliced_metrics_table
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
    
    # Get project root
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    
    # Get paths
    panel_price_path = project_root / config['inputs']['panel_price_path']
    
    # Adjust path if needed
    if not panel_price_path.exists() and 'preprocessed' in str(panel_price_path):
        panel_price_path = project_root / str(config['inputs']['panel_price_path']).replace('preprocessed', 'processed')
    
    print("=" * 80)
    print("GARCH(1,1) VOLATILITY FORECASTING FOR VAR/CVAR EVALUATION (ASSET LEVEL)")
    print("=" * 80)
    print(f"\nLoading data...")
    print(f"  Panel prices: {panel_price_path}")
    
    # Load data
    prices = load_panel_prices(panel_price_path)
    
    # Filter assets if specified
    asset_universe_config = config['inputs'].get('asset_universe', {})
    if asset_universe_config.get('mode') == 'from_columns':
        include = asset_universe_config.get('include')
        exclude = asset_universe_config.get('exclude')
        if include is not None:
            prices = prices[include]
        if exclude is not None:
            prices = prices.drop(columns=exclude, errors='ignore')
    
    data_period = f"{prices.index.min().strftime('%Y-%m-%d')} to {prices.index.max().strftime('%Y-%m-%d')}"
    
    print(f"\nLoaded:")
    print(f"  Prices: {len(prices)} dates, {len(prices.columns)} assets")
    print(f"  Data period: {data_period}")
    
    # Get data settings
    data_settings = config.get('data_settings', {})
    return_type = data_settings.get('return_type', 'log')
    tail_side = data_settings.get('tail_side', 'left')
    missing_data_policy = data_settings.get('missing_data_policy', {})
    min_required_observations = missing_data_policy.get('min_required_observations', 800)
    
    # Get GARCH settings
    garch_settings = config.get('garch_settings', {})
    confidence_levels = garch_settings.get('confidence_levels', [0.95, 0.99])
    estimation_windows = garch_settings.get('estimation_windows', [252, 500])
    
    # Get horizons
    horizons_config = garch_settings.get('horizons', {})
    if isinstance(horizons_config, dict):
        base_horizon = horizons_config.get('base_horizon', 1)
        scaled_horizons = horizons_config.get('scaled_horizons', [])
        scaling_rule = horizons_config.get('scaling_rule', 'sqrt_time')
        horizons = compute_horizons(base_horizon, scaled_horizons, scaling_rule)
    else:
        horizons = horizons_config if isinstance(horizons_config, list) else [1, 10]
    
    # Get computation strategy
    computation_strategy = config.get('computation_strategy', {})
    runtime_instrumentation = computation_strategy.get('runtime_instrumentation', {}).get('enabled', True)
    
    # Get evaluation config
    evaluation_config = config.get('evaluation', {})
    
    # Get cache settings
    cache_config = computation_strategy.get('cache', {})
    cache_enabled = cache_config.get('enabled', True)
    cache_root = project_root / Path(cache_config.get('parameter_store_path', 'cache/garch_asset_parameters.parquet')).parent
    
    # Initialize cache
    cache = None
    if cache_enabled:
        cache_path = project_root / cache_config.get('parameter_store_path', 'cache/garch_asset_parameters.parquet')
        cache = GARCHParameterCache(cache_path)
    
    # Get worker count
    n_workers = _get_worker_count(config, n_jobs)
    
    print(f"\nGARCH Settings:")
    print(f"  Model: GARCH({garch_settings.get('p', 1)},{garch_settings.get('q', 1)})")
    print(f"  Distribution: {garch_settings.get('distribution', 't')}")
    print(f"  Mean model: {'AR' if garch_settings.get('mean_model', {}).get('enabled', False) else 'Zero'}")
    print(f"  Confidence levels: {confidence_levels}")
    print(f"  Horizons: {horizons} days")
    print(f"  Estimation windows: {estimation_windows} days")
    print(f"\nParallelization Settings:")
    print(f"  Workers: {n_workers}")
    print(f"  Cache enabled: {cache_enabled}")
    
    # Compute daily returns
    print(f"\nComputing daily returns (method: {return_type})...")
    start_time_returns = time.time()
    daily_returns = compute_daily_returns(prices, method=return_type)
    
    # Handle missing data
    if missing_data_policy.get('dropna') == 'per_asset':
        daily_returns = daily_returns.dropna(axis=0, how='all')  # Drop dates with all NaN
    
    # Filter assets with insufficient observations
    valid_assets = []
    for asset in daily_returns.columns:
        asset_returns = daily_returns[asset].dropna()
        if len(asset_returns) >= min_required_observations:
            valid_assets.append(asset)
        else:
            print(f"  Skipping {asset}: only {len(asset_returns)} observations (required: {min_required_observations})")
    
    daily_returns = daily_returns[valid_assets]
    returns_compute_time = (time.time() - start_time_returns) * 1000
    
    print(f"  Daily returns: {len(daily_returns)} dates, {len(daily_returns.columns)} assets")
    print(f"  Returns computation time: {returns_compute_time:.2f} ms")
    
    # Prepare asset data
    asset_data_list = [
        (asset, daily_returns[asset].dropna())
        for asset in daily_returns.columns
    ]
    
    print(f"\n{'='*80}")
    print("PROCESSING ASSETS")
    print(f"{'='*80}")
    print(f"  Processing {len(asset_data_list)} assets")
    
    # Process assets (with optional parallelization)
    start_time_total = time.time()
    all_results = []
    
    if n_workers > 1 and len(asset_data_list) > 1:
        # Parallel processing
        worker_func = partial(
            _process_single_asset,
            estimation_windows=estimation_windows,
            confidence_levels=confidence_levels,
            horizons=horizons,
            garch_config=garch_settings,
            evaluation_config=evaluation_config,
            cache=cache,
            runtime_instrumentation=runtime_instrumentation
        )
        
        try:
            with Pool(processes=n_workers) as pool:
                all_results = pool.map(worker_func, asset_data_list)
        except Exception as e:
            warnings.warn(f"Parallel processing failed: {e}. Falling back to sequential.")
            all_results = []
            for asset_data in asset_data_list:
                result = worker_func(asset_data)
                all_results.append(result)
    else:
        # Sequential processing
        worker_func = partial(
            _process_single_asset,
            estimation_windows=estimation_windows,
            confidence_levels=confidence_levels,
            horizons=horizons,
            garch_config=garch_settings,
            evaluation_config=evaluation_config,
            cache=cache,
            runtime_instrumentation=runtime_instrumentation
        )
        for asset_data in asset_data_list:
            result = worker_func(asset_data)
            all_results.append(result)
    
    total_runtime = time.time() - start_time_total
    
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"  Total runtime: {total_runtime/60:.2f} minutes ({total_runtime:.2f} seconds)")
    if cache:
        print(f"  Cache hit ratio: {cache.get_hit_ratio():.2%}")
        cache.save_cache()
    
    # Aggregate results into DataFrames
    print(f"\nAggregating results...")
    
    volatility_series_list = []
    risk_series_list = []
    metrics_list = []
    time_sliced_metrics_list = []
    
    for result in all_results:
        volatility_series_list.extend(result.get('volatility_series', []))
        risk_series_list.extend(result.get('risk_series', []))
        metrics_list.extend(result.get('metrics', []))
        time_sliced_metrics_list.extend(result.get('time_sliced_metrics', []))
    
    volatility_series_df = pd.DataFrame(volatility_series_list) if volatility_series_list else pd.DataFrame()
    risk_series_df = pd.DataFrame(risk_series_list) if risk_series_list else pd.DataFrame()
    metrics_df = pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame()
    time_sliced_metrics_df = pd.DataFrame(time_sliced_metrics_list) if time_sliced_metrics_list else pd.DataFrame()
    
    print(f"  Volatility series: {len(volatility_series_df)} rows")
    print(f"  Risk series: {len(risk_series_df)} rows")
    print(f"  Metrics: {len(metrics_df)} rows")
    print(f"  Time-sliced metrics: {len(time_sliced_metrics_df)} rows")
    
    # Get output paths
    outputs = config.get('outputs', {})
    
    # Save volatility series
    if len(volatility_series_df) > 0:
        vol_path = project_root / outputs.get('volatility_series_store', {}).get('path', 'results/classical_risk/garch_asset_conditional_volatility_series.parquet')
        vol_path.parent.mkdir(parents=True, exist_ok=True)
        volatility_series_df.to_parquet(vol_path, index=False)
        print(f"  Saved volatility series: {vol_path}")
    
    # Save risk series
    if len(risk_series_df) > 0:
        risk_path = project_root / outputs.get('risk_series_store', {}).get('path', 'results/classical_risk/garch_asset_var_cvar_series.parquet')
        risk_path.parent.mkdir(parents=True, exist_ok=True)
        risk_series_df.to_parquet(risk_path, index=False)
        print(f"  Saved risk series: {risk_path}")
    
    # Save metrics table
    if len(metrics_df) > 0:
        metrics_path = project_root / outputs.get('metrics_table', {}).get('path', 'results/classical_risk/garch_asset_level_metrics.parquet')
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_parquet(metrics_path, index=False)
        print(f"  Saved metrics: {metrics_path}")
    
    # Save time-sliced metrics
    if len(time_sliced_metrics_df) > 0:
        time_sliced_path = project_root / outputs.get('time_sliced_metrics_table', {}).get('path', 'results/classical_risk/garch_asset_level_time_sliced_metrics.parquet')
        time_sliced_path.parent.mkdir(parents=True, exist_ok=True)
        time_sliced_metrics_df.to_parquet(time_sliced_path, index=False)
        print(f"  Saved time-sliced metrics: {time_sliced_path}")
    
    # Generate schema JSON
    print(f"\nGenerating schema JSON...")
    schema_path = project_root / outputs.get('garch_results_scheme.json', {}).get('path', 'results/classical_risk/garch_asset_level_results_scheme.json')
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    
    schema = {
        'description': 'GARCH(1,1) VaR/CVaR evaluation results at asset level',
        'files': {
            'volatility_series': {
                'path': str(outputs.get('volatility_series_store', {}).get('path', 'results/classical_risk/garch_asset_conditional_volatility_series.parquet')),
                'columns': outputs.get('volatility_series_store', {}).get('contents', []),
                'row_granularity': 'asset_x_date_x_estimation_window'
            },
            'risk_series': {
                'path': str(outputs.get('risk_series_store', {}).get('path', 'results/classical_risk/garch_asset_var_cvar_series.parquet')),
                'columns': outputs.get('risk_series_store', {}).get('contents', []),
                'row_granularity': 'asset_x_date_x_confidence_level_x_horizon_x_estimation_window'
            },
            'metrics_table': {
                'path': str(outputs.get('metrics_table', {}).get('path', 'results/classical_risk/garch_asset_level_metrics.parquet')),
                'row_granularity': outputs.get('metrics_table', {}).get('row_granularity', 'asset_x_confidence_level_x_horizon_x_window'),
                'columns': 'All metrics from accuracy, tail_behavior_var, tail_behavior_cvar, garch_specific, distribution, runtime'
            },
            'time_sliced_metrics_table': {
                'path': str(outputs.get('time_sliced_metrics_table', {}).get('path', 'results/classical_risk/garch_asset_level_time_sliced_metrics.parquet')),
                'row_granularity': outputs.get('time_sliced_metrics_table', {}).get('row_granularity', 'asset_x_slice_type_x_slice_value_x_confidence_level_x_horizon_x_window')
            }
        },
        'metadata': {
            'task': config.get('task', 'garch_1_1_var_cvar_asset_level_evaluation_optimized'),
            'generated_at': datetime.now().isoformat(),
            'garch_settings': garch_settings
        }
    }
    
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2, default=str)
    print(f"  Saved schema: {schema_path}")
    
    # Generate report
    print(f"\nGenerating summary report...")
    report_path = project_root / outputs.get('report', {}).get('path', 'results/classical_risk/garch_asset_level_report.md')
    report_sections = outputs.get('report', {}).get('include_sections', [])
    
    generate_report(
        metrics_df,
        time_sliced_metrics_df,
        report_path,
        garch_settings=garch_settings,
        report_sections=report_sections
    )
    print(f"  Saved report: {report_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return {
        'volatility_series': volatility_series_df,
        'risk_series': risk_series_df,
        'metrics': metrics_df,
        'time_sliced_metrics': time_sliced_metrics_df
    }


def main():
    """Main entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate GARCH(1,1) Volatility Forecasting VaR/CVaR at asset level'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration JSON file (default: llm.json in same directory)'
    )
    
    args = parser.parse_args()
    
    results = evaluate_garch_var_cvar_asset_level(
        config_path=args.config
    )
    
    print(f"\nResults summary:")
    print(f"  Volatility series: {len(results['volatility_series'])} rows")
    print(f"  Risk series: {len(results['risk_series'])} rows")
    print(f"  Metrics: {len(results['metrics'])} rows")
    print(f"  Time-sliced metrics: {len(results['time_sliced_metrics'])} rows")


if __name__ == "__main__":
    main()
