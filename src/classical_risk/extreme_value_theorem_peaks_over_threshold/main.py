"""
Main evaluation script for EVT-POT VaR and CVaR at asset level.

Orchestrates the entire VaR/CVaR evaluation pipeline with asset-level EVT fitting:
- Data loading and preprocessing
- Asset-level EVT parameter estimation with rolling windows (with caching)
- Rolling VaR/CVaR time series computation
- Backtesting per asset
- Time-sliced metrics computation
- Report generation
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import warnings
from threading import Lock

# Suppress RuntimeWarning about module import in multiprocessing workers
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*found in sys.modules.*')

from .returns import (
    load_panel_prices,
    compute_daily_returns,
    compute_losses_from_returns
)
from .evt_calculator import (
    compute_rolling_asset_evt_parameters,
    EVTParameterCache,
    compute_var_from_evt,
    compute_cvar_from_evt
)
from .backtesting import compute_accuracy_metrics, detect_cvar_violations
from .metrics import (
    compute_tail_metrics,
    compute_cvar_tail_metrics,
    compute_evt_tail_metrics,
    compute_distribution_metrics,
    compute_runtime_metrics
)
from .report_generator import generate_report
from .time_sliced_metrics import compute_time_sliced_metrics


def _process_single_asset_rolling(
    asset_data: Tuple[str, pd.Series],
    estimation_windows: List[int],
    threshold_quantiles: List[float],
    confidence_levels: List[float],
    horizons: List[int],
    scaling_rule: str,
    min_exceedances: int,
    xi_lower: float,
    xi_upper: float,
    gpd_fitting_method: str,
    fallback_quantiles: Optional[List[float]],
    cache: Optional[EVTParameterCache] = None,
    cache_lock: Optional[Lock] = None,
    safety_checks: Optional[Dict] = None
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Process a single asset with rolling windows to produce VaR/CVaR time series.
    
    Args:
        asset_data: Tuple of (asset_name, asset_losses_series)
        estimation_windows: List of estimation window sizes
        threshold_quantiles: List of threshold quantiles
        confidence_levels: List of confidence levels
        horizons: List of horizons
        scaling_rule: Scaling rule for horizon
        min_exceedances: Minimum number of exceedances
        xi_lower: Lower bound for shape parameter
        xi_upper: Upper bound for shape parameter
        gpd_fitting_method: GPD fitting method
        return_type: Return type ('log' or 'simple')
        tail_side: Tail side ('left' or 'right')
        cache: Optional parameter cache
        cache_lock: Optional lock for thread-safe cache access
        safety_checks: Optional safety check configuration
        
    Returns:
        Tuple of (risk_series_list, metrics_list, runtime_dict)
    """
    asset_name, asset_losses = asset_data
    
    if len(asset_losses) < max(estimation_windows):
        return [], [], {}
    
    risk_series_list = []
    metrics_list = []
    runtime_dict = {
        'total_runtime_ms': 0.0,
        'evt_fit_time_ms': 0.0,
        'threshold_selection_time_ms': 0.0,
        'var_compute_time_ms': 0.0,
        'cvar_compute_time_ms': 0.0,
        'backtesting_time_ms': 0.0
    }
    
    start_total = time.time()
    
    # Process each combination of window, quantile, confidence, horizon
    for window in estimation_windows:
        if len(asset_losses) < window:
            continue
            
        for quantile in threshold_quantiles:
            # Compute rolling EVT parameters and VaR/CVaR series
            start_evt = time.time()
            
            try:
                var_series, cvar_series, evt_params_series = compute_rolling_asset_evt_parameters(
                    asset_losses,
                    window,
                    quantile,
                    confidence_levels,
                    horizons,
                    scaling_rule,
                    min_exceedances,
                    xi_lower,
                    xi_upper,
                    gpd_fitting_method,
                    fallback_quantiles=fallback_quantiles,
                    cache=cache,
                    cache_lock=cache_lock
                )
                
                runtime_dict['evt_fit_time_ms'] += (time.time() - start_evt) * 1000
                
                if var_series is None:
                    warnings.warn(f"No VaR series returned (None) for asset {asset_name}, window {window}, quantile {quantile}")
                    continue
                
                if len(var_series) == 0:
                    warnings.warn(f"Empty VaR series for asset {asset_name}, window {window}, quantile {quantile}")
                    continue
                
                # Check if all values are NaN (all filtered by guardrails)
                if var_series.isna().all().all():
                    warnings.warn(f"All VaR values are NaN for asset {asset_name}, window {window}, quantile {quantile} - all values filtered by guardrails")
                    continue
                
                # Create risk series records
                for conf_level in confidence_levels:
                    for horizon in horizons:
                        var_key = f"var_{conf_level}_{horizon}"
                        cvar_key = f"cvar_{conf_level}_{horizon}"
                        
                        if var_key in var_series.columns and cvar_key in cvar_series.columns:
                            var_col = var_series[var_key]
                            cvar_col = cvar_series[cvar_key]
                            
                            # Create risk series DataFrame
                            for date in var_col.index:
                                if pd.isna(var_col[date]) or pd.isna(cvar_col[date]):
                                    continue
                                
                                # Safety check: VaR must be positive
                                if safety_checks and safety_checks.get('enabled', False):
                                    var_positive_check = safety_checks.get('checks', [])
                                    for check in var_positive_check:
                                        if check.get('name') == 'var_positive':
                                            if var_col[date] <= 0:
                                                continue  # Skip this timestamp
                                
                                risk_series_list.append({
                                    'asset': asset_name,
                                    'date': date,
                                    'confidence_level': conf_level,
                                    'horizon': horizon,
                                    'estimation_window': window,
                                    'threshold_quantile': quantile,
                                    'VaR': float(var_col[date]),
                                    'CVaR': float(cvar_col[date])
                                })
                            
                            # Compute backtesting metrics for this configuration
                            start_backtest = time.time()
                            
                            # Align losses with VaR/CVaR
                            aligned_losses = asset_losses.loc[var_col.index]
                            aligned_var = var_col
                            aligned_cvar = cvar_col
                            
                            # Remove NaN values
                            valid_mask = ~(pd.isna(aligned_losses) | pd.isna(aligned_var) | pd.isna(aligned_cvar))
                            aligned_losses = aligned_losses[valid_mask]
                            aligned_var = aligned_var[valid_mask]
                            aligned_cvar = aligned_cvar[valid_mask]
                            
                            if len(aligned_losses) == 0:
                                continue
                            
                            # Compute accuracy metrics using losses
                            accuracy_metrics = compute_accuracy_metrics(
                                aligned_losses,
                                aligned_var,
                                confidence_level=conf_level
                            )
                            
                            # Compute tail metrics using losses
                            tail_metrics = compute_tail_metrics(
                                aligned_losses,
                                aligned_var,
                                confidence_level=conf_level
                            )
                            
                            # Compute CVaR tail metrics using losses
                            cvar_tail_metrics = compute_cvar_tail_metrics(
                                aligned_losses,
                                aligned_cvar,
                                aligned_var,
                                confidence_level=conf_level
                            )
                            
                            # Compute distribution metrics (on returns, not losses)
                            # Convert losses back to returns for distribution metrics
                            aligned_returns = -aligned_losses
                            distribution_metrics = compute_distribution_metrics(aligned_returns)
                            
                            # Compute EVT tail metrics using losses
                            if len(evt_params_series) > 0 and evt_params_series[-1].get('success', False):
                                latest_params = evt_params_series[-1]
                                evt_tail_metrics = compute_evt_tail_metrics(
                                    aligned_losses,
                                    aligned_var,
                                    latest_params.get('threshold', np.nan),
                                    latest_params.get('xi', np.nan),
                                    latest_params.get('beta', np.nan),
                                    confidence_level=conf_level
                                )
                                # Add additional EVT parameters
                                evt_tail_metrics['threshold'] = latest_params.get('threshold', np.nan)
                                evt_tail_metrics['num_exceedances'] = latest_params.get('num_exceedances', 0)
                                
                                # Compute shape-scale stability from series
                                if len(evt_params_series) > 1:
                                    xi_values = [p.get('xi', np.nan) for p in evt_params_series if p.get('success', False) and not pd.isna(p.get('xi', np.nan))]
                                    if len(xi_values) > 1:
                                        evt_tail_metrics['shape_scale_stability'] = float(np.std(xi_values))
                            else:
                                evt_tail_metrics = {
                                    'expected_shortfall_exceedance': np.nan,
                                    'tail_index_xi': np.nan,
                                    'scale_beta': np.nan,
                                    'shape_scale_stability': np.nan,
                                    'threshold': np.nan,
                                    'num_exceedances': 0
                                }
                            
                            # Remove the old evt_tail_metrics assignment that was here
                            
                            runtime_dict['backtesting_time_ms'] += (time.time() - start_backtest) * 1000
                            
                            # Combine all metrics
                            metrics_list.append({
                                'asset': asset_name,
                                'confidence_level': conf_level,
                                'horizon': horizon,
                                'estimation_window': window,
                                'threshold_quantile': quantile,
                                **accuracy_metrics,
                                **tail_metrics,
                                **cvar_tail_metrics,
                                **evt_tail_metrics,
                                **distribution_metrics
                            })
                            
            except Exception as e:
                warnings.warn(f"Error processing asset {asset_name} with window {window}, quantile {quantile}: {e}")
                continue
    
    runtime_dict['total_runtime_ms'] = (time.time() - start_total) * 1000
    
    return risk_series_list, metrics_list, runtime_dict


def evaluate_evt_pot_var_cvar(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    n_jobs: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main function to evaluate VaR and CVaR using EVT-POT methodology at asset level.
    
    This function implements the optimized workflow:
    1. Load and preprocess asset price data
    2. Compute daily returns
    3. For each asset, compute rolling EVT parameters and VaR/CVaR time series
    4. Perform backtesting and compute metrics
    5. Generate time-sliced metrics
    6. Save results
    
    Args:
        config_path: Path to JSON configuration file
        config_dict: Configuration dictionary (if not loading from file)
        n_jobs: Number of parallel workers (default: auto)
        
    Returns:
        Tuple of (risk_series_df, metrics_df, time_sliced_metrics_df)
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
    
    # Adjust path if it says "preprocessed" but file is in "processed"
    if not panel_price_path.exists() and 'preprocessed' in str(panel_price_path):
        panel_price_path = project_root / str(config['inputs']['panel_price_path']).replace('preprocessed', 'processed')
    
    print("=" * 80)
    print("EVT-POT (EXTREME VALUE THEORY - PEAKS OVER THRESHOLD) VAR/CVAR EVALUATION")
    print("Asset-Level Evaluation with Rolling Windows")
    print("=" * 80)
    print(f"\nLoading data...")
    print(f"  Panel prices: {panel_price_path}")
    
    # Load data
    prices = load_panel_prices(panel_price_path)
    
    # Get data period
    data_period = f"{prices.index.min().strftime('%Y-%m-%d')} to {prices.index.max().strftime('%Y-%m-%d')}"
    
    print(f"\nLoaded:")
    print(f"  Prices: {len(prices)} dates, {len(prices.columns)} assets")
    print(f"  Data period: {data_period}")
    
    # Handle asset universe selection
    asset_universe_config = config['inputs'].get('asset_universe', {})
    if asset_universe_config.get('mode') == 'from_columns':
        include_assets = asset_universe_config.get('include')
        exclude_assets = asset_universe_config.get('exclude')
        
        if include_assets:
            prices = prices[include_assets]
        if exclude_assets:
            prices = prices.drop(columns=exclude_assets)
    
    # Data settings
    data_settings = config.get('data_settings', {})
    return_type = data_settings.get('return_type', 'log')
    tail_side = data_settings.get('tail_side', 'left')
    min_required_obs = data_settings.get('missing_data_policy', {}).get('min_required_observations', 800)
    
    # Calendar settings
    calendar_settings = data_settings.get('calendar', {})
    if calendar_settings.get('sort_index', True):
        prices = prices.sort_index()
    if calendar_settings.get('drop_duplicate_dates', True):
        prices = prices[~prices.index.duplicated(keep='first')]
    
    # Compute daily returns
    print(f"\nComputing daily returns (method: {return_type})...")
    daily_returns = compute_daily_returns(prices, method=return_type)
    
    # Filter assets with insufficient data
    valid_assets = []
    for asset in daily_returns.columns:
        asset_returns = daily_returns[asset].dropna()
        if len(asset_returns) >= min_required_obs:
            valid_assets.append(asset)
    
    daily_returns = daily_returns[valid_assets]
    print(f"  Daily returns: {len(daily_returns)} dates, {len(daily_returns.columns)} assets (after filtering)")
    
    # Compute explicit loss series: loss_t = -returns_t
    print(f"\nComputing losses from returns...")
    daily_losses = compute_losses_from_returns(daily_returns)
    print(f"  Daily losses: {len(daily_losses)} dates, {len(daily_losses.columns)} assets")
    
    # Get EVT settings
    evt_settings = config.get('evt_settings', {})
    confidence_levels = evt_settings.get('confidence_levels', [0.95, 0.99])
    horizons_config = evt_settings.get('horizons', {})
    base_horizon = horizons_config.get('base_horizon', 1)
    scaled_horizons = horizons_config.get('scaled_horizons', [10])
    horizons = [base_horizon] + scaled_horizons
    scaling_rule = horizons_config.get('scaling_rule', 'sqrt_time')
    estimation_windows = evt_settings.get('estimation_windows', [252, 500])
    threshold_settings = evt_settings.get('threshold_selection', {})
    threshold_quantiles = threshold_settings.get('quantiles', [0.95])
    min_exceedances = threshold_settings.get('min_exceedances', 50)
    fallback_quantiles = threshold_settings.get('fallback_quantiles_if_insufficient', [0.9, 0.85, 0.8, 0.75, 0.7])
    shape_constraints = evt_settings.get('shape_constraints', {})
    xi_lower = shape_constraints.get('xi_lower_bound', -0.5)
    xi_upper = shape_constraints.get('xi_upper_bound', 0.5)
    gpd_fitting_method = evt_settings.get('gpd_fitting_method', 'pwm')
    
    # Rolling settings
    rolling_settings = evt_settings.get('rolling', {})
    rolling_enabled = rolling_settings.get('enabled', True)
    step_size = rolling_settings.get('step_size', 1)
    
    # Get computation strategy
    computation_strategy = config.get('computation_strategy', {})
    enable_cache = computation_strategy.get('cache', {}).get('enabled', True)
    parallelization_config = computation_strategy.get('rolling_engine', {})
    max_workers_config = parallelization_config.get('max_workers', 'auto')
    chunk_assets = parallelization_config.get('chunk_assets', 1)
    safety_checks = computation_strategy.get('safety_checks', {})
    
    # Determine number of workers
    if n_jobs is None:
        if max_workers_config == 'auto':
            n_jobs = max(1, cpu_count() - 1)  # Leave one core free
        else:
            n_jobs = int(max_workers_config)
    
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
    print(f"  Rolling windows: {'Enabled' if rolling_enabled else 'Disabled'}")
    print(f"  Parameter caching: {'Enabled' if enable_cache else 'Disabled'}")
    print(f"  Parallel workers: {n_jobs}")
    
    # Initialize parameter cache
    cache = None
    cache_lock = None
    manager = None
    if enable_cache:
        outputs = config.get('outputs', {})
        cache_path = outputs.get('parameter_store', {}).get('path')
        if cache_path:
            cache_path = project_root / cache_path
            cache = EVTParameterCache(cache_path)
            # Create a manager and lock for process-safe cache access
            # Note: In multiprocessing, each process has its own cache copy
            # The lock helps with thread-safety if threading is used within processes
            manager = Manager()
            cache_lock = manager.Lock()
            print(f"  Cache path: {cache_path}")
    
    # Prepare asset data (using losses, not returns)
    asset_data_list = [(asset, daily_losses[asset].dropna()) for asset in daily_losses.columns]
    
    print(f"\n{'='*80}")
    print("Processing Assets with Rolling EVT Estimation")
    print(f"{'='*80}")
    print(f"  Total assets: {len(asset_data_list)}")
    print(f"  Parallel workers: {n_jobs}")
    
    # Process assets
    all_risk_series = []
    all_metrics = []
    all_runtimes = []
    
    start_time_total = time.time()
    
    if n_jobs > 1 and len(asset_data_list) > 1:
        # Parallel processing
        print(f"\n  Processing assets in parallel...")
        
        worker_func = partial(
            _process_single_asset_rolling,
            estimation_windows=estimation_windows,
            threshold_quantiles=threshold_quantiles,
            confidence_levels=confidence_levels,
            horizons=horizons,
            scaling_rule=scaling_rule,
            min_exceedances=min_exceedances,
            xi_lower=xi_lower,
            xi_upper=xi_upper,
            gpd_fitting_method=gpd_fitting_method,
            fallback_quantiles=fallback_quantiles,
            cache=cache,
            cache_lock=cache_lock,
            safety_checks=safety_checks
        )
        
        try:
            with Pool(processes=n_jobs) as pool:
                results_iter = pool.imap(worker_func, asset_data_list, chunksize=chunk_assets)
                
                for idx, (risk_series, metrics, runtime) in enumerate(results_iter):
                    all_risk_series.extend(risk_series)
                    all_metrics.extend(metrics)
                    all_runtimes.append(runtime)
                    
                    if (idx + 1) % 10 == 0 or (idx + 1) in [1, 5, 20, 50, 100]:
                        print(f"  Processed {idx + 1}/{len(asset_data_list)} assets...", flush=True)
        except Exception as e:
            warnings.warn(f"Parallel processing failed: {e}. Falling back to sequential processing.")
            # Fall back to sequential
            for idx, asset_data in enumerate(asset_data_list):
                risk_series, metrics, runtime = worker_func(asset_data)
                all_risk_series.extend(risk_series)
                all_metrics.extend(metrics)
                all_runtimes.append(runtime)
                
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(asset_data_list)} assets...", flush=True)
    else:
        # Sequential processing
        print(f"\n  Processing assets sequentially...")
        for idx, asset_data in enumerate(asset_data_list):
            risk_series, metrics, runtime = _process_single_asset_rolling(
                asset_data,
                estimation_windows,
                threshold_quantiles,
                confidence_levels,
                horizons,
                scaling_rule,
                min_exceedances,
                xi_lower,
                xi_upper,
                gpd_fitting_method,
                fallback_quantiles,
                cache=cache,
                cache_lock=cache_lock,
                safety_checks=safety_checks
            )
            all_risk_series.extend(risk_series)
            all_metrics.extend(metrics)
            all_runtimes.append(runtime)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(asset_data_list)} assets...", flush=True)
    
    total_runtime = time.time() - start_time_total
    
    # Save cache
    if cache:
        cache.save_cache()
        cache_hit_ratio = cache.get_hit_ratio()
        print(f"\n  Cache hit ratio: {cache_hit_ratio:.2%}")
    
    # Create DataFrames
    if len(all_risk_series) == 0:
        # Provide more helpful error message
        print(f"\nWARNING: No risk series computed!")
        print(f"  Total assets processed: {len(asset_data_list)}")
        print(f"  Total metrics records: {len(all_metrics)}")
        print(f"  This may indicate:")
        print(f"    - VaR guardrails are too strict (VaR <= threshold)")
        print(f"    - CVaR domain checks failing (xi >= 1)")
        print(f"    - Insufficient exceedances for EVT fitting")
        print(f"    - All computed VaR/CVaR values are invalid")
        raise ValueError("No risk series computed. Check data and configuration. All VaR/CVaR values may have been filtered out by guardrails.")
    
    risk_series_df = pd.DataFrame(all_risk_series)
    metrics_df = pd.DataFrame(all_metrics)
    
    # Runtime metrics should NOT be per-row (IEEE reviewers flag duplicated runtime columns)
    # Store runtime metrics separately, not broadcast to every configuration row
    runtime_metrics_dict = {}
    if len(all_runtimes) > 0:
        runtime_metrics_dict = compute_runtime_metrics([r.get('total_runtime_ms', 0) / 1000 for r in all_runtimes])
        if cache:
            runtime_metrics_dict['cache_hit_ratio'] = cache.get_hit_ratio()
    
    print(f"\nCompleted evaluation:")
    print(f"  Total runtime: {total_runtime/60:.2f} minutes ({total_runtime:.2f} seconds)")
    print(f"  Risk series records: {len(risk_series_df)}")
    print(f"  Metrics records: {len(metrics_df)}")
    
    # Compute time-sliced metrics
    print(f"\n{'='*80}")
    print("Computing Time-Sliced Metrics")
    print(f"{'='*80}")
    
    time_sliced_config = config.get('evaluation', {}).get('time_sliced_metrics', {})
    if time_sliced_config.get('enabled', True):
        slice_by_list = time_sliced_config.get('slice_by', ['year', 'quarter', 'month'])
        min_obs_per_slice = time_sliced_config.get('minimum_observations_per_slice', 60)
        
        all_time_sliced = []
        
        # Group risk series by asset and configuration
        for asset in risk_series_df['asset'].unique():
            asset_risk = risk_series_df[risk_series_df['asset'] == asset]
            
            for conf_level in confidence_levels:
                for horizon in horizons:
                    for window in estimation_windows:
                        for quantile in threshold_quantiles:
                            config_risk = asset_risk[
                                (asset_risk['confidence_level'] == conf_level) &
                                (asset_risk['horizon'] == horizon) &
                                (asset_risk['estimation_window'] == window) &
                                (asset_risk['threshold_quantile'] == quantile)
                            ]
                            
                            if len(config_risk) == 0:
                                continue
                            
                            # Get asset losses
                            asset_losses = daily_losses[asset].dropna()
                            
                            # Create VaR/CVaR series
                            var_series = pd.Series(
                                config_risk.set_index('date')['VaR'],
                                index=pd.to_datetime(config_risk['date'])
                            )
                            cvar_series = pd.Series(
                                config_risk.set_index('date')['CVaR'],
                                index=pd.to_datetime(config_risk['date'])
                            )
                            
                            # Align losses
                            common_dates = asset_losses.index.intersection(var_series.index)
                            if len(common_dates) < min_obs_per_slice:
                                continue
                            
                            aligned_losses = asset_losses.loc[common_dates]
                            aligned_var = var_series.loc[common_dates]
                            aligned_cvar = cvar_series.loc[common_dates]
                            
                            # Compute time-sliced metrics using losses
                            for slice_by in slice_by_list:
                                time_slices = compute_time_sliced_metrics(
                                    aligned_losses,
                                    aligned_var,
                                    cvar_series=aligned_cvar,
                                    confidence_level=conf_level,
                                    slice_by=slice_by
                                )
                                
                                for ts in time_slices:
                                    ts['asset'] = asset
                                    ts['confidence_level'] = conf_level
                                    ts['horizon'] = horizon
                                    ts['estimation_window'] = window
                                    ts['threshold_quantile'] = quantile
                                    ts['slice_type'] = slice_by
                                    all_time_sliced.append(ts)
        
        time_sliced_metrics_df = pd.DataFrame(all_time_sliced)
        print(f"  Time-sliced metrics records: {len(time_sliced_metrics_df)}")
    else:
        time_sliced_metrics_df = pd.DataFrame()
    
    # Save results
    outputs = config.get('outputs', {})
    
    # Save risk series
    if 'risk_series_store' in outputs:
        risk_path = project_root / outputs['risk_series_store']['path']
        risk_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving risk series...")
        print(f"  Path: {risk_path}")
        
        if risk_path.suffix == '.parquet':
            risk_series_df.to_parquet(risk_path, index=False)
        else:
            risk_path = risk_path.with_suffix('.parquet')
            risk_series_df.to_parquet(risk_path, index=False)
        
        print(f"  Saved: {risk_path}")
    
    # Save metrics table
    if 'metrics_table' in outputs:
        metrics_path = project_root / outputs['metrics_table']['path']
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving metrics table...")
        print(f"  Path: {metrics_path}")
        
        if metrics_path.suffix == '.parquet':
            metrics_df.to_parquet(metrics_path, index=False)
        else:
            metrics_path = metrics_path.with_suffix('.parquet')
            metrics_df.to_parquet(metrics_path, index=False)
        
        print(f"  Saved: {metrics_path}")
    
    # Save time-sliced metrics
    if 'time_sliced_metrics_table' in outputs and len(time_sliced_metrics_df) > 0:
        time_sliced_path = project_root / outputs['time_sliced_metrics_table']['path']
        time_sliced_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving time-sliced metrics...")
        print(f"  Path: {time_sliced_path}")
        
        if time_sliced_path.suffix == '.parquet':
            time_sliced_metrics_df.to_parquet(time_sliced_path, index=False)
        else:
            time_sliced_path = time_sliced_path.with_suffix('.parquet')
            time_sliced_metrics_df.to_parquet(time_sliced_path, index=False)
        
        print(f"  Saved: {time_sliced_path}")
    
    # Generate report
    if 'report' in outputs:
        report_path = project_root / outputs['report']['path']
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating report...")
        print(f"  Path: {report_path}")
        
        report_sections = outputs['report'].get('include_sections', [])
        generate_report(
            metrics_df,
            report_path,
            evt_settings=evt_settings,
            report_sections=report_sections
        )
        
        print(f"  Saved: {report_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return risk_series_df, metrics_df, time_sliced_metrics_df


def main():
    """Main entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate EVT-POT VaR/CVaR at asset level with rolling windows'
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
        help='Number of parallel workers (default: auto)'
    )
    
    args = parser.parse_args()
    
    risk_series_df, metrics_df, time_sliced_metrics_df = evaluate_evt_pot_var_cvar(
        config_path=args.config,
        n_jobs=args.n_jobs
    )
    
    print(f"\nResults summary:")
    print(f"  Risk series rows: {len(risk_series_df)}")
    print(f"  Metrics rows: {len(metrics_df)}")
    print(f"  Time-sliced metrics rows: {len(time_sliced_metrics_df)}")
    print(f"\nFirst few risk series rows:")
    print(risk_series_df.head())
    print(f"\nFirst few metrics rows:")
    print(metrics_df.head())


if __name__ == "__main__":
    main()
