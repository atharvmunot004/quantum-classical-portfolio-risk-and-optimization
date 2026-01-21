"""
Main evaluation script for Variance-Covariance VaR at asset level.

Orchestrates the entire VaR evaluation pipeline with asset-level estimation:
- Data loading and preprocessing
- Asset-level returns computation (once)
- Rolling mean and volatility estimation per asset
- Rolling VaR time series computation using normal distribution assumption
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
    compute_daily_returns
)
from .var_calculator import (
    compute_rolling_variance_covariance_var,
    compute_rolling_mean_volatility,
    align_returns_and_var
)
from .backtesting import compute_accuracy_metrics
from .metrics import (
    compute_tail_metrics,
    compute_distribution_metrics,
    compute_runtime_metrics
)
from .report_generator import generate_report
from .time_sliced_metrics import compute_time_sliced_metrics


def _process_single_asset_rolling(
    asset_data: Tuple[str, pd.Series],
    estimation_windows: List[int],
    confidence_levels: List[float],
    horizons: List[int],
    scaling_rule: str,
    mean_estimator: str,
    volatility_estimator: str,
    tail_side: str,
    step_size: int,
    safety_checks: Optional[Dict] = None
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Process a single asset with rolling windows to produce VaR time series.
    
    Args:
        asset_data: Tuple of (asset_name, asset_returns_series)
        estimation_windows: List of estimation window sizes
        confidence_levels: List of confidence levels
        horizons: List of horizons
        scaling_rule: Scaling rule for horizon
        mean_estimator: Method for mean estimation
        volatility_estimator: Method for volatility estimation
        tail_side: Tail side ('left' or 'right')
        step_size: Step size for rolling window
        safety_checks: Optional safety check configuration
        
    Returns:
        Tuple of (risk_series_list, metrics_list, runtime_dict)
    """
    asset_name, asset_returns = asset_data
    
    if len(asset_returns) < max(estimation_windows):
        return [], [], {}
    
    risk_series_list = []
    metrics_list = []
    runtime_dict = {
        'total_runtime_ms': 0.0,
        'returns_compute_time_ms': 0.0,
        'mean_estimation_time_ms': 0.0,
        'volatility_estimation_time_ms': 0.0,
        'var_compute_time_ms': 0.0,
        'backtesting_time_ms': 0.0,
        'time_slicing_time_ms': 0.0
    }
    
    start_total = time.time()
    
    # Process each combination of window, confidence, horizon
    for window in estimation_windows:
        if len(asset_returns) < window:
            continue
        
        # Compute rolling mean and volatility
        start_mean_vol = time.time()
        rolling_mean, rolling_volatility = compute_rolling_mean_volatility(
            asset_returns,
            window=window,
            mean_estimator=mean_estimator,
            volatility_estimator=volatility_estimator,
            step_size=step_size
        )
        mean_vol_time = time.time() - start_mean_vol
        runtime_dict['mean_estimation_time_ms'] += mean_vol_time * 1000
        runtime_dict['volatility_estimation_time_ms'] += mean_vol_time * 1000
        
        # Process each confidence level and horizon
        for conf_level in confidence_levels:
            for horizon in horizons:
                try:
                    # Compute rolling VaR
                    start_var = time.time()
                    rolling_var = compute_rolling_variance_covariance_var(
                        asset_returns,
                        window=window,
                        confidence_level=conf_level,
                        horizon=horizon,
                        scaling_rule=scaling_rule,
                        mean_estimator=mean_estimator,
                        volatility_estimator=volatility_estimator,
                        tail_side=tail_side,
                        step_size=step_size
                    )
                    var_time = time.time() - start_var
                    runtime_dict['var_compute_time_ms'] += var_time * 1000
                    
                    if rolling_var.isna().all():
                        continue
                    
                    # Create risk series records
                    for date in rolling_var.index:
                        if pd.isna(rolling_var[date]):
                            continue
                        
                        var_value = float(rolling_var[date])
                        
                        # Safety check: VaR must be positive
                        if safety_checks and safety_checks.get('enabled', False):
                            var_positive_check = safety_checks.get('checks', [])
                            for check in var_positive_check:
                                if check.get('name') == 'var_positive':
                                    if var_value <= 0:
                                        continue  # Skip this timestamp
                        
                        # Get mean and volatility for this date
                        mean_val = float(rolling_mean[date]) if date in rolling_mean.index and not pd.isna(rolling_mean[date]) else np.nan
                        vol_val = float(rolling_volatility[date]) if date in rolling_volatility.index and not pd.isna(rolling_volatility[date]) else np.nan
                        
                        risk_series_list.append({
                            'asset': asset_name,
                            'date': date,
                            'confidence_level': conf_level,
                            'horizon': horizon,
                            'estimation_window': window,
                            'VaR': var_value,
                            'rolling_mean': mean_val,
                            'rolling_volatility': vol_val
                        })
                    
                    # Compute backtesting metrics
                    start_backtest = time.time()
                    
                    # Align returns with VaR
                    aligned_returns, aligned_var = align_returns_and_var(
                        asset_returns,
                        rolling_var
                    )
                    
                    # Remove NaN values
                    valid_mask = ~(pd.isna(aligned_returns) | pd.isna(aligned_var))
                    aligned_returns = aligned_returns[valid_mask]
                    aligned_var = aligned_var[valid_mask]
                    
                    if len(aligned_returns) == 0:
                        continue
                    
                    # Compute accuracy metrics
                    accuracy_metrics = compute_accuracy_metrics(
                        aligned_returns,
                        aligned_var,
                        confidence_level=conf_level,
                        compute_traffic_light=(conf_level == 0.99)
                    )
                    
                    # Compute tail metrics
                    tail_metrics = compute_tail_metrics(
                        aligned_returns,
                        aligned_var,
                        confidence_level=conf_level
                    )
                    
                    # Compute distribution metrics
                    distribution_metrics = compute_distribution_metrics(aligned_returns)
                    
                    # Add rolling mean and volatility to distribution metrics
                    if len(rolling_mean) > 0 and len(rolling_volatility) > 0:
                        # Use the last valid values
                        valid_mean = rolling_mean.dropna()
                        valid_vol = rolling_volatility.dropna()
                        if len(valid_mean) > 0:
                            distribution_metrics['rolling_mean'] = float(valid_mean.iloc[-1])
                        if len(valid_vol) > 0:
                            distribution_metrics['rolling_volatility'] = float(valid_vol.iloc[-1])
                    
                    runtime_dict['backtesting_time_ms'] += (time.time() - start_backtest) * 1000
                    
                    # Combine all metrics
                    metrics_list.append({
                        'asset': asset_name,
                        'confidence_level': conf_level,
                        'horizon': horizon,
                        'estimation_window': window,
                        **accuracy_metrics,
                        **tail_metrics,
                        **distribution_metrics
                    })
                    
                except Exception as e:
                    warnings.warn(f"Error processing asset {asset_name} with window {window}, conf={conf_level}, h={horizon}: {e}")
                    continue
    
    runtime_dict['total_runtime_ms'] = (time.time() - start_total) * 1000
    
    return risk_series_list, metrics_list, runtime_dict


def evaluate_var(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    n_jobs: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main function to evaluate Variance-Covariance VaR at asset level.
    
    This function implements the workflow:
    1. Load and preprocess asset price data
    2. Compute daily returns (once)
    3. For each asset, compute rolling mean/volatility and VaR time series
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
    print("VARIANCE-COVARIANCE VALUE-AT-RISK EVALUATION")
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
    
    # Compute daily returns (once)
    print(f"\nComputing daily returns (method: {return_type})...")
    start_returns = time.time()
    daily_returns = compute_daily_returns(prices, method=return_type)
    returns_time = (time.time() - start_returns) * 1000
    
    # Filter assets with insufficient data
    valid_assets = []
    for asset in daily_returns.columns:
        asset_returns = daily_returns[asset].dropna()
        if len(asset_returns) >= min_required_obs:
            valid_assets.append(asset)
    
    daily_returns = daily_returns[valid_assets]
    print(f"  Daily returns: {len(daily_returns)} dates, {len(daily_returns.columns)} assets (after filtering)")
    print(f"  Returns computation time: {returns_time:.2f} ms")
    
    # Get Variance-Covariance settings
    varcov_settings = config.get('variance_covariance_settings', {})
    distributional_assumption = varcov_settings.get('distributional_assumption', 'normal')
    mean_estimator = varcov_settings.get('mean_estimator', 'sample_mean')
    volatility_estimator = varcov_settings.get('volatility_estimator', 'sample_std')
    confidence_levels = varcov_settings.get('confidence_levels', [0.95, 0.99])
    horizons_config = varcov_settings.get('horizons', {})
    base_horizon = horizons_config.get('base_horizon', 1)
    scaled_horizons = horizons_config.get('scaled_horizons', [10])
    horizons = [base_horizon] + scaled_horizons
    scaling_rule = horizons_config.get('scaling_rule', 'sqrt_time')
    estimation_windows = varcov_settings.get('estimation_windows', [252, 500])
    
    # Rolling settings
    rolling_settings = varcov_settings.get('rolling', {})
    rolling_enabled = rolling_settings.get('enabled', True)
    step_size = rolling_settings.get('step_size', 1)
    warmup_policy = rolling_settings.get('warmup_policy', 'skip_until_window_full')
    
    # Get computation strategy
    computation_strategy = config.get('computation_strategy', {})
    compute_daily_returns_once = computation_strategy.get('compute_daily_returns_once', True)
    parallelization_config = computation_strategy.get('rolling_engine', {})
    max_workers_config = parallelization_config.get('max_workers', 'auto')
    chunk_assets = parallelization_config.get('chunk_assets', 1)
    safety_checks = computation_strategy.get('safety_checks', {})
    runtime_instrumentation = computation_strategy.get('runtime_instrumentation', {})
    enable_runtime_instrumentation = runtime_instrumentation.get('enabled', True)
    
    # Determine number of workers
    if n_jobs is None:
        if max_workers_config == 'auto':
            n_jobs = max(1, cpu_count() - 1)  # Leave one core free
        else:
            n_jobs = int(max_workers_config) if max_workers_config != 'auto' else max(1, cpu_count() - 1)
    
    print(f"\nVariance-Covariance VaR Settings:")
    print(f"  Distributional assumption: {distributional_assumption}")
    print(f"  Mean estimator: {mean_estimator}")
    print(f"  Volatility estimator: {volatility_estimator}")
    print(f"  Confidence levels: {confidence_levels}")
    print(f"  Horizons: {horizons} days (scaling: {scaling_rule})")
    print(f"  Estimation windows: {estimation_windows} days")
    print(f"  Rolling windows: {'Enabled' if rolling_enabled else 'Disabled'}")
    print(f"  Step size: {step_size}")
    print(f"  Parallel workers: {n_jobs}")
    
    # Prepare asset data
    asset_data_list = [(asset, daily_returns[asset].dropna()) for asset in daily_returns.columns]
    
    print(f"\n{'='*80}")
    print("Processing Assets with Rolling Variance-Covariance Estimation")
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
            confidence_levels=confidence_levels,
            horizons=horizons,
            scaling_rule=scaling_rule,
            mean_estimator=mean_estimator,
            volatility_estimator=volatility_estimator,
            tail_side=tail_side,
            step_size=step_size,
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
                confidence_levels,
                horizons,
                scaling_rule,
                mean_estimator,
                volatility_estimator,
                tail_side,
                step_size,
                safety_checks=safety_checks
            )
            all_risk_series.extend(risk_series)
            all_metrics.extend(metrics)
            all_runtimes.append(runtime)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(asset_data_list)} assets...", flush=True)
    
    total_runtime = time.time() - start_time_total
    
    # Create DataFrames
    if len(all_risk_series) == 0:
        raise ValueError("No risk series computed. Check data and configuration.")
    
    risk_series_df = pd.DataFrame(all_risk_series)
    metrics_df = pd.DataFrame(all_metrics)
    
    # Compute runtime metrics
    runtime_metrics_dict = {}
    if len(all_runtimes) > 0 and enable_runtime_instrumentation:
        runtime_metrics_dict = compute_runtime_metrics([r.get('total_runtime_ms', 0) / 1000 for r in all_runtimes])
        # Add detailed runtime metrics
        if enable_runtime_instrumentation:
            measure_list = runtime_instrumentation.get('measure', [])
            for measure in measure_list:
                if measure in ['mean_estimation_time_ms', 'volatility_estimation_time_ms', 'var_compute_time_ms', 
                              'backtesting_time_ms', 'time_slicing_time_ms']:
                    values = [r.get(measure, 0) for r in all_runtimes]
                    if values:
                        runtime_metrics_dict[measure] = float(np.mean(values))
            runtime_metrics_dict['returns_compute_time_ms'] = returns_time
    
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
            asset_returns = daily_returns[asset].dropna()
            
            for conf_level in confidence_levels:
                for horizon in horizons:
                    for window in estimation_windows:
                        config_risk = asset_risk[
                            (asset_risk['confidence_level'] == conf_level) &
                            (asset_risk['horizon'] == horizon) &
                            (asset_risk['estimation_window'] == window)
                        ]
                        
                        if len(config_risk) == 0:
                            continue
                        
                        # Create VaR series
                        var_series = pd.Series(
                            config_risk.set_index('date')['VaR'],
                            index=pd.to_datetime(config_risk['date'])
                        )
                        
                        # Align returns
                        common_dates = asset_returns.index.intersection(var_series.index)
                        if len(common_dates) < min_obs_per_slice:
                            continue
                        
                        aligned_returns = asset_returns.loc[common_dates]
                        aligned_var = var_series.loc[common_dates]
                        
                        # Compute time-sliced metrics
                        for slice_by in slice_by_list:
                            time_slices = compute_time_sliced_metrics(
                                aligned_returns,
                                aligned_var,
                                confidence_level=conf_level,
                                slice_by=slice_by
                            )
                            
                            for ts in time_slices:
                                ts['asset'] = asset
                                ts['confidence_level'] = conf_level
                                ts['horizon'] = horizon
                                ts['estimation_window'] = window
                                ts['slice_type'] = slice_by
                                all_time_sliced.append(ts)
        
        time_sliced_metrics_df = pd.DataFrame(all_time_sliced)
        print(f"  Time-sliced metrics records: {len(time_sliced_metrics_df)}")
    else:
        time_sliced_metrics_df = pd.DataFrame()
    
    # Save results
    outputs = config.get('outputs', {})
    
    # Save parameter store (mean and volatility)
    if 'parameter_store' in outputs:
        param_path = project_root / outputs['parameter_store']['path']
        param_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract parameters from risk series
        param_records = []
        for _, row in risk_series_df.iterrows():
            if not pd.isna(row.get('rolling_mean')) and not pd.isna(row.get('rolling_volatility')):
                param_records.append({
                    'asset': row['asset'],
                    'date': row['date'],
                    'estimation_window': row['estimation_window'],
                    'rolling_mean': row['rolling_mean'],
                    'rolling_volatility': row['rolling_volatility'],
                    'fit_success': True
                })
        
        if param_records:
            param_df = pd.DataFrame(param_records)
            param_df.to_parquet(param_path, index=False)
            print(f"\nSaved parameter store: {param_path}")
    
    # Save risk series
    if 'risk_series_store' in outputs:
        risk_path = project_root / outputs['risk_series_store']['path']
        risk_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving risk series...")
        print(f"  Path: {risk_path}")
        
        # Select only required columns
        required_cols = outputs['risk_series_store'].get('contents', [
            'asset', 'date', 'confidence_level', 'horizon', 'estimation_window', 'VaR'
        ])
        available_cols = [col for col in required_cols if col in risk_series_df.columns]
        risk_series_to_save = risk_series_df[available_cols]
        
        if risk_path.suffix == '.parquet':
            risk_series_to_save.to_parquet(risk_path, index=False)
        else:
            risk_path = risk_path.with_suffix('.parquet')
            risk_series_to_save.to_parquet(risk_path, index=False)
        
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
            varcov_settings=varcov_settings,
            report_sections=report_sections,
            runtime_metrics=runtime_metrics_dict
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
        description='Evaluate Variance-Covariance VaR at asset level with rolling windows'
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
    
    risk_series_df, metrics_df, time_sliced_metrics_df = evaluate_var(
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
