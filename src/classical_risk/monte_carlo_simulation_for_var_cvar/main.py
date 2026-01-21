"""
Main evaluation script for Monte Carlo Simulation VaR and CVaR at asset level.

Implements asset-level Monte Carlo evaluation following llm.json spec.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import warnings
import logging

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from .returns import load_panel_prices, compute_daily_returns
from .monte_carlo_calculator import simulate_returns, simulate_returns_paths, compute_var_cvar, fit_student_t_df
from .backtesting import (
    detect_var_violations,
    compute_hit_rate,
    compute_violation_ratio,
    kupiec_test,
    christoffersen_test,
    traffic_light_zone
)
from .metrics import (
    compute_tail_metrics,
    compute_cvar_tail_metrics,
    compute_distribution_metrics
)
from .time_sliced_metrics import compute_time_sliced_metrics
from .report_generator import generate_report


def _process_single_asset(
    asset: str,
    asset_returns: pd.Series,
    estimation_windows: List[int],
    confidence_levels: List[float],
    horizons: List[int],
    methods: List[Dict],
    num_simulations: int,
    random_seed: Optional[int],
    tail_side: str = 'left'
) -> List[Dict]:
    """
    Process a single asset for Monte Carlo VaR/CVaR evaluation.
    
    Optimizations applied:
    - SIM_001: Simulate once per (asset, window, method, horizon), compute VaR/CVaR for all confidence levels
    - SIM_002: Cache Student-t degrees of freedom per (asset, estimation_window)
    - SIM_004: Simulate max horizon once and reuse paths for all horizons
    """
    results = []
    returns_array = asset_returns.values.astype(np.float64)
    asset_dates = asset_returns.index
    num_dates = len(returns_array)
    
    if num_dates == 0:
        return results
    
    max_horizon = max(horizons) if horizons else 1
    
    # Cache for Student-t degrees of freedom: (estimation_window) -> df
    df_cache: Dict[int, Optional[float]] = {}
    
    for estimation_window in estimation_windows:
        min_periods = min(estimation_window, num_dates)
        
        # Pre-compute Student-t df for this estimation_window if needed
        # We'll compute it on first use and cache it
        student_t_df_cache: Optional[float] = None
        
        for time_idx in range(min_periods - 1, num_dates):
            window_start = max(0, time_idx - estimation_window + 1)
            window_end = time_idx + 1
            window_returns = returns_array[window_start:window_end]
            
            if len(window_returns) < min_periods or not np.all(np.isfinite(window_returns)):
                continue
            
            # Safety check: sufficient variation
            if np.std(window_returns) <= 0:
                continue
            
            current_date = asset_dates[time_idx]
            
            for method_config in methods:
                if not method_config.get('enabled', True):
                    continue
                
                method_name = method_config['name']
                
                # SIM_002: Cache Student-t df per estimation_window
                if method_name == 'parametric_student_t' and student_t_df_cache is None:
                    try:
                        df_config = method_config.get('fit', {}).get('df', {})
                        df_mode = df_config.get('mode', 'mle_or_fixed')
                        if df_mode == 'mle_or_fixed':
                            bounds = tuple(df_config.get('bounds', [2.1, 50.0]))
                            student_t_df_cache = fit_student_t_df(window_returns, bounds)
                        else:
                            student_t_df_cache = df_config.get('fixed_df', 5.0)
                    except Exception as e:
                        logger.debug(f"Failed to fit Student-t df for {asset}: {e}")
                        student_t_df_cache = method_config.get('fit', {}).get('df', {}).get('fallback_df', 5.0)
                
                # Set random seed per asset-method-time
                seed = random_seed
                if seed is not None:
                    seed = seed + hash(f"{asset}_{method_name}_{time_idx}") % 10000
                
                # SIM_004: Simulate max horizon once and reuse paths
                try:
                    # Simulate paths up to max_horizon
                    simulated_paths, fitted_params = simulate_returns_paths(
                        window_returns,
                        method_name,
                        method_config,
                        num_simulations,
                        max_horizon,
                        seed,
                        cached_df=student_t_df_cache if method_name == 'parametric_student_t' else None
                    )
                    
                    if simulated_paths is None or simulated_paths.size == 0:
                        continue
                    
                    if simulated_paths.shape != (num_simulations, max_horizon):
                        continue
                    
                    if not np.any(np.isfinite(simulated_paths)):
                        continue
                    
                    # SIM_001: Loop confidence levels only for compute_var_cvar
                    for horizon in horizons:
                        # Extract horizon-specific paths and aggregate
                        horizon_paths = simulated_paths[:, :horizon]
                        simulated = horizon_paths.sum(axis=1)
                        
                        if not np.any(np.isfinite(simulated)):
                            continue
                        
                        # Compute VaR and CVaR for all confidence levels
                        for confidence_level in confidence_levels:
                            var, cvar = compute_var_cvar(simulated, confidence_level, tail_side)
                            
                            if not (np.isfinite(var) and np.isfinite(cvar) and var > 0 and cvar >= var):
                                continue
                            
                            result = {
                                'asset': asset,
                                'date': current_date,
                                'method': method_name,
                                'confidence_level': confidence_level,
                                'horizon': horizon,
                                'estimation_window': estimation_window,
                                'VaR': float(var),
                                'CVaR': float(cvar),
                                'num_simulations': num_simulations,
                                **fitted_params
                            }
                            results.append(result)
                            
                except Exception as e:
                    # Log error for debugging but continue processing
                    logger.debug(f"Error in {asset} {method_name} at {current_date}: {e}")
                    continue
    
    return results
    
    return results


def _compute_asset_metrics(
    asset: str,
    asset_returns: pd.Series,
    var_series: pd.Series,
    cvar_series: pd.Series,
    confidence_level: float,
    horizon: int,
    estimation_window: int,
    method: str
) -> Dict:
    """
    Compute all metrics for a single asset-method-configuration.
    
    MET_001: Violations are computed once and reused across all backtesting metrics.
    """
    common_dates = asset_returns.index.intersection(var_series.index).intersection(cvar_series.index)
    if len(common_dates) == 0:
        return {}
    
    aligned_returns = asset_returns.loc[common_dates]
    aligned_var = var_series.loc[common_dates]
    aligned_cvar = cvar_series.loc[common_dates]
    
    valid_mask = ~(aligned_returns.isna() | aligned_var.isna() | aligned_cvar.isna())
    if valid_mask.sum() == 0:
        return {}
    
    aligned_returns = aligned_returns[valid_mask]
    aligned_var = aligned_var[valid_mask]
    aligned_cvar = aligned_cvar[valid_mask]
    
    # MET_001: Compute violations once and reuse
    violations = aligned_returns < -aligned_var
    expected_violation_rate = 1 - confidence_level
    
    # Reuse violations for all metrics
    hit_rate = compute_hit_rate(violations)
    violation_ratio_val = compute_violation_ratio(violations, expected_violation_rate)
    kupiec_results = kupiec_test(violations, confidence_level)
    christoffersen_results = christoffersen_test(violations, confidence_level)
    traffic_light = traffic_light_zone(violations, confidence_level)
    
    tail_metrics = compute_tail_metrics(aligned_returns, aligned_var, confidence_level)
    cvar_tail_metrics = compute_cvar_tail_metrics(
        aligned_returns, aligned_cvar, aligned_var, confidence_level
    )
    distribution_metrics = compute_distribution_metrics(aligned_returns)
    
    return {
        'asset': asset,
        'method': method,
        'confidence_level': confidence_level,
        'horizon': horizon,
        'estimation_window': estimation_window,
        'hit_rate': float(hit_rate),
        'violation_ratio': float(violation_ratio_val),
        'num_violations': int(violations.sum()),
        'total_observations': int(len(violations)),
        'expected_violations': float(len(violations) * expected_violation_rate),
        'kupiec_unconditional_coverage': float(kupiec_results['p_value']),
        'kupiec_test_statistic': float(kupiec_results['test_statistic']),
        'kupiec_reject_null': bool(kupiec_results['reject_null']),
        'christoffersen_independence': float(christoffersen_results['independence_p_value']),
        'christoffersen_independence_statistic': float(christoffersen_results['independence_test_statistic']),
        'christoffersen_independence_reject_null': bool(christoffersen_results['independence_reject_null']),
        'christoffersen_conditional_coverage': float(christoffersen_results['conditional_coverage_p_value']),
        'christoffersen_conditional_coverage_statistic': float(christoffersen_results['conditional_coverage_test_statistic']),
        'christoffersen_conditional_coverage_reject_null': bool(christoffersen_results['conditional_coverage_reject_null']),
        'traffic_light_zone': str(traffic_light),
        **tail_metrics,
        **cvar_tail_metrics,
        **distribution_metrics
    }


def evaluate_monte_carlo_var_cvar(
    config_path: Optional[Path] = None,
    config_dict: Optional[Dict] = None,
    n_jobs: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Main function to evaluate VaR and CVaR using Monte Carlo simulation at asset level."""
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
    
    project_root = Path(__file__).parent.parent.parent.parent
    panel_price_path = project_root / config['inputs']['panel_price_path']
    
    if not panel_price_path.exists() and 'preprocessed' in str(panel_price_path):
        panel_price_path = project_root / str(config['inputs']['panel_price_path']).replace('preprocessed', 'processed')
    
    print("=" * 80)
    print("MONTE CARLO SIMULATION FOR VAR/CVAR EVALUATION (ASSET LEVEL)")
    print("=" * 80)
    print(f"\nLoading data...")
    
    prices = load_panel_prices(panel_price_path)
    asset_universe_config = config['inputs']['asset_universe']
    
    if asset_universe_config.get('mode') == 'from_columns':
        assets = list(prices.columns)
        if asset_universe_config.get('include'):
            assets = [a for a in assets if a in asset_universe_config['include']]
        if asset_universe_config.get('exclude'):
            assets = [a for a in assets if a not in asset_universe_config['exclude']]
    
    prices = prices[assets]
    data_settings = config['data_settings']
    return_type = data_settings.get('return_type', 'log')
    
    calendar_settings = data_settings.get('calendar', {})
    if calendar_settings.get('sort_index', True):
        prices = prices.sort_index()
    if calendar_settings.get('drop_duplicate_dates', True):
        prices = prices[~prices.index.duplicated(keep='first')]
    
    missing_policy = data_settings.get('missing_data_policy', {})
    dropna_mode = missing_policy.get('dropna', 'per_asset')
    min_required = missing_policy.get('min_required_observations', 800)
    
    if dropna_mode == 'per_asset':
        valid_assets = []
        for asset in assets:
            asset_prices = prices[asset].dropna()
            if len(asset_prices) >= min_required:
                valid_assets.append(asset)
        assets = valid_assets
        prices = prices[assets]
    
    print(f"  Prices: {len(prices)} dates, {len(prices.columns)} assets")
    
    print(f"\nComputing daily returns...")
    daily_returns = compute_daily_returns(prices, method=return_type)
    
    mc_settings = config['monte_carlo_settings']
    methods = [m for m in mc_settings.get('methods', []) if m.get('enabled', True)]
    estimation_windows = mc_settings.get('estimation_windows', [252])
    num_simulations = mc_settings.get('num_simulations', 20000)
    confidence_levels = mc_settings.get('confidence_levels', [0.95, 0.99])
    
    horizons_config = mc_settings.get('horizons', {})
    if isinstance(horizons_config, dict):
        base_horizon = horizons_config.get('base_horizon', 1)
        scaled_horizons = horizons_config.get('scaled_horizons', [])
        horizons = [base_horizon] + scaled_horizons
        horizons = sorted(set(horizons))
    else:
        horizons = horizons_config if isinstance(horizons_config, list) else [1]
    
    risk_measure_def = mc_settings.get('risk_measure_definition', {})
    tail_side = risk_measure_def.get('tail_side', 'left')
    random_seed = mc_settings.get('random_seed', 42)
    
    print(f"\nMonte Carlo Settings:")
    print(f"  Methods: {[m['name'] for m in methods]}")
    print(f"  Simulations: {num_simulations:,}")
    print(f"  Windows: {estimation_windows}")
    print(f"  Confidence levels: {confidence_levels}")
    print(f"  Horizons: {horizons}")
    
    comp_strategy = config.get('computation_strategy', {})
    parallel_config = comp_strategy.get('rolling_engine', {})
    max_workers = parallel_config.get('max_workers', 'auto')
    
    if max_workers == 'auto':
        import os
        max_workers = os.cpu_count() or 1
    
    if n_jobs is not None:
        max_workers = n_jobs
    
    print(f"\nProcessing {len(assets)} assets with {max_workers} workers...")
    start_time = time.time()
    all_risk_series = []
    
    # SIM_005: Use ProcessPoolExecutor for true multi-core parallelism
    # Prepare arguments for pickling (ProcessPoolExecutor requires picklable arguments)
    process_args = []
    valid_assets = []
    for asset in assets:
        asset_returns = daily_returns[asset].dropna()
        if len(asset_returns) < min_required:
            continue
        valid_assets.append(asset)
        process_args.append((
            asset, asset_returns, estimation_windows,
            confidence_levels, horizons, methods,
            num_simulations, random_seed, tail_side
        ))
    
    # Try ProcessPoolExecutor first, fallback to ThreadPoolExecutor or sequential on failure
    executor_class = ProcessPoolExecutor
    use_parallel = True
    
    # On Windows, ProcessPoolExecutor can have issues, so we'll try it but fallback gracefully
    try:
        # Test if ProcessPoolExecutor works by trying to create one
        test_executor = ProcessPoolExecutor(max_workers=1)
        test_executor.shutdown(wait=False)
        executor_class = ProcessPoolExecutor
        print(f"Using ProcessPoolExecutor with {max_workers} workers")
    except Exception as e:
        print(f"ProcessPoolExecutor not available ({e}), falling back to ThreadPoolExecutor")
        executor_class = ThreadPoolExecutor
    
    # Use executor for CPU-bound Monte Carlo simulation
    try:
        with executor_class(max_workers=max_workers) as executor:
            futures = {}
            for asset, args_tuple in zip(valid_assets, process_args):
                future = executor.submit(_process_single_asset, *args_tuple)
                futures[future] = asset
            
            # PROG_001: Add progress monitoring
            completed_assets = 0
            total_assets = len(valid_assets)
            heartbeat_interval = 30  # seconds
            last_heartbeat = time.time()
            error_count = 0
            
            for future in as_completed(futures):
                asset = futures[future]
                completed_assets += 1
                
                try:
                    asset_results = future.result(timeout=3600)  # 1 hour timeout per asset
                    if len(asset_results) > 0:
                        all_risk_series.extend(asset_results)
                        print(f"  {asset}: {len(asset_results)} results")
                    else:
                        print(f"  {asset}: 0 results (check debug logs)")
                    
                    # Log progress
                    elapsed = time.time() - start_time
                    if completed_assets % max(1, total_assets // 10) == 0 or elapsed - last_heartbeat >= heartbeat_interval:
                        elapsed_minutes = elapsed / 60
                        print(
                            f"Progress: {completed_assets}/{total_assets} assets | "
                            f"Elapsed: {elapsed_minutes:.1f} min | "
                            f"Results: {len(all_risk_series):,} rows | "
                            f"Errors: {error_count}"
                        )
                        logger.info(
                            f"Processed {completed_assets}/{total_assets} assets | "
                            f"Elapsed: {elapsed_minutes:.1f} min | "
                            f"Results: {len(all_risk_series):,} rows | "
                            f"Errors: {error_count}"
                        )
                        last_heartbeat = elapsed
                        
                except Exception as e:
                    error_count += 1
                    error_msg = f"Error processing {asset}: {e}"
                    print(f"  ERROR: {error_msg}")
                    logger.error(error_msg, exc_info=True)
                    continue
    except Exception as e:
        # Fallback to sequential processing if parallel fails completely
        print(f"Parallel processing failed: {e}")
        print("Falling back to sequential processing...")
        logger.warning(f"Parallel processing failed: {e}, falling back to sequential")
        
        for asset, args_tuple in zip(valid_assets, process_args):
            try:
                asset_results = _process_single_asset(*args_tuple)
                if len(asset_results) > 0:
                    all_risk_series.extend(asset_results)
                    print(f"  {asset}: {len(asset_results)} results")
            except Exception as e:
                error_msg = f"Error processing {asset} (sequential): {e}"
                print(f"  ERROR: {error_msg}")
                logger.error(error_msg, exc_info=True)
                continue
    
    if len(all_risk_series) == 0:
        raise ValueError("No results computed. Check data and configuration.")
    
    risk_series_df = pd.DataFrame(all_risk_series)
    print(f"\nComputed risk series: {len(risk_series_df):,} rows")
    
    print(f"\nComputing metrics...")
    all_metrics = []
    
    for asset in risk_series_df['asset'].unique():
        asset_risk = risk_series_df[risk_series_df['asset'] == asset]
        asset_returns = daily_returns[asset].dropna()
        
        for method in asset_risk['method'].unique():
            for estimation_window in estimation_windows:
                for confidence_level in confidence_levels:
                    for horizon in horizons:
                        asset_config_risk = asset_risk[
                            (asset_risk['method'] == method) &
                            (asset_risk['estimation_window'] == estimation_window) &
                            (asset_risk['confidence_level'] == confidence_level) &
                            (asset_risk['horizon'] == horizon)
                        ]
                        
                        if len(asset_config_risk) == 0:
                            continue
                        
                        var_series = pd.Series(
                            asset_config_risk['VaR'].values,
                            index=asset_config_risk['date']
                        )
                        cvar_series = pd.Series(
                            asset_config_risk['CVaR'].values,
                            index=asset_config_risk['date']
                        )
                        
                        metrics = _compute_asset_metrics(
                            asset, asset_returns, var_series, cvar_series,
                            confidence_level, horizon, estimation_window, method
                        )
                        
                        if metrics:
                            all_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(all_metrics)
    print(f"  Computed metrics for {len(metrics_df)} configurations")
    
    time_sliced_metrics = []
    if config.get('evaluation', {}).get('time_sliced_metrics', {}).get('enabled', True):
        print(f"\nComputing time-sliced metrics...")
        slice_by = config['evaluation']['time_sliced_metrics'].get('slice_by', ['year', 'quarter', 'month'])
        
        for asset in risk_series_df['asset'].unique():
            asset_risk = risk_series_df[risk_series_df['asset'] == asset]
            asset_returns = daily_returns[asset].dropna()
            
            for method in asset_risk['method'].unique():
                for estimation_window in estimation_windows:
                    for confidence_level in confidence_levels:
                        for horizon in horizons:
                            asset_config_risk = asset_risk[
                                (asset_risk['method'] == method) &
                                (asset_risk['estimation_window'] == estimation_window) &
                                (asset_risk['confidence_level'] == confidence_level) &
                                (asset_risk['horizon'] == horizon)
                            ]
                            
                            if len(asset_config_risk) == 0:
                                continue
                            
                            var_series = pd.Series(
                                asset_config_risk['VaR'].values,
                                index=asset_config_risk['date']
                            )
                            cvar_series = pd.Series(
                                asset_config_risk['CVaR'].values,
                                index=asset_config_risk['date']
                            )
                            
                            for slice_type in slice_by:
                                sliced = compute_time_sliced_metrics(
                                    asset_returns, var_series, cvar_series,
                                    confidence_level, slice_by=slice_type
                                )
                                
                                for slice_data in sliced:
                                    time_sliced_metrics.append({
                                        'asset': asset,
                                        'method': method,
                                        'slice_type': slice_type,
                                        'slice_value': slice_data['slice'],
                                        'confidence_level': confidence_level,
                                        'horizon': horizon,
                                        'estimation_window': estimation_window,
                                        **slice_data
                                    })
        
        time_sliced_metrics_df = pd.DataFrame(time_sliced_metrics)
        print(f"  Computed time-sliced metrics: {len(time_sliced_metrics_df)} rows")
    else:
        time_sliced_metrics_df = pd.DataFrame()
    
    total_runtime = time.time() - start_time
    print(f"\nCompleted evaluation: {total_runtime/60:.2f} minutes")
    
    # Save outputs
    outputs = config.get('outputs', {})
    
    risk_series_config = outputs.get('risk_series_store', {})
    if risk_series_config:
        risk_series_path = project_root / risk_series_config.get('path', 'results/classical_risk/mcs_asset_var_cvar_series.parquet')
        risk_series_path.parent.mkdir(parents=True, exist_ok=True)
        risk_series_df.to_parquet(risk_series_path, index=False)
        print(f"\nSaved risk series: {risk_series_path}")
    
    metrics_config = outputs.get('metrics_table', {})
    if metrics_config:
        metrics_path = project_root / metrics_config.get('path', 'results/classical_risk/mcs_asset_level_metrics.parquet')
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_parquet(metrics_path, index=False)
        print(f"Saved metrics: {metrics_path}")
    
    time_sliced_config = outputs.get('time_sliced_metrics_table', {})
    if time_sliced_config and len(time_sliced_metrics_df) > 0:
        time_sliced_path = project_root / time_sliced_config.get('path', 'results/classical_risk/mcs_asset_level_time_sliced_metrics.parquet')
        time_sliced_path.parent.mkdir(parents=True, exist_ok=True)
        time_sliced_metrics_df.to_parquet(time_sliced_path, index=False)
        print(f"Saved time-sliced metrics: {time_sliced_path}")
    
    scheme_config = outputs.get('mcs_results_scheme.json', {})
    if scheme_config:
        scheme_path = project_root / scheme_config.get('path', 'results/classical_risk/mcs_asset_level_results_scheme.json')
        scheme_path.parent.mkdir(parents=True, exist_ok=True)
        
        scheme = {
            'risk_series': {
                'path': str(risk_series_config.get('path', '')),
                'rows': len(risk_series_df),
                'columns': list(risk_series_df.columns)
            },
            'metrics': {
                'path': str(metrics_config.get('path', '')),
                'rows': len(metrics_df),
                'columns': list(metrics_df.columns)
            },
            'time_sliced_metrics': {
                'path': str(time_sliced_config.get('path', '')),
                'rows': len(time_sliced_metrics_df),
                'columns': list(time_sliced_metrics_df.columns) if len(time_sliced_metrics_df) > 0 else []
            }
        }
        
        with open(scheme_path, 'w') as f:
            json.dump(scheme, f, indent=2, default=str)
        print(f"Saved results scheme: {scheme_path}")
    
    report_config = outputs.get('report', {})
    if report_config:
        report_path = project_root / report_config.get('path', 'results/classical_risk/mcs_asset_level_report.md')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_sections = report_config.get('include_sections', [
            'methodology_overview',
            'monte_carlo_methods_and_assumptions',
            'rolling_forecast_construction',
            'backtesting_results',
            'time_sliced_backtesting',
            'tail_risk_behavior',
            'distributional_characteristics',
            'key_insights'
        ])
        
        generate_report(
            metrics_df, report_path,
            monte_carlo_settings=mc_settings,
            risk_series_df=risk_series_df,
            time_sliced_metrics_df=time_sliced_metrics_df,
            report_sections=report_sections
        )
        print(f"Generated report: {report_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return risk_series_df, metrics_df, time_sliced_metrics_df


def main():
    """Main entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate Monte Carlo Simulation VaR/CVaR at asset level'
    )
    parser.add_argument('--config', type=str, default=None,
        help='Path to configuration JSON file (default: llm.json in same directory)')
    parser.add_argument('--n-jobs', type=int, default=None,
        help='Number of parallel workers (default: auto)')
    
    args = parser.parse_args()
    
    risk_series_df, metrics_df, time_sliced_metrics_df = evaluate_monte_carlo_var_cvar(
        config_path=args.config, n_jobs=args.n_jobs
    )
    
    print(f"\nResults summary:")
    print(f"  Risk series: {len(risk_series_df)} rows")
    print(f"  Metrics: {len(metrics_df)} rows")
    print(f"  Time-sliced metrics: {len(time_sliced_metrics_df)} rows")


if __name__ == "__main__":
    main()
