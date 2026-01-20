"""
Main evaluation script for Monte Carlo Simulation VaR and CVaR at asset level.

Implements asset-level Monte Carlo evaluation following llm.json spec:
- Asset-level only: no portfolio aggregation
- Rolling window parameter estimation per asset
- Forward return paths simulated per asset
- Empirical VaR and CVaR computed per asset and time
- All backtesting, tail diagnostics, and time-sliced evaluations per asset
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

from .returns import load_panel_prices, compute_daily_returns
from .monte_carlo_calculator import (
    estimate_asset_return_distribution,
    simulate_asset_return_scenarios,
    scale_horizon_covariance,
    compute_var_cvar_from_simulations_efficient
)
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
    dates: pd.DatetimeIndex,
    estimation_windows: List[int],
    confidence_levels: List[float],
    horizons: List[int],
    num_simulations: int,
    distribution_type: str,
    mean_model: Dict,
    covariance_model: Dict,
    scaling_rule: str,
    mc_settings: Dict,
    cache_dir: Optional[Path],
    scenario_dtype: np.dtype = np.float32
) -> List[Dict]:
    """Process a single asset for Monte Carlo VaR/CVaR evaluation."""
    results = []
    returns_array = asset_returns.values.astype(np.float64)
    asset_dates = asset_returns.index  # Use actual asset dates, not full dates index
    num_dates = len(returns_array)
    
    if num_dates == 0:
        return results
    
    for estimation_window in estimation_windows:
        min_periods = min(estimation_window, num_dates)
        
        for time_idx in range(min_periods - 1, num_dates):
            window_start = max(0, time_idx - estimation_window + 1)
            window_end = time_idx + 1
            window_returns = returns_array[window_start:window_end]
            
            if len(window_returns) < min_periods:
                continue
            
            window_returns_df = pd.DataFrame({asset: window_returns})
            
            for horizon in horizons:
                for confidence_level in confidence_levels:
                    try:
                        mean_returns, cov_matrix = estimate_asset_return_distribution(
                            window_returns_df, mean_model, covariance_model
                        )
                        
                        if horizon > 1:
                            cov_matrix = scale_horizon_covariance(cov_matrix, horizon, scaling_rule)
                        
                        # Extract single asset parameters
                        if cov_matrix.ndim == 2:
                            asset_variance = cov_matrix[0, 0]
                        else:
                            asset_variance = float(cov_matrix)
                        
                        if isinstance(mean_returns, np.ndarray):
                            asset_mean = float(mean_returns[0])
                        else:
                            asset_mean = float(mean_returns)
                        
                        # Ensure variance is positive
                        if asset_variance <= 0:
                            continue
                        
                        # Set random seed if specified
                        if random_seed := mc_settings.get('random_seed'):
                            np.random.seed(random_seed + hash(asset) % 10000)
                        
                        # Simulate returns
                        simulated_returns = np.random.normal(
                            asset_mean, np.sqrt(asset_variance), size=num_simulations
                        ).astype(scenario_dtype)
                        
                        # Compute VaR and CVaR
                        var_value, cvar_value = compute_var_cvar_from_simulations_efficient(
                            simulated_returns.reshape(1, -1), confidence_level
                        )
                        
                        # Get the date for this time index
                        current_date = asset_dates[time_idx]
                        
                        result = {
                            'asset': asset,
                            'date': current_date,
                            'confidence_level': confidence_level,
                            'horizon': horizon,
                            'estimation_window': estimation_window,
                            'VaR': float(var_value[0]),
                            'CVaR': float(cvar_value[0]),
                            'estimated_mean': float(asset_mean),
                            'estimated_volatility': float(np.sqrt(asset_variance)),
                            'num_simulations': num_simulations,
                            'distribution': distribution_type
                        }
                        results.append(result)
                    except Exception as e:
                        # Log first few errors for debugging
                        if len(results) == 0 and len([r for r in results if r.get('asset') == asset]) < 3:
                            print(f"    Warning: Error processing {asset} at time_idx={time_idx}: {e}")
                        continue
    
    return results


def _compute_asset_metrics(
    asset: str,
    asset_returns: pd.Series,
    var_series: pd.Series,
    cvar_series: pd.Series,
    dates: pd.DatetimeIndex,
    confidence_level: float,
    horizon: int,
    estimation_window: int,
    mc_settings: Dict
) -> Dict:
    """Compute all metrics for a single asset."""
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
    
    violations = detect_var_violations(aligned_returns, aligned_var, confidence_level)
    hit_rate = compute_hit_rate(violations)
    expected_violation_rate = 1 - confidence_level
    violation_ratio_val = compute_violation_ratio(violations, expected_violation_rate)
    
    kupiec_results = kupiec_test(violations, confidence_level)
    christoffersen_results = christoffersen_test(violations, confidence_level)
    
    traffic_light_config = mc_settings.get('backtesting', {}).get('traffic_light', {})
    if traffic_light_config.get('enabled', True):
        window_size_days = traffic_light_config.get('window_size_days', 250)
        alpha = traffic_light_config.get('alpha', 0.99)
        traffic_light = traffic_light_zone(violations, confidence_level, window_size_days, alpha)
    else:
        traffic_light = traffic_light_zone(violations, confidence_level)
    
    tail_metrics = compute_tail_metrics(aligned_returns, aligned_var, confidence_level)
    cvar_tail_metrics = compute_cvar_tail_metrics(
        aligned_returns, aligned_cvar, aligned_var, confidence_level
    )
    distribution_metrics = compute_distribution_metrics(aligned_returns)
    
    return {
        'asset': asset,
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
    config_path: Optional[Union[str, Path]] = None,
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
    
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    panel_price_path = project_root / config['inputs']['panel_price_path']
    
    if not panel_price_path.exists() and 'preprocessed' in str(panel_price_path):
        panel_price_path = project_root / str(config['inputs']['panel_price_path']).replace('preprocessed', 'processed')
    
    print("=" * 80)
    print("MONTE CARLO SIMULATION FOR VAR/CVAR EVALUATION (ASSET LEVEL)")
    print("=" * 80)
    print(f"\nLoading data...")
    print(f"  Panel prices: {panel_price_path}")
    
    prices = load_panel_prices(panel_price_path)
    asset_universe_config = config['inputs']['asset_universe']
    
    if asset_universe_config.get('mode') == 'from_columns':
        assets = list(prices.columns)
        if asset_universe_config.get('include'):
            assets = [a for a in assets if a in asset_universe_config['include']]
        if asset_universe_config.get('exclude'):
            assets = [a for a in assets if a not in asset_universe_config['exclude']]
    else:
        assets = list(prices.columns)
    
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
    
    print(f"\nLoaded:")
    print(f"  Prices: {len(prices)} dates, {len(prices.columns)} assets")
    print(f"  Data period: {prices.index.min().strftime('%Y-%m-%d')} to {prices.index.max().strftime('%Y-%m-%d')}")
    
    print(f"\nComputing daily returns...")
    daily_returns = compute_daily_returns(prices, method=return_type)
    print(f"  Daily returns: {len(daily_returns)} dates")
    
    mc_settings = config['monte_carlo_settings']
    distribution = mc_settings.get('distribution', 'normal')
    parameter_estimation = mc_settings.get('parameter_estimation', {})
    estimation_windows = mc_settings.get('estimation_windows', [252])
    simulation_config = mc_settings.get('simulation', {})
    num_simulations = simulation_config.get('num_simulations', 100000)
    confidence_levels = mc_settings.get('confidence_levels', [0.95, 0.99])
    
    horizons_config = mc_settings.get('horizons', {})
    if isinstance(horizons_config, dict):
        base_horizon = horizons_config.get('base_horizon', 1)
        scaled_horizons = horizons_config.get('scaled_horizons', [])
        scaling_rule = horizons_config.get('scaling_rule', 'sqrt_time')
        horizons = [base_horizon]
        horizons.extend(scaled_horizons)
        horizons = sorted(set(horizons))
    else:
        horizons = horizons_config if isinstance(horizons_config, list) else [1]
        scaling_rule = 'sqrt_time'
    
    mean_model = {
        'enabled': parameter_estimation.get('demean_returns', True),
        'estimator': parameter_estimation.get('method', 'rolling_sample_moments')
    }
    covariance_model = {'estimator': 'sample_covariance', 'shrinkage': {'enabled': False}}
    distribution_type = 'multivariate_normal' if distribution == 'normal' else distribution
    
    print(f"\nMonte Carlo Settings:")
    print(f"  Distribution: {distribution}")
    print(f"  Number of simulations: {num_simulations:,}")
    print(f"  Estimation windows: {estimation_windows}")
    print(f"  Confidence levels: {confidence_levels}")
    print(f"  Horizons: {horizons}")
    print(f"  Scaling rule: {scaling_rule}")
    
    comp_strategy = config.get('computation_strategy', {})
    parallel_config = comp_strategy.get('rolling_engine', {})
    max_workers = parallel_config.get('max_workers', 'auto')
    
    if max_workers == 'auto':
        import os
        max_workers = os.cpu_count() or 1
    
    if n_jobs is not None:
        max_workers = n_jobs
    
    print(f"\nParallelization:")
    print(f"  Enabled: True")
    print(f"  Workers: {max_workers}")
    
    cache_config = comp_strategy.get('cache', {})
    cache_enabled = cache_config.get('enabled', True)
    cache_dir = None
    if cache_enabled:
        cache_path_str = cache_config.get('parameter_store_path', 'cache/monte_carlo_asset_parameters.parquet')
        cache_dir = project_root / Path(cache_path_str).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {len(assets)} assets...")
    start_time = time.time()
    all_risk_series = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for asset in assets:
            asset_returns = daily_returns[asset].dropna()
            if len(asset_returns) < min_required:
                continue
            
            future = executor.submit(
                _process_single_asset,
                asset, asset_returns, daily_returns.index,
                estimation_windows, confidence_levels, horizons,
                num_simulations, distribution_type, mean_model,
                covariance_model, scaling_rule, mc_settings,
                cache_dir, np.float32
            )
            futures[future] = asset
        
        for future in as_completed(futures):
            asset = futures[future]
            try:
                asset_results = future.result()
                if len(asset_results) > 0:
                    all_risk_series.extend(asset_results)
                else:
                    print(f"  Warning: No results for asset {asset}")
            except Exception as e:
                print(f"  Error processing {asset}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if len(all_risk_series) == 0:
        raise ValueError("No results computed. Check data and configuration.")
    
    risk_series_df = pd.DataFrame(all_risk_series)
    print(f"\nComputed risk series:")
    print(f"  Total rows: {len(risk_series_df):,}")
    
    print(f"\nComputing metrics...")
    all_metrics = []
    
    for asset in risk_series_df['asset'].unique():
        asset_risk = risk_series_df[risk_series_df['asset'] == asset]
        asset_returns = daily_returns[asset].dropna()
        
        for estimation_window in estimation_windows:
            for confidence_level in confidence_levels:
                for horizon in horizons:
                    asset_config_risk = asset_risk[
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
                        daily_returns.index, confidence_level, horizon,
                        estimation_window, mc_settings
                    )
                    
                    if metrics:
                        all_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(all_metrics)
    print(f"  Computed metrics for {len(metrics_df)} asset-configurations")
    
    time_sliced_metrics = []
    if config.get('evaluation', {}).get('time_sliced_metrics', {}).get('enabled', True):
        print(f"\nComputing time-sliced metrics...")
        slice_by = config['evaluation']['time_sliced_metrics'].get('slice_by', ['year', 'quarter', 'month'])
        
        for asset in risk_series_df['asset'].unique():
            asset_risk = risk_series_df[risk_series_df['asset'] == asset]
            asset_returns = daily_returns[asset].dropna()
            
            for estimation_window in estimation_windows:
                for confidence_level in confidence_levels:
                    for horizon in horizons:
                        asset_config_risk = asset_risk[
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
    print(f"\nCompleted evaluation:")
    print(f"  Total runtime: {total_runtime/60:.2f} minutes ({total_runtime:.2f} seconds)")
    
    outputs = config.get('outputs', {})
    
    risk_series_config = outputs.get('risk_series_store', {})
    if risk_series_config:
        risk_series_path = project_root / risk_series_config.get('path', 'results/classical_risk/monte_carlo_asset_var_cvar_series.parquet')
        risk_series_path.parent.mkdir(parents=True, exist_ok=True)
        risk_series_df.to_parquet(risk_series_path, index=False)
        print(f"\nSaved risk series: {risk_series_path}")
    
    metrics_config = outputs.get('metrics_table', {})
    if metrics_config:
        metrics_path = project_root / metrics_config.get('path', 'results/classical_risk/monte_carlo_asset_level_metrics.parquet')
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_parquet(metrics_path, index=False)
        print(f"Saved metrics: {metrics_path}")
    
    time_sliced_config = outputs.get('time_sliced_metrics_table', {})
    if time_sliced_config and len(time_sliced_metrics_df) > 0:
        time_sliced_path = project_root / time_sliced_config.get('path', 'results/classical_risk/monte_carlo_asset_level_time_sliced_metrics.parquet')
        time_sliced_path.parent.mkdir(parents=True, exist_ok=True)
        time_sliced_metrics_df.to_parquet(time_sliced_path, index=False)
        print(f"Saved time-sliced metrics: {time_sliced_path}")
    
    scheme_config = outputs.get('monte_carlo_results_scheme.json', {})
    if scheme_config:
        scheme_path = project_root / scheme_config.get('path', 'results/classical_risk/monte_carlo_asset_level_results_scheme.json')
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
        report_path = project_root / report_config.get('path', 'results/classical_risk/monte_carlo_asset_level_report.md')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_sections = report_config.get('include_sections', [
            'methodology_overview', 'distributional_assumptions',
            'parameter_estimation', 'monte_carlo_simulation_design',
            'var_cvar_construction', 'backtesting_results',
            'time_sliced_backtesting', 'tail_risk_behavior',
            'distributional_characteristics', 'computational_performance',
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

