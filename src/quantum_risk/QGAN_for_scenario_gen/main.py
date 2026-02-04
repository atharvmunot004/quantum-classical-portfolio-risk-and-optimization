"""
Main evaluation script for QGAN-based scenario generation at asset level.

Orchestrates the entire pipeline:
- Data loading, returns, losses
- Rolling QGAN training
- Scenario generation
- Risk metrics (VaR/CVaR from scenarios)
- Evaluation metrics (distribution fidelity, tail preservation, stylized facts)
- Time-sliced metrics
- Report generation
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
import warnings

from .returns import (
    load_panel_prices,
    compute_daily_returns,
    compute_losses_from_returns,
    standardize_returns,
    inverse_standardize
)
from .qgan_model import QuantumGenerator, ClassicalDiscriminator
from .discretization import create_uniform_grid, map_bitstrings_to_returns
from .qgan_training import QGANTrainer
from .scenario_generation import generate_scenarios, compute_var_cvar_from_scenarios
from .metrics import (
    compute_distribution_fidelity_metrics,
    compute_tail_metrics,
    compute_stylized_facts_metrics,
    compute_mode_collapse_score,
    compute_quantum_specific_metrics
)
from .time_sliced_metrics import compute_time_sliced_metrics
from .report_generator import generate_report
from .cache import QGANParameterCache


def _process_single_asset_single_window(
    asset_name: str,
    asset_returns: pd.Series,
    asset_losses: pd.Series,
    window: int,
    step_size: int,
    qgan_settings: Dict,
    data_settings: Dict,
    safety_checks: Optional[Dict],
    cache: Optional[QGANParameterCache],
    date_idx: int
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """
    Process a single asset for a single rolling window at a specific date.
    
    Returns:
        risk_series_list, metrics_list, scenario_list, runtime_dict
    """
    risk_series_list = []
    metrics_list = []
    scenario_list = []
    runtime = {
        'total_runtime_ms': 0.0,
        'preprocessing_time_ms': 0.0,
        'qgan_training_time_ms': 0.0,
        'scenario_generation_time_ms': 0.0,
        'risk_compute_time_ms': 0.0,
        'evaluation_time_ms': 0.0,
    }
    
    start_total = time.time()
    
    # Check if we have enough data
    if date_idx < window:
        return risk_series_list, metrics_list, scenario_list, runtime
    
    # Get window data
    window_start = max(0, date_idx - window)
    window_end = date_idx
    window_returns = asset_returns.iloc[window_start:window_end]
    window_losses = asset_losses.iloc[window_start:window_end]
    
    if len(window_returns) < window * 0.8:  # Require at least 80% of window
        return risk_series_list, metrics_list, scenario_list, runtime
    
    current_date = asset_returns.index[date_idx]
    
    # Check cache
    cache_key = {
        'asset': asset_name,
        'date': current_date,
        'estimation_window': window,
        'generator_num_qubits': qgan_settings.get('generator', {}).get('num_qubits', 6),
        'ansatz_layers': qgan_settings.get('generator', {}).get('ansatz', {}).get('layers', 3),
        'shots': qgan_settings.get('execution', {}).get('shots', 4096),
        'batch_size': qgan_settings.get('optimization', {}).get('batch_size', 256),
        'max_iterations': qgan_settings.get('optimization', {}).get('max_iterations', 300),
        'return_type': data_settings.get('return_type', 'log'),
    }
    
    # Preprocessing (always needed for grid and stats)
    start_prep = time.time()
    
    # Standardize returns
    std_config = data_settings.get('standardization', {})
    if std_config.get('enabled', True):
        window_returns_std, stats = standardize_returns(
            window_returns.values,
            method=std_config.get('method', 'robust_zscore'),
            clip_range=tuple(std_config.get('clip_range', [-6.0, 6.0]))
        )
    else:
        window_returns_std = window_returns.values
        stats = {'mean': 0, 'std': 1, 'median': 0, 'mad': 1}
    
    # Create discretization grid
    data_repr = qgan_settings.get('data_representation', {})
    dist_support = data_repr.get('distribution_support', {})
    discretization = data_repr.get('discretization', {})
    
    grid = create_uniform_grid(
        window_returns_std,
        num_bins=discretization.get('num_bins', 64),
        clip_quantiles=(
            dist_support.get('lower_quantile', 0.001),
            dist_support.get('upper_quantile', 0.999)
        )
    )
    
    runtime['preprocessing_time_ms'] = (time.time() - start_prep) * 1000
    
    # Initialize QGAN components (always needed)
    gen_cfg = qgan_settings.get('generator', {})
    disc_cfg = qgan_settings.get('discriminator', {})
    exec_cfg = qgan_settings.get('execution', {})
    
    generator = QuantumGenerator(
        num_qubits=gen_cfg.get('num_qubits', 6),
        ansatz_layers=gen_cfg.get('ansatz', {}).get('layers', 3),
        entanglement=gen_cfg.get('ansatz', {}).get('entanglement', 'linear'),
        rotation_gates=gen_cfg.get('ansatz', {}).get('rotation_gates', ['ry', 'rz']),
        backend=exec_cfg.get('backend', 'aer_simulator'),
        shots=exec_cfg.get('shots', 4096),
        seed=qgan_settings.get('optimization', {}).get('random_seed', 42)
    )
    
    # Check for GPU availability
    use_gpu = qgan_settings.get('execution', {}).get('use_gpu', True)
    
    discriminator = ClassicalDiscriminator(
            input_dim=1,
            hidden_layers=disc_cfg.get('hidden_layers', [64, 32]),
            activation=disc_cfg.get('activation', 'relu'),
            dropout=disc_cfg.get('dropout', 0.1),
            label_smoothing=disc_cfg.get('label_smoothing', 0.0),
            seed=qgan_settings.get('optimization', {}).get('random_seed', 42),
            use_gpu=use_gpu
        )
    
    # Check cache for trained parameters
    cached_result = cache.get(cache_key) if cache else None
    training_result = None
    
    if cached_result and cached_result.get('fit_success', False):
        generator_params = np.array(cached_result.get('generator_params', []))
        training_time = cached_result.get('training_time_ms', 0)
        runtime['qgan_training_time_ms'] = training_time
        # Reconstruct training result for metrics
        training_result = {
            'generator_params': generator_params,
            'generator_losses': cached_result.get('generator_losses', []),
            'discriminator_losses': cached_result.get('discriminator_losses', []),
            'final_generator_loss': cached_result.get('generator_final_loss', np.nan),
            'final_discriminator_loss': cached_result.get('discriminator_final_loss', np.nan),
            'training_time_ms': training_time,
        }
    else:
        # Train QGAN
        start_train = time.time()
        trainer = QGANTrainer(
            generator=generator,
            discriminator=discriminator,
            grid=grid,
            optimizer_config=qgan_settings.get('optimization', {}),
            early_stopping=qgan_settings.get('optimization', {}).get('early_stopping', {}),
            seed=qgan_settings.get('optimization', {}).get('random_seed', 42)
        )
        
        opt_cfg = qgan_settings.get('optimization', {})
        adv_steps = opt_cfg.get('adversarial_steps', {})
        
        training_result = trainer.train(
            real_data=window_returns_std,
            max_iterations=opt_cfg.get('max_iterations', 300),
            batch_size=opt_cfg.get('batch_size', 256),
            disc_steps=adv_steps.get('discriminator_steps_per_iter', 1),
            gen_steps=adv_steps.get('generator_steps_per_iter', 1),
            verbose=False
        )
        
        generator_params = training_result['generator_params']
        runtime['qgan_training_time_ms'] = training_result['training_time_ms']
        
        # Cache results
        if cache:
            cache.set(cache_key, {
                'generator_params': generator_params.tolist(),
                'generator_final_loss': training_result['final_generator_loss'],
                'discriminator_final_loss': training_result['final_discriminator_loss'],
                'generator_losses': training_result.get('generator_losses', []),
                'discriminator_losses': training_result.get('discriminator_losses', []),
                'fit_success': True,
                'training_time_ms': training_result['training_time_ms'],
            })
    
    # Generate scenarios
    start_scen = time.time()
    scen_cfg = qgan_settings.get('scenario_generation', {})
    num_scenarios = scen_cfg.get('num_scenarios_per_timestamp', 10000)
    horizons = scen_cfg.get('horizons', {})
    
    scenarios = generate_scenarios(
        generator=generator,
        generator_params=generator_params,
        grid=grid,
        num_scenarios=num_scenarios,
        horizons=horizons
    )
    runtime['scenario_generation_time_ms'] = (time.time() - start_scen) * 1000
    
    # Compute risk metrics from scenarios
    start_risk = time.time()
    confidence_levels = [0.95, 0.99]
    risk_metrics = compute_var_cvar_from_scenarios(scenarios, confidence_levels)
    runtime['risk_compute_time_ms'] = (time.time() - start_risk) * 1000
    
    # Store risk series
    for horizon, horizon_data in risk_metrics.items():
        for conf, risk_vals in horizon_data.items():
            risk_series_list.append({
                'asset': asset_name,
                'date': current_date,
                'confidence_level': conf,
                'horizon': horizon,
                'estimation_window': window,
                'VaR_generated': risk_vals['VaR'],
                'CVaR_generated': risk_vals['CVaR'],
            })
    
    # Store scenarios
    for horizon, scen_data in scenarios.items():
        for i, (ret, loss) in enumerate(zip(scen_data['returns'], scen_data['losses'])):
            scenario_list.append({
                'asset': asset_name,
                'date': current_date,
                'scenario_id': i,
                'horizon': horizon,
                'generated_return': ret,
                'generated_loss': loss,
            })
    
    # Evaluation metrics
    start_eval = time.time()
    if len(scenarios) > 0:
        # Use first horizon for evaluation
        first_horizon = list(scenarios.keys())[0]
        gen_returns = scenarios[first_horizon]['returns']
        gen_losses = scenarios[first_horizon]['losses']
        
        # Inverse standardize generated returns
        if std_config.get('enabled', True):
            gen_returns_orig = inverse_standardize(
                gen_returns, stats, std_config.get('method', 'robust_zscore')
            )
        else:
            gen_returns_orig = gen_returns
        
        # Real returns for comparison (out-of-sample)
        if date_idx < len(asset_returns):
            real_return_os = asset_returns.iloc[date_idx]
            real_loss_os = asset_losses.iloc[date_idx]
        else:
            real_return_os = np.nan
            real_loss_os = np.nan
        
        # Distribution fidelity (compare generated to training window)
        dist_metrics = compute_distribution_fidelity_metrics(
            window_returns.values, gen_returns_orig[:len(window_returns)]
        )
        
        # Tail metrics
        tail_metrics = compute_tail_metrics(
            window_losses.values, gen_losses[:len(window_losses)], confidence_levels
        )
        
        # Stylized facts
        stylized_metrics = compute_stylized_facts_metrics(
            window_returns.values, gen_returns_orig[:len(window_returns)]
        )
        
        # Mode collapse
        mode_collapse = compute_mode_collapse_score(gen_returns)
        
        # Quantum-specific metrics
        quantum_metrics = {}
        if training_result is not None:
            quantum_metrics = compute_quantum_specific_metrics(
                generator, training_result, qgan_settings
            )
        
        metrics_list.append({
            'asset': asset_name,
            'date': current_date,
            'estimation_window': window,
            **dist_metrics,
            **tail_metrics,
            **stylized_metrics,
            'mode_collapse_score': mode_collapse,
            **quantum_metrics,
        })
    
    runtime['evaluation_time_ms'] = (time.time() - start_eval) * 1000
    runtime['total_runtime_ms'] = (time.time() - start_total) * 1000
    
    return risk_series_list, metrics_list, scenario_list, runtime


def _process_single_asset(
    asset_name: str,
    asset_returns: pd.Series,
    asset_losses: pd.Series,
    estimation_windows: List[int],
    qgan_settings: Dict,
    data_settings: Dict,
    safety_checks: Optional[Dict],
    cache: Optional[QGANParameterCache],
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """
    Process a single asset across all rolling windows.
    """
    all_risk = []
    all_metrics = []
    all_scenarios = []
    all_runtimes = []
    
    rolling_cfg = qgan_settings.get('rolling', {})
    step_size = rolling_cfg.get('step_size', 5)
    warmup_policy = rolling_cfg.get('warmup_policy', 'skip_until_window_full')
    
    for window in estimation_windows:
        if len(asset_returns) < window:
            continue
        
        # Determine date indices to process
        if warmup_policy == 'skip_until_window_full':
            start_idx = window
        else:
            start_idx = 0
        
        # Process at step_size intervals
        for date_idx in range(start_idx, len(asset_returns), step_size):
            try:
                risk, metrics, scenarios, runtime = _process_single_asset_single_window(
                    asset_name, asset_returns, asset_losses, window, step_size,
                    qgan_settings, data_settings, safety_checks, cache, date_idx
                )
                if len(risk) > 0:
                    all_risk.extend(risk)
                if len(metrics) > 0:
                    all_metrics.extend(metrics)
                if len(scenarios) > 0:
                    all_scenarios.extend(scenarios)
                all_runtimes.append(runtime)
            except Exception as e:
                import traceback
                warnings.warn(f"Error processing {asset_name} window={window} date_idx={date_idx}: {e}")
                if date_idx == start_idx:  # Only print traceback for first error
                    traceback.print_exc()
                continue
    
    # Aggregate runtime
    agg_runtime = {}
    if all_runtimes:
        for key in all_runtimes[0].keys():
            agg_runtime[key] = sum(r.get(key, 0) for r in all_runtimes)
    
    return all_risk, all_metrics, all_scenarios, agg_runtime


def evaluate_qgan_scenarios(
    config_path: Optional[Path] = None,
    config_dict: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main evaluation function for QGAN scenario generation.
    
    Returns:
        risk_df, metrics_df, scenarios_df, time_sliced_df
    """
    if config_dict is None:
        config_path = config_path or Path(__file__).parent / "llm.json"
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = config_dict
    
    project_root = Path(__file__).parent.parent.parent.parent
    panel_path = project_root / config['inputs']['panel_price_path']
    if not panel_path.exists() and 'preprocessed' in str(panel_path):
        panel_path = project_root / str(config['inputs']['panel_price_path']).replace(
            'preprocessed', 'processed'
        )
    
    print("=" * 80)
    print("QGAN SCENARIO GENERATION ASSET-LEVEL EVALUATION")
    print("=" * 80)
    print(f"\nLoading: {panel_path}")
    
    prices = load_panel_prices(panel_path)
    asset_cfg = config['inputs'].get('asset_universe', {})
    if asset_cfg.get('include'):
        prices = prices[asset_cfg['include']]
    if asset_cfg.get('exclude'):
        prices = prices.drop(columns=asset_cfg['exclude'], errors='ignore')
    
    data_settings = config.get('data_settings', {})
    if data_settings.get('calendar', {}).get('sort_index', True):
        prices = prices.sort_index()
    if data_settings.get('calendar', {}).get('drop_duplicate_dates', True):
        prices = prices[~prices.index.duplicated(keep='first')]
    
    return_type = data_settings.get('return_type', 'log')
    min_obs = data_settings.get('missing_data_policy', {}).get('min_required_observations', 800)
    
    returns = compute_daily_returns(prices, method=return_type)
    losses = compute_losses_from_returns(returns)
    
    valid = [c for c in losses.columns if losses[c].dropna().shape[0] >= min_obs]
    losses = losses[valid]
    returns = returns[valid]
    
    print(f"  Assets: {len(losses.columns)}, dates: {len(losses)}")
    
    qgan_settings = config.get('qgan_settings', {})
    estimation_windows = qgan_settings.get('rolling', {}).get('estimation_windows', [252, 500])
    safety_checks = config.get('computation_strategy', {}).get('safety_checks')
    cache = None
    cache_cfg = config.get('computation_strategy', {}).get('cache', {})
    if cache_cfg.get('enabled') and cache_cfg.get('parameter_store_path'):
        cache_path = project_root / cache_cfg['parameter_store_path']
        cache = QGANParameterCache(cache_path, cache_cfg.get('key_fields'))
    
    print(f"  Windows: {estimation_windows}")
    
    all_risk = []
    all_metrics = []
    all_scenarios = []
    all_runtimes = []
    
    for idx, asset in enumerate(losses.columns):
        print(f"  Processing asset {idx+1}/{len(losses.columns)}: {asset}...", flush=True)
        asset_returns = returns[asset].dropna()
        asset_losses = losses[asset].dropna()
        
        # Align indices
        common_idx = asset_returns.index.intersection(asset_losses.index)
        asset_returns = asset_returns.loc[common_idx]
        asset_losses = asset_losses.loc[common_idx]
        
        risk, metrics, scenarios, runtime = _process_single_asset(
            asset,
            asset_returns,
            asset_losses,
            estimation_windows,
            qgan_settings,
            data_settings,
            safety_checks,
            cache,
        )
        all_risk.extend(risk)
        all_metrics.extend(metrics)
        all_scenarios.extend(scenarios)
        all_runtimes.append(runtime)
    
    if not all_risk:
        raise ValueError("No risk series computed.")
    
    risk_df = pd.DataFrame(all_risk)
    metrics_df = pd.DataFrame(all_metrics)
    scenarios_df = pd.DataFrame(all_scenarios)
    
    # Aggregate runtime metrics
    runtime_dict = {}
    if all_runtimes:
        for key in all_runtimes[0].keys():
            values = [r.get(key, 0) for r in all_runtimes if r.get(key, 0) > 0]
            if values:
                runtime_dict[key] = float(np.mean(values))
                if 'training_time' in key.lower():
                    runtime_dict[key.replace('mean', 'p95')] = float(np.percentile(values, 95))
    
    # Add summary runtime metrics
    if all_runtimes:
        total_times = [r.get('total_runtime_ms', 0) for r in all_runtimes]
        training_times = [r.get('qgan_training_time_ms', 0) for r in all_runtimes if r.get('qgan_training_time_ms', 0) > 0]
        if total_times:
            runtime_dict['total_runtime_ms'] = float(np.sum(total_times))
            runtime_dict['runtime_per_asset_ms'] = float(np.mean(total_times))
        if training_times:
            runtime_dict['mean_training_time_ms'] = float(np.mean(training_times))
            runtime_dict['p95_training_time_ms'] = float(np.percentile(training_times, 95))
    
    if cache and cache_cfg.get('cache_hit_ratio_metric'):
        runtime_dict['cache_hit_ratio'] = cache.get_hit_ratio()
    
    print(f"\nCompleted: {len(risk_df)} risk records, {len(metrics_df)} metric records, {len(scenarios_df)} scenarios")
    
    # Time-sliced metrics
    time_sliced_config = config.get('evaluation', {}).get('time_sliced_metrics', {})
    time_sliced_df = pd.DataFrame()
    if time_sliced_config.get('enabled', True):
        slice_by_list = time_sliced_config.get('slice_by', ['year', 'quarter', 'month'])
        min_obs_slice = time_sliced_config.get('minimum_observations_per_slice', 60)
        confidence_levels = [0.95, 0.99]
        all_ts = []
        
        for asset in risk_df['asset'].unique():
            asset_returns_full = returns[asset].dropna()
            asset_losses_full = losses[asset].dropna()
            common_idx = asset_returns_full.index.intersection(asset_losses_full.index)
            asset_returns_full = asset_returns_full.loc[common_idx]
            asset_losses_full = asset_losses_full.loc[common_idx]
            
            # Get generated scenarios for this asset
            asset_scenarios = scenarios_df[scenarios_df['asset'] == asset]
            if asset_scenarios.empty:
                continue
            
            # Group by date and aggregate scenarios
            for date in asset_scenarios['date'].unique():
                date_scenarios = asset_scenarios[asset_scenarios['date'] == date]
                gen_returns_series = pd.Series(
                    date_scenarios['generated_return'].values,
                    index=[date] * len(date_scenarios)
                )
                gen_losses_series = pd.Series(
                    date_scenarios['generated_loss'].values,
                    index=[date] * len(date_scenarios)
                )
                
                # Get real data for this date
                if date in asset_returns_full.index:
                    real_ret_series = pd.Series([asset_returns_full.loc[date]], index=[date])
                    real_loss_series = pd.Series([asset_losses_full.loc[date]], index=[date])
                    
                    for sb in slice_by_list:
                        for ts in compute_time_sliced_metrics(
                            real_ret_series, gen_returns_series,
                            real_loss_series, gen_losses_series,
                            confidence_levels, sb, min_obs_slice
                        ):
                            ts.update(
                                asset=asset,
                                estimation_window=estimation_windows[0],  # Use first window
                                slice_type=sb,
                            )
                            all_ts.append(ts)
        
        if all_ts:
            time_sliced_df = pd.DataFrame(all_ts)
    
    # Save outputs
    outputs = config.get('outputs', {})
    
    if 'parameter_store' in outputs:
        out_path = project_root / outputs['parameter_store']['path']
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Parameters are already cached, just save cache
        if cache:
            cache.save_cache([])
    
    if 'scenario_store' in outputs:
        out_path = project_root / outputs['scenario_store']['path']
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cols = outputs['scenario_store'].get('contents', scenarios_df.columns.tolist())
        cols = [c for c in cols if c in scenarios_df.columns]
        scenarios_df[cols].to_parquet(out_path, index=False)
        print(f"Saved scenarios: {out_path}")
    
    if 'risk_series_store' in outputs:
        out_path = project_root / outputs['risk_series_store']['path']
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cols = outputs['risk_series_store'].get('contents', risk_df.columns.tolist())
        cols = [c for c in cols if c in risk_df.columns]
        risk_df[cols].to_parquet(out_path, index=False)
        print(f"Saved risk series: {out_path}")
    
    if 'metrics_table' in outputs:
        out_path = project_root / outputs['metrics_table']['path']
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_parquet(out_path, index=False)
        print(f"Saved metrics: {out_path}")
    
    if 'time_sliced_metrics_table' in outputs and not time_sliced_df.empty:
        out_path = project_root / outputs['time_sliced_metrics_table']['path']
        out_path.parent.mkdir(parents=True, exist_ok=True)
        time_sliced_df.to_parquet(out_path, index=False)
        print(f"Saved time-sliced metrics: {out_path}")
    
    if 'report' in outputs:
        out_path = project_root / outputs['report']['path']
        generate_report(
            metrics_df,
            out_path,
            quantum_settings=qgan_settings,
            report_sections=outputs['report'].get('include_sections'),
            runtime_metrics=runtime_dict,
        )
        print(f"Saved report: {out_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return risk_df, metrics_df, scenarios_df, time_sliced_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='QGAN scenario generation asset-level evaluation')
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    risk_df, metrics_df, scenarios_df, ts_df = evaluate_qgan_scenarios(config_path=args.config)
    print(f"\nRisk rows: {len(risk_df)}, Metrics rows: {len(metrics_df)}, "
          f"Scenarios: {len(scenarios_df)}, Time-sliced: {len(ts_df)}")
    print(risk_df.head())


if __name__ == "__main__":
    main()
