"""
Main evaluation script for QAOA-based CVaR at asset level.

Orchestrates the entire pipeline:
- Data loading, returns, losses
- Rolling QAOA with CVaR objective
- Backtesting, metrics, time-sliced metrics
- Report generation
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
import warnings

from .returns import load_panel_prices, compute_daily_returns, compute_losses_from_returns
from .qaoa_cvar import run_rolling_qaoa_cvar
from .backtesting import compute_accuracy_metrics
from .metrics import (
    compute_tail_metrics,
    compute_distribution_metrics,
    compute_runtime_metrics,
)
from .report_generator import generate_report
from .time_sliced_metrics import compute_time_sliced_metrics


def _process_single_asset(
    asset_name: str,
    asset_losses: pd.Series,
    estimation_windows: List[int],
    confidence_levels: List[float],
    quantum_risk_settings: Dict,
    problem_encoding: Dict,
    qaoa_settings: Dict,
    safety_checks: Optional[Dict],
    cache: Optional[Any],
) -> Tuple[List[Dict], List[Dict], Dict]:
    risk_series_list = []
    metrics_list = []
    runtime = {
        'total_runtime_ms': 0.0,
        'returns_compute_time_ms': 0.0,
        'loss_discretization_time_ms': 0.0,
        'hamiltonian_build_time_ms': 0.0,
        'qaoa_optimize_time_ms': 0.0,
        'cvar_eval_time_ms': 0.0,
        'backtesting_time_ms': 0.0,
        'time_slicing_time_ms': 0.0,
    }

    start_total = time.time()
    losses_arr = asset_losses.values
    num_qubits = problem_encoding.get('num_state_qubits', 8)
    discretization = problem_encoding.get('loss_discretization', {})
    discretization_config = {
        'clip_quantiles': discretization.get('clip_quantiles', [0.001, 0.999]),
        'rescale_to_unit_interval': discretization.get('rescale_to_unit_interval', True),
    }
    optimizer_cfg = qaoa_settings.get('optimizer', {})
    exec_cfg = qaoa_settings.get('execution', {})
    reps_grid = qaoa_settings.get('ansatz', {}).get('reps_grid', [1])

    for window in estimation_windows:
        if len(asset_losses) < window:
            continue

        for conf in confidence_levels:
            for reps in reps_grid:
                try:
                    cvar_series, best_energy_series, param_records = run_rolling_qaoa_cvar(
                        losses=losses_arr,
                        window=window,
                        confidence_level=conf,
                        step_size=quantum_risk_settings.get('rolling', {}).get('step_size', 1),
                        num_qubits=num_qubits,
                        reps=reps,
                        shots=exec_cfg.get('shots', 4000),
                        maxiter=optimizer_cfg.get('maxiter', 200),
                        tol=optimizer_cfg.get('tol', 1e-4),
                        seed=quantum_risk_settings.get('random_seed', 42),
                        discretization_config=discretization_config,
                    )

                    step = quantum_risk_settings.get('rolling', {}).get('step_size', 1)
                    valid_idx = np.where(~np.isnan(cvar_series))[0]
                    dates = asset_losses.index[valid_idx]

                    param_by_idx = {r['index']: r for r in param_records}
                    for j, i in enumerate(valid_idx):
                        cvar_val = float(cvar_series[i])
                        best_e = float(best_energy_series[i])
                        pr = param_by_idx.get(i, {})
                        nfev = pr.get('nfev', 0)
                        best_params = pr.get('best_parameters', [])
                        if safety_checks and safety_checks.get('enabled'):
                            for c in safety_checks.get('checks', []):
                                if c.get('name') == 'cvar_nonnegative' and cvar_val < 0:
                                    continue
                                if c.get('name') == 'cvar_monotonicity':
                                    pass
                        risk_series_list.append({
                            'asset': asset_name,
                            'date': dates[j],
                            'confidence_level': conf,
                            'estimation_window': window,
                            'reps': reps,
                            'CVaR': cvar_val,
                            'best_energy': best_e,
                            'nfev': nfev,
                            'best_parameters': best_params,
                        })

                    aligned_losses = asset_losses.loc[dates]
                    aligned_cvar = pd.Series(cvar_series[valid_idx], index=dates)
                    valid = ~(aligned_losses.isna() | aligned_cvar.isna())
                    aligned_losses = aligned_losses[valid]
                    aligned_cvar = aligned_cvar[valid]

                    if len(aligned_losses) == 0:
                        continue

                    start_bt = time.time()
                    acc = compute_accuracy_metrics(
                        aligned_losses, aligned_cvar, confidence_level=conf,
                        compute_traffic_light=(conf == 0.99),
                    )
                    tail = compute_tail_metrics(aligned_losses, aligned_cvar, confidence_level=conf)
                    rets = -aligned_losses
                    dist_met = compute_distribution_metrics(rets)
                    runtime['backtesting_time_ms'] += (time.time() - start_bt) * 1000

                    if param_records:
                        runtime['qaoa_optimize_time_ms'] += sum(
                            r.get('total_time_ms', 0) for r in param_records
                        )

                    metrics_list.append({
                        'asset': asset_name,
                        'confidence_level': conf,
                        'estimation_window': window,
                        'reps': reps,
                        **acc,
                        **tail,
                        **dist_met,
                    })

                except Exception as e:
                    warnings.warn(f"Error processing {asset_name} w={window} conf={conf} reps={reps}: {e}")

    runtime['total_runtime_ms'] = (time.time() - start_total) * 1000
    return risk_series_list, metrics_list, runtime


def evaluate_qaoa_cvar(
    config_path: Optional[Path] = None,
    config_dict: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    print("QAOA CVaR ASSET-LEVEL EVALUATION")
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

    qr = config.get('quantum_risk_settings', {})
    estimation_windows = qr.get('estimation_windows', [252, 500])
    confidence_levels = qr.get('confidence_levels', [0.95, 0.99])
    problem_encoding = config.get('problem_encoding', {})
    qaoa_settings = config.get('qaoa_settings', {})
    safety_checks = config.get('computation_strategy', {}).get('safety_checks')
    cache = None
    cache_cfg = config.get('computation_strategy', {}).get('cache', {})
    if cache_cfg.get('enabled') and cache_cfg.get('parameter_store_path'):
        from .cache import QAOAParameterCache
        cache_path = project_root / cache_cfg['parameter_store_path']
        cache = QAOAParameterCache(cache_path, cache_cfg.get('key_fields'))

    print(f"  Windows: {estimation_windows}, confidence: {confidence_levels}")

    all_risk = []
    all_metrics = []
    all_runtimes = []

    for idx, asset in enumerate(losses.columns):
        print(f"  Processing asset {idx+1}/{len(losses.columns)}: {asset}...", flush=True)
        rs, ms, rt = _process_single_asset(
            asset,
            losses[asset].dropna(),
            estimation_windows,
            confidence_levels,
            qr,
            problem_encoding,
            qaoa_settings,
            safety_checks,
            cache,
        )
        all_risk.extend(rs)
        all_metrics.extend(ms)
        all_runtimes.append(rt)

    if not all_risk:
        raise ValueError("No risk series computed.")

    risk_df = pd.DataFrame(all_risk)
    metrics_df = pd.DataFrame(all_metrics)

    runtime_dict = {}
    if all_runtimes:
        total_ms = [r['total_runtime_ms'] for r in all_runtimes]
        runtime_dict = compute_runtime_metrics(total_ms)
        for m in config.get('computation_strategy', {}).get('runtime_instrumentation', {}).get('measure', []):
            vals = [r.get(m, 0) for r in all_runtimes]
            if vals:
                runtime_dict[m] = float(np.mean(vals))
    if cache and cache_cfg.get('cache_hit_ratio_metric'):
        runtime_dict['cache_hit_ratio'] = cache.get_hit_ratio()

    print(f"\nCompleted: {len(risk_df)} risk records, {len(metrics_df)} metric records")

    time_sliced_config = config.get('evaluation', {}).get('time_sliced_metrics', {})
    time_sliced_df = pd.DataFrame()
    if time_sliced_config.get('enabled', True):
        slice_by_list = time_sliced_config.get('slice_by', ['year', 'quarter', 'month'])
        min_obs_slice = time_sliced_config.get('minimum_observations_per_slice', 60)
        all_ts = []
        for asset in risk_df['asset'].unique():
            ar = risk_df[risk_df['asset'] == asset]
            al = losses[asset].dropna()
            for conf in confidence_levels:
                for w in estimation_windows:
                    for reps in qaoa_settings.get('ansatz', {}).get('reps_grid', [1]):
                        sub = ar[
                            (ar['confidence_level'] == conf) &
                            (ar['estimation_window'] == w) &
                            (ar['reps'] == reps)
                        ]
                        if sub.empty:
                            continue
                        cvar_s = pd.Series(sub.set_index('date')['CVaR'])
                        common = al.index.intersection(cvar_s.index)
                        if len(common) < min_obs_slice:
                            continue
                        al_c = al.loc[common]
                        cvar_c = cvar_s.loc[common]
                        for sb in slice_by_list:
                            for ts in compute_time_sliced_metrics(
                                al_c, cvar_c, None, conf, sb, min_obs_slice
                            ):
                                ts.update(
                                    asset=asset,
                                    confidence_level=conf,
                                    estimation_window=w,
                                    reps=reps,
                                    slice_type=sb,
                                )
                                all_ts.append(ts)
        if all_ts:
            time_sliced_df = pd.DataFrame(all_ts)

    outputs = config.get('outputs', {})

    if 'parameter_store' in outputs:
        out_path = project_root / outputs['parameter_store']['path']
        out_path.parent.mkdir(parents=True, exist_ok=True)
        param_recs = []
        for _, r in risk_df.iterrows():
            bp = r.get('best_parameters', [])
            bp_str = str(bp) if isinstance(bp, list) else str(bp)
            param_recs.append({
                'asset': r['asset'],
                'date': r['date'],
                'estimation_window': r['estimation_window'],
                'confidence_level': r['confidence_level'],
                'reps': r.get('reps', 1),
                'num_state_qubits': problem_encoding.get('num_state_qubits', 8),
                'optimizer': qaoa_settings.get('optimizer', {}).get('type', 'cobyla'),
                'shots': qaoa_settings.get('execution', {}).get('shots', 4000),
                'best_energy': r.get('best_energy', np.nan),
                'best_parameters': bp_str,
                'optimizer_success': True,
            })
        if param_recs:
            pd.DataFrame(param_recs).drop_duplicates(
                subset=['asset', 'date', 'estimation_window', 'confidence_level', 'reps']
            ).to_parquet(out_path, index=False)
            print(f"Saved parameters: {out_path}")

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
            quantum_settings={
                **qr,
                'problem_encoding': problem_encoding,
                'qaoa_settings': qaoa_settings,
            },
            report_sections=outputs['report'].get('include_sections'),
            runtime_metrics=runtime_dict,
        )
        print(f"Saved report: {out_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    return risk_df, metrics_df, time_sliced_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='QAOA CVaR asset-level evaluation')
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    risk_df, metrics_df, ts_df = evaluate_qaoa_cvar(config_path=args.config)
    print(f"\nRisk rows: {len(risk_df)}, Metrics rows: {len(metrics_df)}, Time-sliced: {len(ts_df)}")
    print(risk_df.head())


if __name__ == "__main__":
    main()
