"""
Main execution script for QAOA Portfolio CVaR Optimization.

Single-fit protocol: Optimize once per unique asset-set, then backtest fixed weights
on the test period without any re-optimization.
"""
import ast
import json
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .returns import (
    load_panel_prices,
    compute_daily_returns,
    load_baseline_portfolios,
    extract_asset_sets_from_portfolios
)
from .precompute_registry import PrecomputeRegistry
from .portfolio_evaluator import PortfolioEvaluator
from .metrics import evaluate_portfolio_performance
from .qubo_builder import QUBOBuilder
from .report_generator import generate_report
from .terminal_logging import get_logger
from .parallel_processing import get_optimal_worker_count


def _resolve_path(path_str: str, project_root: Path) -> Path:
    """Resolve path relative to project root."""
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    full = project_root / path_str
    if full.exists():
        return full
    if p.exists():
        return p.resolve()
    return full


def _get_train_test_split(returns: pd.DataFrame, train_years: int = 3) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split returns into train and test periods."""
    dates = returns.index
    if len(dates) < train_years * 252:
        train_end_idx = max(1, len(dates) // 2)
    else:
        train_end_idx = train_years * 252
    train_end = dates[train_end_idx - 1]
    test_start_idx = train_end_idx
    if test_start_idx >= len(dates):
        test_start_idx = train_end_idx - 1
    test_start = dates[test_start_idx]
    train_dates = dates[:train_end_idx]
    test_dates = dates[test_start_idx:]
    return train_dates, test_dates


def _process_single_asset_set(args):
    """
    Worker: Run QAOA optimization for one unique asset set.
    Single-fit: use fixed train window, no date dimension.
    Uses absolute imports for multiprocessing compatibility on Windows (spawn).
    """
    (asset_set, train_returns_subset, target_k_list, return_weights, risk_weights,
     diversification_weights, confidence_levels, reps_grid, shots, maxiter, use_gpu,
     registry_root, project_root) = args

    try:
        import sys
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        from src.quantum_optimization.QAOA_for_CVaR.returns import generate_scenario_matrix
        from src.quantum_optimization.QAOA_for_CVaR.qubo_builder import QUBOBuilder
        from src.quantum_optimization.QAOA_for_CVaR.qaoa_portfolio import run_qaoa_optimization
        from src.quantum_optimization.QAOA_for_CVaR.precompute_registry import PrecomputeRegistry

        asset_list = list(asset_set)
        n = len(asset_list)
        if n < 2:
            return []

        registry = PrecomputeRegistry(registry_root=registry_root, persist_to_disk=True)
        results = []

        for target_k in target_k_list:
            if target_k > n or target_k < 2:
                continue
            for rw in return_weights:
                for rskw in risk_weights:
                    for dw in diversification_weights:
                        for alpha in confidence_levels:
                            for reps in reps_grid:
                                weight_config = (rw, rskw, dw, target_k, alpha)
                                cached = registry.get_qaoa_result_single_fit(asset_set, weight_config, reps)
                                if cached is not None:
                                    results.append({
                                        'asset_set': str(asset_set),
                                        'num_assets': n,
                                        'target_k': target_k,
                                        'return_weight': rw,
                                        'risk_weight': rskw,
                                        'diversification_weight': dw,
                                        'confidence_level': alpha,
                                        'reps': reps,
                                        'best_energy': cached.best_energy,
                                        'best_solution': cached.best_solution.tolist(),
                                        'nfev': cached.nfev,
                                        'circuit_depth': cached.circuit_depth,
                                        'circuit_width': cached.circuit_width,
                                        'shots': cached.shots,
                                        'total_time_ms': cached.total_time_ms,
                                    })
                                    continue

                                scenario_matrix = generate_scenario_matrix(
                                    train_returns_subset, min(252, len(train_returns_subset))
                                )
                                expected_returns = train_returns_subset.mean().values
                                corr = train_returns_subset.corr().values

                                qubo = QUBOBuilder(
                                    expected_returns=expected_returns,
                                    scenario_matrix=scenario_matrix,
                                    correlation_matrix=corr,
                                    return_weight=rw,
                                    risk_weight=rskw,
                                    diversification_weight=dw,
                                    target_k=target_k,
                                    confidence_level=alpha,
                                    use_gpu=use_gpu
                                )
                                Q, constant = qubo.build_qubo()

                                res = run_qaoa_optimization(
                                    Q, constant,
                                    reps=reps,
                                    shots=shots,
                                    alpha=alpha,
                                    maxiter=maxiter
                                )
                                registry.store_qaoa_result_single_fit(asset_set, weight_config, reps, res)

                                weights = res.best_solution / res.best_solution.sum() if res.best_solution.sum() > 0 else res.best_solution
                                results.append({
                                    'asset_set': str(asset_set),
                                    'num_assets': n,
                                    'target_k': target_k,
                                    'return_weight': rw,
                                    'risk_weight': rskw,
                                    'diversification_weight': dw,
                                    'confidence_level': alpha,
                                    'reps': reps,
                                    'best_energy': res.best_energy,
                                    'best_solution': res.best_solution.tolist(),
                                    'weights': weights.tolist(),
                                    'nfev': res.nfev,
                                    'circuit_depth': res.circuit_depth,
                                    'circuit_width': res.circuit_width,
                                    'shots': res.shots,
                                    'total_time_ms': res.total_time_ms,
                                })
        return results
    except Exception as e:
        return []


def run_qaoa_portfolio_optimization(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    num_portfolios_limit: Optional[int] = None
) -> Dict:
    """
    Run QAOA portfolio CVaR optimization (single-fit protocol).

    Stage A: Deduplicate baseline portfolios into unique asset sets
    Stage B: Run QAOA once per unique asset-set
    Stage C: Backtest fixed weights on test period

    Args:
        config_path: Path to llm.json
        config_dict: Optional config dict
        num_portfolios_limit: Limit baseline portfolios (None = use all, e.g. 100K)
    """
    logger = get_logger(use_progress_bars=True, clear_on_update=False, flush_immediately=True)
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    if config_dict is None:
        config_path = config_path or Path(__file__).parent / 'llm.json'
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = config_dict

    logger.section("QAOA Portfolio CVaR - Single-Fit Protocol")
    use_gpu = False
    logger.info("GPU: disabled (CPU parallelization)")

    inputs = config['inputs']
    execution_plan = config['execution_plan']
    precompute_config = config.get('precompute_registry', {})
    registry_root = precompute_config.get('registry_root', 'cache/qaoa_cvar_single_fit')
    portfolio_problem = config['portfolio_problem']
    cvar_objective = config['cvar_objective']
    qaoa_settings = config['qaoa_settings']
    outputs = config['outputs']
    single_fit = config.get('single_fit_protocol', {})
    train_years = single_fit.get('train_test_split', {}).get('default_rule_if_not_overridden', {}).get('train_years', 3)

    # Paths
    panel_path = _resolve_path(inputs['panel_price_path'], project_root)
    baseline_path = _resolve_path(inputs.get('baseline_portfolios_path', 'portfolios/portfolios.parquet'), project_root)

    # Load data
    logger.progress("Loading prices...")
    prices = load_panel_prices(str(panel_path))
    logger.info(f"  Prices: {len(prices):,} dates, {len(prices.columns):,} assets")

    returns = compute_daily_returns(prices, method=inputs.get('return_type', 'log'))
    logger.info(f"  Returns: {len(returns):,} rows")

    # Train/test split
    train_dates, test_dates = _get_train_test_split(returns, train_years)
    train_returns = returns.loc[train_dates]
    test_returns = returns.loc[test_dates]
    logger.info(f"  Train: {len(train_dates):,} days, Test: {len(test_dates):,} days")

    # Initialize registry
    registry = PrecomputeRegistry(registry_root=registry_root, persist_to_disk=precompute_config.get('persist_to_disk', True))

    # Use reduced grid for faster execution with 100K portfolios (full grid = 1000 sets × 96 configs)
    fast_mode = config.get('fast_mode', True)
    target_k_list = portfolio_problem['constraints']['cardinality'].get('target_k', [5, 10])
    return_weights = cvar_objective['objective_weights'].get('return_weight', [0.25, 0.75])
    risk_weights = cvar_objective['objective_weights'].get('risk_weight', [1.0, 2.0])
    diversification_weights = cvar_objective['objective_weights'].get('diversification_weight', [0.1, 0.5])
    confidence_levels = inputs.get('confidence_levels', [0.95, 0.99])
    reps_grid = qaoa_settings['ansatz'].get('reps_grid', [1, 2, 3])
    shots = qaoa_settings['execution'].get('shots', 5000)
    maxiter = qaoa_settings['optimizer'].get('maxiter', 250)

    if fast_mode:
        target_k_list = [5]
        return_weights, risk_weights, diversification_weights = [0.5], [1.0], [0.2]
        confidence_levels = [0.95]
        reps_grid = [1]
        logger.info("  Fast mode: reduced grid (1 target_k, 1 weight config, 1 rep)")
    else:
        # Weight sweep: limit to avoid explosion (llm says 12 configs)
        multi_obj = config.get('multi_objective_analysis', {})
        num_weight_configs = multi_obj.get('weight_sweep', {}).get('num_weight_configs', 12)
        if num_weight_configs < len(return_weights) * len(risk_weights) * len(diversification_weights):
            # Use grid sample
            import itertools
            all_combos = list(itertools.product(return_weights, risk_weights, diversification_weights))
            idx = np.linspace(0, len(all_combos) - 1, min(num_weight_configs, len(all_combos)), dtype=int)
            weight_combos = [all_combos[i] for i in idx]
            return_weights = [c[0] for c in weight_combos]
            risk_weights = [c[1] for c in weight_combos]
            diversification_weights = [c[2] for c in weight_combos]

    # Stage A: Unique asset sets
    logger.section("Stage A: Unique Asset Sets")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline portfolios not found: {baseline_path}")

    baseline = load_baseline_portfolios(str(baseline_path))
    if num_portfolios_limit:
        baseline = baseline.head(num_portfolios_limit)
    logger.info(f"  Loaded {len(baseline):,} baseline portfolios")

    portfolio_asset_sets = extract_asset_sets_from_portfolios(baseline)
    unique_asset_sets = portfolio_asset_sets['asset_set'].unique()
    unique_asset_sets = [tuple(sorted(list(s))) for s in unique_asset_sets]
    unique_asset_sets = list(dict.fromkeys(unique_asset_sets))
    # Cap unique sets for faster run (100K portfolios -> ~1000 unique sets; full run takes hours)
    max_unique_sets = config.get('max_unique_asset_sets')  # e.g. 50 for quick test
    if max_unique_sets and len(unique_asset_sets) > max_unique_sets:
        unique_asset_sets = unique_asset_sets[:int(max_unique_sets)]
        logger.info(f"  Capped to {len(unique_asset_sets):,} unique asset sets (max_unique_asset_sets={max_unique_sets})")
    logger.success(f"  Found {len(unique_asset_sets):,} unique asset sets")

    if execution_plan.get('stage_A_build_unique_asset_sets', True):
        out_dir = Path(outputs.get('unique_asset_sets', 'results/quantum_optimization/unique_asset_sets.parquet')).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        unique_df = pd.DataFrame([{'asset_set': str(s), 'num_assets': len(s)} for s in unique_asset_sets])
        unique_df.to_parquet(out_dir / 'qaoa_single_fit_unique_asset_sets.parquet', index=False)

    # Stage B: QAOA per unique asset set
    logger.section("Stage B: Single In-Sample QAOA per Asset Set")
    parallel_config = config.get('parallelization', {})
    n_jobs_raw = parallel_config.get('n_jobs', None)
    try:
        n_jobs = int(n_jobs_raw) if n_jobs_raw not in (None, 'auto', '') else None
    except (ValueError, TypeError):
        n_jobs = None
    n_jobs = get_optimal_worker_count(
        max_workers=n_jobs,
        cpu_percent=float(parallel_config.get('cpu_percent', 0.5))
    )
    logger.info(f"  Using {n_jobs} workers")

    # Prepare tasks: each task = (asset_set, train_returns_subset, ...)
    tasks = []
    for asset_set in unique_asset_sets:
        asset_list = list(asset_set)
        common = [c for c in asset_list if c in train_returns.columns]
        if len(common) < 2:
            continue
        train_sub = train_returns[common].dropna(how='all')
        if len(train_sub) < 60:
            continue
        tasks.append((
            asset_set,
            train_sub,
            target_k_list,
            return_weights,
            risk_weights,
            diversification_weights,
            confidence_levels,
            reps_grid,
            shots,
            maxiter,
            use_gpu,
            registry_root,
            str(project_root),
        ))

    logger.info(f"  Prepared {len(tasks):,} optimization tasks")

    all_results = []
    if n_jobs > 1 and len(tasks) > 1:
        with Pool(processes=n_jobs) as pool:
            pbar = logger.create_progress_bar(total=len(tasks), desc="QAOA", unit="set")
            for chunk in pool.imap(_process_single_asset_set, tasks, chunksize=max(1, len(tasks) // (n_jobs * 4))):
                all_results.extend(chunk)
                logger.update_progress_bar(pbar, n=1)
            logger.close_progress_bar(pbar)
    else:
        pbar = logger.create_progress_bar(total=len(tasks), desc="QAOA", unit="set")
        for t in tasks:
            all_results.extend(_process_single_asset_set(t))
            logger.update_progress_bar(pbar, n=1)
        logger.close_progress_bar(pbar)

    results_df = pd.DataFrame(all_results)
    logger.success(f"  Completed {len(results_df):,} optimizations")

    # Add weights if missing
    if 'weights' not in results_df.columns and 'best_solution' in results_df.columns:
        def sol_to_weights(sol):
            if isinstance(sol, str):
                sol = ast.literal_eval(sol)
            arr = np.array(sol, dtype=float)
            s = arr.sum()
            return (arr / s).tolist() if s > 0 else arr.tolist()
        results_df['weights'] = results_df['best_solution'].apply(sol_to_weights)

    # Stage C: Backtest fixed weights on test period
    logger.section("Stage C: Backtest Fixed Weights on Test Period")
    performance_results = []
    risk_free = inputs.get('risk_free_rate', 0.0001)

    eval_pbar = logger.create_progress_bar(total=len(results_df), desc="Backtest", unit="portfolio")
    for idx, row in results_df.iterrows():
        try:
            asset_set_str = row.get('asset_set', '')
            try:
                asset_names = list(ast.literal_eval(asset_set_str))
            except (ValueError, SyntaxError):
                continue
            weights = row.get('weights')
            if weights is None:
                continue
            if isinstance(weights, str):
                weights = ast.literal_eval(weights)
            weights = np.array(weights, dtype=float)
            common = [c for c in asset_names if c in test_returns.columns]
            if len(common) != len(asset_names) or len(test_returns) == 0:
                continue
            test_sub = test_returns[common]
            perf = evaluate_portfolio_performance(test_sub, weights, common, risk_free_rate=risk_free)
            perf.update({
                'asset_set': asset_set_str,
                'target_k': row.get('target_k'),
                'return_weight': row.get('return_weight'),
                'risk_weight': row.get('risk_weight'),
                'reps': row.get('reps'),
            })
            performance_results.append(perf)
        except Exception:
            pass
        logger.update_progress_bar(eval_pbar, n=1)
    logger.close_progress_bar(eval_pbar)
    performance_df = pd.DataFrame(performance_results)
    logger.success(f"  Evaluated {len(performance_df):,} portfolios on test period")

    # Save outputs
    logger.section("Saving Results")
    out_base = Path(outputs.get('qaoa_selected_portfolios', 'results/quantum_optimization/qaoa_single_fit_selected_portfolios.parquet')).parent
    out_base.mkdir(parents=True, exist_ok=True)

    results_df.to_parquet(out_base / 'qaoa_single_fit_selected_portfolios.parquet', index=False)
    if not results_df.empty and 'best_solution' in results_df.columns:
        samples_df = results_df[['asset_set', 'target_k', 'return_weight', 'risk_weight', 'best_energy', 'best_solution', 'weights']].head(5000)
        samples_df.to_parquet(out_base / 'qaoa_single_fit_samples.parquet', index=False)
    performance_df.to_parquet(out_base / 'qaoa_single_fit_performance_test.parquet', index=False)

    report_path = out_base / 'qaoa_single_fit_result_summary.md'
    generate_report({'results': results_df, 'samples': results_df.head(1000), 'performance': performance_df}, config, report_path)
    logger.success(f"  Report: {report_path}")

    stats = {
        'Unique asset sets': len(unique_asset_sets),
        'Total optimizations': len(results_df),
        'Portfolios backtested': len(performance_df)
    }
    logger.summary(stats)
    return {'results': results_df, 'performance': performance_df, 'unique_asset_sets': unique_asset_sets}


if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    num_limit = int(sys.argv[2]) if len(sys.argv) > 2 else None  # e.g. 100000
    try:
        run_qaoa_portfolio_optimization(config_path=config_path, num_portfolios_limit=num_limit)
        print("\nSuccess!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
