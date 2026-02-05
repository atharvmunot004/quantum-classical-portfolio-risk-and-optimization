"""
Main execution script for QAOA Portfolio CVaR Optimization.

Orchestrates the entire optimization pipeline with precomputation and reuse.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

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


def _process_single_optimization(args):
    """
    Worker function for parallel processing of a single optimization task.
    
    Args:
        args: Tuple of (asset_universe, date, target_k, return_weight, risk_weight,
                       diversification_weight, confidence_level, reps, returns_path,
                       registry_root, estimation_window, shots, maxiter, use_gpu,
                       use_baseline_portfolios)
    
    Returns:
        Tuple of (result_dict, sample_dicts) or None on error
    """
    (asset_universe, date, target_k, return_weight, risk_weight,
     diversification_weight, confidence_level, reps, returns_path,
     registry_root, estimation_window, shots, maxiter, use_gpu,
     use_baseline_portfolios) = args
    
    try:
        # Reload returns in worker process (shared file-based)
        from .returns import load_panel_prices, compute_daily_returns
        from .precompute_registry import PrecomputeRegistry
        from .portfolio_evaluator import PortfolioEvaluator
        
        prices = load_panel_prices(returns_path)
        returns = compute_daily_returns(prices, method='log')
        
        # Create local evaluator
        registry = PrecomputeRegistry(registry_root=registry_root, persist_to_disk=True)
        evaluator = PortfolioEvaluator(
            returns=returns,
            registry=registry,
            estimation_window=estimation_window,
            target_k=[target_k],
            return_weights=[return_weight],
            risk_weights=[risk_weight],
            diversification_weights=[diversification_weight],
            confidence_levels=[confidence_level],
            reps_grid=[reps],
            shots=shots,
            maxiter=maxiter,
            use_gpu=use_gpu
        )
        
        # Run optimization
        result = evaluator.optimize_portfolio(
            asset_set=asset_universe,
            date=date,
            return_weight=return_weight,
            risk_weight=risk_weight,
            diversification_weight=diversification_weight,
            target_k=target_k,
            confidence_level=confidence_level,
            reps=reps
        )
        
        # Convert solution to weights
        weights = result.best_solution / result.best_solution.sum() if result.best_solution.sum() > 0 else result.best_solution
        
        # Build result dict
        result_dict = {
            'date': date,
            'target_k': target_k,
            'return_weight': return_weight,
            'risk_weight': risk_weight,
            'diversification_weight': diversification_weight,
            'confidence_level': confidence_level,
            'reps': reps,
            'best_energy': result.best_energy,
            'best_solution': result.best_solution.tolist(),
            'weights': weights.tolist(),
            'nfev': result.nfev,
            'circuit_depth': result.circuit_depth,
            'circuit_width': result.circuit_width,
            'shots': result.shots,
            'total_time_ms': result.total_time_ms
        }
        
        if use_baseline_portfolios:
            result_dict['asset_set'] = str(asset_universe)
            result_dict['num_assets'] = len(asset_universe)
        
        # Build sample dicts
        sample_dicts = []
        for sample_solution, sample_energy in result.samples[:10]:
            sample_weights = sample_solution / sample_solution.sum() if sample_solution.sum() > 0 else sample_solution
            sample_dict = {
                'date': date,
                'target_k': target_k,
                'return_weight': return_weight,
                'risk_weight': risk_weight,
                'diversification_weight': diversification_weight,
                'confidence_level': confidence_level,
                'reps': reps,
                'energy': sample_energy,
                'solution': sample_solution.tolist(),
                'weights': sample_weights.tolist()
            }
            if use_baseline_portfolios:
                sample_dict['asset_set'] = str(asset_universe)
                sample_dict['num_assets'] = len(asset_universe)
            sample_dicts.append(sample_dict)
        
        return (result_dict, sample_dicts)
    except Exception as e:
        return None


def run_qaoa_portfolio_optimization(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None
) -> Dict:
    """
    Run QAOA portfolio CVaR optimization pipeline.
    
    Args:
        config_path: Path to llm.json configuration file
        config_dict: Optional configuration dictionary (overrides config_path)
        
    Returns:
        Dictionary with optimization results
    """
    # Initialize logger
    logger = get_logger(
        use_progress_bars=True,
        clear_on_update=False,  # Don't clear on every update, only on major sections
        flush_immediately=True
    )
    
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
        config = config
    
    logger.section("QAOA Portfolio CVaR Optimization")
    
    # Force disable GPU - using CPU only with multi-core parallelization
    use_gpu = False
    logger.info("GPU acceleration: DISABLED (using multi-core CPU parallelization)")
    
    # Get CPU configuration for parallel processing
    parallel_config = config.get('parallelization', {})
    cpu_percent = parallel_config.get('cpu_percent', 0.8)  # Use 80% of cores by default
    n_jobs = parallel_config.get('n_jobs', None)  # None = auto
    
    if n_jobs is None:
        n_jobs = get_optimal_worker_count(cpu_percent=cpu_percent)
    
    total_cores = cpu_count()
    logger.info(f"CPU parallelization: Using {n_jobs} workers out of {total_cores} cores ({n_jobs/total_cores*100:.1f}%)")
    
    # Extract settings
    inputs = config['inputs']
    execution_plan = config['execution_plan']
    precompute_registry_config = config['precompute_registry']
    portfolio_problem = config['portfolio_problem']
    cvar_objective = config['cvar_objective']
    qubo_construction = config['qubo_construction']
    qaoa_settings = config['qaoa_settings']
    outputs = config['outputs']
    
    # Load data
    logger.progress("Loading data...")
    prices = load_panel_prices(inputs['panel_price_path'])
    logger.info(f"  Loaded prices: {len(prices):,} dates, {len(prices.columns):,} assets")
    
    # Compute returns
    logger.progress("Computing returns...")
    returns = compute_daily_returns(
        prices,
        method=inputs.get('return_type', 'log')
    )
    logger.info(f"  Computed {len(returns):,} daily returns")
    
    # Initialize registry
    logger.progress("Initializing precompute registry...")
    registry = PrecomputeRegistry(
        registry_root=precompute_registry_config.get('registry_root', 'cache/qaoa_cvar_precompute'),
        persist_to_disk=precompute_registry_config.get('persist_to_disk', True)
    )
    logger.success("Registry initialized")
    
    # Extract optimization parameters
    target_k_list = portfolio_problem['constraints']['cardinality'].get('target_k', [5, 10])
    return_weights = cvar_objective['objective_weights'].get('return_weight', [0.25, 0.75])
    risk_weights = cvar_objective['objective_weights'].get('risk_weight', [1.0, 2.0])
    diversification_weights = cvar_objective['objective_weights'].get('diversification_weight', [0.1, 0.5])
    confidence_levels = inputs.get('confidence_levels', [0.95, 0.99])
    reps_grid = qaoa_settings['ansatz'].get('reps_grid', [1, 2, 3])
    
    # Initialize evaluator
    logger.progress("Initializing portfolio evaluator...")
    evaluator = PortfolioEvaluator(
        returns=returns,
        registry=registry,
        estimation_window=252,
        target_k=target_k_list,
        return_weights=return_weights,
        risk_weights=risk_weights,
        diversification_weights=diversification_weights,
        confidence_levels=confidence_levels,
        reps_grid=reps_grid,
        shots=qaoa_settings['execution'].get('shots', 5000),
        maxiter=qaoa_settings['optimizer'].get('maxiter', 250),
        use_gpu=use_gpu
    )
    logger.success("Portfolio evaluator initialized")
    
    # Determine optimization dates (use rebalancing frequency)
    rebalance_frequency = 21  # Default
    max_window = 252
    start_date = returns.index[max_window]
    end_date = returns.index[-1]
    
    rebalance_dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date in returns.index:
            rebalance_dates.append(current_date)
        current_idx = returns.index.get_loc(current_date)
        if current_idx + rebalance_frequency < len(returns):
            current_date = returns.index[current_idx + rebalance_frequency]
        else:
            break
    
    logger.info(f"Optimization dates: {len(rebalance_dates)}")
    
    # Load 100K portfolios if specified
    baseline_portfolios = None
    portfolio_asset_sets = None
    use_baseline_portfolios = False
    
    if inputs.get('baseline_portfolios_path'):
        try:
            logger.progress("Loading baseline portfolios...")
            baseline_portfolios = load_baseline_portfolios(inputs['baseline_portfolios_path'])
            logger.success(f"Loaded {len(baseline_portfolios):,} baseline portfolios")
            
            # Extract asset sets from portfolios
            logger.progress("Extracting asset sets from portfolios...")
            portfolio_asset_sets = extract_asset_sets_from_portfolios(baseline_portfolios)
            
            # Get unique asset sets
            unique_asset_sets = portfolio_asset_sets['asset_set'].unique()
            logger.info(f"Found {len(unique_asset_sets):,} unique asset sets")
            logger.info(f"Asset set sizes: {portfolio_asset_sets['num_assets'].value_counts().to_dict()}")
            
            use_baseline_portfolios = True
        except FileNotFoundError as e:
            logger.warning(f"Baseline portfolios file not found: {e}")
            logger.info("Continuing with full asset universe optimization")
    
    # Get asset universe
    if use_baseline_portfolios and portfolio_asset_sets is not None:
        # Use asset sets from portfolios
        asset_universes = [tuple(sorted(list(asset_set))) for asset_set in unique_asset_sets]
        logger.info(f"Will optimize {len(asset_universes):,} unique asset sets from {len(baseline_portfolios):,} portfolios")
    else:
        # Use full asset universe
        asset_universes = [tuple(sorted(returns.columns.tolist()))]
        logger.info("Using full asset universe")
    
    # Calculate total number of optimizations
    if use_baseline_portfolios:
        total_optimizations = (
            len(asset_universes) *
            len(rebalance_dates) *
            len(target_k_list) *
            len(return_weights) *
            len(risk_weights) *
            len(diversification_weights) *
            len(confidence_levels) *
            len(reps_grid)
        )
    else:
        total_optimizations = (
            len(rebalance_dates) *
            len(target_k_list) *
            len(return_weights) *
            len(risk_weights) *
            len(diversification_weights) *
            len(confidence_levels) *
            len(reps_grid)
        )
    logger.info(f"Total optimizations to perform: {total_optimizations:,}")
    
    # Stage A & B: Precompute and optimize
    if execution_plan.get('stage_A_precompute_per_unique_asset_set_and_date', True) and \
       execution_plan.get('stage_B_qaoa_optimize_and_sample_candidates', True):
        logger.section("Stage A & B: Precomputation and QAOA Optimization")
        
        all_results = []
        all_samples = []
        
        # Prepare all optimization tasks
        logger.progress("Preparing optimization tasks...")
        tasks = []
        for asset_universe in asset_universes:
            for date in rebalance_dates:
                for target_k in target_k_list:
                    if use_baseline_portfolios and target_k > len(asset_universe):
                        continue
                    for return_weight in return_weights:
                        for risk_weight in risk_weights:
                            for diversification_weight in diversification_weights:
                                for confidence_level in confidence_levels:
                                    for reps in reps_grid:
                                        tasks.append((
                                            asset_universe, date, target_k, return_weight,
                                            risk_weight, diversification_weight, confidence_level, reps,
                                            inputs['panel_price_path'],  # returns_path
                                            precompute_registry_config.get('registry_root', 'cache/qaoa_cvar_precompute'),
                                            252,  # estimation_window
                                            qaoa_settings['execution'].get('shots', 5000),
                                            qaoa_settings['optimizer'].get('maxiter', 250),
                                            use_gpu,
                                            use_baseline_portfolios
                                        ))
        
        logger.info(f"Prepared {len(tasks):,} optimization tasks")
        
        # Create progress bar for overall progress
        overall_pbar = logger.create_progress_bar(
            total=len(tasks),
            desc="Overall Progress",
            unit="opt"
        )
        
        # Process tasks in parallel
        if n_jobs > 1 and len(tasks) > 1:
            logger.progress(f"Processing {len(tasks):,} optimizations in parallel using {n_jobs} workers...")
            try:
                with Pool(processes=n_jobs) as pool:
                    results_iter = pool.imap(_process_single_optimization, tasks, chunksize=max(1, len(tasks) // (n_jobs * 4)))
                    opt_count = 0
                    for result_data in results_iter:
                        if result_data is not None:
                            result_dict, sample_dicts = result_data
                            all_results.append(result_dict)
                            all_samples.extend(sample_dicts)
                        opt_count += 1
                        logger.update_progress_bar(
                            overall_pbar,
                            n=1,
                            desc=f"Completed {opt_count}/{len(tasks)} optimizations"
                        )
            except Exception as e:
                logger.warning(f"Parallel processing failed: {e}. Falling back to sequential.")
                # Fallback to sequential
                opt_count = 0
                for task in tasks:
                    result_data = _process_single_optimization(task)
                    if result_data is not None:
                        result_dict, sample_dicts = result_data
                        all_results.append(result_dict)
                        all_samples.extend(sample_dicts)
                    opt_count += 1
                    logger.update_progress_bar(overall_pbar, n=1)
        else:
            # Sequential processing
            logger.progress(f"Processing {len(tasks):,} optimizations sequentially...")
            opt_count = 0
            for task in tasks:
                result_data = _process_single_optimization(task)
                if result_data is not None:
                    result_dict, sample_dicts = result_data
                    all_results.append(result_dict)
                    all_samples.extend(sample_dicts)
                opt_count += 1
                logger.update_progress_bar(overall_pbar, n=1)
        
        logger.close_progress_bar(overall_pbar)
        
        results_df = pd.DataFrame(all_results)
        samples_df = pd.DataFrame(all_samples)
        
        logger.success(f"Optimization complete!")
        logger.info(f"  Total optimizations: {len(results_df):,}")
        logger.info(f"  Total samples: {len(samples_df):,}")
        
        results_df = pd.DataFrame(all_results)
        samples_df = pd.DataFrame(all_samples)
        
        print(f"\nOptimization complete!")
        print(f"  Total optimizations: {len(results_df)}")
        print(f"  Total samples: {len(samples_df)}")
    
    # Stage C: Batch evaluation
    if execution_plan.get('stage_C_batch_evaluate_candidates_and_select', True) and 'results_df' in locals():
        logger.section("Stage C: Batch Evaluation")
        
        performance_results = []
        
        eval_pbar = logger.create_progress_bar(
            total=len(results_df),
            desc="Evaluating Portfolios",
            unit="portfolio"
        )
        
        for idx, row in results_df.iterrows():
            try:
                date = row['date']
                
                # Get solution and convert to weights
                solution_list = row.get('best_solution', None)
                if solution_list is None:
                    continue
                
                # Handle different storage formats
                if isinstance(solution_list, str):
                    import ast
                    try:
                        solution_list = ast.literal_eval(solution_list)
                    except:
                        continue
                elif hasattr(solution_list, '__len__') and not isinstance(solution_list, str):
                    pass  # Already a list/array
                else:
                    continue  # Skip if not a valid sequence
                
                solution = np.array(solution_list, dtype=float)
                if solution.ndim != 1 or len(solution) != len(asset_universe):
                    continue
                
                # Convert to weights (equal weight on selected assets)
                if solution.sum() > 0:
                    weights = solution / solution.sum()
                else:
                    continue  # Skip if no assets selected
                
                asset_names = list(asset_universe)
            except Exception as e:
                continue  # Skip this row if there's any error
            
            # Get out-of-sample returns (next period)
            try:
                date_idx = returns.index.get_loc(date)
            except KeyError:
                continue
            
            if date_idx + 1 < len(returns):
                oos_returns = returns.iloc[date_idx + 1:date_idx + 22]  # Next ~21 days
                
                if len(oos_returns) == 0:
                    continue
                
                try:
                    perf_metrics = evaluate_portfolio_performance(
                        oos_returns,
                        weights,
                        asset_names,
                        risk_free_rate=inputs.get('risk_free_rate', 0.0001)
                    )
                    
                    perf_metrics.update({
                        'date': date,
                        'target_k': row['target_k'],
                        'return_weight': row['return_weight'],
                        'risk_weight': row['risk_weight'],
                        'diversification_weight': row['diversification_weight'],
                        'confidence_level': row['confidence_level'],
                        'reps': row['reps']
                    })
                    
                    performance_results.append(perf_metrics)
                except Exception as e:
                    logger.warning(f"  Error evaluating portfolio: {e}")
                    continue
            
            logger.update_progress_bar(eval_pbar, n=1)
        
        logger.close_progress_bar(eval_pbar)
        
        performance_df = pd.DataFrame(performance_results)
        logger.success(f"Evaluated {len(performance_df):,} portfolios")
    
    # Save outputs
    logger.section("Saving Results")
    output_base = Path(outputs['qaoa_selected_portfolios']).parent
    output_base.mkdir(parents=True, exist_ok=True)
    
    if 'results_df' in locals() and not results_df.empty:
        results_path = output_base / Path(outputs['qaoa_selected_portfolios']).name
        results_df.to_parquet(results_path)
        logger.success(f"Saved selected portfolios: {results_path}")
    
    if 'samples_df' in locals() and not samples_df.empty:
        samples_path = output_base / Path(outputs['qaoa_samples']).name
        samples_df.to_parquet(samples_path)
        logger.success(f"Saved samples: {samples_path}")
    
    if 'performance_df' in locals() and not performance_df.empty:
        perf_path = output_base / Path(outputs['portfolio_performance']).name
        performance_df.to_parquet(perf_path)
        logger.success(f"Saved performance: {perf_path}")
    
    # Generate and save report
    report_path = output_base / Path(outputs.get('summary_report', 'qaoa_result_summary.md')).name
    logger.progress("Generating report...")
    report_content = generate_report(
        {
            'results': results_df if 'results_df' in locals() else pd.DataFrame(),
            'samples': samples_df if 'samples_df' in locals() else pd.DataFrame(),
            'performance': performance_df if 'performance_df' in locals() else pd.DataFrame()
        },
        config,
        report_path
    )
    logger.success(f"Saved report: {report_path}")
    
    # Print summary
    stats = {
        'Total optimizations': len(results_df) if 'results_df' in locals() else 0,
        'Total samples': len(samples_df) if 'samples_df' in locals() else 0,
        'Portfolios evaluated': len(performance_df) if 'performance_df' in locals() else 0
    }
    logger.summary(stats)
    
    logger.section("Optimization Pipeline Complete!")
    
    return {
        'results': results_df if 'results_df' in locals() else pd.DataFrame(),
        'samples': samples_df if 'samples_df' in locals() else pd.DataFrame(),
        'performance': performance_df if 'performance_df' in locals() else pd.DataFrame()
    }


if __name__ == '__main__':
    import sys
    
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    try:
        results = run_qaoa_portfolio_optimization(config_path=config_path)
        print("\nSuccess!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
