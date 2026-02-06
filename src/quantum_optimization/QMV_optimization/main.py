"""
Main execution script for QMV Portfolio Optimization.

Orchestrates the entire optimization pipeline with precomputation and reuse.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Union
import warnings

from .returns import (
    load_panel_prices,
    compute_daily_returns,
    load_baseline_portfolios,
    extract_asset_sets_from_portfolios
)
from .precompute_registry import PrecomputeRegistry
from .portfolio_optimizer import QMVPortfolioOptimizer
from .metrics import compute_portfolio_metrics_batch
from .report_generator import generate_report


def run_qmv_portfolio_optimization(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None
) -> Dict:
    """
    Run QMV portfolio optimization pipeline.
    
    Args:
        config_path: Path to llm.json configuration file
        config_dict: Optional configuration dictionary (overrides config_path)
        
    Returns:
        Dictionary with optimization results
    """
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
    
    print("=" * 80)
    print("QMV Portfolio Optimization")
    print("=" * 80)
    print()
    
    # Extract settings
    inputs = config['inputs']
    execution_plan = config['execution_plan']
    precompute_registry_config = config['precompute_registry']
    mean_variance_settings = config['mean_variance_settings']
    weight_encoding = config['weight_encoding']
    qubo_construction = config['qubo_construction']
    solver_settings = config['solver_settings']
    outputs = config['outputs']
    
    # Load data
    print("Loading data...")
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    
    # Resolve paths
    panel_path = Path(inputs['panel_price_path'])
    if not panel_path.is_absolute() and not panel_path.exists():
        panel_path = project_root / inputs['panel_price_path']
    
    baseline_path = Path(inputs.get('baseline_portfolios_path', 'portfolios/portfolios.parquet'))
    if not baseline_path.is_absolute() and not baseline_path.exists():
        baseline_path = project_root / inputs.get('baseline_portfolios_path', 'portfolios/portfolios.parquet')
    
    prices = load_panel_prices(str(panel_path))
    print(f"  Loaded prices: {len(prices):,} dates, {len(prices.columns):,} assets")
    
    # Load baseline portfolios
    print("\nLoading baseline portfolios...")
    baseline_portfolios = load_baseline_portfolios(str(baseline_path))
    print(f"  Loaded {len(baseline_portfolios):,} baseline portfolios")
    
    # Compute returns
    print("\nComputing returns...")
    returns = compute_daily_returns(
        prices,
        method=inputs.get('return_type', 'log')
    )
    print(f"  Computed {len(returns):,} daily returns")
    
    # Initialize registry
    print("\nInitializing precompute registry...")
    registry = PrecomputeRegistry(
        registry_root=precompute_registry_config.get('registry_root', 'cache/qmv_precompute'),
        persist_to_disk=precompute_registry_config.get('persist_to_disk', True)
    )
    
    # Extract optimization parameters
    estimation_window = mean_variance_settings.get('estimation_window', 252)
    lambda_risk_grid = mean_variance_settings['objective'].get('lambda_risk_grid', [0.1, 0.25, 0.5, 1.0, 2.0])
    bits_per_asset = weight_encoding.get('bits_per_asset', 4)
    weight_step = weight_encoding.get('weight_step', 0.0625)
    max_weight_per_asset = weight_encoding.get('max_weight_per_asset', 0.25)
    budget_penalty = qubo_construction['penalty_weights'].get('budget_penalty', 20.0)
    max_weight_penalty = qubo_construction['penalty_weights'].get('max_weight_penalty', 10.0)
    
    # Initialize optimizer
    print("\nInitializing QMV portfolio optimizer...")
    optimizer = QMVPortfolioOptimizer(
        returns=returns,
        registry=registry,
        estimation_window=estimation_window,
        bits_per_asset=bits_per_asset,
        weight_step=weight_step,
        max_weight_per_asset=max_weight_per_asset,
        budget_penalty=budget_penalty,
        max_weight_penalty=max_weight_penalty,
        num_reads=solver_settings.get('annealing_parameters', {}).get('num_reads', 5000),
        random_seed=solver_settings.get('random_seed', 42)
    )
    
    # Determine optimization dates
    rebalance_frequency = 21  # Default
    start_date = returns.index[estimation_window]
    end_date = returns.index[-1]
    
    rebalance_dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date in returns.index:
            rebalance_dates.append(current_date)
        try:
            current_idx = returns.index.get_loc(current_date)
            if current_idx + rebalance_frequency < len(returns):
                current_date = returns.index[current_idx + rebalance_frequency]
            else:
                break
        except:
            break
    
    print(f"\nOptimization dates: {len(rebalance_dates)}")
    
    # Extract unique asset sets from baseline portfolios
    print("\nExtracting unique asset sets from portfolios...")
    portfolio_asset_sets = extract_asset_sets_from_portfolios(baseline_portfolios)
    unique_asset_sets = portfolio_asset_sets['asset_set'].unique()
    unique_asset_sets = [tuple(sorted(list(s))) for s in unique_asset_sets]
    unique_asset_sets = list(dict.fromkeys(unique_asset_sets))
    print(f"  Found {len(unique_asset_sets):,} unique asset sets")
    
    # Get asset universe (all assets in returns)
    asset_universe = tuple(sorted(returns.columns.tolist()))
    
    # Stage A: Precompute per unique asset set
    if execution_plan.get('stage_A_precompute_per_unique_asset_set', True):
        print("\n" + "=" * 80)
        print("Stage A: Precomputation per Unique Asset Set")
        print("=" * 80)
        
        # Precompute for each unique asset set
        test_date = rebalance_dates[0] if rebalance_dates else returns.index[-1]
        valid_asset_sets = []
        for asset_set in unique_asset_sets:
            # Filter to assets that exist in returns
            asset_list = [a for a in asset_set if a in returns.columns]
            if len(asset_list) >= 2:  # Need at least 2 assets
                valid_asset_sets.append(tuple(sorted(asset_list)))
        
        print(f"  Precomputing statistics for {len(valid_asset_sets):,} valid asset sets...")
        for idx, asset_set_tuple in enumerate(valid_asset_sets):
            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"  Progress: {idx + 1}/{len(valid_asset_sets)} asset sets")
            optimizer.precompute_asset_set_statistics(asset_set_tuple, test_date)
        print(f"  Precomputation complete! Processed {len(valid_asset_sets):,} asset sets")
    
    # Stage B: Build QUBO and solve for each unique asset set
    if execution_plan.get('stage_B_build_qubo_and_solve', True):
        print("\n" + "=" * 80)
        print("Stage B: Build QUBO and Solve")
        print("=" * 80)
        
        all_results = []
        
        # Use first rebalance date for optimization
        opt_date = rebalance_dates[0] if rebalance_dates else returns.index[-1]
        print(f"Optimizing on date: {opt_date.date()}")
        
        # Process each unique asset set
        valid_asset_sets = []
        for asset_set in unique_asset_sets:
            # Filter to assets that exist in returns
            asset_list = [a for a in asset_set if a in returns.columns]
            if len(asset_list) >= 2:  # Need at least 2 assets
                valid_asset_sets.append(tuple(sorted(asset_list)))
        
        print(f"  Optimizing {len(valid_asset_sets):,} unique asset sets with {len(lambda_risk_grid)} lambda values each...")
        total_optimizations = len(valid_asset_sets) * len(lambda_risk_grid)
        optimization_count = 0
        
        for asset_set_idx, asset_set_tuple in enumerate(valid_asset_sets):
            if (asset_set_idx + 1) % 50 == 0 or asset_set_idx == 0:
                print(f"  Progress: {asset_set_idx + 1}/{len(valid_asset_sets)} asset sets ({optimization_count}/{total_optimizations} optimizations)")
            
            # Optimize for each lambda_risk value
            for lambda_risk in lambda_risk_grid:
                optimization_count += 1
                try:
                    result = optimizer.optimize_portfolio(
                        asset_set=asset_set_tuple,
                        date=opt_date,
                        lambda_risk=lambda_risk
                    )
                    result['asset_set'] = str(asset_set_tuple)
                    all_results.append(result)
                except Exception as e:
                    if (asset_set_idx + 1) % 50 == 0:
                        print(f"    Error optimizing {asset_set_tuple} with lambda={lambda_risk}: {e}")
                    continue
        
        results_df = pd.DataFrame(all_results)
        print(f"\nOptimization complete!")
        print(f"  Total optimizations: {len(results_df)}")
    
    # Stage C: Expand weights and batch evaluate all portfolios
    if execution_plan.get('stage_C_expand_weights_and_batch_evaluate', True) and 'results_df' in locals():
        print("\n" + "=" * 80)
        print("Stage C: Batch Evaluation for All Portfolios")
        print("=" * 80)
        
        # Create mapping from asset_set to optimized weights
        asset_set_to_weights = {}
        for _, row in results_df.iterrows():
            asset_set_str = row.get('asset_set', '')
            if asset_set_str not in asset_set_to_weights:
                asset_set_to_weights[asset_set_str] = []
            asset_set_to_weights[asset_set_str].append({
                'weights': np.array(row['weights']),
                'lambda_risk': row['lambda_risk'],
                'energy': row['energy'],
                'runtime_ms': row['runtime_ms']
            })
        
        # Use first lambda_risk for each asset set (or best energy)
        optimized_weights_by_asset_set = {}
        for asset_set_str, solutions in asset_set_to_weights.items():
            # Use solution with lowest energy (best optimization)
            best_solution = min(solutions, key=lambda x: x['energy'])
            optimized_weights_by_asset_set[asset_set_str] = best_solution
        
        # Map optimized weights to all portfolios
        print("\nMapping optimized weights to all portfolios...")
        all_portfolio_weights = []
        all_portfolio_metadata = []
        
        # Create a mapping from portfolio_id to asset_set for faster lookup
        portfolio_to_asset_set = {}
        for _, row in portfolio_asset_sets.iterrows():
            portfolio_id = row['portfolio_id']
            asset_set = row['asset_set']
            portfolio_to_asset_set[portfolio_id] = asset_set
        
        for portfolio_id, row in baseline_portfolios.iterrows():
            # Get asset set for this portfolio
            if portfolio_id not in portfolio_to_asset_set:
                continue
            
            asset_set = portfolio_to_asset_set[portfolio_id]
            asset_set_str = str(asset_set)
            
            if asset_set_str in optimized_weights_by_asset_set:
                solution = optimized_weights_by_asset_set[asset_set_str]
                weights = solution['weights']
                
                # Expand weights to full asset universe
                full_weights = np.zeros(len(asset_universe))
                asset_list = list(asset_set)
                asset_universe_list = list(asset_universe)
                for i, asset in enumerate(asset_list):
                    if asset in asset_universe_list:
                        asset_idx = asset_universe_list.index(asset)
                        if i < len(weights):
                            full_weights[asset_idx] = weights[i]
                
                all_portfolio_weights.append(full_weights)
                all_portfolio_metadata.append({
                    'portfolio_id': portfolio_id,
                    'asset_set': asset_set_str,
                    'lambda_risk': solution['lambda_risk'],
                    'energy': solution['energy'],
                    'runtime_ms': solution['runtime_ms']
                })
        
        print(f"  Mapped weights to {len(all_portfolio_weights):,} portfolios")
        
        # Batch compute metrics for all portfolios
        if all_portfolio_weights:
            print("\nComputing metrics for all portfolios...")
            weights_matrix = np.array(all_portfolio_weights)
            
            # Use out-of-sample period for evaluation
            opt_date = rebalance_dates[0] if rebalance_dates else returns.index[-1]
            try:
                date_idx = returns.index.get_loc(opt_date)
            except KeyError:
                date_idx = len(returns) - 1
            
            # Use returns after optimization date for evaluation
            if date_idx + 1 < len(returns):
                eval_returns = returns.iloc[date_idx + 1:]
            else:
                eval_returns = returns.iloc[-252:]  # Use last year if no future data
            
            if len(eval_returns) > 0:
                try:
                    # Compute metrics in batch
                    perf_metrics = compute_portfolio_metrics_batch(
                        eval_returns,
                        weights_matrix,
                        list(asset_universe),
                        risk_free_rate=inputs.get('risk_free_rate', 0.0001)
                    )
                    
                    # Combine with metadata
                    performance_df = perf_metrics.copy()
                    for i, metadata in enumerate(all_portfolio_metadata):
                        for key, value in metadata.items():
                            performance_df.loc[i, key] = value
                    
                    print(f"  Computed metrics for {len(performance_df):,} portfolios")
                except Exception as e:
                    print(f"  Error computing batch metrics: {e}")
                    import traceback
                    traceback.print_exc()
                    performance_df = pd.DataFrame()
            else:
                print("  No evaluation period available")
                performance_df = pd.DataFrame()
        else:
            performance_df = pd.DataFrame()
    
    # Save outputs
    print("\nSaving results...")
    output_base = Path(outputs['solutions']).parent
    output_base.mkdir(parents=True, exist_ok=True)
    
    if 'results_df' in locals() and not results_df.empty:
        solutions_path = output_base / Path(outputs['solutions']).name
        results_df.to_parquet(solutions_path)
        print(f"  Saved solutions: {solutions_path}")
        
        # Save weights
        weights_path = output_base / Path(outputs['portfolio_weights']).name
        weights_data = []
        for _, row in results_df.iterrows():
            for i, asset in enumerate(asset_universe):
                weights_data.append({
                    'date': row['date'],
                    'lambda_risk': row['lambda_risk'],
                    'asset': asset,
                    'weight': row['weights'][i] if i < len(row['weights']) else 0.0
                })
        weights_df = pd.DataFrame(weights_data)
        weights_df.to_parquet(weights_path)
        print(f"  Saved portfolio weights: {weights_path}")
    
    if 'performance_df' in locals() and not performance_df.empty:
        metrics_path = output_base / Path(outputs['metrics_table']).name
        performance_df.to_parquet(metrics_path)
        print(f"  Saved metrics: {metrics_path}")
    
    # Generate and save report
    report_path = output_base / Path(outputs.get('summary_report', 'qmv_result_summary.md')).name
    print(f"\nGenerating report...")
    report_content = generate_report(
        {
            'results': results_df if 'results_df' in locals() else pd.DataFrame(),
            'performance': performance_df if 'performance_df' in locals() else pd.DataFrame()
        },
        config,
        report_path
    )
    print(f"  Saved report: {report_path}")
    
    print("\n" + "=" * 80)
    print("Optimization pipeline complete!")
    print("=" * 80)
    
    return {
        'results': results_df if 'results_df' in locals() else pd.DataFrame(),
        'performance': performance_df if 'performance_df' in locals() else pd.DataFrame()
    }


if __name__ == '__main__':
    import sys
    
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    try:
        results = run_qmv_portfolio_optimization(config_path=config_path)
        print("\nSuccess!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
