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
    compute_daily_returns
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
    prices = load_panel_prices(inputs['panel_price_path'])
    print(f"  Loaded prices: {len(prices):,} dates, {len(prices.columns):,} assets")
    
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
    
    # Get asset universe
    asset_universe = tuple(sorted(returns.columns.tolist()))
    
    # Stage A: Precompute per unique asset set
    if execution_plan.get('stage_A_precompute_per_unique_asset_set', True):
        print("\n" + "=" * 80)
        print("Stage A: Precomputation per Unique Asset Set")
        print("=" * 80)
        
        # Precompute for asset universe
        test_date = rebalance_dates[0] if rebalance_dates else returns.index[-1]
        print(f"Precomputing statistics for asset set: {asset_universe}")
        optimizer.precompute_asset_set_statistics(asset_universe, test_date)
        print("  Precomputation complete!")
    
    # Stage B: Build QUBO and solve
    if execution_plan.get('stage_B_build_qubo_and_solve', True):
        print("\n" + "=" * 80)
        print("Stage B: Build QUBO and Solve")
        print("=" * 80)
        
        all_results = []
        
        # Limit to first few dates for testing
        test_dates = rebalance_dates[:5] if len(rebalance_dates) > 5 else rebalance_dates
        
        for date_idx, date in enumerate(test_dates):
            print(f"\nProcessing date {date_idx + 1}/{len(test_dates)}: {date.date()}")
            
            for lambda_risk in lambda_risk_grid:
                try:
                    result = optimizer.optimize_portfolio(
                        asset_set=asset_universe,
                        date=date,
                        lambda_risk=lambda_risk
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"  Error optimizing: {e}")
                    continue
        
        results_df = pd.DataFrame(all_results)
        print(f"\nOptimization complete!")
        print(f"  Total optimizations: {len(results_df)}")
    
    # Stage C: Expand weights and batch evaluate
    if execution_plan.get('stage_C_expand_weights_and_batch_evaluate', True) and 'results_df' in locals():
        print("\n" + "=" * 80)
        print("Stage C: Batch Evaluation")
        print("=" * 80)
        
        # Collect weights
        weights_list = []
        metadata_list = []
        
        for _, row in results_df.iterrows():
            weights = np.array(row['weights'])
            weights_list.append(weights)
            metadata_list.append({
                'date': row['date'],
                'lambda_risk': row['lambda_risk'],
                'energy': row['energy'],
                'runtime_ms': row['runtime_ms']
            })
        
        if weights_list:
            weights_matrix = np.array(weights_list)
            
            # Evaluate out-of-sample performance
            performance_results = []
            
            for i, (weights, metadata) in enumerate(zip(weights_list, metadata_list)):
                date = metadata['date']
                
                try:
                    date_idx = returns.index.get_loc(date)
                except KeyError:
                    continue
                
                if date_idx + 1 < len(returns):
                    oos_returns = returns.iloc[date_idx + 1:date_idx + 22]  # Next ~21 days
                    
                    if len(oos_returns) > 0:
                        try:
                            perf_metrics = compute_portfolio_metrics_batch(
                                oos_returns,
                                weights.reshape(1, -1),
                                list(asset_universe),
                                risk_free_rate=inputs.get('risk_free_rate', 0.0001)
                            )
                            
                            perf_dict = perf_metrics.iloc[0].to_dict()
                            perf_dict.update(metadata)
                            performance_results.append(perf_dict)
                        except Exception as e:
                            print(f"  Error evaluating portfolio: {e}")
                            continue
            
            performance_df = pd.DataFrame(performance_results)
            print(f"  Evaluated {len(performance_df)} portfolios")
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
