"""
Main execution script for Quantum Annealing Multi-Objective Portfolio Optimization.

Orchestrates the entire optimization pipeline.
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
    filter_asset_universe
)
from .portfolio_optimizer import PortfolioOptimizer
from .report_generator import generate_report
from .metrics import compute_time_sliced_metrics


def run_quantum_annealing_optimization(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None
) -> Dict:
    """
    Run quantum annealing portfolio optimization pipeline.
    
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
        config = config_dict
    
    print("=" * 80)
    print("Quantum Annealing Multi-Objective Portfolio Optimization")
    print("=" * 80)
    print()
    
    # Extract settings
    inputs = config['inputs']
    data_settings = config['data_settings']
    optimization_windows = config['optimization_windows']
    portfolio_objectives = config['portfolio_objectives']
    constraints = config['constraints']
    qubo_construction = config['qubo_construction']
    quantum_settings = config['quantum_annealing_settings']
    outputs = config['outputs']
    
    # Extract precomputed portfolios path
    precomputed_portfolios_path = inputs.get('precomputed_portfolios_path')
    
    # Load data
    print("Loading data...")
    prices = load_panel_prices(inputs['panel_price_path'])
    print(f"  Loaded prices: {len(prices):,} dates, {len(prices.columns):,} assets")
    
    # Filter asset universe
    asset_universe_config = inputs.get('asset_universe', {})
    if asset_universe_config.get('mode') == 'filtered':
        max_assets = asset_universe_config.get('max_assets', 40)
        print(f"\nFiltering asset universe (max {max_assets} assets)...")
        # We'll filter after computing returns
    else:
        max_assets = None
    
    # Compute returns
    print("\nComputing returns...")
    returns = compute_daily_returns(
        prices,
        method=data_settings.get('return_type', 'log')
    )
    print(f"  Computed {len(returns):,} daily returns")
    
    # Handle missing data
    missing_policy = data_settings.get('missing_data_policy', {})
    if missing_policy.get('dropna') == 'panel_intersection':
        returns = returns.dropna()
        print(f"  After panel intersection: {len(returns):,} dates, {len(returns.columns):,} assets")
    
    # Filter asset universe
    if max_assets is not None:
        min_obs = missing_policy.get('min_required_observations', 1000)
        returns = filter_asset_universe(
            returns,
            max_assets=max_assets,
            selection_method=asset_universe_config.get('selection_method', 'liquidity_and_data_availability'),
            min_required_observations=min_obs
        )
        print(f"  After filtering: {len(returns):,} dates, {len(returns.columns):,} assets")
    
    # Extract optimization parameters
    estimation_windows = optimization_windows['estimation_windows']
    rebalance_frequency = optimization_windows['rebalance_frequency']
    
    # Extract weight ranges
    obj_weights = qubo_construction['objective_weights']
    return_weights = obj_weights.get('return_weight', [0.5, 1.0])
    risk_weights = obj_weights.get('risk_weight', [1.0, 2.0])
    diversification_weights = obj_weights.get('diversification_weight', [0.1, 0.5])
    
    # Extract penalty weights
    penalty_weights = qubo_construction['penalty_weights']
    budget_penalty = penalty_weights.get('budget_penalty', 10.0)
    cardinality_penalty = penalty_weights.get('cardinality_penalty', 8.0)
    
    # Extract constraints
    min_assets = constraints['cardinality']['min_assets']
    max_assets_constraint = constraints['cardinality']['max_assets']
    
    # Extract risk settings
    confidence_levels = portfolio_objectives['risk']['confidence_levels']
    confidence_level = confidence_levels[0] if confidence_levels else 0.95
    
    # Extract quantum settings
    annealing_params = quantum_settings.get('annealing_parameters', {})
    quantum_config = {
        'backend': quantum_settings.get('backend', 'dwave_or_simulated_annealer'),
        'num_reads': annealing_params.get('num_reads', 5000),
        'annealing_time_us': annealing_params.get('annealing_time_us', [20, 50])[0],
        'chain_strength': annealing_params.get('chain_strength', 'auto'),
        'auto_scale': annealing_params.get('auto_scale', True),
        'optimize_embedding': quantum_settings.get('embedding', {}).get('optimize_embedding', True),
        'embedding_retries': quantum_settings.get('embedding', {}).get('embedding_retries', 5),
        'random_seed': quantum_settings.get('random_seed', 42)
    }
    
    # Initialize optimizer
    print("\nInitializing portfolio optimizer...")
    optimizer = PortfolioOptimizer(
        returns=returns,
        estimation_windows=estimation_windows,
        rebalance_frequency=rebalance_frequency,
        return_weights=return_weights if isinstance(return_weights, list) else [return_weights],
        risk_weights=risk_weights if isinstance(risk_weights, list) else [risk_weights],
        diversification_weights=diversification_weights if isinstance(diversification_weights, list) else [diversification_weights],
        budget_penalty=budget_penalty,
        cardinality_penalty=cardinality_penalty,
        min_assets=min_assets,
        max_assets=max_assets_constraint,
        confidence_level=confidence_level,
        quantum_settings=quantum_config,
        warmup_policy=optimization_windows.get('warmup_policy', 'skip_until_window_full'),
        precomputed_portfolios_path=precomputed_portfolios_path,
        num_top_portfolios=quantum_settings.get('annealing_parameters', {}).get('num_reads', 5000)  # Use num_reads as top N
    )
    
    # Run optimization
    print("\nRunning optimization...")
    results = optimizer.optimize()
    
    print(f"\nOptimization complete!")
    print(f"  Total optimizations: {results['num_optimizations']}")
    print(f"  Portfolio weight records: {len(results['portfolio_weights'])}")
    print(f"  Performance records: {len(results['portfolio_performance'])}")
    
    # Save outputs
    print("\nSaving results...")
    output_base = Path(outputs['portfolio_weights_store']['path']).parent
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Save portfolio weights
    weights_path = output_base / Path(outputs['portfolio_weights_store']['path']).name
    if not results['portfolio_weights'].empty:
        results['portfolio_weights'].to_parquet(weights_path)
        print(f"  Saved portfolio weights: {weights_path}")
    
    # Save portfolio performance
    perf_path = output_base / Path(outputs['portfolio_performance_store']['path']).name
    if not results['portfolio_performance'].empty:
        results['portfolio_performance'].to_parquet(perf_path)
        print(f"  Saved portfolio performance: {perf_path}")
    
    # Save metrics
    metrics_path = output_base / Path(outputs['metrics_table']['path']).name
    if not results['optimization_metrics'].empty:
        results['optimization_metrics'].to_parquet(metrics_path)
        print(f"  Saved metrics: {metrics_path}")
    
    # Generate and save report
    report_path = output_base / Path(outputs['report']['path']).name
    print(f"\nGenerating report...")
    report_content = generate_report(results, config, report_path)
    print(f"  Saved report: {report_path}")
    
    print("\n" + "=" * 80)
    print("Optimization pipeline complete!")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    import sys
    
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    try:
        results = run_quantum_annealing_optimization(config_path=config_path)
        print("\nSuccess!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
