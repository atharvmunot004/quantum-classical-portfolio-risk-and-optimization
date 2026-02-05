"""
Main execution script for QAE Portfolio CVaR Evaluation.

Orchestrates the entire evaluation pipeline with precomputation and reuse.
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
    compute_portfolio_losses
)
from .precompute_registry import PrecomputeRegistry
from .portfolio_evaluator import PortfolioEvaluator
from .report_generator import generate_report


def create_sample_portfolios(
    asset_universe: list,
    num_portfolios: int = 100,
    min_assets: int = 3,
    max_assets: int = 8
) -> pd.DataFrame:
    """
    Create sample portfolios for testing.
    
    Args:
        asset_universe: List of asset names
        num_portfolios: Number of portfolios to create
        min_assets: Minimum assets per portfolio
        max_assets: Maximum assets per portfolio
        
    Returns:
        DataFrame of portfolio weights (N portfolios x M assets)
    """
    np.random.seed(42)
    
    portfolios = []
    
    for i in range(num_portfolios):
        # Random number of assets
        n_assets = np.random.randint(min_assets, max_assets + 1)
        
        # Random asset selection
        selected_assets = np.random.choice(asset_universe, size=n_assets, replace=False)
        
        # Random weights (normalized)
        weights = np.random.random(n_assets)
        weights = weights / weights.sum()
        
        # Create portfolio row
        portfolio_row = pd.Series(0.0, index=asset_universe)
        portfolio_row[selected_assets] = weights
        
        portfolios.append(portfolio_row)
    
    portfolio_df = pd.DataFrame(portfolios)
    portfolio_df.index = [f'portfolio_{i}' for i in range(num_portfolios)]
    
    return portfolio_df


def run_qae_portfolio_evaluation(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None
) -> Dict:
    """
    Run QAE portfolio CVaR evaluation pipeline.
    
    Args:
        config_path: Path to llm.json configuration file
        config_dict: Optional configuration dictionary (overrides config_path)
        
    Returns:
        Dictionary with evaluation results
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
    print("QAE Portfolio CVaR Evaluation")
    print("=" * 80)
    print()
    
    # Extract settings
    inputs = config['inputs']
    execution_plan = config['execution_plan']
    precompute_registry_config = config['precompute_registry']
    quantum_risk_settings = config['quantum_risk_settings']
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
    
    # Load or create portfolios
    print("\nLoading portfolios...")
    # Check for precomputed_portfolios_path first, then fallback to portfolio_weights_path
    portfolio_weights_path = inputs.get('precomputed_portfolios_path') or inputs.get('portfolio_weights_path', 'portfolios/precomputed_weights.parquet')
    portfolio_weights_path = Path(portfolio_weights_path)
    
    # Handle relative paths
    if not portfolio_weights_path.is_absolute() and not portfolio_weights_path.exists():
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        portfolio_weights_path = project_root / portfolio_weights_path
    
    if portfolio_weights_path.exists():
        portfolio_weights = pd.read_parquet(portfolio_weights_path)
        print(f"  Loaded {len(portfolio_weights):,} portfolios from {portfolio_weights_path}")
        
        # Handle different formats: wide (portfolio_id x assets) or long (portfolio_id, asset, weight)
        if 'portfolio_id' in portfolio_weights.columns and 'asset' in portfolio_weights.columns:
            # Long format - convert to wide
            print("  Converting from long format to wide format...")
            portfolio_weights = portfolio_weights.pivot(index='portfolio_id', columns='asset', values='weight')
            portfolio_weights = portfolio_weights.fillna(0.0)
            print(f"  Converted to wide format: {len(portfolio_weights):,} portfolios x {len(portfolio_weights.columns):,} assets")
        
        # Align with asset universe
        common_assets = portfolio_weights.columns.intersection(returns.columns)
        if len(common_assets) < len(portfolio_weights.columns):
            print(f"  Aligning with asset universe: {len(common_assets):,} common assets")
            portfolio_weights = portfolio_weights[common_assets]
            
            # Add missing assets with zero weights
            missing_assets = returns.columns.difference(common_assets)
            for asset in missing_assets:
                portfolio_weights[asset] = 0.0
            
            # Reorder to match returns
            portfolio_weights = portfolio_weights[returns.columns]
    else:
        # Create sample portfolios
        print(f"  Portfolio file not found at {portfolio_weights_path}, creating sample portfolios...")
        asset_universe = list(returns.columns)
        portfolio_weights = create_sample_portfolios(
            asset_universe,
            num_portfolios=100,
            min_assets=3,
            max_assets=8
        )
        
        # Save sample portfolios
        portfolio_weights_path.parent.mkdir(parents=True, exist_ok=True)
        portfolio_weights.to_parquet(portfolio_weights_path)
        print(f"  Created and saved {len(portfolio_weights):,} sample portfolios")
    
    # Initialize registry
    print("\nInitializing precompute registry...")
    registry = PrecomputeRegistry(
        registry_root=precompute_registry_config.get('registry_root', 'cache/qae_precompute'),
        persist_to_disk=precompute_registry_config.get('persist_to_disk', True)
    )
    
    # Create portfolio to asset set mapping
    print("\nCreating portfolio to asset set mapping...")
    portfolio_asset_set_map = registry.create_portfolio_to_asset_set_map(portfolio_weights)
    print(f"  Mapped {len(portfolio_asset_set_map):,} portfolios to asset sets")
    
    # Extract unique asset sets
    unique_asset_sets = [tuple(row['asset_set']) for _, row in portfolio_asset_set_map.iterrows()]
    unique_asset_sets = list(set(unique_asset_sets))
    print(f"  Found {len(unique_asset_sets):,} unique asset sets")
    
    # Initialize evaluator
    print("\nInitializing portfolio evaluator...")
    quantum_encoding = quantum_risk_settings.get('quantum_encoding', {})
    qae_settings = quantum_risk_settings.get('qae', {})
    
    evaluator = PortfolioEvaluator(
        returns=returns,
        registry=registry,
        num_state_qubits=quantum_encoding.get('num_state_qubits', 6),
        estimation_window=quantum_risk_settings.get('distribution_model', {}).get('estimation_window', 252),
        epsilon_target=qae_settings.get('epsilon_target', 0.01),
        confidence_alpha=qae_settings.get('confidence_alpha', 0.05),
        shots=qae_settings.get('shots', 2000),
        use_shrinkage=quantum_risk_settings.get('distribution_model', {}).get('shrinkage', {}).get('enabled', True)
    )
    
    # Stage A: Precompute quantum risk per asset set
    if execution_plan.get('stage_A_precompute_quantum_risk_per_asset_set', True):
        print("\n" + "=" * 80)
        print("Stage A: Precomputing Quantum Risk per Asset Set")
        print("=" * 80)
        
        confidence_levels = inputs.get('confidence_levels', [0.95, 0.99])
        
        precompute_results = evaluator.precompute_quantum_risk_per_asset_set(
            unique_asset_sets,
            confidence_levels
        )
        
        print(f"\nPrecomputation complete!")
        print(f"  Distribution params: {len(precompute_results['distribution_params'])}")
        print(f"  VaR thresholds: {len(precompute_results['var_thresholds'])}")
        print(f"  QAE tail expectations: {len(precompute_results['qae_tail_expectations'])}")
    
    # Stage C: Batch portfolio evaluation
    if execution_plan.get('stage_C_batch_portfolio_loss_evaluation', True):
        print("\n" + "=" * 80)
        print("Stage C: Batch Portfolio CVaR Evaluation")
        print("=" * 80)
        
        confidence_levels = inputs.get('confidence_levels', [0.95, 0.99])
        
        evaluation_results = evaluator.evaluate_portfolios_batch(
            portfolio_weights,
            confidence_levels
        )
        
        print(f"\nEvaluation complete!")
        print(f"  Evaluated {len(evaluation_results):,} portfolio-confidence combinations")
    
    # Save outputs
    print("\nSaving results...")
    output_base = Path(outputs['quantum_cvar_store']).parent
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Save CVaR results
    if 'evaluation_results' in locals() and not evaluation_results.empty:
        cvar_path = output_base / Path(outputs['quantum_cvar_store']).name
        evaluation_results.to_parquet(cvar_path)
        print(f"  Saved CVaR results: {cvar_path}")
    
    # Save metrics
    if 'evaluation_results' in locals() and not evaluation_results.empty:
        metrics_path = output_base / Path(outputs['metrics_table']).name
        evaluation_results.to_parquet(metrics_path)
        print(f"  Saved metrics: {metrics_path}")
    
    # Save portfolio to asset set map
    map_path = output_base / "portfolio_asset_set_map.parquet"
    portfolio_asset_set_map.to_parquet(map_path)
    print(f"  Saved portfolio-asset set map: {map_path}")
    
    # Generate and save report
    report_path = output_base / Path(outputs.get('summary_report', 'qae_result_summary.md')).name
    print(f"\nGenerating report...")
    report_content = generate_report(
        {
            'evaluation_results': evaluation_results if 'evaluation_results' in locals() else pd.DataFrame(),
            'portfolio_asset_set_map': portfolio_asset_set_map,
            'precompute_results': precompute_results if 'precompute_results' in locals() else {}
        },
        config,
        report_path
    )
    print(f"  Saved report: {report_path}")
    
    print("\n" + "=" * 80)
    print("Evaluation pipeline complete!")
    print("=" * 80)
    
    return {
        'evaluation_results': evaluation_results if 'evaluation_results' in locals() else pd.DataFrame(),
        'portfolio_asset_set_map': portfolio_asset_set_map,
        'precompute_results': precompute_results if 'precompute_results' in locals() else {}
    }


if __name__ == '__main__':
    import sys
    
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    try:
        results = run_qae_portfolio_evaluation(config_path=config_path)
        print("\nSuccess!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
