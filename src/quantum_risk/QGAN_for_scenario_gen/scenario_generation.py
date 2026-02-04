"""
Scenario generation from trained QGAN models.
"""
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd

from .qgan_model import QuantumGenerator
from .discretization import map_bitstrings_to_returns


def generate_scenarios(
    generator: QuantumGenerator,
    generator_params: np.ndarray,
    grid: Dict,
    num_scenarios: int = 10000,
    horizons: Dict = None
) -> Dict:
    """
    Generate scenarios from trained QGAN.
    
    Args:
        generator: Trained QuantumGenerator
        generator_params: Trained generator parameters
        grid: Discretization grid
        num_scenarios: Number of scenarios to generate per timestamp
        horizons: Horizon configuration dict
        
    Returns:
        Dict with 'returns', 'losses', 'scenario_ids'
    """
    horizons = horizons or {'base_horizon': 1, 'scaled_horizons': [1]}
    
    base_horizon = horizons.get('base_horizon', 1)
    scaled_horizons = horizons.get('scaled_horizons', [1])
    aggregation_rule = horizons.get('aggregation_rule', 'sum_log_returns')
    
    all_scenarios = {}
    
    for horizon in scaled_horizons:
        if horizon == base_horizon:
            # Generate 1-step scenarios
            bitstrings = generator.generate_samples(generator_params, num_samples=num_scenarios)
            returns_1step = map_bitstrings_to_returns(
                bitstrings, grid, generator.num_qubits
            )
            
            if aggregation_rule == 'sum_log_returns':
                # For multi-step, sum log returns
                if horizon > 1:
                    # Generate multiple 1-step scenarios and aggregate
                    returns_multi = []
                    for _ in range(horizon):
                        bitstrings_h = generator.generate_samples(
                            generator_params, num_samples=num_scenarios
                        )
                        returns_h = map_bitstrings_to_returns(
                            bitstrings_h, grid, generator.num_qubits
                        )
                        returns_multi.append(returns_h)
                    returns = np.sum(returns_multi, axis=0)
                else:
                    returns = returns_1step
            else:
                returns = returns_1step
            
            losses = -returns  # Loss = -return
            
            all_scenarios[horizon] = {
                'returns': returns,
                'losses': losses,
                'scenario_ids': np.arange(len(returns))
            }
    
    return all_scenarios


def compute_var_cvar_from_scenarios(
    scenarios: Dict,
    confidence_levels: List[float] = [0.95, 0.99]
) -> Dict:
    """
    Compute VaR and CVaR from generated scenarios.
    
    Args:
        scenarios: Dict from generate_scenarios
        confidence_levels: List of confidence levels
        
    Returns:
        Dict with VaR and CVaR for each horizon and confidence level
    """
    results = {}
    
    for horizon, data in scenarios.items():
        losses = data['losses']
        results[horizon] = {}
        
        for conf in confidence_levels:
            alpha = 1 - conf
            var = np.quantile(losses, conf)
            tail_losses = losses[losses >= var]
            cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var
            
            results[horizon][conf] = {
                'VaR': var,
                'CVaR': cvar
            }
    
    return results
