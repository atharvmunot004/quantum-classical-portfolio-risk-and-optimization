"""
Discretization utilities for mapping continuous returns to discrete bins
and vice versa for QGAN scenario generation.
"""
import numpy as np
from typing import Tuple, Dict


def create_uniform_grid(
    data: np.ndarray,
    num_bins: int = 64,
    clip_quantiles: Tuple[float, float] = (0.001, 0.999)
) -> Dict:
    """
    Create uniform discretization grid for returns.
    
    Args:
        data: Array of returns
        num_bins: Number of bins
        clip_quantiles: Quantiles for clipping outliers
        
    Returns:
        Dict with 'bin_edges', 'bin_centers', 'min_val', 'max_val'
    """
    # Clip outliers
    min_val = np.quantile(data, clip_quantiles[0])
    max_val = np.quantile(data, clip_quantiles[1])
    
    # Create uniform grid
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'min_val': min_val,
        'max_val': max_val,
        'num_bins': num_bins
    }


def discretize_returns(
    returns: np.ndarray,
    grid: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize continuous returns to bin indices.
    
    Args:
        returns: Continuous returns
        grid: Grid dict from create_uniform_grid
        
    Returns:
        bin_indices: Indices of bins (0 to num_bins-1)
        bin_centers: Corresponding bin center values
    """
    bin_edges = grid['bin_edges']
    bin_centers = grid['bin_centers']
    
    # Clip to range
    returns_clipped = np.clip(returns, grid['min_val'], grid['max_val'])
    
    # Find bin indices
    bin_indices = np.digitize(returns_clipped, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_centers) - 1)
    
    # Map to bin centers
    discretized = bin_centers[bin_indices]
    
    return bin_indices, discretized


def map_bitstrings_to_returns(
    bitstrings: np.ndarray,
    grid: Dict,
    num_qubits: int
) -> np.ndarray:
    """
    Map quantum generator bitstrings to continuous return values.
    
    Args:
        bitstrings: Integer bitstrings from quantum generator (0 to 2^num_qubits - 1)
        grid: Grid dict from create_uniform_grid
        num_qubits: Number of qubits
        
    Returns:
        Continuous return values
    """
    num_bins = grid['num_bins']
    max_bin_idx = num_bins - 1
    
    # Map bitstrings to bin indices (scale from [0, 2^num_qubits-1] to [0, num_bins-1])
    if num_bins <= 2**num_qubits:
        # Direct mapping
        bin_indices = (bitstrings * max_bin_idx) // (2**num_qubits - 1)
    else:
        # Use modulo
        bin_indices = bitstrings % num_bins
    
    bin_indices = np.clip(bin_indices, 0, max_bin_idx)
    
    # Map to bin centers
    bin_centers = grid['bin_centers']
    returns = bin_centers[bin_indices]
    
    return returns
