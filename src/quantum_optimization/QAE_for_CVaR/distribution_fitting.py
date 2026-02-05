"""
Distribution Fitting Module for QAE Portfolio CVaR.

Fits multivariate normal distributions to asset returns for portfolio loss modeling.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.covariance import LedoitWolf


@dataclass
class DistributionParams:
    """Parameters of multivariate normal distribution."""
    mean: np.ndarray
    cov: np.ndarray
    asset_set: tuple
    estimation_window: int


def fit_multivariate_normal(
    returns: pd.DataFrame,
    asset_set: tuple,
    estimation_window: int,
    use_shrinkage: bool = True,
    shrinkage_method: str = 'ledoit_wolf'
) -> DistributionParams:
    """
    Fit multivariate normal distribution to asset returns.
    
    Args:
        returns: DataFrame of returns (T x N)
        asset_set: Tuple of asset names in the set
        estimation_window: Window size for estimation
        use_shrinkage: Whether to use shrinkage estimator
        shrinkage_method: Shrinkage method ('ledoit_wolf')
        
    Returns:
        DistributionParams with mean and covariance
    """
    # Select assets
    asset_list = list(asset_set)
    returns_subset = returns[asset_list]
    
    # Use last estimation_window observations
    if len(returns_subset) > estimation_window:
        returns_window = returns_subset.iloc[-estimation_window:]
    else:
        returns_window = returns_subset
    
    # Remove any NaN rows
    returns_window = returns_window.dropna()
    
    if len(returns_window) < 10:
        raise ValueError(f"Insufficient data for asset set {asset_set}: {len(returns_window)} observations")
    
    # Compute mean
    mean = returns_window.mean().values
    
    # Compute covariance
    if use_shrinkage and shrinkage_method == 'ledoit_wolf':
        lw = LedoitWolf()
        lw.fit(returns_window)
        cov = lw.covariance_
    else:
        cov = returns_window.cov().values
    
    return DistributionParams(
        mean=mean,
        cov=cov,
        asset_set=asset_set,
        estimation_window=estimation_window
    )


def discretize_portfolio_loss_distribution(
    dist_params: DistributionParams,
    weights: np.ndarray,
    num_state_qubits: int = 6,
    support_clipping_quantiles: Tuple[float, float] = (0.001, 0.999),
    rescale_to_unit_interval: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize portfolio loss distribution for quantum encoding.
    
    Args:
        dist_params: Distribution parameters
        weights: Portfolio weights (aligned with asset_set)
        num_state_qubits: Number of qubits for state encoding
        support_clipping_quantiles: Quantiles for clipping support
        rescale_to_unit_interval: Whether to rescale to [0, 1]
        
    Returns:
        Tuple of (probabilities, bin_centers)
    """
    # Portfolio loss mean: -mu @ w
    portfolio_loss_mean = -dist_params.mean @ weights
    
    # Portfolio loss variance: w^T @ Sigma @ w
    portfolio_loss_var = weights.T @ dist_params.cov @ weights
    
    # Portfolio loss std
    portfolio_loss_std = np.sqrt(portfolio_loss_var)
    
    # Clip support using quantiles
    from scipy import stats
    q_low, q_high = support_clipping_quantiles
    
    # Approximate quantiles using normal distribution
    loss_low = stats.norm.ppf(q_low, loc=portfolio_loss_mean, scale=portfolio_loss_std)
    loss_high = stats.norm.ppf(q_high, loc=portfolio_loss_mean, scale=portfolio_loss_std)
    
    # Discretize into 2^n bins
    n_bins = 2 ** num_state_qubits
    bin_edges = np.linspace(loss_low, loss_high, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute probabilities for each bin
    probs = np.diff(stats.norm.cdf(
        bin_edges,
        loc=portfolio_loss_mean,
        scale=portfolio_loss_std
    ))
    
    # Normalize (should already be normalized, but ensure)
    probs = probs / probs.sum()
    
    # Rescale bin centers to [0, 1] if requested
    if rescale_to_unit_interval:
        bin_centers_scaled = (bin_centers - loss_low) / (loss_high - loss_low)
        return probs, bin_centers_scaled
    else:
        return probs, bin_centers
