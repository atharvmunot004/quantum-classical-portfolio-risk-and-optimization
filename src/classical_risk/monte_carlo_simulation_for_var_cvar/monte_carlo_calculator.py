"""
Monte Carlo Simulation for VaR and CVaR calculation module.

Implements efficient, memory-safe rolling VaR and CVaR calculation using:
- Vectorized batch portfolio projection (single BLAS matmul)
- Efficient VaR/CVaR computation using np.partition (no full sorts)
- Float32 for simulations/projections, float64 for outputs
- Asset-level simulation with portfolio projection for efficiency
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Union, Optional, List, Tuple, Dict
import warnings
from sklearn.covariance import LedoitWolf


def estimate_asset_return_distribution(
    returns_window: pd.DataFrame,
    mean_model: Dict,
    covariance_model: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate asset return distribution parameters (mean and covariance).
    
    Args:
        returns_window: DataFrame of historical returns (window size x num assets)
        mean_model: Dictionary with 'enabled' and 'estimator' keys
        covariance_model: Dictionary with 'estimator' and optional 'shrinkage' keys
        
    Returns:
        Tuple of (mean_returns, covariance_matrix) as numpy arrays
    """
    returns_array = returns_window.values  # (window_size, num_assets)
    
    # Estimate mean
    if mean_model.get('enabled', True):
        if mean_model.get('estimator') == 'sample_mean':
            mean_returns = returns_array.mean(axis=0)
        else:
            mean_returns = returns_array.mean(axis=0)
    else:
        mean_returns = np.zeros(returns_array.shape[1])
    
    # Estimate covariance
    if covariance_model.get('estimator') == 'sample_covariance':
        shrinkage = covariance_model.get('shrinkage', {})
        if shrinkage.get('enabled', False) and shrinkage.get('method') == 'ledoit_wolf':
            # Use Ledoit-Wolf shrinkage estimator
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns_array).covariance_
        else:
            # Sample covariance
            cov_matrix = np.cov(returns_array.T)
        
        # Ensure covariance matrix is positive semi-definite
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        # Add small regularization to avoid singular matrix issues
        cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-8
    else:
        # Default to sample covariance
        cov_matrix = np.cov(returns_array.T)
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-8
    
    return mean_returns, cov_matrix


def simulate_asset_return_scenarios(
    mean_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    num_simulations: int = 10000,
    horizon: int = 1,
    distribution_type: str = 'multivariate_normal',
    random_seed: Optional[int] = None,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Simulate asset return scenarios for Monte Carlo simulation.
    
    Uses float32 by default to halve memory footprint.
    
    Args:
        mean_returns: Array of mean returns for each asset (num_assets,)
        covariance_matrix: Covariance matrix (num_assets x num_assets)
        num_simulations: Number of Monte Carlo simulations
        horizon: Time horizon in days (always 1 for pre-scaled covariance)
        distribution_type: Distribution assumption ('multivariate_normal')
        random_seed: Random seed for reproducibility
        dtype: Data type for scenarios (default: float32)
        
    Returns:
        Array of simulated asset returns (num_simulations, num_assets) as float32
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if distribution_type == 'multivariate_normal':
        # Check for singular matrix
        try:
            # Simulate asset returns - single period (covariance already scaled for multi-period)
            simulated_asset_returns = np.random.multivariate_normal(
                mean_returns,
                covariance_matrix,
                size=num_simulations
            ).astype(dtype)
        except np.linalg.LinAlgError:
            # If matrix is singular, use diagonal approximation
            cov_matrix = np.diag(np.diag(covariance_matrix))
            simulated_asset_returns = np.random.multivariate_normal(
                mean_returns,
                cov_matrix,
                size=num_simulations
            ).astype(dtype)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
    
    return simulated_asset_returns


def project_portfolio_returns_batch(
    asset_return_scenarios: np.ndarray,
    portfolio_weights_batch: np.ndarray
) -> np.ndarray:
    """
    Project portfolio returns for a batch of portfolios using vectorized BLAS matmul.
    
    Formula: R_batch = W_batch @ R_assets.T
    where:
    - R_assets: (num_simulations, num_assets) - asset return scenarios
    - W_batch: (batch_size, num_assets) - portfolio weights for batch
    - R_batch: (batch_size, num_simulations) - portfolio return scenarios
    
    This is a single vectorized operation, avoiding Python loops.
    
    Args:
        asset_return_scenarios: Array of simulated asset returns (num_simulations, num_assets)
        portfolio_weights_batch: Array of portfolio weights (batch_size, num_assets)
        
    Returns:
        Array of simulated portfolio returns (batch_size, num_simulations) as float32
    """
    # Single BLAS matmul: R_batch = W_batch @ R_assets.T
    # This is the vectorized operation - no loops!
    portfolio_returns_batch = np.dot(portfolio_weights_batch, asset_return_scenarios.T)
    
    return portfolio_returns_batch


def compute_var_cvar_from_simulations_efficient(
    simulated_returns: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute VaR and CVaR from simulated returns using efficient np.partition.
    
    Uses np.partition instead of full sort to avoid O(n log n) complexity.
    Only partitions to find the quantile index, then computes CVaR from tail.
    
    Args:
        simulated_returns: Array of simulated portfolio returns
            Shape: (num_portfolios, num_simulations) for batch
            Shape: (num_simulations,) for single portfolio
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        
    Returns:
        Tuple of (VaR, CVaR) arrays
        - For batch: (num_portfolios,) arrays
        - For single: scalar values
    """
    # Handle both single portfolio and batch cases
    is_batch = simulated_returns.ndim == 2
    if not is_batch:
        simulated_returns = simulated_returns.reshape(1, -1)
    
    num_portfolios, num_simulations = simulated_returns.shape
    alpha = 1 - confidence_level
    
    # Calculate quantile index
    var_index = int(np.floor(alpha * num_simulations))
    var_index = max(0, min(var_index, num_simulations - 1))
    
    # Use np.partition to find VaR without full sort
    # Partition to put the var_index-th smallest element in the correct position
    partitioned = np.partition(simulated_returns, var_index, axis=1)
    var_values = -partitioned[:, var_index]  # VaR is positive (loss amount)
    
    # Compute CVaR: mean of returns in the tail (worst alpha% of outcomes)
    # Select tail returns without full sort
    tail_indices = np.argpartition(simulated_returns, var_index, axis=1)[:, :var_index + 1]
    
    # Get tail returns efficiently
    batch_indices = np.arange(num_portfolios)[:, None]
    tail_returns = simulated_returns[batch_indices, tail_indices]
    cvar_values = -tail_returns.mean(axis=1)  # CVaR is positive (loss amount)
    
    if not is_batch:
        return var_values[0], cvar_values[0]
    
    return var_values, cvar_values


def project_portfolio_returns(
    asset_return_scenarios: np.ndarray,
    portfolio_weights: np.ndarray,
    horizon: int = 1
) -> np.ndarray:
    """
    Project portfolio returns from asset return scenarios using linear projection.
    
    Formula: R_p = W^T R_assets
    
    Args:
        asset_return_scenarios: Array of simulated asset returns
            Shape: (num_simulations, num_assets) for single-period or pre-scaled multi-period
        portfolio_weights: Array of portfolio weights (num_assets,)
        horizon: Time horizon in days (for reference, but scenarios may already be scaled)
        
    Returns:
        Array of simulated portfolio returns (num_simulations,)
    """
    # Check the shape to determine if we need to aggregate over horizon
    if asset_return_scenarios.ndim == 3:
        # Shape: (num_simulations, horizon, num_assets)
        # Sum over horizon: (num_simulations, num_assets)
        asset_returns_aggregated = asset_return_scenarios.sum(axis=1)
        portfolio_returns = np.dot(asset_returns_aggregated, portfolio_weights)
    elif asset_return_scenarios.ndim == 2:
        # Shape: (num_simulations, num_assets) - single period or pre-scaled multi-period
        portfolio_returns = np.dot(asset_return_scenarios, portfolio_weights)
    else:
        raise ValueError(f"Unexpected asset_return_scenarios shape: {asset_return_scenarios.shape}")
    
    return portfolio_returns


def simulate_portfolio_returns(
    returns_window: pd.DataFrame,
    portfolio_weights: np.ndarray,
    num_simulations: int = 10000,
    horizon: int = 1,
    distribution_type: str = 'multivariate_normal',
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate portfolio returns using Monte Carlo method.
    
    Args:
        returns_window: DataFrame of historical returns (window size x num assets)
        portfolio_weights: Array of portfolio weights
        num_simulations: Number of Monte Carlo simulations
        horizon: Time horizon in days
        distribution_type: Distribution assumption ('multivariate_normal')
        random_seed: Random seed for reproducibility
        
    Returns:
        Array of simulated portfolio returns (num_simulations,)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Align assets
    returns_array = returns_window.values  # (window_size, num_assets)
    
    if distribution_type == 'multivariate_normal':
        # Estimate mean and covariance from historical returns
        mean_returns = returns_array.mean(axis=0)
        cov_matrix = np.cov(returns_array.T)
        
        # Ensure covariance matrix is positive semi-definite
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        # Add small regularization to avoid singular matrix issues
        cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-8
        
        # Check for singular matrix
        try:
            # Simulate asset returns
            simulated_asset_returns = np.random.multivariate_normal(
                mean_returns,
                cov_matrix,
                size=(num_simulations, horizon)
            )
        except np.linalg.LinAlgError:
            # If matrix is still singular, use diagonal approximation
            cov_matrix = np.diag(np.diag(cov_matrix))
            simulated_asset_returns = np.random.multivariate_normal(
                mean_returns,
                cov_matrix,
                size=(num_simulations, horizon)
            )
        
        # If horizon > 1, sum returns over horizon
        if horizon > 1:
            simulated_asset_returns = simulated_asset_returns.sum(axis=1)
        else:
            simulated_asset_returns = simulated_asset_returns.squeeze()
        
        # Compute portfolio returns: weighted sum
        portfolio_returns = np.dot(simulated_asset_returns, portfolio_weights)
        
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
    
    return portfolio_returns


def compute_var_cvar_from_simulations(
    simulated_returns: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Compute VaR and CVaR from simulated returns (legacy single-portfolio version).
    
    Args:
        simulated_returns: Array of simulated portfolio returns
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        
    Returns:
        Tuple of (VaR, CVaR)
    """
    # Sort returns (ascending order - losses are negative)
    sorted_returns = np.sort(simulated_returns)
    
    # Calculate VaR (Value at Risk) - quantile of losses
    alpha = 1 - confidence_level
    var_index = int(np.floor(alpha * len(sorted_returns)))
    var_index = max(0, min(var_index, len(sorted_returns) - 1))
    
    var = -sorted_returns[var_index]  # VaR is positive (loss amount)
    
    # Calculate CVaR (Conditional Value at Risk) - expected loss beyond VaR
    # CVaR is the mean of returns in the tail (worst alpha% of outcomes)
    tail_returns = sorted_returns[:var_index + 1]
    cvar = -tail_returns.mean() if len(tail_returns) > 0 else var
    
    return var, cvar


def scale_horizon_covariance(
    covariance_matrix: np.ndarray,
    horizon: int,
    scaling_rule: str = 'sqrt_time'
) -> np.ndarray:
    """
    Scale covariance matrix for multi-period horizon.
    
    Args:
        covariance_matrix: Single-period covariance matrix
        horizon: Time horizon in days
        scaling_rule: Scaling rule ('sqrt_time' for sqrt(T) scaling)
        
    Returns:
        Scaled covariance matrix
    """
    if scaling_rule == 'sqrt_time':
        # Scale by sqrt(T) for returns: Var(h*R) = h * Var(R) for h-day returns
        # But for multi-period: if daily returns have cov Σ, then h-day returns have cov h*Σ
        # VaR scales as sqrt(h) * σ for normal returns
        return covariance_matrix * horizon
    else:
        return covariance_matrix


def compute_rolling_var(
    returns: pd.DataFrame,
    portfolio_weights: pd.Series,
    window: int = 252,
    confidence_level: float = 0.95,
    horizon: int = 1,
    num_simulations: int = 10000,
    distribution_type: str = 'multivariate_normal',
    random_seed: Optional[int] = None,
    min_periods: Optional[int] = None,
    mean_model: Optional[Dict] = None,
    covariance_model: Optional[Dict] = None,
    scaling_rule: str = 'sqrt_time',
    asset_scenarios: Optional[Dict] = None
) -> pd.Series:
    """
    Compute rolling VaR using Monte Carlo simulation.
    
    Supports asset-level simulation with portfolio projection, or legacy per-portfolio simulation.
    
    Args:
        returns: DataFrame of asset returns with dates as index
        portfolio_weights: Series of portfolio weights with assets as index
        window: Rolling window size in days
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        horizon: Time horizon in days
        num_simulations: Number of Monte Carlo simulations
        distribution_type: Distribution assumption
        random_seed: Random seed for reproducibility
        min_periods: Minimum number of periods required for calculation
        mean_model: Optional dict with mean estimation settings
        covariance_model: Optional dict with covariance estimation settings
        scaling_rule: Scaling rule for multi-period horizon ('sqrt_time')
        asset_scenarios: Optional dict mapping (window_start_idx, window_end_idx) to pre-computed asset scenarios
        
    Returns:
        Series of rolling VaR values with dates as index
    """
    if min_periods is None:
        min_periods = min(window, len(returns))
    
    # Set default models if not provided
    if mean_model is None:
        mean_model = {'enabled': True, 'estimator': 'sample_mean'}
    if covariance_model is None:
        covariance_model = {'estimator': 'sample_covariance', 'shrinkage': {'enabled': False}}
    
    # Align assets
    common_assets = returns.columns.intersection(portfolio_weights.index)
    if len(common_assets) == 0:
        raise ValueError("No common assets between returns and weights")
    
    returns_aligned = returns[common_assets]
    weights_aligned = portfolio_weights[common_assets].values
    weights_aligned = weights_aligned / weights_aligned.sum()  # Normalize
    
    # Compute rolling VaR
    rolling_var = pd.Series(index=returns.index, dtype=float)
    
    for i in range(len(returns_aligned)):
        if i < min_periods - 1:
            rolling_var.iloc[i] = np.nan
            continue
        
        # Get window of returns
        start_idx = max(0, i - window + 1)
        window_returns = returns_aligned.iloc[start_idx:i+1]
        
        if len(window_returns) < min_periods:
            rolling_var.iloc[i] = np.nan
            continue
        
        try:
            # Check if we have pre-computed asset scenarios
            window_key = (start_idx, i)
            if asset_scenarios is not None and window_key in asset_scenarios:
                asset_return_scenarios = asset_scenarios[window_key]
                # Project portfolio returns from asset scenarios
                simulated_returns = project_portfolio_returns(
                    asset_return_scenarios,
                    weights_aligned,
                    horizon=horizon
                )
            else:
                # Use asset-level simulation
                # Estimate distribution
                mean_returns, cov_matrix = estimate_asset_return_distribution(
                    window_returns,
                    mean_model,
                    covariance_model
                )
                
                # Scale covariance for horizon
                if horizon > 1:
                    cov_matrix = scale_horizon_covariance(cov_matrix, horizon, scaling_rule)
                
                # Simulate asset returns
                asset_return_scenarios = simulate_asset_return_scenarios(
                    mean_returns,
                    cov_matrix,
                    num_simulations=num_simulations,
                    horizon=1 if horizon > 1 else horizon,  # Already scaled covariance
                    distribution_type=distribution_type,
                    random_seed=random_seed
                )
                
                # Project portfolio returns
                simulated_returns = project_portfolio_returns(
                    asset_return_scenarios,
                    weights_aligned,
                    horizon=horizon
                )
            
            # Compute VaR from simulations
            var, _ = compute_var_cvar_from_simulations(
                simulated_returns,
                confidence_level=confidence_level
            )
            
            rolling_var.iloc[i] = var
            
        except Exception as e:
            rolling_var.iloc[i] = np.nan
    
    return rolling_var


def compute_rolling_cvar(
    returns: pd.DataFrame,
    portfolio_weights: pd.Series,
    window: int = 252,
    confidence_level: float = 0.95,
    horizon: int = 1,
    num_simulations: int = 10000,
    distribution_type: str = 'multivariate_normal',
    random_seed: Optional[int] = None,
    min_periods: Optional[int] = None,
    mean_model: Optional[Dict] = None,
    covariance_model: Optional[Dict] = None,
    scaling_rule: str = 'sqrt_time',
    asset_scenarios: Optional[Dict] = None
) -> pd.Series:
    """
    Compute rolling CVaR using Monte Carlo simulation.
    
    Supports asset-level simulation with portfolio projection, or legacy per-portfolio simulation.
    
    Args:
        returns: DataFrame of asset returns with dates as index
        portfolio_weights: Series of portfolio weights with assets as index
        window: Rolling window size in days
        confidence_level: Confidence level (e.g., 0.95 for 95% CVaR)
        horizon: Time horizon in days
        num_simulations: Number of Monte Carlo simulations
        distribution_type: Distribution assumption
        random_seed: Random seed for reproducibility
        min_periods: Minimum number of periods required for calculation
        mean_model: Optional dict with mean estimation settings
        covariance_model: Optional dict with covariance estimation settings
        scaling_rule: Scaling rule for multi-period horizon ('sqrt_time')
        asset_scenarios: Optional dict mapping (window_start_idx, window_end_idx) to pre-computed asset scenarios
        
    Returns:
        Series of rolling CVaR values with dates as index
    """
    if min_periods is None:
        min_periods = min(window, len(returns))
    
    # Set default models if not provided
    if mean_model is None:
        mean_model = {'enabled': True, 'estimator': 'sample_mean'}
    if covariance_model is None:
        covariance_model = {'estimator': 'sample_covariance', 'shrinkage': {'enabled': False}}
    
    # Align assets
    common_assets = returns.columns.intersection(portfolio_weights.index)
    if len(common_assets) == 0:
        raise ValueError("No common assets between returns and weights")
    
    returns_aligned = returns[common_assets]
    weights_aligned = portfolio_weights[common_assets].values
    weights_aligned = weights_aligned / weights_aligned.sum()  # Normalize
    
    # Compute rolling CVaR
    rolling_cvar = pd.Series(index=returns.index, dtype=float)
    
    for i in range(len(returns_aligned)):
        if i < min_periods - 1:
            rolling_cvar.iloc[i] = np.nan
            continue
        
        # Get window of returns
        start_idx = max(0, i - window + 1)
        window_returns = returns_aligned.iloc[start_idx:i+1]
        
        if len(window_returns) < min_periods:
            rolling_cvar.iloc[i] = np.nan
            continue
        
        try:
            # Check if we have pre-computed asset scenarios
            window_key = (start_idx, i)
            if asset_scenarios is not None and window_key in asset_scenarios:
                asset_return_scenarios = asset_scenarios[window_key]
                # Project portfolio returns from asset scenarios
                simulated_returns = project_portfolio_returns(
                    asset_return_scenarios,
                    weights_aligned,
                    horizon=horizon
                )
            else:
                # Use asset-level simulation
                # Estimate distribution
                mean_returns, cov_matrix = estimate_asset_return_distribution(
                    window_returns,
                    mean_model,
                    covariance_model
                )
                
                # Scale covariance for horizon
                if horizon > 1:
                    cov_matrix = scale_horizon_covariance(cov_matrix, horizon, scaling_rule)
                
                # Simulate asset returns
                asset_return_scenarios = simulate_asset_return_scenarios(
                    mean_returns,
                    cov_matrix,
                    num_simulations=num_simulations,
                    horizon=1 if horizon > 1 else horizon,  # Already scaled covariance
                    distribution_type=distribution_type,
                    random_seed=random_seed
                )
                
                # Project portfolio returns
                simulated_returns = project_portfolio_returns(
                    asset_return_scenarios,
                    weights_aligned,
                    horizon=horizon
                )
            
            # Compute CVaR from simulations
            _, cvar = compute_var_cvar_from_simulations(
                simulated_returns,
                confidence_level=confidence_level
            )
            
            rolling_cvar.iloc[i] = cvar
            
        except Exception as e:
            rolling_cvar.iloc[i] = np.nan
    
    return rolling_cvar


def align_returns_and_var(
    returns: pd.Series,
    var_series: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Align returns and VaR series to common dates.
    
    Args:
        returns: Series of returns with dates as index
        var_series: Series of VaR values with dates as index
        
    Returns:
        Tuple of (aligned_returns, aligned_var)
    """
    # Find common dates
    common_dates = returns.index.intersection(var_series.index)
    
    if len(common_dates) == 0:
        raise ValueError("No common dates between returns and VaR series")
    
    aligned_returns = returns.loc[common_dates]
    aligned_var = var_series.loc[common_dates]
    
    return aligned_returns, aligned_var


def align_returns_and_cvar(
    returns: pd.Series,
    cvar_series: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Align returns and CVaR series to common dates.
    
    Args:
        returns: Series of returns with dates as index
        cvar_series: Series of CVaR values with dates as index
        
    Returns:
        Tuple of (aligned_returns, aligned_cvar)
    """
    # Find common dates
    common_dates = returns.index.intersection(cvar_series.index)
    
    if len(common_dates) == 0:
        raise ValueError("No common dates between returns and CVaR series")
    
    aligned_returns = returns.loc[common_dates]
    aligned_cvar = cvar_series.loc[common_dates]
    
    return aligned_returns, aligned_cvar
