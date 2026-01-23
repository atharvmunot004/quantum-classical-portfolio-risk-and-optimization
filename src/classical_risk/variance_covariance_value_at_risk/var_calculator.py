"""
Variance-Covariance Value-at-Risk calculation module.

Implements rolling VaR calculation using the parametric Variance-Covariance method
with normal distribution assumption:
VaR = μ - z_α * σ * √(horizon)

Where:
- μ = mean return (sample mean)
- σ = standard deviation (sample std)
- z_α = z-score for confidence level α
- horizon = time horizon in days
"""
import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple, Dict, Any
from scipy import stats


def compute_variance_covariance_var(
    mean_return: float,
    volatility: float,
    confidence_level: float = 0.95,
    horizon: int = 1,
    scaling_rule: str = 'sqrt_time',
    tail_side: str = 'left'
) -> float:
    """
    Calculate Variance-Covariance Value-at-Risk using normal distribution assumption.
    
    Standard formula for left-tail risk:
    VaR = -μ + z_{1-α} * σ * √(horizon)
    
    Where:
    - μ = mean return
    - σ = standard deviation of returns
    - z_{1-α} = (1-α) quantile of standard normal distribution (positive for α < 0.5)
    - α = 1 - confidence_level (e.g., α = 0.05 for 95% VaR)
    - horizon = time horizon in days
    
    For 95% VaR: z_{0.95} ≈ 1.645, so VaR = -μ + 1.645 * σ * √(h)
    
    Args:
        mean_return: Mean return (μ)
        volatility: Standard deviation of returns (σ)
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        horizon: Time horizon in days
        scaling_rule: How to scale for horizon ('sqrt_time' or 'linear')
        tail_side: 'left' for left-tail risk (default)
        
    Returns:
        VaR value (positive number representing potential loss)
    """
    if volatility <= 0:
        return np.nan
    
    alpha = 1 - confidence_level
    
    # Get z-score for the quantile
    if tail_side == 'left':
        # For left tail: use z_{1-α} which is positive for α < 0.5
        # e.g., for 95% VaR (α = 0.05), z_{0.95} ≈ 1.645
        z_score = stats.norm.ppf(1 - alpha)
    else:
        # For right tail: use z_α
        z_score = stats.norm.ppf(alpha)
    
    # Compute 1-day VaR using standard formula: VaR = -μ + z_{1-α} * σ
    # This ensures VaR is positive (loss magnitude) when μ is small or negative
    var_1day = -mean_return + z_score * volatility
    
    # Ensure VaR is non-negative (representing a loss)
    var_1day = max(0.0, var_1day)
    
    # Scale for horizon
    if horizon == 1:
        return float(var_1day)
    elif scaling_rule == 'sqrt_time':
        # Square root scaling: VaR_h = VaR_1 * sqrt(h)
        return float(var_1day * np.sqrt(horizon))
    elif scaling_rule == 'linear':
        # Linear scaling: VaR_h = VaR_1 * h
        return float(var_1day * horizon)
    else:
        # Default to sqrt_time
        return float(var_1day * np.sqrt(horizon))


def estimate_mean_volatility(
    returns: Union[pd.Series, np.ndarray],
    mean_estimator: str = 'sample_mean',
    volatility_estimator: str = 'sample_std'
) -> Tuple[float, float]:
    """
    Estimate mean and volatility from returns.
    
    Args:
        returns: Array or Series of returns
        mean_estimator: Method for mean estimation ('sample_mean')
        volatility_estimator: Method for volatility estimation ('sample_std')
        
    Returns:
        Tuple of (mean_return, volatility)
    """
    if isinstance(returns, pd.Series):
        returns_array = returns.dropna().values
    else:
        returns_array = np.array(returns)
        returns_array = returns_array[~np.isnan(returns_array)]
    
    if len(returns_array) == 0:
        return np.nan, np.nan
    
    # Estimate mean
    if mean_estimator == 'sample_mean':
        mean_return = float(np.mean(returns_array))
    else:
        raise ValueError(f"Unknown mean_estimator: {mean_estimator}")
    
    # Estimate volatility
    if volatility_estimator == 'sample_std':
        # Sample standard deviation (ddof=1 for unbiased estimate)
        volatility = float(np.std(returns_array, ddof=1))
    elif volatility_estimator == 'population_std':
        # Population standard deviation (ddof=0)
        volatility = float(np.std(returns_array, ddof=0))
    else:
        raise ValueError(f"Unknown volatility_estimator: {volatility_estimator}")
    
    return mean_return, volatility


def compute_rolling_variance_covariance_var(
    asset_returns: pd.Series,
    window: int = 252,
    confidence_level: float = 0.95,
    horizon: int = 1,
    scaling_rule: str = 'sqrt_time',
    mean_estimator: str = 'sample_mean',
    volatility_estimator: str = 'sample_std',
    tail_side: str = 'left',
    min_periods: Optional[int] = None,
    step_size: int = 1
) -> pd.Series:
    """
    Compute rolling Variance-Covariance VaR using a rolling window.
    
    For each window position, estimates mean and volatility, then computes VaR
    using the normal distribution assumption.
    
    Args:
        asset_returns: Series of asset returns with dates as index
        window: Rolling window size in days
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        horizon: Time horizon in days
        scaling_rule: How to scale for horizon ('sqrt_time' or 'linear')
        mean_estimator: Method for mean estimation ('sample_mean')
        volatility_estimator: Method for volatility estimation ('sample_std')
        tail_side: 'left' for left-tail risk
        min_periods: Minimum number of periods required for calculation
        step_size: Step size for rolling window (default: 1 for daily rolling)
        
    Returns:
        Series of rolling VaR values with dates as index
    """
    if min_periods is None:
        min_periods = min(window, len(asset_returns))
    
    # Initialize output series
    rolling_var = pd.Series(index=asset_returns.index, dtype=float)
    
    # Compute rolling mean and volatility, then VaR
    for i in range(0, len(asset_returns), step_size):
        date = asset_returns.index[i]
        
        if i < min_periods - 1:
            rolling_var.loc[date] = np.nan
            continue
        
        # Get window of returns
        start_idx = max(0, i - window + 1)
        window_returns = asset_returns.iloc[start_idx:i+1]
        
        if len(window_returns) < min_periods:
            rolling_var.loc[date] = np.nan
            continue
        
        # Estimate mean and volatility
        mean_return, volatility = estimate_mean_volatility(
            window_returns,
            mean_estimator=mean_estimator,
            volatility_estimator=volatility_estimator
        )
        
        if np.isnan(mean_return) or np.isnan(volatility) or volatility <= 0:
            rolling_var.loc[date] = np.nan
            continue
        
        # Calculate Variance-Covariance VaR for this window
        var_value = compute_variance_covariance_var(
            mean_return,
            volatility,
            confidence_level=confidence_level,
            horizon=horizon,
            scaling_rule=scaling_rule,
            tail_side=tail_side
        )
        
        rolling_var.loc[date] = var_value
    
    return rolling_var


def compute_rolling_mean_volatility(
    asset_returns: pd.Series,
    window: int = 252,
    mean_estimator: str = 'sample_mean',
    volatility_estimator: str = 'sample_std',
    min_periods: Optional[int] = None,
    step_size: int = 1
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute rolling mean and volatility series.
    
    Args:
        asset_returns: Series of asset returns with dates as index
        window: Rolling window size in days
        mean_estimator: Method for mean estimation ('sample_mean')
        volatility_estimator: Method for volatility estimation ('sample_std')
        min_periods: Minimum number of periods required for calculation
        step_size: Step size for rolling window (default: 1)
        
    Returns:
        Tuple of (rolling_mean_series, rolling_volatility_series)
    """
    if min_periods is None:
        min_periods = min(window, len(asset_returns))
    
    rolling_mean = pd.Series(index=asset_returns.index, dtype=float)
    rolling_volatility = pd.Series(index=asset_returns.index, dtype=float)
    
    for i in range(0, len(asset_returns), step_size):
        date = asset_returns.index[i]
        
        if i < min_periods - 1:
            rolling_mean.loc[date] = np.nan
            rolling_volatility.loc[date] = np.nan
            continue
        
        # Get window of returns
        start_idx = max(0, i - window + 1)
        window_returns = asset_returns.iloc[start_idx:i+1]
        
        if len(window_returns) < min_periods:
            rolling_mean.loc[date] = np.nan
            rolling_volatility.loc[date] = np.nan
            continue
        
        # Estimate mean and volatility
        mean_return, volatility = estimate_mean_volatility(
            window_returns,
            mean_estimator=mean_estimator,
            volatility_estimator=volatility_estimator
        )
        
        rolling_mean.loc[date] = mean_return
        rolling_volatility.loc[date] = volatility
    
    return rolling_mean, rolling_volatility


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
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    aligned_returns = returns.loc[common_dates]
    aligned_var = var_series.loc[common_dates]
    
    return aligned_returns, aligned_var
