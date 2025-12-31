"""
Historical Value-at-Risk calculation module.

Implements rolling VaR calculation using the historical simulation method
with empirical quantiles and linear interpolation.
"""
import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
from scipy import stats


def compute_historical_var(
    returns: Union[pd.Series, np.ndarray],
    confidence_level: float = 0.95,
    horizon: int = 1,
    scaling_rule: str = 'sqrt_time',
    quantile_method: str = 'empirical',
    interpolation: str = 'linear'
) -> float:
    """
    Calculate Historical Value-at-Risk using empirical quantiles.
    
    Args:
        returns: Array or Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        horizon: Time horizon in days
        scaling_rule: How to scale for horizon ('sqrt_time' or 'linear')
        quantile_method: Method for quantile ('empirical')
        interpolation: Interpolation method ('linear', 'lower', 'higher', 'midpoint', 'nearest')
        
    Returns:
        VaR value (positive number representing potential loss)
    """
    if isinstance(returns, pd.Series):
        returns_array = returns.dropna().values
    else:
        returns_array = np.array(returns)
        returns_array = returns_array[~np.isnan(returns_array)]
    
    if len(returns_array) == 0:
        return np.nan
    
    # Compute quantile level (left tail)
    alpha = 1 - confidence_level
    quantile_level = alpha * 100  # Convert to percentile
    
    # Compute empirical quantile
    if quantile_method == 'empirical':
        if interpolation == 'linear':
            var_1day = -np.percentile(returns_array, quantile_level, interpolation='linear')
        elif interpolation == 'lower':
            var_1day = -np.percentile(returns_array, quantile_level, interpolation='lower')
        elif interpolation == 'higher':
            var_1day = -np.percentile(returns_array, quantile_level, interpolation='higher')
        elif interpolation == 'midpoint':
            var_1day = -np.percentile(returns_array, quantile_level, interpolation='midpoint')
        elif interpolation == 'nearest':
            var_1day = -np.percentile(returns_array, quantile_level, interpolation='nearest')
        else:
            # Default to linear
            var_1day = -np.percentile(returns_array, quantile_level, interpolation='linear')
    else:
        raise ValueError(f"Unknown quantile_method: {quantile_method}")
    
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


def compute_rolling_historical_var(
    portfolio_returns: pd.Series,
    window: int = 252,
    confidence_level: float = 0.95,
    horizon: int = 1,
    scaling_rule: str = 'sqrt_time',
    quantile_method: str = 'empirical',
    interpolation: str = 'linear',
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Compute rolling Historical VaR using a rolling window.
    
    Args:
        portfolio_returns: Series of portfolio returns with dates as index
        window: Rolling window size in days
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        horizon: Time horizon in days
        scaling_rule: How to scale for horizon ('sqrt_time' or 'linear')
        quantile_method: Method for quantile ('empirical')
        interpolation: Interpolation method ('linear', 'lower', 'higher', 'midpoint', 'nearest')
        min_periods: Minimum number of periods required for calculation
        
    Returns:
        Series of rolling VaR values with dates as index
    """
    if min_periods is None:
        min_periods = min(window, len(portfolio_returns))
    
    rolling_var = pd.Series(index=portfolio_returns.index, dtype=float)
    
    for i in range(len(portfolio_returns)):
        if i < min_periods - 1:
            rolling_var.iloc[i] = np.nan
            continue
        
        # Get window of returns
        start_idx = max(0, i - window + 1)
        window_returns = portfolio_returns.iloc[start_idx:i+1]
        
        if len(window_returns) < min_periods:
            rolling_var.iloc[i] = np.nan
            continue
        
        # Calculate Historical VaR for this window
        var_value = compute_historical_var(
            window_returns,
            confidence_level=confidence_level,
            horizon=horizon,
            scaling_rule=scaling_rule,
            quantile_method=quantile_method,
            interpolation=interpolation
        )
        
        rolling_var.iloc[i] = var_value
    
    return rolling_var


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
