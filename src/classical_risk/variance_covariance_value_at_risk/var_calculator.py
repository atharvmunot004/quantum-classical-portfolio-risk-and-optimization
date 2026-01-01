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
        # Handle different NumPy versions for percentile calculation
        # NumPy 1.22.0+ uses np.quantile with 'method' parameter
        # NumPy 1.9.0-1.21.x uses np.percentile with 'interpolation' parameter
        # Older NumPy uses np.percentile without interpolation parameter
        
        # First try np.quantile with method (NumPy 1.22.0+)
        try:
            interpolation_to_method = {
                'linear': 'linear',
                'lower': 'lower',
                'higher': 'higher',
                'midpoint': 'midpoint',
                'nearest': 'nearest'
            }
            method = interpolation_to_method.get(interpolation, 'linear')
            var_1day = -np.quantile(returns_array, quantile_level / 100.0, method=method)
        except (TypeError, ValueError, AttributeError):
            # Fallback to np.percentile with interpolation (NumPy 1.9.0-1.21.x)
            try:
                var_1day = -np.percentile(returns_array, quantile_level, interpolation=interpolation)
            except TypeError:
                # Fallback for very old NumPy: use default percentile
                # Default behavior approximates 'linear' interpolation
                var_1day = -np.percentile(returns_array, quantile_level)
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
    min_periods: Optional[int] = None,
    use_vectorized: bool = True
) -> pd.Series:
    """
    Compute rolling Historical VaR using a rolling window.
    
    Uses vectorized operations when possible for better performance.
    
    Args:
        portfolio_returns: Series of portfolio returns with dates as index
        window: Rolling window size in days
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        horizon: Time horizon in days
        scaling_rule: How to scale for horizon ('sqrt_time' or 'linear')
        quantile_method: Method for quantile ('empirical')
        interpolation: Interpolation method ('linear', 'lower', 'higher', 'midpoint', 'nearest')
        min_periods: Minimum number of periods required for calculation
        use_vectorized: Whether to use vectorized rolling operations (faster)
        
    Returns:
        Series of rolling VaR values with dates as index
    """
    if min_periods is None:
        min_periods = min(window, len(portfolio_returns))
    
    # Compute quantile level (left tail)
    alpha = 1 - confidence_level
    quantile_level = alpha * 100  # Convert to percentile
    
    # Scale factor for horizon
    if horizon == 1:
        horizon_scale = 1.0
    elif scaling_rule == 'sqrt_time':
        horizon_scale = np.sqrt(horizon)
    elif scaling_rule == 'linear':
        horizon_scale = horizon
    else:
        horizon_scale = np.sqrt(horizon)
    
    if use_vectorized and len(portfolio_returns) > window:
        # Use pandas rolling with custom quantile calculation for better performance
        try:
            # Convert interpolation to method for np.quantile
            interpolation_to_method = {
                'linear': 'linear',
                'lower': 'lower',
                'higher': 'higher',
                'midpoint': 'midpoint',
                'nearest': 'nearest'
            }
            method = interpolation_to_method.get(interpolation, 'linear')
            
            # Use pandas rolling quantile which is vectorized
            def compute_var(series):
                """Compute VaR for a window of returns."""
                if len(series) < min_periods:
                    return np.nan
                try:
                    # Try np.quantile with method (NumPy 1.22.0+)
                    var_1day = -np.quantile(series.values, quantile_level / 100.0, method=method)
                except (TypeError, ValueError, AttributeError):
                    try:
                        # Fallback to np.percentile with interpolation (NumPy 1.9.0-1.21.x)
                        var_1day = -np.percentile(series.values, quantile_level, interpolation=interpolation)
                    except TypeError:
                        # Fallback for very old NumPy
                        var_1day = -np.percentile(series.values, quantile_level)
                
                return float(var_1day * horizon_scale)
            
            # Use rolling apply with vectorized quantile
            rolling_var = portfolio_returns.rolling(
                window=window,
                min_periods=min_periods
            ).apply(compute_var, raw=False)
            
            return rolling_var
            
        except Exception:
            # Fall back to loop-based approach if vectorized fails
            pass
    
    # Loop-based approach (fallback or when use_vectorized=False)
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
