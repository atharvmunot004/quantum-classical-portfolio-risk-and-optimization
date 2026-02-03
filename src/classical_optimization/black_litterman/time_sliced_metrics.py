"""
Time-sliced metrics computation module for Black-Litterman optimization.

Computes metrics sliced by time periods (e.g., by calendar year) to analyze
temporal patterns in portfolio performance.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from .metrics import (
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio
)


def compute_time_sliced_returns(
    portfolio_returns: pd.Series,
    time_unit: str = 'calendar_year',
    slicing_method: Dict = None,
    return_computation: Dict = None
) -> pd.DataFrame:
    """
    Compute time-sliced returns for portfolio.
    
    Args:
        portfolio_returns: Series of portfolio returns with dates as index
        time_unit: 'calendar_year', 'quarter', or 'month'
        slicing_method: Dictionary with slicing configuration
        return_computation: Dictionary with return computation configuration
        
    Returns:
        DataFrame with time-sliced returns
    """
    if len(portfolio_returns) == 0:
        return pd.DataFrame()
    
    # Ensure index is datetime
    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
    
    portfolio_returns = portfolio_returns.sort_index()
    
    # Default slicing method
    if slicing_method is None:
        slicing_method = {
            'type': 'fixed_non_overlapping',
            'alignment': 'year_end'
        }
    
    # Default return computation
    if return_computation is None:
        return_computation = {
            'method': 'realized_historical',
            'compounding': 'log_sum',
            'annualization': False
        }
    
    # Group by time period
    if time_unit == 'calendar_year':
        groups = portfolio_returns.groupby(portfolio_returns.index.year)
    elif time_unit == 'quarter':
        groups = portfolio_returns.groupby([
            portfolio_returns.index.year,
            portfolio_returns.index.quarter
        ])
    elif time_unit == 'month':
        groups = portfolio_returns.groupby([
            portfolio_returns.index.year,
            portfolio_returns.index.month
        ])
    else:
        raise ValueError(f"Unknown time_unit: {time_unit}")
    
    time_slices = []
    
    for period, period_returns in groups:
        if len(period_returns) < 1:
            continue
        
        period_name = str(period) if isinstance(period, tuple) else str(period)
        
        # Compute returns based on method
        if return_computation.get('compounding') == 'log_sum':
            # Sum of log returns
            yearly_return = period_returns.sum()
        elif return_computation.get('compounding') == 'geometric':
            # Geometric compounding: (1 + r1) * (1 + r2) * ... - 1
            yearly_return = (1 + period_returns).prod() - 1
        else:
            # Simple sum
            yearly_return = period_returns.sum()
        
        # Annualize if requested
        if return_computation.get('annualization', False):
            if time_unit == 'calendar_year':
                pass  # Already annual
            elif time_unit == 'quarter':
                yearly_return = yearly_return * 4
            elif time_unit == 'month':
                yearly_return = yearly_return * 12
        
        time_slices.append({
            'period': period_name,
            'start_date': period_returns.index[0],
            'end_date': period_returns.index[-1],
            'yearly_return': yearly_return,
            'cumulative_return': (1 + period_returns).prod() - 1,
            'num_observations': len(period_returns)
        })
    
    if len(time_slices) == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame(time_slices)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    
    return df


def compute_time_sliced_risk_metrics(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    time_unit: str = 'calendar_year'
) -> pd.DataFrame:
    """
    Compute time-sliced risk metrics.
    
    Args:
        portfolio_returns: Series of portfolio returns with dates as index
        risk_free_rate: Risk-free rate
        time_unit: 'calendar_year', 'quarter', or 'month'
        
    Returns:
        DataFrame with time-sliced risk metrics
    """
    if len(portfolio_returns) == 0:
        return pd.DataFrame()
    
    # Ensure index is datetime
    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
    
    portfolio_returns = portfolio_returns.sort_index()
    
    # Group by time period
    if time_unit == 'calendar_year':
        groups = portfolio_returns.groupby(portfolio_returns.index.year)
    elif time_unit == 'quarter':
        groups = portfolio_returns.groupby([
            portfolio_returns.index.year,
            portfolio_returns.index.quarter
        ])
    elif time_unit == 'month':
        groups = portfolio_returns.groupby([
            portfolio_returns.index.year,
            portfolio_returns.index.month
        ])
    else:
        raise ValueError(f"Unknown time_unit: {time_unit}")
    
    time_slices = []
    
    for period, period_returns in groups:
        if len(period_returns) < 2:
            continue
        
        period_name = str(period) if isinstance(period, tuple) else str(period)
        
        volatility = period_returns.std()
        max_dd = compute_max_drawdown(period_returns)
        downside_returns = period_returns[period_returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0.0
        
        # Annualize if needed
        if time_unit == 'calendar_year':
            pass  # Already annual
        elif time_unit == 'quarter':
            volatility = volatility * np.sqrt(4)
            downside_deviation = downside_deviation * np.sqrt(4)
        elif time_unit == 'month':
            volatility = volatility * np.sqrt(12)
            downside_deviation = downside_deviation * np.sqrt(12)
        
        sharpe = compute_sharpe_ratio(period_returns, risk_free_rate, annualize=False)
        sortino = compute_sortino_ratio(period_returns, risk_free_rate, annualize=False)
        calmar = compute_calmar_ratio(period_returns, annualize=False)
        
        time_slices.append({
            'period': period_name,
            'start_date': period_returns.index[0],
            'end_date': period_returns.index[-1],
            'volatility': volatility,
            'max_drawdown': max_dd,
            'downside_deviation': downside_deviation,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'num_observations': len(period_returns)
        })
    
    if len(time_slices) == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame(time_slices)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    
    return df


def compute_time_sliced_tail_metrics(
    portfolio_returns: pd.Series,
    time_unit: str = 'calendar_year',
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Compute time-sliced tail risk metrics (VaR, CVaR).
    
    Args:
        portfolio_returns: Series of portfolio returns with dates as index
        time_unit: 'calendar_year', 'quarter', or 'month'
        confidence_level: Confidence level for VaR/CVaR (default: 0.95)
        
    Returns:
        DataFrame with time-sliced tail risk metrics
    """
    if len(portfolio_returns) == 0:
        return pd.DataFrame()
    
    # Ensure index is datetime
    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
    
    portfolio_returns = portfolio_returns.sort_index()
    
    # Group by time period
    if time_unit == 'calendar_year':
        groups = portfolio_returns.groupby(portfolio_returns.index.year)
    elif time_unit == 'quarter':
        groups = portfolio_returns.groupby([
            portfolio_returns.index.year,
            portfolio_returns.index.quarter
        ])
    elif time_unit == 'month':
        groups = portfolio_returns.groupby([
            portfolio_returns.index.year,
            portfolio_returns.index.month
        ])
    else:
        raise ValueError(f"Unknown time_unit: {time_unit}")
    
    time_slices = []
    
    for period, period_returns in groups:
        if len(period_returns) < 2:
            continue
        
        period_name = str(period) if isinstance(period, tuple) else str(period)
        
        # VaR (Value at Risk)
        var_percentile = (1 - confidence_level) * 100
        var_value = np.percentile(period_returns, var_percentile)
        var_95 = abs(var_value) if var_value < 0 else var_value
        
        # CVaR (Conditional Value at Risk)
        cvar_returns = period_returns[period_returns <= var_value]
        cvar_value = cvar_returns.mean() if len(cvar_returns) > 0 else np.nan
        cvar_95 = abs(cvar_value) if not np.isnan(cvar_value) and cvar_value < 0 else (cvar_value if not np.isnan(cvar_value) else np.nan)
        
        time_slices.append({
            'period': period_name,
            'start_date': period_returns.index[0],
            'end_date': period_returns.index[-1],
            f'value_at_risk_{int(confidence_level * 100)}': var_95,
            f'conditional_value_at_risk_{int(confidence_level * 100)}': cvar_95,
            'num_observations': len(period_returns)
        })
    
    if len(time_slices) == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame(time_slices)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    
    return df


def compare_time_sliced_prior_vs_posterior(
    prior_portfolio_returns: pd.Series,
    posterior_portfolio_returns: pd.Series,
    market_portfolio_returns: pd.Series,
    time_unit: str = 'calendar_year'
) -> pd.DataFrame:
    """
    Compare prior, posterior, and market portfolios across time slices.
    
    Args:
        prior_portfolio_returns: Series of prior portfolio returns
        posterior_portfolio_returns: Series of posterior portfolio returns
        market_portfolio_returns: Series of market portfolio returns
        time_unit: 'calendar_year', 'quarter', or 'month'
        
    Returns:
        DataFrame with comparison metrics for each time slice
    """
    # Align all series
    common_index = (
        prior_portfolio_returns.index
        .intersection(posterior_portfolio_returns.index)
        .intersection(market_portfolio_returns.index)
    )
    
    if len(common_index) == 0:
        return pd.DataFrame()
    
    prior_aligned = prior_portfolio_returns.loc[common_index]
    posterior_aligned = posterior_portfolio_returns.loc[common_index]
    market_aligned = market_portfolio_returns.loc[common_index]
    
    # Ensure index is datetime
    if not isinstance(common_index, pd.DatetimeIndex):
        common_index = pd.to_datetime(common_index)
        prior_aligned.index = common_index
        posterior_aligned.index = common_index
        market_aligned.index = common_index
    
    # Group by time period
    if time_unit == 'calendar_year':
        groups = prior_aligned.groupby(prior_aligned.index.year)
    elif time_unit == 'quarter':
        groups = prior_aligned.groupby([
            prior_aligned.index.year,
            prior_aligned.index.quarter
        ])
    elif time_unit == 'month':
        groups = prior_aligned.groupby([
            prior_aligned.index.year,
            prior_aligned.index.month
        ])
    else:
        raise ValueError(f"Unknown time_unit: {time_unit}")
    
    time_slices = []
    
    for period, period_prior in groups:
        if len(period_prior) < 1:
            continue
        
        period_name = str(period) if isinstance(period, tuple) else str(period)
        period_dates = period_prior.index
        
        period_posterior = posterior_aligned.loc[period_dates]
        period_market = market_aligned.loc[period_dates]
        
        # Compute returns
        prior_return = period_prior.sum()
        posterior_return = period_posterior.sum()
        market_return = period_market.sum()
        
        # Active returns
        posterior_active_return = posterior_return - market_return
        posterior_minus_prior_return = posterior_return - prior_return
        posterior_minus_market_return = posterior_return - market_return
        
        # Tracking error
        active_returns = period_posterior - period_market
        tracking_error = active_returns.std() if len(active_returns) > 1 else np.nan
        
        # Information ratio
        info_ratio = (posterior_active_return / tracking_error) if tracking_error > 0 and not np.isnan(tracking_error) else np.nan
        
        time_slices.append({
            'period': period_name,
            'start_date': period_dates[0],
            'end_date': period_dates[-1],
            'prior_return': prior_return,
            'posterior_return': posterior_return,
            'market_return': market_return,
            'active_return': posterior_active_return,
            'tracking_error': tracking_error,
            'information_ratio': info_ratio,
            'posterior_minus_prior_return': posterior_minus_prior_return,
            'posterior_minus_market_return': posterior_minus_market_return,
            'num_observations': len(period_dates)
        })
    
    if len(time_slices) == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame(time_slices)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    
    return df


def analyze_temporal_performance_stability(
    portfolio_returns: pd.Series,
    time_unit: str = 'calendar_year',
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Analyze temporal stability of portfolio performance.
    
    Args:
        portfolio_returns: Series of portfolio returns with dates as index
        time_unit: 'calendar_year', 'quarter', or 'month'
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary with stability metrics
    """
    if len(portfolio_returns) == 0:
        return {}
    
    # Compute time-sliced returns
    returns_df = compute_time_sliced_returns(portfolio_returns, time_unit=time_unit)
    
    if len(returns_df) == 0:
        return {}
    
    # Compute time-sliced risk metrics
    risk_df = compute_time_sliced_risk_metrics(portfolio_returns, risk_free_rate, time_unit=time_unit)
    
    if len(risk_df) == 0:
        return {}
    
    # Merge on period
    merged = returns_df.merge(risk_df, on='period', suffixes=('', '_risk'))
    
    if len(merged) == 0:
        return {}
    
    # Stability metrics
    yearly_returns = merged['yearly_return'].values
    yearly_sharpes = merged['sharpe_ratio'].values
    
    # Return volatility of returns (volatility of period returns)
    return_volatility_of_returns = np.std(yearly_returns) if len(yearly_returns) > 1 else np.nan
    
    # Sharpe variance over time
    sharpe_variance_over_time = np.var(yearly_sharpes) if len(yearly_sharpes) > 1 else np.nan
    
    # Drawdown persistence (max consecutive negative periods)
    negative_periods = yearly_returns < 0
    if np.any(negative_periods):
        # Find longest sequence of negative returns
        max_consecutive = 0
        current_consecutive = 0
        for is_negative in negative_periods:
            if is_negative:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        drawdown_persistence = max_consecutive
    else:
        drawdown_persistence = 0
    
    # Worst and best year returns
    worst_year_return = np.min(yearly_returns) if len(yearly_returns) > 0 else np.nan
    best_year_return = np.max(yearly_returns) if len(yearly_returns) > 0 else np.nan
    
    return {
        'return_volatility_of_returns': return_volatility_of_returns,
        'sharpe_variance_over_time': sharpe_variance_over_time,
        'drawdown_persistence': drawdown_persistence,
        'worst_year_return': worst_year_return,
        'best_year_return': best_year_return
    }


def compute_batch_time_sliced_metrics(
    returns_matrix: np.ndarray,
    dates: pd.DatetimeIndex,
    portfolio_ids: np.ndarray,
    risk_free_rate: float = 0.0,
    time_unit: str = 'calendar_year'
) -> pd.DataFrame:
    """
    Compute time-sliced metrics in batch from returns matrix.
    
    Args:
        returns_matrix: Returns matrix (T x N_portfolios)
        dates: DatetimeIndex of length T
        portfolio_ids: Array of portfolio IDs of length N_portfolios
        risk_free_rate: Risk-free rate
        time_unit: 'calendar_year', 'quarter', or 'month'
        
    Returns:
        DataFrame in long format: portfolio_id, period, yearly_return, sharpe, sortino, max_drawdown, var, cvar, etc.
    """
    from .metrics import compute_batch_metrics_vectorized
    
    T, N = returns_matrix.shape
    
    if len(dates) != T:
        raise ValueError(f"dates length ({len(dates)}) must match returns_matrix time dimension ({T})")
    if len(portfolio_ids) != N:
        raise ValueError(f"portfolio_ids length ({len(portfolio_ids)}) must match returns_matrix portfolio dimension ({N})")
    
    # Group dates by time unit
    if time_unit == 'calendar_year':
        periods = dates.year
    elif time_unit == 'quarter':
        periods = list(zip(dates.year, dates.quarter))
    elif time_unit == 'month':
        periods = list(zip(dates.year, dates.month))
    else:
        raise ValueError(f"Unknown time_unit: {time_unit}")
    
    unique_periods = sorted(set(periods))
    
    results = []
    
    for period in unique_periods:
        # Get indices for this period
        if time_unit == 'calendar_year':
            period_mask = dates.year == period
        elif time_unit == 'quarter':
            period_mask = (dates.year == period[0]) & (dates.quarter == period[1])
        else:  # month
            period_mask = (dates.year == period[0]) & (dates.month == period[1])
        
        period_returns = returns_matrix[period_mask, :]
        
        if period_returns.shape[0] < 1:
            continue
        
        # Compute metrics for this period
        period_metrics = compute_batch_metrics_vectorized(period_returns, risk_free_rate, use_gpu=False)
        
        # Yearly return (sum of log returns)
        yearly_returns = np.sum(period_returns, axis=0)
        
        # Add to results
        period_name = str(period) if isinstance(period, tuple) else str(period)
        for i, portfolio_id in enumerate(portfolio_ids):
            results.append({
                'portfolio_id': portfolio_id,
                'period': period_name,
                'yearly_return': yearly_returns[i],
                'sharpe_ratio': period_metrics['sharpe_ratio'][i],
                'sortino_ratio': period_metrics['sortino_ratio'][i],
                'max_drawdown': period_metrics['max_drawdown'][i],
                'volatility': period_metrics['volatility'][i],
                'value_at_risk': period_metrics['value_at_risk'][i],
                'conditional_value_at_risk': period_metrics['conditional_value_at_risk'][i],
                'skewness': period_metrics['skewness'][i],
                'kurtosis': period_metrics['kurtosis'][i],
                'jarque_bera_p_value': period_metrics['jarque_bera_p_value'][i]
            })
    
    if len(results) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(results)