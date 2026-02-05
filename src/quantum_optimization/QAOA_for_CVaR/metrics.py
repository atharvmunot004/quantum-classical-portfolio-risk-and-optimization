"""
Performance Metrics Module for QAOA Portfolio CVaR Optimization.

Computes portfolio performance metrics and optimization quality metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from scipy import stats


def compute_portfolio_return(
    returns: pd.Series,
    weights: np.ndarray,
    annualized: bool = True
) -> float:
    """Compute portfolio return."""
    portfolio_returns = (returns.values @ weights)
    total_return = (1 + portfolio_returns).prod() - 1
    
    if annualized:
        periods_per_year = 252
        n_periods = len(portfolio_returns)
        total_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    return total_return


def compute_portfolio_volatility(
    returns: pd.Series,
    weights: np.ndarray,
    annualized: bool = True
) -> float:
    """Compute portfolio volatility."""
    portfolio_returns = returns.values @ weights
    volatility = portfolio_returns.std()
    
    if annualized:
        volatility = volatility * np.sqrt(252)
    
    return volatility


def compute_portfolio_cvar(
    returns: pd.Series,
    weights: np.ndarray,
    confidence_level: float = 0.95,
    annualized: bool = True
) -> float:
    """Compute portfolio CVaR."""
    portfolio_returns = returns.values @ weights
    portfolio_losses = -portfolio_returns
    
    var_level = 1 - confidence_level
    var = np.quantile(portfolio_losses, var_level)
    
    cvar = portfolio_losses[portfolio_losses >= var].mean()
    
    if annualized:
        cvar = cvar * np.sqrt(252)
    
    return cvar


def compute_max_drawdown(portfolio_returns: pd.Series) -> float:
    """Compute maximum drawdown."""
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()


def compute_sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualized: bool = True
) -> float:
    """Compute Sharpe ratio."""
    excess_returns = portfolio_returns - risk_free_rate / 252
    
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = excess_returns.mean() / excess_returns.std()
    
    if annualized:
        sharpe = sharpe * np.sqrt(252)
    
    return sharpe


def evaluate_portfolio_performance(
    returns: pd.DataFrame,
    weights: np.ndarray,
    asset_names: List[str],
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Evaluate portfolio performance metrics.
    
    Args:
        returns: DataFrame of returns
        weights: Portfolio weights
        asset_names: List of asset names
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary of performance metrics
    """
    # Align weights with returns
    aligned_weights = np.zeros(len(returns.columns))
    for i, asset in enumerate(asset_names):
        if asset in returns.columns:
            idx = returns.columns.get_loc(asset)
            aligned_weights[idx] = weights[i]
    
    # Compute portfolio returns
    portfolio_returns = pd.Series(
        (returns.values @ aligned_weights),
        index=returns.index
    )
    
    metrics = {
        'realized_return': compute_portfolio_return(portfolio_returns, np.ones(len(portfolio_returns))),
        'realized_volatility': compute_portfolio_volatility(portfolio_returns, np.ones(len(portfolio_returns))),
        'realized_cvar': compute_portfolio_cvar(portfolio_returns, np.ones(len(portfolio_returns))),
        'max_drawdown': compute_max_drawdown(portfolio_returns),
        'sharpe_ratio': compute_sharpe_ratio(portfolio_returns, risk_free_rate)
    }
    
    return metrics
