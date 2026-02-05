"""
Performance Metrics Module for QMV Portfolio Optimization.

Computes batch portfolio performance metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def compute_portfolio_returns_batch(
    returns: pd.DataFrame,
    weights_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute portfolio returns for multiple portfolios in batch.
    
    Args:
        returns: DataFrame of returns (T x N assets)
        weights_matrix: Weight matrix (N portfolios x N assets)
        
    Returns:
        Portfolio returns matrix (T x N portfolios)
    """
    # returns: T x N_assets
    # weights_matrix: N_portfolios x N_assets
    # Result: T x N_portfolios
    
    portfolio_returns = returns.values @ weights_matrix.T
    
    return portfolio_returns


def compute_realized_return_batch(
    portfolio_returns: np.ndarray,
    annualized: bool = True
) -> np.ndarray:
    """
    Compute realized returns for multiple portfolios.
    
    Args:
        portfolio_returns: Portfolio returns matrix (T x N portfolios)
        annualized: Whether to annualize
        
    Returns:
        Array of realized returns (N portfolios,)
    """
    total_returns = (1 + portfolio_returns).prod(axis=0) - 1
    
    if annualized:
        periods_per_year = 252
        n_periods = portfolio_returns.shape[0]
        total_returns = (1 + total_returns) ** (periods_per_year / n_periods) - 1
    
    return total_returns


def compute_realized_volatility_batch(
    portfolio_returns: np.ndarray,
    annualized: bool = True
) -> np.ndarray:
    """
    Compute realized volatility for multiple portfolios.
    
    Args:
        portfolio_returns: Portfolio returns matrix (T x N portfolios)
        annualized: Whether to annualize
        
    Returns:
        Array of realized volatilities (N portfolios,)
    """
    volatilities = portfolio_returns.std(axis=0)
    
    if annualized:
        volatilities = volatilities * np.sqrt(252)
    
    return volatilities


def compute_sharpe_ratio_batch(
    portfolio_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualized: bool = True
) -> np.ndarray:
    """
    Compute Sharpe ratios for multiple portfolios.
    
    Args:
        portfolio_returns: Portfolio returns matrix (T x N portfolios)
        risk_free_rate: Risk-free rate
        annualized: Whether to annualize
        
    Returns:
        Array of Sharpe ratios (N portfolios,)
    """
    excess_returns = portfolio_returns - risk_free_rate / 252
    
    mean_returns = excess_returns.mean(axis=0)
    std_returns = excess_returns.std(axis=0)
    
    # Avoid division by zero
    sharpe_ratios = np.where(std_returns > 1e-10, mean_returns / std_returns, 0.0)
    
    if annualized:
        sharpe_ratios = sharpe_ratios * np.sqrt(252)
    
    return sharpe_ratios


def compute_max_drawdown_batch(
    portfolio_returns: np.ndarray
) -> np.ndarray:
    """
    Compute maximum drawdown for multiple portfolios.
    
    Args:
        portfolio_returns: Portfolio returns matrix (T x N portfolios)
        
    Returns:
        Array of maximum drawdowns (N portfolios,)
    """
    cumulative = (1 + portfolio_returns).cumprod(axis=0)
    running_max = np.maximum.accumulate(cumulative, axis=0)
    drawdown = (cumulative - running_max) / running_max
    
    max_drawdowns = drawdown.min(axis=0)
    
    return max_drawdowns


def compute_turnover_batch(
    weights_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute turnover for multiple portfolios.
    
    Args:
        weights_matrix: Weight matrix (N portfolios x N assets)
        
    Returns:
        Array of turnovers (N portfolios,)
    """
    # For single-period optimization, turnover is zero
    # This would be computed across rebalancing dates
    turnovers = np.zeros(weights_matrix.shape[0])
    
    return turnovers


def compute_portfolio_metrics_batch(
    returns: pd.DataFrame,
    weights_matrix: np.ndarray,
    asset_names: List[str],
    risk_free_rate: float = 0.0
) -> pd.DataFrame:
    """
    Compute comprehensive portfolio metrics for multiple portfolios in batch.
    
    Args:
        returns: DataFrame of returns (T x N assets)
        weights_matrix: Weight matrix (N portfolios x N assets)
        asset_names: List of asset names
        risk_free_rate: Risk-free rate
        
    Returns:
        DataFrame of metrics (N portfolios x metrics)
    """
    # Align weights with returns
    aligned_weights = np.zeros((weights_matrix.shape[0], len(returns.columns)))
    
    for i, asset in enumerate(asset_names):
        if asset in returns.columns:
            asset_idx = returns.columns.get_loc(asset)
            aligned_weights[:, asset_idx] = weights_matrix[:, i]
    
    # Compute portfolio returns
    portfolio_returns = compute_portfolio_returns_batch(returns, aligned_weights)
    
    # Compute metrics
    metrics = {
        'realized_return': compute_realized_return_batch(portfolio_returns),
        'realized_volatility': compute_realized_volatility_batch(portfolio_returns),
        'sharpe_ratio': compute_sharpe_ratio_batch(portfolio_returns, risk_free_rate),
        'max_drawdown': compute_max_drawdown_batch(portfolio_returns),
        'turnover': compute_turnover_batch(aligned_weights)
    }
    
    return pd.DataFrame(metrics)
