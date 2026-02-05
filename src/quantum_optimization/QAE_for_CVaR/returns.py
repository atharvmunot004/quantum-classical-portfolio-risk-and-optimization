"""
Returns computation module for QAE Portfolio CVaR Evaluation.

Computes daily returns from price data and handles data preprocessing.
"""
import pandas as pd
import numpy as np
from typing import Union, Optional
from pathlib import Path


def load_panel_prices(panel_price_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load panel price data from parquet or CSV file.
    
    Args:
        panel_price_path: Path to panel price file
        
    Returns:
        DataFrame with dates as index and assets as columns
    """
    original_path = str(panel_price_path)
    panel_price_path = Path(panel_price_path)
    
    if panel_price_path.is_absolute() and panel_price_path.exists():
        pass
    elif panel_price_path.exists():
        pass
    else:
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        panel_price_path = project_root / original_path
        
        if not panel_price_path.exists():
            attempted_paths = [
                str(Path(original_path).resolve()),
                str(panel_price_path)
            ]
            raise FileNotFoundError(
                f"Panel price file not found. Attempted paths:\n"
                f"  1. {attempted_paths[0]}\n"
                f"  2. {attempted_paths[1]}\n"
            )
    
    if panel_price_path.suffix == '.parquet':
        prices = pd.read_parquet(panel_price_path)
    elif panel_price_path.suffix == '.csv':
        prices = pd.read_csv(panel_price_path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {panel_price_path.suffix}")
    
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    
    prices = prices.sort_index()
    
    if prices.index.duplicated().any():
        prices = prices[~prices.index.duplicated(keep='first')]
    
    return prices


def compute_daily_returns(
    prices: pd.DataFrame,
    method: str = 'log'
) -> pd.DataFrame:
    """
    Compute daily returns from price data.
    
    Args:
        prices: DataFrame with dates as index and assets as columns
        method: 'log' for log returns, 'simple' for simple returns
        
    Returns:
        DataFrame of daily returns with same index and columns as prices
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    elif method == 'simple':
        returns = (prices / prices.shift(1)) - 1
    else:
        raise ValueError(f"Unknown method: {method}. Use 'log' or 'simple'")
    
    returns = returns.dropna()
    return returns


def compute_portfolio_losses(
    returns: pd.DataFrame,
    weights: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute portfolio losses from returns and weights.
    
    Loss definition: L_p = -R @ w (negative portfolio return)
    
    Args:
        returns: DataFrame of returns (T x N)
        weights: DataFrame of portfolio weights (N portfolios x M assets)
        
    Returns:
        DataFrame of portfolio losses (T x N portfolios)
    """
    # Align assets
    common_assets = returns.columns.intersection(weights.columns)
    if len(common_assets) == 0:
        raise ValueError("No common assets between returns and weights")
    
    returns_aligned = returns[common_assets]
    weights_aligned = weights[common_assets]
    
    # Compute portfolio returns: R @ w
    portfolio_returns = returns_aligned @ weights_aligned.T
    
    # Portfolio losses: -portfolio_returns
    portfolio_losses = -portfolio_returns
    
    return portfolio_losses
