"""
Returns and losses computation for QAOA CVaR asset-level evaluation.

Computes daily returns from price data and loss series.
Loss definition: loss_t = -returns_t (consistent with risk measure conventions).
"""
import pandas as pd
import numpy as np
from typing import Union, Optional
from pathlib import Path


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


def compute_losses_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute loss series from returns.

    Global invariant: loss_t = -returns_t
    This ensures consistent loss definition across all modules.

    Args:
        returns: DataFrame of returns with dates as index and assets as columns

    Returns:
        DataFrame of losses with same index and columns as returns
    """
    return -returns


def load_panel_prices(panel_price_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load panel price data from parquet or CSV file.

    Args:
        panel_price_path: Path to panel price file

    Returns:
        DataFrame with dates as index and assets as columns
    """
    panel_price_path = Path(panel_price_path)

    if not panel_price_path.exists():
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        panel_price_path = project_root / panel_price_path

        if not panel_price_path.exists():
            raise FileNotFoundError(f"Panel price file not found: {panel_price_path}")

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
