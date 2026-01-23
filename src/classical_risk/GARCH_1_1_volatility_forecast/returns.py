"""Returns computation utilities.

- compute_daily_returns: log/simple returns
- compute_portfolio_returns: weighted aggregation
- load_panel_prices / load_portfolio_weights: parquet/csv loaders
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


def compute_daily_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    """Compute daily returns from price data.

    Args:
        prices: DataFrame with DatetimeIndex and asset columns.
        method: 'log' or 'simple'.

    Returns:
        DataFrame of returns (first row dropped).
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    elif method == 'simple':
        returns = (prices / prices.shift(1)) - 1.0
    else:
        raise ValueError(f"Unknown method: {method}. Use 'log' or 'simple'.")

    return returns.dropna(how='all')


def compute_portfolio_returns(
    asset_returns: pd.DataFrame,
    portfolio_weights: pd.Series,
    align_assets: bool = True
) -> pd.Series:
    """Compute portfolio returns from asset returns and weights."""
    if align_assets:
        common_assets = asset_returns.columns.intersection(portfolio_weights.index)
        if len(common_assets) == 0:
            raise ValueError("No common assets between returns and weights")
        asset_returns = asset_returns[common_assets]
        portfolio_weights = portfolio_weights[common_assets]

    w = portfolio_weights.astype(float)
    s = float(w.sum())
    if s == 0:
        raise ValueError("Portfolio weights sum to zero")
    w = w / s

    return (asset_returns * w).sum(axis=1)


def load_panel_prices(panel_price_path: Union[str, Path]) -> pd.DataFrame:
    """Load a panel of prices (parquet/csv)."""
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

    return prices.sort_index()


def load_portfolio_weights(portfolio_weights_path: Union[str, Path]) -> pd.DataFrame:
    """Load portfolio weights (parquet/csv)."""
    portfolio_weights_path = Path(portfolio_weights_path)

    if not portfolio_weights_path.exists():
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        portfolio_weights_path = project_root / portfolio_weights_path

    if not portfolio_weights_path.exists():
        raise FileNotFoundError(f"Portfolio weights file not found: {portfolio_weights_path}")

    if portfolio_weights_path.suffix == '.parquet':
        weights = pd.read_parquet(portfolio_weights_path)
    elif portfolio_weights_path.suffix == '.csv':
        weights = pd.read_csv(portfolio_weights_path, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {portfolio_weights_path.suffix}")

    return weights
