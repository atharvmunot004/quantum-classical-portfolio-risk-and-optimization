"""Panel price loading and log/simple returns for qPCA factor risk."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union


def load_panel_prices(panel_price_path: Union[str, Path]) -> pd.DataFrame:
    """Load panel price data from parquet or CSV. Index = dates, columns = assets."""
    path = Path(panel_price_path)
    if not path.exists():
        raise FileNotFoundError(f"Panel price file not found: {path}")
    if path.suffix == ".parquet":
        prices = pd.read_parquet(path)
    elif path.suffix == ".csv":
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    if prices.index.duplicated().any():
        prices = prices[~prices.index.duplicated(keep="first")]
    return prices


def compute_daily_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute daily returns. method: 'log' or 'simple'."""
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = (prices / prices.shift(1)) - 1
    else:
        raise ValueError(f"Unknown method: {method}")
    return returns.dropna(how="all")
