"""
Returns and losses computation for QGAN scenario generation.

Computes daily returns from price data and loss series.
Loss definition: loss_t = -returns_t (consistent with risk measure conventions).
"""
import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple, Dict
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


def standardize_returns(
    returns: np.ndarray,
    method: str = 'robust_zscore',
    clip_range: Optional[tuple] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Standardize returns using specified method.

    Args:
        returns: Array of returns
        method: 'robust_zscore' or 'zscore'
        clip_range: Optional tuple (min, max) for clipping

    Returns:
        Standardized returns and statistics dict (mean, std, median, mad)
    """
    returns = returns[~np.isnan(returns)]
    
    if method == 'robust_zscore':
        median = np.median(returns)
        mad = np.median(np.abs(returns - median))
        if mad == 0:
            mad = np.std(returns)
        standardized = (returns - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std for normal
        stats = {'median': median, 'mad': mad, 'mean': np.mean(returns), 'std': np.std(returns)}
    else:  # zscore
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            std = 1.0
        standardized = (returns - mean) / std
        stats = {'mean': mean, 'std': std, 'median': np.median(returns), 'mad': np.median(np.abs(returns - np.median(returns)))}
    
    if clip_range:
        standardized = np.clip(standardized, clip_range[0], clip_range[1])
    
    return standardized, stats


def inverse_standardize(
    standardized: np.ndarray,
    stats: Dict[str, float],
    method: str = 'robust_zscore'
) -> np.ndarray:
    """
    Inverse standardization to recover original scale.

    Args:
        standardized: Standardized values
        stats: Statistics dict from standardize_returns
        method: 'robust_zscore' or 'zscore'

    Returns:
        Returns in original scale
    """
    if method == 'robust_zscore':
        return standardized * (1.4826 * stats['mad']) + stats['median']
    else:
        return standardized * stats['std'] + stats['mean']
