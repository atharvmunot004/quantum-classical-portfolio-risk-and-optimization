"""
Returns computation module for Quantum Annealing Portfolio Optimization.

Computes daily returns from price data and handles data preprocessing.
"""
import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
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
            attempted_paths = []
            first_attempt = Path(original_path)
            if first_attempt.is_absolute():
                attempted_paths.append(str(first_attempt))
            else:
                attempted_paths.append(str(first_attempt.resolve()))
            attempted_paths.append(str(panel_price_path))
            
            raise FileNotFoundError(
                f"Panel price file not found. Attempted paths:\n"
                f"  1. {attempted_paths[0]}\n"
                f"  2. {attempted_paths[1]}\n"
                f"Please check that the file exists or update the path in llm.json"
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


def filter_asset_universe(
    returns: pd.DataFrame,
    max_assets: int = 40,
    selection_method: str = 'liquidity_and_data_availability',
    min_required_observations: int = 1000
) -> pd.DataFrame:
    """
    Filter asset universe based on liquidity and data availability.
    
    Args:
        returns: DataFrame of returns
        max_assets: Maximum number of assets to select
        selection_method: Selection method ('liquidity_and_data_availability')
        min_required_observations: Minimum required observations per asset
        
    Returns:
        Filtered returns DataFrame
    """
    # Drop assets with insufficient data
    returns = returns.dropna(axis=1, how='all')
    
    # Count non-null observations per asset
    valid_counts = returns.notna().sum()
    
    # Filter by minimum observations
    valid_assets = valid_counts[valid_counts >= min_required_observations].index
    
    if len(valid_assets) == 0:
        raise ValueError(f"No assets with at least {min_required_observations} observations")
    
    returns_filtered = returns[valid_assets]
    
    # If we have more assets than max_assets, select by liquidity (trading volume proxy)
    # Use variance as proxy for liquidity (higher variance = more trading activity)
    if len(valid_assets) > max_assets:
        asset_variances = returns_filtered.var()
        top_assets = asset_variances.nlargest(max_assets).index
        returns_filtered = returns_filtered[top_assets]
    
    # Drop any remaining NaN rows (panel intersection)
    returns_filtered = returns_filtered.dropna()
    
    return returns_filtered


def compute_rolling_mean_return(
    returns: pd.DataFrame,
    window: int,
    annualized: bool = True
) -> pd.Series:
    """
    Compute rolling mean return.
    
    Args:
        returns: DataFrame of returns
        window: Rolling window size
        annualized: Whether to annualize returns
        
    Returns:
        Series of expected returns (last value in window)
    """
    mean_returns = returns.rolling(window=window).mean().iloc[-1]
    
    if annualized:
        mean_returns = mean_returns * 252  # Annualize assuming 252 trading days
    
    return mean_returns


def compute_rolling_correlation_matrix(
    returns: pd.DataFrame,
    window: int
) -> pd.DataFrame:
    """
    Compute rolling correlation matrix.
    
    Args:
        returns: DataFrame of returns
        window: Rolling window size
        
    Returns:
        Correlation matrix DataFrame
    """
    # Compute correlation for the last window
    window_returns = returns.iloc[-window:] if len(returns) >= window else returns
    corr_matrix = window_returns.corr()
    
    # Fill NaN with 0 (for assets with no variance)
    corr_matrix = corr_matrix.fillna(0)
    
    return corr_matrix


def compute_cvar(
    returns: pd.DataFrame,
    confidence_level: float = 0.95,
    window: Optional[int] = None
) -> pd.Series:
    """
    Compute Conditional Value at Risk (CVaR).
    
    Args:
        returns: DataFrame of returns
        confidence_level: Confidence level (e.g., 0.95)
        window: Optional rolling window size
        
    Returns:
        Series of CVaR values
    """
    if window is not None:
        returns_window = returns.iloc[-window:]
    else:
        returns_window = returns
    
    # Compute VaR (Value at Risk) at confidence level
    var_level = 1 - confidence_level
    var = returns_window.quantile(var_level, axis=0)
    
    # CVaR is the mean of returns below VaR (for each asset)
    cvar = pd.Series(index=returns_window.columns, dtype=float)
    
    for asset in returns_window.columns:
        asset_returns = returns_window[asset]
        var_value = var[asset]
        below_var = asset_returns[asset_returns <= var_value]
        
        if len(below_var) > 0:
            cvar[asset] = below_var.mean()
        else:
            # If no returns below VaR, use the VaR value itself
            cvar[asset] = var_value
    
    # Fill any remaining NaN with 0
    cvar = cvar.fillna(0)
    
    return cvar


def load_precomputed_portfolios(
    portfolios_path: Union[str, Path],
    asset_universe: Optional[pd.Index] = None
) -> pd.DataFrame:
    """
    Load precomputed portfolios from parquet file.
    
    Args:
        portfolios_path: Path to portfolios parquet file
        asset_universe: Optional asset universe to filter portfolios
        
    Returns:
        DataFrame of portfolio weights (portfolio_id x assets) or (portfolio_id, asset, weight) format
    """
    portfolios_path = Path(portfolios_path)
    
    # Handle relative paths
    if not portfolios_path.is_absolute() and not portfolios_path.exists():
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        portfolios_path = project_root / portfolios_path
    
    if not portfolios_path.exists():
        raise FileNotFoundError(f"Portfolios file not found: {portfolios_path}")
    
    portfolios = pd.read_parquet(portfolios_path)
    
    # Handle different formats:
    # 1. Wide format: portfolio_id as index, assets as columns
    # 2. Long format: portfolio_id, asset, weight columns
    
    if 'portfolio_id' in portfolios.columns and 'asset' in portfolios.columns:
        # Long format - convert to wide
        portfolios_wide = portfolios.pivot(index='portfolio_id', columns='asset', values='weight')
        portfolios_wide = portfolios_wide.fillna(0.0)
        portfolios = portfolios_wide
    elif 'portfolio_id' in portfolios.index.names or portfolios.index.name == 'portfolio_id':
        # Already wide format
        pass
    else:
        # Assume index is portfolio_id
        portfolios.index.name = 'portfolio_id'
    
    # Filter to asset universe if provided
    if asset_universe is not None:
        # Get intersection of assets
        common_assets = portfolios.columns.intersection(asset_universe)
        if len(common_assets) == 0:
            raise ValueError("No common assets between portfolios and asset universe")
        portfolios = portfolios[common_assets]
        
        # Add missing assets with zero weights
        missing_assets = asset_universe.difference(common_assets)
        for asset in missing_assets:
            portfolios[asset] = 0.0
        
        # Reorder columns to match asset_universe
        portfolios = portfolios[asset_universe]
    
    return portfolios
