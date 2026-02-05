"""
Returns computation module for QAOA Portfolio CVaR Optimization.

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


def generate_scenario_matrix(
    returns: pd.DataFrame,
    window: int,
    num_scenarios: Optional[int] = None
) -> np.ndarray:
    """
    Generate scenario matrix from historical returns.
    
    Args:
        returns: DataFrame of returns
        window: Rolling window size
        num_scenarios: Number of scenarios (None to use all window observations)
        
    Returns:
        Scenario matrix (num_scenarios x num_assets)
    """
    if len(returns) < window:
        window = len(returns)
    
    window_returns = returns.iloc[-window:]
    
    if num_scenarios is None:
        num_scenarios = len(window_returns)
    
    # Sample scenarios (with replacement if needed)
    if num_scenarios <= len(window_returns):
        scenario_indices = np.random.choice(
            len(window_returns),
            size=num_scenarios,
            replace=False
        )
    else:
        scenario_indices = np.random.choice(
            len(window_returns),
            size=num_scenarios,
            replace=True
        )
    
    scenario_matrix = window_returns.iloc[scenario_indices].values
    
    return scenario_matrix


def load_baseline_portfolios(baseline_portfolios_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load baseline portfolios from parquet or CSV file.
    
    Args:
        baseline_portfolios_path: Path to baseline portfolios file
        
    Returns:
        DataFrame with portfolio weights (portfolio_id x assets)
    """
    original_path = str(baseline_portfolios_path)
    baseline_portfolios_path = Path(baseline_portfolios_path)
    
    # Handle relative paths
    if baseline_portfolios_path.is_absolute() and baseline_portfolios_path.exists():
        pass
    elif baseline_portfolios_path.exists():
        pass
    else:
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        baseline_portfolios_path = project_root / original_path
        
        if not baseline_portfolios_path.exists():
            attempted_paths = [
                str(Path(original_path).resolve()),
                str(baseline_portfolios_path)
            ]
            raise FileNotFoundError(
                f"Baseline portfolios file not found. Attempted paths:\n"
                f"  1. {attempted_paths[0]}\n"
                f"  2. {attempted_paths[1]}\n"
                f"Please check that the file exists or update the path in llm.json"
            )
    
    if baseline_portfolios_path.suffix == '.parquet':
        portfolios = pd.read_parquet(baseline_portfolios_path)
    elif baseline_portfolios_path.suffix == '.csv':
        portfolios = pd.read_csv(baseline_portfolios_path, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {baseline_portfolios_path.suffix}")
    
    # Ensure index is named portfolio_id if not already
    if portfolios.index.name is None:
        portfolios.index.name = 'portfolio_id'
    
    return portfolios


def extract_asset_sets_from_portfolios(portfolios: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique asset sets from portfolios.
    
    Args:
        portfolios: DataFrame with portfolio weights (portfolio_id x assets)
        
    Returns:
        DataFrame mapping portfolio_id to asset_set (tuple of asset names)
    """
    asset_sets = []
    
    for portfolio_id, row in portfolios.iterrows():
        # Get assets with non-zero weights
        active_assets = row[row > 1e-10].index.tolist()
        asset_set = tuple(sorted(active_assets))
        asset_sets.append({
            'portfolio_id': portfolio_id,
            'asset_set': asset_set,
            'num_assets': len(active_assets)
        })
    
    asset_set_df = pd.DataFrame(asset_sets)
    return asset_set_df
