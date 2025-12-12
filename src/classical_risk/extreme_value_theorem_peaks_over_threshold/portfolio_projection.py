"""
Portfolio tail projection module for EVT-POT.

Implements portfolio-level VaR and CVaR projection from asset-level EVT parameters
using weighted tail expectation aggregation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import warnings

# Import compute_var_from_evt and compute_cvar_from_evt from evt_calculator
from .evt_calculator import compute_var_from_evt, compute_cvar_from_evt


def project_portfolio_var_cvar(
    portfolio_weights: pd.Series,
    asset_evt_parameters: Dict[Tuple[str, int, float], Dict[str, Any]],
    asset_returns: pd.DataFrame,
    estimation_window: int,
    threshold_quantile: float,
    confidence_level: float = 0.99,
    horizon: int = 1,
    scaling_rule: str = 'sqrt_time',
    aggregation_method: str = 'weighted_tail_expectation',
    assume_tail_dependence: bool = False
) -> Tuple[float, float, Dict]:
    """
    Project portfolio VaR and CVaR from asset-level EVT parameters.
    
    This function aggregates asset-level EVT parameters to portfolio-level
    risk measures using weighted tail expectation.
    
    Args:
        portfolio_weights: Series of portfolio weights (asset -> weight)
        asset_evt_parameters: Dictionary mapping (asset, window, quantile) to EVT params
        asset_returns: DataFrame of asset returns (for computing exceedance counts)
        estimation_window: Estimation window size
        threshold_quantile: Threshold quantile
        confidence_level: Confidence level for VaR/CVaR
        horizon: Time horizon in days
        scaling_rule: Scaling rule for horizon
        aggregation_method: Aggregation method ('weighted_tail_expectation')
        assume_tail_dependence: Whether to assume tail dependence (not implemented)
        
    Returns:
        Tuple of (portfolio_var, portfolio_cvar, diagnostics_dict)
    """
    # Get common assets
    common_assets = portfolio_weights.index.intersection(asset_returns.columns)
    if len(common_assets) == 0:
        return np.nan, np.nan, {'error': 'No common assets'}
    
    portfolio_weights = portfolio_weights[common_assets]
    portfolio_weights = portfolio_weights / portfolio_weights.sum()  # Normalize
    
    # Collect asset-level VaR and CVaR
    asset_vars = {}
    asset_cvars = {}
    asset_params = {}
    
    for asset in common_assets:
        key = (asset, estimation_window, threshold_quantile)
        if key not in asset_evt_parameters:
            continue
        
        params = asset_evt_parameters[key]
        if not params.get('success', False):
            continue
        
        # Get asset returns for this window
        asset_ret = asset_returns[asset].dropna()
        if len(asset_ret) < estimation_window:
            continue
        
        window_returns = asset_ret.iloc[-estimation_window:]
        n = len(window_returns)
        
        # Get EVT parameters
        threshold = params['threshold']
        xi = params['xi']
        beta = params['beta']
        nu = params['num_exceedances']
        
        # Compute asset-level VaR
        asset_var = compute_var_from_evt(
            threshold, xi, beta, n, nu,
            confidence_level, horizon, scaling_rule
        )
        
        if np.isnan(asset_var):
            continue
        
        # Compute asset-level CVaR
        asset_cvar = compute_cvar_from_evt(asset_var, threshold, xi, beta)
        
        asset_vars[asset] = asset_var
        asset_cvars[asset] = asset_cvar
        asset_params[asset] = params
    
    if len(asset_vars) == 0:
        return np.nan, np.nan, {'error': 'No valid asset EVT parameters'}
    
    # Aggregate to portfolio level
    if aggregation_method == 'weighted_tail_expectation':
        portfolio_var, portfolio_cvar = _weighted_tail_expectation(
            portfolio_weights,
            asset_vars,
            asset_cvars,
            asset_params,
            confidence_level
        )
    else:
        # Fallback: simple weighted average
        portfolio_var = np.sum([portfolio_weights[asset] * asset_vars[asset] 
                                for asset in asset_vars.keys()])
        portfolio_cvar = np.sum([portfolio_weights[asset] * asset_cvars[asset] 
                                 for asset in asset_cvars.keys()])
    
    diagnostics = {
        'num_assets_used': len(asset_vars),
        'aggregation_method': aggregation_method,
        'assume_tail_dependence': assume_tail_dependence
    }
    
    return float(portfolio_var), float(portfolio_cvar), diagnostics


def _weighted_tail_expectation(
    portfolio_weights: pd.Series,
    asset_vars: Dict[str, float],
    asset_cvars: Dict[str, float],
    asset_params: Dict[str, Dict],
    confidence_level: float
) -> Tuple[float, float]:
    """
    Compute portfolio VaR/CVaR using weighted tail expectation.
    
    This method weights asset-level tail expectations by portfolio weights,
    accounting for the fact that portfolio tail risk is a weighted combination
    of asset tail risks.
    
    Args:
        portfolio_weights: Series of portfolio weights
        asset_vars: Dictionary of asset VaR values
        asset_cvars: Dictionary of asset CVaR values
        asset_params: Dictionary of asset EVT parameters
        confidence_level: Confidence level
        
    Returns:
        Tuple of (portfolio_var, portfolio_cvar)
    """
    # For VaR: use weighted average of asset VaRs
    # This is a simplified approach; more sophisticated methods would
    # account for tail dependence
    portfolio_var = np.sum([
        portfolio_weights[asset] * asset_vars[asset]
        for asset in asset_vars.keys()
    ])
    
    # For CVaR: use weighted average of asset CVaRs
    # This assumes tail independence (can be extended for dependence)
    portfolio_cvar = np.sum([
        portfolio_weights[asset] * asset_cvars[asset]
        for asset in asset_cvars.keys()
    ])
    
    return float(portfolio_var), float(portfolio_cvar)


def compute_rolling_portfolio_var_cvar(
    portfolio_weights: pd.Series,
    asset_evt_parameters: Dict[Tuple[str, int, float], Dict[str, Any]],
    asset_returns: pd.DataFrame,
    estimation_window: int,
    threshold_quantile: float,
    confidence_level: float = 0.99,
    horizon: int = 1,
    scaling_rule: str = 'sqrt_time',
    aggregation_method: str = 'weighted_tail_expectation'
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute rolling portfolio VaR and CVaR series.
    
    This function computes a time series of portfolio VaR and CVaR values
    by projecting asset-level EVT parameters at each time point.
    
    Args:
        portfolio_weights: Series of portfolio weights (constant over time)
        asset_evt_parameters: Dictionary of asset EVT parameters
        asset_returns: DataFrame of asset returns
        estimation_window: Estimation window size
        threshold_quantile: Threshold quantile
        confidence_level: Confidence level
        horizon: Time horizon
        scaling_rule: Scaling rule
        aggregation_method: Aggregation method
        
    Returns:
        Tuple of (portfolio_var_series, portfolio_cvar_series)
    """
    # For now, compute a single VaR/CVaR value using the most recent window
    # In a full implementation, this would roll through time
    
    portfolio_var, portfolio_cvar, _ = project_portfolio_var_cvar(
        portfolio_weights,
        asset_evt_parameters,
        asset_returns,
        estimation_window,
        threshold_quantile,
        confidence_level,
        horizon,
        scaling_rule,
        aggregation_method
    )
    
    # Create series with single value (or extend to rolling window)
    var_series = pd.Series(
        [portfolio_var] * len(asset_returns.index),
        index=asset_returns.index
    )
    cvar_series = pd.Series(
        [portfolio_cvar] * len(asset_returns.index),
        index=asset_returns.index
    )
    
    return var_series, cvar_series

