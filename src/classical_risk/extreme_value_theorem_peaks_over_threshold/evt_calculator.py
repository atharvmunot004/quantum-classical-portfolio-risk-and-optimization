"""
EVT-POT calculator for VaR and CVaR estimation using Extreme Value Theory.

Implements Peaks Over Threshold (POT) methodology with Generalized Pareto Distribution (GPD)
for asset-level EVT fitting with rolling windows.

Key features:
- Asset-level EVT parameter estimation
- Rolling window estimation for time series VaR/CVaR
- PWM (Probability Weighted Moments) method for GPD fitting
- Parameter caching for computational efficiency (per asset/window/quantile/time)
"""
import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple, Dict, Any, List
import warnings
from pathlib import Path
import pickle
import hashlib
from threading import Lock


def extract_exceedances(
    returns: pd.Series,
    threshold: float
) -> pd.Series:
    """
    Extract exceedances over threshold.
    
    Args:
        returns: Series of returns
        threshold: Threshold value (for losses, so threshold is positive)
        
    Returns:
        Series of exceedances (losses - threshold) where losses > threshold
    """
    losses = -returns
    exceedances = losses[losses > threshold] - threshold
    
    return exceedances


def fit_gpd_pwm(
    exceedances: pd.Series,
    xi_lower: float = -0.5,
    xi_upper: float = 0.5
) -> Tuple[float, float, Dict]:
    """
    Fit Generalized Pareto Distribution using Probability Weighted Moments (PWM).
    
    PWM method is more robust than MLE for small samples and provides
    better estimates for the GPD parameters.
    
    Args:
        exceedances: Series of exceedances (losses - threshold)
        xi_lower: Lower bound for shape parameter xi
        xi_upper: Upper bound for shape parameter xi
        
    Returns:
        Tuple of (xi (shape), beta (scale), diagnostics_dict)
    """
    if len(exceedances) == 0:
        return np.nan, np.nan, {'error': 'No exceedances'}
    
    exceedances_array = exceedances.values
    n = len(exceedances_array)
    
    if n < 3:
        # Fallback to simple method of moments
        mean_exceedance = exceedances_array.mean()
        var_exceedance = exceedances_array.var() if n > 1 else 0
        
        if var_exceedance <= 0 or mean_exceedance <= 0:
            xi_est = 0.0
            beta_est = mean_exceedance if mean_exceedance > 0 else 0.001
        else:
            xi_est = 0.5 * (1 - (mean_exceedance**2 / var_exceedance))
            beta_est = 0.5 * mean_exceedance * (1 + (mean_exceedance**2 / var_exceedance))
            xi_est = np.clip(xi_est, xi_lower + 0.01, xi_upper - 0.01)
            beta_est = max(beta_est, 0.001)
        
        return float(xi_est), float(beta_est), {
            'success': True,
            'method': 'method_of_moments',
            'num_exceedances': n
        }
    
    # Sort exceedances
    sorted_exceedances = np.sort(exceedances_array)
    
    # Compute probability weighted moments
    # M_0 = mean
    m0 = sorted_exceedances.mean()
    
    # M_1 = weighted mean with weights (n-i)/(n-1)
    weights = np.array([(n - i - 1) / (n - 1) for i in range(n)])
    m1 = np.mean(sorted_exceedances * weights)
    
    # M_2 = weighted mean with weights (n-i)(n-i-1)/((n-1)(n-2))
    if n > 2:
        weights2 = np.array([(n - i - 1) * (n - i - 2) / ((n - 1) * (n - 2)) for i in range(n)])
        m2 = np.mean(sorted_exceedances * weights2)
    else:
        m2 = m1
    
    # Estimate parameters using PWM
    # For GPD, the relationship is:
    # xi = (3*m1 - m0) / (m0 - m1) if m0 != m1
    # beta = (m0 * m1) / (m0 - m1) if m0 != m1
    
    if abs(m0 - m1) < 1e-10:
        # Fallback: use method of moments
        var_exceedance = sorted_exceedances.var()
        if var_exceedance > 0 and m0 > 0:
            xi_est = 0.5 * (1 - (m0**2 / var_exceedance))
            beta_est = 0.5 * m0 * (1 + (m0**2 / var_exceedance))
        else:
            xi_est = 0.0
            beta_est = max(m0, 0.001)
    else:
        # PWM estimates
        xi_est = (3 * m1 - m0) / (m0 - m1)
        beta_est = (m0 * m1) / (m0 - m1)
    
    # Apply constraints
    xi_est = np.clip(xi_est, xi_lower, xi_upper)
    beta_est = max(beta_est, 0.001)
    
    # Validate parameters
    if np.isnan(xi_est) or np.isnan(beta_est) or beta_est <= 0:
        # Fallback to simple estimates
        xi_est = 0.0
        beta_est = max(m0, 0.001)
    
    diagnostics = {
        'success': True,
        'xi': float(xi_est),
        'beta': float(beta_est),
        'method': 'pwm',
        'num_exceedances': n,
        'm0': float(m0),
        'm1': float(m1),
        'm2': float(m2)
    }
    
    return float(xi_est), float(beta_est), diagnostics


def select_threshold_quantile(
    returns: pd.Series,
    quantile: float = 0.95,
    min_exceedances: int = 50
) -> Tuple[float, int]:
    """
    Select threshold using quantile method.
    
    Args:
        returns: Series of returns
        quantile: Quantile level for threshold (e.g., 0.95 for 95th percentile)
        min_exceedances: Minimum number of exceedances required
        
    Returns:
        Tuple of (threshold, number_of_exceedances)
    """
    losses = -returns
    threshold = float(losses.quantile(quantile))
    
    # Count exceedances
    exceedances = losses[losses > threshold]
    num_exceedances = len(exceedances)
    
    # If not enough exceedances, try lower quantiles
    if num_exceedances < min_exceedances:
        for q in [0.90, 0.85, 0.80, 0.75, 0.70]:
            threshold = losses.quantile(q)
            exceedances = losses[losses > threshold]
            num_exceedances = len(exceedances)
            if num_exceedances >= min_exceedances:
                break
    
    return float(threshold), num_exceedances


def compute_asset_level_evt_parameters(
    asset_returns: pd.Series,
    estimation_window: int,
    threshold_quantile: float,
    min_exceedances: int = 50,
    xi_lower: float = -0.5,
    xi_upper: float = 0.5
) -> Dict[str, Any]:
    """
    Compute EVT parameters for a single asset using rolling window.
    
    This function fits EVT parameters at the asset level for a given
    estimation window and threshold quantile.
    
    Args:
        asset_returns: Series of asset returns
        estimation_window: Size of rolling window for estimation
        threshold_quantile: Quantile for threshold selection
        min_exceedances: Minimum number of exceedances required
        xi_lower: Lower bound for shape parameter
        xi_upper: Upper bound for shape parameter
        
    Returns:
        Dictionary with EVT parameters and metadata
    """
    if len(asset_returns) < estimation_window:
        return {
            'success': False,
            'error': 'Insufficient data',
            'xi': np.nan,
            'beta': np.nan,
            'threshold': np.nan,
            'num_exceedances': 0
        }
    
    # Use the most recent window
    window_returns = asset_returns.iloc[-estimation_window:]
    
    # Select threshold
    threshold, num_exceedances = select_threshold_quantile(
        window_returns,
        threshold_quantile,
        min_exceedances
    )
    
    if num_exceedances < min_exceedances:
        return {
            'success': False,
            'error': 'Insufficient exceedances',
            'xi': np.nan,
            'beta': np.nan,
            'threshold': threshold,
            'num_exceedances': num_exceedances
        }
    
    # Extract exceedances
    exceedances = extract_exceedances(window_returns, threshold)
    
    if len(exceedances) < min_exceedances:
        return {
            'success': False,
            'error': 'Insufficient exceedances after extraction',
            'xi': np.nan,
            'beta': np.nan,
            'threshold': threshold,
            'num_exceedances': len(exceedances)
        }
    
    # Fit GPD using PWM
    xi, beta, diagnostics = fit_gpd_pwm(
        exceedances,
        xi_lower,
        xi_upper
    )
    
    if np.isnan(xi) or np.isnan(beta) or beta <= 0:
        return {
            'success': False,
            'error': 'Invalid GPD parameters',
            'xi': xi,
            'beta': beta,
            'threshold': threshold,
            'num_exceedances': len(exceedances)
        }
    
    return {
        'success': True,
        'xi': float(xi),
        'beta': float(beta),
        'threshold': float(threshold),
        'num_exceedances': int(num_exceedances),
        'estimation_window': estimation_window,
        'threshold_quantile': threshold_quantile,
        **diagnostics
    }


def compute_var_from_evt(
    threshold: float,
    xi: float,
    beta: float,
    n: int,
    nu: int,
    confidence_level: float = 0.99,
    horizon: int = 1,
    scaling_rule: str = 'sqrt_time'
) -> float:
    """
    Compute VaR from EVT-POT using GPD parameters.
    
    Args:
        threshold: Threshold used for POT
        xi: GPD shape parameter
        beta: GPD scale parameter
        n: Total number of observations
        nu: Number of exceedances
        confidence_level: Confidence level (e.g., 0.99)
        horizon: Time horizon in days
        scaling_rule: Scaling rule for horizon ('sqrt_time' or 'linear')
        
    Returns:
        VaR value
    """
    if nu == 0 or n == 0:
        return np.nan
    
    # Probability of exceedance
    p_exceed = nu / n
    
    # Target probability for VaR
    p_target = 1 - confidence_level
    
    # Adjust for horizon
    if scaling_rule == 'sqrt_time':
        # Square root scaling
        p_target_horizon = 1 - (confidence_level ** (1.0 / np.sqrt(horizon)))
    elif scaling_rule == 'linear':
        # Linear scaling
        p_target_horizon = 1 - (confidence_level ** (1.0 / horizon))
    else:
        # Default to linear
        p_target_horizon = 1 - (confidence_level ** (1.0 / horizon))
    
    if p_target_horizon <= p_exceed:
        # VaR is above threshold
        if abs(xi) < 1e-8:  # Exponential case
            var = threshold + beta * np.log(p_exceed / p_target_horizon)
        else:
            var = threshold + (beta / xi) * (((p_exceed / p_target_horizon) ** (-xi)) - 1)
    else:
        # VaR is below threshold - use empirical approximation
        # This is a simplified case, in practice we'd use empirical quantile
        var = threshold * (p_target_horizon / p_exceed)
    
    return float(var)


def compute_cvar_from_evt(
    var_value: float,
    threshold: float,
    xi: float,
    beta: float
) -> float:
    """
    Compute CVaR (Expected Shortfall) from EVT-POT using GPD parameters.
    
    Args:
        var_value: VaR value (already computed)
        threshold: Threshold used for POT
        xi: GPD shape parameter
        beta: GPD scale parameter
        
    Returns:
        CVaR value
    """
    if np.isnan(var_value) or var_value <= threshold:
        # If VaR is below threshold, use simplified approximation
        return var_value + beta
    
    # CVaR for GPD
    if abs(xi) < 1e-8:  # Exponential case
        cvar = var_value + beta
    elif xi < 1:  # Finite mean case
        cvar = var_value + (beta - xi * (var_value - threshold)) / (1 - xi)
    else:
        # Infinite mean case
        cvar = np.inf
    
    return float(cvar)


class EVTParameterCache:
    """
    Cache for EVT parameters to avoid recomputation.
    
    Stores EVT parameters keyed by (asset, date, estimation_window, threshold_quantile)
    to support per-time-index caching for rolling windows.
    """
    
    def __init__(self, cache_path: Optional[Union[str, Path]] = None):
        """
        Initialize parameter cache.
        
        Args:
            cache_path: Optional path to save/load cache from disk (parquet format)
        """
        # Cache structure: (asset, date, estimation_window, threshold_quantile) -> parameters
        self.cache: Dict[Tuple[str, pd.Timestamp, int, float], Dict[str, Any]] = {}
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Load cache if it exists
        if self.cache_path and self.cache_path.exists():
            try:
                self.load_cache()
            except Exception as e:
                warnings.warn(f"Failed to load cache: {e}")
    
    def _make_key(self, asset: str, date: pd.Timestamp, estimation_window: int, threshold_quantile: float) -> Tuple[str, pd.Timestamp, int, float]:
        """Create cache key."""
        return (asset, date, estimation_window, threshold_quantile)
    
    def get(
        self,
        asset: str,
        date: Optional[pd.Timestamp],
        estimation_window: int,
        threshold_quantile: float
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached EVT parameters.
        
        Args:
            asset: Asset name
            date: Date for time-indexed cache (None for non-time-indexed)
            estimation_window: Estimation window size
            threshold_quantile: Threshold quantile
            
        Returns:
            Cached parameters or None if not found
        """
        # For backward compatibility, if date is None, try to find any cached entry
        if date is None:
            # Search for any entry with matching asset, window, quantile
            for key, value in self.cache.items():
                if key[0] == asset and key[2] == estimation_window and key[3] == threshold_quantile:
                    self.cache_hits += 1
                    return value
            self.cache_misses += 1
            return None
        
        key = self._make_key(asset, date, estimation_window, threshold_quantile)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        else:
            self.cache_misses += 1
            return None
    
    def set(
        self,
        asset: str,
        date: Optional[pd.Timestamp],
        estimation_window: int,
        threshold_quantile: float,
        parameters: Dict[str, Any]
    ):
        """
        Store EVT parameters in cache.
        
        Args:
            asset: Asset name
            date: Date for time-indexed cache (None for non-time-indexed)
            estimation_window: Estimation window size
            threshold_quantile: Threshold quantile
            parameters: EVT parameters dictionary
        """
        if date is None:
            # Use a dummy date for backward compatibility
            date = pd.Timestamp('2000-01-01')
        
        key = self._make_key(asset, date, estimation_window, threshold_quantile)
        self.cache[key] = parameters
    
    def save_cache(self):
        """Save cache to disk as parquet."""
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert cache to DataFrame for parquet storage
            cache_records = []
            for (asset, date, window, quantile), params in self.cache.items():
                record = {
                    'asset': asset,
                    'date': date,
                    'estimation_window': window,
                    'threshold_quantile': quantile,
                    **params
                }
                cache_records.append(record)
            
            if cache_records:
                cache_df = pd.DataFrame(cache_records)
                cache_df.to_parquet(self.cache_path, index=False)
    
    def load_cache(self):
        """Load cache from disk (parquet format)."""
        if self.cache_path and self.cache_path.exists():
            try:
                cache_df = pd.read_parquet(self.cache_path)
                cache_df['date'] = pd.to_datetime(cache_df['date'])
                
                for _, row in cache_df.iterrows():
                    key = (row['asset'], row['date'], row['estimation_window'], row['threshold_quantile'])
                    params = row.drop(['asset', 'date', 'estimation_window', 'threshold_quantile']).to_dict()
                    self.cache[key] = params
            except Exception as e:
                warnings.warn(f"Failed to load cache from parquet: {e}. Cache will be empty.")
                self.cache = {}
    
    def get_hit_ratio(self) -> float:
        """Get cache hit ratio."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total
    
    def clear(self):
        """Clear cache."""
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0


def compute_all_asset_evt_parameters(
    daily_returns: pd.DataFrame,
    estimation_windows: List[int],
    threshold_quantiles: List[float],
    min_exceedances: int = 50,
    xi_lower: float = -0.5,
    xi_upper: float = 0.5,
    cache: Optional[EVTParameterCache] = None,
    n_jobs: Optional[int] = None
) -> Dict[Tuple[str, int, float], Dict[str, Any]]:
    """
    Compute EVT parameters for all assets, estimation windows, and threshold quantiles.
    
    This is the core function that implements asset-level EVT fitting as specified
    in the computation strategy.
    
    Args:
        daily_returns: DataFrame of daily returns (dates x assets)
        estimation_windows: List of estimation window sizes
        threshold_quantiles: List of threshold quantiles
        min_exceedances: Minimum number of exceedances
        xi_lower: Lower bound for shape parameter
        xi_upper: Upper bound for shape parameter
        cache: Optional parameter cache
        n_jobs: Number of parallel workers (not implemented yet)
        
    Returns:
        Dictionary mapping (asset, estimation_window, threshold_quantile) to parameters
    """
    all_parameters = {}
    
    for asset in daily_returns.columns:
        asset_returns = daily_returns[asset].dropna()
        
        for window in estimation_windows:
            for quantile in threshold_quantiles:
                # Check cache first
                if cache:
                    cached = cache.get(asset, window, quantile)
                    if cached:
                        all_parameters[(asset, window, quantile)] = cached
                        continue
                
                # Compute EVT parameters
                params = compute_asset_level_evt_parameters(
                    asset_returns,
                    window,
                    quantile,
                    min_exceedances,
                    xi_lower,
                    xi_upper
                )
                
                all_parameters[(asset, window, quantile)] = params
                
                # Store in cache
                if cache:
                    cache.set(asset, window, quantile, params)
    
    return all_parameters


def compute_rolling_asset_evt_parameters(
    asset_returns: pd.Series,
    estimation_window: int,
    threshold_quantile: float,
    confidence_levels: List[float],
    horizons: List[int],
    scaling_rule: str,
    min_exceedances: int,
    xi_lower: float,
    xi_upper: float,
    gpd_fitting_method: str,
    return_type: str,
    tail_side: str,
    step_size: int = 1,
    cache: Optional[EVTParameterCache] = None,
    cache_lock: Optional[Lock] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[Dict]]:
    """
    Compute rolling EVT parameters and VaR/CVaR time series for a single asset.
    
    This function implements rolling window estimation where EVT parameters are
    fitted for each window position, producing time series of VaR and CVaR values.
    
    Args:
        asset_returns: Series of asset returns with dates as index
        estimation_window: Size of rolling estimation window
        threshold_quantile: Threshold quantile for POT
        confidence_levels: List of confidence levels
        horizons: List of horizons
        scaling_rule: Scaling rule for horizon
        min_exceedances: Minimum number of exceedances
        xi_lower: Lower bound for shape parameter
        xi_upper: Upper bound for shape parameter
        gpd_fitting_method: GPD fitting method ('pwm' or 'mle')
        return_type: Return type ('log' or 'simple')
        tail_side: Tail side ('left' or 'right')
        step_size: Step size for rolling window (default: 1)
        cache: Optional parameter cache
        cache_lock: Optional lock for thread-safe cache access
        
    Returns:
        Tuple of (var_series_df, cvar_series_df, evt_params_list)
        - var_series_df: DataFrame with columns for each conf_level/horizon combination
        - cvar_series_df: DataFrame with columns for each conf_level/horizon combination
        - evt_params_list: List of EVT parameter dictionaries for each window
    """
    if len(asset_returns) < estimation_window:
        return None, None, []
    
    # Prepare data based on tail_side
    if tail_side == 'left':
        # Focus on left tail (losses)
        working_returns = asset_returns.copy()
    else:
        # Focus on right tail (gains) - negate returns
        working_returns = -asset_returns.copy()
    
    # Initialize output structures
    var_series_dict = {}
    cvar_series_dict = {}
    evt_params_list = []
    dates_list = []
    
    # Rolling window loop
    for start_idx in range(0, len(asset_returns) - estimation_window + 1, step_size):
        end_idx = start_idx + estimation_window
        window_returns = working_returns.iloc[start_idx:end_idx]
        window_date = asset_returns.index[end_idx - 1]  # Use end date of window
        
        # Check cache
        cached_params = None
        if cache:
            if cache_lock:
                cache_lock.acquire()
            try:
                cached_params = cache.get(asset_returns.name, window_date, estimation_window, threshold_quantile)
            finally:
                if cache_lock:
                    cache_lock.release()
        
        if cached_params and cached_params.get('success', False):
            params = cached_params
        else:
            # Compute EVT parameters for this window
            params = compute_asset_level_evt_parameters(
                window_returns,
                estimation_window,
                threshold_quantile,
                min_exceedances,
                xi_lower,
                xi_upper
            )
            
            # Store in cache
            if cache and params.get('success', False):
                if cache_lock:
                    cache_lock.acquire()
                try:
                    cache.set(asset_returns.name, window_date, estimation_window, threshold_quantile, params)
                finally:
                    if cache_lock:
                        cache_lock.release()
        
        if not params.get('success', False):
            # Skip this window if fitting failed
            continue
        
        # Store EVT parameters
        evt_params_list.append({
            'date': window_date,
            'success': params.get('success', False),
            'xi': params.get('xi', np.nan),
            'beta': params.get('beta', np.nan),
            'threshold': params.get('threshold', np.nan),
            'num_exceedances': params.get('num_exceedances', 0)
        })
        
        # Compute VaR and CVaR for each confidence level and horizon
        n = len(window_returns)
        nu = params.get('num_exceedances', 0)
        threshold = params.get('threshold', np.nan)
        xi = params.get('xi', np.nan)
        beta = params.get('beta', np.nan)
        
        for conf_level in confidence_levels:
            for horizon in horizons:
                # Compute VaR
                var_value = compute_var_from_evt(
                    threshold, xi, beta, n, nu,
                    conf_level, horizon, scaling_rule
                )
                
                # Compute CVaR
                cvar_value = compute_cvar_from_evt(var_value, threshold, xi, beta)
                
                # Store in series dictionaries
                var_key = f"var_{conf_level}_{horizon}"
                cvar_key = f"cvar_{conf_level}_{horizon}"
                
                if var_key not in var_series_dict:
                    var_series_dict[var_key] = []
                    cvar_series_dict[cvar_key] = []
                
                var_series_dict[var_key].append(var_value)
                cvar_series_dict[cvar_key].append(cvar_value)
        
        dates_list.append(window_date)
    
    if len(dates_list) == 0:
        return None, None, []
    
    # Create DataFrames
    var_series_df = pd.DataFrame(var_series_dict, index=pd.DatetimeIndex(dates_list))
    cvar_series_df = pd.DataFrame(cvar_series_dict, index=pd.DatetimeIndex(dates_list))
    
    return var_series_df, cvar_series_df, evt_params_list
