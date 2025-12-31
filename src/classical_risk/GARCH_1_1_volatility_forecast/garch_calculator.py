"""
GARCH model calculator for volatility forecasting and VaR/CVaR estimation.

Implements GARCH(1,1) model fitting at asset level with parameter caching,
conditional volatility computation, and portfolio volatility projection
via variance aggregation.

GPU-accelerated using PyTorch for improved performance on large datasets.
"""
import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple, Dict, Any, List
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
import time
import pickle

try:
    import torch
    TORCH_AVAILABLE = True
    # Set device: use GPU if available, otherwise CPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        # Set memory fraction to avoid OOM errors
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)
        except:
            pass
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    warnings.warn("PyTorch not available. Falling back to CPU-only computation.")

# Suppress convergence warnings from GARCH model fitting (handled by fallback mechanism)
# These warnings occur when GARCH fitting encounters numerical issues, but our fallback handles these cases
# Filter by message content since ConvergenceWarning may be a UserWarning
warnings.filterwarnings('ignore', message='.*optimizer returned code.*')
warnings.filterwarnings('ignore', message='.*Inequality constraints incompatible.*')
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
warnings.filterwarnings('ignore', message='.*See scipy.optimize.fmin_slsqp.*')
# Try to import and suppress ConvergenceWarning category if available
try:
    from arch.univariate.base import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
except (ImportError, AttributeError):
    pass
# Suppress UserWarnings from arch module (ConvergenceWarning is often a UserWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='arch')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='arch')

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn("arch library not available. GARCH functionality will not work. Install with: pip install arch")


def _to_tensor(data: Union[pd.Series, np.ndarray, Any], dtype=None) -> Any:
    """
    Convert pandas Series, numpy array, or tensor to PyTorch tensor on GPU.
    
    Args:
        data: Input data (Series, array, or tensor)
        dtype: Target dtype (default: float32 for memory efficiency)
        
    Returns:
        PyTorch tensor on appropriate device, or numpy array if PyTorch unavailable
    """
    if not TORCH_AVAILABLE:
        if isinstance(data, pd.Series):
            return data.values
        return np.asarray(data)
    
    if dtype is None:
        dtype = torch.float32
    
    if isinstance(data, torch.Tensor):
        return data.to(device=DEVICE, dtype=dtype)
    elif isinstance(data, pd.Series):
        return torch.tensor(data.values, device=DEVICE, dtype=dtype)
    else:
        return torch.tensor(data, device=DEVICE, dtype=dtype)


def _to_numpy(tensor: Union[torch.Tensor, np.ndarray, pd.Series]) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array (moving to CPU if needed).
    
    Args:
        tensor: PyTorch tensor, numpy array, or pandas Series
        
    Returns:
        Numpy array
    """
    if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, pd.Series):
        return tensor.values
    else:
        return np.asarray(tensor)


class GARCHParameterCache:
    """
    Cache for GARCH parameters to avoid recomputation.
    
    Stores GARCH parameters and conditional volatility series keyed by (asset, estimation_window).
    """
    
    def __init__(self, cache_path: Optional[Union[str, Path]] = None):
        """
        Initialize parameter cache.
        
        Args:
            cache_path: Optional path to save/load cache from disk
        """
        self.cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Load cache if it exists
        if self.cache_path and self.cache_path.exists():
            try:
                self.load_cache()
            except Exception as e:
                warnings.warn(f"Failed to load cache: {e}")
    
    def _make_key(self, asset: str, estimation_window: int) -> Tuple[str, int]:
        """Create cache key."""
        return (asset, estimation_window)
    
    def get(
        self,
        asset: str,
        estimation_window: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached GARCH parameters and conditional volatility.
        
        Args:
            asset: Asset name
            estimation_window: Estimation window size
            
        Returns:
            Cached parameters or None if not found
        """
        key = self._make_key(asset, estimation_window)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        else:
            self.cache_misses += 1
            return None
    
    def set(
        self,
        asset: str,
        estimation_window: int,
        parameters: Dict[str, Any]
    ):
        """
        Store GARCH parameters and conditional volatility in cache.
        
        Args:
            asset: Asset name
            estimation_window: Estimation window size
            parameters: Dictionary with 'conditional_volatility' Series and optionally 'fit_result'
        """
        key = self._make_key(asset, estimation_window)
        self.cache[key] = parameters.copy()
    
    def save_cache(self):
        """Save cache to disk (parquet format for conditional volatility)."""
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # If path ends with .parquet, save as parquet
            if self.cache_path.suffix == '.parquet':
                # Save conditional volatilities as parquet
                cache_data = []
                for (asset, window), value in self.cache.items():
                    cond_vol = value.get('conditional_volatility')
                    if cond_vol is not None and isinstance(cond_vol, pd.Series):
                        for date, vol in cond_vol.items():
                            cache_data.append({
                                'asset': asset,
                                'estimation_window': window,
                                'date': date,
                                'conditional_volatility': vol
                            })
                
                if len(cache_data) > 0:
                    cache_df = pd.DataFrame(cache_data)
                    cache_df.to_parquet(self.cache_path, index=False)
            else:
                # Fallback to pickle for backward compatibility
                cache_to_save = {}
                for key, value in self.cache.items():
                    cache_to_save[key] = {
                        'conditional_volatility': value.get('conditional_volatility'),
                        'fit_result_params': value.get('fit_result_params'),
                        'long_run_variance': value.get('long_run_variance')
                    }
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(cache_to_save, f)
    
    def load_cache(self):
        """Load cache from disk (supports both parquet and pickle)."""
        if self.cache_path and self.cache_path.exists():
            if self.cache_path.suffix == '.parquet':
                # Load from parquet
                try:
                    cache_df = pd.read_parquet(self.cache_path)
                    # Reconstruct cache dictionary
                    for (asset, window), group in cache_df.groupby(['asset', 'estimation_window']):
                        cond_vol = pd.Series(
                            group['conditional_volatility'].values,
                            index=group['date'].values,
                            name='conditional_volatility'
                        )
                        self.cache[(asset, window)] = {
                            'conditional_volatility': cond_vol
                        }
                except Exception as e:
                    warnings.warn(f"Failed to load parquet cache: {e}")
            else:
                # Load from pickle
                try:
                    with open(self.cache_path, 'rb') as f:
                        cache_loaded = pickle.load(f)
                    self.cache = cache_loaded
                except Exception as e:
                    warnings.warn(f"Failed to load pickle cache: {e}")
    
    def get_hit_ratio(self) -> float:
        """Get cache hit ratio."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


def fit_garch_model(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    rescale: bool = True
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Fit a GARCH(p,q) model to return series.
    
    Args:
        returns: Series of returns
        p: GARCH order (number of lagged conditional variances)
        q: ARCH order (number of lagged squared residuals)
        dist: Distribution assumption ('normal', 't', 'ged')
        mean: Mean model ('Zero', 'AR', 'Constant')
        vol: Volatility model ('GARCH', 'EGARCH', etc.)
        rescale: Whether to rescale data
        
    Returns:
        Tuple of (fitted model, fit results)
    """
    if not ARCH_AVAILABLE:
        raise ImportError("arch library is required. Install with: pip install arch")
    
    if len(returns.dropna()) < max(p, q) + 10:
        return None, None
    
    try:
        # Fit GARCH model
        model = arch_model(
            returns.dropna(),
            vol=vol,
            p=p,
            q=q,
            dist=dist,
            mean=mean,
            rescale=rescale
        )
        
        # Suppress ConvergenceWarnings during fitting
        # Use a comprehensive warning suppression context
        with warnings.catch_warnings(record=False):
            warnings.simplefilter('ignore')
            # Also temporarily redirect stderr to suppress warnings printed directly by arch library
            import sys
            import io
            old_stderr = sys.stderr
            try:
                # Redirect stderr to a null stream during fitting
                sys.stderr = io.StringIO()
                # Fit with display='off' to suppress output
                fit_result = model.fit(update_freq=0, disp='off')
            finally:
                # Restore stderr
                sys.stderr = old_stderr
        
        return model, fit_result
    except Exception as e:
        # Only warn for actual exceptions, not convergence issues
        return None, None


def compute_conditional_volatility(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True
) -> pd.Series:
    """
    Compute conditional volatility from GARCH model fitted to entire series (GPU-accelerated fallback).
    
    Args:
        returns: Series of returns
        p: GARCH order
        q: ARCH order
        dist: Distribution assumption
        mean: Mean model
        vol: Volatility model
        fallback_long_run_variance: If True, use long-run variance if model fails
        
    Returns:
        Series of conditional volatility estimates
    """
    if not ARCH_AVAILABLE:
        raise ImportError("arch library is required. Install with: pip install arch")
    
    model, fit_result = fit_garch_model(returns, p, q, dist, mean, vol)
    
    if fit_result is None:
        if fallback_long_run_variance:
            # Use long-run variance as fallback (GPU-accelerated if available)
            if TORCH_AVAILABLE:
                returns_tensor = _to_tensor(returns)
                long_run_vol = torch.std(returns_tensor).item()
            else:
                long_run_vol = returns.std()
            return pd.Series(long_run_vol, index=returns.index, name='conditional_volatility')
        else:
            return pd.Series(np.nan, index=returns.index, name='conditional_volatility')
    
    # Get conditional volatility from fitted model
    conditional_vol = fit_result.conditional_volatility
    
    # Align with original returns index
    conditional_vol_series = pd.Series(
        conditional_vol,
        index=returns.dropna().index,
        name='conditional_volatility'
    )
    
    return conditional_vol_series


def forecast_volatility(
    returns: pd.Series,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True,
    use_gpu_for_fallback: bool = False
) -> float:
    """
    Forecast volatility for a given horizon using GARCH model (GPU-accelerated fallback).
    
    Args:
        returns: Series of historical returns
        horizon: Forecast horizon (in periods)
        p: GARCH order
        q: ARCH order
        dist: Distribution assumption
        mean: Mean model
        vol: Volatility model
        fallback_long_run_variance: If True, use long-run variance if model fails
        use_gpu_for_fallback: If True, use GPU for fallback std calculation
        
    Returns:
        Forecasted volatility (annualized if returns are daily)
    """
    if not ARCH_AVAILABLE:
        raise ImportError("arch library is required. Install with: pip install arch")
    
    model, fit_result = fit_garch_model(returns, p, q, dist, mean, vol)
    
    if fit_result is None:
        if fallback_long_run_variance:
            # Use long-run variance as fallback (GPU-accelerated if available)
            if use_gpu_for_fallback and TORCH_AVAILABLE:
                returns_tensor = _to_tensor(returns)
                std_val = torch.std(returns_tensor).item() * np.sqrt(horizon)
                return float(std_val)
            else:
                return returns.std() * np.sqrt(horizon)
        else:
            return np.nan
    
    try:
        # Suppress warnings during forecast
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Forecast volatility
            forecast = fit_result.forecast(horizon=horizon, reindex=False)
        
        # Get variance forecast
        variance_forecast = forecast.variance.iloc[-1, -1]  # Last forecast
        
        # Convert to volatility (standard deviation)
        vol_forecast = np.sqrt(variance_forecast) * np.sqrt(horizon)
        
        return float(vol_forecast)
    except Exception as e:
        # Silently fall back on forecast failure (GPU-accelerated if available)
        if fallback_long_run_variance:
            if use_gpu_for_fallback and TORCH_AVAILABLE:
                returns_tensor = _to_tensor(returns)
                std_val = torch.std(returns_tensor).item() * np.sqrt(horizon)
                return float(std_val)
            else:
                return returns.std() * np.sqrt(horizon)
        else:
            return np.nan


def compute_rolling_volatility_forecast(
    returns: pd.Series,
    window: int,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True,
    batch_size: int = 50
) -> pd.Series:
    """
    Compute rolling volatility forecasts using GARCH models (GPU-accelerated where applicable).
    
    For each date, fits GARCH to a rolling window of historical returns
    and forecasts volatility for the next horizon periods.
    
    Args:
        returns: Series of returns with dates as index
        window: Rolling window size for model fitting
        horizon: Forecast horizon (in periods)
        p: GARCH order
        q: ARCH order
        dist: Distribution assumption
        mean: Mean model
        vol: Volatility model
        fallback_long_run_variance: If True, use long-run variance if model fails
        batch_size: Number of windows to process before clearing GPU cache (for memory management)
        
    Returns:
        Series of volatility forecasts with dates as index
    """
    if not ARCH_AVAILABLE:
        raise ImportError("arch library is required. Install with: pip install arch")
    
    # Convert entire returns series to GPU tensor once (if available) for faster slicing
    if TORCH_AVAILABLE:
        returns_tensor_full = _to_tensor(returns)
        use_gpu = isinstance(returns_tensor_full, torch.Tensor) and returns_tensor_full.is_cuda
    else:
        use_gpu = False
    
    forecasts = []
    forecast_dates = []
    
    # Need at least window + some buffer for GARCH fitting
    min_window = max(window, max(p, q) + 10)
    
    for i in range(min_window, len(returns)):
        # Get rolling window of returns
        window_returns = returns.iloc[i - window:i]
        
        # Forecast volatility (GARCH fitting is still CPU-bound, but fallback std calc can use GPU)
        vol_forecast = forecast_volatility(
            window_returns,
            horizon=horizon,
            p=p,
            q=q,
            dist=dist,
            mean=mean,
            vol=vol,
            fallback_long_run_variance=fallback_long_run_variance,
            use_gpu_for_fallback=use_gpu
        )
        
        # Forecast date is the date for which we're forecasting (current date)
        forecast_date = returns.index[i]
        
        forecasts.append(vol_forecast)
        forecast_dates.append(forecast_date)
        
        # Clear GPU cache periodically to manage memory
        if use_gpu and (i - min_window) % batch_size == 0:
            torch.cuda.empty_cache()
    
    # Create Series with forecasts
    forecast_series = pd.Series(
        forecasts,
        index=forecast_dates,
        name='volatility_forecast'
    )
    
    return forecast_series


def var_from_volatility(
    volatility: Union[pd.Series, float],
    confidence_level: float = 0.95,
    horizon: int = 1,
    dist: str = 'normal',
    use_gpu: bool = True
) -> Union[pd.Series, float]:
    """
    Compute VaR from volatility forecast using distributional assumption (GPU-accelerated).
    
    VaR = volatility * z_score(confidence_level) * sqrt(horizon)
    
    Args:
        volatility: Volatility forecast (Series or scalar)
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        horizon: Time horizon (e.g., 1 for 1 day)
        dist: Distribution assumption ('normal', 't')
        use_gpu: If True, use GPU acceleration when available
        
    Returns:
        VaR estimate (Series or scalar, same type as volatility)
    """
    from scipy import stats
    
    # Z-score for confidence level (computed once, cached on CPU)
    alpha = 1 - confidence_level
    
    if dist == 'normal':
        z_score = stats.norm.ppf(confidence_level)
    elif dist == 't':
        # Use t-distribution with 4 degrees of freedom (common choice)
        z_score = stats.t.ppf(confidence_level, df=4)
    else:
        # Default to normal
        z_score = stats.norm.ppf(confidence_level)
    
    # Check if GPU should be used
    use_gpu_accel = use_gpu and TORCH_AVAILABLE and DEVICE is not None and DEVICE.type == 'cuda'
    
    # Convert z_score to tensor for GPU operations (if available)
    horizon_sqrt = np.sqrt(horizon)
    if use_gpu_accel:
        z_score_tensor = torch.tensor(z_score, device=DEVICE, dtype=torch.float32)
        horizon_tensor = torch.tensor(horizon_sqrt, device=DEVICE, dtype=torch.float32)
    else:
        z_score_tensor = z_score
        horizon_tensor = horizon_sqrt
    
    # Handle both Series and scalar
    is_scalar = isinstance(volatility, (int, float)) or (isinstance(volatility, np.ndarray) and volatility.ndim == 0)
    
    if is_scalar:
        # Scalar case
        vol_scalar = float(volatility)
        if use_gpu_accel:
            vol_tensor = torch.tensor(vol_scalar, device=DEVICE, dtype=torch.float32)
            var_tensor = vol_tensor * z_score_tensor * horizon_tensor
            return float(var_tensor.item())
        else:
            return vol_scalar * z_score * horizon_sqrt
    else:
        # Series case - use GPU if available
        if use_gpu_accel and isinstance(volatility, pd.Series):
            vol_tensor = _to_tensor(volatility)
            var_tensor = vol_tensor * z_score_tensor * horizon_tensor
            var_array = _to_numpy(var_tensor)
            return pd.Series(var_array, index=volatility.index, name='VaR')
        else:
            # CPU fallback
            var = volatility * z_score * horizon_sqrt
            if isinstance(volatility, pd.Series):
                return pd.Series(var, index=volatility.index, name='VaR')
            return var


def cvar_from_volatility(
    volatility: Union[pd.Series, float],
    confidence_level: float = 0.95,
    horizon: int = 1,
    dist: str = 'normal',
    use_gpu: bool = True
) -> Union[pd.Series, float]:
    """
    Compute CVaR from volatility forecast using distributional assumption (GPU-accelerated).
    
    CVaR = volatility * expected_shortfall_factor * sqrt(horizon)
    
    Args:
        volatility: Volatility forecast (Series or scalar)
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        horizon: Time horizon (e.g., 1 for 1 day)
        dist: Distribution assumption ('normal', 't')
        use_gpu: If True, use GPU acceleration when available
        
    Returns:
        CVaR estimate (Series or scalar, same type as volatility)
    """
    from scipy import stats
    
    # Expected shortfall (ES) for normal distribution (computed once on CPU)
    alpha = 1 - confidence_level
    
    if dist == 'normal':
        # For normal: ES = -sigma * phi(z_alpha) / alpha
        z_alpha = stats.norm.ppf(confidence_level)
        phi_z_alpha = stats.norm.pdf(z_alpha)
        es_factor = phi_z_alpha / alpha
    elif dist == 't':
        # For t-distribution with 4 df
        z_alpha = stats.t.ppf(confidence_level, df=4)
        # Approximate ES for t-distribution
        es_factor = stats.t.pdf(z_alpha, df=4) / alpha * (4 + z_alpha**2) / (4 - 1)
    else:
        # Default to normal
        z_alpha = stats.norm.ppf(confidence_level)
        phi_z_alpha = stats.norm.pdf(z_alpha)
        es_factor = phi_z_alpha / alpha
    
    # Check if GPU should be used
    use_gpu_accel = use_gpu and TORCH_AVAILABLE and DEVICE is not None and DEVICE.type == 'cuda'
    
    # Convert es_factor to tensor for GPU operations (if available)
    horizon_sqrt = np.sqrt(horizon)
    if use_gpu_accel:
        es_factor_tensor = torch.tensor(es_factor, device=DEVICE, dtype=torch.float32)
        horizon_tensor = torch.tensor(horizon_sqrt, device=DEVICE, dtype=torch.float32)
    else:
        es_factor_tensor = es_factor
        horizon_tensor = horizon_sqrt
    
    # Handle both Series and scalar
    is_scalar = isinstance(volatility, (int, float)) or (isinstance(volatility, np.ndarray) and volatility.ndim == 0)
    
    if is_scalar:
        # Scalar case
        vol_scalar = float(volatility)
        if use_gpu_accel:
            vol_tensor = torch.tensor(vol_scalar, device=DEVICE, dtype=torch.float32)
            cvar_tensor = vol_tensor * es_factor_tensor * horizon_tensor
            return float(cvar_tensor.item())
        else:
            return vol_scalar * es_factor * horizon_sqrt
    else:
        # Series case - use GPU if available
        if use_gpu_accel and isinstance(volatility, pd.Series):
            vol_tensor = _to_tensor(volatility)
            cvar_tensor = vol_tensor * es_factor_tensor * horizon_tensor
            cvar_array = _to_numpy(cvar_tensor)
            return pd.Series(cvar_array, index=volatility.index, name='CVaR')
        else:
            # CPU fallback
            cvar = volatility * es_factor * horizon_sqrt
            if isinstance(volatility, pd.Series):
                return pd.Series(cvar, index=volatility.index, name='CVaR')
            return cvar


def compute_rolling_var_from_garch(
    returns: pd.Series,
    window: int,
    confidence_level: float = 0.95,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True,
    batch_size: int = 50
) -> pd.Series:
    """
    Compute rolling VaR using GARCH volatility forecasts (GPU-accelerated).
    
    Args:
        returns: Series of returns
        window: Rolling window size for GARCH fitting
        confidence_level: Confidence level for VaR
        horizon: Forecast horizon
        p: GARCH order
        q: ARCH order
        dist: Distribution assumption
        mean: Mean model
        vol: Volatility model
        fallback_long_run_variance: If True, use long-run variance if model fails
        batch_size: Number of windows to process before clearing GPU cache (for memory management)
        
    Returns:
        Series of VaR estimates
    """
    # Get volatility forecasts (with GPU acceleration)
    vol_forecasts = compute_rolling_volatility_forecast(
        returns,
        window=window,
        horizon=horizon,
        p=p,
        q=q,
        dist=dist,
        mean=mean,
        vol=vol,
        fallback_long_run_variance=fallback_long_run_variance,
        batch_size=batch_size
    )
    
    # Convert volatility to VaR (GPU-accelerated)
    var_series = var_from_volatility(
        vol_forecasts,
        confidence_level=confidence_level,
        horizon=horizon,
        dist=dist
    )
    
    return var_series


def compute_rolling_cvar_from_garch(
    returns: pd.Series,
    window: int,
    confidence_level: float = 0.95,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True,
    batch_size: int = 50
) -> pd.Series:
    """
    Compute rolling CVaR using GARCH volatility forecasts (GPU-accelerated).
    
    Args:
        returns: Series of returns
        window: Rolling window size for GARCH fitting
        confidence_level: Confidence level for CVaR
        horizon: Forecast horizon
        p: GARCH order
        q: ARCH order
        dist: Distribution assumption
        mean: Mean model
        vol: Volatility model
        fallback_long_run_variance: If True, use long-run variance if model fails
        batch_size: Number of windows to process before clearing GPU cache (for memory management)
        
    Returns:
        Series of CVaR estimates
    """
    # Get volatility forecasts (with GPU acceleration)
    vol_forecasts = compute_rolling_volatility_forecast(
        returns,
        window=window,
        horizon=horizon,
        p=p,
        q=q,
        dist=dist,
        mean=mean,
        vol=vol,
        fallback_long_run_variance=fallback_long_run_variance,
        batch_size=batch_size
    )
    
    # Convert volatility to CVaR (GPU-accelerated)
    cvar_series = cvar_from_volatility(
        vol_forecasts,
        confidence_level=confidence_level,
        horizon=horizon,
        dist=dist
    )
    
    return cvar_series


def align_returns_and_var(
    returns: pd.Series,
    var_series: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Align returns and VaR series to common dates.
    
    Args:
        returns: Series of returns
        var_series: Series of VaR estimates
        
    Returns:
        Tuple of (aligned returns, aligned VaR)
    """
    common_dates = returns.index.intersection(var_series.index)
    
    if len(common_dates) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    aligned_returns = returns.loc[common_dates]
    aligned_var = var_series.loc[common_dates]
    
    return aligned_returns, aligned_var


def align_returns_and_cvar(
    returns: pd.Series,
    cvar_series: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Align returns and CVaR series to common dates.
    
    Args:
        returns: Series of returns
        cvar_series: Series of CVaR estimates
        
    Returns:
        Tuple of (aligned returns, aligned CVaR)
    """
    common_dates = returns.index.intersection(cvar_series.index)
    
    if len(common_dates) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    aligned_returns = returns.loc[common_dates]
    aligned_cvar = cvar_series.loc[common_dates]
    
    return aligned_returns, aligned_cvar


def compute_asset_level_conditional_volatility(
    asset_returns: pd.Series,
    estimation_window: int,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True,
    cache: Optional[GARCHParameterCache] = None
) -> pd.Series:
    """
    Compute rolling conditional volatility for a single asset using GARCH.
    
    This function implements asset-level GARCH fitting as specified in the
    computation strategy. For each date, fits GARCH to a rolling window and
    computes conditional volatility.
    
    Args:
        asset_returns: Series of asset returns
        estimation_window: Rolling window size for GARCH fitting
        p: GARCH order
        q: ARCH order
        dist: Distribution assumption
        mean: Mean model
        vol: Volatility model
        fallback_long_run_variance: If True, use long-run variance if model fails
        cache: Optional parameter cache
        
    Returns:
        Series of conditional volatility estimates
    """
    # Check cache first
    asset_name = asset_returns.name if hasattr(asset_returns, 'name') else 'asset'
    if cache:
        cached = cache.get(asset_name, estimation_window)
        if cached and 'conditional_volatility' in cached:
            return cached['conditional_volatility'].copy()
    
    # Need at least window + some buffer for GARCH fitting
    min_window = max(estimation_window, max(p, q) + 10)
    
    conditional_vols = []
    vol_dates = []
    
    for i in range(min_window, len(asset_returns)):
        # Get rolling window of returns
        window_returns = asset_returns.iloc[i - estimation_window:i]
        
        # Compute conditional volatility for this window
        cond_vol_series = compute_conditional_volatility(
            window_returns,
            p=p,
            q=q,
            dist=dist,
            mean=mean,
            vol=vol,
            fallback_long_run_variance=fallback_long_run_variance
        )
        
        # Get the last conditional volatility value (most recent)
        if len(cond_vol_series) > 0:
            last_vol = cond_vol_series.iloc[-1]
            conditional_vols.append(last_vol)
            vol_dates.append(asset_returns.index[i])
    
    # Create Series
    conditional_vol_series = pd.Series(
        conditional_vols,
        index=vol_dates,
        name='conditional_volatility'
    )
    
    # Store in cache
    if cache:
        cache.set(asset_name, estimation_window, {
            'conditional_volatility': conditional_vol_series
        })
    
    return conditional_vol_series


def _fit_single_asset_window(
    args: Tuple[str, pd.Series, int, int, int, str, str, str, bool, Optional[GARCHParameterCache]]
) -> Tuple[Tuple[str, int], pd.Series]:
    """
    Worker function to fit GARCH for a single asset/window combination.
    
    Args:
        args: Tuple of (asset, asset_returns, window, p, q, dist, mean, vol, fallback_long_run_variance, cache)
        
    Returns:
        Tuple of ((asset, window), conditional_volatility_series)
    """
    asset, asset_returns, window, p, q, dist, mean, vol, fallback_long_run_variance, cache = args
    
    try:
        cond_vol = compute_asset_level_conditional_volatility(
            asset_returns,
            window,
            p=p,
            q=q,
            dist=dist,
            mean=mean,
            vol=vol,
            fallback_long_run_variance=fallback_long_run_variance,
            cache=cache
        )
        return ((asset, window), cond_vol)
    except Exception:
        # Return empty series on failure
        return ((asset, window), pd.Series(dtype=float))


def compute_all_asset_conditional_volatilities(
    daily_returns: pd.DataFrame,
    estimation_windows: List[int],
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True,
    cache: Optional[GARCHParameterCache] = None,
    n_jobs: Optional[int] = None
) -> Dict[Tuple[str, int], pd.Series]:
    """
    Compute conditional volatilities for all assets and estimation windows.
    
    This is the core function that implements asset-level GARCH fitting as specified
    in the computation strategy. Fits GARCH once per asset and estimation window.
    Supports parallelization across asset/window combinations.
    
    Args:
        daily_returns: DataFrame of daily returns (dates x assets)
        estimation_windows: List of estimation window sizes
        p: GARCH order
        q: ARCH order
        dist: Distribution assumption
        mean: Mean model
        vol: Volatility model
        fallback_long_run_variance: If True, use long-run variance if model fails
        cache: Optional parameter cache (shared across workers)
        n_jobs: Number of parallel workers (None for sequential, 1 for sequential, >1 for parallel)
        
    Returns:
        Dictionary mapping (asset, estimation_window) to conditional volatility Series
    """
    # Prepare all asset/window combinations
    tasks = []
    for asset in daily_returns.columns:
        asset_returns = daily_returns[asset].dropna()
        for window in estimation_windows:
            tasks.append((asset, asset_returns, window, p, q, dist, mean, vol, fallback_long_run_variance, cache))
    
    # Determine if we should parallelize
    if n_jobs is None or n_jobs <= 1 or len(tasks) == 0:
        # Sequential processing
        all_conditional_vols = {}
        for task in tasks:
            key, cond_vol = _fit_single_asset_window(task)
            all_conditional_vols[key] = cond_vol
        return all_conditional_vols
    
    # Parallel processing
    all_conditional_vols = {}
    
    try:
        with Pool(processes=n_jobs) as pool:
            results = pool.map(_fit_single_asset_window, tasks)
        
        for key, cond_vol in results:
            all_conditional_vols[key] = cond_vol
    except Exception as e:
        # Graceful degradation: fall back to sequential
        warnings.warn(f"Parallel asset-level GARCH fitting failed: {e}. Falling back to sequential.")
        all_conditional_vols = {}
        for task in tasks:
            try:
                key, cond_vol = _fit_single_asset_window(task)
                all_conditional_vols[key] = cond_vol
            except Exception:
                continue
    
    return all_conditional_vols


def compute_portfolio_volatilities_batch_gpu(
    portfolio_weights_list: List[pd.Series],
    asset_conditional_vols: Dict[str, pd.Series],
    covariance_matrix: Optional[pd.DataFrame] = None,
    use_dynamic_covariance: bool = False
) -> List[pd.Series]:
    """
    Compute portfolio volatilities for multiple portfolios in a GPU batch.
    
    This function processes multiple portfolios simultaneously on GPU for better performance.
    
    Args:
        portfolio_weights_list: List of portfolio weight Series
        asset_conditional_vols: Dictionary mapping asset names to conditional volatility Series
        covariance_matrix: Optional covariance matrix
        use_dynamic_covariance: Whether to use dynamic covariance
        
    Returns:
        List of portfolio volatility Series
    """
    if not TORCH_AVAILABLE or DEVICE is None or DEVICE.type != 'cuda':
        # Fallback to CPU processing
        return [compute_portfolio_volatility_from_assets(
            weights, asset_conditional_vols, covariance_matrix, use_dynamic_covariance, use_gpu=False
        ) for weights in portfolio_weights_list]
    
    # GPU batch processing
    results = []
    for weights in portfolio_weights_list:
        vol_series = compute_portfolio_volatility_from_assets(
            weights, asset_conditional_vols, covariance_matrix, use_dynamic_covariance, use_gpu=True
        )
        results.append(vol_series)
    
    return results


def compute_portfolio_volatility_from_assets(
    portfolio_weights: pd.Series,
    asset_conditional_vols: Dict[str, pd.Series],
    covariance_matrix: Optional[pd.DataFrame] = None,
    use_dynamic_covariance: bool = False,
    use_gpu: bool = True
) -> pd.Series:
    """
    Compute portfolio volatility via variance aggregation from asset-level conditional variances.
    
    Portfolio variance = w' * Σ * w
    where w is portfolio weights and Σ is the covariance matrix.
    
    If use_dynamic_covariance=False, uses simple variance aggregation:
    Portfolio variance ≈ Σ(w_i² * σ_i²) for uncorrelated assets
    
    GPU-accelerated for batch processing of multiple dates.
    
    Args:
        portfolio_weights: Series of portfolio weights (assets as index)
        asset_conditional_vols: Dictionary mapping asset names to conditional volatility Series
        covariance_matrix: Optional covariance matrix (if use_dynamic_covariance=True)
        use_dynamic_covariance: If True, use full covariance matrix; else use variance aggregation
        use_gpu: If True, use GPU acceleration when available
        
    Returns:
        Series of portfolio volatility estimates
    """
    # Get common assets
    common_assets = portfolio_weights.index.intersection(set(asset_conditional_vols.keys()))
    if len(common_assets) == 0:
        return pd.Series(dtype=float)
    
    # Normalize weights
    weights = portfolio_weights[common_assets] / portfolio_weights[common_assets].sum()
    
    # Get conditional volatility series for common assets
    vol_series_list = []
    for asset in common_assets:
        vol_series = asset_conditional_vols[asset]
        vol_series_list.append(vol_series)
    
    # Find common dates across all volatility series
    if len(vol_series_list) == 0:
        return pd.Series(dtype=float)
    
    common_dates = vol_series_list[0].index
    for vol_series in vol_series_list[1:]:
        common_dates = common_dates.intersection(vol_series.index)
    
    if len(common_dates) == 0:
        return pd.Series(dtype=float)
    
    # Align all series to common dates
    aligned_vols = {}
    for asset in common_assets:
        aligned_vols[asset] = asset_conditional_vols[asset].loc[common_dates]
    
    # Check if GPU is available and should be used
    use_gpu_accel = use_gpu and TORCH_AVAILABLE and DEVICE is not None and DEVICE.type == 'cuda'
    
    if use_dynamic_covariance and covariance_matrix is not None:
        # Use full covariance matrix (more accurate but computationally expensive)
        common_assets_list = list(common_assets)
        cov_subset = covariance_matrix.loc[common_assets_list, common_assets_list]
        weights_array = weights[common_assets_list].values
        
        if use_gpu_accel:
            # GPU-accelerated batch processing
            try:
                # Convert weights to tensor
                weights_tensor = _to_tensor(weights_array)
                
                # Build correlation matrix once
                corr_matrix = np.corrcoef(cov_subset.values)
                corr_tensor = _to_tensor(corr_matrix)
                
                # Build conditional volatility matrix (num_dates x num_assets)
                cond_vols_matrix = np.array([[aligned_vols[asset].loc[date] for asset in common_assets_list] 
                                            for date in common_dates])
                cond_vols_tensor = _to_tensor(cond_vols_matrix)
                
                # Compute conditional covariance matrices for all dates at once
                # cond_cov = outer(vols, vols) * corr for each date
                # Shape: (num_dates, num_assets, num_assets)
                cond_vols_expanded = cond_vols_tensor.unsqueeze(2)  # (num_dates, num_assets, 1)
                cond_vols_expanded_T = cond_vols_tensor.unsqueeze(1)  # (num_dates, 1, num_assets)
                cond_cov_tensor = cond_vols_expanded * cond_vols_expanded_T * corr_tensor.unsqueeze(0)
                
                # Portfolio variance = w' * Σ * w for all dates
                # weights_tensor: (num_assets,)
                # cond_cov_tensor: (num_dates, num_assets, num_assets)
                # Result: (num_dates,)
                weights_expanded = weights_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, num_assets)
                weights_col = weights_tensor.unsqueeze(0).unsqueeze(2)  # (1, num_assets, 1)
                # Compute w' * Σ for all dates: (num_dates, 1, num_assets)
                w_sigma = torch.bmm(weights_expanded.expand(cond_cov_tensor.shape[0], -1, -1), cond_cov_tensor)
                # Compute (w' * Σ) * w for all dates: (num_dates, 1, 1)
                portfolio_var_tensor = torch.bmm(w_sigma, weights_col.expand(cond_cov_tensor.shape[0], -1, -1)).squeeze()
                
                # Take square root
                portfolio_vols = _to_numpy(torch.sqrt(portfolio_var_tensor))
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                # Fallback to CPU
                portfolio_vols = []
                for date in common_dates:
                    cond_vols_array = np.array([aligned_vols[asset].loc[date] for asset in common_assets_list])
                    corr_matrix = np.corrcoef(cov_subset.values)
                    cond_cov_matrix = np.outer(cond_vols_array, cond_vols_array) * corr_matrix
                    portfolio_var = weights_array @ cond_cov_matrix @ weights_array
                    portfolio_vols.append(np.sqrt(portfolio_var))
        else:
            # CPU processing
            portfolio_vols = []
            for date in common_dates:
                cond_vols_array = np.array([aligned_vols[asset].loc[date] for asset in common_assets_list])
                corr_matrix = np.corrcoef(cov_subset.values)
                cond_cov_matrix = np.outer(cond_vols_array, cond_vols_array) * corr_matrix
                portfolio_var = weights_array @ cond_cov_matrix @ weights_array
                portfolio_vols.append(np.sqrt(portfolio_var))
    else:
        # Simple variance aggregation (assumes low correlation or uses diagonal)
        if use_gpu_accel:
            # GPU-accelerated batch processing
            try:
                # Build weights tensor
                weights_array = weights[common_assets].values
                weights_tensor = _to_tensor(weights_array)
                
                # Build conditional volatility matrix (num_dates x num_assets)
                cond_vols_matrix = np.array([[aligned_vols[asset].loc[date] for asset in common_assets] 
                                            for date in common_dates])
                cond_vols_tensor = _to_tensor(cond_vols_matrix)
                
                # Portfolio variance ≈ Σ(w_i² * σ_i²) for all dates
                # weights_tensor: (num_assets,)
                # cond_vols_tensor: (num_dates, num_assets)
                weights_squared = weights_tensor ** 2  # (num_assets,)
                portfolio_var_tensor = torch.sum(cond_vols_tensor ** 2 * weights_squared.unsqueeze(0), dim=1)
                
                # Take square root
                portfolio_vols = _to_numpy(torch.sqrt(portfolio_var_tensor))
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                # Fallback to CPU
                portfolio_vols = []
                for date in common_dates:
                    portfolio_var = 0.0
                    for asset in common_assets:
                        w_i = weights[asset]
                        sigma_i = aligned_vols[asset].loc[date]
                        portfolio_var += (w_i ** 2) * (sigma_i ** 2)
                    portfolio_vols.append(np.sqrt(portfolio_var))
        else:
            # CPU processing
            portfolio_vols = []
            for date in common_dates:
                portfolio_var = 0.0
                for asset in common_assets:
                    w_i = weights[asset]
                    sigma_i = aligned_vols[asset].loc[date]
                    portfolio_var += (w_i ** 2) * (sigma_i ** 2)
                portfolio_vols.append(np.sqrt(portfolio_var))
    
    portfolio_vol_series = pd.Series(
        portfolio_vols,
        index=common_dates,
        name='portfolio_volatility'
    )
    
    return portfolio_vol_series


def compute_horizons(
    base_horizon: int = 1,
    scaled_horizons: Optional[List[int]] = None,
    scaling_rule: str = 'sqrt_time'
) -> List[int]:
    """
    Compute list of horizons from base horizon and scaled horizons.
    
    Args:
        base_horizon: Base horizon (e.g., 1 day)
        scaled_horizons: List of scaling factors (e.g., [10] means 10x base horizon)
        scaling_rule: Scaling rule ('sqrt_time' for square root scaling)
        
    Returns:
        List of horizon values
    """
    horizons = [base_horizon]
    
    if scaled_horizons:
        for scale in scaled_horizons:
            if scaling_rule == 'sqrt_time':
                # For sqrt scaling: horizon = base_horizon * sqrt(scale)
                horizon = int(base_horizon * np.sqrt(scale))
            else:
                # Linear scaling: horizon = base_horizon * scale
                horizon = base_horizon * scale
            if horizon not in horizons:
                horizons.append(horizon)
    
    return sorted(horizons)

