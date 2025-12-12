"""
GARCH model calculator for volatility forecasting and VaR/CVaR estimation.

Implements GARCH(1,1) model fitting, conditional volatility computation,
and rolling volatility forecasting for portfolio risk assessment.

GPU-accelerated using PyTorch for improved performance on large datasets.
"""
import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple, Dict, Any
import warnings
import time

try:
    import torch
    TORCH_AVAILABLE = True
    # Set device: use GPU if available, otherwise CPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
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
    dist: str = 'normal'
) -> Union[pd.Series, float]:
    """
    Compute VaR from volatility forecast using distributional assumption (GPU-accelerated).
    
    VaR = volatility * z_score(confidence_level) * sqrt(horizon)
    
    Args:
        volatility: Volatility forecast (Series or scalar)
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        horizon: Time horizon (e.g., 1 for 1 day)
        dist: Distribution assumption ('normal', 't')
        
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
    
    # Convert z_score to tensor for GPU operations (if available)
    horizon_sqrt = np.sqrt(horizon)
    if TORCH_AVAILABLE:
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
        if TORCH_AVAILABLE:
            vol_tensor = torch.tensor(vol_scalar, device=DEVICE, dtype=torch.float32)
            var_tensor = vol_tensor * z_score_tensor * horizon_tensor
            return float(var_tensor.item())
        else:
            return vol_scalar * z_score * horizon_sqrt
    else:
        # Series case - use GPU if available
        if TORCH_AVAILABLE and isinstance(volatility, pd.Series):
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
    dist: str = 'normal'
) -> Union[pd.Series, float]:
    """
    Compute CVaR from volatility forecast using distributional assumption (GPU-accelerated).
    
    CVaR = volatility * expected_shortfall_factor * sqrt(horizon)
    
    Args:
        volatility: Volatility forecast (Series or scalar)
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        horizon: Time horizon (e.g., 1 for 1 day)
        dist: Distribution assumption ('normal', 't')
        
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
    
    # Convert es_factor to tensor for GPU operations (if available)
    horizon_sqrt = np.sqrt(horizon)
    if TORCH_AVAILABLE:
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
        if TORCH_AVAILABLE:
            vol_tensor = torch.tensor(vol_scalar, device=DEVICE, dtype=torch.float32)
            cvar_tensor = vol_tensor * es_factor_tensor * horizon_tensor
            return float(cvar_tensor.item())
        else:
            return vol_scalar * es_factor * horizon_sqrt
    else:
        # Series case - use GPU if available
        if TORCH_AVAILABLE and isinstance(volatility, pd.Series):
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

