"""
GPU acceleration utilities for Black-Litterman optimization.

Provides GPU-accelerated matrix operations using CuPy (NumPy-compatible)
with automatic fallback to NumPy if GPU is not available.
"""
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Any, List, Dict
import warnings

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    
    # Check if GPU is available
    try:
        cp.cuda.Device(0).use()
        GPU_AVAILABLE = True
        GPU_DEVICE = cp.cuda.Device(0)
    except Exception:
        GPU_AVAILABLE = False
        GPU_DEVICE = None
except ImportError:
    CUPY_AVAILABLE = False
    GPU_AVAILABLE = False
    GPU_DEVICE = None
    cp = None  # Set to None when CuPy is not available


def get_array_module(use_gpu: bool = False):
    """
    Get the appropriate array module (CuPy for GPU, NumPy for CPU).
    
    Args:
        use_gpu: Whether to use GPU (if available)
        
    Returns:
        Array module (cupy or numpy)
    """
    if use_gpu and CUPY_AVAILABLE and GPU_AVAILABLE:
        return cp
    return np


def to_gpu_array(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    use_gpu: bool = False
) -> Union[Any, np.ndarray]:
    """
    Convert data to GPU array if GPU is available and requested.
    
    Args:
        data: Input data (DataFrame, Series, or array)
        use_gpu: Whether to use GPU
        
    Returns:
        GPU array (CuPy) or CPU array (NumPy)
    """
    xp = get_array_module(use_gpu)
    
    if isinstance(data, pd.DataFrame):
        return xp.asarray(data.values)
    elif isinstance(data, pd.Series):
        return xp.asarray(data.values)
    elif isinstance(data, np.ndarray):
        return xp.asarray(data)
    else:
        return xp.asarray(data)


def to_cpu_array(data: Union[Any, np.ndarray]) -> np.ndarray:
    """
    Convert GPU array back to CPU NumPy array.
    
    Args:
        data: GPU or CPU array
        
    Returns:
        NumPy array on CPU
    """
    if CUPY_AVAILABLE and cp is not None:
        try:
            # Check if it's a CuPy array by checking for device attribute
            if hasattr(data, 'device') and hasattr(data, 'get'):
                return cp.asnumpy(data)
        except Exception:
            pass
    return np.asarray(data)


def compute_covariance_gpu(
    returns: Union[pd.DataFrame, np.ndarray, Any],
    use_gpu: bool = False
) -> Union[Any, np.ndarray]:
    """
    Compute covariance matrix on GPU.
    
    Args:
        returns: Returns data
        use_gpu: Whether to use GPU
        
    Returns:
        Covariance matrix
    """
    xp = get_array_module(use_gpu)
    returns_array = to_gpu_array(returns, use_gpu)
    
    # Compute covariance: cov = E[(X - μ)(X - μ)^T]
    mean = xp.mean(returns_array, axis=0, keepdims=True)
    centered = returns_array - mean
    cov = xp.dot(centered.T, centered) / (returns_array.shape[0] - 1)
    
    return cov


def matrix_inverse_gpu(
    matrix: Union[np.ndarray, Any],
    use_gpu: bool = False
) -> Union[Any, np.ndarray]:
    """
    Compute matrix inverse on GPU.
    
    Args:
        matrix: Input matrix
        use_gpu: Whether to use GPU
        
    Returns:
        Inverse matrix
    """
    xp = get_array_module(use_gpu)
    matrix_array = to_gpu_array(matrix, use_gpu)
    
    try:
        inv = xp.linalg.inv(matrix_array)
    except (np.linalg.LinAlgError, Exception):
        # Fallback to pseudo-inverse if singular
        inv = xp.linalg.pinv(matrix_array)
    
    return inv


def matrix_multiply_gpu(
    A: Union[np.ndarray, Any],
    B: Union[np.ndarray, Any],
    use_gpu: bool = False
) -> Union[Any, np.ndarray]:
    """
    Matrix multiplication on GPU.
    
    Args:
        A: First matrix
        B: Second matrix
        use_gpu: Whether to use GPU
        
    Returns:
        Matrix product A @ B
    """
    xp = get_array_module(use_gpu)
    A_array = to_gpu_array(A, use_gpu)
    B_array = to_gpu_array(B, use_gpu)
    
    return xp.dot(A_array, B_array)


def solve_linear_system_gpu(
    A: Union[np.ndarray, Any],
    b: Union[np.ndarray, Any],
    use_gpu: bool = False
) -> Union[Any, np.ndarray]:
    """
    Solve linear system Ax = b on GPU.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        use_gpu: Whether to use GPU
        
    Returns:
        Solution vector x
    """
    xp = get_array_module(use_gpu)
    A_array = to_gpu_array(A, use_gpu)
    b_array = to_gpu_array(b, use_gpu)
    
    try:
        x = xp.linalg.solve(A_array, b_array)
    except (np.linalg.LinAlgError, Exception):
        # Fallback to least squares if singular
        x = xp.linalg.lstsq(A_array, b_array, rcond=None)[0]
    
    return x


def batch_covariance_gpu(
    returns_batch: List[Union[pd.DataFrame, np.ndarray]],
    use_gpu: bool = False
) -> List[Union[Any, np.ndarray]]:
    """
    Compute covariance matrices for a batch of return series on GPU.
    
    Args:
        returns_batch: List of return DataFrames/arrays
        use_gpu: Whether to use GPU
        
    Returns:
        List of covariance matrices
    """
    if not use_gpu or not GPU_AVAILABLE:
        # Fallback to CPU
        return [compute_covariance_gpu(ret, use_gpu=False) for ret in returns_batch]
    
    xp = get_array_module(use_gpu)
    covariances = []
    
    for returns in returns_batch:
        cov = compute_covariance_gpu(returns, use_gpu=True)
        covariances.append(cov)
    
    return covariances


def is_gpu_available() -> bool:
    """Check if GPU is available for computation."""
    return GPU_AVAILABLE and CUPY_AVAILABLE


def get_gpu_info() -> dict:
    """
    Get GPU information if available.
    
    Returns:
        Dictionary with GPU information
    """
    if not is_gpu_available():
        return {
            'available': False,
            'library': None,
            'device': None
        }
    
    try:
        if cp is None:
            return {
                'available': False,
                'library': None,
                'device': None
            }
        device = GPU_DEVICE
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        
        return {
            'available': True,
            'library': 'CuPy',
            'device': str(device),
            'memory_used_mb': mempool.used_bytes() / (1024**2),
            'memory_total_mb': mempool.total_bytes() / (1024**2) if hasattr(mempool, 'total_bytes') else None
        }
    except Exception:
        return {
            'available': True,
            'library': 'CuPy',
            'device': 'Unknown'
        }


def clear_gpu_cache():
    """Clear GPU memory cache if using CuPy."""
    if CUPY_AVAILABLE and GPU_AVAILABLE and cp is not None:
        try:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        except Exception:
            pass


def batch_portfolio_returns_matrix_multiply(
    returns_matrix: np.ndarray,
    weight_matrix: np.ndarray,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Compute portfolio returns for all portfolios in batch: R(T x N_assets) @ W(N_assets x N_portfolios).
    
    Args:
        returns_matrix: Returns matrix (T x N_assets)
        weight_matrix: Weight matrix (N_assets x N_portfolios)
        use_gpu: Whether to use GPU
        
    Returns:
        Portfolio returns matrix (T x N_portfolios)
    """
    xp = get_array_module(use_gpu)
    R = to_gpu_array(returns_matrix, use_gpu)
    W = to_gpu_array(weight_matrix, use_gpu)
    
    portfolio_returns = xp.dot(R, W)
    
    return to_cpu_array(portfolio_returns)


def batch_cumprod_running_max_for_drawdown(
    returns_matrix: np.ndarray,
    use_gpu: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cumulative product and running max for drawdown calculation.
    
    Args:
        returns_matrix: Returns matrix (T x N_portfolios)
        use_gpu: Whether to use GPU
        
    Returns:
        Tuple of (cumulative, running_max) arrays
    """
    xp = get_array_module(use_gpu)
    R = to_gpu_array(returns_matrix, use_gpu)
    
    # Cumulative product: (1 + r1) * (1 + r2) * ...
    cumulative = xp.cumprod(1 + R, axis=0)
    
    # Running max along time axis
    running_max = xp.maximum.accumulate(cumulative, axis=0)
    
    return to_cpu_array(cumulative), to_cpu_array(running_max)


def batch_percentiles_for_var_cvar(
    returns_matrix: np.ndarray,
    percentiles: List[float],
    use_gpu: bool = False
) -> np.ndarray:
    """
    Compute percentiles along time axis for VaR/CVaR calculation.
    
    Args:
        returns_matrix: Returns matrix (T x N_portfolios)
        percentiles: List of percentile values (e.g., [5, 95])
        use_gpu: Whether to use GPU
        
    Returns:
        Percentile values array (len(percentiles) x N_portfolios)
    """
    xp = get_array_module(use_gpu)
    R = to_gpu_array(returns_matrix, use_gpu)
    
    # Compute percentiles along axis 0 (time)
    if use_gpu and CUPY_AVAILABLE and GPU_AVAILABLE:
        # CuPy doesn't have percentile, use numpy
        percentiles_array = np.percentile(to_cpu_array(R), percentiles, axis=0)
    else:
        percentiles_array = np.percentile(R, percentiles, axis=0)
    
    return percentiles_array


def batch_mean_std_skew_kurtosis(
    returns_matrix: np.ndarray,
    use_gpu: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compute mean, std, skewness, kurtosis along time axis.
    
    Args:
        returns_matrix: Returns matrix (T x N_portfolios)
        use_gpu: Whether to use GPU
        
    Returns:
        Dictionary with 'mean', 'std', 'skew', 'kurt' arrays
    """
    xp = get_array_module(use_gpu)
    R = to_gpu_array(returns_matrix, use_gpu)
    
    mean = xp.mean(R, axis=0)
    std = xp.std(R, axis=0, ddof=1)
    
    # Skewness and kurtosis: convert to CPU for scipy.stats
    R_cpu = to_cpu_array(R)
    from scipy import stats
    skew = np.apply_along_axis(stats.skew, 0, R_cpu)
    kurt = np.apply_along_axis(stats.kurtosis, 0, R_cpu)
    
    return {
        'mean': to_cpu_array(mean),
        'std': to_cpu_array(std),
        'skew': skew,
        'kurt': kurt
    }
