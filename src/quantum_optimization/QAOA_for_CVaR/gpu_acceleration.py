"""
GPU acceleration utilities for QAOA Portfolio CVaR Optimization.

Provides GPU-accelerated matrix operations using CuPy (NumPy-compatible)
with automatic fallback to NumPy if GPU is not available.
"""
import numpy as np
from typing import Optional, Union, Tuple, Any
import warnings

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    
    # Check if GPU is available
    try:
        # Try to use GPU device 0
        device = cp.cuda.Device(0)
        device.use()
        # Test a simple operation to ensure GPU is working
        test_array = cp.array([1.0, 2.0, 3.0])
        _ = cp.sum(test_array)
        # Clear test array from GPU memory
        del test_array
        cp.get_default_memory_pool().free_all_blocks()
        GPU_AVAILABLE = True
        GPU_DEVICE = device
    except Exception as e:
        import warnings
        warnings.warn(f"GPU initialization failed: {e}. Falling back to CPU.")
        GPU_AVAILABLE = False
        GPU_DEVICE = None
except ImportError:
    CUPY_AVAILABLE = False
    GPU_AVAILABLE = False
    GPU_DEVICE = None
    cp = None


def should_use_gpu(use_gpu: bool, matrix_size: int = 0, batch_size: int = 0) -> bool:
    """
    Determine if GPU should be used based on problem size.
    
    GPU is only beneficial for:
    - Large matrices (>100x100) or
    - Large batch operations (>1000 items)
    
    Args:
        use_gpu: Whether GPU is requested
        matrix_size: Size of matrix (n x n)
        batch_size: Number of items in batch
        
    Returns:
        Whether to actually use GPU
    """
    if not use_gpu or not CUPY_AVAILABLE or not GPU_AVAILABLE:
        return False
    
    # GPU threshold: only use for larger problems
    # Small matrices (< 50x50) are faster on CPU due to GPU overhead
    if matrix_size > 0:
        if matrix_size < 50:
            return False  # Too small for GPU
    
    # Batch operations: use GPU if batch is large enough
    if batch_size > 0:
        if batch_size < 1000:
            return False  # Too small batch for GPU
    
    return True


def get_array_module(use_gpu: bool = False, matrix_size: int = 0, batch_size: int = 0):
    """
    Get the appropriate array module (CuPy for GPU, NumPy for CPU).
    
    Args:
        use_gpu: Whether to use GPU (if available)
        matrix_size: Size of matrix (n x n) - used to decide if GPU is beneficial
        batch_size: Number of items in batch - used to decide if GPU is beneficial
        
    Returns:
        Array module (cupy or numpy)
    """
    if should_use_gpu(use_gpu, matrix_size, batch_size):
        return cp
    return np


def to_gpu_array(
    data: Union[np.ndarray, Any],
    use_gpu: bool = False
) -> Union[Any, np.ndarray]:
    """
    Convert NumPy array to GPU array.
    
    Args:
        data: NumPy array or CuPy array
        use_gpu: Whether to use GPU
        
    Returns:
        GPU array if use_gpu and GPU available, else CPU array
    """
    if not use_gpu or not GPU_AVAILABLE:
        if isinstance(data, np.ndarray):
            return data
        elif CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return np.asarray(data)
    
    if isinstance(data, cp.ndarray):
        return data
    elif isinstance(data, np.ndarray):
        return cp.asarray(data)
    else:
        return cp.asarray(np.asarray(data))


def to_cpu_array(data: Union[np.ndarray, Any]) -> np.ndarray:
    """
    Convert GPU array to CPU array.
    
    Args:
        data: GPU or CPU array
        
    Returns:
        NumPy array
    """
    if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return np.asarray(data)


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
    
    result = xp.dot(A_array, B_array)
    
    if use_gpu and GPU_AVAILABLE:
        return to_cpu_array(result)
    return result


def qubo_energy_gpu(
    solutions: np.ndarray,
    Q: np.ndarray,
    constant: float = 0.0,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Compute QUBO energies for multiple solutions on GPU.
    
    Energy = solution^T @ Q @ solution + constant
    
    Args:
        solutions: Binary solutions matrix (num_solutions x num_vars)
        Q: QUBO matrix (num_vars x num_vars)
        constant: Constant term
        use_gpu: Whether to use GPU
        
    Returns:
        Array of energies (num_solutions,)
    """
    if not use_gpu or not GPU_AVAILABLE:
        # CPU fallback
        energies = np.sum((solutions @ Q) * solutions, axis=1) + constant
        return energies
    
    # GPU computation
    solutions_gpu = to_gpu_array(solutions, use_gpu=True)
    Q_gpu = to_gpu_array(Q, use_gpu=True)
    constant_gpu = cp.asarray(constant)
    
    # Compute energies: sum((solutions @ Q) * solutions, axis=1) + constant
    energies_gpu = cp.sum((solutions_gpu @ Q_gpu) * solutions_gpu, axis=1) + constant_gpu
    
    return to_cpu_array(energies_gpu)


def batch_scenario_loss_gpu(
    scenario_matrix: np.ndarray,
    weights: np.ndarray,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Compute scenario losses for multiple portfolios on GPU.
    
    Loss = -scenario_matrix @ weights
    
    Args:
        scenario_matrix: Scenario matrix (num_scenarios x num_assets)
        weights: Portfolio weights matrix (num_portfolios x num_assets)
        use_gpu: Whether to use GPU
        
    Returns:
        Loss matrix (num_scenarios x num_portfolios)
    """
    num_portfolios = weights.shape[0] if weights.ndim > 1 else 1
    
    # Only use GPU for large batch operations (>1000 portfolios)
    use_gpu_here = should_use_gpu(use_gpu, batch_size=num_portfolios)
    
    if not use_gpu_here:
        # CPU fallback
        losses = -scenario_matrix @ weights.T
        return losses
    
    # GPU computation
    scenario_gpu = to_gpu_array(scenario_matrix, use_gpu=True)
    weights_gpu = to_gpu_array(weights, use_gpu=True)
    
    # Compute losses: -scenario @ weights^T
    losses_gpu = -cp.dot(scenario_gpu, weights_gpu.T)
    
    return to_cpu_array(losses_gpu)


def compute_cvar_gpu(
    losses: np.ndarray,
    alpha: float = 0.95,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Compute CVaR for multiple portfolios on GPU.
    
    Args:
        losses: Loss matrix (num_scenarios x num_portfolios)
        alpha: Confidence level
        use_gpu: Whether to use GPU
        
    Returns:
        CVaR values (num_portfolios,)
    """
    num_portfolios = losses.shape[1] if losses.ndim > 1 else 1
    
    # Only use GPU for large batch operations (>1000 portfolios)
    use_gpu_here = should_use_gpu(use_gpu, batch_size=num_portfolios)
    
    if not use_gpu_here:
        # CPU fallback
        sorted_losses = np.sort(losses, axis=0)
        tail_size = max(1, int((1 - alpha) * len(sorted_losses)))
        tail_losses = sorted_losses[-tail_size:]
        cvar = np.mean(tail_losses, axis=0)
        return cvar
    
    # GPU computation
    losses_gpu = to_gpu_array(losses, use_gpu=True)
    
    # Sort losses
    sorted_losses_gpu = cp.sort(losses_gpu, axis=0)
    tail_size = max(1, int((1 - alpha) * len(sorted_losses_gpu)))
    tail_losses_gpu = sorted_losses_gpu[-tail_size:]
    
    # Compute mean
    cvar_gpu = cp.mean(tail_losses_gpu, axis=0)
    
    return to_cpu_array(cvar_gpu)


def is_gpu_available() -> bool:
    """Check if GPU is available for computation."""
    return GPU_AVAILABLE


def get_gpu_info() -> dict:
    """
    Get GPU information.
    
    Returns:
        Dictionary with GPU information
    """
    if not GPU_AVAILABLE:
        return {
            'available': False,
            'device_name': None,
            'memory_total': None,
            'memory_free': None
        }
    
    try:
        mempool = cp.get_default_memory_pool()
        meminfo = cp.cuda.runtime.memGetInfo()
        
        return {
            'available': True,
            'device_name': cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8'),
            'memory_total': meminfo[1],
            'memory_free': meminfo[0],
            'memory_used': meminfo[1] - meminfo[0]
        }
    except Exception as e:
        return {
            'available': True,
            'error': str(e)
        }
