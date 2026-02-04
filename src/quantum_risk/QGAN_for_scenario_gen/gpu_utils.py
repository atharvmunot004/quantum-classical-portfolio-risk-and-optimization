"""
GPU utility functions for QGAN optimization.
"""
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_device(use_gpu: bool = True):
    """
    Get the best available device (GPU or CPU).
    
    Args:
        use_gpu: Whether to prefer GPU
        
    Returns:
        torch.device object
    """
    if not TORCH_AVAILABLE:
        return None
    
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = torch.device('cpu')
        if use_gpu:
            warnings.warn("GPU requested but not available. Using CPU.")
        return device


def clear_gpu_cache():
    """Clear GPU cache if available."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gpu_memory_info():
    """Get GPU memory information."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None
    
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
        'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
        'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
    }
