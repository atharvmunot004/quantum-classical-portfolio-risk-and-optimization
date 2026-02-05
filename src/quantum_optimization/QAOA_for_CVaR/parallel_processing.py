"""
Parallel processing utilities for QAOA optimization.

Provides CPU-based multiprocessing with resource limits.
"""
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Callable, List, Tuple, Any, Optional
import warnings


def get_optimal_worker_count(
    max_workers: Optional[int] = None,
    cpu_percent: float = 0.8
) -> int:
    """
    Get optimal number of worker processes.
    
    Args:
        max_workers: Maximum number of workers (None for auto)
        cpu_percent: Percentage of CPU cores to use (default 0.8 = 80%)
        
    Returns:
        Number of worker processes
    """
    total_cores = cpu_count()
    
    if max_workers is None:
        # Use 80% of cores, but leave at least 1 core free
        workers = max(1, int(total_cores * cpu_percent))
        workers = min(workers, total_cores - 1)  # Leave at least 1 core free
        return max(1, workers)
    
    # Respect user-specified limit
    return max(1, min(max_workers, total_cores))


def process_optimizations_parallel(
    optimization_tasks: List[Tuple],
    worker_func: Callable,
    n_jobs: Optional[int] = None,
    cpu_percent: float = 0.8,
    chunksize: int = 1
) -> List[Any]:
    """
    Process optimization tasks in parallel using multiprocessing.
    
    Args:
        optimization_tasks: List of task arguments tuples
        worker_func: Worker function to process each task
        n_jobs: Number of parallel jobs (None for auto)
        cpu_percent: Percentage of CPU cores to use
        chunksize: Chunk size for imap
        
    Returns:
        List of results
    """
    if not optimization_tasks:
        return []
    
    # Determine number of workers
    if n_jobs is None:
        n_jobs = get_optimal_worker_count(cpu_percent=cpu_percent)
    else:
        n_jobs = get_optimal_worker_count(max_workers=n_jobs)
    
    # If only one task or one worker, process sequentially
    if len(optimization_tasks) == 1 or n_jobs == 1:
        return [worker_func(task) for task in optimization_tasks]
    
    # Parallel processing
    results = []
    try:
        with Pool(processes=n_jobs) as pool:
            # Use imap for better progress tracking
            results_iter = pool.imap(worker_func, optimization_tasks, chunksize=chunksize)
            results = list(results_iter)
    except Exception as e:
        warnings.warn(f"Parallel processing failed: {e}. Falling back to sequential.")
        # Fallback to sequential
        results = [worker_func(task) for task in optimization_tasks]
    
    return results
