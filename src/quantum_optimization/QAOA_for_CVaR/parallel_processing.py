"""
Parallel processing utilities for QAOA optimization.
CPU-only, with strict worker and memory limits.
"""
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, List, Optional, Tuple

import warnings


def get_optimal_worker_count(
    max_workers: Optional[int] = None,
    cpu_percent: float = 0.5,
    max_workers_cap: int = 8,
) -> int:
    """
    Number of worker processes; keeps CPU and memory in check.
    max_workers_cap limits memory (each worker loads returns).
    """
    total = cpu_count()
    if max_workers is not None:
        n = max(1, min(max_workers, total))
    else:
        pct = max(0.1, min(1.0, float(cpu_percent)))
        n = max(1, min(int(total * pct), total - 1, max_workers_cap))
    return n


def process_optimizations_parallel(
    optimization_tasks: List[Tuple],
    worker_func: Callable,
    n_jobs: Optional[int] = None,
    cpu_percent: float = 0.5,
    chunksize: int = 1,
) -> List[Any]:
    if not optimization_tasks:
        return []
    n_jobs = get_optimal_worker_count(max_workers=n_jobs, cpu_percent=cpu_percent)
    if len(optimization_tasks) == 1 or n_jobs == 1:
        return [worker_func(t) for t in optimization_tasks]
    try:
        with Pool(processes=n_jobs) as pool:
            return list(pool.imap(worker_func, optimization_tasks, chunksize=chunksize))
    except Exception as e:
        warnings.warn(f"Parallel failed: {e}. Falling back to sequential.")
        return [worker_func(t) for t in optimization_tasks]
