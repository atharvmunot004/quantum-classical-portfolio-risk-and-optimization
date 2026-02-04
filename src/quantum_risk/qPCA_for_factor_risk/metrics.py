"""Factor quality and classical alignment metrics for qPCA."""
import numpy as np
from typing import Dict, Optional


def explained_variance_ratios(evals: np.ndarray) -> np.ndarray:
    """Per-component explained variance ratio (evals already for top-k)."""
    total = np.sum(evals) + 1e-12
    return np.array(evals) / total


def cumulative_explained_variance(evals: np.ndarray) -> np.ndarray:
    """Cumulative explained variance ratio."""
    total = np.sum(evals) + 1e-12
    return np.cumsum(evals) / total


def factor_stability_cosine_similarity(
    evecs_prev: Optional[np.ndarray],
    evecs_curr: np.ndarray,
) -> float:
    """
    Cosine similarity between factor subspaces of adjacent windows.
    Returns mean absolute cosine (column alignment) or 0 if no previous.
    """
    if evecs_prev is None or evecs_prev.size == 0 or evecs_curr.size == 0:
        return 0.0
    k = min(evecs_prev.shape[1], evecs_curr.shape[1])
    if k == 0:
        return 0.0
    sims = []
    for j in range(k):
        s = np.abs(np.dot(evecs_prev[:, j], evecs_curr[:, j]))
        sims.append(min(1.0, float(s)))
    return float(np.mean(sims))


def subspace_distance(V1: np.ndarray, V2: np.ndarray) -> float:
    """Frobenius norm of (V1 - V2) for same-shaped orthonormal matrices."""
    if V1.shape != V2.shape:
        return 1.0
    return float(np.linalg.norm(V1 - V2, "fro"))
