"""Classical PCA baseline for alignment metrics (principal angles, explained variance gap, exposure correlation)."""
import numpy as np
from typing import Tuple, Dict


def classical_pca_eigen(
    rho: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classical PCA on same density matrix. Returns (eigenvalues, eigenvectors as columns)."""
    evals, evecs = np.linalg.eigh(rho)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx][:top_k]
    evecs = evecs[:, idx][:, :top_k]
    return evals, evecs


def principal_angle_distance(
    V_q: np.ndarray,
    V_c: np.ndarray,
) -> float:
    """
    Principal angle (subspace) distance between qPCA and classical PCA factor subspaces.
    V_q, V_c: (n x k) orthonormal factor matrices.
    Returns mean principal angle in radians or 0 if degenerate.
    """
    if V_q.size == 0 or V_c.size == 0:
        return 0.0
    # Singular values of V_q^T V_c are cos(angles)
    M = V_q.T @ V_c
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    return float(np.mean(angles))


def explained_variance_gap(
    evals_q: np.ndarray,
    evals_c: np.ndarray,
) -> float:
    """Absolute gap in cumulative explained variance (same order)."""
    cum_q = np.cumsum(evals_q) / (np.sum(evals_q) + 1e-12)
    cum_c = np.cumsum(evals_c) / (np.sum(evals_c) + 1e-12)
    k = min(len(evals_q), len(evals_c))
    return float(np.abs(cum_q[:k] - cum_c[:k]).max())


def exposure_correlation(
    exp_q: np.ndarray,
    exp_c: np.ndarray,
) -> float:
    """Mean correlation of factor exposures (per factor) between qPCA and classical."""
    k = min(exp_q.shape[1], exp_c.shape[1])
    if k == 0:
        return 0.0
    corrs = []
    for j in range(k):
        c = np.corrcoef(exp_q[:, j], exp_c[:, j])[0, 1]
        corrs.append(c if np.isfinite(c) else 0.0)
    return float(np.mean(corrs))
