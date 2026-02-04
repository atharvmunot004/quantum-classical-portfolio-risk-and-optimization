"""
Factor exposures (projection) and factor-implied risk proxies (VaR/CVaR, Gaussian).
"""
import numpy as np
from typing import Tuple, List, Dict
from scipy import stats


def factor_exposures_projection(
    returns_window: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """
    Per-asset factor exposures by projecting asset returns onto factor eigenvectors.
    returns_window: (T x N), eigenvectors: (N x k).
    Returns exposures (N x k): exposure[i, j] = loading of asset i on factor j.
    """
    # Each column of returns_window is one asset's time series. Factors are columns of evecs.
    # Projection: B = X^T @ F where F (T x k) = returns @ evecs. So asset i exposure to factor j
    # = cov(asset i, factor j) proportional to (returns[:, i] @ factor_scores[:, j]).
    # Standard: loadings = eigenvectors (each column is the loading vector for that factor).
    # So exposure[i, j] = evecs[i, j].
    return np.asarray(eigenvectors, dtype=float)


def idiosyncratic_variance(
    returns_window: np.ndarray,
    exposures: np.ndarray,
    factor_variances: np.ndarray,
) -> np.ndarray:
    """
    Per-asset idiosyncratic variance = total var - factor-explained var.
    returns_window (T x N), exposures (N x k), factor_variances (k,).
    """
    n_assets = returns_window.shape[1]
    total_var = np.var(returns_window, axis=0)
    # Factor-explained var for asset i: sum_j exposure[i,j]^2 * factor_var[j]
    explained_var = np.sum(exposures ** 2 * factor_variances, axis=1)
    idio = total_var - explained_var
    idio = np.maximum(idio, 1e-12)
    return idio


def factor_gaussian_var_cvar(
    exposures: np.ndarray,
    factor_variances: np.ndarray,
    idiosyncratic_var: np.ndarray,
    confidence_level: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Factor-implied VaR and CVaR (Gaussian) per asset.
    exposures (N x k), factor_variances (k,), idiosyncratic_var (N,).
    Returns (var_per_asset, cvar_per_asset) for left tail (losses).
    """
    # Total variance of asset i = sum_j exposure[i,j]^2 * factor_var[j] + idio[i]
    total_var = np.sum(exposures ** 2 * factor_variances, axis=1) + idiosyncratic_var
    total_std = np.sqrt(np.maximum(total_var, 1e-12))
    # VaR (left tail): -mu + z_alpha * sigma; assuming zero mean, VaR = z_alpha * sigma (positive = loss)
    alpha = 1 - confidence_level
    z = stats.norm.ppf(alpha)
    var_factor = -z * total_std  # z is negative for alpha < 0.5
    # CVaR (Gaussian): sigma * phi(z_alpha) / alpha
    cvar_factor = total_std * stats.norm.pdf(z) / alpha
    return var_factor, cvar_factor


def factor_shock_implied_loss(
    exposures: np.ndarray,
    factor_std: np.ndarray,
    shock_sigma: float,
) -> np.ndarray:
    """
    Factor shock scenario: shock each factor by shock_sigma std, implied loss per asset.
    exposures (N x k), factor_std (k,). Returns (N,) loss per asset.
    """
    # Loss = - return; factor shock delta_f = shock_sigma * factor_std; delta_r_i = sum_j exposure[i,j] * delta_f[j]
    delta_f = shock_sigma * factor_std
    delta_return = exposures @ delta_f
    loss = -delta_return
    return loss
