"""Performance metrics for GARCH VaR/CVaR evaluation.

Correctness improvements:
- Quantile loss is always defined, even when there are zero violations.
- RMSE between VaR/CVaR and realized losses is computed on the FULL sample (not only violations),
  which avoids NaNs and is standard for forecast evaluation.
- Exceedance magnitude stats remain NaN when there are zero exceedances.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def compute_tail_metrics(returns: pd.Series, var_series: pd.Series, confidence_level: float = 0.95) -> Dict[str, float]:
    """Tail metrics for VaR using IEEE loss convention."""
    losses = -returns
    violations = losses > var_series
    alpha = float(1.0 - confidence_level)

    # Pinball (quantile) loss is always defined
    quantile_loss = float(np.mean(alpha * np.maximum(losses - var_series, 0.0) + (1.0 - alpha) * np.maximum(var_series - losses, 0.0)))

    # RMSE on full sample is always defined (if non-empty)
    rmse_full = float(np.sqrt(np.mean((var_series - losses) ** 2))) if len(losses) else np.nan

    if int(violations.sum()) == 0:
        return {
            'mean_exceedance': np.nan,
            'max_exceedance': np.nan,
            'std_exceedance': np.nan,
            'quantile_loss_score': quantile_loss,
            'rmse_var_vs_losses': rmse_full,
            'num_violations': 0,
            'configuration_valid': True
        }

    exceedances = losses[violations] - var_series[violations]

    return {
        'mean_exceedance': float(exceedances.mean()),
        'max_exceedance': float(exceedances.max()),
        'std_exceedance': float(exceedances.std()),
        'quantile_loss_score': quantile_loss,
        'rmse_var_vs_losses': rmse_full,
        'num_violations': int(violations.sum()),
        'configuration_valid': True
    }


def compute_cvar_tail_metrics(
    returns: pd.Series,
    cvar_series: pd.Series,
    var_series: pd.Series,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """Tail metrics for CVaR (Expected Shortfall) using IEEE loss convention."""
    losses = -returns
    cvar_violations = losses > cvar_series

    rmse_full = float(np.sqrt(np.mean((cvar_series - losses) ** 2))) if len(losses) else np.nan

    if int(cvar_violations.sum()) == 0:
        return {
            'cvar_mean_exceedance': np.nan,
            'cvar_max_exceedance': np.nan,
            'cvar_std_exceedance': np.nan,
            'rmse_cvar_vs_losses': rmse_full,
            'cvar_num_violations': 0
        }

    cvar_exceedances = losses[cvar_violations] - cvar_series[cvar_violations]

    return {
        'cvar_mean_exceedance': float(cvar_exceedances.mean()),
        'cvar_max_exceedance': float(cvar_exceedances.max()),
        'cvar_std_exceedance': float(cvar_exceedances.std()),
        'rmse_cvar_vs_losses': rmse_full,
        'cvar_num_violations': int(cvar_violations.sum())
    }


def compute_structure_metrics(portfolio_weights: pd.Series, covariance_matrix: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Portfolio structure metrics."""
    w = portfolio_weights.to_numpy(dtype=float)
    active = w[w > 1e-10]

    hhi = float(np.sum(w ** 2))
    enc = float(1.0 / hhi) if hhi > 0 else np.nan

    cond_num = np.nan
    if covariance_matrix is not None:
        try:
            cov = covariance_matrix.to_numpy(dtype=float)
            eig = np.linalg.eigvals(cov)
            eig = eig[np.real(eig) > 1e-10]
            if len(eig) > 0:
                cond_num = float(np.max(np.real(eig)) / np.min(np.real(eig)))
        except Exception:
            cond_num = np.nan

    return {
        'portfolio_size': int(len(active)),
        'num_active_assets': int(len(active)),
        'hhi_concentration': hhi,
        'effective_number_of_assets': enc,
        'covariance_condition_number': cond_num
    }


def compute_distribution_metrics(returns: pd.Series) -> Dict[str, float]:
    """Distribution diagnostics (skew, excess kurtosis, Jarque-Bera)."""
    r = returns.dropna()
    if len(r) < 3:
        return {'skewness': np.nan, 'kurtosis': np.nan, 'jarque_bera_p_value': np.nan, 'jarque_bera_statistic': np.nan}

    skewness = float(stats.skew(r))
    kurtosis = float(stats.kurtosis(r))  # excess kurtosis
    jb_stat, jb_p_value = stats.jarque_bera(r)

    return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'jarque_bera_p_value': float(jb_p_value),
        'jarque_bera_statistic': float(jb_stat)
    }


def compute_garch_metrics(garch_params: Dict[str, float], conditional_vol: pd.Series) -> Dict[str, float]:
    """GARCH-specific metrics (persistence, long-run vol, etc.)."""
    alpha = garch_params.get('alpha', np.nan)
    beta = garch_params.get('beta', np.nan)
    omega = garch_params.get('omega', np.nan)
    alpha_plus_beta = garch_params.get('alpha_plus_beta', np.nan)
    unconditional_variance = garch_params.get('unconditional_variance', np.nan)

    out = {
        'omega': omega,
        'alpha': alpha,
        'beta': beta,
        'alpha_plus_beta': alpha_plus_beta,
        'unconditional_variance': unconditional_variance,
        'last_conditional_sigma': garch_params.get('last_conditional_sigma', np.nan),
        'fit_success': float(bool(garch_params.get('fit_success', False))),
        'convergence_flag': float(bool(garch_params.get('convergence_flag', False))),
        'loglikelihood': garch_params.get('loglikelihood', np.nan),
        'aic': garch_params.get('aic', np.nan),
        'bic': garch_params.get('bic', np.nan)
    }

    if not np.isnan(alpha_plus_beta) and 0 < alpha_plus_beta < 1:
        out['persistence_half_life'] = float(np.log(0.5) / np.log(alpha_plus_beta))
    else:
        out['persistence_half_life'] = np.nan

    if not np.isnan(omega) and not np.isnan(alpha_plus_beta) and (1 - alpha_plus_beta) > 0:
        lr_var = omega / (1.0 - alpha_plus_beta)
        out['long_run_volatility'] = float(np.sqrt(lr_var)) if lr_var > 0 else np.nan
    else:
        out['long_run_volatility'] = np.nan

    return out


def compute_runtime_metrics(runtimes: list[float]) -> Dict[str, float]:
    """Runtime summary statistics (ms)."""
    if len(runtimes) == 0:
        return {
            'runtime_per_portfolio_ms': np.nan,
            'p95_runtime_ms': np.nan,
            'mean_runtime_ms': np.nan,
            'median_runtime_ms': np.nan,
            'min_runtime_ms': np.nan,
            'max_runtime_ms': np.nan
        }

    a = np.asarray(runtimes, dtype=float)
    return {
        'runtime_per_portfolio_ms': float(np.mean(a) * 1000.0),
        'p95_runtime_ms': float(np.percentile(a, 95) * 1000.0),
        'mean_runtime_ms': float(np.mean(a) * 1000.0),
        'median_runtime_ms': float(np.median(a) * 1000.0),
        'min_runtime_ms': float(np.min(a) * 1000.0),
        'max_runtime_ms': float(np.max(a) * 1000.0)
    }
