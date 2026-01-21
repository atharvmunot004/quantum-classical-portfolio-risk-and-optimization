"""
Monte Carlo simulation methods for VaR/CVaR calculation.

Implements multiple simulation methods per llm.json spec:
- Historical bootstrap (iid or block)
- Parametric normal
- Parametric Student-t
- Filtered EWMA bootstrap (optional)
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


def fit_student_t_df(returns: np.ndarray, bounds: Tuple[float, float] = (2.1, 50.0)) -> float:
    """Fit Student-t degrees of freedom using MLE."""
    def neg_log_likelihood(df):
        try:
            return -np.sum(stats.t.logpdf(returns, df=df, loc=returns.mean(), scale=returns.std()))
        except:
            return np.inf
    
    result = minimize_scalar(neg_log_likelihood, bounds=bounds, method='bounded')
    if result.success:
        return max(bounds[0], min(bounds[1], result.x))
    return bounds[0] + (bounds[1] - bounds[0]) / 2


def simulate_historical_bootstrap(
    returns_window: np.ndarray,
    num_simulations: int,
    horizon: int,
    bootstrap_type: str = 'iid',
    block_length: Optional[int] = None,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate returns using historical bootstrap.
    
    Args:
        returns_window: Historical returns array (window_size,)
        num_simulations: Number of simulations
        horizon: Time horizon (simulate path and aggregate)
        bootstrap_type: 'iid' or 'block'
        block_length: Block length for block bootstrap
        random_seed: Random seed
        
    Returns:
        Simulated horizon returns (num_simulations,)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n = len(returns_window)
    if n == 0:
        return np.full(num_simulations, np.nan)
    
    if horizon == 1:
        if bootstrap_type == 'block' and block_length is not None:
            # Block bootstrap
            num_blocks = (n + block_length - 1) // block_length
            simulated = []
            for _ in range(num_simulations):
                block_idx = np.random.randint(0, num_blocks)
                start_idx = block_idx * block_length
                end_idx = min(start_idx + block_length, n)
                if start_idx < n:
                    simulated.append(returns_window[start_idx:end_idx])
            return np.concatenate(simulated)[:num_simulations]
        else:
            # IID bootstrap
            return np.random.choice(returns_window, size=num_simulations)
    else:
        # Vectorized path simulation: simulate daily returns and sum
        if bootstrap_type == 'block' and block_length is not None:
            # Block bootstrap for path - partially vectorized
            num_blocks = (n + block_length - 1) // block_length
            # Generate block indices for all simulations and horizons
            block_indices = np.random.randint(0, num_blocks, size=(num_simulations, horizon))
            # Generate positions within blocks
            positions = np.random.randint(0, block_length, size=(num_simulations, horizon))
            
            path_returns = np.zeros((num_simulations, horizon))
            for sim_idx in range(num_simulations):
                for h_idx in range(horizon):
                    block_idx = block_indices[sim_idx, h_idx]
                    start_idx = block_idx * block_length
                    end_idx = min(start_idx + block_length, n)
                    if start_idx < n:
                        pos = min(positions[sim_idx, h_idx], end_idx - start_idx - 1)
                        actual_idx = start_idx + pos
                        if actual_idx < n:
                            path_returns[sim_idx, h_idx] = returns_window[actual_idx]
            return path_returns.sum(axis=1)
        else:
            # Fully vectorized IID bootstrap for path
            # Generate random indices for all simulations and horizons at once
            indices = np.random.randint(0, n, size=(num_simulations, horizon))
            path_returns = returns_window[indices]
            return path_returns.sum(axis=1)


def simulate_parametric_normal(
    returns_window: np.ndarray,
    num_simulations: int,
    horizon: int,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """Simulate returns using parametric normal distribution."""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    mu = returns_window.mean()
    sigma = returns_window.std()
    
    if sigma <= 0 or not np.isfinite(sigma):
        return np.full(num_simulations, np.nan)
    
    if horizon == 1:
        return np.random.normal(mu, sigma, size=num_simulations)
    else:
        # Path simulation: simulate daily returns and sum
        daily_returns = np.random.normal(mu, sigma, size=(num_simulations, horizon))
        return daily_returns.sum(axis=1)


def simulate_parametric_student_t(
    returns_window: np.ndarray,
    num_simulations: int,
    horizon: int,
    df: Optional[float] = None,
    df_mode: str = 'mle_or_fixed',
    fixed_df: float = 5.0,
    bounds: Tuple[float, float] = (2.1, 50.0),
    fallback_df: float = 5.0,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Simulate returns using parametric Student-t distribution.
    
    Returns:
        Tuple of (simulated_returns, fitted_df)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    mu = returns_window.mean()
    sigma = returns_window.std()
    
    if sigma <= 0 or not np.isfinite(sigma):
        return np.full(num_simulations, np.nan), fallback_df
    
    # Fit degrees of freedom (can be pre-computed and passed via df parameter)
    if df is None:
        if df_mode == 'mle_or_fixed':
            try:
                df = fit_student_t_df(returns_window, bounds)
            except:
                df = fixed_df
        else:
            df = fixed_df
    
    df = max(bounds[0], min(bounds[1], df))
    
    # Scale parameter for Student-t
    scale = sigma * np.sqrt((df - 2) / df) if df > 2 else sigma
    
    if horizon == 1:
        simulated = stats.t.rvs(df=df, loc=mu, scale=scale, size=num_simulations)
    else:
        # Vectorized path simulation
        daily_returns = stats.t.rvs(df=df, loc=mu, scale=scale, size=(num_simulations, horizon))
        simulated = daily_returns.sum(axis=1)
    
    return simulated, df


def simulate_filtered_ewma_bootstrap(
    returns_window: np.ndarray,
    num_simulations: int,
    horizon: int,
    lambda_param: float = 0.94,
    min_variance_floor: float = 1e-12,
    bootstrap_type: str = 'iid',
    block_length: Optional[int] = None,
    mu_rule: str = 'zero_or_sample_mean',
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate returns using filtered EWMA bootstrap.
    
    Computes standardized residuals, resamples them, then reconstructs returns
    using EWMA volatility forecast.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n = len(returns_window)
    if n == 0:
        return np.full(num_simulations, np.nan)
    
    # Compute EWMA volatility
    sigma_sq = np.zeros(n)
    sigma_sq[0] = returns_window[0] ** 2
    
    for i in range(1, n):
        sigma_sq[i] = lambda_param * sigma_sq[i-1] + (1 - lambda_param) * returns_window[i-1]**2
    
    sigma = np.sqrt(np.maximum(sigma_sq, min_variance_floor))
    
    # Compute standardized residuals
    mu = returns_window.mean() if mu_rule == 'zero_or_sample_mean' else 0.0
    standardized_residuals = (returns_window - mu) / (sigma + 1e-10)
    
    # Forecast next period volatility
    next_sigma_sq = lambda_param * sigma_sq[-1] + (1 - lambda_param) * returns_window[-1]**2
    next_sigma = np.sqrt(max(next_sigma_sq, min_variance_floor))
    
    if horizon == 1:
        # Resample standardized residuals
        if bootstrap_type == 'block' and block_length is not None:
            num_blocks = (n + block_length - 1) // block_length
            resampled_residuals = []
            for _ in range(num_simulations):
                block_idx = np.random.randint(0, num_blocks)
                start_idx = block_idx * block_length
                end_idx = min(start_idx + block_length, n)
                if start_idx < n:
                    idx = np.random.randint(start_idx, end_idx)
                    resampled_residuals.append(standardized_residuals[idx])
            resampled_residuals = np.array(resampled_residuals[:num_simulations])
        else:
            resampled_residuals = np.random.choice(standardized_residuals, size=num_simulations)
        
        # Reconstruct returns
        return mu + next_sigma * resampled_residuals
    else:
        # Vectorized path simulation: simulate daily returns and sum
        current_sigma_sq = sigma_sq[-1]
        
        # Resample residuals for all paths and horizons
        if bootstrap_type == 'block' and block_length is not None:
            num_blocks = (n + block_length - 1) // block_length
            # Generate block indices
            block_indices = np.random.randint(0, num_blocks, size=(num_simulations, horizon))
            positions = np.random.randint(0, block_length, size=(num_simulations, horizon))
            
            residuals = np.zeros((num_simulations, horizon))
            for sim_idx in range(num_simulations):
                for h_idx in range(horizon):
                    block_idx = block_indices[sim_idx, h_idx]
                    start_idx = block_idx * block_length
                    end_idx = min(start_idx + block_length, n)
                    if start_idx < n:
                        pos = min(positions[sim_idx, h_idx], end_idx - start_idx - 1)
                        actual_idx = start_idx + pos
                        if actual_idx < n:
                            residuals[sim_idx, h_idx] = standardized_residuals[actual_idx]
        else:
            # Fully vectorized IID bootstrap
            residual_indices = np.random.randint(0, n, size=(num_simulations, horizon))
            residuals = standardized_residuals[residual_indices]
        
        # Simulate paths with EWMA volatility updating
        path_returns = np.zeros((num_simulations, horizon))
        path_sigma_sq = np.full(num_simulations, current_sigma_sq)
        
        for h_idx in range(horizon):
            path_sigma = np.sqrt(np.maximum(path_sigma_sq, min_variance_floor))
            path_returns[:, h_idx] = mu + path_sigma * residuals[:, h_idx]
            # Update sigma for next period
            path_sigma_sq = lambda_param * path_sigma_sq + (1 - lambda_param) * path_returns[:, h_idx]**2
        
        return path_returns.sum(axis=1)


def compute_var_cvar(
    simulated_returns: np.ndarray,
    confidence_level: float,
    tail_side: str = 'left'
) -> Tuple[float, float]:
    """
    Compute VaR and CVaR from simulated returns.
    
    Args:
        simulated_returns: Simulated returns array
        confidence_level: Confidence level (e.g., 0.95)
        tail_side: 'left' for left-tail risk
        
    Returns:
        Tuple of (VaR, CVaR) as positive loss numbers
    """
    if len(simulated_returns) == 0 or not np.any(np.isfinite(simulated_returns)):
        return np.nan, np.nan
    
    # Convert returns to losses: loss = -return
    losses = -simulated_returns
    
    # Compute VaR (quantile of loss distribution)
    # For left-tail risk at confidence_level (e.g., 0.95), we want the worst (1-confidence_level) outcomes
    # This means: "95% of the time, losses will be <= VaR"
    # So VaR is the confidence_level quantile (e.g., 95th percentile) of the loss distribution
    quantile_level = confidence_level  # e.g., 0.95 for 95% confidence
    var_idx = int(np.floor(quantile_level * len(losses)))
    var_idx = max(0, min(var_idx, len(losses) - 1))
    
    # Use partition for efficiency - partition at the quantile index
    partitioned = np.partition(losses, var_idx)
    var = partitioned[var_idx]
    
    # Ensure VaR is positive (representing a loss)
    # If var is negative, it means the quantile falls in the negative loss region (gains)
    # In this case, VaR should be the maximum positive loss, or 0 if all losses are negative
    if var < 0:
        positive_losses = losses[losses > 0]
        if len(positive_losses) > 0:
            # Use the maximum positive loss as VaR
            var = positive_losses.max()
        else:
            # All outcomes are gains, VaR should be 0
            var = 0.0
    
    # Compute CVaR: mean of losses exceeding VaR (tail losses)
    tail_losses = losses[losses >= var]
    cvar = tail_losses.mean() if len(tail_losses) > 0 else var
    
    # Ensure CVaR >= VaR and both are non-negative
    if cvar < var:
        cvar = var
    
    # Final check: both should be non-negative
    var = max(0.0, var)
    cvar = max(var, cvar)
    
    return float(var), float(cvar)


def simulate_returns_paths(
    returns_window: np.ndarray,
    method: str,
    method_config: Dict,
    num_simulations: int,
    max_horizon: int,
    random_seed: Optional[int] = None,
    cached_df: Optional[float] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Simulate daily return paths up to max_horizon using specified method.
    
    This function simulates paths once and returns the full path array,
    which can then be sliced for different horizons.
    
    Args:
        returns_window: Historical returns array
        method: Method name ('historical_bootstrap', 'parametric_normal', etc.)
        method_config: Method-specific configuration
        num_simulations: Number of simulations
        max_horizon: Maximum time horizon (simulates paths up to this)
        random_seed: Random seed
        cached_df: Cached Student-t degrees of freedom (for parametric_student_t)
        
    Returns:
        Tuple of (simulated_paths (num_simulations, max_horizon), fitted_params_dict)
    """
    fitted_params = {}
    
    if method == 'historical_bootstrap':
        bootstrap_type = method_config.get('bootstrap_type', 'iid')
        block_bootstrap = method_config.get('block_bootstrap', {})
        block_length = block_bootstrap.get('block_length') if block_bootstrap.get('enabled') else None
        
        n = len(returns_window)
        if n == 0:
            return np.full((num_simulations, max_horizon), np.nan), {}
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        if bootstrap_type == 'block' and block_length is not None:
            num_blocks = (n + block_length - 1) // block_length
            block_indices = np.random.randint(0, num_blocks, size=(num_simulations, max_horizon))
            positions = np.random.randint(0, block_length, size=(num_simulations, max_horizon))
            
            paths = np.zeros((num_simulations, max_horizon))
            for sim_idx in range(num_simulations):
                for h_idx in range(max_horizon):
                    block_idx = block_indices[sim_idx, h_idx]
                    start_idx = block_idx * block_length
                    end_idx = min(start_idx + block_length, n)
                    if start_idx < n:
                        pos = min(positions[sim_idx, h_idx], end_idx - start_idx - 1)
                        actual_idx = start_idx + pos
                        if actual_idx < n:
                            paths[sim_idx, h_idx] = returns_window[actual_idx]
        else:
            # Fully vectorized IID bootstrap
            indices = np.random.randint(0, n, size=(num_simulations, max_horizon))
            paths = returns_window[indices]
        
        return paths, {}
        
    elif method == 'parametric_normal':
        if random_seed is not None:
            np.random.seed(random_seed)
        
        mu = returns_window.mean()
        sigma = returns_window.std()
        
        if sigma <= 0 or not np.isfinite(sigma):
            return np.full((num_simulations, max_horizon), np.nan), {}
        
        paths = np.random.normal(mu, sigma, size=(num_simulations, max_horizon))
        fitted_params['window_mu'] = mu
        fitted_params['window_sigma'] = sigma
        
        return paths, fitted_params
        
    elif method == 'parametric_student_t':
        if random_seed is not None:
            np.random.seed(random_seed)
        
        mu = returns_window.mean()
        sigma = returns_window.std()
        
        if sigma <= 0 or not np.isfinite(sigma):
            df_config = method_config.get('fit', {}).get('df', {})
            fallback_df = df_config.get('fallback_df', 5.0)
            return np.full((num_simulations, max_horizon), np.nan), {'student_t_df': fallback_df}
        
        # Use cached df if provided, otherwise fit
        if cached_df is not None:
            df = cached_df
        else:
            df_config = method_config.get('fit', {}).get('df', {})
            df_mode = df_config.get('mode', 'mle_or_fixed')
            fixed_df = df_config.get('fixed_df', 5.0)
            bounds = tuple(df_config.get('bounds', [2.1, 50.0]))
            
            if df_mode == 'mle_or_fixed':
                try:
                    df = fit_student_t_df(returns_window, bounds)
                except:
                    df = fixed_df
            else:
                df = fixed_df
        
        bounds = tuple(method_config.get('fit', {}).get('df', {}).get('bounds', [2.1, 50.0]))
        df = max(bounds[0], min(bounds[1], df))
        
        scale = sigma * np.sqrt((df - 2) / df) if df > 2 else sigma
        
        paths = stats.t.rvs(df=df, loc=mu, scale=scale, size=(num_simulations, max_horizon))
        fitted_params['window_mu'] = mu
        fitted_params['window_sigma'] = sigma
        fitted_params['student_t_df'] = df
        
        return paths, fitted_params
        
    elif method == 'filtered_ewma_bootstrap':
        if random_seed is not None:
            np.random.seed(random_seed)
        
        n = len(returns_window)
        if n == 0:
            return np.full((num_simulations, max_horizon), np.nan), {}
        
        filter_config = method_config.get('filter', {})
        lambda_param = filter_config.get('lambda', 0.94)
        min_variance_floor = filter_config.get('min_variance_floor', 1e-12)
        
        # Compute EWMA volatility
        sigma_sq = np.zeros(n)
        sigma_sq[0] = returns_window[0] ** 2
        for i in range(1, n):
            sigma_sq[i] = lambda_param * sigma_sq[i-1] + (1 - lambda_param) * returns_window[i-1]**2
        
        sigma = np.sqrt(np.maximum(sigma_sq, min_variance_floor))
        
        # Compute standardized residuals
        mu = returns_window.mean() if method_config.get('reconstruction', {}).get('mu_rule', 'zero_or_sample_mean') == 'zero_or_sample_mean' else 0.0
        standardized_residuals = (returns_window - mu) / (sigma + 1e-10)
        
        current_sigma_sq = sigma_sq[-1]
        
        shock_config = method_config.get('shock_resampling', {})
        bootstrap_type = shock_config.get('bootstrap_type', 'iid')
        block_bootstrap = shock_config.get('block_bootstrap', {})
        block_length = block_bootstrap.get('block_length') if block_bootstrap.get('enabled') else None
        
        # Resample residuals
        if bootstrap_type == 'block' and block_length is not None:
            num_blocks = (n + block_length - 1) // block_length
            block_indices = np.random.randint(0, num_blocks, size=(num_simulations, max_horizon))
            positions = np.random.randint(0, block_length, size=(num_simulations, max_horizon))
            
            residuals = np.zeros((num_simulations, max_horizon))
            for sim_idx in range(num_simulations):
                for h_idx in range(max_horizon):
                    block_idx = block_indices[sim_idx, h_idx]
                    start_idx = block_idx * block_length
                    end_idx = min(start_idx + block_length, n)
                    if start_idx < n:
                        pos = min(positions[sim_idx, h_idx], end_idx - start_idx - 1)
                        actual_idx = start_idx + pos
                        if actual_idx < n:
                            residuals[sim_idx, h_idx] = standardized_residuals[actual_idx]
        else:
            residual_indices = np.random.randint(0, n, size=(num_simulations, max_horizon))
            residuals = standardized_residuals[residual_indices]
        
        # Simulate paths with EWMA volatility updating
        paths = np.zeros((num_simulations, max_horizon))
        path_sigma_sq = np.full(num_simulations, current_sigma_sq)
        
        for h_idx in range(max_horizon):
            path_sigma = np.sqrt(np.maximum(path_sigma_sq, min_variance_floor))
            paths[:, h_idx] = mu + path_sigma * residuals[:, h_idx]
            path_sigma_sq = lambda_param * path_sigma_sq + (1 - lambda_param) * paths[:, h_idx]**2
        
        fitted_params['ewma_lambda'] = lambda_param
        return paths, fitted_params
        
    else:
        raise ValueError(f"Unknown method: {method}")


def simulate_returns(
    returns_window: np.ndarray,
    method: str,
    method_config: Dict,
    num_simulations: int,
    horizon: int,
    random_seed: Optional[int] = None,
    cached_df: Optional[float] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Simulate returns using specified method.
    
    Args:
        returns_window: Historical returns array
        method: Method name ('historical_bootstrap', 'parametric_normal', etc.)
        method_config: Method-specific configuration
        num_simulations: Number of simulations
        horizon: Time horizon
        random_seed: Random seed
        
    Returns:
        Tuple of (simulated_returns, fitted_params_dict)
    """
    fitted_params = {}
    
    if method == 'historical_bootstrap':
        bootstrap_type = method_config.get('bootstrap_type', 'iid')
        block_bootstrap = method_config.get('block_bootstrap', {})
        block_length = block_bootstrap.get('block_length') if block_bootstrap.get('enabled') else None
        
        simulated = simulate_historical_bootstrap(
            returns_window, num_simulations, horizon,
            bootstrap_type, block_length, random_seed
        )
        
    elif method == 'parametric_normal':
        simulated = simulate_parametric_normal(
            returns_window, num_simulations, horizon, random_seed
        )
        fitted_params['window_mu'] = returns_window.mean()
        fitted_params['window_sigma'] = returns_window.std()
        
    elif method == 'parametric_student_t':
        df_config = method_config.get('fit', {}).get('df', {})
        df_mode = df_config.get('mode', 'mle_or_fixed')
        fixed_df = df_config.get('fixed_df', 5.0)
        bounds = tuple(df_config.get('bounds', [2.1, 50.0]))
        fallback_df = df_config.get('fallback_df', 5.0)
        
        simulated, fitted_df = simulate_parametric_student_t(
            returns_window, num_simulations, horizon,
            df=cached_df, df_mode=df_mode, fixed_df=fixed_df,
            bounds=bounds, fallback_df=fallback_df, random_seed=random_seed
        )
        fitted_params['window_mu'] = returns_window.mean()
        fitted_params['window_sigma'] = returns_window.std()
        fitted_params['student_t_df'] = fitted_df
        
    elif method == 'filtered_ewma_bootstrap':
        filter_config = method_config.get('filter', {})
        lambda_param = filter_config.get('lambda', 0.94)
        min_variance_floor = filter_config.get('min_variance_floor', 1e-12)
        
        shock_config = method_config.get('shock_resampling', {})
        bootstrap_type = shock_config.get('bootstrap_type', 'iid')
        block_bootstrap = shock_config.get('block_bootstrap', {})
        block_length = block_bootstrap.get('block_length') if block_bootstrap.get('enabled') else None
        
        reconstruction = method_config.get('reconstruction', {})
        mu_rule = reconstruction.get('mu_rule', 'zero_or_sample_mean')
        
        simulated = simulate_filtered_ewma_bootstrap(
            returns_window, num_simulations, horizon,
            lambda_param, min_variance_floor, bootstrap_type,
            block_length, mu_rule, random_seed
        )
        fitted_params['ewma_lambda'] = lambda_param
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    fitted_params['fit_success'] = np.any(np.isfinite(simulated))
    fitted_params['effective_sample_size'] = len(returns_window)
    
    return simulated, fitted_params
