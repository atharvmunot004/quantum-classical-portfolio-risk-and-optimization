"""
Risk Parity / Equal Risk Contribution (ERC) Portfolio Optimizer.

Implements the Equal Risk Contribution portfolio optimization method, which aims
to equalize the risk contribution of each asset in the portfolio.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import time
import warnings
import logging

logger = logging.getLogger(__name__)

from sklearn.covariance import LedoitWolf
try:
    from sklearn.covariance import OAS
    OAS_AVAILABLE = True
except ImportError:
    OAS_AVAILABLE = False
    logger.warning("OAS estimator not available, will use Ledoit-Wolf as fallback")

# Try to import optimization libraries (optional, not used in fixed-point mode)
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def compute_covariance_matrix(
    returns: pd.DataFrame,
    method: str = 'sample',
    window: Optional[int] = None,
    use_shrinkage: bool = False,
    shrinkage_method: str = 'ledoit_wolf',
    ewma_lambda: float = 0.94,
    ensure_psd: bool = True,
    min_eigenvalue: float = 1e-10,
    skip_psd_if_positive: bool = True
) -> Tuple[pd.DataFrame, float]:
    """
    Compute covariance matrix from returns.
    
    Args:
        returns: DataFrame of returns with dates as index and assets as columns
        method: 'sample', 'oas', or 'ewma'
        window: Optional rolling window size (if None, uses all data)
        use_shrinkage: Whether to apply shrinkage estimator (for sample method)
        shrinkage_method: Shrinkage method ('ledoit_wolf' or 'oas')
        ewma_lambda: Decay factor for EWMA (default 0.94)
        ensure_psd: Ensure positive semi-definite matrix
        min_eigenvalue: Minimum eigenvalue threshold for PSD fix
        
    Returns:
        Tuple of (covariance_matrix, computation_time_ms)
    """
    start_time = time.time()
    
    if window is not None and len(returns) > window:
        returns_window = returns.iloc[-window:].copy()
    else:
        returns_window = returns.copy()
    
    returns_window = returns_window.dropna(axis=1, how='any')
    
    if len(returns_window.columns) == 0:
        raise ValueError("No assets with sufficient data for covariance estimation")
    
    # Compute covariance based on method
    if method == 'sample':
        if use_shrinkage:
            if shrinkage_method == 'ledoit_wolf':
                lw = LedoitWolf()
                cov_array = lw.fit(returns_window.values).covariance_
            elif shrinkage_method == 'oas':
                if OAS_AVAILABLE:
                    oas = OAS()
                    cov_array = oas.fit(returns_window.values).covariance_
                else:
                    logger.warning("OAS not available, using Ledoit-Wolf instead")
                    lw = LedoitWolf()
                    cov_array = lw.fit(returns_window.values).covariance_
            else:
                logger.warning(f"Unknown shrinkage method {shrinkage_method}, using sample covariance")
                cov_array = returns_window.cov().values
        else:
            cov_array = returns_window.cov().values
            
    elif method == 'oas':
        if OAS_AVAILABLE:
            oas = OAS()
            cov_array = oas.fit(returns_window.values).covariance_
        else:
            logger.warning("OAS not available, using Ledoit-Wolf instead")
            lw = LedoitWolf()
            cov_array = lw.fit(returns_window.values).covariance_
        
    elif method == 'ewma':
        # EWMA covariance estimation
        cov_array = _compute_ewma_covariance(returns_window.values, ewma_lambda)
        
    else:
        logger.warning(f"Unknown method {method}, using sample covariance")
        cov_array = returns_window.cov().values
    
    # Ensure PSD
    if ensure_psd:
        cov_array = _ensure_psd(cov_array, min_eigenvalue, skip_if_positive=skip_psd_if_positive)
    
    cov_matrix = pd.DataFrame(
        cov_array,
        index=returns_window.columns,
        columns=returns_window.columns
    )
    
    computation_time = (time.time() - start_time) * 1000
    logger.debug(f"Covariance estimation ({method}) took {computation_time:.2f}ms")
    
    return cov_matrix, computation_time


def _compute_ewma_covariance(returns: np.ndarray, lambda_param: float = 0.94) -> np.ndarray:
    """Compute EWMA covariance matrix (vectorized for performance)."""
    n_obs, n_assets = returns.shape
    
    # Vectorized weight computation
    indices = np.arange(n_obs)
    weights = (1 - lambda_param) * (lambda_param ** (n_obs - 1 - indices))
    weights = weights / weights.sum()
    
    # Vectorized mean computation
    mean_returns = np.average(returns, axis=0, weights=weights)
    centered = returns - mean_returns
    
    # Vectorized covariance computation using outer product
    # weights is (n_obs,), centered is (n_obs, n_assets)
    # We want: sum(weights[i] * centered[i, :] @ centered[i, :].T) for all i
    weighted_centered = centered * weights[:, np.newaxis]  # (n_obs, n_assets)
    cov = weighted_centered.T @ centered  # (n_assets, n_assets)
    
    return cov


def _ensure_psd(cov_matrix: np.ndarray, min_eigenvalue: float = 1e-10, skip_if_positive: bool = True) -> np.ndarray:
    """Ensure covariance matrix is positive semi-definite using eigenvalue clipping."""
    if skip_if_positive:
        # Quick check: if all diagonal elements are positive and matrix is symmetric
        if np.all(np.diag(cov_matrix) > 0):
            # Check minimum eigenvalue only
            min_eig = np.linalg.eigvalsh(cov_matrix)[0]
            if min_eig >= min_eigenvalue:
                return cov_matrix  # Already PSD, skip expensive reconstruction
    
    # Full PSD fix
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    eigenvals = np.maximum(eigenvals, min_eigenvalue)
    cov_psd = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    return cov_psd


def calculate_risk_contributions(
    weights: np.ndarray,
    covariance_matrix: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate risk contributions for each asset in the portfolio.
    
    The risk contribution of asset i is:
    RC_i = w_i * (Σ * w)_i / σ_p
    
    where:
    - w_i: weight of asset i
    - (Σ * w)_i: marginal contribution to risk
    - σ_p: portfolio volatility
    
    Args:
        weights: Portfolio weights array of shape (n_assets,)
        covariance_matrix: Covariance matrix DataFrame
        
    Returns:
        Tuple of (risk_contributions, marginal_risk_contributions, portfolio_volatility)
    """
    weights = weights.reshape(-1, 1)
    Sigma = covariance_matrix.values
    
    # Portfolio variance
    portfolio_variance = (weights.T @ Sigma @ weights)[0, 0]
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    if portfolio_volatility == 0:
        return np.zeros(len(weights)), np.zeros(len(weights)), 0.0
    
    # Marginal risk contributions: Σ * w
    marginal_contrib = (Sigma @ weights).flatten()
    
    # Risk contributions: w_i * (Σ * w)_i / σ_p
    risk_contrib = (weights.flatten() * marginal_contrib) / portfolio_volatility
    
    return risk_contrib, marginal_contrib, portfolio_volatility


def optimize_risk_parity_portfolio_fixed_point(
    covariance_matrix: pd.DataFrame,
    constraints: Dict,
    max_iterations: int = 100,
    convergence_tol: float = 1e-6,
    early_stop_threshold: Optional[float] = None
) -> Tuple[pd.Series, Dict, float]:
    """
    Optimize ERC portfolio using fixed-point cyclic coordinate descent.
    
    This is a fast iterative algorithm that doesn't require scipy.
    Algorithm: iteratively update weights to equalize risk contributions.
    
    Args:
        covariance_matrix: Covariance matrix DataFrame
        constraints: Dictionary of portfolio constraints
        max_iterations: Maximum number of iterations
        convergence_tol: Convergence tolerance
        
    Returns:
        Tuple of (optimal_weights, optimization_info, solver_time_ms)
    """
    start_time = time.time()
    
    n = len(covariance_matrix)
    Sigma = covariance_matrix.values
    
    # Get constraints
    weight_bounds = constraints.get('weight_bounds', [0.0, 1.0])
    min_weight = max(0.0, weight_bounds[0])  # Long-only
    max_weight = min(1.0, weight_bounds[1])
    max_weight_per_asset = constraints.get('max_weight_per_asset', 1.0)
    max_weight = min(max_weight, max_weight_per_asset)
    
    # Initialize with equal weights
    w = np.ones(n) / n
    
    # Fixed-point iteration
    for iteration in range(max_iterations):
        w_old = w.copy()
        
        # Calculate portfolio volatility
        portfolio_var = w.T @ Sigma @ w
        portfolio_std = np.sqrt(portfolio_var)
        
        if portfolio_std < 1e-10:
            break
        
        # Calculate marginal risk contributions: Σ * w
        marginal_contrib = Sigma @ w
        
        # Calculate risk contributions: w_i * (Σ * w)_i / σ_p
        risk_contrib = w * marginal_contrib / portfolio_std
        
        # Update weights using inverse risk contributions (to equalize)
        # w_i ∝ 1 / RC_i, normalized to sum to 1
        inv_rc = 1.0 / (risk_contrib + 1e-10)  # Add small epsilon to avoid division by zero
        w = inv_rc / inv_rc.sum()
        
        # Project to constraints
        w = np.clip(w, min_weight, max_weight)
        w = w / w.sum()  # Renormalize
        
        # Check convergence
        weight_change = np.max(np.abs(w - w_old))
        if weight_change < convergence_tol:
            break
        
        # Early stopping on risk parity deviation
        if early_stop_threshold is not None:
            portfolio_var = w.T @ Sigma @ w
            portfolio_std = np.sqrt(portfolio_var)
            if portfolio_std > 1e-10:
                marginal_contrib = Sigma @ w
                risk_contrib = w * marginal_contrib / portfolio_std
                target_rc = portfolio_std / n
                deviation = np.sqrt(np.mean((risk_contrib - target_rc)**2)) / target_rc if target_rc > 0 else np.inf
                if deviation <= early_stop_threshold:
                    break
    
    solver_time = (time.time() - start_time) * 1000
    
    # Calculate final risk contributions for info
    portfolio_var = w.T @ Sigma @ w
    portfolio_std = np.sqrt(portfolio_var)
    marginal_contrib = Sigma @ w
    risk_contrib = w * marginal_contrib / portfolio_std
    target_rc = portfolio_std / n
    deviation = np.sqrt(np.mean((risk_contrib - target_rc)**2)) / target_rc if target_rc > 0 else 0.0
    
    optimal_weights_series = pd.Series(w, index=covariance_matrix.index)
    
    opt_info = {
        'status': 'success' if weight_change < convergence_tol else 'max_iterations',
        'iterations': iteration + 1,
        'convergence': weight_change,
        'risk_parity_deviation': deviation
    }
    
    return optimal_weights_series, opt_info, solver_time


def optimize_risk_parity_portfolio(
    covariance_matrix: pd.DataFrame,
    constraints: Dict,
    optimization_settings: Dict,
    initial_weights: Optional[np.ndarray] = None,
    previous_weights: Optional[np.ndarray] = None,
    perf_opts: Optional[Dict] = None
) -> Tuple[pd.Series, Dict, float]:
    """
    Optimize portfolio for Equal Risk Contribution (ERC).
    
    The optimization problem minimizes the sum of squared differences between
    each asset's risk contribution and the target equal risk contribution.
    
    Objective: minimize sum((RC_i - target_RC)^2)
    
    where target_RC = σ_p / n (equal risk contribution per asset)
    
    Args:
        covariance_matrix: Covariance matrix DataFrame
        constraints: Dictionary of portfolio constraints
        optimization_settings: Dictionary of optimization solver settings
        initial_weights: Optional initial weights guess (default: equal weights)
        previous_weights: Previous period weights for turnover constraint
        
    Returns:
        Tuple of (optimal_weights, optimization_info, solver_time_ms)
    """
    # Check if fixed-point mode is enabled
    rp_settings = optimization_settings.get('risk_parity_settings', {}) if isinstance(optimization_settings, dict) else {}
    solver_config = rp_settings.get('solver', {}) if isinstance(rp_settings, dict) else {}
    
    # Also check in perf_opts
    if perf_opts:
        exec_mode = perf_opts.get('execution_mode', {})
        if exec_mode.get('strategy') == 'static_erc_single_solve':
            disable_nlp = rp_settings.get('disable_nonlinear_programming', False)
            disable_scipy = rp_settings.get('disable_scipy_minimize', False)
            
            if disable_nlp or disable_scipy:
                # Use fixed-point algorithm
                max_iter = solver_config.get('max_iterations', 100)
                conv_tol = solver_config.get('convergence_tol', 1e-6)
                return optimize_risk_parity_portfolio_fixed_point(
                    covariance_matrix,
                    constraints,
                    max_iterations=max_iter,
                    convergence_tol=conv_tol
                )
    
    # Fallback to scipy-based optimization (legacy)
    start_time = time.time()
    
    if not SCIPY_AVAILABLE:
        # Try fixed-point as fallback
        logger.warning("scipy not available, using fixed-point algorithm")
        return optimize_risk_parity_portfolio_fixed_point(
            covariance_matrix,
            constraints,
            max_iterations=100,
            convergence_tol=1e-6
        )
    
    n = len(covariance_matrix)
    Sigma = covariance_matrix.values
    
    # Get performance optimization settings
    if perf_opts is None:
        perf_opts = {}
    solver_opts = perf_opts.get('risk_parity_solver_optimization', {})
    warm_start = solver_opts.get('warm_start', {})
    early_stop = solver_opts.get('early_stopping', {})
    tolerance_sched = solver_opts.get('tolerance_schedule', {})
    
    # Get objective settings
    obj_settings = optimization_settings.get('objective', {})
    loss_type = obj_settings.get('loss', 'squared')
    epsilon = obj_settings.get('epsilon_for_numerical_stability', 1e-12)
    
    # Warm start: use previous weights if available
    if initial_weights is None:
        if warm_start.get('enable', False) and warm_start.get('use_previous_rebalance_weights', False) and previous_weights is not None:
            w0 = previous_weights.copy()
            if len(w0) != n:
                w0 = np.ones(n) / n
        else:
            w0 = np.ones(n) / n
    else:
        w0 = initial_weights.copy()
        if len(w0) != n:
            raise ValueError(f"Initial weights length {len(w0)} does not match assets {n}")
    
    # Objective function: minimize sum of squared differences in risk contributions
    def objective(w):
        w = w.reshape(-1, 1)
        portfolio_var = (w.T @ Sigma @ w)[0, 0]
        portfolio_std = np.sqrt(portfolio_var + epsilon)
        
        if portfolio_std == 0:
            return 1e10  # Penalty for zero volatility
        
        # Risk contributions
        marginal_contrib = (Sigma @ w).flatten()
        risk_contrib = w.flatten() * marginal_contrib / portfolio_std
        
        # Target: equal risk contributions
        target_contrib = portfolio_std / n
        
        # Loss function
        if loss_type == 'squared':
            return np.sum((risk_contrib - target_contrib) ** 2)
        elif loss_type == 'absolute':
            return np.sum(np.abs(risk_contrib - target_contrib))
        else:
            return np.sum((risk_contrib - target_contrib) ** 2)
    
    # Constraints
    constraint_list = []
    
    # Fully invested constraint: sum(w) = 1
    if constraints.get('fully_invested', True):
        constraint_list.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
    
    # Turnover constraint (can be disabled via constraints_acceleration)
    constraints_accel = perf_opts.get('constraints_acceleration', {}) if perf_opts else {}
    disable_hard_turnover = constraints_accel.get('turnover_constraint', {}).get('disable_hard_constraint', False)
    
    if not disable_hard_turnover and previous_weights is not None and constraints.get('turnover_constraint', {}).get('enable', False):
        max_turnover = constraints['turnover_constraint'].get('max_turnover_per_rebalance', 0.30)
        constraint_list.append({
            'type': 'ineq',
            'fun': lambda w: max_turnover - np.sum(np.abs(w - previous_weights))
        })
    
    # Bounds
    weight_bounds = constraints.get('weight_bounds', [0.0, 1.0])
    min_weight = weight_bounds[0]
    max_weight = weight_bounds[1]
    
    # Long-only constraint
    if constraints.get('long_only', True) or constraints.get('no_short_selling', False):
        min_weight = max(0.0, min_weight)
    
    # Max weight per asset constraint
    max_weight_per_asset = constraints.get('max_weight_per_asset', 1.0)
    max_weight = min(max_weight, max_weight_per_asset)
    
    bounds = [(min_weight, max_weight) for _ in range(n)]
    
    # Solver stack reduction
    solver_stack_reduction = solver_opts.get('solver_stack_reduction', {})
    if solver_stack_reduction.get('enable', False):
        primary = solver_stack_reduction.get('primary_solver', 'slsqp')
        fallback = solver_stack_reduction.get('fallback_solver', 'none')
        solver_stack = [
            {'backend': 'scipy', 'solver': primary, 'tolerance': optimization_settings.get('tolerance', 1e-8), 'max_iterations': optimization_settings.get('max_iterations', 5000)}
        ]
        if fallback != 'none':
            solver_stack.append({'backend': 'scipy', 'solver': fallback, 'tolerance': optimization_settings.get('tolerance', 1e-8), 'max_iterations': optimization_settings.get('max_iterations', 5000)})
        disable_multi_start = solver_stack_reduction.get('disable_multi_start_if_warm_start_available', False)
        if disable_multi_start and previous_weights is not None:
            n_starts = 1
        else:
            n_starts = min(solver_stack_reduction.get('max_multi_starts', 1), 1)
        noise_scale = 0.02
    else:
        solver_stack = optimization_settings.get('solver_stack', [])
        if not solver_stack:
            solver_stack = [{
                'backend': 'scipy',
                'solver': 'slsqp',
                'tolerance': optimization_settings.get('tolerance', 1e-8),
                'max_iterations': optimization_settings.get('max_iterations', 5000)
            }]
        multi_start = optimization_settings.get('initialization', {}).get('multi_start', {})
        n_starts = multi_start.get('n_starts', 1) if multi_start.get('enable', False) else 1
        noise_scale = multi_start.get('noise_scale', 0.02)
    
    # Tolerance schedule
    if tolerance_sched.get('adaptive_tolerance', False):
        initial_tol = tolerance_sched.get('initial_tolerance', 1e-5)
        final_tol = tolerance_sched.get('final_tolerance', 1e-8)
    else:
        initial_tol = optimization_settings.get('tolerance', 1e-8)
        final_tol = initial_tol
    
    best_result = None
    best_obj = np.inf
    best_weights = None
    best_info = None
    solver_used = None
    
    # Early stopping threshold
    early_stop_threshold = early_stop.get('risk_parity_deviation_threshold', 1e-4) if early_stop.get('enable', False) else None
    max_no_improvement = early_stop.get('max_no_improvement_iterations', 50) if early_stop.get('enable', False) else None
    
    # Try multiple starts
    no_improvement_count = 0
    for start_idx in range(n_starts):
        if start_idx > 0:
            # Add noise to initial guess
            w0_perturbed = w0 + np.random.normal(0, noise_scale, n)
            w0_perturbed = np.clip(w0_perturbed, min_weight, max_weight)
            w0_perturbed = w0_perturbed / w0_perturbed.sum()
        else:
            w0_perturbed = w0
        
        # Try each solver in stack
        for solver_idx, solver_config in enumerate(solver_stack):
            backend = solver_config.get('backend', 'scipy')
            solver_name = solver_config.get('solver', 'slsqp')
            
            # Adaptive tolerance: start loose, tighten if needed
            if tolerance_sched.get('adaptive_tolerance', False) and solver_idx == 0:
                tolerance = initial_tol
            else:
                tolerance = solver_config.get('tolerance', final_tol)
            
            max_iter = solver_config.get('max_iterations', 5000)
            
            if backend == 'scipy':
                if solver_name == 'slsqp':
                    method = 'SLSQP'
                elif solver_name == 'trust-constr':
                    method = 'trust-constr'
                else:
                    method = 'SLSQP'
            else:
                continue
            
            try:
                result = minimize(
                    objective,
                    w0_perturbed,
                    method=method,
                    bounds=bounds,
                    constraints=constraint_list,
                    options={
                        'ftol': tolerance,
                        'maxiter': max_iter,
                        'disp': False
                    }
                )
                
                # Check early stopping
                if early_stop_threshold is not None and result.success:
                    # Calculate risk parity deviation
                    w_test = result.x.reshape(-1, 1)
                    var_test = (w_test.T @ Sigma @ w_test)[0, 0]
                    std_test = np.sqrt(var_test + epsilon)
                    if std_test > 0:
                        marginal_test = (Sigma @ w_test).flatten()
                        rc_test = w_test.flatten() * marginal_test / std_test
                        target_rc = std_test / n
                        deviation = np.sqrt(np.mean((rc_test - target_rc)**2)) / target_rc if target_rc > 0 else np.inf
                        
                        if deviation <= early_stop_threshold:
                            # Good enough, use this result
                            best_result = result
                            best_obj = result.fun
                            best_weights = result.x
                            solver_used = f"{backend}_{solver_name}"
                            best_info = {
                                'status': 'success',
                                'message': 'Early stopping: threshold reached',
                                'iterations': getattr(result, 'nit', None),
                                'objective_value': result.fun,
                                'solver': solver_used,
                                'start': start_idx,
                                'early_stopped': True
                            }
                            break
                
                if result.success and result.fun < best_obj:
                    best_result = result
                    best_obj = result.fun
                    best_weights = result.x
                    solver_used = f"{backend}_{solver_name}"
                    best_info = {
                        'status': 'success',
                        'message': getattr(result, 'message', 'Optimization successful'),
                        'iterations': getattr(result, 'nit', None),
                        'objective_value': result.fun,
                        'solver': solver_used,
                        'start': start_idx
                    }
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if max_no_improvement is not None and no_improvement_count >= max_no_improvement:
                        break
                    
            except Exception as e:
                logger.debug(f"Solver {solver_name} failed: {e}")
                continue
        
        if best_weights is not None and early_stop_threshold is not None:
            break  # Early stopping triggered
    
    # Fallback if no solver succeeded
    if best_weights is None:
        logger.warning("All solvers failed, using equal weights or previous weights")
        if previous_weights is not None:
            best_weights = previous_weights.copy()
        else:
            best_weights = np.ones(n) / n
        best_info = {
            'status': 'failed',
            'message': 'All solvers failed, using fallback weights',
            'solver': 'fallback'
        }
    else:
        # Ensure weights sum to 1 and respect bounds
        best_weights = best_weights / np.sum(best_weights)
        best_weights = np.clip(best_weights, min_weight, max_weight)
        best_weights = best_weights / np.sum(best_weights)
    
    solver_time = (time.time() - start_time) * 1000
    
    # Create weights Series
    optimal_weights_series = pd.Series(best_weights, index=covariance_matrix.index)
    
    return optimal_weights_series, best_info, solver_time

