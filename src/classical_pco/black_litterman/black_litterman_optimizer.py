"""
Black-Litterman Portfolio Optimizer.

Implements the Black-Litterman model for portfolio optimization with investor views.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.covariance import LedoitWolf
import time
import warnings

# Try to import GPU acceleration
try:
    from .gpu_acceleration import (
        get_array_module,
        to_gpu_array,
        to_cpu_array,
        compute_covariance_gpu,
        matrix_inverse_gpu,
        matrix_multiply_gpu,
        solve_linear_system_gpu,
        is_gpu_available,
        clear_gpu_cache
    )
    GPU_ACCELERATION_AVAILABLE = True
except ImportError:
    GPU_ACCELERATION_AVAILABLE = False
    is_gpu_available = lambda: False
    clear_gpu_cache = lambda: None

# Try to import optimization libraries
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    # Only warn in main process, suppress in worker processes
    import multiprocessing
    try:
        # If this is a worker process, multiprocessing.current_process() will have a name != 'MainProcess'
        current_process = multiprocessing.current_process()
        if current_process.name == 'MainProcess':
            warnings.warn("cvxpy not available. Using scipy.optimize instead.", UserWarning, stacklevel=2)
    except (AttributeError, RuntimeError):
        # Fallback: warn anyway if we can't determine (e.g., before multiprocessing is initialized)
        warnings.warn("cvxpy not available. Using scipy.optimize instead.", UserWarning, stacklevel=2)

try:
    import osqp
    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False

if not CVXPY_AVAILABLE:
    from scipy.optimize import minimize


def compute_covariance_matrix(
    returns: pd.DataFrame,
    method: str = 'sample',
    window: Optional[int] = None,
    use_shrinkage: bool = False,
    shrinkage_method: str = 'ledoit_wolf',
    estimation_windows: Optional[List[int]] = None,
    use_gpu: bool = False
) -> Tuple[pd.DataFrame, float]:
    """
    Compute covariance matrix from returns.
    
    Args:
        returns: DataFrame of returns with dates as index and assets as columns
        method: 'sample' for sample covariance
        window: Optional rolling window size (if None, uses all data)
        use_shrinkage: Whether to apply shrinkage estimator
        shrinkage_method: Shrinkage method ('ledoit_wolf')
        estimation_windows: Optional list of estimation windows (uses first if provided)
        
    Returns:
        Tuple of (covariance_matrix, computation_time_ms)
    """
    start_time = time.time()
    
    # Use estimation_windows if provided, otherwise use window
    if estimation_windows is not None and len(estimation_windows) > 0:
        window = estimation_windows[0]  # Use first window
    
    if window is not None and len(returns) > window:
        returns_window = returns.iloc[-window:]
    else:
        returns_window = returns
    
    returns_window = returns_window.dropna(axis=1, how='any')
    
    if len(returns_window.columns) == 0:
        raise ValueError("No assets with sufficient data for covariance estimation")
    
    # Use GPU acceleration if available and requested
    if use_gpu and GPU_ACCELERATION_AVAILABLE and is_gpu_available():
        try:
            cov_array = compute_covariance_gpu(returns_window, use_gpu=True)
            cov_array = to_cpu_array(cov_array)  # Convert back to CPU for DataFrame
            cov_matrix = pd.DataFrame(
                cov_array,
                index=returns_window.columns,
                columns=returns_window.columns
            )
        except Exception as e:
            warnings.warn(f"GPU covariance computation failed, falling back to CPU: {e}")
            use_gpu = False
    
    if not use_gpu or not GPU_ACCELERATION_AVAILABLE:
        # CPU computation
        if use_shrinkage and shrinkage_method == 'ledoit_wolf':
            lw = LedoitWolf()
            cov_array = lw.fit(returns_window.values).covariance_
            cov_matrix = pd.DataFrame(
                cov_array,
                index=returns_window.columns,
                columns=returns_window.columns
            )
        else:
            cov_matrix = returns_window.cov()
    
    computation_time = (time.time() - start_time) * 1000
    
    return cov_matrix, computation_time


def derive_market_equilibrium_returns(
    returns: pd.DataFrame,
    baseline_portfolios: pd.DataFrame,
    covariance_matrix: pd.DataFrame,
    risk_aversion: float = 2.5,
    derive_from_portfolios: bool = True,
    market_weight_source: str = 'baseline_portfolios',
    use_cap_weighted: bool = False
) -> Tuple[pd.Series, pd.Series, float]:
    """
    Derive market equilibrium returns (prior) from baseline portfolios.
    
    The Black-Litterman model uses market equilibrium returns as the prior:
    π = λ * Σ * w_market
    
    where:
    - π: equilibrium returns (prior)
    - λ: risk aversion parameter
    - Σ: covariance matrix
    - w_market: market weights (derived from baseline portfolios)
    
    Args:
        returns: DataFrame of returns
        baseline_portfolios: DataFrame of baseline portfolios
        covariance_matrix: Covariance matrix DataFrame
        risk_aversion: Risk aversion parameter (lambda)
        derive_from_portfolios: Whether to derive weights from portfolios
        market_weight_source: Source of market weights
        use_cap_weighted: Whether to use cap-weighted benchmark
        
    Returns:
        Tuple of (prior_returns, market_weights, computation_time_ms)
    """
    start_time = time.time()
    
    # Align assets
    assets = returns.columns.intersection(covariance_matrix.index)
    
    if len(assets) == 0:
        raise ValueError("No common assets for market equilibrium computation")
    
    # Get market weights from baseline portfolios
    if derive_from_portfolios and market_weight_source == 'baseline_portfolios':
        # Use average weights across all baseline portfolios
        portfolio_weights_list = []
        
        for portfolio_id in baseline_portfolios.index:
            portfolio_data = baseline_portfolios.loc[portfolio_id]
            if isinstance(portfolio_data, pd.Series):
                # Extract weights if available
                weights = {}
                for asset in assets:
                    if asset in portfolio_data.index:
                        weight = portfolio_data[asset]
                        if pd.notna(weight) and weight > 0:
                            weights[asset] = weight
                
                if weights:
                    portfolio_weights_list.append(pd.Series(weights))
        
        if portfolio_weights_list:
            # Average weights across portfolios
            avg_weights = pd.concat(portfolio_weights_list, axis=1).mean(axis=1)
            market_weights = avg_weights.reindex(assets, fill_value=0)
            # Normalize
            total_weight = market_weights.sum()
            if total_weight > 0:
                market_weights = market_weights / total_weight
            else:
                # Equal weights fallback
                market_weights = pd.Series(1.0 / len(assets), index=assets)
        else:
            # Equal weights fallback
            market_weights = pd.Series(1.0 / len(assets), index=assets)
    else:
        # Equal weights fallback
        market_weights = pd.Series(1.0 / len(assets), index=assets)
    
    # Align covariance matrix
    Sigma = covariance_matrix.loc[assets, assets].values
    w_market = market_weights[assets].values
    
    # Compute equilibrium returns: π = λ * Σ * w_market
    prior_returns_array = risk_aversion * (Sigma @ w_market)
    
    # Annualize if needed (assuming daily returns)
    prior_returns_array = prior_returns_array * 252
    
    prior_returns = pd.Series(prior_returns_array, index=assets)
    
    computation_time = (time.time() - start_time) * 1000
    
    return prior_returns, market_weights, computation_time


def generate_synthetic_views(
    returns: pd.DataFrame,
    num_views: int = 5,
    view_generation_method: str = 'return_differentials',
    random_seed: Optional[int] = None
) -> Dict:
    """
    Generate synthetic investor views based on return patterns.
    
    Args:
        returns: DataFrame of returns
        num_views: Number of views to generate
        view_generation_method: Method for generating views ('return_differentials')
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing synthetic views
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    asset_names = returns.columns.tolist()
    num_assets = len(asset_names)
    
    if num_views > num_assets * (num_assets - 1) // 2:
        num_views = num_assets * (num_assets - 1) // 2
    
    views = {
        "absolute_views": [],
        "relative_views": []
    }
    
    if view_generation_method == 'return_differentials':
        # Compute mean returns
        mean_returns = returns.mean()
        
        # Generate relative views based on return differentials
        # Select pairs with largest return differences
        return_pairs = []
        for i, asset1 in enumerate(asset_names):
            for j, asset2 in enumerate(asset_names[i+1:], start=i+1):
                return_diff = mean_returns[asset1] - mean_returns[asset2]
                return_pairs.append((asset1, asset2, return_diff))
        
        # Sort by absolute return difference
        return_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Generate views from top pairs
        for i, (asset1, asset2, return_diff) in enumerate(return_pairs[:num_views]):
            # Create relative view: asset1 will outperform asset2
            confidence = np.random.choice([0.50, 0.75, 0.90])
            views["relative_views"].append({
                "assets": [asset1, asset2],
                "return_diff": return_diff * 252,  # Annualize
                "confidence": confidence
            })
    
    return views


def parse_and_scale_views(
    investor_views: Dict,
    asset_names: List[str],
    covariance_matrix: pd.DataFrame,
    tau: float = 0.025,
    view_type: str = 'absolute_relative_mixed',
    uncertainty_matrix: str = 'diagonal',
    confidence_levels: List[float] = [0.50, 0.75, 0.90]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Parse investor views and construct P, Q, and Omega matrices.
    
    P: Pick matrix (k x n) - maps views to assets
    Q: View vector (k x 1) - expected returns from views
    Omega: Uncertainty matrix (k x k) - uncertainty in views
    
    Args:
        investor_views: Dictionary containing investor views
        asset_names: List of asset names
        covariance_matrix: Covariance matrix DataFrame
        tau: Scaling factor for uncertainty
        view_type: Type of views ('absolute_relative_mixed')
        uncertainty_matrix: Type of uncertainty matrix ('diagonal', 'idzorek')
        confidence_levels: List of confidence levels for views
        
    Returns:
        Tuple of (P, Q, Omega, computation_time_ms)
    """
    start_time = time.time()
    
    # Parse views
    absolute_views = investor_views.get('absolute_views', [])
    relative_views = investor_views.get('relative_views', [])
    
    num_views = len(absolute_views) + len(relative_views)
    num_assets = len(asset_names)
    
    if num_views == 0:
        # No views - return empty matrices
        P = np.zeros((0, num_assets))
        Q = np.zeros((0, 1))
        Omega = np.zeros((0, 0))
        return P, Q, Omega, (time.time() - start_time) * 1000
    
    # Initialize matrices
    P = np.zeros((num_views, num_assets))
    Q = np.zeros((num_views, 1))
    
    view_idx = 0
    
    # Process absolute views
    for view in absolute_views:
        asset = view.get('asset')
        if asset in asset_names:
            asset_idx = asset_names.index(asset)
            P[view_idx, asset_idx] = 1.0
            Q[view_idx, 0] = view.get('return', 0.0)
            view_idx += 1
    
    # Process relative views
    for view in relative_views:
        assets = view.get('assets', [])
        return_diff = view.get('return_diff', 0.0)
        
        if len(assets) >= 2:
            asset1 = assets[0]
            asset2 = assets[1]
            
            if asset1 in asset_names and asset2 in asset_names:
                asset1_idx = asset_names.index(asset1)
                asset2_idx = asset_names.index(asset2)
                P[view_idx, asset1_idx] = 1.0
                P[view_idx, asset2_idx] = -1.0
                Q[view_idx, 0] = return_diff
                view_idx += 1
    
    # Trim matrices if some views were invalid
    if view_idx < num_views:
        P = P[:view_idx, :]
        Q = Q[:view_idx, :]
        num_views = view_idx
    
    # Construct Omega (uncertainty matrix)
    if uncertainty_matrix == 'diagonal':
        # Diagonal Omega: Omega = tau * P * Σ * P^T
        Sigma = covariance_matrix.loc[asset_names, asset_names].values
        Omega = tau * (P @ Sigma @ P.T)
        
        # Ensure diagonal
        Omega = np.diag(np.diag(Omega))
    elif uncertainty_matrix == 'idzorek':
        # Idzorek method: uses confidence levels
        Sigma = covariance_matrix.loc[asset_names, asset_names].values
        base_omega = tau * (P @ Sigma @ P.T)
        
        # Use confidence levels if provided
        confidences = []
        for view in absolute_views + relative_views:
            conf = view.get('confidence', confidence_levels[0] if confidence_levels else 0.5)
            confidences.append(conf)
        
        # Pad if needed
        while len(confidences) < num_views:
            confidences.append(confidence_levels[0] if confidence_levels else 0.5)
        
        # Scale by confidence: higher confidence = lower uncertainty
        omega_diag = np.diag(base_omega) / np.array(confidences[:num_views])
        Omega = np.diag(omega_diag)
    else:
        # Default: diagonal
        Sigma = covariance_matrix.loc[asset_names, asset_names].values
        Omega = tau * (P @ Sigma @ P.T)
        Omega = np.diag(np.diag(Omega))
    
    computation_time = (time.time() - start_time) * 1000
    
    return P, Q, Omega, computation_time


def compute_posterior_bl_returns(
    prior_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.025,
    use_gpu: bool = False
) -> Tuple[pd.Series, float]:
    """
    Compute Black-Litterman posterior returns.
    
    μ_BL = [(τΣ)^(-1) + P^T * Omega^(-1) * P]^(-1) * [(τΣ)^(-1) * π + P^T * Omega^(-1) * Q]
    
    Args:
        prior_returns: Prior (equilibrium) returns Series
        covariance_matrix: Covariance matrix DataFrame
        P: Pick matrix (k x n)
        Q: View vector (k x 1)
        Omega: Uncertainty matrix (k x k)
        tau: Scaling factor
        
    Returns:
        Tuple of (posterior_returns, computation_time_ms)
    """
    start_time = time.time()
    
    assets = prior_returns.index.intersection(covariance_matrix.index)
    if len(assets) == 0:
        raise ValueError("No common assets for posterior computation")
    
    pi = prior_returns[assets].values.reshape(-1, 1)
    Sigma = covariance_matrix.loc[assets, assets].values
    n = len(assets)
    
    # Handle case with no views
    if P.shape[0] == 0:
        # No views - posterior equals prior
        posterior_returns = prior_returns.copy()
        return posterior_returns, (time.time() - start_time) * 1000
    
    # Use GPU acceleration if available and requested
    # Do not use GPU for small matrices (n <= 10) per JSON spec
    use_gpu_for_inverse = use_gpu and GPU_ACCELERATION_AVAILABLE and is_gpu_available() and n > 10
    
    if use_gpu_for_inverse:
        try:
            xp = get_array_module(use_gpu=True)
            pi_gpu = to_gpu_array(pi, use_gpu=True)
            Sigma_gpu = to_gpu_array(Sigma, use_gpu=True)
            P_gpu = to_gpu_array(P, use_gpu=True)
            Q_gpu = to_gpu_array(Q, use_gpu=True)
            Omega_gpu = to_gpu_array(Omega, use_gpu=True)
            
            # Compute (τΣ)^(-1)
            tau_Sigma_gpu = tau * Sigma_gpu
            tau_Sigma_inv = matrix_inverse_gpu(tau_Sigma_gpu, use_gpu=True)
            
            # Compute Omega^(-1)
            Omega_inv = matrix_inverse_gpu(Omega_gpu, use_gpu=True)
            
            # Compute M_inv = [(τΣ)^(-1) + P^T * Omega^(-1) * P]
            P_T = xp.transpose(P_gpu)
            M_inv = tau_Sigma_inv + matrix_multiply_gpu(
                matrix_multiply_gpu(P_T, Omega_inv, use_gpu=True),
                P_gpu,
                use_gpu=True
            )
            
            # Compute M = M_inv^(-1)
            M = matrix_inverse_gpu(M_inv, use_gpu=True)
            
            # Compute posterior mean: μ_BL = M * [(τΣ)^(-1) * π + P^T * Omega^(-1) * Q]
            term1 = matrix_multiply_gpu(tau_Sigma_inv, pi_gpu, use_gpu=True)
            term2 = matrix_multiply_gpu(
                matrix_multiply_gpu(P_T, Omega_inv, use_gpu=True),
                Q_gpu,
                use_gpu=True
            )
            mu_bl_gpu = matrix_multiply_gpu(
                M,
                term1 + term2,
                use_gpu=True
            )
            
            # Convert back to CPU
            mu_bl = to_cpu_array(mu_bl_gpu)
        except Exception as e:
            warnings.warn(f"GPU posterior computation failed, falling back to CPU: {e}")
            use_gpu_for_inverse = False
    
    if not use_gpu_for_inverse:
        # CPU computation
        # Compute (τΣ)^(-1)
        tau_Sigma = tau * Sigma
        try:
            tau_Sigma_inv = np.linalg.inv(tau_Sigma)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            tau_Sigma_inv = np.linalg.pinv(tau_Sigma)
        
        # Compute Omega^(-1)
        try:
            Omega_inv = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            Omega_inv = np.linalg.pinv(Omega)
        
        # Compute posterior covariance of returns
        # M = [(τΣ)^(-1) + P^T * Omega^(-1) * P]^(-1)
        M_inv = tau_Sigma_inv + P.T @ Omega_inv @ P
        
        try:
            M = np.linalg.inv(M_inv)
        except np.linalg.LinAlgError:
            M = np.linalg.pinv(M_inv)
        
        # Compute posterior mean
        # μ_BL = M * [(τΣ)^(-1) * π + P^T * Omega^(-1) * Q]
        mu_bl = M @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q)
    
    posterior_returns = pd.Series(mu_bl.flatten(), index=assets)
    
    computation_time = (time.time() - start_time) * 1000
    
    return posterior_returns, computation_time


def compute_posterior_covariance(
    covariance_matrix: pd.DataFrame,
    P: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.025,
    method: str = 'black_litterman',
    use_gpu: bool = False
) -> Tuple[pd.DataFrame, float]:
    """
    Compute Black-Litterman posterior covariance matrix.
    
    Σ_BL = Σ + M
    
    where M = [(τΣ)^(-1) + P^T * Omega^(-1) * P]^(-1)
    
    Args:
        covariance_matrix: Prior covariance matrix DataFrame
        P: Pick matrix (k x n)
        Omega: Uncertainty matrix (k x k)
        tau: Scaling factor
        method: Method for computing posterior covariance
        
    Returns:
        Tuple of (posterior_covariance, computation_time_ms)
    """
    start_time = time.time()
    
    assets = covariance_matrix.index
    Sigma = covariance_matrix.values
    n = len(assets)
    
    if method == 'black_litterman':
        # Handle case with no views
        if P.shape[0] == 0:
            # No views - posterior equals prior
            posterior_cov = covariance_matrix.copy()
            return posterior_cov, (time.time() - start_time) * 1000
        
        # Use GPU acceleration if available and requested
        # Do not use GPU for small matrices (n <= 10) per JSON spec
        use_gpu_for_inverse = use_gpu and GPU_ACCELERATION_AVAILABLE and is_gpu_available() and n > 10
        
        if use_gpu_for_inverse:
            try:
                xp = get_array_module(use_gpu=True)
                Sigma_gpu = to_gpu_array(Sigma, use_gpu=True)
                P_gpu = to_gpu_array(P, use_gpu=True)
                Omega_gpu = to_gpu_array(Omega, use_gpu=True)
                
                # Compute (τΣ)^(-1)
                tau_Sigma_gpu = tau * Sigma_gpu
                tau_Sigma_inv = matrix_inverse_gpu(tau_Sigma_gpu, use_gpu=True)
                
                # Compute Omega^(-1)
                Omega_inv = matrix_inverse_gpu(Omega_gpu, use_gpu=True)
                
                # Compute M = [(τΣ)^(-1) + P^T * Omega^(-1) * P]^(-1)
                P_T = xp.transpose(P_gpu)
                M_inv = tau_Sigma_inv + matrix_multiply_gpu(
                    matrix_multiply_gpu(P_T, Omega_inv, use_gpu=True),
                    P_gpu,
                    use_gpu=True
                )
                M = matrix_inverse_gpu(M_inv, use_gpu=True)
                
                # Posterior covariance: Σ_BL = Σ + M
                posterior_cov_array = to_cpu_array(Sigma_gpu + M)
            except Exception as e:
                warnings.warn(f"GPU posterior covariance computation failed, falling back to CPU: {e}")
                use_gpu_for_inverse = False
        
        if not use_gpu_for_inverse:
            # CPU computation
            # Compute (τΣ)^(-1)
            tau_Sigma = tau * Sigma
            try:
                tau_Sigma_inv = np.linalg.inv(tau_Sigma)
            except np.linalg.LinAlgError:
                tau_Sigma_inv = np.linalg.pinv(tau_Sigma)
            
            # Compute Omega^(-1)
            try:
                Omega_inv = np.linalg.inv(Omega)
            except np.linalg.LinAlgError:
                Omega_inv = np.linalg.pinv(Omega)
            
            # Compute M = [(τΣ)^(-1) + P^T * Omega^(-1) * P]^(-1)
            M_inv = tau_Sigma_inv + P.T @ Omega_inv @ P
            
            try:
                M = np.linalg.inv(M_inv)
            except np.linalg.LinAlgError:
                M = np.linalg.pinv(M_inv)
            
            # Posterior covariance: Σ_BL = Σ + M
            posterior_cov_array = Sigma + M
    else:
        # Default: use prior covariance
        posterior_cov_array = Sigma
    
    posterior_cov = pd.DataFrame(
        posterior_cov_array,
        index=assets,
        columns=assets
    )
    
    computation_time = (time.time() - start_time) * 1000
    
    return posterior_cov, computation_time


def optimize_portfolio(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    objective: str = 'maximize_posterior_sharpe',
    risk_aversion: Optional[float] = None,
    risk_free_rate: float = 0.0,
    constraints: Optional[Dict] = None,
    solver: str = 'osqp',
    tolerance: float = 1e-6,
    fallback_solver: Optional[str] = None,
    use_closed_form_when_possible: bool = True
) -> Tuple[pd.Series, Dict, float]:
    """
    Optimize portfolio using posterior returns and covariance.
    
    Args:
        expected_returns: Series of expected returns (posterior)
        covariance_matrix: Covariance matrix DataFrame (posterior)
        objective: Optimization objective
        risk_aversion: Risk aversion parameter (lambda)
        risk_free_rate: Risk-free rate
        constraints: Dictionary of constraints
        solver: Solver backend
        tolerance: Optimization tolerance
        
    Returns:
        Tuple of (optimal_weights, optimization_info, solver_time_ms)
    """
    start_time = time.time()
    
    assets = expected_returns.index.intersection(covariance_matrix.index)
    if len(assets) == 0:
        raise ValueError("No common assets between expected returns and covariance matrix")
    
    mu = expected_returns[assets].values
    Sigma = covariance_matrix.loc[assets, assets].values
    n = len(assets)
    
    if constraints is None:
        constraints = {
            'long_only': True,
            'fully_invested': True,
            'no_short_selling': True,
            'weight_bounds': [0.0, 1.0],
            'max_weight_per_asset': None
        }
    
    # Try closed-form solution first if requested
    if solver == 'closed_form_preferred' or solver == 'closed_form' or (use_closed_form_when_possible and solver != 'closed_form'):
        # Closed-form solution for mean-variance optimization
        # w* = (1/λ) * Σ^(-1) * (μ - rf * 1)
        # With constraints: long-only, fully-invested, no max_weight_per_asset
        
        use_closed_form = (
            constraints.get('fully_invested', True) and
            constraints.get('long_only', True) and
            constraints.get('max_weight_per_asset') is None and
            constraints.get('weight_bounds', [0.0, 1.0]) == [0.0, 1.0] and
            objective in ['mean_variance', 'risk_return_tradeoff']
        )
        
        if use_closed_form:
            try:
                if risk_aversion is None:
                    risk_aversion = 1.0
                
                # Closed-form: w* = (1/λ) * Σ^(-1) * (μ - rf * 1)
                ones = np.ones(n)
                mu_excess = mu - risk_free_rate * ones
                
                try:
                    Sigma_inv = np.linalg.inv(Sigma)
                except np.linalg.LinAlgError:
                    Sigma_inv = np.linalg.pinv(Sigma)
                
                weights_unconstrained = (1.0 / risk_aversion) * (Sigma_inv @ mu_excess)
                
                # Apply long-only constraint (clip negative weights)
                weights_array = np.maximum(weights_unconstrained, 0.0)
                
                # Normalize to satisfy fully-invested constraint
                weight_sum = weights_array.sum()
                if weight_sum > 0:
                    weights_array = weights_array / weight_sum
                else:
                    # Fallback to equal weights if all negative
                    weights_array = np.ones(n) / n
                
                # Check if solution satisfies constraints
                if np.all(weights_array >= 0) and np.abs(weights_array.sum() - 1.0) < tolerance:
                    solver_time = (time.time() - start_time) * 1000
                    optimization_info = {
                        'status': 'optimal',
                        'objective_value': float(weights_array @ mu - risk_aversion * (weights_array @ Sigma @ weights_array)),
                        'solver': 'closed_form'
                    }
                else:
                    # Fall through to numerical solver
                    use_closed_form = False
            except Exception as e:
                # Fall through to numerical solver
                use_closed_form = False
                warnings.warn(f"Closed-form solution failed: {e}. Falling back to numerical solver.")
        
        if not use_closed_form:
            # Fallback to numerical solver
            if solver == 'closed_form_preferred':
                solver = fallback_solver if fallback_solver is not None else 'osqp'
            elif solver == 'closed_form':
                solver = fallback_solver if fallback_solver is not None else 'osqp'
    
    if CVXPY_AVAILABLE and solver in ['cvxpy', 'osqp']:
        w = cp.Variable(n)
        
        if objective == 'maximize_posterior_sharpe' or objective == 'mean_variance':
            # Maximize Sharpe ratio: (μ^T w - rf) / sqrt(w^T Σ w)
            # Reformulated as: minimize -μ^T w + λ * w^T Σ w
            if risk_aversion is None:
                risk_aversion = 1.0
            objective_func = cp.Minimize(
                -cp.sum(mu @ w) + risk_aversion * cp.quad_form(w, Sigma)
            )
        elif objective == 'risk_return_tradeoff':
            if risk_aversion is None:
                risk_aversion = 1.0
            objective_func = cp.Minimize(
                -cp.sum(mu @ w) + risk_aversion * cp.quad_form(w, Sigma)
            )
        elif objective == 'min_variance':
            objective_func = cp.Minimize(cp.quad_form(w, Sigma))
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        constraint_list = []
        
        if constraints.get('fully_invested', True):
            constraint_list.append(cp.sum(w) == 1.0)
        
        if constraints.get('long_only', True) or constraints.get('no_short_selling', True):
            constraint_list.append(w >= 0.0)
        
        weight_bounds = constraints.get('weight_bounds', [0.0, 1.0])
        constraint_list.append(w >= weight_bounds[0])
        constraint_list.append(w <= weight_bounds[1])
        
        max_weight = constraints.get('max_weight_per_asset')
        if max_weight is not None:
            constraint_list.append(w <= max_weight)
        
        problem = cp.Problem(objective_func, constraint_list)
        
        if solver == 'osqp' and OSQP_AVAILABLE:
            problem.solve(solver=cp.OSQP, eps_abs=tolerance, eps_rel=tolerance)
        else:
            problem.solve(solver=cp.ECOS, abstol=tolerance, reltol=tolerance)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise RuntimeError(f"Optimization failed with status: {problem.status}")
        
        weights_array = w.value
        solver_time = (time.time() - start_time) * 1000
        
        optimization_info = {
            'status': problem.status,
            'objective_value': problem.value,
            'solver': solver
        }
    else:
        # Fallback to scipy.optimize
        if objective == 'min_variance':
            def objective_func(w):
                return w @ Sigma @ w
        elif objective == 'risk_return_tradeoff':
            if risk_aversion is None:
                risk_aversion = 1.0
            def objective_func(w):
                return -mu @ w + risk_aversion * (w @ Sigma @ w)
        else:
            if risk_aversion is None:
                risk_aversion = 1.0
            def objective_func(w):
                return -mu @ w + risk_aversion * (w @ Sigma @ w)
        
        constraints_opt = []
        
        if constraints.get('fully_invested', True):
            constraints_opt.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            })
        
        bounds = []
        weight_bounds = constraints.get('weight_bounds', [0.0, 1.0])
        max_weight = constraints.get('max_weight_per_asset')
        
        for i in range(n):
            if max_weight is not None:
                bounds.append((weight_bounds[0], min(weight_bounds[1], max_weight)))
            else:
                bounds.append((weight_bounds[0], weight_bounds[1]))
        
        w0 = np.ones(n) / n  # Initial guess: equal weights
        
        # Try SLSQP first
        result = minimize(
            objective_func,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_opt,
            options={'ftol': tolerance, 'maxiter': 1000}
        )
        
        # If SLSQP fails, try L-BFGS-B with relaxed constraints
        if not result.success:
            try:
                # Try with L-BFGS-B (doesn't support equality constraints, so we'll use penalty)
                def objective_with_penalty(w):
                    obj = objective_func(w)
                    # Add penalty for constraint violation
                    constraint_violation = abs(np.sum(w) - 1.0)
                    penalty = 1e6 * constraint_violation
                    return obj + penalty
                
                result = minimize(
                    objective_with_penalty,
                    w0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': tolerance, 'maxiter': 1000}
                )
                
                # Normalize to satisfy fully-invested constraint
                if result.success:
                    weights_array = result.x
                    weights_array = np.maximum(weights_array, 0.0)  # Ensure non-negative
                    weight_sum = weights_array.sum()
                    if weight_sum > 0:
                        weights_array = weights_array / weight_sum
                    else:
                        weights_array = np.ones(n) / n
                    result.success = True
                    result.fun = objective_func(weights_array)
            except Exception:
                pass
        
        # If still failed, use closed-form approximation as last resort
        if not result.success:
            warnings.warn(f"scipy.optimize failed, using closed-form approximation as fallback")
            try:
                if risk_aversion is None:
                    risk_aversion = 1.0
                
                # Closed-form approximation
                ones = np.ones(n)
                mu_excess = mu - risk_free_rate * ones
                
                try:
                    Sigma_inv = np.linalg.inv(Sigma)
                except np.linalg.LinAlgError:
                    Sigma_inv = np.linalg.pinv(Sigma)
                
                weights_unconstrained = (1.0 / risk_aversion) * (Sigma_inv @ mu_excess)
                weights_array = np.maximum(weights_unconstrained, 0.0)
                weight_sum = weights_array.sum()
                if weight_sum > 0:
                    weights_array = weights_array / weight_sum
                else:
                    weights_array = np.ones(n) / n
                
                # Apply max_weight constraint if needed
                max_weight = constraints.get('max_weight_per_asset')
                if max_weight is not None:
                    weights_array = np.minimum(weights_array, max_weight)
                    weights_array = weights_array / weights_array.sum()  # Renormalize
                
                result.success = True
                result.fun = objective_func(weights_array)
            except Exception as e:
                # Ultimate fallback: equal weights
                warnings.warn(f"All optimization methods failed, using equal weights: {e}")
                weights_array = np.ones(n) / n
                result.success = True
                result.fun = objective_func(weights_array)
        
        # Use result.x if successful, otherwise use fallback weights_array
        if result.success and hasattr(result, 'x'):
            weights_array = result.x
        solver_time = (time.time() - start_time) * 1000
        
        optimization_info = {
            'status': 'optimal' if result.success else 'failed',
            'objective_value': float(result.fun),
            'solver': 'scipy' if result.success else 'fallback'
        }
    
    optimal_weights = pd.Series(weights_array, index=assets)
    
    return optimal_weights, optimization_info, solver_time


def generate_efficient_frontier(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    num_portfolios: int = 50,
    constraints: Optional[Dict] = None,
    risk_free_rate: float = 0.0,
    solver: str = 'osqp'
) -> pd.DataFrame:
    """
    Generate efficient frontier using posterior returns and covariance.
    
    Args:
        expected_returns: Series of expected returns (posterior)
        covariance_matrix: Covariance matrix DataFrame (posterior)
        num_portfolios: Number of portfolios on frontier
        constraints: Dictionary of constraints
        risk_free_rate: Risk-free rate
        solver: Solver backend
        
    Returns:
        DataFrame with columns: expected_return, volatility, sharpe_ratio, weights
    """
    assets = expected_returns.index.intersection(covariance_matrix.index)
    mu = expected_returns[assets].values
    Sigma = covariance_matrix.loc[assets, assets].values
    n = len(assets)
    
    if constraints is None:
        constraints = {
            'long_only': True,
            'fully_invested': True,
            'weight_bounds': [0.0, 1.0]
        }
    
    # Find min and max return portfolios
    min_var_weights, _, _ = optimize_portfolio(
        expected_returns[assets],
        covariance_matrix.loc[assets, assets],
        objective='min_variance',
        constraints=constraints,
        solver=solver
    )
    
    min_var_return = mu @ min_var_weights.values
    min_var_vol = np.sqrt(min_var_weights.values @ Sigma @ min_var_weights.values)
    
    # Max return (unconstrained)
    max_return_idx = np.argmax(mu)
    max_return = mu[max_return_idx]
    
    # Generate target returns
    target_returns = np.linspace(min_var_return, max_return, num_portfolios)
    
    frontier_data = []
    
    for target_return in target_returns:
        try:
            # Optimize for target return
            if CVXPY_AVAILABLE:
                w = cp.Variable(n)
                objective = cp.Minimize(cp.quad_form(w, Sigma))
                
                constraint_list = [
                    cp.sum(w) == 1.0,
                    mu @ w >= target_return
                ]
                
                if constraints.get('long_only', True):
                    constraint_list.append(w >= 0.0)
                
                weight_bounds = constraints.get('weight_bounds', [0.0, 1.0])
                constraint_list.append(w >= weight_bounds[0])
                constraint_list.append(w <= weight_bounds[1])
                
                problem = cp.Problem(objective, constraint_list)
                problem.solve(solver=cp.ECOS if solver != 'osqp' else cp.OSQP)
                
                if problem.status in ['optimal', 'optimal_inaccurate']:
                    weights = w.value
                    exp_return = mu @ weights
                    volatility = np.sqrt(weights @ Sigma @ weights)
                    sharpe = (exp_return - risk_free_rate) / volatility if volatility > 0 else np.nan
                    
                    frontier_data.append({
                        'expected_return': exp_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe,
                        **{asset: weight for asset, weight in zip(assets, weights)}
                    })
        except Exception:
            continue
    
    if len(frontier_data) == 0:
        return pd.DataFrame()
    
    frontier_df = pd.DataFrame(frontier_data)
    
    return frontier_df

