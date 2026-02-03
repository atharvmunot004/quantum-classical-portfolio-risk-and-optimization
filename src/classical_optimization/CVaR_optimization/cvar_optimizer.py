"""
CVaR Portfolio Optimizer using Rockafellar-Uryasev Linear Programming Formulation.

Implements the linear programming approach to CVaR optimization as described in:
Rockafellar, R.T. and Uryasev, S. (2000). Optimization of conditional value-at-risk.
Journal of Risk, 2, 21-41.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import time
import warnings

# Try to import optimization libraries
try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. CVaR optimization may not work.")

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

try:
    # Try to import HiGHS solver (if available)
    from scipy.optimize import linprog
    HIGHS_AVAILABLE = True
except ImportError:
    HIGHS_AVAILABLE = False


def setup_cvar_linear_program(
    scenario_matrix: np.ndarray,
    expected_returns: Optional[np.ndarray] = None,
    confidence_level: float = 0.95,
    target_return: Optional[float] = None,
    lambda_cvar: float = 1.0,
    lambda_mean_return: float = 0.0,
    constraints: Optional[Dict] = None
) -> Dict:
    """
    Set up CVaR optimization as a linear program using Rockafellar-Uryasev formulation.
    
    The optimization problem is:
    minimize: lambda_cvar * CVaR_alpha + lambda_mean_return * (-mean_return)
    subject to: constraints
    
    Where CVaR_alpha = VaR_alpha + (1/(1-alpha)) * E[max(0, -portfolio_return - VaR_alpha)]
    
    This is reformulated as a linear program:
    minimize: lambda_cvar * (VaR + (1/(1-alpha)) * sum(z_s) / S) + lambda_mean_return * (-mean_return)
    subject to:
        z_s >= -portfolio_return_s - VaR for all scenarios s
        z_s >= 0 for all scenarios s
        portfolio constraints
    
    Args:
        scenario_matrix: Scenario matrix of shape (num_scenarios, num_assets)
        expected_returns: Expected returns array of shape (num_assets,)
        confidence_level: Confidence level alpha (e.g., 0.95 for 95% CVaR)
        target_return: Optional target return constraint
        lambda_cvar: Weight on CVaR in objective
        lambda_mean_return: Weight on (negative) mean return in objective
        constraints: Dictionary of portfolio constraints
        
    Returns:
        Dictionary with LP formulation details
    """
    num_scenarios, num_assets = scenario_matrix.shape
    
    if expected_returns is None:
        # Use sample mean from scenarios
        expected_returns = scenario_matrix.mean(axis=0)
    
    # Alpha parameter (tail probability)
    alpha = confidence_level
    
    # Set up decision variables:
    # - w: portfolio weights (num_assets)
    # - VaR: Value at Risk (scalar)
    # - z_s: auxiliary variables for each scenario (num_scenarios)
    num_vars = num_assets + 1 + num_scenarios  # weights + VaR + z variables
    
    # Objective function coefficients
    # minimize: lambda_cvar * (VaR + (1/(1-alpha)) * mean(z)) + lambda_mean_return * (-w^T * mu)
    c = np.zeros(num_vars)
    
    # Coefficient for VaR
    c[num_assets] = lambda_cvar
    
    # Coefficient for z variables: lambda_cvar * (1/((1-alpha) * num_scenarios))
    z_coeff = lambda_cvar / ((1 - alpha) * num_scenarios)
    c[num_assets + 1:] = z_coeff
    
    # Coefficient for mean return (negative because we minimize)
    if lambda_mean_return > 0:
        c[:num_assets] = -lambda_mean_return * expected_returns
    
    # Constraints matrix A and bounds b
    # Constraint 1: z_s >= -portfolio_return_s - VaR for all scenarios s
    # This is: z_s + VaR + sum(w_i * return_si) >= 0
    # Rearranged: z_s + VaR + w^T * return_s >= 0
    
    # Build constraint matrix
    A_ub = []  # Inequality constraints (upper bounds)
    b_ub = []  # Right-hand side
    
    # For each scenario: z_s + VaR + portfolio_return_s >= 0
    for s in range(num_scenarios):
        row = np.zeros(num_vars)
        # Coefficients for weights: -return_s (because we have -portfolio_return_s)
        row[:num_assets] = -scenario_matrix[s, :]
        # Coefficient for VaR: 1
        row[num_assets] = 1
        # Coefficient for z_s: 1
        row[num_assets + 1 + s] = 1
        A_ub.append(row)
        b_ub.append(0.0)
    
    # Constraint 2: z_s >= 0 (handled by variable bounds)
    
    # Portfolio constraints
    A_eq = []  # Equality constraints
    b_eq = []  # Right-hand side
    
    # Fully invested constraint: sum(w_i) = 1
    if constraints.get('fully_invested', True):
        row = np.zeros(num_vars)
        row[:num_assets] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)
    
    # Target return constraint (if specified)
    if target_return is not None:
        row = np.zeros(num_vars)
        row[:num_assets] = expected_returns
        A_eq.append(row)
        b_eq.append(target_return)
    
    # Variable bounds
    # Weights bounds
    weight_bounds = constraints.get('weight_bounds', [0.0, 1.0])
    min_weight = weight_bounds[0] if constraints.get('long_only', True) or constraints.get('no_short_selling', True) else -np.inf
    max_weight = weight_bounds[1]
    max_weight_per_asset = constraints.get('max_weight_per_asset', 1.0)
    max_weight = min(max_weight, max_weight_per_asset)
    
    bounds = []
    # Weight bounds
    for _ in range(num_assets):
        bounds.append((min_weight, max_weight))
    
    # VaR bounds: need to bound from below to prevent unbounded problem
    # When minimizing CVaR without return constraints, VaR can go to -infinity
    # We bound VaR based on feasible portfolio returns
    # Lower bound: worst-case portfolio return given constraints
    if constraints.get('long_only', True) or constraints.get('no_short_selling', True):
        # For long-only portfolios: minimum return is the worst return across all scenarios
        # This happens when we invest 100% in the worst-performing asset
        worst_case_return = np.min(scenario_matrix)
    else:
        # For portfolios with shorting: compute worst-case considering weight bounds
        # Worst case: all negative weights at minimum, all positive at maximum
        worst_returns = np.min(scenario_matrix, axis=0)
        best_returns = np.max(scenario_matrix, axis=0)
        # Conservative estimate: assume we can short up to -1.0 per asset
        worst_case_return = np.sum(np.minimum(worst_returns * max_weight, best_returns * min_weight))
    
    # Set VaR lower bound: allow VaR to be below worst-case by a reasonable margin
    # This ensures the problem is bounded while allowing flexibility
    var_lower_bound = worst_case_return - 10.0  # Allow significant downside margin
    
    # Upper bound: best-case portfolio return
    if constraints.get('long_only', True) or constraints.get('no_short_selling', True):
        best_case_return = np.max(scenario_matrix)
    else:
        best_returns = np.max(scenario_matrix, axis=0)
        worst_returns = np.min(scenario_matrix, axis=0)
        best_case_return = np.sum(np.maximum(best_returns * max_weight, worst_returns * min_weight))
    
    var_upper_bound = best_case_return + 10.0  # Allow significant upside margin
    
    bounds.append((var_lower_bound, var_upper_bound))
    
    # z variable bounds (non-negative)
    for _ in range(num_scenarios):
        bounds.append((0.0, None))
    
    return {
        'c': c,
        'A_ub': np.array(A_ub) if A_ub else None,
        'b_ub': np.array(b_ub) if b_ub else None,
        'A_eq': np.array(A_eq) if A_eq else None,
        'b_eq': np.array(b_eq) if b_eq else None,
        'bounds': bounds,
        'num_assets': num_assets,
        'num_scenarios': num_scenarios,
        'scenario_matrix': scenario_matrix,
        'expected_returns': expected_returns,
        'confidence_level': confidence_level
    }


def optimize_cvar_portfolio(
    scenario_matrix: np.ndarray,
    expected_returns: Optional[Union[np.ndarray, pd.Series]] = None,
    confidence_level: float = 0.95,
    target_return: Optional[float] = None,
    lambda_cvar: float = 1.0,
    lambda_mean_return: float = 0.0,
    constraints: Optional[Dict] = None,
    solver: str = 'highs',
    tolerance: float = 1e-6,
    max_iterations: int = 100000,
    portfolio_id: Optional[Union[int, str]] = None
) -> Tuple[pd.Series, Dict, float]:
    """
    Optimize portfolio to minimize CVaR using linear programming.
    
    Args:
        scenario_matrix: Scenario matrix of shape (num_scenarios, num_assets)
        expected_returns: Expected returns array
        confidence_level: Confidence level alpha
        target_return: Optional target return constraint
        lambda_cvar: Weight on CVaR in objective
        lambda_mean_return: Weight on mean return in objective
        constraints: Dictionary of portfolio constraints
        solver: Solver backend ('highs', 'scipy', 'cvxpy')
        tolerance: Optimization tolerance
        max_iterations: Maximum iterations
        
    Returns:
        Tuple of (optimal_weights, optimization_info, solver_time_ms)
    """
    start_time = time.time()
    
    if constraints is None:
        constraints = {}
    
    # Convert expected_returns to array if it's a Series
    expected_returns_array = expected_returns
    if isinstance(expected_returns, pd.Series):
        expected_returns_array = expected_returns.values
    
    # Set up linear program
    lp_formulation = setup_cvar_linear_program(
        scenario_matrix,
        expected_returns=expected_returns_array,
        confidence_level=confidence_level,
        target_return=target_return,
        lambda_cvar=lambda_cvar,
        lambda_mean_return=lambda_mean_return,
        constraints=constraints
    )
    
    num_assets = lp_formulation['num_assets']
    num_scenarios = lp_formulation['num_scenarios']
    
    # Log LP setup details for first few portfolios
    if portfolio_id is not None and str(portfolio_id) in ['0', '1', '2']:
        print(f"    Setting up LP: {num_assets} assets, {num_scenarios} scenarios, confidence={confidence_level:.2f}")
    
    # Solve linear program
    if solver == 'highs' or solver == 'scipy':
        # Use scipy.optimize.linprog
        # Note: HiGHS doesn't recognize 'tol' option, so we only include it for other methods
        options = {
            'maxiter': max_iterations,
            'disp': False
        }
        if solver != 'highs':
            # For non-HiGHS methods, include tolerance
            options['tol'] = tolerance
        
        result = linprog(
            c=lp_formulation['c'],
            A_ub=lp_formulation['A_ub'],
            b_ub=lp_formulation['b_ub'],
            A_eq=lp_formulation['A_eq'],
            b_eq=lp_formulation['b_eq'],
            bounds=lp_formulation['bounds'],
            method='highs' if solver == 'highs' else 'revised simplex',
            options=options
        )
        
        if not result.success:
            # Log failure for first few portfolios, suppress warnings for others to avoid spam
            should_log = portfolio_id is not None and str(portfolio_id) in ['0', '1', '2']
            if should_log:
                print(f"    Optimization failed: {result.message}")
            # Only warn for first few portfolios to avoid terminal spam
            if should_log:
                warnings.warn(f"Optimization failed: {result.message}", category=UserWarning, stacklevel=2)
            # Return equal weights as fallback
            optimal_weights = pd.Series(
                np.ones(num_assets) / num_assets,
                index=range(num_assets)
            )
            opt_info = {
                'status': 'failed',
                'message': result.message
            }
        else:
            # Extract weights (first num_assets variables)
            num_assets = lp_formulation['num_assets']
            weights_array = result.x[:num_assets]
            
            # Get asset names - use expected_returns index if available
            if isinstance(expected_returns, pd.Series):
                asset_names = expected_returns.index
            elif expected_returns is not None and hasattr(expected_returns, 'index'):
                asset_names = expected_returns.index
            elif hasattr(scenario_matrix, 'columns'):
                asset_names = scenario_matrix.columns
            else:
                asset_names = range(num_assets)
            
            # Ensure we have the right number of asset names
            if len(asset_names) >= num_assets:
                asset_names = asset_names[:num_assets]
            else:
                asset_names = list(asset_names) + list(range(len(asset_names), num_assets))
            
            optimal_weights = pd.Series(weights_array, index=asset_names)
            
            # Extract VaR and CVaR
            var_value = result.x[num_assets]
            z_values = result.x[num_assets + 1:]
            cvar_value = var_value + np.mean(z_values) / (1 - confidence_level)
            
            opt_info = {
                'status': 'success',
                'var': var_value,
                'cvar': cvar_value,
                'objective_value': result.fun,
                'num_iterations': result.nit if hasattr(result, 'nit') else None
            }
    
    elif solver == 'cvxpy' and CVXPY_AVAILABLE:
        # Use CVXPY (alternative implementation)
        num_assets = lp_formulation['num_assets']
        num_scenarios = lp_formulation['num_scenarios']
        alpha = confidence_level
        
        # Decision variables
        w = cp.Variable(num_assets)
        var = cp.Variable()
        z = cp.Variable(num_scenarios)
        
        # Portfolio returns for each scenario
        portfolio_returns = scenario_matrix @ w
        
        # Objective
        cvar_term = var + (1 / ((1 - alpha) * num_scenarios)) * cp.sum(z)
        mean_return_term = -lp_formulation['expected_returns'] @ w
        objective = cp.Minimize(
            lambda_cvar * cvar_term + lambda_mean_return * mean_return_term
        )
        
        # Constraints
        constraints_list = []
        
        # z_s >= -portfolio_return_s - VaR
        constraints_list.append(z >= -portfolio_returns - var)
        constraints_list.append(z >= 0)
        
        # Portfolio constraints
        if constraints.get('fully_invested', True):
            constraints_list.append(cp.sum(w) == 1.0)
        
        if constraints.get('long_only', True) or constraints.get('no_short_selling', True):
            constraints_list.append(w >= 0)
        
        max_weight = constraints.get('max_weight_per_asset', 1.0)
        constraints_list.append(w <= max_weight)
        
        if target_return is not None:
            constraints_list.append(lp_formulation['expected_returns'] @ w >= target_return)
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            warnings.warn(f"CVXPY optimization failed: {problem.status}")
            optimal_weights = pd.Series(
                np.ones(num_assets) / num_assets,
                index=range(num_assets)
            )
            opt_info = {'status': 'failed', 'message': problem.status}
        else:
            optimal_weights = pd.Series(w.value, index=range(num_assets))
            opt_info = {
                'status': 'success',
                'var': var.value,
                'cvar': var.value + np.mean(z.value) / (1 - alpha),
                'objective_value': problem.value
            }
    else:
        raise ValueError(f"Unsupported solver: {solver}")
    
    solver_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return optimal_weights, opt_info, solver_time


def generate_cvar_return_frontier(
    scenario_matrix: np.ndarray,
    expected_returns: np.ndarray,
    confidence_level: float = 0.95,
    target_return_grid: List[float] = None,
    constraints: Optional[Dict] = None,
    solver: str = 'highs',
    tolerance: float = 1e-6
) -> pd.DataFrame:
    """
    Generate CVaR-return efficient frontier.
    
    Args:
        scenario_matrix: Scenario matrix
        expected_returns: Expected returns array
        confidence_level: Confidence level alpha
        target_return_grid: List of target returns for frontier
        constraints: Portfolio constraints
        solver: Solver backend
        tolerance: Optimization tolerance
        
    Returns:
        DataFrame with columns: target_return, cvar, expected_return, weights, ...
    """
    if target_return_grid is None:
        # Auto-generate grid
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        target_return_grid = np.linspace(min_return, max_return, 20)
    
    frontier_results = []
    
    for target_return in target_return_grid:
        try:
            weights, opt_info, _ = optimize_cvar_portfolio(
                scenario_matrix,
                expected_returns=expected_returns,
                confidence_level=confidence_level,
                target_return=target_return,
                lambda_cvar=1.0,
                lambda_mean_return=0.0,
                constraints=constraints,
                solver=solver,
                tolerance=tolerance
            )
            
            if opt_info.get('status') == 'success':
                # Compute actual portfolio return
                portfolio_return = expected_returns @ weights.values
                
                # Convert weights Series to dict for parquet compatibility
                # Store weights as a dictionary that can be serialized
                weights_dict = weights.to_dict()
                
                frontier_results.append({
                    'target_return': target_return,
                    'expected_return': portfolio_return,
                    'cvar': opt_info.get('cvar', np.nan),
                    'var': opt_info.get('var', np.nan),
                    'weights': weights_dict,  # Store as dict instead of Series
                    'optimization_status': 'success'
                })
        except Exception as e:
            warnings.warn(f"Failed to optimize for target return {target_return}: {e}")
            continue
    
    if len(frontier_results) == 0:
        return pd.DataFrame()
    
    # Convert to DataFrame
    frontier_df = pd.DataFrame(frontier_results)
    
    # Expand weights dict into separate columns for parquet compatibility
    if 'weights' in frontier_df.columns and len(frontier_df) > 0:
        # Get asset names from first weights entry
        first_weights = frontier_df['weights'].iloc[0]
        if isinstance(first_weights, dict):
            asset_names = list(first_weights.keys())
        elif isinstance(first_weights, pd.Series):
            asset_names = first_weights.index.tolist()
        else:
            asset_names = []
        
        # Expand weights into separate columns
        if len(asset_names) > 0:
            for asset in asset_names:
                frontier_df[f'weight_{asset}'] = frontier_df['weights'].apply(
                    lambda w: w.get(asset, 0.0) if isinstance(w, dict) else (w[asset] if isinstance(w, pd.Series) and asset in w.index else 0.0)
                )
        
        # Drop the original weights column (contains dict/Series objects not compatible with parquet)
        frontier_df = frontier_df.drop(columns=['weights'])
    
    return frontier_df

