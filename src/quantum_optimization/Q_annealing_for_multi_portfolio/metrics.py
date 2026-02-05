"""
Performance Metrics Module for Quantum Portfolio Optimization.

Computes portfolio performance metrics and optimization quality metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats


def compute_realized_return(
    portfolio_returns: pd.Series,
    annualized: bool = True
) -> float:
    """
    Compute realized return.
    
    Args:
        portfolio_returns: Series of portfolio returns
        annualized: Whether to annualize
        
    Returns:
        Realized return
    """
    total_return = (1 + portfolio_returns).prod() - 1
    
    if annualized:
        periods_per_year = 252
        n_periods = len(portfolio_returns)
        total_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    return total_return


def compute_realized_volatility(
    portfolio_returns: pd.Series,
    annualized: bool = True
) -> float:
    """
    Compute realized volatility.
    
    Args:
        portfolio_returns: Series of portfolio returns
        annualized: Whether to annualize
        
    Returns:
        Realized volatility
    """
    volatility = portfolio_returns.std()
    
    if annualized:
        volatility = volatility * np.sqrt(252)
    
    return volatility


def compute_realized_cvar(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    annualized: bool = True
) -> float:
    """
    Compute realized Conditional Value at Risk (CVaR).
    
    Args:
        portfolio_returns: Series of portfolio returns
        confidence_level: Confidence level
        annualized: Whether to annualize
        
    Returns:
        Realized CVaR (negative value)
    """
    var_level = 1 - confidence_level
    var = portfolio_returns.quantile(var_level)
    
    # CVaR is mean of returns below VaR
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    
    if annualized:
        cvar = cvar * np.sqrt(252)
    
    return cvar


def compute_max_drawdown(portfolio_returns: pd.Series) -> float:
    """
    Compute maximum drawdown.
    
    Args:
        portfolio_returns: Series of portfolio returns
        
    Returns:
        Maximum drawdown (negative value)
    """
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()


def compute_sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualized: bool = True
) -> float:
    """
    Compute Sharpe ratio.
    
    Args:
        portfolio_returns: Series of portfolio returns
        risk_free_rate: Risk-free rate
        annualized: Whether to annualize
        
    Returns:
        Sharpe ratio
    """
    excess_returns = portfolio_returns - risk_free_rate / 252
    
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = excess_returns.mean() / excess_returns.std()
    
    if annualized:
        sharpe = sharpe * np.sqrt(252)
    
    return sharpe


def compute_turnover(
    weights: pd.DataFrame,
    rebalance_frequency: int = 21
) -> float:
    """
    Compute portfolio turnover.
    
    Args:
        weights: DataFrame of portfolio weights (dates x assets)
        rebalance_frequency: Rebalancing frequency in days
        
    Returns:
        Average turnover per rebalancing period
    """
    if len(weights) < 2:
        return 0.0
    
    turnovers = []
    
    for i in range(1, len(weights)):
        prev_weights = weights.iloc[i - 1]
        curr_weights = weights.iloc[i]
        
        # Turnover is sum of absolute weight changes
        turnover = (curr_weights - prev_weights).abs().sum()
        turnovers.append(turnover)
    
    return np.mean(turnovers) if turnovers else 0.0


def compute_portfolio_performance_metrics(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Compute comprehensive portfolio performance metrics.
    
    Args:
        portfolio_returns: Series of portfolio returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {
        'realized_return': compute_realized_return(portfolio_returns),
        'realized_volatility': compute_realized_volatility(portfolio_returns),
        'realized_cvar': compute_realized_cvar(portfolio_returns),
        'max_drawdown': compute_max_drawdown(portfolio_returns),
        'sharpe_ratio': compute_sharpe_ratio(portfolio_returns, risk_free_rate)
    }
    
    return metrics


def compute_optimization_quality_metrics(
    annealing_results: Dict,
    qubo_builder
) -> Dict[str, float]:
    """
    Compute optimization quality metrics.
    
    Args:
        annealing_results: Results from quantum annealer
        qubo_builder: QUBOBuilder instance
        
    Returns:
        Dictionary of optimization quality metrics
    """
    energies = annealing_results['energies']
    best_energy = annealing_results['best_energy']
    
    # Energy gap: difference between best and second best
    sorted_energies = np.sort(energies)
    if len(sorted_energies) > 1:
        energy_gap = sorted_energies[1] - sorted_energies[0]
    else:
        energy_gap = 0.0
    
    # Pareto front size (approximate: number of unique solutions)
    samples = annealing_results['samples']
    unique_solutions = len(set(tuple(s) for s in samples))
    
    metrics = {
        'best_energy': float(best_energy),
        'energy_gap': float(energy_gap),
        'pareto_front_size': unique_solutions
    }
    
    return metrics


def compute_quantum_specific_metrics(
    annealing_results: Dict
) -> Dict[str, Optional[float]]:
    """
    Compute quantum-specific metrics.
    
    Args:
        annealing_results: Results from quantum annealer
        
    Returns:
        Dictionary of quantum-specific metrics
    """
    metrics = {
        'num_reads': len(annealing_results['energies']),
        'annealing_time_us': None,  # Set by caller
        'chain_break_fraction': annealing_results.get('chain_break_fraction'),
        'embedding_size': annealing_results.get('embedding_size'),
        'logical_to_physical_qubit_ratio': annealing_results.get('logical_to_physical_ratio')
    }
    
    return metrics


def compute_time_sliced_metrics(
    portfolio_returns: pd.Series,
    weights: pd.DataFrame,
    slice_by: str = 'year',
    risk_free_rate: float = 0.0
) -> pd.DataFrame:
    """
    Compute time-sliced performance metrics.
    
    Args:
        portfolio_returns: Series of portfolio returns
        weights: DataFrame of portfolio weights
        slice_by: 'year' or 'quarter'
        risk_free_rate: Risk-free rate
        
    Returns:
        DataFrame of time-sliced metrics
    """
    if slice_by == 'year':
        grouper = portfolio_returns.index.year
    elif slice_by == 'quarter':
        grouper = pd.Grouper(freq='Q')
    else:
        raise ValueError(f"Unknown slice_by: {slice_by}")
    
    results = []
    
    for period, period_returns in portfolio_returns.groupby(grouper):
        if len(period_returns) == 0:
            continue
        
        period_metrics = compute_portfolio_performance_metrics(
            period_returns,
            risk_free_rate
        )
        
        # Compute turnover for this period
        period_start = period_returns.index[0]
        period_end = period_returns.index[-1]
        period_weights = weights.loc[period_start:period_end]
        
        if len(period_weights) > 1:
            period_metrics['turnover'] = compute_turnover(period_weights)
        else:
            period_metrics['turnover'] = 0.0
        
        period_metrics['period'] = period
        results.append(period_metrics)
    
    return pd.DataFrame(results)
