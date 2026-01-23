"""
Black-Litterman Portfolio Optimization Module.

This module implements the Black-Litterman model for portfolio optimization,
combining market equilibrium returns with synthetic investor views to produce posterior
expected returns and optimal portfolio allocations.
"""

from .main import run_black_litterman_optimization
from .black_litterman_optimizer import (
    compute_covariance_matrix,
    derive_market_equilibrium_returns,
    generate_synthetic_views,
    parse_and_scale_views,
    compute_posterior_bl_returns,
    compute_posterior_covariance,
    optimize_portfolio,
    generate_efficient_frontier
)
from .returns import (
    load_panel_prices,
    load_baseline_portfolios,
    load_investor_views,
    compute_daily_returns
)
from .metrics import (
    compute_portfolio_statistics,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_information_ratio,
    compute_tracking_error,
    compute_alpha_vs_market,
    compute_bl_specific_metrics,
    compute_structure_metrics,
    compute_risk_metrics,
    compute_distribution_metrics,
    compute_comparison_metrics,
    compute_runtime_metrics
)
from .time_sliced_metrics import (
    compute_time_sliced_returns,
    compute_time_sliced_risk_metrics,
    compute_time_sliced_tail_metrics,
    compare_time_sliced_prior_vs_posterior,
    analyze_temporal_performance_stability
)
from .report_generator import generate_report, generate_metrics_schema
from .cache import BLCache

# GPU acceleration utilities (optional)
try:
    from .gpu_acceleration import (
        is_gpu_available,
        get_gpu_info,
        clear_gpu_cache,
        compute_covariance_gpu,
        matrix_inverse_gpu,
        matrix_multiply_gpu
    )
    GPU_ACCELERATION_AVAILABLE = True
except ImportError:
    GPU_ACCELERATION_AVAILABLE = False

__all__ = [
    'run_black_litterman_optimization',
    'compute_covariance_matrix',
    'derive_market_equilibrium_returns',
    'generate_synthetic_views',
    'parse_and_scale_views',
    'compute_posterior_bl_returns',
    'compute_posterior_covariance',
    'optimize_portfolio',
    'generate_efficient_frontier',
    'load_panel_prices',
    'load_baseline_portfolios',
    'load_investor_views',
    'compute_daily_returns',
    'compute_portfolio_statistics',
    'compute_sharpe_ratio',
    'compute_sortino_ratio',
    'compute_max_drawdown',
    'compute_calmar_ratio',
    'compute_information_ratio',
    'compute_tracking_error',
    'compute_alpha_vs_market',
    'compute_bl_specific_metrics',
    'compute_structure_metrics',
    'compute_risk_metrics',
    'compute_distribution_metrics',
    'compute_comparison_metrics',
    'compute_runtime_metrics',
    'compute_time_sliced_returns',
    'compute_time_sliced_risk_metrics',
    'compute_time_sliced_tail_metrics',
    'compare_time_sliced_prior_vs_posterior',
    'analyze_temporal_performance_stability',
    'generate_report',
    'generate_metrics_schema',
    'BLCache'
]

# Add GPU utilities to exports if available
if GPU_ACCELERATION_AVAILABLE:
    __all__.extend([
        'is_gpu_available',
        'get_gpu_info',
        'clear_gpu_cache',
        'compute_covariance_gpu',
        'matrix_inverse_gpu',
        'matrix_multiply_gpu'
    ])

__version__ = '1.0.0'

