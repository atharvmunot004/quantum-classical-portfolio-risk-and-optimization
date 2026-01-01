"""
Historical Value-at-Risk Evaluation Module.

This module implements comprehensive Historical VaR evaluation including:
- Asset-level returns computation (optimized: computed once)
- Portfolio returns via linear projection: R_p(t) = W^T R_assets(t)
- Rolling Historical VaR calculation using empirical quantiles
- Backtesting with violation analysis
- Comprehensive performance metrics
- Report generation

Key optimization: Asset returns are computed once at the asset level and
reused across portfolio weight realizations via linear projection.
"""

# Lazy import to avoid RuntimeWarning when running main.py as a module
def evaluate_var(*args, **kwargs):
    """Lazy import wrapper for evaluate_var to avoid import conflicts."""
    from .main import evaluate_var as _evaluate_var
    return _evaluate_var(*args, **kwargs)
from .returns import (
    compute_daily_returns,
    compute_portfolio_returns,
    construct_asset_return_matrix,
    compute_portfolio_returns_linear_projection
)
from .var_calculator import (
    compute_rolling_historical_var,
    compute_historical_var,
    align_returns_and_var
)
from .backtesting import detect_var_violations, compute_accuracy_metrics
from .metrics import (
    compute_tail_metrics,
    compute_structure_metrics,
    compute_distribution_metrics,
    compute_runtime_metrics
)
from .time_sliced_metrics import compute_time_sliced_metrics

__all__ = [
    'evaluate_var',
    'compute_daily_returns',
    'compute_portfolio_returns',
    'construct_asset_return_matrix',
    'compute_portfolio_returns_linear_projection',
    'compute_rolling_historical_var',
    'compute_historical_var',
    'align_returns_and_var',
    'detect_var_violations',
    'compute_accuracy_metrics',
    'compute_tail_metrics',
    'compute_structure_metrics',
    'compute_distribution_metrics',
    'compute_runtime_metrics',
    'compute_time_sliced_metrics',
]
