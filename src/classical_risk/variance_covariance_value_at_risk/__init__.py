"""
Variance-Covariance Value-at-Risk Evaluation Module.

This module implements comprehensive Variance-Covariance VaR evaluation including:
- Asset-level returns computation (optimized: computed once)
- Rolling mean and volatility estimation per asset
- Rolling Variance-Covariance VaR calculation using normal distribution assumption
- Backtesting with violation analysis
- Comprehensive performance metrics
- Report generation

Key features:
- Asset-level evaluation (no portfolio aggregation)
- Normal distribution assumption: VaR = -μ + z_{1-α} * σ * √(h)
- Rolling window estimation for mean and volatility
- Parameter caching for computational efficiency
"""

# Lazy import to avoid RuntimeWarning when running main.py as a module
def evaluate_var(*args, **kwargs):
    """Lazy import wrapper for evaluate_var to avoid import conflicts."""
    from .main import evaluate_var as _evaluate_var
    return _evaluate_var(*args, **kwargs)

from .returns import (
    compute_daily_returns,
    load_panel_prices
)
from .var_calculator import (
    compute_rolling_variance_covariance_var,
    compute_rolling_mean_volatility,
    compute_variance_covariance_var,
    estimate_mean_volatility,
    align_returns_and_var
)
from .backtesting import (
    detect_var_violations,
    compute_accuracy_metrics
)
from .metrics import (
    compute_tail_metrics,
    compute_distribution_metrics,
    compute_runtime_metrics
)
from .time_sliced_metrics import compute_time_sliced_metrics
from .report_generator import generate_report

__all__ = [
    'evaluate_var',
    'compute_daily_returns',
    'load_panel_prices',
    'compute_rolling_variance_covariance_var',
    'compute_rolling_mean_volatility',
    'compute_variance_covariance_var',
    'estimate_mean_volatility',
    'align_returns_and_var',
    'detect_var_violations',
    'compute_accuracy_metrics',
    'compute_tail_metrics',
    'compute_distribution_metrics',
    'compute_runtime_metrics',
    'compute_time_sliced_metrics',
    'generate_report',
]
