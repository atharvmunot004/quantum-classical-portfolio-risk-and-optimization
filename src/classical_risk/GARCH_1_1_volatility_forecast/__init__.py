"""GARCH(1,1) Volatility Forecasting for VaR/CVaR Evaluation.

This package provides functionality for evaluating Value at Risk (VaR) and
Conditional Value at Risk (CVaR) using rolling, asset-level GARCH(1,1)
volatility forecasting.

IEEE convention used throughout:
- return_t is the arithmetic/log return.
- loss_t = -return_t.
- VaR_t and CVaR_t are stored as POSITIVE loss magnitudes.
- A VaR violation occurs if loss_t > VaR_t.

The main public entry-point is:
- evaluate_garch_var_cvar_asset_level
"""

from .main import evaluate_garch_var_cvar_asset_level
from .returns import compute_daily_returns, compute_portfolio_returns, load_panel_prices, load_portfolio_weights
from .garch_calculator import (
    fit_garch_model,
    compute_conditional_volatility,
    forecast_volatility,
    compute_rolling_volatility_forecast,
    compute_rolling_var_from_garch,
    compute_rolling_cvar_from_garch,
    var_from_volatility,
    cvar_from_volatility,
    GARCHParameterCache,
    compute_asset_level_conditional_volatility,
    compute_all_asset_conditional_volatilities,
    compute_portfolio_volatility_from_assets,
    compute_horizons,
    compute_rolling_garch_asset_level
)
from .backtesting import (
    detect_var_violations,
    detect_cvar_violations,
    compute_accuracy_metrics
)
from .metrics import (
    compute_tail_metrics,
    compute_cvar_tail_metrics,
    compute_structure_metrics,
    compute_distribution_metrics,
    compute_runtime_metrics,
    compute_garch_metrics
)
from .time_sliced_metrics import compute_time_sliced_metrics
from .report_generator import generate_report

__all__ = [
    'evaluate_garch_var_cvar_asset_level',
    'compute_daily_returns',
    'compute_portfolio_returns',
    'load_panel_prices',
    'load_portfolio_weights',
    'fit_garch_model',
    'compute_conditional_volatility',
    'forecast_volatility',
    'compute_rolling_volatility_forecast',
    'compute_rolling_var_from_garch',
    'compute_rolling_cvar_from_garch',
    'var_from_volatility',
    'cvar_from_volatility',
    'GARCHParameterCache',
    'compute_asset_level_conditional_volatility',
    'compute_all_asset_conditional_volatilities',
    'compute_portfolio_volatility_from_assets',
    'compute_horizons',
    'compute_rolling_garch_asset_level',
    'detect_var_violations',
    'detect_cvar_violations',
    'compute_accuracy_metrics',
    'compute_tail_metrics',
    'compute_cvar_tail_metrics',
    'compute_structure_metrics',
    'compute_distribution_metrics',
    'compute_runtime_metrics',
    'compute_garch_metrics',
    'compute_time_sliced_metrics',
    'generate_report'
]
