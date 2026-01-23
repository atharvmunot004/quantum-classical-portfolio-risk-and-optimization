"""
Monte Carlo Simulation for Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) Evaluation Module.

This module implements comprehensive asset-level VaR and CVaR evaluation using Monte Carlo simulation including:
- Asset-level daily returns computation
- Multiple simulation methods (historical bootstrap, parametric normal, Student-t, filtered EWMA)
- Rolling VaR and CVaR calculation via Monte Carlo simulation per asset
- Path simulation for multi-horizon forecasts
- Backtesting with violation analysis for both VaR and CVaR per asset
- Comprehensive performance metrics per asset
- Time-sliced metrics (year/quarter/month) per asset
- Report generation
"""

from .main import evaluate_monte_carlo_var_cvar
from .returns import load_panel_prices, compute_daily_returns
from .monte_carlo_calculator import (
    simulate_returns,
    simulate_returns_paths,
    compute_var_cvar,
    simulate_historical_bootstrap,
    simulate_parametric_normal,
    simulate_parametric_student_t,
    simulate_filtered_ewma_bootstrap,
    fit_student_t_df
)
from .backtesting import (
    detect_var_violations,
    detect_cvar_violations,
    compute_hit_rate,
    compute_violation_ratio,
    kupiec_test,
    christoffersen_test,
    traffic_light_zone,
    compute_accuracy_metrics
)
from .metrics import (
    compute_tail_metrics,
    compute_cvar_tail_metrics,
    compute_distribution_metrics
)
from .time_sliced_metrics import compute_time_sliced_metrics
from .report_generator import generate_report

__all__ = [
    'evaluate_monte_carlo_var_cvar',
    'load_panel_prices',
    'compute_daily_returns',
    'simulate_returns',
    'simulate_returns_paths',
    'compute_var_cvar',
    'simulate_historical_bootstrap',
    'simulate_parametric_normal',
    'simulate_parametric_student_t',
    'simulate_filtered_ewma_bootstrap',
    'fit_student_t_df',
    'detect_var_violations',
    'detect_cvar_violations',
    'compute_hit_rate',
    'compute_violation_ratio',
    'kupiec_test',
    'christoffersen_test',
    'traffic_light_zone',
    'compute_accuracy_metrics',
    'compute_tail_metrics',
    'compute_cvar_tail_metrics',
    'compute_distribution_metrics',
    'compute_time_sliced_metrics',
    'generate_report',
]
