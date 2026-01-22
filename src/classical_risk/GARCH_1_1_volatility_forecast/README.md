# GARCH(1,1) Volatility Forecasting Value-at-Risk Evaluation

This module implements comprehensive evaluation of GARCH(1,1) volatility forecasting for Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) at the asset level.

## Overview

The GARCH(1,1) model captures time-varying volatility by modeling conditional variance as a function of past squared innovations and past conditional variances. This approach provides dynamic risk measures that adapt to changing market conditions.

**GARCH(1,1) Model Specification:**

```
σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
```

Where:
- σ²ₜ = conditional variance at time t
- ω = intercept (long-run variance component)
- α = ARCH coefficient (sensitivity to past squared innovations)
- β = GARCH coefficient (persistence of volatility)
- εₜ₋₁ = past innovation (return - mean)
- Stationarity condition: α + β < 1

**Key Design Principle**: GARCH(1,1) is fitted and evaluated strictly on individual asset return series. Conditional volatility and rolling forecasts are produced as time series via rolling estimation windows. VaR/CVaR are computed per asset under the chosen conditional distribution and scaled horizons. All backtesting, tail diagnostics, distributional diagnostics, and time-sliced evaluation (year/quarter/month) are computed per asset to support empirical evaluation without portfolio aggregation assumptions.

## Features

- **Asset-Level Evaluation**: Processes each asset independently (no portfolio aggregation)
- **Daily Returns Computation**: Computes log or simple returns from price data (once, then reused)
- **Rolling GARCH Parameter Estimation**: Estimates GARCH(1,1) parameters (ω, α, β) using rolling windows per asset
- **Conditional Volatility**: Computes time-varying conditional volatility from fitted GARCH models
- **One-Step-Ahead Volatility Forecasts**: Generates rolling one-step-ahead volatility forecasts for VaR/CVaR computation
- **Rolling VaR/CVaR**: Implements rolling window VaR and CVaR calculation using forecasted volatility
- **Distribution Support**: Supports normal and Student-t distributions for conditional returns
- **Parameter Caching**: Caches GARCH parameters and conditional volatility for computational efficiency
- **Backtesting**: Comprehensive backtesting with violation analysis per asset
- **Accuracy Metrics**: Hit rate, violation ratio, Kupiec test, Christoffersen tests, traffic light zones
- **Tail Risk Metrics**: Mean exceedance, max exceedance, quantile loss, RMSE for both VaR and CVaR
- **GARCH-Specific Metrics**: Parameter estimates (ω, α, β), persistence (α+β), unconditional variance, fit diagnostics (AIC, BIC, log-likelihood)
- **Distribution Metrics**: Skewness, kurtosis, Jarque-Bera normality test
- **Time-Sliced Metrics**: Backtesting metrics by year, quarter, and month
- **Safety Checks**: Guardrails for volatility positivity, stationarity, and parameter constraints
- **Fallback Mechanisms**: EWMA or sample variance fallback when GARCH fitting fails
- **Report Generation**: Comprehensive markdown reports

## Usage

### Command Line

```bash
# Run with default configuration (llm.json in same directory)
python -m src.classical_risk.GARCH_1_1_volatility_forecast.main

# Specify custom configuration file
python -m src.classical_risk.GARCH_1_1_volatility_forecast.main --config path/to/config.json
```

### Python API

```python
from src.classical_risk.GARCH_1_1_volatility_forecast import evaluate_garch_var_cvar_asset_level

# Load configuration and evaluate (default: auto workers)
results = evaluate_garch_var_cvar_asset_level(config_path='llm.json')

# Access results
volatility_series_df = results['volatility_series']
risk_series_df = results['risk_series']
metrics_df = results['metrics']
time_sliced_metrics_df = results['time_sliced_metrics']

# Or use configuration dictionary
config = {
    'inputs': {
        'panel_price_path': 'data/preprocessed/panel_price.parquet',
        'asset_universe': {
            'mode': 'from_columns',
            'include': None,
            'exclude': None
        }
    },
    'garch_settings': {
        'model_type': 'GARCH',
        'p': 1,
        'q': 1,
        'distribution': 't',
        'student_t_df': 5,
        'mean_model': {
            'enabled': False
        },
        'confidence_levels': [0.95, 0.99],
        'horizons': {
            'base_horizon': 1,
            'scaled_horizons': [10],
            'scaling_rule': 'sqrt_time'
        },
        'estimation_windows': [252, 500]
    },
    # ... other settings
}

results = evaluate_garch_var_cvar_asset_level(config_dict=config)
```

## Configuration

The module uses a JSON configuration file (`llm.json`) with the following structure:

```json
{
  "task": "garch_1_1_var_cvar_asset_level_evaluation_optimized",
  "inputs": {
    "panel_price_path": "data/preprocessed/panel_price.parquet",
    "asset_universe": {
      "mode": "from_columns",
      "include": null,
      "exclude": null
    }
  },
  "data_settings": {
    "return_type": "log",
    "tail_side": "left",
    "calendar": {
      "sort_index": true,
      "drop_duplicate_dates": true
    },
    "missing_data_policy": {
      "dropna": "per_asset",
      "min_required_observations": 800
    }
  },
  "garch_settings": {
    "model_type": "GARCH",
    "p": 1,
    "q": 1,
    "distribution": "t",
    "student_t_df": 5,
    "mean_model": {
      "enabled": false
    },
    "estimation_windows": [252, 500],
    "rolling": {
      "enabled": true,
      "step_size": 1,
      "warmup_policy": "skip_until_window_full",
      "forecast_method": "rolling",
      "forecast_target": "one_step_ahead"
    },
    "constraints": {
      "omega_lower_bound": 1e-12,
      "alpha_lower_bound": 0.0,
      "beta_lower_bound": 0.0,
      "stationarity_enforced": true,
      "stationarity_rule": "alpha + beta < 1"
    },
    "fallback_long_run_variance": true,
    "fallback_policy": {
      "on_fit_fail": "use_ewma_or_sample_var",
      "ewma_lambda": 0.94,
      "min_variance_floor": 1e-12
    },
    "confidence_levels": [0.95, 0.99],
    "horizons": {
      "base_horizon": 1,
      "scaled_horizons": [10],
      "scaling_rule": "sqrt_time"
    }
  },
  "computation_strategy": {
    "compute_daily_returns_once": true,
    "rolling_engine": {
      "max_workers": "auto",
      "chunk_assets": 1
    },
    "cache": {
      "enabled": true,
      "parameter_store_path": "cache/garch_asset_parameters.parquet"
    },
    "runtime_instrumentation": {
      "enabled": true
    },
    "safety_checks": {
      "enabled": true
    }
  },
  "evaluation": {
    "scope": "per_asset",
    "backtesting": {
      "enabled": true,
      "tests": [
        "hit_rate",
        "violation_ratio",
        "kupiec_unconditional_coverage",
        "christoffersen_independence",
        "christoffersen_conditional_coverage",
        "traffic_light_zone"
      ]
    },
    "time_sliced_metrics": {
      "enabled": true,
      "slice_by": ["year", "quarter", "month"],
      "minimum_observations_per_slice": 60
    }
  },
  "outputs": {
    "parameter_store": {
      "path": "cache/garch_asset_parameters.parquet"
    },
    "volatility_series_store": {
      "path": "results/classical_risk/garch_asset_conditional_volatility_series.parquet"
    },
    "risk_series_store": {
      "path": "results/classical_risk/garch_asset_var_cvar_series.parquet"
    },
    "metrics_table": {
      "path": "results/classical_risk/garch_asset_level_metrics.parquet"
    },
    "time_sliced_metrics_table": {
      "path": "results/classical_risk/garch_asset_level_time_sliced_metrics.parquet"
    },
    "garch_results_scheme.json": {
      "path": "results/classical_risk/garch_asset_level_results_scheme.json"
    },
    "report": {
      "path": "results/classical_risk/garch_asset_level_report.md"
    }
  }
}
```

### Key Configuration Parameters

- **`model_type`**: GARCH model type (`'GARCH'` for standard GARCH)
- **`p`** / **`q`**: GARCH order parameters (typically p=1, q=1 for GARCH(1,1))
- **`distribution`**: Conditional distribution (`'normal'` or `'t'` for Student-t)
- **`student_t_df`**: Degrees of freedom for t-distribution (if distribution='t')
- **`mean_model.enabled`**: Whether to include AR mean model (typically False for zero-mean)
- **`estimation_windows`**: Rolling window sizes in days (e.g., [252, 500])
- **`confidence_levels`**: VaR/CVaR confidence levels (e.g., [0.95, 0.99])
- **`constraints.stationarity_enforced`**: Whether to enforce α + β < 1
- **`fallback_long_run_variance`**: Use long-run variance fallback when fitting fails
- **`fallback_policy.on_fit_fail`**: Fallback method (`'use_ewma_or_sample_var'`)

## Output

The module generates:

1. **Volatility Series** (`garch_asset_conditional_volatility_series.parquet`): Time series of conditional volatility (σₜ) and one-step-ahead forecast volatility per asset, date, and estimation window. Each row represents volatility estimates at a specific point in time.

2. **Risk Series** (`garch_asset_var_cvar_series.parquet`): Time series of VaR and CVaR values per asset, date, confidence level, horizon, and estimation window. Each row represents a single VaR/CVaR estimate at a specific point in time.

3. **Metrics Table** (`garch_asset_level_metrics.parquet`): Aggregated backtesting metrics over the full evaluation period per asset. Row granularity: asset × confidence_level × horizon × estimation_window. Includes:
   - **Accuracy metrics**: hit rate, violation ratio, Kupiec test, Christoffersen tests, traffic light zone
   - **Tail behavior metrics (VaR)**: mean exceedance, max exceedance, quantile loss, RMSE
   - **Tail behavior metrics (CVaR)**: CVaR mean exceedance, max exceedance, RMSE
   - **GARCH-specific metrics**: ω, α, β, α+β (persistence), unconditional variance, fit success, convergence flag, log-likelihood, AIC, BIC
   - **Distributional characteristics**: skewness, kurtosis, Jarque-Bera test

4. **Time-Sliced Metrics** (`garch_asset_level_time_sliced_metrics.parquet`): Backtesting metrics computed for specific time periods (year, quarter, month). Enables analysis of temporal patterns in VaR/CVaR performance across different market regimes.

5. **Parameter Store** (`cache/garch_asset_parameters.parquet`): Cached GARCH parameters (ω, α, β) and conditional volatility estimated per asset, date, and estimation window. Used for computational efficiency.

6. **Results Schema** (`garch_asset_level_results_scheme.json`): JSON schema documenting the structure of all output files, including row counts and column descriptions.

7. **Report** (`garch_asset_level_report.md`): Comprehensive markdown report with:
   - Methodology overview
   - GARCH specification and estimation details
   - Rolling forecast construction
   - VaR/CVaR construction
   - Backtesting results
   - Time-sliced backtesting
   - Tail risk behavior
   - Distributional characteristics
   - Computational performance
   - Key insights and recommendations

## Module Structure

- `main.py`: Main orchestration script for asset-level evaluation
- `garch_calculator.py`: GARCH model fitting, conditional volatility computation, and VaR/CVaR calculation
- `returns.py`: Asset-level returns computation
- `backtesting.py`: Violation detection and accuracy metrics
- `metrics.py`: Tail risk, GARCH-specific, and distribution metrics (asset-level)
- `time_sliced_metrics.py`: Time-sliced backtesting metrics
- `report_generator.py`: Markdown report generation

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.23.0
- scipy >= 1.9.0
- pyarrow >= 10.0.0
- arch >= 5.0.0 (for GARCH model estimation)

## Performance

- **Parallel Processing**: The module uses multiprocessing to process assets in parallel
  - Default: Uses 80% of available CPU cores
  - Can be controlled via `computation_strategy.rolling_engine.max_workers`
  - Assets are processed independently, enabling efficient parallelization
  - Expected speedup: ~N× where N is the number of CPU cores

- **Optimization Features**:
  - Daily returns computed once and reused
  - GARCH parameter caching (ω, α, β, conditional volatility) for rolling windows
  - Efficient GARCH fitting using `arch` library
  - Memory-efficient processing per asset
  - Fallback mechanisms for failed GARCH fits (EWMA or sample variance)

## Methodology Notes

- **Distributional Assumption**: Conditional returns follow normal or Student-t distribution given past information
- **GARCH Estimation**: Maximum likelihood estimation using `arch` library with stationarity constraints
- **Volatility Forecasting**: One-step-ahead forecasts computed from fitted GARCH models
- **Horizon Scaling**: Square-root-of-time rule applied to conditional volatility (σ_h = σ_1 * √(h))
- **Rolling Windows**: Configurable window sizes (e.g., 252, 500 days) with step size control
- **Warmup Policy**: Skips periods until window is full
- **Stationarity**: Enforces α + β < 1 to ensure covariance stationarity
- **Fallback Policy**: Uses EWMA (λ=0.94) or sample variance when GARCH fitting fails or violates constraints

## Notes

- **Asset-Level Only**: This module evaluates VaR/CVaR strictly per asset. No portfolio aggregation is performed.
- **GARCH(1,1) Model**: The standard GARCH(1,1) specification is used (p=1, q=1). The model captures volatility clustering and persistence.
- **VaR Violations**: Occur when actual loss > VaR (losses are positive values representing tail risk: loss_t = -return_t)
- **CVaR Violations**: Occur when actual loss > CVaR (CVaR represents expected loss given VaR is exceeded)
- **Traffic Light Zones**: Follow Basel guidelines (adjusted for confidence level). Computed only for 99% VaR per specification.
- **Parameter Caching**: GARCH parameters (ω, α, β) and conditional volatility are cached to avoid recomputation. Cache hit ratio is tracked for performance monitoring.
- **Time-Sliced Metrics**: Enable analysis of temporal patterns in VaR/CVaR performance across different market regimes.
- **Persistence Interpretation**: 
  - α + β close to 1 indicates high volatility persistence (slow mean reversion)
  - Lower α + β indicates faster mean reversion to long-run variance
  - α captures sensitivity to recent shocks (ARCH effect)
  - β captures volatility clustering (GARCH effect)
- **Distribution Choice**:
  - Student-t distribution better captures fat tails in financial returns
  - Normal distribution is simpler but may underestimate tail risk
  - Degrees of freedom (df) for t-distribution typically set to 5-10
- **Safety Checks**: 
  - Volatility must be positive (σₜ > 0)
  - VaR must be positive (VaR > 0)
  - Stationarity must be maintained (α + β < 1)
  - Fallback to EWMA or sample variance when constraints violated
- **Numerical Stability**: Returns are scaled internally (by 100) during GARCH fitting for numerical stability, then rescaled back to original units for VaR/CVaR computation.

