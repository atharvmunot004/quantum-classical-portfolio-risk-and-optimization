# Monte Carlo Simulation for Value-at-Risk and Conditional Value-at-Risk Evaluation

This module implements comprehensive evaluation of Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) at the asset level using Monte Carlo simulation methods.

## Overview

Monte Carlo simulation generates synthetic return paths by sampling from historical data or parametric distributions, then computes VaR and CVaR as quantiles of the simulated loss distribution. Unlike parametric methods that assume a specific distribution, Monte Carlo methods can capture complex return dynamics through:

1. **Historical Bootstrap**: Resampling from historical returns (iid or block bootstrap)
2. **Parametric Normal**: Sampling from fitted normal distribution
3. **Parametric Student-t**: Sampling from fitted Student-t distribution (captures fat tails)
4. **Filtered EWMA Bootstrap**: Volatility-filtered simulation using EWMA volatility estimates

**Key Design Principle**: Monte Carlo simulation is performed strictly per asset using rolling windows under alternative data-generating assumptions. No portfolio aggregation or cross-asset covariance modeling is performed, ensuring methodological comparability with other single-asset risk models.

## Features

- **Asset-Level Evaluation**: Processes each asset independently (no portfolio aggregation)
- **Multiple Simulation Methods**: Historical bootstrap, parametric normal, Student-t, and filtered EWMA bootstrap
- **Daily Returns Computation**: Computes log or simple returns from price data (once, then reused)
- **Rolling VaR and CVaR**: Implements rolling window VaR/CVaR calculation via Monte Carlo simulation per asset
- **Path Simulation**: For multi-day horizons, simulates return paths and aggregates (sum of daily returns)
- **Backtesting**: Comprehensive backtesting with violation analysis for both VaR and CVaR per asset
- **Accuracy Metrics**: Hit rate, violation ratio, Kupiec test, Christoffersen test, traffic light zones
- **Tail Risk Metrics**: Mean exceedance, max exceedance, quantile loss, RMSE for both VaR and CVaR
- **Distribution Metrics**: Skewness, kurtosis, Jarque-Bera normality test
- **Time-Sliced Metrics**: Backtesting metrics by year, quarter, and month
- **Parameter Caching**: Caches fitted parameters and risk series for computational efficiency
- **Report Generation**: Comprehensive markdown reports

## Usage

### Command Line

```bash
# Run with default configuration (llm.json in same directory)
python -m src.classical_risk.monte_carlo_simulation_for_var_cvar.main

# Specify custom configuration file
python -m src.classical_risk.monte_carlo_simulation_for_var_cvar.main --config path/to/config.json

# Use specific number of parallel workers
python -m src.classical_risk.monte_carlo_simulation_for_var_cvar.main --n-jobs 8

# Sequential processing (for debugging)
python -m src.classical_risk.monte_carlo_simulation_for_var_cvar.main --n-jobs 1
```

### Python API

```python
from src.classical_risk.monte_carlo_simulation_for_var_cvar import evaluate_monte_carlo_var_cvar

# Load configuration and evaluate (default: auto workers)
risk_series_df, metrics_df, time_sliced_metrics_df = evaluate_monte_carlo_var_cvar(config_path='llm.json')

# Use specific number of workers
risk_series_df, metrics_df, time_sliced_metrics_df = evaluate_monte_carlo_var_cvar(
    config_path='llm.json', 
    n_jobs=8
)

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
    'monte_carlo_settings': {
        'methods': [
            {
                'name': 'historical_bootstrap',
                'enabled': True,
                'bootstrap_type': 'iid'
            },
            {
                'name': 'parametric_normal',
                'enabled': True
            },
            {
                'name': 'parametric_student_t',
                'enabled': True,
                'fit': {
                    'df': {
                        'mode': 'mle_or_fixed',
                        'fixed_df': 5
                    }
                }
            }
        ],
        'estimation_windows': [252, 500],
        'confidence_levels': [0.95, 0.99],
        'horizons': {
            'base_horizon': 1,
            'scaled_horizons': [10],
            'horizon_handling': 'path_simulation'
        },
        'num_simulations': 20000,
        'random_seed': 42
    },
    # ... other settings
}

risk_series_df, metrics_df, time_sliced_metrics_df = evaluate_monte_carlo_var_cvar(config_dict=config, n_jobs=8)
```

## Configuration

The module uses a JSON configuration file (`llm.json`) with the following structure:

```json
{
  "task": "monte_carlo_var_cvar_asset_level_evaluation_optimized",
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
  "monte_carlo_settings": {
    "simulation_family": "returns_based",
    "methods": [
      {
        "name": "historical_bootstrap",
        "enabled": true,
        "bootstrap_type": "iid",
        "block_bootstrap": {
          "enabled": false,
          "block_length": 5
        }
      },
      {
        "name": "parametric_normal",
        "enabled": true,
        "fit": {
          "mu": "sample_mean",
          "sigma": "sample_std"
        }
      },
      {
        "name": "parametric_student_t",
        "enabled": true,
        "fit": {
          "mu": "sample_mean",
          "sigma": "sample_std",
          "df": {
            "mode": "mle_or_fixed",
            "fixed_df": 5,
            "bounds": [2.1, 50.0]
          }
        }
      },
      {
        "name": "filtered_ewma_bootstrap",
        "enabled": false,
        "filter": {
          "type": "ewma",
          "lambda": 0.94
        }
      }
    ],
    "estimation_windows": [252, 500],
    "rolling": {
      "enabled": true,
      "step_size": 1,
      "warmup_policy": "skip_until_window_full"
    },
    "confidence_levels": [0.95, 0.99],
    "horizons": {
      "base_horizon": 1,
      "scaled_horizons": [10],
      "horizon_handling": "path_simulation"
    },
    "num_simulations": 20000,
    "path_aggregation_rule": "sum",
    "random_seed": 42
  },
  "computation_strategy": {
    "compute_daily_returns_once": true,
    "rolling_engine": {
      "max_workers": "auto",
      "chunk_assets": 1
    },
    "cache": {
      "enabled": true,
      "fitted_params_store_path": "cache/mcs_asset_fitted_parameters.parquet",
      "risk_series_store_path": "cache/mcs_asset_var_cvar_series.parquet"
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
    "fitted_parameter_store": {
      "path": "cache/mcs_asset_fitted_parameters.parquet"
    },
    "risk_series_store": {
      "path": "results/classical_risk/mcs_asset_var_cvar_series.parquet"
    },
    "metrics_table": {
      "path": "results/classical_risk/mcs_asset_level_metrics.parquet"
    },
    "time_sliced_metrics_table": {
      "path": "results/classical_risk/mcs_asset_level_time_sliced_metrics.parquet"
    },
    "mcs_results_scheme.json": {
      "path": "results/classical_risk/mcs_asset_level_results_scheme.json"
    },
    "report": {
      "path": "results/classical_risk/mcs_asset_level_report.md"
    }
  }
}
```

## Output

The module generates:

1. **Risk Series** (`mcs_asset_var_cvar_series.parquet`): Time series of VaR and CVaR values per asset, date, method, confidence level, horizon, and estimation window. Each row represents a single VaR/CVaR estimate at a specific point in time.

2. **Metrics Table** (`mcs_asset_level_metrics.parquet`): Aggregated backtesting metrics over the full evaluation period per asset. Row granularity: asset × method × confidence_level × horizon × estimation_window. Includes:
   - Accuracy metrics: hit rate, violation ratio, Kupiec test, Christoffersen tests, traffic light zone
   - Tail behavior metrics: mean exceedance, max exceedance, quantile loss, RMSE (for both VaR and CVaR)
   - Distributional characteristics: rolling mean, rolling volatility, skewness, kurtosis, Jarque-Bera test
   - Simulation-specific metrics: method, num_simulations, fitted parameters (mu, sigma, Student-t df, EWMA lambda)

3. **Time-Sliced Metrics** (`mcs_asset_level_time_sliced_metrics.parquet`): Backtesting metrics computed for specific time periods (year, quarter, month). Enables analysis of temporal patterns in VaR/CVaR performance across different market regimes.

4. **Fitted Parameter Store** (`cache/mcs_asset_fitted_parameters.parquet`): Cached fitted parameters (mean, volatility, Student-t degrees of freedom, EWMA lambda) estimated per asset, date, method, and estimation window. Used for computational efficiency and analysis.

5. **Results Schema** (`mcs_asset_level_results_scheme.json`): JSON schema documenting the structure of all output files, including row counts and column descriptions.

6. **Report** (`mcs_asset_level_report.md`): Comprehensive markdown report with:
   - Methodology overview
   - Monte Carlo methods and assumptions
   - Rolling forecast construction
   - Path simulation for multi-horizon forecasts
   - VaR/CVaR construction details
   - Backtesting results
   - Time-sliced backtesting
   - Tail risk behavior
   - Distributional characteristics
   - Computational performance
   - Key insights and recommendations

## Module Structure

- `main.py`: Main orchestration script for asset-level evaluation
- `returns.py`: Asset-level returns computation
- `monte_carlo_calculator.py`: Monte Carlo simulation methods (historical bootstrap, parametric normal, Student-t, filtered EWMA) and VaR/CVaR computation
- `backtesting.py`: Violation detection and accuracy metrics for both VaR and CVaR
- `metrics.py`: Tail risk and distribution metrics (asset-level)
- `time_sliced_metrics.py`: Time-sliced backtesting metrics
- `report_generator.py`: Markdown report generation

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.23.0
- scipy >= 1.9.0
- pyarrow >= 10.0.0

## Performance

- **Parallel Processing**: The module uses multiprocessing to process assets in parallel
  - Default: Uses all available CPU cores minus one
  - Can be controlled with `--n-jobs` parameter
  - Assets are processed independently, enabling efficient parallelization
  - Expected speedup: ~N× where N is the number of CPU cores

- **Optimization Features**:
  - Daily returns computed once and reused
  - Parameter caching for fitted parameters (mean, volatility, Student-t df, EWMA lambda)
  - Risk series caching to avoid recomputation
  - Efficient path simulation for multi-horizon forecasts
  - Vectorized simulation where possible
  - Memory-efficient processing per asset

- **Simulation Optimizations**:
  - Simulate once per (asset, window, method, horizon), compute VaR/CVaR for all confidence levels
  - Cache Student-t degrees of freedom per (asset, estimation_window)
  - Simulate max horizon once and reuse paths for all horizons

## Methodology Notes

### Simulation Methods

1. **Historical Bootstrap**: Resamples from historical returns in the estimation window
   - IID bootstrap: Independent random sampling with replacement
   - Block bootstrap: Preserves short-range dependence by resampling blocks

2. **Parametric Normal**: Fits normal distribution (mean, volatility) to window returns, then samples

3. **Parametric Student-t**: Fits Student-t distribution (mean, volatility, degrees of freedom) to capture fat tails
   - Degrees of freedom estimated via MLE or fixed value
   - Bounded between 2.1 and 50.0

4. **Filtered EWMA Bootstrap**: 
   - Estimates time-varying volatility using EWMA filter
   - Computes standardized residuals
   - Resamples standardized residuals and reconstructs returns using forecasted volatility

### Horizon Handling

- **Path Simulation**: For multi-day horizons, simulates daily return paths and aggregates (sum of daily returns)
- Avoids square-root-of-time scaling unless explicitly benchmarking
- More accurate for capturing path-dependent risk

### Risk Measure Definitions

- **VaR**: Positive loss number representing the α-quantile of the loss distribution (loss = -return)
- **CVaR**: Expected shortfall - mean loss exceeding VaR threshold
- **Tail Side**: Left-tail risk (losses when returns are negative)

### Rolling Windows

- Configurable window sizes (e.g., 252, 500 days) with step size control
- Warmup policy: Skips periods until window is full
- Parameters re-estimated at each rolling step

## Notes

- **Asset-Level Only**: This module evaluates VaR/CVaR strictly per asset. No portfolio aggregation is performed.
- **Multiple Methods**: Supports comparison across different simulation methods (historical, parametric normal, Student-t, filtered EWMA)
- **VaR Violations**: Occur when actual loss > VaR (left-tail risk: loss_t = -return_t)
- **CVaR Violations**: Occur when actual loss > CVaR (more severe than VaR violations)
- **Traffic Light Zones**: Follow Basel guidelines (adjusted for confidence level). Computed only for 99% VaR per specification.
- **Parameter Caching**: Fitted parameters and risk series are cached to avoid recomputation. Cache hit ratio is tracked for performance monitoring.
- **Time-Sliced Metrics**: Enable analysis of temporal patterns in VaR/CVaR performance across different market regimes.
- **Random Seed**: Configurable random seed ensures reproducibility of Monte Carlo simulations
- **Number of Simulations**: Default 20,000 simulations provides good balance between accuracy and computational cost. Increase for higher precision.

