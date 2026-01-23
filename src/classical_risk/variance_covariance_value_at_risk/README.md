# Variance-Covariance Value-at-Risk Evaluation

This module implements comprehensive evaluation of Variance-Covariance (Parametric) Value-at-Risk at the asset level.

## Overview

The Variance-Covariance method assumes returns follow a normal distribution and calculates VaR as:

```
VaR_h = -μ + z_{1-α} * σ * √(h)
```

Where:
- μ = mean return (estimated from rolling window)
- σ = standard deviation of returns (estimated from rolling window)
- z_{1-α} = (1-α) quantile of standard normal distribution (positive for α < 0.5)
- α = 1 - confidence_level (e.g., α = 0.05 for 95% VaR)
- h = time horizon in days
- √(h) = square root scaling for multi-day horizons

**Key Design Principle**: VaR is estimated strictly per asset using rolling windows under the assumption of conditional normality. No portfolio aggregation or cross-asset covariance modeling is performed, ensuring methodological comparability with other single-asset risk models.

## Features

- **Asset-Level Evaluation**: Processes each asset independently (no portfolio aggregation)
- **Daily Returns Computation**: Computes log or simple returns from price data (once, then reused)
- **Rolling Mean and Volatility Estimation**: Estimates parameters using rolling windows per asset
- **Rolling VaR**: Implements rolling window VaR calculation with normal distribution assumption
- **Backtesting**: Comprehensive backtesting with violation analysis per asset
- **Accuracy Metrics**: Hit rate, violation ratio, Kupiec test, Christoffersen test
- **Tail Risk Metrics**: Mean exceedance, max exceedance, quantile loss, RMSE
- **Distribution Metrics**: Skewness, kurtosis, Jarque-Bera normality test
- **Time-Sliced Metrics**: Backtesting metrics by year, quarter, and month
- **Parameter Caching**: Caches rolling mean and volatility for computational efficiency
- **Report Generation**: Comprehensive markdown reports

## Usage

### Command Line

```bash
# Run with default configuration (llm.json in same directory)
python -m src.classical_risk.variance_covariance_value_at_risk.main

# Specify custom configuration file
python -m src.classical_risk.variance_covariance_value_at_risk.main --config path/to/config.json

# Use specific number of parallel workers
python -m src.classical_risk.variance_covariance_value_at_risk.main --n-jobs 8

# Sequential processing (for debugging)
python -m src.classical_risk.variance_covariance_value_at_risk.main --n-jobs 1
```

### Python API

```python
from src.classical_risk.variance_covariance_value_at_risk import evaluate_var

# Load configuration and evaluate (default: auto workers)
risk_series_df, metrics_df, time_sliced_metrics_df = evaluate_var(config_path='llm.json')

# Use specific number of workers
risk_series_df, metrics_df, time_sliced_metrics_df = evaluate_var(
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
    'variance_covariance_settings': {
        'distributional_assumption': 'normal',
        'mean_estimator': 'sample_mean',
        'volatility_estimator': 'sample_std',
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

risk_series_df, metrics_df, time_sliced_metrics_df = evaluate_var(config_dict=config, n_jobs=8)
```

## Configuration

The module uses a JSON configuration file (`llm.json`) with the following structure:

```json
{
  "task": "variance_covariance_var_asset_level_evaluation",
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
  "variance_covariance_settings": {
    "distributional_assumption": "normal",
    "mean_estimator": "sample_mean",
    "volatility_estimator": "sample_std",
    "confidence_levels": [0.95, 0.99],
    "horizons": {
      "base_horizon": 1,
      "scaled_horizons": [10],
      "scaling_rule": "sqrt_time"
    },
    "estimation_windows": [252, 500],
    "rolling": {
      "enabled": true,
      "step_size": 1,
      "warmup_policy": "skip_until_window_full"
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
      "parameter_store_path": "cache/varcov_asset_parameters.parquet"
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
      "path": "cache/varcov_asset_parameters.parquet"
    },
    "risk_series_store": {
      "path": "results/classical_risk/varcov_asset_var_series.parquet"
    },
    "metrics_table": {
      "path": "results/classical_risk/varcov_asset_level_metrics.parquet"
    },
    "time_sliced_metrics_table": {
      "path": "results/classical_risk/varcov_asset_level_time_sliced_metrics.parquet"
    },
    "results_scheme.json": {
      "path": "results/classical_risk/varcov_asset_level_results_scheme.json"
    },
    "report": {
      "path": "results/classical_risk/varcov_asset_level_report.md"
    }
  }
}
```

## Output

The module generates:

1. **Risk Series** (`varcov_asset_var_series.parquet`): Time series of VaR values per asset, date, confidence level, horizon, and estimation window. Each row represents a single VaR estimate at a specific point in time.

2. **Metrics Table** (`varcov_asset_level_metrics.parquet`): Aggregated backtesting metrics over the full evaluation period per asset. Row granularity: asset × confidence_level × horizon × estimation_window. Includes:
   - Accuracy metrics: hit rate, violation ratio, Kupiec test, Christoffersen tests, traffic light zone
   - Tail behavior metrics: mean exceedance, max exceedance, quantile loss, RMSE
   - Distributional characteristics: rolling mean, rolling volatility, skewness, kurtosis, Jarque-Bera test

3. **Time-Sliced Metrics** (`varcov_asset_level_time_sliced_metrics.parquet`): Backtesting metrics computed for specific time periods (year, quarter, month). Enables analysis of temporal patterns in VaR performance.

4. **Parameter Store** (`cache/varcov_asset_parameters.parquet`): Cached rolling mean and volatility parameters estimated per asset, date, and estimation window. Used for computational efficiency.

5. **Results Schema** (`varcov_asset_level_results_scheme.json`): JSON schema documenting the structure of all output files, including row counts and column descriptions.

6. **Report** (`varcov_asset_level_report.md`): Comprehensive markdown report with:
   - Methodology overview
   - Normality assumption analysis
   - Rolling mean and volatility estimation details
   - Variance-Covariance VaR construction
   - Backtesting results
   - Time-sliced backtesting
   - Distributional characteristics
   - Computational performance
   - Key insights and recommendations

## Module Structure

- `main.py`: Main orchestration script for asset-level evaluation
- `returns.py`: Asset-level returns computation
- `var_calculator.py`: Variance-Covariance VaR calculation with rolling windows
- `backtesting.py`: Violation detection and accuracy metrics
- `metrics.py`: Tail risk and distribution metrics (asset-level)
- `time_sliced_metrics.py`: Time-sliced backtesting metrics
- `report_generator.py`: Markdown report generation
- `update_results_scheme.py`: Script to update results schema JSON with actual data

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
  - Parameter caching for rolling mean and volatility
  - Efficient rolling window implementation
  - Memory-efficient processing per asset

## Methodology Notes

- **Distributional Assumption**: Normal distribution (tested using Jarque-Bera test)
- **Mean Estimation**: Sample mean from rolling window
- **Volatility Estimation**: Sample standard deviation from rolling window
- **Horizon Scaling**: Square-root-of-time rule (VaR_h = VaR_1 * √(h))
- **Rolling Windows**: Configurable window sizes (e.g., 252, 500 days) with step size control
- **Warmup Policy**: Skips periods until window is full

## Notes

- **Asset-Level Only**: This module evaluates VaR strictly per asset. No portfolio aggregation is performed.
- **Normality Assumption**: The method assumes returns follow a normal distribution. Results include normality tests (Jarque-Bera) to validate this assumption.
- **VaR Violations**: Occur when actual return < -VaR (left-tail risk)
- **Traffic Light Zones**: Follow Basel guidelines (adjusted for confidence level). Computed only for 99% VaR per specification.
- **Parameter Caching**: Rolling mean and volatility are cached to avoid recomputation. Cache hit ratio is tracked for performance monitoring.
- **Time-Sliced Metrics**: Enable analysis of temporal patterns in VaR performance across different market regimes.

## Updating Results Schema

After generating results, update the schema JSON file with actual row counts:

```bash
python -m src.classical_risk.variance_covariance_value_at_risk.update_results_scheme
```

This script reads the parquet files and updates the schema JSON with actual row counts and column information.
