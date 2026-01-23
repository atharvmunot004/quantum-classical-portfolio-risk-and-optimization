# Extreme Value Theory (EVT) - Peaks Over Threshold (POT) Value-at-Risk Evaluation

This module implements comprehensive evaluation of Extreme Value Theory (EVT) - Peaks Over Threshold (POT) Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) at the asset level.

## Overview

The EVT-POT method models extreme tail events by focusing on exceedances over a high threshold and fitting a Generalized Pareto Distribution (GPD) to the tail. This approach provides more accurate estimates of extreme risk measures compared to parametric methods that assume normal distributions.

**Key Design Principle**: EVT-POT is fitted and evaluated strictly on individual asset return series. VaR/CVaR are produced as time series via rolling estimation windows. All backtesting and diagnostics are computed per asset and time-sliced (year/quarter/month) to support empirical evaluation without portfolio-level aggregation assumptions.

### Theoretical Foundation

The Peaks Over Threshold method is based on the Pickands-Balkema-de Haan Theorem:
- For a sufficiently high threshold u, the distribution of exceedances (X - u | X > u) converges to a Generalized Pareto Distribution (GPD) as u increases.

**GPD Parameters:**
- **Shape parameter (ξ)**: Determines tail behavior
  - ξ > 0: Heavy-tailed (Pareto-type)
  - ξ = 0: Exponential tail
  - ξ < 0: Light-tailed (bounded)
- **Scale parameter (β)**: Controls dispersion

**VaR and CVaR from EVT:**
- VaR is computed from the GPD distribution fitted to exceedances
- CVaR (Expected Shortfall) is the expected loss given that VaR is exceeded
- Both are scaled by horizon using square-root-of-time rule

## Features

- **Asset-Level Evaluation**: Processes each asset independently (no portfolio aggregation)
- **Daily Returns and Losses Computation**: Computes log or simple returns from price data, then converts to losses (loss_t = -returns_t)
- **Rolling EVT Parameter Estimation**: Estimates GPD parameters (ξ, β) using rolling windows per asset
- **Threshold Selection**: Automatic threshold selection using quantile-based method with fallback options
- **Rolling VaR/CVaR**: Implements rolling window VaR and CVaR calculation using EVT-POT methodology
- **GPD Fitting Methods**: Supports Probability Weighted Moments (PWM) method for robust parameter estimation
- **Parameter Caching**: Caches EVT parameters (threshold, ξ, β) for computational efficiency
- **Backtesting**: Comprehensive backtesting with violation analysis per asset
- **Accuracy Metrics**: Hit rate, violation ratio, Kupiec test, Christoffersen tests, traffic light zones
- **Tail Risk Metrics**: Mean exceedance, max exceedance, quantile loss, RMSE for both VaR and CVaR
- **EVT-Specific Metrics**: Tail index (ξ), scale parameter (β), shape-scale stability, expected shortfall exceedance
- **Distribution Metrics**: Skewness, kurtosis, Jarque-Bera normality test
- **Time-Sliced Metrics**: Backtesting metrics by year, quarter, and month
- **Safety Checks**: Guardrails for VaR positivity, minimum exceedances, and parameter constraints
- **Report Generation**: Comprehensive markdown reports

## Usage

### Command Line

```bash
# Run with default configuration (llm.json in same directory)
python -m src.classical_risk.extreme_value_theorem_peaks_over_threshold.main

# Specify custom configuration file
python -m src.classical_risk.extreme_value_theorem_peaks_over_threshold.main --config path/to/config.json

# Use specific number of parallel workers
python -m src.classical_risk.extreme_value_theorem_peaks_over_threshold.main --n-jobs 8

# Sequential processing (for debugging)
python -m src.classical_risk.extreme_value_theorem_peaks_over_threshold.main --n-jobs 1
```

### Python API

```python
from src.classical_risk.extreme_value_theorem_peaks_over_threshold import evaluate_evt_pot_var_cvar

# Load configuration and evaluate (default: auto workers)
risk_series_df, metrics_df, time_sliced_metrics_df = evaluate_evt_pot_var_cvar(config_path='llm.json')

# Use specific number of workers
risk_series_df, metrics_df, time_sliced_metrics_df = evaluate_evt_pot_var_cvar(
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
    'evt_settings': {
        'method': 'peaks_over_threshold',
        'distribution': 'generalized_pareto',
        'gpd_fitting_method': 'pwm',
        'threshold_selection': {
            'quantiles': [0.95],
            'min_exceedances': 50,
            'fallback_quantiles_if_insufficient': [0.9, 0.85, 0.8, 0.75, 0.7]
        },
        'shape_constraints': {
            'xi_lower_bound': -0.5,
            'xi_upper_bound': 0.5
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

risk_series_df, metrics_df, time_sliced_metrics_df = evaluate_evt_pot_var_cvar(config_dict=config, n_jobs=8)
```

## Configuration

The module uses a JSON configuration file (`llm.json`) with the following structure:

```json
{
  "task": "evt_pot_var_cvar_asset_level_evaluation_optimized",
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
  "evt_settings": {
    "method": "peaks_over_threshold",
    "distribution": "generalized_pareto",
    "gpd_fitting_method": "pwm",
    "threshold_selection": {
      "method": "quantile",
      "quantiles": [0.95],
      "min_exceedances": 50,
      "fallback_quantiles_if_insufficient": [0.9, 0.85, 0.8, 0.75, 0.7]
    },
    "shape_constraints": {
      "xi_lower_bound": -0.5,
      "xi_upper_bound": 0.5
    },
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
      "parameter_store_path": "cache/evt_asset_parameters.parquet"
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
      "path": "cache/evt_asset_parameters.parquet"
    },
    "risk_series_store": {
      "path": "results/classical_risk/evt_asset_var_cvar_series.parquet"
    },
    "metrics_table": {
      "path": "results/classical_risk/evt_asset_level_metrics.parquet"
    },
    "time_sliced_metrics_table": {
      "path": "results/classical_risk/evt_asset_level_time_sliced_metrics.parquet"
    },
    "evt_results_scheme.json": {
      "path": "results/classical_risk/evt_asset_level_results_scheme.json"
    },
    "report": {
      "path": "results/classical_risk/evt_asset_level_report.md"
    }
  }
}
```

### Key Configuration Parameters

- **`gpd_fitting_method`**: Method for fitting GPD (`'pwm'` for Probability Weighted Moments, recommended)
- **`threshold_selection.quantiles`**: Quantile levels for threshold selection (e.g., 0.95 means 95th percentile of losses)
- **`threshold_selection.min_exceedances`**: Minimum number of exceedances required for GPD fitting
- **`threshold_selection.fallback_quantiles_if_insufficient`**: Fallback quantiles if minimum exceedances not met
- **`shape_constraints.xi_lower_bound`** / **`xi_upper_bound`**: Bounds for shape parameter ξ (typically [-0.5, 0.5])
- **`confidence_levels`**: VaR/CVaR confidence levels (e.g., [0.95, 0.99])
- **`estimation_windows`**: Rolling window sizes in days (e.g., [252, 500])

## Output

The module generates:

1. **Risk Series** (`evt_asset_var_cvar_series.parquet`): Time series of VaR and CVaR values per asset, date, confidence level, horizon, estimation window, and threshold quantile. Each row represents a single VaR/CVaR estimate at a specific point in time.

2. **Metrics Table** (`evt_asset_level_metrics.parquet`): Aggregated backtesting metrics over the full evaluation period per asset. Row granularity: asset × confidence_level × horizon × estimation_window × threshold_quantile. Includes:
   - **Accuracy metrics**: hit rate, violation ratio, Kupiec test, Christoffersen tests, traffic light zone
   - **Tail behavior metrics (VaR)**: mean exceedance, max exceedance, quantile loss, RMSE
   - **Tail behavior metrics (CVaR)**: CVaR mean exceedance, max exceedance, RMSE
   - **EVT-specific metrics**: tail index (ξ), scale parameter (β), shape-scale stability, expected shortfall exceedance, threshold, number of exceedances
   - **Distributional characteristics**: skewness, kurtosis, Jarque-Bera test

3. **Time-Sliced Metrics** (`evt_asset_level_time_sliced_metrics.parquet`): Backtesting metrics computed for specific time periods (year, quarter, month). Enables analysis of temporal patterns in VaR/CVaR performance across different market regimes.

4. **Parameter Store** (`cache/evt_asset_parameters.parquet`): Cached EVT parameters (threshold, ξ, β, number of exceedances) estimated per asset, date, estimation window, and threshold quantile. Used for computational efficiency.

5. **Results Schema** (`evt_asset_level_results_scheme.json`): JSON schema documenting the structure of all output files, including row counts and column descriptions.

6. **Report** (`evt_asset_level_report.md`): Comprehensive markdown report with:
   - Methodology overview
   - EVT theory and POT framework
   - Threshold selection procedure
   - GPD parameter estimation details
   - EVT-based VaR/CVaR construction
   - Backtesting results
   - Time-sliced backtesting
   - Tail risk behavior
   - Distributional characteristics
   - Computational performance
   - Key insights and recommendations

## Module Structure

- `main.py`: Main orchestration script for asset-level evaluation
- `evt_calculator.py`: EVT-POT calculation with GPD fitting and rolling windows
- `returns.py`: Asset-level returns and losses computation
- `backtesting.py`: Violation detection and accuracy metrics
- `metrics.py`: Tail risk, EVT-specific, and distribution metrics (asset-level)
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
  - EVT parameter caching (threshold, ξ, β) for rolling windows
  - Efficient GPD fitting using PWM method
  - Memory-efficient processing per asset
  - Fallback threshold selection for insufficient exceedances

## Methodology Notes

- **Distributional Assumption**: Generalized Pareto Distribution (GPD) fitted to exceedances over threshold
- **GPD Fitting Method**: Probability Weighted Moments (PWM) - more robust than MLE for small samples
- **Threshold Selection**: Quantile-based method (e.g., 95th percentile of losses) with automatic fallback
- **Shape Parameter Constraints**: ξ typically constrained to [-0.5, 0.5] for financial applications
- **Minimum Exceedances**: Requires minimum number of exceedances (default: 50) for reliable GPD fitting
- **Horizon Scaling**: Square-root-of-time rule (VaR_h = VaR_1 * √(h))
- **Rolling Windows**: Configurable window sizes (e.g., 252, 500 days) with step size control
- **Warmup Policy**: Skips periods until window is full
- **Loss Space**: All calculations performed in loss space (loss_t = -returns_t) for EVT consistency

## Notes

- **Asset-Level Only**: This module evaluates VaR/CVaR strictly per asset. No portfolio aggregation is performed.
- **EVT-POT Methodology**: The method specifically models extreme tail events, making it more suitable for high confidence levels (99%, 99.5%) than parametric methods.
- **VaR Violations**: Occur when actual loss > VaR (losses are positive values representing tail risk)
- **CVaR Violations**: Occur when actual loss > CVaR (CVaR represents expected loss given VaR is exceeded)
- **Traffic Light Zones**: Follow Basel guidelines (adjusted for confidence level). Computed only for 99% VaR per specification.
- **Parameter Caching**: EVT parameters (threshold, ξ, β) are cached to avoid recomputation. Cache hit ratio is tracked for performance monitoring.
- **Time-Sliced Metrics**: Enable analysis of temporal patterns in VaR/CVaR performance across different market regimes.
- **Shape Parameter Interpretation**: 
  - Positive ξ indicates heavy tails (higher extreme risk)
  - ξ close to 0 indicates exponential tails
  - Negative ξ indicates light tails (bounded support)
- **Safety Checks**: 
  - VaR must be positive (VaR > 0)
  - Minimum exceedances must be met for GPD fitting
  - Shape parameter must be within bounds
  - CVaR domain checks (requires ξ < 1 for finite CVaR)

