# Monte Carlo Simulation for VaR/CVaR Evaluation Report (Asset Level)

**Generated:** 2026-01-21 01:08:27

**Scope:** Asset-level evaluation only (no portfolio aggregation)

## Methodology Overview

Monte Carlo simulation generates multiple scenarios of asset returns based on historical data.
VaR and CVaR are then calculated from the distribution of simulated returns per asset.

**Design Principle:**
- Asset-level only: evaluation is performed strictly on individual asset return series
- No portfolio aggregation: all metrics computed per asset
- Rolling window parameter estimation per asset
- Forward return paths simulated per asset

## Monte Carlo Methods and Assumptions

### Historical Bootstrap

- **Bootstrap Type:** iid

### Parametric Normal

- **Mean:** sample_mean
- **Volatility:** sample_std

### Parametric Student T

- **Mean:** sample_mean
- **Volatility:** sample_std
- **DF Mode:** mle_or_fixed
- **Fixed DF:** 5

### Horizon Handling

- **Method:** path_simulation
- **Base Horizon:** 1 days
- **Scaled Horizons:** [10] days
- **Path Aggregation:** sum

## Rolling Forecast Construction

- **Step Size:** 1
- **Forecast Method:** rolling
- **Forecast Target:** one_step_ahead
- **Estimation Windows:** [252, 500] days
- **Number of Simulations:** 20,000

## Backtesting Results

- **Average Hit Rate:** 0.0144

- **Average Violation Ratio:** 0.5414

### Traffic Light Zones

- **Green:** 172 (71.7%)
- **Red:** 60 (25.0%)
- **Yellow:** 8 (3.3%)

- **Kupiec Test Passed:** 80/240 (33.3%)

- **Christoffersen Conditional Coverage Passed:** 53/240 (22.1%)

## Time-Sliced Backtesting

### By Year

- **Average Hit Rate:** 0.0133


### By Quarter

- **Average Hit Rate:** 0.0141


### By Month

- **Average Hit Rate:** 0.0144


## Tail Risk Behavior

- **Average Mean Exceedance (VaR):** 0.021937

- **Average Max Exceedance (VaR):** 0.089403

- **Average CVaR Mean Exceedance:** 0.022377

## Distributional Characteristics

- **Average Skewness:** -0.2719

- **Average Excess Kurtosis:** 9.3860

- **Jarque-Bera Normality Test Passed:** 0/240 (0.0%)

## Key Insights

### Method Comparison

```
                      hit_rate  violation_ratio  kupiec_unconditional_coverage
method                                                                        
historical_bootstrap  0.014869         0.477510                         0.4875
parametric_normal     0.014226         0.592673                         0.2250
parametric_student_t  0.014104         0.554017                         0.2875
```

### Confidence Level Comparison

```
                  hit_rate  violation_ratio
confidence_level                           
0.95              0.022464         0.449273
0.99              0.006335         0.633527
```
