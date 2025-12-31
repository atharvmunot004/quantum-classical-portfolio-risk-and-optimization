# GARCH(1,1) Volatility Forecasting for VaR/CVaR Evaluation Report

**Generated:** 2025-12-30 05:07:14

## Model Configuration

### GARCH(1,1) Model Specification

The GARCH(1,1) model specifies the conditional variance as:

σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁

Where:
- σ²ₜ is the conditional variance at time t
- ω is the constant term
- α captures the ARCH effect (lagged squared residuals)
- β captures the GARCH effect (lagged conditional variance)
- εₜ are the residuals

### Model Parameters

- **GARCH Order (p):** 1
- **ARCH Order (q):** 1
- **Distribution:** normal
- **Mean Model:** Zero
- **Return Type:** log
- **Estimation Windows:** [500] days
- **Confidence Levels:** [0.99]
- **Base Horizon:** 1 days
- **Scaled Horizons:** [10]
- **Scaling Rule:** sqrt_time
- **Forecast Method:** rolling
- **Fallback to Long-Run Variance:** True

### Design Principle

Asset-level GARCH parameters and conditional volatility paths are estimated
once and reused across all portfolio batches to ensure statistical consistency,
computational efficiency, and reproducibility across batched execution.

## Batch Execution Summary

- **Total Portfolio-Configuration Combinations:** 3942000
- **Unique Portfolios:** 1000

## Backtesting Results

- **Average Hit Rate:** 0.0000

- **Average Violation Ratio:** 0.0000
  - Ratio > 1 indicates overestimation of risk
  - Ratio < 1 indicates underestimation of risk

### Traffic Light Zones

- **Green:** 3942000 configurations (100.0%)

- **Kupiec Test Passed:** 0/3942000 configurations (0.0%)

## Tail Risk Analysis

- **Average Mean Exceedance (VaR):** nan

- **Average Max Exceedance (VaR):** nan

- **Average CVaR Mean Exceedance:** nan

- **Average CVaR Max Exceedance:** nan

## Computational Performance

- **Average Total Runtime per Portfolio:** 123911.46 ms

## Detailed Metrics

### Summary Statistics by Metric

```
       portfolio_id  confidence_level    horizon  estimation_window           var          cvar   hit_rate  violation_ratio  kupiec_unconditional_coverage  kupiec_test_statistic  christoffersen_independence  christoffersen_independence_statistic  christoffersen_conditional_coverage  christoffersen_conditional_coverage_statistic  num_violations  total_observations  expected_violations  mean_exceedance  max_exceedance  std_exceedance  quantile_loss_score  rmse_var_vs_losses  cvar_mean_exceedance  cvar_max_exceedance  cvar_std_exceedance  portfolio_size  num_active_assets  hhi_concentration  effective_number_of_assets  covariance_condition_number      skewness      kurtosis  jarque_bera_p_value  jarque_bera_statistic  runtime_per_portfolio_ms  p95_runtime_ms  mean_runtime_ms  median_runtime_ms  min_runtime_ms  max_runtime_ms
count  3.942000e+06        3942000.00  3942000.0          3942000.0  3.942000e+06  3.942000e+06  3942000.0        3942000.0                            0.0                    0.0                          0.0                                    0.0                                  0.0                                            0.0       3942000.0           3942000.0           3942000.00              0.0             0.0             0.0                  0.0                 0.0                   0.0                  0.0                  0.0    3.942000e+06       3.942000e+06       3.942000e+06                3.942000e+06                 3.942000e+06  3.942000e+06  3.942000e+06            3942000.0           3.942000e+06              3.942000e+06    3.942000e+06     3.942000e+06       3.942000e+06    3.942000e+06    3.942000e+06
mean   9.949950e+04              0.99        2.0              500.0  2.813905e+00  3.223791e+00        0.0              0.0                            NaN                    NaN                          NaN                                    NaN                                  NaN                                            NaN             0.0              1971.0                19.71              NaN             NaN             NaN                  NaN                 NaN                   NaN                  NaN                  NaN    6.006000e+00       6.006000e+00       3.376722e-01                3.675193e+00                 1.188948e+01 -9.272793e-01  1.515851e+01                  0.0           2.062622e+04              1.239115e+05    1.239115e+05     1.239115e+05       1.239115e+05    1.239115e+05    1.239115e+05
std    2.886750e+02              0.00        1.0                0.0  1.495924e+00  1.713827e+00        0.0              0.0                            NaN                    NaN                          NaN                                    NaN                                  NaN                                            NaN             0.0                 0.0                 0.00              NaN             NaN             NaN                  NaN                 NaN                   NaN                  NaN                  NaN    2.512761e+00       2.512761e+00       1.769459e-01                1.548869e+00                 0.000000e+00  4.313446e-01  4.146361e+00                  0.0           1.025339e+04              1.455192e-11    1.455192e-11     1.455192e-11       1.455192e-11    1.455192e-11    1.455192e-11
min    9.900000e+04              0.99        1.0              500.0  3.724651e-01  4.267200e-01        0.0              0.0                            NaN                    NaN                          NaN                                    NaN                                  NaN                                            NaN             0.0              1971.0                19.71              NaN             NaN             NaN                  NaN                 NaN                   NaN                  NaN                  NaN    2.000000e+00       2.000000e+00       1.276638e-01                1.005968e+00                 1.188948e+01 -1.677212e+00  4.187520e+00                  0.0           1.440098e+03              1.239115e+05    1.239115e+05     1.239115e+05       1.239115e+05    1.239115e+05    1.239115e+05
25%    9.924975e+04              0.99        1.0              500.0  1.829062e+00  2.095491e+00        0.0              0.0                            NaN                    NaN                          NaN                                    NaN                                  NaN                                            NaN             0.0              1971.0                19.71              NaN             NaN             NaN                  NaN                 NaN                   NaN                  NaN                  NaN    4.000000e+00       4.000000e+00       2.079101e-01                2.412764e+00                 1.188948e+01 -1.250473e+00  1.218809e+01                  0.0           1.246124e+04              1.239115e+05    1.239115e+05     1.239115e+05       1.239115e+05    1.239115e+05    1.239115e+05
50%    9.949950e+04              0.99        2.0              500.0  2.504465e+00  2.869276e+00        0.0              0.0                            NaN                    NaN                          NaN                                    NaN                                  NaN                                            NaN             0.0              1971.0                19.71              NaN             NaN             NaN                  NaN                 NaN                   NaN                  NaN                  NaN    6.000000e+00       6.000000e+00       2.858835e-01                3.497933e+00                 1.188948e+01 -9.906915e-01  1.542449e+01                  0.0           1.982140e+04              1.239115e+05    1.239115e+05     1.239115e+05       1.239115e+05    1.239115e+05    1.239115e+05
75%    9.974925e+04              0.99        3.0              500.0  3.357019e+00  3.846018e+00        0.0              0.0                            NaN                    NaN                          NaN                                    NaN                                  NaN                                            NaN             0.0              1971.0                19.71              NaN             NaN             NaN                  NaN                 NaN                   NaN                  NaN                  NaN    8.000000e+00       8.000000e+00       4.144647e-01                4.809775e+00                 1.188948e+01 -6.856648e-01  1.827206e+01                  0.0           2.787988e+04              1.239115e+05    1.239115e+05     1.239115e+05       1.239115e+05    1.239115e+05    1.239115e+05
max    9.999900e+04              0.99        3.0              500.0  5.024687e+01  5.756605e+01        0.0              0.0                            NaN                    NaN                          NaN                                    NaN                                  NaN                                            NaN             0.0              1971.0                19.71              NaN             NaN             NaN                  NaN                 NaN                   NaN                  NaN                  NaN    1.000000e+01       1.000000e+01       9.940674e-01                7.833076e+00                 1.188948e+01  7.077154e-01  2.375687e+01                  0.0           4.703994e+04              1.239115e+05    1.239115e+05     1.239115e+05       1.239115e+05    1.239115e+05    1.239115e+05
```
