# Extreme Value Theory (EVT) - Peaks Over Threshold (POT) VaR/CVaR Evaluation Report

**Generated:** 2026-01-06 09:39:49

## Methodology Overview

### Extreme Value Theory - Peaks Over Threshold

EVT-POT is a statistical approach for modeling extreme tail events in financial returns.
It focuses on exceedances over a high threshold and fits a Generalized Pareto Distribution (GPD)
to model the tail behavior, providing more accurate estimates of extreme risk measures.

**Advantages:**
- Specifically designed for extreme tail events
- Captures heavy-tailed distributions better than parametric methods
- Provides tail index (shape parameter) for tail risk assessment
- More robust for high confidence levels (e.g., 99%, 99.5%)

## Threshold Selection Procedure

### Threshold Selection Methods

- **Selection Method:** quantile
- **Automatic Selection:** False
- **Quantiles Tested:** [0.95]
- **Minimum Exceedances Required:** 50

## GPD Parameter Estimation

### Maximum Likelihood Estimation

GPD parameters are estimated using Maximum Likelihood Estimation (MLE) on exceedances.

- **Fitting Method:** pwm
- **Shape Parameter (ξ) Bounds:** [-0.5, 0.5]

- **Average Tail Index (ξ):** -0.4078
  - Negative ξ indicates light-tailed distribution (bounded)

## Summary Statistics

- **Total Asset-Configuration Combinations:** 80
- **Number of Assets Evaluated:** 10

## Backtesting Results

### Hit Rate Calibration Check

**Confidence Level 95%:**
  - Expected violation rate: 0.0500
  - Observed hit rate: 0.0576
  - Difference: 0.0076
  - ✓ Within acceptable bounds [0.030, 0.080]

**Confidence Level 99%:**
  - Expected violation rate: 0.0100
  - Observed hit rate: 0.0440
  - Difference: 0.0340
  - ⚠ Outside acceptable bounds [0.005, 0.020]

- **Average Violation Ratio:** 2.7753
  - Ratio > 1 indicates underestimation of risk (excessive VaR breaches)
  - Ratio < 1 indicates overestimation of risk
  - ⚠ Warning: Violation ratio outside acceptable bounds [0.7, 1.3]

### Traffic Light Zones

- **Green:** 45 configurations (56.2%)
- **Red:** 30 configurations (37.5%)
- **Yellow:** 5 configurations (6.2%)

- **Kupiec Test Passed:** 3/80 configurations (3.8%)

## Tail Risk Behavior

- **Average Mean Exceedance (VaR):** 0.018998

- **Average Max Exceedance (VaR):** 0.122403

- **Average Expected Shortfall Exceedance:** 0.018998

- **Average CVaR Mean Exceedance:** 0.021817

- **Average CVaR Max Exceedance:** 0.114812

- **Average Tail Index (ξ):** -0.4078 (std: 0.0708)
  - ⚠ Warning: 32/80 (40.0%) configurations near lower bound (-0.5)
    This may indicate threshold instability or poor tail regime

- **Average Shape-Scale Stability:** 0.0762
  - Lower values indicate more stable tail behavior

## Computational Performance

## Key Insights

### Findings

- **Risk Underestimation:** EVT-POT tends to underestimate risk (excessive VaR breaches)

- **Light-Tailed Distribution:** Negative tail index indicates bounded tails

### Recommendations

- EVT-POT is particularly effective for high confidence levels (99%, 99.5%)
- Monitor tail index (ξ) for stability across assets and time
- Adjust threshold selection based on available data and required exceedances
- Consider EVT-POT for assets with heavy-tailed return distributions
- Rolling window estimation provides time-varying risk measures

## Detailed Metrics

### Summary Statistics by Metric

```
       confidence_level    horizon  estimation_window  threshold_quantile   hit_rate  violation_ratio  kupiec_unconditional_coverage  kupiec_test_statistic  christoffersen_independence  christoffersen_independence_statistic  christoffersen_conditional_coverage  christoffersen_conditional_coverage_statistic  num_violations  total_observations  expected_violations  mean_exceedance  max_exceedance  std_exceedance  quantile_loss_score  rmse_var_vs_losses  cvar_mean_exceedance  cvar_max_exceedance  cvar_std_exceedance  rmse_cvar_vs_losses  expected_shortfall_exceedance  tail_index_xi  scale_beta  shape_scale_stability  threshold  num_exceedances   skewness   kurtosis  jarque_bera_p_value  jarque_bera_statistic
count         80.000000  80.000000          80.000000               80.00  80.000000        80.000000                      80.000000              80.000000                 8.000000e+01                              80.000000                            80.000000                                      80.000000       80.000000           80.000000            80.000000        80.000000       80.000000       80.000000            80.000000           80.000000             80.000000            80.000000            78.000000            80.000000                      80.000000      80.000000   80.000000              80.000000  80.000000        80.000000  80.000000  80.000000         8.000000e+01              80.000000
mean           0.970000   5.500000         376.000000                0.95   0.050785         2.775259                       0.015693             187.527507                 2.136283e-01                               5.398401                             0.004064                                     192.925909      108.150000         2096.000000            62.880000         0.018998        0.122403        0.023595             0.037138            0.029744              0.021817             0.114812             0.024899             0.032514                       0.018998      -0.407838    0.002533               0.076214   0.011797        50.500000  -0.271897   9.385981        1.593101e-308            8325.163937
std            0.020126   4.528391         124.782342                0.00   0.049808         3.535271                       0.076510             217.585248                 3.258471e-01                               5.564584                             0.020195                                     218.828790      109.134118          124.782342            42.423724         0.009240        0.048274        0.010750             0.020570            0.013119              0.017116             0.048674             0.011473             0.018558                       0.009240       0.070792    0.000501               0.011305   0.003447         0.503155   0.431507   2.505422         0.000000e+00            3876.986883
min            0.950000   1.000000         252.000000                0.95   0.001521         0.050710                       0.000000               0.489831                 5.992169e-08                               0.009146                             0.000000                                       3.767271        3.000000         1972.000000            19.720000         0.008235        0.028624        0.009902             0.012610            0.012866              0.008648             0.016385             0.010459             0.013552                       0.008235      -0.500000    0.001889               0.059530   0.007070        50.000000  -0.931832   4.078599         0.000000e+00            1411.469588
25%            0.950000   1.000000         252.000000                0.95   0.004564         0.216216                       0.000000              22.284662                 4.475460e-03                               1.334968                             0.000000                                      22.293809       10.000000         1972.000000            21.580000         0.011074        0.083677        0.015308             0.018741            0.018875              0.011514             0.077980             0.016257             0.019895                       0.011074      -0.467510    0.002071               0.066177   0.008856        50.000000  -0.554171   7.709968         0.000000e+00            5321.639316
50%            0.970000   5.500000         376.000000                0.95   0.033999         1.285383                       0.000000             132.297302                 3.732575e-02                               4.335561                             0.000000                                     135.447526       69.000000         2096.000000            60.400000         0.014555        0.121241        0.019368             0.033731            0.023564              0.015557             0.115251             0.020779             0.025001                       0.014555      -0.404904    0.002382               0.076327   0.012042        50.500000  -0.355953   9.829733         0.000000e+00            8340.755120
75%            0.990000  10.000000         500.000000                0.95   0.093966         3.461186                       0.000002             237.659556                 2.480579e-01                               8.080152                             0.000014                                     243.508863      191.250000         2220.000000           101.700000         0.027037        0.155509        0.031426             0.053831            0.040577              0.028097             0.150542             0.033943             0.042745                       0.027037      -0.351928    0.002938               0.084258   0.014237        51.000000  -0.005238  11.172227         0.000000e+00           11318.195192
max            0.990000  10.000000         500.000000                0.95   0.143694        11.396396                       0.484003             794.781624                 9.238093e-01                              29.365937                             0.152036                                     805.031720      319.000000         2220.000000           111.000000         0.040194        0.247098        0.055613             0.096377            0.059806              0.141614             0.241237             0.055199             0.141614                       0.040194      -0.272324    0.003511               0.095926   0.018314        51.000000   0.709688  12.972163        3.186202e-307           14226.861044
```
