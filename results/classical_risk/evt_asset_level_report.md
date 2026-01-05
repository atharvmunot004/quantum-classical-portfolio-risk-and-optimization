# Extreme Value Theory (EVT) - Peaks Over Threshold (POT) VaR/CVaR Evaluation Report

**Generated:** 2026-01-05 19:42:42

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

- **Average Hit Rate:** 0.4876

- **Average Violation Ratio:** 35.8952
  - Ratio > 1 indicates overestimation of risk
  - Ratio < 1 indicates underestimation of risk

### Traffic Light Zones

- **Red:** 79 configurations (98.8%)
- **Yellow:** 1 configurations (1.2%)

- **Kupiec Test Passed:** 0/80 configurations (0.0%)

## Tail Risk Behavior

- **Average Mean Exceedance (VaR):** 0.013864

- **Average Max Exceedance (VaR):** 0.165215

- **Average Expected Shortfall Exceedance:** nan

- **Average CVaR Mean Exceedance:** 0.013092

- **Average CVaR Max Exceedance:** 0.161730

- **Average Tail Index (ξ):** -0.4078

- **Average Shape-Scale Stability:** 0.0762
  - Lower values indicate more stable tail behavior

## Computational Performance

- **Average Total Runtime per Portfolio:** 22847.86 ms

## Key Insights

### Findings

- **Risk Overestimation:** EVT-POT tends to overestimate risk

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
       confidence_level    horizon  estimation_window  threshold_quantile   hit_rate  violation_ratio  kupiec_unconditional_coverage  kupiec_test_statistic  christoffersen_independence  christoffersen_independence_statistic  christoffersen_conditional_coverage  christoffersen_conditional_coverage_statistic  num_violations  total_observations  expected_violations  mean_exceedance  max_exceedance  std_exceedance  quantile_loss_score  rmse_var_vs_losses  cvar_mean_exceedance  cvar_max_exceedance  cvar_std_exceedance  rmse_cvar_vs_losses  expected_shortfall_exceedance  tail_index_xi  scale_beta  shape_scale_stability  threshold  num_exceedances   skewness   kurtosis  jarque_bera_p_value  jarque_bera_statistic  runtime_per_portfolio_ms  p95_runtime_ms  mean_runtime_ms  median_runtime_ms  min_runtime_ms  max_runtime_ms  cache_hit_ratio
count         80.000000  80.000000          80.000000               80.00  80.000000        80.000000                           80.0              80.000000                 8.000000e+01                              80.000000                                 80.0                                      80.000000       80.000000           80.000000            80.000000        80.000000       80.000000       80.000000            80.000000           80.000000             80.000000            80.000000            80.000000            80.000000                            0.0      80.000000   80.000000              80.000000  80.000000        80.000000  80.000000  80.000000         8.000000e+01              80.000000                 80.000000       80.000000        80.000000          80.000000        80.00000       80.000000             80.0
mean           0.970000   5.500000         376.000000                0.95   0.487615        35.895175                            0.0            6185.457831                 2.621116e-01                               4.295926                                  0.0                                    6189.753757     1038.387500         2096.000000            62.880000         0.013864        0.165215        0.014210             0.007852            0.019972              0.013092             0.161730             0.014386             0.019556                            NaN      -0.407838    0.002533               0.076214   0.011797        50.500000  -0.271897   9.385981        1.593101e-308            8325.163937              22847.857332    23046.081114     22847.857332       22917.146564     22568.37368    23047.686577              0.0
std            0.020126   4.528391         124.782342                0.00   0.248366        33.034026                            0.0            5623.075996                 3.153827e-01                               5.595798                                  0.0                                    5624.360161      555.239633          124.782342            42.423724         0.005332        0.053766        0.003803             0.006595            0.006148              0.004549             0.052924             0.003807             0.005532                            NaN       0.070792    0.000501               0.011305   0.003447         0.503155   0.431507   2.505422         0.000000e+00            3876.986883                  0.000000        0.000000         0.000000           0.000000         0.00000        0.000000              0.0
min            0.950000   1.000000         252.000000                0.95   0.117647         2.352941                            0.0             139.962002                 5.375713e-08                              -1.952026                                  0.0                                     141.414287      232.000000         1972.000000            19.720000         0.008246        0.081409        0.008955             0.001762            0.012169              0.008078             0.078426             0.008973             0.012225                            NaN      -0.500000    0.001889               0.059530   0.007070        50.000000  -0.931832   4.078599         0.000000e+00            1411.469588              22847.857332    23046.081114     22847.857332       22917.146564     22568.37368    23047.686577              0.0
25%            0.950000   1.000000         252.000000                0.95   0.291075         6.377675                            0.0            1560.331616                 1.446248e-02                               0.655026                                  0.0                                    1561.129713      574.000000         1972.000000            21.580000         0.010388        0.130716        0.011813             0.003240            0.016088              0.010108             0.127934             0.011882             0.016161                            NaN      -0.467510    0.002071               0.066177   0.008856        50.000000  -0.554171   7.709968         0.000000e+00            5321.639316              22847.857332    23046.081114     22847.857332       22917.146564     22568.37368    23047.686577              0.0
50%            0.970000   5.500000         376.000000                0.95   0.496697        20.547302                            0.0            4096.104624                 1.299185e-01                               2.297621                                  0.0                                    4096.735549     1033.500000         2096.000000            60.400000         0.011972        0.163378        0.013091             0.005726            0.018050              0.011758             0.160298             0.013478             0.017946                            NaN      -0.404904    0.002382               0.076327   0.012042        50.500000  -0.355953   9.829733         0.000000e+00            8340.755120              22847.857332    23046.081114     22847.857332       22917.146564     22568.37368    23047.686577              0.0
75%            0.990000  10.000000         500.000000                0.95   0.673835        67.383481                            0.0           10648.412174                 4.199839e-01                               6.011455                                  0.0                                   10658.785497     1437.500000         2220.000000           101.700000         0.015099        0.186334        0.015576             0.010000            0.022033              0.013994             0.183051             0.015706             0.021324                            NaN      -0.351928    0.002938               0.084258   0.014237        51.000000  -0.005238  11.172227         0.000000e+00           11318.195192              22847.857332    23046.081114     22847.857332       22917.146564     22568.37368    23047.686577              0.0
max            0.990000  10.000000         500.000000                0.95   0.936036        93.603604                            0.0           18086.384914                 1.000000e+00                              29.576340                                  0.0                                   18088.110221     2078.000000         2220.000000           111.000000         0.038911        0.334712        0.029887             0.036066            0.049060              0.035084             0.328863             0.028353             0.045104                            NaN      -0.272324    0.003511               0.095926   0.018314        51.000000   0.709688  12.972163        3.186202e-307           14226.861044              22847.857332    23046.081114     22847.857332       22917.146564     22568.37368    23047.686577              0.0
```
