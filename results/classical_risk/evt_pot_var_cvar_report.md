# Extreme Value Theory (EVT) - Peaks Over Threshold (POT) VaR/CVaR Evaluation Report

**Generated:** 2025-12-13 02:53:44

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

## EVT Theory and POT Framework

### Theoretical Foundation

Extreme Value Theory (EVT) provides a framework for modeling the tail behavior of distributions.
The Peaks Over Threshold (POT) method focuses on observations exceeding a high threshold.

**Key Theorem:** Pickands-Balkema-de Haan Theorem
- For a sufficiently high threshold u, the distribution of exceedances (X - u | X > u)
  converges to a Generalized Pareto Distribution (GPD) as u increases.

**GPD Distribution:**
- Shape parameter (ξ): Determines tail behavior
  - ξ > 0: Heavy-tailed (Pareto-type)
  - ξ = 0: Exponential tail
  - ξ < 0: Light-tailed (bounded)
- Scale parameter (β): Controls dispersion

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

- **Average Tail Index (ξ):** -0.4490
  - Negative ξ indicates light-tailed distribution (bounded)

## EVT-Based VaR/CVaR Calculation

### From GPD to Risk Measures

**VaR Calculation:**
- VaR is computed using the GPD distribution fitted to exceedances
- Accounts for probability of exceedance and tail behavior
- Formula: VaR = u + (β/ξ) · [((n/nu) · (1-α))^(-ξ) - 1]
  - Where u is threshold, β is scale, ξ is shape, n is sample size, nu is exceedances, α is confidence level

**CVaR Calculation:**
- CVaR (Expected Shortfall) is the expected loss given VaR is exceeded
- Formula: CVaR = VaR + (β - ξ·(VaR - u)) / (1 - ξ) for ξ < 1

- **Confidence Levels:** [0.99]
- **Horizons:** {'base_horizon': 1, 'scaled_horizons': [10], 'scaling_rule': 'sqrt_time'} days
- **Estimation Windows:** [500] days

## Summary Statistics

- **Total Portfolio-Configuration Combinations:** 200000

## Backtesting Results

- **Average Hit Rate:** 0.5179

- **Average Violation Ratio:** 51.7932
  - Ratio > 1 indicates overestimation of risk
  - Ratio < 1 indicates underestimation of risk

### Traffic Light Zones

- **Red:** 200000 configurations (100.0%)

- **Kupiec Test Passed:** 0/200000 configurations (0.0%)

## Tail Risk Behavior

- **Average Mean Exceedance (VaR):** 0.009455

- **Average Max Exceedance (VaR):** 0.143127

- **Average Expected Shortfall Exceedance:** nan

- **Average CVaR Mean Exceedance:** 0.008860

- **Average CVaR Max Exceedance:** 0.140509

- **Average Tail Index (ξ):** -0.4490

- **Average Shape-Scale Stability:** nan
  - Lower values indicate more stable tail behavior

## Portfolio Structure Effects

- **Average Number of Active Assets:** 5.98

- **Average HHI Concentration:** 0.3383

## Stability and Robustness Checks

- **Tail Index (ξ) Standard Deviation:** 0.0000
  - Lower values indicate more stable tail behavior across portfolios

- **Average Shape-Scale Stability:** nan

- **Average Skewness:** -0.8125

- **Average Excess Kurtosis:** 14.1372

- **Normality Tests Passed (Jarque-Bera):** 0/200000 configurations (0.0%)

## Computational Performance

- **Average EVT Fitting Time per Configuration:** 0.00 ms

- **Average Threshold Selection Time per Configuration:** 0.00 ms

- **Average Total Runtime per Portfolio:** 2.59 ms

## Key Insights

### Findings

- **Risk Overestimation:** EVT-POT tends to overestimate risk

- **Light-Tailed Distribution:** Negative tail index indicates bounded tails

### Recommendations

- EVT-POT is particularly effective for high confidence levels (99%, 99.5%)
- Monitor tail index (ξ) for stability across portfolios
- Adjust threshold selection based on available data and required exceedances
- Consider EVT-POT for portfolios with heavy-tailed return distributions

## Detailed Metrics

### Summary Statistics by Metric

```
        portfolio_id  confidence_level        horizon  estimation_window  threshold_quantile  var_runtime_ms  evt_fitting_time_ms  threshold_selection_time_ms       hit_rate  violation_ratio  kupiec_unconditional_coverage  kupiec_test_statistic  christoffersen_independence  christoffersen_independence_statistic  christoffersen_conditional_coverage  christoffersen_conditional_coverage_statistic  num_violations  total_observations  expected_violations  mean_exceedance  max_exceedance  std_exceedance  quantile_loss_score  rmse_var_vs_losses  cvar_mean_exceedance  cvar_max_exceedance  cvar_std_exceedance  rmse_cvar_vs_losses  expected_shortfall_exceedance  tail_index_xi    scale_beta  shape_scale_stability  portfolio_size  num_active_assets  hhi_concentration  effective_number_of_assets  covariance_condition_number       skewness       kurtosis  jarque_bera_p_value  jarque_bera_statistic  runtime_per_portfolio_ms  p95_runtime_ms  mean_runtime_ms  median_runtime_ms  min_runtime_ms  max_runtime_ms  cache_hit_ratio
count  200000.000000      2.000000e+05  200000.000000           200000.0        2.000000e+05   200000.000000             200000.0                     200000.0  200000.000000    200000.000000                       200000.0          200000.000000                 2.000000e+05                          200000.000000                             200000.0                                  200000.000000   200000.000000            200000.0         2.000000e+05    200000.000000   200000.000000   200000.000000        200000.000000       200000.000000         200000.000000        200000.000000        200000.000000        200000.000000                            0.0   2.000000e+05  2.000000e+05                    0.0   200000.000000      200000.000000      200000.000000               200000.000000                 2.000000e+05  200000.000000  200000.000000             200000.0          200000.000000              2.000000e+05    2.000000e+05     2.000000e+05      200000.000000        200000.0   200000.000000         200000.0
mean    49999.500000      9.900000e-01       5.500000              500.0        9.500000e-01        1.294393                  0.0                          0.0       0.517932        51.793174                            0.0            8941.586485                 1.792298e-01                               4.081335                                  0.0                                    8945.667820     1279.809325              2471.0         2.471000e+01         0.009455        0.143127        0.009803             0.005259            0.013691              0.008860             0.140509             0.010133             0.013515                            NaN  -4.490237e-01  2.617174e-03                    NaN        5.978000           5.978000           0.338342                    3.682122                 1.188948e+01      -0.812503      14.137199                  0.0           22476.503214              2.588786e+00    1.423436e+01     2.588786e+00           1.000166             0.0     1023.832798              1.0
std     28867.585627      4.440903e-16       4.500011                0.0        5.551129e-16        6.137102                  0.0                          0.0       0.230619        23.061851                            0.0            5365.007203                 2.398141e-01                               3.501885                                  0.0                                    5364.376504      569.858327                 0.0         1.065817e-14         0.002070        0.029488        0.001457             0.003107            0.002090              0.001583             0.029395             0.001668             0.001916                            NaN   1.665339e-16  4.336820e-19                    NaN        2.571577           2.571577           0.176871                    1.583869                 1.776361e-15       0.387078       3.899553                  0.0           11265.393844              4.440903e-16    1.065817e-14     4.440903e-16           0.000000             0.0        0.000000              0.0
min         0.000000      9.900000e-01       1.000000              500.0        9.500000e-01        0.000000                  0.0                          0.0       0.195063        19.506273                            0.0            2040.579923                 8.599843e-08                              -1.981214                                  0.0                                    2045.689533      482.000000              2471.0         2.471000e+01         0.005751        0.068763        0.007229             0.001374            0.009550              0.005850             0.066387             0.007048             0.009831                            NaN  -4.490237e-01  2.617174e-03                    NaN        2.000000           2.000000           0.105833                    1.000105                 1.188948e+01      -1.575624       3.816761                  0.0            1500.048871              2.588786e+00    1.423436e+01     2.588786e+00           1.000166             0.0     1023.832798              1.0
25%     24999.750000      9.900000e-01       1.000000              500.0        9.500000e-01        0.000000                  0.0                          0.0       0.288952        28.895184                            0.0            3640.317574                 1.442516e-02                               1.388411                                  0.0                                    3647.990371      714.000000              2471.0         2.471000e+01         0.007671        0.123354        0.008784             0.002260            0.012272              0.007712             0.120779             0.008871             0.012221                            NaN  -4.490237e-01  2.617174e-03                    NaN        4.000000           4.000000           0.207012                    2.370002                 1.188948e+01      -1.101840      11.394055                  0.0           13574.470181              2.588786e+00    1.423436e+01     2.588786e+00           1.000166             0.0     1023.832798              1.0
50%     49999.500000      9.900000e-01       5.500000              500.0        9.500000e-01        0.000000                  0.0                          0.0       0.502226        50.222582                            0.0            8030.929676                 7.326126e-02                               3.208396                                  0.0                                    8031.955815     1241.000000              2471.0         2.471000e+01         0.009353        0.140630        0.009553             0.005096            0.013328              0.008640             0.138040             0.009962             0.013141                            NaN  -4.490237e-01  2.617174e-03                    NaN        6.000000           6.000000           0.281837                    3.548152                 1.188948e+01      -0.863913      14.346070                  0.0           21491.995291              2.588786e+00    1.423436e+01     2.588786e+00           1.000166             0.0     1023.832798              1.0
75%     74999.250000      9.900000e-01      10.000000              500.0        9.500000e-01        1.013041                  0.0                          0.0       0.746257        74.625658                            0.0           14197.270221                 2.386736e-01                               5.985353                                  0.0                                   14197.533272     1844.000000              2471.0         2.471000e+01         0.010744        0.159597        0.010479             0.007843            0.014683              0.009686             0.156941             0.011024             0.014354                            NaN  -4.490237e-01  2.617174e-03                    NaN        8.000000           8.000000           0.421941                    4.830638                 1.188948e+01      -0.583140      17.118301                  0.0           30624.636536              2.588786e+00    1.423436e+01     2.588786e+00           1.000166             0.0     1023.832798              1.0
max     99999.000000      9.900000e-01      10.000000              500.0        9.500000e-01     1023.832798                  0.0                          0.0       0.887900        88.789964                            0.0           18478.984707                 1.000000e+00                              28.666046                                  0.0                                   18482.037157     2194.000000              2471.0         2.471000e+01         0.019239        0.273625        0.018518             0.016457            0.024840              0.016798             0.270410             0.019463             0.024405                            NaN  -4.490237e-01  2.617174e-03                    NaN       10.000000          10.000000           0.999895                    9.448865                 1.188948e+01       0.657064      23.933537                  0.0           59837.050569              2.588786e+00    1.423436e+01     2.588786e+00           1.000166             0.0     1023.832798              1.0
```
