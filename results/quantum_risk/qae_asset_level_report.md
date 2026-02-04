# QAE (Quantum Amplitude Estimation) VaR/CVaR Asset-Level Report

**Generated:** 2026-02-04 00:47:26

## Quantum Methodology Overview

Quantum Amplitude Estimation (QAE) is applied to individual asset loss distributions.
VaR is estimated via CDF bisection using QAE; CVaR via Rockafellar-Uryasev with tail expectation.

## Distribution Encoding and State Preparation

- **Distribution family:** student_t
- **State qubits:** 6
- Discretization: uniform grid in loss space, rescaled to unit interval

## QAE VaR and CVaR Construction

- **VaR:** Bisection on CDF(L <= x) to find x such that CDF(x) ≈ 1-α
- **CVaR:** VaR + (1/(1-α)) × E[(L - VaR)^+]

## Backtesting Results

**Confidence 95%:** expected violation rate 0.0500, observed 0.0492
**Confidence 99%:** expected violation rate 0.0100, observed 0.0157

- **Average violation ratio:** 1.2787

- **Green:** 12 configurations
- **Yellow:** 8 configurations

## Runtime

- **total_runtime_ms:** 2274754.89
- **p95_runtime_ms:** 8475915.67
- **mean_runtime_ms:** 2274754.89
- **runtime_per_asset_ms:** 2274754.89
- **returns_compute_time_ms:** 0.00
- **distribution_fit_time_ms:** 0.00
- **state_preparation_time_ms:** 0.00
- **qae_runtime_ms:** 0.00
- **var_search_time_ms:** 4535.23
- **cvar_compute_time_ms:** 440077.24
- **backtesting_time_ms:** 27.81
- **time_slicing_time_ms:** 0.00

## Key Insights

- Risk underestimation: excessive VaR breaches

## Summary Statistics

- **Asset-config combinations:** 40
- **Assets:** 10

```
       confidence_level  estimation_window   hit_rate  violation_ratio  kupiec_unconditional_coverage  kupiec_test_statistic  christoffersen_independence  christoffersen_independence_statistic  christoffersen_conditional_coverage  christoffersen_conditional_coverage_statistic  num_violations  total_observations  expected_violations  mean_exceedance  max_exceedance  std_exceedance  quantile_loss_score  rmse_var_vs_losses  cvar_mean_exceedance  cvar_max_exceedance  cvar_std_exceedance  rmse_cvar_vs_losses   skewness   kurtosis  jarque_bera_p_value  jarque_bera_statistic
count         40.000000          40.000000  40.000000        40.000000                      40.000000              40.000000                 4.000000e+01                              40.000000                         4.000000e+01                                      40.000000        40.00000           40.000000            40.000000        40.000000       40.000000       40.000000            40.000000           40.000000             40.000000            40.000000            40.000000            40.000000  40.000000  40.000000         4.000000e+01              40.000000
mean           0.970000         376.000000   0.032476         1.278662                       0.289873               3.575104                 9.677959e-02                               6.428521                         6.930264e-02                                      10.003626        68.05000         2096.000000            62.880000         0.015256        0.123772        0.021300             0.031495            0.026050              0.162301             1.045302             0.316798             0.354939  -0.271897   9.385981        1.593101e-308            8325.163937
std            0.020255         125.579682   0.017244         0.328543                       0.318544               3.880411                 1.466909e-01                               6.716989                         1.295701e-01                                       6.897070        36.19247          125.579682            42.694805         0.004126        0.041592        0.006174             0.010137            0.007172              0.297266             3.006322             0.900006             0.904396   0.434264   2.521431         0.000000e+00            3901.760214
min            0.950000         252.000000   0.012677         0.837838                       0.000170               0.003851                 2.433215e-08                               0.359974                         1.108538e-07                                       0.711426        25.00000         1972.000000            19.720000         0.009055        0.054603        0.011355             0.019412            0.014484              0.013653             0.085709             0.018419             0.022763  -0.931832   4.078599         0.000000e+00            1411.469588
25%            0.950000         252.000000   0.015766         0.981073                       0.011787               0.401910                 4.664563e-03                               2.451345                         1.789307e-03                                       4.979546        34.75000         1972.000000            21.580000         0.012136        0.097155        0.017392             0.023784            0.020758              0.021625             0.132523             0.031458             0.037157  -0.554171   7.709968         0.000000e+00            5321.639316
50%            0.970000         376.000000   0.030405         1.191684                       0.169646               1.887055                 2.795861e-02                               4.835015                         1.277717e-02                                       8.720190        64.00000         2096.000000            60.400000         0.014165        0.128787        0.021291             0.031062            0.025984              0.036877             0.190306             0.050523             0.062751  -0.355953   9.829733         0.000000e+00            8340.755120
75%            0.990000         500.000000   0.048784         1.576577                       0.530685               6.342597                 1.174726e-01                               8.034537                         8.292922e-02                                      12.652641       102.25000         2220.000000           101.700000         0.017558        0.145178        0.025579             0.036623            0.031059              0.205325             0.871993             0.350071             0.470625  -0.005238  11.172227         0.000000e+00           11318.195192
max            0.990000         500.000000   0.055781         1.891892                       0.950520              14.135412                 5.485206e-01                              31.113910                         7.006735e-01                                      32.030107       120.00000         2220.000000           111.000000         0.024833        0.231874        0.034831             0.065023            0.041784              1.735939            18.991174             5.722928             5.726076   0.709688  12.972163        3.186202e-307           14226.861044
```