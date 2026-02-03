# QAE (Quantum Amplitude Estimation) VaR/CVaR Asset-Level Report

**Generated:** 2026-02-03 19:16:38

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

**Confidence 95%:** expected violation rate 0.0500, observed 0.0529
**Confidence 99%:** expected violation rate 0.0100, observed 0.0199

- **Average violation ratio:** 1.5234

- **Green:** 7 configurations
- **Yellow:** 12 configurations
- **Red:** 1 configurations

## Runtime

- **total_runtime_ms:** 33230.35
- **p95_runtime_ms:** 44128.10
- **mean_runtime_ms:** 33230.35
- **runtime_per_asset_ms:** 33230.35
- **returns_compute_time_ms:** 0.00
- **distribution_fit_time_ms:** 0.00
- **state_preparation_time_ms:** 0.00
- **qae_runtime_ms:** 0.00
- **var_search_time_ms:** 201.35
- **cvar_compute_time_ms:** 266.22
- **backtesting_time_ms:** 20.79
- **time_slicing_time_ms:** 0.00

## Key Insights

- Risk underestimation: excessive VaR breaches

## Summary Statistics

- **Asset-config combinations:** 40
- **Assets:** 10

```
       confidence_level  estimation_window   hit_rate  violation_ratio  kupiec_unconditional_coverage  kupiec_test_statistic  christoffersen_independence  christoffersen_independence_statistic  christoffersen_conditional_coverage  christoffersen_conditional_coverage_statistic  num_violations  total_observations  expected_violations  mean_exceedance  max_exceedance  std_exceedance  quantile_loss_score  rmse_var_vs_losses  cvar_mean_exceedance  cvar_max_exceedance  cvar_std_exceedance  rmse_cvar_vs_losses   skewness   kurtosis  jarque_bera_p_value  jarque_bera_statistic
count         40.000000          40.000000  40.000000        40.000000                      38.000000              38.000000                    38.000000                              38.000000                            38.000000                                      38.000000        40.00000           40.000000            40.000000        38.000000       38.000000       33.000000            38.000000           38.000000             31.000000            31.000000            23.000000            31.000000  40.000000  40.000000         4.000000e+01              40.000000
mean           0.970000         376.000000   0.036386         1.523384                       0.492294               1.134761                     0.605185                               0.618894                             0.539219                                       1.753655         3.67500          100.000000             3.000000         0.010987        0.022443        0.009802             0.031745            0.013513              0.008882             0.016701             0.009082             0.010902  -0.051049   2.321553         5.420707e-02              45.813813
std            0.020255         125.579682   0.023171         0.954975                       0.332996               1.375124                     0.256806                               1.205804                             0.296490                                       1.742823         2.44307            6.076436             2.037293         0.006165        0.015085        0.006316             0.010870            0.007880              0.005243             0.013246             0.006321             0.006963   0.589698   2.144139         1.925539e-01              75.507471
min            0.950000         252.000000   0.000000         0.000000                       0.025449               0.003496                     0.010639                               0.000000                             0.037951                                       0.022728         0.00000           94.000000             0.940000         0.000892        0.001438        0.000779             0.018978            0.001048              0.000317             0.000317             0.000443             0.000317  -1.027411   0.147712         3.028299e-60               0.258067
25%            0.950000         252.000000   0.018868         0.943396                       0.124106               0.038200                     0.450860                               0.077675                             0.245879                                       0.588251         2.00000           94.000000             1.030000         0.006051        0.011796        0.004866             0.023623            0.008028              0.004592             0.006607             0.003760             0.005480  -0.383571   1.186507         1.972918e-08               8.772688
50%            0.970000         376.000000   0.031915         1.132075                       0.413764               0.667958                     0.654691                               0.200037                             0.625473                                       0.938495         3.00000          100.000000             2.880000         0.010771        0.019428        0.008131             0.030013            0.012131              0.008254             0.015385             0.008464             0.010227  -0.114418   1.566103         1.714316e-03              12.949162
75%            0.990000         500.000000   0.053191         1.933962                       0.856091               2.365852                     0.780474                               0.568488                             0.745183                                       2.810286         5.00000          106.000000             4.850000         0.014947        0.031489        0.013687             0.036889            0.017484              0.012601             0.022118             0.013385             0.014851   0.449781   2.262245         1.324497e-02              36.835684
max            0.990000         500.000000   0.103774         3.773585                       0.952848               4.993053                     1.000000                               6.524707                             0.988701                                       6.542911        11.00000          106.000000             5.300000         0.024329        0.060819        0.024645             0.068495            0.032364              0.019463             0.048276             0.020539             0.024469   1.149499   8.256009         8.789446e-01             274.094209
```