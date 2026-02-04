# QAOA CVaR Asset-Level Risk Report

**Generated:** 2026-02-03 19:35:09

## Quantum Methodology Overview

QAOA is used as a quantum optimization routine for asset-level tail-risk scoring.
A discretized tail-loss objective is encoded as a cost Hamiltonian and optimized
using CVaR-based sampling of measurement outcomes (mean of worst alpha fraction).

## QAOA Cost and Mixer Construction

- **Cost Hamiltonian:** Ising form, built from discretized loss objective (quantile grid)
- **State qubits:** 8
- **Mixer:** Standard transverse-field X mixer for unconstrained binary search

## CVaR-Based Sampling and Objective

Objective = mean of worst (1-alpha) fraction of sampled loss outcomes from QAOA measurements.
This CVaR objective is evaluated during optimization; the final CVaR estimate
is the tail-average of the optimized state's measurement distribution.

## Backtesting and Tail Diagnostics

- **Mean exceedance rate vs CVaR:** 0.0119
- **Total exceedances:** 3000
- **Green zone:** 60 configurations

## Runtime

- **total_runtime_ms:** 1143.57
- **runtime_per_asset_ms:** 1143.57
- **mean_qaoa_optimize_time_ms:** 1143.57
- **p95_qaoa_optimize_time_ms:** 1256.28
- **returns_compute_time_ms:** 0.00
- **loss_discretization_time_ms:** 0.00
- **hamiltonian_build_time_ms:** 0.00
- **qaoa_optimize_time_ms:** 0.00
- **cvar_eval_time_ms:** 0.00
- **backtesting_time_ms:** 35.40
- **time_slicing_time_ms:** 0.00
- **cache_hit_ratio:** 0.00

## Key Insights

- Risk overestimation

## Summary Statistics

- **Asset-config combinations:** 120
- **Assets:** 10

```
       confidence_level  estimation_window       reps    hit_rate  violation_ratio  num_exceedances_vs_cvar  exceedance_rate_vs_cvar  total_observations  expected_violations  mean_exceedance_given_cvar  max_exceedance_given_cvar  tail_mean_loss  tail_max_loss  rmse_cvar_vs_losses    skewness    kurtosis  jarque_bera_statistic  jarque_bera_p_value
count        120.000000         120.000000  120.00000  120.000000       120.000000               120.000000               120.000000          120.000000           120.000000                  120.000000                 120.000000      120.000000     120.000000           120.000000  120.000000  120.000000             120.000000         1.200000e+02
mean           0.970000         376.000000    2.00000    0.011911         0.441896                25.000000                 0.011911         2096.000000            62.880000                    0.017253                   0.092196        0.061436       0.160161             0.027441   -0.271897    9.385981            8325.163937        1.593101e-308
std            0.020084         124.519918    0.81992    0.007021         0.104762                14.889509                 0.007021          124.519918            42.334505                    0.005548                   0.040087        0.016881       0.047311             0.008454    0.430599    2.500153            3868.833379         0.000000e+00
min            0.950000         252.000000    1.00000    0.003043         0.297297                 6.000000                 0.003043         1972.000000            19.720000                    0.006416                   0.025272        0.036416       0.092891             0.010860   -0.931832    4.078599            1411.469588         0.000000e+00
25%            0.950000         252.000000    1.00000    0.004564         0.365112                10.000000                 0.004564         1972.000000            21.580000                    0.014101                   0.060322        0.047736       0.134754             0.022705   -0.554171    7.709968            5321.639316         0.000000e+00
50%            0.970000         376.000000    2.00000    0.011261         0.419622                24.000000                 0.011261         2096.000000            60.400000                    0.015711                   0.089763        0.057599       0.159513             0.027330   -0.355953    9.829733            8340.755120         0.000000e+00
75%            0.990000         500.000000    3.00000    0.018309         0.456389                39.250000                 0.018309         2220.000000           101.700000                    0.021471                   0.118616        0.072947       0.177522             0.032968   -0.005238   11.172227           11318.195192         0.000000e+00
max            0.990000         500.000000    3.00000    0.021805         0.765766                47.000000                 0.021805         2220.000000           111.000000                    0.031432                   0.196472        0.102967       0.264367             0.047959    0.709688   12.972163           14226.861044        3.186202e-307
```