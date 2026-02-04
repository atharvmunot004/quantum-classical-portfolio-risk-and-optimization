# qPCA Factor Risk Asset-Level Report

**Generated:** 2026-02-04 08:55:05

## Quantum Methodology Overview

Quantum PCA (qPCA) extracts dominant latent factors from rolling-window return
covariance via density-matrix eigenstructure. Factors and explained variance are
computed per window; asset-level factor exposures and factor-implied risk proxies
(VaR/CVaR) support comparison with classical PCA-based factor risk.

## Density Matrix Construction and Shrinkage

Cross-asset covariance is estimated with Ledoit-Wolf shrinkage per rolling window,
then trace-normalized to a density matrix for qPCA. This ensures PSD and unit trace.

## qPCA Eigenspectrum Estimation

Eigenvalues and eigenvectors are obtained by classical eigensolver (simulating
quantum phase estimation). Circuit depth/width and precision bits are reported
for hardware feasibility discussion.

## Factor Exposure Construction

Per-asset factor exposures are the loadings from projecting each asset onto the
top-k qPCA factors (eigenvectors) in each window.

## Factor Risk Proxy Methodology

Factor-implied VaR and CVaR use a Gaussian factor model: total variance =
factor-explained variance + idiosyncratic variance; tail quantiles from normal.

## Factor Stability and Regime Shifts

- **Mean cumulative explained variance (top factors):** 0.7478

## Classical PCA Alignment Results

- **principal_angle_distance_vs_classical:** mean = 0.0000
- **explained_variance_gap:** mean = 0.0000
- **exposure_correlation:** mean = 1.0000

## Runtime and State Preparation Overhead

- **total_runtime_ms:** 65789.43
- **returns_compute_time_ms:** 0.00
- **covariance_build_time_ms:** 1470.94
- **density_matrix_build_time_ms:** 1511.41
- **state_preparation_time_ms:** 0.00
- **qpca_runtime_ms:** 487.11
- **projection_time_ms:** 16.69
- **factor_risk_compute_time_ms:** 28722.59
- **time_slicing_time_ms:** 1878.40
- **cache_hit_ratio:** 0.00

## Key Insights

- High alignment between qPCA and classical PCA factor exposures.

## Summary Statistics

- **Metric rows:** 60
- **Assets:** 10
```
       estimation_window      top_k  explained_variance_ratio_k  cumulative_explained_variance_k  factor_stability_cosine_similarity  principal_angle_distance_vs_classical  explained_variance_gap  exposure_correlation  factor_var_95  factor_var_99  factor_cvar_95  factor_cvar_99  idiosyncratic_variance  precision_bits   shots  qpca_circuit_depth  qpca_circuit_width
count          60.000000  60.000000                   60.000000                     6.000000e+01                           60.000000                           6.000000e+01                    60.0                  60.0   6.000000e+01   6.000000e+01       60.000000    6.000000e+01               60.000000            60.0    60.0                60.0                60.0
mean          376.000000   6.000000                    0.105963                     1.000000e+00                            0.993724                           2.288432e-08                     0.0                   1.0   1.644854e+00   2.326348e+00        2.062713    2.665214e+00                0.921879             6.0  4000.0                96.0                10.0
std           125.046432   2.968764                    0.065919                     2.868577e-13                            0.006508                           5.389779e-09                     0.0                   0.0   8.672324e-17   1.734465e-16        0.000000    4.128852e-16                0.019795             0.0     0.0                 0.0                 0.0
min           252.000000   3.000000                    0.029637                     1.000000e+00                            0.983691                           1.411763e-08                     0.0                   1.0   1.644854e+00   2.326348e+00        2.062713    2.665214e+00                0.900000             6.0  4000.0                96.0                10.0
25%           252.000000   3.000000                    0.030860                     1.000000e+00                            0.987915                           1.866906e-08                     0.0                   1.0   1.644854e+00   2.326348e+00        2.062713    2.665214e+00                0.900000             6.0  4000.0                96.0                10.0
50%           376.000000   5.000000                    0.097891                     1.000000e+00                            0.995544                           2.362804e-08                     0.0                   1.0   1.644854e+00   2.326348e+00        2.062713    2.665214e+00                0.918609             6.0  4000.0                96.0                10.0
75%           500.000000  10.000000                    0.189454                     1.000000e+00                            0.999765                           2.673433e-08                     0.0                   1.0   1.644854e+00   2.326348e+00        2.062713    2.665214e+00                0.936936             6.0  4000.0                96.0                10.0
max           500.000000  10.000000                    0.190046                     1.000000e+00                            0.999888                           3.052883e-08                     0.0                   1.0   1.644854e+00   2.326348e+00        2.062713    2.665214e+00                0.966365             6.0  4000.0                96.0                10.0
```