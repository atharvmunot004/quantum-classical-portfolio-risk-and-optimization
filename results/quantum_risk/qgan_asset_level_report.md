# QGAN Scenario Generation Asset-Level Risk Report

**Generated:** 2026-02-04 08:43:05

## Quantum Methodology Overview

Quantum Generative Adversarial Networks (QGAN) are trained per asset
to learn the empirical distribution of returns/losses within rolling windows
and to generate realistic synthetic scenarios. Generated scenarios are used
for asset-level risk assessment (VaR/CVaR estimation and tail diagnostics).

## Data Preprocessing and Discretization

- **Number of qubits:** 6
- **Discretization bins:** 64
- **Standardization:** robust_zscore

## QGAN Architecture and Training

- **Generator:** Hardware-efficient ansatz with 3 layers
- **Discriminator:** MLP with hidden layers [64, 32]
- **Max iterations:** 50
- **Batch size:** 256

## Scenario Generation Protocol

- **Scenarios per timestamp:** 10000
- **Horizons:** [10]

## Distribution and Tail Fidelity Results

- **Mean Wasserstein distance:** 0.018385
- **Mean KS statistic:** 0.362875
- **Mean VaR error (95%):** 3.718566
- **Mean CVaR error (95%):** 4.002872

## Stylized Facts Validation

- **Volatility clustering proxy:** 0.094836
- **Leptokurtosis gap:** 5.221234

## Runtime and Scalability

- **total_runtime_ms:** 190338.46
- **preprocessing_time_ms:** 43.88
- **qgan_training_time_ms:** 19091.21
- **scenario_generation_time_ms:** 719.61
- **risk_compute_time_ms:** 85.98
- **evaluation_time_ms:** 638.21
- **runtime_per_asset_ms:** 19033.85
- **mean_training_time_ms:** 15996.59
- **p95_training_time_ms:** 19091.21
- **cache_hit_ratio:** 0.00

## Key Insights

- Excellent distribution fidelity: low Wasserstein distance
- Good diversity: low mode collapse

## Summary Statistics

- **Asset-config combinations:** 450
- **Assets:** 10

```
       estimation_window  wasserstein_distance  ks_statistic  js_divergence  moment_error_mean  moment_error_var  moment_error_skew  moment_error_kurt  var_error_95  cvar_error_95  tail_mass_error_95  extreme_quantile_error_95  var_error_99  cvar_error_99  tail_mass_error_99  extreme_quantile_error_99  acf_squared_returns_lag1_5  leptokurtosis_gap  downside_skew_preservation  volatility_clustering_proxy  mode_collapse_score  generator_circuit_depth  generator_circuit_width  num_qubits   shots  generator_loss_final  generator_loss_mean  generator_loss_std  generator_loss_min  generator_loss_max  discriminator_loss_final  discriminator_loss_mean  discriminator_loss_std  discriminator_loss_min  discriminator_loss_max
count              450.0            450.000000    450.000000     450.000000       4.500000e+02        450.000000         450.000000         450.000000    450.000000     450.000000          450.000000                 450.000000    450.000000     450.000000               450.0                 450.000000                  450.000000         450.000000                  450.000000                   450.000000                450.0                    450.0                    450.0       450.0   450.0            450.000000           450.000000          450.000000          450.000000          450.000000                450.000000               450.000000              450.000000              450.000000              450.000000
mean               252.0              0.018385      0.362875       0.482726       8.593459e-03          0.000846           0.609307           5.221234      3.718566       4.002872            0.007937                   4.235311      4.187244       4.245605                 0.0                   4.235311                    0.094836           5.221234                    0.609307                     0.094836                  1.0                      6.0                      6.0         6.0  4096.0              0.681906             0.682507            0.000809            0.680953            0.684545                  0.697221                 0.696907                0.000419                0.695849                0.697712
std                  0.0              0.007629      0.063812       0.035869       6.686980e-03          0.000656           0.595803           4.289337      0.983516       1.022383            0.000000                   1.041566      1.047962       1.049319                 0.0                   1.041566                    0.055603           4.289337                    0.595803                     0.055603                  0.0                      0.0                      0.0         0.0     0.0              0.004980             0.004860            0.000173            0.005205            0.004564                  0.002471                 0.002405                0.000094                0.002242                0.002590
min                252.0              0.006468      0.222222       0.361918       8.173882e-07          0.000148           0.000654           1.113030      1.796164       1.943604            0.007937                   2.147780      2.087925       2.148515                 0.0                   2.147780                    0.020596           1.113030                    0.000654                     0.020596                  1.0                      6.0                      6.0         6.0  4096.0              0.671310             0.672095            0.000450            0.670033            0.674519                  0.692036                 0.691967                0.000229                0.691288                0.692425
25%                252.0              0.012795      0.309524       0.456486       3.305841e-03          0.000410           0.175378           2.459240      2.859501       3.101706            0.007937                   3.308978      3.260305       3.315627                 0.0                   3.308978                    0.059080           2.459240                    0.175378                     0.059080                  1.0                      6.0                      6.0         6.0  4096.0              0.677505             0.678254            0.000661            0.676139            0.680745                  0.695077                 0.694827                0.000339                0.693972                0.695449
50%                252.0              0.016802      0.365079       0.485910       7.472939e-03          0.000661           0.416529           3.754861      3.621604       3.908230            0.007937                   4.153185      4.103682       4.162312                 0.0                   4.153185                    0.078951           3.754861                    0.416529                     0.078951                  1.0                      6.0                      6.0         6.0  4096.0              0.682054             0.682621            0.000792            0.681015            0.684427                  0.697159                 0.696857                0.000409                0.695829                0.697660
75%                252.0              0.022420      0.412698       0.507468       1.310558e-02          0.000993           0.895584           6.667912      4.665389       5.006504            0.007937                   5.270934      5.227506       5.284037                 0.0                   5.270934                    0.111883           6.667912                    0.895584                     0.111883                  1.0                      6.0                      6.0         6.0  4096.0              0.686032             0.686511            0.000971            0.685324            0.688307                  0.699315                 0.698941                0.000509                0.697701                0.699960
max                252.0              0.053701      0.523810       0.557718       4.921729e-02          0.004762           3.856291          36.943814      5.319182       5.613497            0.007937                   5.840146      5.818020       5.858113                 0.0                   5.840146                    0.336136          36.943814                    3.856291                     0.336136                  1.0                      6.0                      6.0         6.0  4096.0              0.692957             0.693105            0.001102            0.692233            0.694402                  0.702513                 0.702119                0.000577                0.700940                0.703186
```