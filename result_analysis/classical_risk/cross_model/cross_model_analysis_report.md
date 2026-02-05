# Cross-Model VaR/CVaR Asset-Level Analysis Report

**Analysis Task:** cross_model_var_cvar_asset_level_analysis
**Models Compared:** variance_covariance, garch_1_1, monte_carlo, evt_pot

## Experimental Setup

- **Analysis Level:** asset
- **Common Dimensions:** asset, confidence_level, horizon, estimation_window
- **Alignment Policy:** Require common assets, windows, confidence levels, and horizons

## Coverage Accuracy Comparison

| model | metric | mean | median | p95 | expected_hit_rate | abs_deviation |
| --- | --- | --- | --- | --- | --- | --- |
| variance_covariance | hit_rate | 0.0142 | 0.0067 | 0.0431 | 0.0300 | 0.0158 |
| variance_covariance | violation_ratio | 0.5936 | 0.3660 | 1.7577 |  |  |
| garch_1_1 | hit_rate | 0.0078 | 0.0036 | 0.0254 | 0.0300 | 0.0222 |
| garch_1_1 | violation_ratio | 0.2479 | 0.1685 | 0.5479 |  |  |
| monte_carlo | hit_rate | 0.0144 | 0.0043 | 0.0469 | 0.0300 | 0.0156 |
| monte_carlo | violation_ratio | 0.5414 | 0.3435 | 1.5213 |  |  |
| evt_pot | hit_rate | 0.0508 | 0.0340 | 0.1365 | 0.0300 | 0.0208 |
| evt_pot | violation_ratio | 2.7753 | 1.2854 | 10.5878 |  |  |

## Statistical Backtesting Comparison

| model | test | rejection_rate | mean_p_value | significance_level |
| --- | --- | --- | --- | --- |
| variance_covariance | kupiec_unconditional_coverage | 72.5000 |  | 0.0500 |
| variance_covariance | christoffersen_independence | 25.0000 |  | 0.0500 |
| variance_covariance | christoffersen_conditional_coverage | 81.2500 |  | 0.0500 |
| garch_1_1 | kupiec_unconditional_coverage | 100.0000 |  | 0.0500 |
| garch_1_1 | christoffersen_independence | 3.7500 |  | 0.0500 |
| garch_1_1 | christoffersen_conditional_coverage | 90.0000 |  | 0.0500 |
| monte_carlo | kupiec_unconditional_coverage | 55.4167 |  | 0.0500 |
| monte_carlo | christoffersen_independence | 28.7500 |  | 0.0500 |
| monte_carlo | christoffersen_conditional_coverage | 66.6667 |  | 0.0500 |
| evt_pot | kupiec_unconditional_coverage | 96.2500 |  | 0.0500 |
| evt_pot | christoffersen_independence | 56.2500 |  | 0.0500 |
| evt_pot | christoffersen_conditional_coverage | 97.5000 |  | 0.0500 |

## Tail Risk Comparison

| model | risk_type | metric | aggregation | value |
| --- | --- | --- | --- | --- |
| variance_covariance | VaR | mean_exceedance | mean | 0.0235 |
| variance_covariance | VaR | mean_exceedance | median | 0.0188 |
| variance_covariance | VaR | mean_exceedance | p95 | 0.0557 |
| variance_covariance | VaR | max_exceedance | mean | 0.0877 |
| variance_covariance | VaR | max_exceedance | median | 0.0800 |
| variance_covariance | VaR | max_exceedance | p95 | 0.1720 |
| variance_covariance | VaR | rmse_var_vs_losses | mean | 0.0316 |
| variance_covariance | VaR | rmse_var_vs_losses | median | 0.0292 |
| variance_covariance | VaR | rmse_var_vs_losses | p95 | 0.0635 |
| garch_1_1 | VaR | mean_exceedance | mean | 0.0198 |
| garch_1_1 | VaR | mean_exceedance | median | 0.0156 |
| garch_1_1 | VaR | mean_exceedance | p95 | 0.0375 |
| garch_1_1 | VaR | max_exceedance | mean | 0.0641 |
| garch_1_1 | VaR | max_exceedance | median | 0.0558 |
| garch_1_1 | VaR | max_exceedance | p95 | 0.1264 |
| garch_1_1 | VaR | rmse_var_vs_losses | mean | 0.0655 |
| garch_1_1 | VaR | rmse_var_vs_losses | median | 0.0596 |
| garch_1_1 | VaR | rmse_var_vs_losses | p95 | 0.1030 |
| monte_carlo | VaR | mean_exceedance | mean | 0.0219 |
| monte_carlo | VaR | mean_exceedance | median | 0.0185 |
| monte_carlo | VaR | mean_exceedance | p95 | 0.0522 |
| monte_carlo | VaR | max_exceedance | mean | 0.0894 |
| monte_carlo | VaR | max_exceedance | median | 0.0890 |
| monte_carlo | VaR | max_exceedance | p95 | 0.1753 |
| monte_carlo | VaR | rmse_var_vs_losses | mean | 0.0308 |
| monte_carlo | VaR | rmse_var_vs_losses | median | 0.0285 |
| monte_carlo | VaR | rmse_var_vs_losses | p95 | 0.0594 |
| evt_pot | VaR | mean_exceedance | mean | 0.0190 |
| evt_pot | VaR | mean_exceedance | median | 0.0146 |
| evt_pot | VaR | mean_exceedance | p95 | 0.0346 |
| evt_pot | VaR | max_exceedance | mean | 0.1224 |
| evt_pot | VaR | max_exceedance | median | 0.1212 |
| evt_pot | VaR | max_exceedance | p95 | 0.2109 |
| evt_pot | VaR | rmse_var_vs_losses | mean | 0.0297 |
| evt_pot | VaR | rmse_var_vs_losses | median | 0.0236 |
| evt_pot | VaR | rmse_var_vs_losses | p95 | 0.0533 |
| garch_1_1 | CVaR | cvar_mean_exceedance | mean | 0.0794 |
| garch_1_1 | CVaR | rmse_cvar_vs_losses | mean | 0.0863 |
| monte_carlo | CVaR | cvar_mean_exceedance | mean | 0.0224 |
| evt_pot | CVaR | cvar_mean_exceedance | mean | 0.0218 |
| evt_pot | CVaR | rmse_cvar_vs_losses | mean | 0.0325 |

## Distributional Assumption Effects

| model | metric | mean | median | std |
| --- | --- | --- | --- | --- |
| variance_covariance | skewness | -0.2719 | -0.3560 | 0.4315 |
| variance_covariance | kurtosis | 9.3860 | 9.8297 | 2.5054 |
| variance_covariance | jarque_bera_p_value | 0.0000 | 0.0000 | 0.0000 |
| garch_1_1 | skewness | -0.2718 | -0.3557 | 0.4317 |
| garch_1_1 | kurtosis | 9.3872 | 9.8239 | 2.5047 |
| garch_1_1 | jarque_bera_p_value | 0.0000 | 0.0000 | 0.0000 |
| monte_carlo | skewness | -0.2719 | -0.3560 | 0.4297 |
| monte_carlo | kurtosis | 9.3860 | 9.8297 | 2.4949 |
| monte_carlo | jarque_bera_p_value | 0.0000 | 0.0000 | 0.0000 |
| evt_pot | skewness | -0.2719 | -0.3560 | 0.4315 |
| evt_pot | kurtosis | 9.3860 | 9.8297 | 2.5054 |
| evt_pot | jarque_bera_p_value | 0.0000 | 0.0000 | 0.0000 |

## Regime-Dependent Performance (Time-Sliced Analysis)

| slice_dimension | slice_value | model | metric | mean | count |
| --- | --- | --- | --- | --- | --- |
| year | 2016 | variance_covariance | hit_rate | 0.0047 | 120 |
| year | 2016 | variance_covariance | violation_ratio | 0.2187 | 120 |
| year | 2016 | monte_carlo | hit_rate | 0.0047 | 360 |
| year | 2016 | monte_carlo | violation_ratio | 0.1771 | 360 |
| year | 2016 | evt_pot | hit_rate | 0.0391 | 120 |
| year | 2016 | evt_pot | violation_ratio | 1.9062 | 120 |
| year | 2017 | variance_covariance | hit_rate | 0.0065 | 800 |
| year | 2017 | variance_covariance | violation_ratio | 0.2208 | 800 |
| year | 2017 | garch_1_1 | hit_rate | 0.0031 | 200 |
| year | 2017 | garch_1_1 | violation_ratio | 0.0784 | 200 |
| year | 2017 | monte_carlo | hit_rate | 0.0070 | 2400 |
| year | 2017 | monte_carlo | violation_ratio | 0.2184 | 2400 |
| year | 2017 | evt_pot | hit_rate | 0.0427 | 800 |
| year | 2017 | evt_pot | violation_ratio | 2.3005 | 800 |
| year | 2018 | variance_covariance | hit_rate | 0.0194 | 1360 |
| year | 2018 | variance_covariance | violation_ratio | 0.7815 | 1360 |
| year | 2018 | garch_1_1 | hit_rate | 0.0080 | 400 |
| year | 2018 | garch_1_1 | violation_ratio | 0.2338 | 400 |
| year | 2018 | monte_carlo | hit_rate | 0.0199 | 4080 |
| year | 2018 | monte_carlo | violation_ratio | 0.7179 | 4080 |
| year | 2018 | evt_pot | hit_rate | 0.0646 | 1360 |
| year | 2018 | evt_pot | violation_ratio | 3.5439 | 1360 |
| year | 2019 | variance_covariance | hit_rate | 0.0128 | 1360 |
| year | 2019 | variance_covariance | violation_ratio | 0.4750 | 1360 |
| year | 2019 | garch_1_1 | hit_rate | 0.0065 | 400 |
| year | 2019 | garch_1_1 | violation_ratio | 0.1838 | 400 |
| year | 2019 | monte_carlo | hit_rate | 0.0131 | 4080 |
| year | 2019 | monte_carlo | violation_ratio | 0.4505 | 4080 |
| year | 2019 | evt_pot | hit_rate | 0.0473 | 1360 |
| year | 2019 | evt_pot | violation_ratio | 2.5605 | 1360 |
| year | 2020 | variance_covariance | hit_rate | 0.0275 | 1360 |
| year | 2020 | variance_covariance | violation_ratio | 1.3414 | 1360 |
| year | 2020 | garch_1_1 | hit_rate | 0.0146 | 320 |
| year | 2020 | garch_1_1 | violation_ratio | 0.5420 | 320 |
| year | 2020 | monte_carlo | hit_rate | 0.0274 | 4080 |
| year | 2020 | monte_carlo | violation_ratio | 1.2056 | 4080 |
| year | 2020 | evt_pot | hit_rate | 0.0800 | 1360 |
| year | 2020 | evt_pot | violation_ratio | 4.4794 | 1360 |
| year | 2021 | variance_covariance | hit_rate | 0.0062 | 1360 |
| year | 2021 | variance_covariance | violation_ratio | 0.2070 | 1360 |
| year | 2021 | garch_1_1 | hit_rate | 0.0049 | 400 |
| year | 2021 | garch_1_1 | violation_ratio | 0.1227 | 400 |
| year | 2021 | monte_carlo | hit_rate | 0.0066 | 4080 |
| year | 2021 | monte_carlo | violation_ratio | 0.1890 | 4080 |
| year | 2021 | evt_pot | hit_rate | 0.0351 | 1360 |
| year | 2021 | evt_pot | violation_ratio | 1.8991 | 1360 |
| year | 2022 | variance_covariance | hit_rate | 0.0158 | 1360 |
| year | 2022 | variance_covariance | violation_ratio | 0.6209 | 1360 |
| year | 2022 | garch_1_1 | hit_rate | 0.0081 | 400 |
| year | 2022 | garch_1_1 | violation_ratio | 0.2339 | 400 |

## Robustness Analysis

### Window Sensitivity

| model | estimation_window | metric | mean |
| --- | --- | --- | --- |
| variance_covariance | 252 | hit_rate | 0.0140 |
| variance_covariance | 252 | violation_ratio | 0.5840 |
| variance_covariance | 500 | hit_rate | 0.0144 |
| variance_covariance | 500 | violation_ratio | 0.6032 |
| garch_1_1 | 252 | hit_rate | 0.0076 |
| garch_1_1 | 252 | violation_ratio | 0.2434 |
| garch_1_1 | 500 | hit_rate | 0.0080 |
| garch_1_1 | 500 | violation_ratio | 0.2524 |
| monte_carlo | 252 | hit_rate | 0.0140 |
| monte_carlo | 252 | violation_ratio | 0.5244 |
| monte_carlo | 500 | hit_rate | 0.0148 |
| monte_carlo | 500 | violation_ratio | 0.5584 |
| evt_pot | 252 | hit_rate | 0.0645 |
| evt_pot | 252 | violation_ratio | 3.5338 |
| evt_pot | 500 | hit_rate | 0.0370 |
| evt_pot | 500 | violation_ratio | 2.0167 |

### Horizon Scaling

| model | horizon | metric | mean |
| --- | --- | --- | --- |
| variance_covariance | 1 | violation_ratio | 1.1539 |
| variance_covariance | 1 | rmse_var_vs_losses | 0.0269 |
| variance_covariance | 10 | violation_ratio | 0.0333 |
| variance_covariance | 10 | rmse_var_vs_losses | 0.0371 |
| garch_1_1 | 1 | violation_ratio | 0.4317 |
| garch_1_1 | 1 | rmse_var_vs_losses | 0.0495 |
| monte_carlo | 1 | violation_ratio | 1.0513 |
| monte_carlo | 1 | rmse_var_vs_losses | 0.0274 |
| monte_carlo | 10 | violation_ratio | 0.0315 |
| monte_carlo | 10 | rmse_var_vs_losses | 0.0351 |
| evt_pot | 1 | violation_ratio | 5.2526 |
| evt_pot | 1 | rmse_var_vs_losses | 0.0205 |
| evt_pot | 10 | violation_ratio | 0.2979 |
| evt_pot | 10 | rmse_var_vs_losses | 0.0390 |

## Overall Model Ranking

| model | confidence_level | horizon | composite_score | component_abs(hit_rate - expected_hit_rate) | component_violation_ratio | component_rmse_var_vs_losses | component_kupiec_rejection_rate | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| monte_carlo | 0.9500 | 1 | 0.0000 | 0.0592 | 0.0000 | 0.0442 | 0.0952 | 1 |
| monte_carlo | 0.9900 | 1 | 0.0200 | 0.0000 | 0.0245 | 0.1408 | 0.0000 | 2 |
| variance_covariance | 0.9500 | 1 | 0.0226 | 0.0970 | 0.0074 | 0.0475 | 0.5000 | 3 |
| variance_covariance | 0.9900 | 1 | 0.0579 | 0.0263 | 0.0502 | 0.1245 | 0.3571 | 4 |
| evt_pot | 0.9900 | 10 | 0.0819 | 0.0335 | 0.0572 | 0.2506 | 0.7857 | 5 |
| garch_1_1 | 0.9500 | 1 | 0.0913 | 0.3369 | 0.0542 | 0.2323 | 1.0000 | 6 |
| garch_1_1 | 0.9900 | 1 | 0.0979 | 0.0392 | 0.0627 | 0.4969 | 1.0000 | 7 |
| monte_carlo | 0.9900 | 10 | 0.1197 | 0.0918 | 0.1141 | 0.1237 | 0.3571 | 8 |
| variance_covariance | 0.9900 | 10 | 0.1266 | 0.0905 | 0.1128 | 0.1426 | 0.5714 | 9 |
| evt_pot | 0.9500 | 10 | 0.1377 | 0.5685 | 0.0994 | 0.2204 | 1.0000 | 10 |
| garch_1_1 | 0.9900 | 3 | 0.1495 | 0.0882 | 0.1105 | 1.0000 | 1.0000 | 11 |
| garch_1_1 | 0.9500 | 3 | 0.1505 | 0.6161 | 0.1087 | 0.5221 | 1.0000 | 12 |
| monte_carlo | 0.9500 | 10 | 0.1529 | 0.6443 | 0.1142 | 0.2223 | 1.0000 | 13 |
| variance_covariance | 0.9500 | 10 | 0.1542 | 0.6484 | 0.1149 | 0.2612 | 1.0000 | 14 |
| evt_pot | 0.9500 | 1 | 0.1787 | 0.7846 | 0.1415 | 0.0000 | 1.0000 | 15 |
| evt_pot | 0.9900 | 1 | 1.0000 | 1.0000 | 1.0000 | 0.0147 | 1.0000 | 16 |

## Discussion and Implications

This cross-model analysis compares multiple VaR/CVaR estimation methods 
across common dimensions. Key findings include:

- Coverage accuracy varies across models
- Statistical backtesting results show different rejection rates
- Tail risk behavior differs significantly between models
- Distributional assumptions impact model performance
- Model rankings depend on the specific metric and context
