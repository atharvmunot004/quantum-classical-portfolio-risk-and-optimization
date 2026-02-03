# EVT-POT asset-level analysis report

Config: `llm.json`. Total configurations: 80.

## Coverage accuracy results

- Hit rate (mean): 0.050785318787347176
- Violation ratio (mean): 2.7752590318513213
- Within tolerance (%): 67.5

## Statistical backtesting
- kupiec_unconditional_coverage_rejection_rate: 96.25%
- christoffersen_independence_rejection_rate: 56.25%
- christoffersen_conditional_coverage_rejection_rate: 97.5%

## Tail risk characteristics
- mean_exceedance_mean: 0.018997834874846554
- mean_exceedance_median: 0.014555365600955869
- mean_exceedance_p95: 0.03464955975045022
- max_exceedance_mean: 0.12240273606444099
- max_exceedance_median: 0.12124052398517593
- max_exceedance_p95: 0.21093422233613363
- std_exceedance_mean: 0.02359502659733275
- std_exceedance_median: 0.01936798982689889
- std_exceedance_p95: 0.0420148804913129
- quantile_loss_score_mean: 0.03713752641357969
- quantile_loss_score_median: 0.033730627400432665
- quantile_loss_score_p95: 0.07029542411333843
- rmse_var_vs_losses_mean: 0.029743530632166115
- rmse_var_vs_losses_median: 0.02356445581261428
- rmse_var_vs_losses_p95: 0.05328955924810871
- cvar_mean_exceedance_mean: 0.02181733096479193
- cvar_mean_exceedance_median: 0.015557247239868132
- cvar_mean_exceedance_p95: 0.04044040394018313
- cvar_max_exceedance_mean: 0.114812288056245
- cvar_max_exceedance_median: 0.11525118622727226
- cvar_max_exceedance_p95: 0.19294202210998812
- cvar_std_exceedance_mean: 0.024898571305898495
- cvar_std_exceedance_median: 0.02077928307849928
- cvar_std_exceedance_p95: 0.04415310894025231
- rmse_cvar_vs_losses_mean: 0.03251414430741147
- rmse_cvar_vs_losses_median: 0.025000632343825983
- rmse_cvar_vs_losses_p95: 0.0565111229852786

## EVT parameter stability
- tail_index_xi_mean: -0.4078381669157453
- tail_index_xi_std: 0.07079230893864538
- tail_index_xi_cv: -0.17357941134839952
- scale_beta_mean: 0.002532635804399317
- scale_beta_std: 0.0005013464925105177
- scale_beta_cv: 0.19795443610157185
- threshold_mean: 0.01179717942842783
- threshold_std: 0.0034468396040095487
- threshold_cv: 0.29217489018634835
- num_exceedances_mean: 50.5
- num_exceedances_std: 0.5031546054266276
- num_exceedances_cv: 0.009963457533200546
- xi_violations_count: 0
- xi_violations_pct: 0.0

## Regime sensitivity
See time-sliced analysis table.

## Robustness checks
- threshold_sensitivity: (see summary table)
- window_sensitivity: (see summary table)

## Computational performance
No runtime metrics in asset-level metrics.

## Discussion

Analysis performed per llm.json: coverage accuracy, backtests, tail risk, EVT diagnostics, 
time-sliced and robustness checks, and ranking. See summary and time-sliced parquet outputs.
