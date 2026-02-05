# Cross Model Rankings

**Source:** `cross_model_rankings.parquet`
**Rows:** 16
**Columns:** 9

## Data Table

| model | confidence_level | horizon | composite_score | component_abs(hit_rate - expected_hit_rate) | component_violation_ratio | component_rmse_var_vs_losses | component_kupiec_rejection_rate | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| monte_carlo | 0.950000 | 1 | 0.000000 | 0.059189 | 0.000000 | 0.044167 | 0.095238 | 1 |
| monte_carlo | 0.990000 | 1 | 0.019972 | 0.000000 | 0.024504 | 0.140813 | 0.000000 | 2 |
| variance_covariance | 0.950000 | 1 | 0.022632 | 0.096953 | 0.007368 | 0.047534 | 0.500000 | 3 |
| variance_covariance | 0.990000 | 1 | 0.057890 | 0.026311 | 0.050170 | 0.124505 | 0.357143 | 4 |
| evt_pot | 0.990000 | 10 | 0.081910 | 0.033479 | 0.057162 | 0.250558 | 0.785714 | 5 |
| garch_1_1 | 0.950000 | 1 | 0.091341 | 0.336931 | 0.054187 | 0.232305 | 1.000000 | 6 |
| garch_1_1 | 0.990000 | 1 | 0.097902 | 0.039193 | 0.062737 | 0.496861 | 1.000000 | 7 |
| monte_carlo | 0.990000 | 10 | 0.119729 | 0.091808 | 0.114062 | 0.123728 | 0.357143 | 8 |
| variance_covariance | 0.990000 | 10 | 0.126630 | 0.090483 | 0.112770 | 0.142598 | 0.571429 | 9 |
| evt_pot | 0.950000 | 10 | 0.137692 | 0.568452 | 0.099357 | 0.220377 | 1.000000 | 10 |
| garch_1_1 | 0.990000 | 3 | 0.149538 | 0.088151 | 0.110495 | 1.000000 | 1.000000 | 11 |
| garch_1_1 | 0.950000 | 3 | 0.150508 | 0.616145 | 0.108662 | 0.522051 | 1.000000 | 12 |
| monte_carlo | 0.950000 | 10 | 0.152935 | 0.644283 | 0.114151 | 0.222268 | 1.000000 | 13 |
| variance_covariance | 0.950000 | 10 | 0.154173 | 0.648364 | 0.114948 | 0.261171 | 1.000000 | 14 |
| evt_pot | 0.950000 | 1 | 0.178716 | 0.784596 | 0.141526 | 0.000000 | 1.000000 | 15 |
| evt_pot | 0.990000 | 1 | 1.000000 | 1.000000 | 1.000000 | 0.014721 | 1.000000 | 16 |

## Column Descriptions

- **model** (object): 16 non-null values
- **confidence_level** (float64): 16 non-null values
- **horizon** (int64): 16 non-null values
- **composite_score** (float64): 16 non-null values
- **component_abs(hit_rate - expected_hit_rate)** (float64): 16 non-null values
- **component_violation_ratio** (float64): 16 non-null values
- **component_rmse_var_vs_losses** (float64): 16 non-null values
- **component_kupiec_rejection_rate** (float64): 16 non-null values
- **rank** (int64): 16 non-null values