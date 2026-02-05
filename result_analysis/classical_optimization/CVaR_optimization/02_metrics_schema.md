# Metrics Schema

**Number of metrics:** 47
**Number of portfolios:** 100000

## Schema Definition

| Metric Name | Data Type | Description |
|-------------|-----------|-------------|
| portfolio_id | int64 | N/A |
| confidence_level | float64 | N/A |
| expected_return | float64 | N/A |
| volatility | float64 | N/A |
| var | float64 | N/A |
| cvar | float64 | N/A |
| sharpe_ratio | float64 | N/A |
| sortino_ratio | float64 | N/A |
| cvar_sharpe_ratio | float64 | N/A |
| cvar_sortino_ratio | float64 | N/A |
| cvar_calmar_ratio | float64 | N/A |
| return_over_cvar | float64 | N/A |
| return_over_var | float64 | N/A |
| max_drawdown | float64 | N/A |
| calmar_ratio | float64 | N/A |
| num_assets_in_portfolio | int64 | N/A |
| hhi_concentration | float64 | N/A |
| effective_number_of_assets | float64 | N/A |
| weight_entropy | float64 | N/A |
| pairwise_correlation_mean | float64 | N/A |
| value_at_risk | float64 | N/A |
| conditional_value_at_risk | float64 | N/A |
| expected_shortfall | float64 | N/A |
| downside_deviation | float64 | N/A |
| semivariance | float64 | N/A |
| tail_ratio | float64 | N/A |
| cvar_var_ratio | float64 | N/A |
| worst_case_loss | float64 | N/A |
| expected_loss_beyond_cvar | float64 | N/A |
| skewness | float64 | N/A |
| kurtosis | float64 | N/A |
| jarque_bera_p_value | float64 | N/A |
| tail_index | float64 | N/A |
| extreme_value_risk | float64 | N/A |
| cvar_coherence_check | float64 | N/A |
| cvar_consistency_across_conf_levels | float64 | N/A |
| cvar_95 | float64 | N/A |
| cvar_99 | float64 | N/A |
| var_95 | float64 | N/A |
| var_99 | float64 | N/A |
| runtime_per_optimization_ms | float64 | N/A |
| scenario_construction_time_ms | float64 | N/A |
| solver_time_ms | float64 | N/A |
| optimization_status | object | N/A |
| metrics_present_count | int64 | N/A |
| metrics_missing_count | int64 | N/A |
| metrics_missing_list | object | N/A |