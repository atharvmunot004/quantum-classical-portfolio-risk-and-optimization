# Metrics Schema

**Number of metrics:** 52
**Number of portfolios:** 2619000

## Schema Definition

| Metric Name | Data Type | Description |
|-------------|-----------|-------------|
| portfolio_id | int64 | N/A |
| method | object | N/A |
| expected_return | float64 | N/A |
| volatility | float64 | N/A |
| portfolio_variance | float64 | N/A |
| sharpe_ratio | float64 | N/A |
| sortino_ratio | float64 | N/A |
| max_drawdown | float64 | N/A |
| calmar_ratio | float64 | N/A |
| risk_contribution_vector | float64 | N/A |
| risk_contribution_variance | float64 | N/A |
| risk_contribution_std | float64 | N/A |
| risk_contribution_coefficient_of_variation | float64 | N/A |
| max_risk_contribution | float64 | N/A |
| min_risk_contribution | float64 | N/A |
| equal_risk_gap | float64 | N/A |
| volatility_contribution_per_asset | float64 | N/A |
| marginal_risk_contribution_vector | float64 | N/A |
| risk_parity_deviation_score | float64 | N/A |
| risk_concentration_index | float64 | N/A |
| num_assets_in_portfolio | int64 | N/A |
| hhi_concentration | float64 | N/A |
| effective_number_of_assets | float64 | N/A |
| weight_entropy | float64 | N/A |
| pairwise_correlation_mean | float64 | N/A |
| value_at_risk | float64 | N/A |
| conditional_value_at_risk | float64 | N/A |
| downside_deviation | float64 | N/A |
| semivariance | float64 | N/A |
| skewness | float64 | N/A |
| kurtosis | float64 | N/A |
| jarque_bera_p_value | float64 | N/A |
| baseline_portfolio_volatility | float64 | N/A |
| baseline_portfolio_sharpe | float64 | N/A |
| baseline_portfolio_expected_return | float64 | N/A |
| volatility_reduction_vs_baseline | float64 | N/A |
| sharpe_improvement_vs_baseline | float64 | N/A |
| risk_contribution_improvement_vs_baseline | float64 | N/A |
| erc_vs_equal_weight_volatility | float64 | N/A |
| erc_vs_equal_weight_sharpe | float64 | N/A |
| erc_vs_equal_weight_risk_contributions | float64 | N/A |
| difference_in_risk_contributions_vs_baseline | float64 | N/A |
| runtime_per_optimization_ms | float64 | N/A |
| covariance_estimation_time_ms | float64 | N/A |
| risk_contribution_calculation_time_ms | float64 | N/A |
| solver_time_ms | float64 | N/A |
| optimization_status | object | N/A |
| covariance_estimator | object | N/A |
| estimation_window | int64 | N/A |
| rebalance_date | datetime64[ns] | N/A |
| rebalancing_frequency | object | N/A |
| stage | object | N/A |