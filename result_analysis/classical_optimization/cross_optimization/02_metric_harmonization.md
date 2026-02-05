# Metric Harmonization

## Canonical Metric Set

- expected_return
- volatility
- sharpe_ratio
- sortino_ratio
- max_drawdown
- calmar_ratio
- value_at_risk
- conditional_value_at_risk
- downside_deviation
- skewness
- kurtosis
- jarque_bera_p_value
- runtime_per_optimization_ms
- solver_time_ms

## Alias Resolution

Each tool may use different column names for the same metric. The harmonization process 
resolves these aliases to canonical metric names for consistent comparison.

## Sign Convention Normalization

- **Drawdown:** Normalized to fraction in [0,1]
- **VaR/CVaR:** Normalized to positive loss magnitude
