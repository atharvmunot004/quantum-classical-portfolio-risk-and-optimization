# Global Summary Table

## Tool-level Summary

| tool_name | expected_return_mean | expected_return_p50 | expected_return_p75 | expected_return_p95 | volatility_mean | volatility_p50 | volatility_p75 | volatility_p95 | sharpe_ratio_mean | sharpe_ratio_p50 | sharpe_ratio_p75 | sharpe_ratio_p95 | sortino_ratio_mean | sortino_ratio_p50 | sortino_ratio_p75 | sortino_ratio_p95 | max_drawdown_mean | max_drawdown_p50 | max_drawdown_p75 | max_drawdown_p95 | conditional_value_at_risk_mean | conditional_value_at_risk_p50 | conditional_value_at_risk_p75 | conditional_value_at_risk_p95 | runtime_per_optimization_ms_mean | runtime_per_optimization_ms_p50 | runtime_per_optimization_ms_p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| markowitz_mean_variance | 0.051343 | 0.051343 | 0.051343 | 0.051343 | 0.007537 | 0.007537 | 0.007537 | 0.007537 | 1.020271 | 1.020271 | 1.020271 | 1.020271 | 0.947459 | 0.947459 | 0.947459 | 0.947459 | 0.376434 | 0.376434 | 0.376434 | 0.376434 | 0.026873 | 0.026873 | 0.026873 | 0.026873 | 10.709042 | 10.884404 | 21.489871 |
| black_litterman | 0.165323 | 0.181512 | 0.202368 | 0.202368 | 0.178806 | 0.192284 | 0.203716 | 0.205518 | 0.911668 | 0.918794 | 0.992890 | 0.992890 | 1.279305 | 1.288047 | 1.380985 | 1.381362 | 0.381676 | 0.406219 | 0.455072 | 0.455072 | 0.025684 | 0.027814 | 0.029462 | 0.029641 |  |  |  |
| cvar_optimization | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |  |  |  |  |  |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 18.240952 | 16.162753 | 33.915865 |
| risk_parity_erc | 0.171425 | 0.178032 | 0.222296 | 0.311526 | 0.009479 | 0.008319 | 0.010694 | 0.015112 | 1.192119 | 1.015770 | 1.659360 | 2.473647 | 1.162547 | 0.941455 | 1.661221 | 2.512061 | 0.190352 | 0.165685 | 0.191615 | 0.376434 | 0.023192 | 0.019841 | 0.027899 | 0.040146 | 13.404349 | 13.678789 | 25.599241 |

## Interpretation

This table provides comprehensive tool-level summary statistics. Key observations include 
which tools dominate by risk-adjusted return (Sharpe ratio) versus tail risk (CVaR, drawdown).
