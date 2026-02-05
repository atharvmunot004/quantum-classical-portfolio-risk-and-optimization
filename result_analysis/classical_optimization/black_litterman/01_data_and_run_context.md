# Data and Run Context

## Configuration Summary
- Tool: black_litterman
- Trading days per year: 252
- Risk-free rate: 0.0001

## Input Files
- optimal_portfolios: `results/classical_optimization/bl_optimal_weights.parquet`
- posterior_returns: `results/classical_optimization/bl_posterior_returns.parquet`
- posterior_covariance: `results/classical_optimization/bl_posterior_covariance.parquet`
- portfolio_daily_returns_matrix: `results/classical_optimization/bl_portfolio_daily_returns.npy`
- metrics_table: `results/classical_optimization/bl_metrics.parquet`
- time_sliced_metrics: `results/classical_optimization/bl_time_sliced_metrics.parquet`

## Runtime Profile
```json
{
  "stages": {
    "stage_A_precompute": 13320.483632802963,
    "stage_B_expand_weights": 1.1128818988800049,
    "stage_C_batch_evaluate": 985.7040407657623
  },
  "total_time": 14320.97786450386,
  "gpu_info": {
    "available": false,
    "library": null,
    "device": null
  }
}
```