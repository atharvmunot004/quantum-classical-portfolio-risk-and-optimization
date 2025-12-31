# Monte Carlo Simulation for VaR/CVaR Evaluation Report

**Generated:** 2025-12-31 11:42:06

## Model Configuration

### Distribution Estimation

- **Mean Model:** sample_mean (enabled: True)
- **Covariance Model:** sample_covariance
  - **Shrinkage:** ledoit_wolf

## Kernel Loop Order Validation

This implementation follows the time-first batch execution design:

**Loop Order:** `time -> batch -> simulations`

- **Outer loop:** time_index (rolling window end)
- **Inner loop:** portfolio_batch
- **Forbidden in batched path:**
  - `process_single_portfolio`
  - `compute_rolling_var`
  - `compute_rolling_cvar`

**Design Rationale:**
Rolling Monte Carlo risk must be computed time-first: at each rolling window end index,
scenarios are generated/loaded once and reused to compute VaR/CVaR for the entire
portfolio batch via a single BLAS matmul, then ALL metrics are updated via streaming accumulators.
This eliminates per-portfolio rolling loops while preserving all metrics.

## Methodology Overview

### Monte Carlo Simulation for VaR and CVaR

Monte Carlo simulation generates multiple scenarios of portfolio returns based on historical data.
VaR and CVaR are then calculated from the distribution of simulated returns.

**Advantages:**
- Can capture non-normal distributions
- Flexible with different distribution assumptions
- Provides both VaR and CVaR estimates

**Method:**
1. Estimate mean and covariance from historical returns
2. Simulate multiple scenarios of future returns
3. Calculate VaR as the quantile of simulated losses
4. Calculate CVaR as the expected loss beyond VaR

## Simulation Design

### Simulation Parameters

- **Number of Simulations:** 10,000
- **Distribution Type:** multivariate_normal
- **Random Seed:** 42
- **Confidence Levels:** [0.95, 0.99]
- **Base Horizon:** 1 days
- **Scaled Horizons:** [10] days
- **Scaling Rule:** sqrt_time
- **Estimation Windows:** [252] days

### Design Principle

- Asset-level simulation with portfolio projection
- Scenarios estimated once per estimation window and reused across portfolios
- Batched execution for scalability

## Batch Execution Summary

- **Total Portfolios Evaluated:** 100000
- **Total Configurations:** 400000

## Aggregate Backtesting Results

- **Average Hit Rate:** 0.0158

- **Average Violation Ratio:** 0.6553
  - Ratio > 1 indicates overestimation of risk
  - Ratio < 1 indicates underestimation of risk

### Traffic Light Zones

- **Green:** 245341 portfolios (61.3%)
- **Red:** 100000 portfolios (25.0%)
- **Yellow:** 54659 portfolios (13.7%)

- **Kupiec Test Passed:** 104014/400000 portfolios (26.0%)

## Tail Behavior Summary

- **Average Mean Exceedance (VaR):** 0.017187

- **Average Max Exceedance (VaR):** 0.074679

- **Average CVaR Mean Exceedance:** 0.016353

- **Average CVaR Max Exceedance:** 0.067354

## Tail Behavior Summary

- **Average Mean Exceedance (VaR):** 0.017187

- **Average Max Exceedance (VaR):** 0.074679

- **Average CVaR Mean Exceedance:** 0.016353

- **Average CVaR Max Exceedance:** 0.067354

- **Average Number of Active Assets:** 5.98

- **Average HHI Concentration:** 0.3383
  - Higher values indicate more concentrated portfolios

- **Average Skewness:** -0.9077
  - Negative values indicate left-skewed (tail risk)

- **Average Excess Kurtosis:** 15.3296
  - Positive values indicate fat tails

## Runtime Statistics

## Memory Statistics

Memory usage statistics are tracked during batch execution.
Check batch_progress.json in the cache directory for detailed memory metrics.

**Key Metrics:**
- Peak RSS (Resident Set Size): Peak memory usage during execution
- Swap Usage: Should be 0 MB (swap usage indicates memory pressure)

**Memory Optimization Features:**
- Float32 used for simulations and projections (halves memory footprint)
- Thread-based parallelism (avoids process memory duplication)
- Shard-based I/O (reduces memory spikes from large DataFrames)

## Summary Statistics

### Aggregate Metrics

```
        portfolio_id  confidence_level        horizon  estimation_window       hit_rate  violation_ratio  kupiec_unconditional_coverage  kupiec_test_statistic  christoffersen_independence  christoffersen_independence_statistic  christoffersen_conditional_coverage  christoffersen_conditional_coverage_statistic  num_violations  total_observations  expected_violations  mean_exceedance  max_exceedance  std_exceedance  quantile_loss_score  rmse_var_vs_losses  cvar_mean_exceedance  cvar_max_exceedance  cvar_std_exceedance       skewness       kurtosis  jarque_bera_p_value  jarque_bera_statistic  portfolio_size  num_active_assets  hhi_concentration  effective_number_of_assets  covariance_condition_number
count  400000.000000         400000.00  400000.000000           400000.0  400000.000000    400000.000000                  391778.000000           3.917780e+05                 3.917780e+05                          391778.000000                        391778.000000                                  391778.000000   400000.000000            400000.0        400000.000000     3.917780e+05    3.917780e+05    3.348730e+05        391778.000000        3.917780e+05          3.627130e+05         3.627130e+05         2.784210e+05  400000.000000  400000.000000             400000.0          400000.000000   400000.000000      400000.000000      400000.000000               400000.000000                 4.000000e+05
mean    49999.500000              0.97       5.500000              252.0       0.015837         0.655331                       0.100285           5.979548e+01                 4.974970e-01                               3.463318                             0.017020                                      63.258798       35.157283              2220.0            66.600000     1.718716e-02    7.467870e-02    1.888308e-02             0.001478        2.418822e-02          1.635330e-02         6.735353e-02         1.913736e-02      -0.907716      15.329559                  0.0           23819.561083        5.977960           5.977960           0.338342                    3.682122                 3.256387e+01
std     28867.549542              0.02       4.500006                0.0       0.017857         0.675176                       0.211747           7.866193e+01                 4.410300e-01                               4.721879                             0.061168                                      76.344321       39.641533                 0.0            44.400056     9.500370e-03    4.138569e-02    7.996431e-03             0.001033        9.831032e-03          9.779993e-03         4.346121e-02         6.428601e-03       0.425098       4.300559                  0.0           12139.851654        2.571562           2.571562           0.176871                    1.583867                 7.105436e-15
min         0.000000              0.95       1.000000              252.0       0.000000         0.000000                       0.000000          -1.705303e-13                 2.405368e-10                               0.000021                             0.000000                                       0.032661        0.000000              2220.0            22.200000     3.129244e-07    3.129244e-07    7.533749e-07             0.000370        3.129244e-07          5.289912e-07         5.289912e-07         5.004938e-07      -1.728951       4.083071                  0.0            1542.139647        2.000000           2.000000           0.105833                    1.000105                 3.256387e+01
25%     24999.750000              0.95       1.000000              252.0       0.000901         0.045045                       0.000000           3.244080e+00                 1.245709e-02                               0.008123                             0.000000                                      10.939200        2.000000              2220.0            22.200000     1.060700e-02    3.786369e-02    1.465734e-02             0.000689        1.813595e-02          1.130011e-02         2.586865e-02         1.636383e-02      -1.225303      12.232428                  0.0           14067.986784        4.000000           4.000000           0.207012                    2.370002                 3.256387e+01
50%     49999.500000              0.97       5.500000              252.0       0.006081         0.391892                       0.000004           2.118610e+01                 6.483363e-01                               0.208004                             0.000002                                      26.566488       13.500000              2220.0            66.600000     1.434948e-02    7.514229e-02    1.806635e-02             0.001181        2.289714e-02          1.447596e-02         7.178381e-02         1.971723e-02      -0.970461      15.577352                  0.0           22801.312099        6.000000           6.000000           0.281837                    3.548152                 3.256387e+01
75%     74999.250000              0.99      10.000000              252.0       0.026802         1.063063                       0.071682           1.801297e+02                 9.281869e-01                               6.244623                             0.004213                                     180.162225       59.500000              2220.0           111.000000     2.175854e-02    1.069386e-01    2.195053e-02             0.002500        2.858827e-02          1.828976e-02         1.026990e-01         2.287224e-02      -0.660677      18.631012                  0.0           32598.120373        8.000000           8.000000           0.421941                    4.830638                 3.256387e+01
max     99999.000000              0.99      10.000000              252.0       0.057658         2.567568                       1.000000           2.162210e+02                 9.963209e-01                              40.106099                             0.983802                                     216.221932      128.000000              2220.0           111.000000     1.127578e-01    2.154096e-01    7.993707e-02             0.005690        1.127578e-01          7.670262e-02         2.030709e-01         5.361895e-02       0.698602      26.010184                  0.0           63540.096362       10.000000          10.000000           0.999895                    9.448865                 3.256387e+01
```
