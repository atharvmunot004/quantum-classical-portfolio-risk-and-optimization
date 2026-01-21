# Variance-Covariance Value-at-Risk Evaluation Report
Asset-Level Evaluation

**Generated:** 2026-01-22 01:15:47

## Methodology Overview

### Variance-Covariance (Parametric) VaR

The Variance-Covariance method assumes returns follow a normal distribution
and calculates VaR using parametric estimation:

```
VaR_h = -μ + z_{1-α} * σ * √(h)
```

Where:
- μ = mean return (estimated from rolling window)
- σ = standard deviation of returns (estimated from rolling window)
- z_{1-α} = (1-α) quantile of standard normal distribution
- α = 1 - confidence_level (e.g., α = 0.05 for 95% VaR)
- h = time horizon in days
- √(h) = square root scaling for multi-day horizons

This method is evaluated strictly per asset using rolling windows
under the assumption of conditional normality.

### VaR Settings

- **Distributional Assumption:** normal
- **Mean Estimator:** sample_mean
- **Volatility Estimator:** sample_std
- **Confidence Levels:** [0.95, 0.99]
- **Horizons:** [1, 10] days
- **Scaling Rule:** sqrt_time
- **Estimation Windows:** [252, 500] days

## Normality Assumption

The Variance-Covariance method assumes returns follow a normal distribution.
This assumption is tested using the Jarque-Bera test for normality.

- **Normality Tests Passed (Jarque-Bera, p > 0.05):** 0/80 assets (0.0%)

**Warning:** Less than 50% of assets pass the normality test.
This suggests the normal distribution assumption may not hold for many assets.

## Rolling Mean and Volatility Estimation

Mean and volatility are estimated using rolling windows:

- **Mean Estimator:** sample_mean
- **Volatility Estimator:** sample_std

- **Average Rolling Mean:** 0.000302

- **Average Rolling Volatility:** 0.013503

## Variance-Covariance VaR Construction

VaR is constructed using the parametric formula with estimated mean and volatility.
For each rolling window position:
1. Estimate mean (μ) and volatility (σ) from the window
2. Compute VaR = -μ + z_{1-α} * σ * √(h)
3. Advance window by step size and repeat

## Summary Statistics

- **Total Assets Evaluated:** 10
- **Total Configuration Combinations:** 80

## Backtesting Results

- **Average Hit Rate:** 0.0142

- **Average Violation Ratio:** 0.5936
  - Ratio > 1 indicates overestimation of risk
  - Ratio < 1 indicates underestimation of risk
  - Ratio ≈ 1 indicates accurate risk estimation

### Traffic Light Zones

- **Green:** 33 asset-configurations (41.2%)
- **Yellow:** 7 asset-configurations (8.8%)

- **Kupiec Test Passed (p > 0.05):** 16/80 configurations (20.0%)

- **Christoffersen Conditional Coverage Test Passed (p > 0.05):** 9/80 configurations (11.2%)

## Time-Sliced Backtesting

Backtesting metrics are computed for specific time periods (year, quarter, month)
to analyze temporal patterns in VaR performance.

See time-sliced metrics table for detailed results by time period.

## Distributional Characteristics

- **Average Skewness:** -0.2719
  - Negative values indicate left-skewed distributions (tail risk)
  - Positive values indicate right-skewed distributions

- **Average Excess Kurtosis:** 9.3860
  - Positive values indicate fat tails (excess kurtosis > 0)
  - Normal distribution has excess kurtosis = 0

**Warning:** Average excess kurtosis > 3 indicates significant fat tails.
This violates the normality assumption of Variance-Covariance VaR.

- **Average Jarque-Bera p-value:** 0.0000
  - p > 0.05 suggests normality cannot be rejected
  - p < 0.05 suggests non-normality

## Computational Performance

- **Average Runtime per Asset:** 12924.53 ms

- **95th Percentile Runtime:** 13218.43 ms

- **Mean Estimation Time:** 1981.22 ms

- **Volatility Estimation Time:** 1981.22 ms

- **VaR Computation Time:** 9881.49 ms

## Key Insights

### Findings

- **Risk Underestimation:** VaR tends to underestimate risk (violation ratio < 0.8)
  This may indicate the normal distribution assumption fails to capture tail risk.

- **Fat Tails Detected:** Returns exhibit significant fat tails (excess kurtosis > 3)
  This violates the normality assumption and may limit VaR accuracy.

- **Normality Assumption Violated:** Less than 50% of assets pass normality tests.
  Consider alternative methods (e.g., EVT, GARCH) for assets with non-normal returns.

### Recommendations

- Use Variance-Covariance VaR as a baseline for comparison with other methods
- For assets with fat tails or non-normal returns, consider EVT or GARCH methods
- Monitor assets in 'red' traffic light zone more closely
- Adjust confidence levels or horizons based on backtesting results
- Consider longer estimation windows for more stable parameter estimates
