# Historical Value-at-Risk Evaluation Report

**Generated:** 2025-12-31 20:37:00

## Methodology Overview

### Historical Simulation VaR

The Historical Simulation method uses empirical quantiles from historical return distributions.
VaR is calculated as:

```
VaR_h = -Q_α(R_1) * √(h)
```

Where:
- Q_α(R_1) = α-quantile of historical 1-day returns
- α = 1 - confidence_level (e.g., 0.05 for 95% VaR)
- h = time horizon in days
- √(h) = square root scaling for multi-day horizons

The method uses empirical quantiles with linear interpolation and computes
asset-level returns once, then projects to portfolios via linear combination:
R_p(t) = W^T R_assets(t)

### VaR Settings

- **Confidence Levels:** [0.95, 0.99]
- **Horizons:** {'base_horizon': 1, 'scaled_horizons': [10], 'scaling_rule': 'sqrt_time', 'assumptions': {'iid_returns': True, 'note': 'Square-root-of-time scaling is applied as a benchmark approximation; limitations are discussed relative to EVT and GARCH-based models.'}} days
- **Estimation Windows:** [252] days

## Summary Statistics

- **Total Portfolios Evaluated:** 400000

## Backtesting Results

- **Average Hit Rate:** 0.0144

- **Average Violation Ratio:** 0.5034
  - Ratio > 1 indicates overestimation of risk
  - Ratio < 1 indicates underestimation of risk

### Traffic Light Zones

- **Green:** 200000 portfolios (50.0%)

- **Kupiec Test Passed:** 193217/400000 portfolios (48.3%)

## Tail Risk Analysis

- **Average Mean Exceedance:** 0.017490

- **Average Max Exceedance:** 0.085876

## Portfolio Structure Effects

- **Average Number of Active Assets:** 5.98

- **Average HHI Concentration:** 0.3383
  - Higher values indicate more concentrated portfolios

- **Average Effective Number of Assets:** 3.68

## Robustness and Normality Checks

- **Average Skewness:** -0.8125
  - Negative values indicate left-skewed (tail risk)

- **Average Excess Kurtosis:** 14.1372
  - Positive values indicate fat tails

- **Normality Tests Passed (Jarque-Bera):** 0/400000 portfolios (0.0%)

## Computational Performance

- **Average Runtime per Portfolio:** 1221.16 ms

- **95th Percentile Runtime:** 1397.73 ms

## Key Insights

### Findings

- **Risk Underestimation:** VaR tends to underestimate risk (violation ratio < 0.8)

- **Fat Tails Detected:** Returns exhibit fat tails, which may limit VaR accuracy

### Recommendations

- Consider alternative VaR methods (e.g., Monte Carlo, Historical Simulation) for portfolios with fat-tailed returns
- Monitor portfolios in 'red' traffic light zone more closely
- Adjust confidence levels or horizons based on backtesting results

## Figures

_Figure generation can be extended here with visualization libraries._
