# Markowitz Mean-Variance Portfolio Optimization Report

**Generated:** 2026-01-24 04:42:55

## Methodology Overview

### Markowitz Mean-Variance Optimization

The Markowitz (1952) mean-variance framework optimizes portfolios by balancing expected return and risk (variance).
The optimization problem is:

```
minimize: w^T Σ w - λ * μ^T w
subject to: Σ w_i = 1, w_i >= 0 (long-only)
```

Where:
- w = portfolio weights vector
- Σ = covariance matrix
- μ = expected returns vector
- λ = risk aversion parameter

### Optimization Settings

- **Objective:** min_variance
- **Long Only:** True
- **Fully Invested:** True
- **Max Weight per Asset:** 0.25
- **Covariance Method:** sample
- **Estimation Windows:** [252] days
- **Shrinkage:** ledoit_wolf

## Mean-Variance Theory

The efficient frontier represents the set of portfolios that offer the highest expected return for a given level of risk, or the lowest risk for a given expected return.

Key concepts:
- **Efficient Frontier:** Optimal risk-return combinations
- **Minimum Variance Portfolio:** Portfolio with lowest risk
- **Maximum Sharpe Portfolio:** Portfolio with highest risk-adjusted return
- **Risk-Return Tradeoff:** Higher returns require accepting higher risk

## Summary Statistics

- **Total Portfolios Optimized:** 100000

## Portfolio Optimization Results

- **Average Expected Return:** 0.051343 (5.13%)

- **Average Volatility:** 0.007537 (0.75%)

- **Average Portfolio Variance:** 0.000057

- **Average Sharpe Ratio:** 1.0203
- **Maximum Sharpe Ratio:** 1.0203

- **Average Sortino Ratio:** 0.9475

## Risk-Return Tradeoff Analysis

- **Return-Volatility Correlation:** 1.0000
  - Positive correlation indicates higher returns come with higher risk

### Sharpe Ratio Distribution

- **Mean:** 1.0203
- **Std:** 0.0000
- **Min:** 1.0203
- **Max:** 1.0203

## Portfolio Structure Effects

- **Average Number of Assets:** 10.00

- **Average HHI Concentration:** 0.1000
  - Higher values indicate more concentrated portfolios
  - HHI ranges from 1/n (equal weights) to 1 (single asset)

- **Average Effective Number of Assets:** 10.00

- **Average Weight Entropy:** 2.3026
  - Higher entropy indicates better diversification

- **Average Pairwise Correlation:** 0.3041

## Risk Metrics

- **Average Value at Risk (VaR):** 0.016231
- **Average Conditional VaR (CVaR):** 0.026873

## Robustness and Normality Checks

- **Average Skewness:** -1.3695
  - Negative skewness indicates left tail risk

- **Average Excess Kurtosis:** 20.9034
  - Positive kurtosis indicates fat tails

- **Normality Rejection Rate (p < 0.05):** 100.00%
  - Higher rate indicates more non-normal return distributions

## Computational Performance

- **Average Runtime per Optimization:** 10.71 ms

- **Average Covariance Estimation Time:** 0.00 ms

- **Average Solver Time:** 0.94 ms

## Limitations of Mean-Variance Framework

The Markowitz mean-variance framework, while foundational, has several well-known limitations:

### 1. Normality Assumption
- Assumes returns follow a normal distribution
- Real-world returns often exhibit skewness, kurtosis, and fat tails
- May underestimate tail risk and extreme events

- **Observed average skewness:** -1.3695 (indicates non-normal distribution)

- **Observed average excess kurtosis:** 20.9034 (indicates fat tails)

### 2. Estimation Error
- Covariance matrix and expected returns are estimated from historical data
- Estimation error is not explicitly modeled
- Small sample sizes can lead to unstable covariance estimates
- Out-of-sample performance may differ significantly from in-sample

### 3. Static Framework
- Assumes parameters (returns, covariances) are constant over time
- Does not account for time-varying volatility or regime changes
- May not adapt well to changing market conditions

### 4. Risk Measure Limitations
- Variance as risk measure treats upside and downside volatility equally
- Does not distinguish between good and bad volatility
- May not capture tail risk adequately (VaR/CVaR provide better tail risk measures)

### 5. Sensitivity to Inputs
- Optimal portfolios are highly sensitive to expected return estimates
- Small changes in inputs can lead to large changes in optimal weights
- This sensitivity makes the framework less robust in practice

### 6. Transaction Costs and Constraints
- Does not explicitly model transaction costs
- May suggest frequent rebalancing which is costly
- Integer constraints (e.g., minimum lot sizes) not naturally handled

### Alternative Approaches
These limitations motivate alternative frameworks:
- **Black-Litterman:** Incorporates investor views and addresses estimation error
- **CVaR Optimization:** Focuses on tail risk rather than variance
- **Robust Optimization:** Accounts for parameter uncertainty
- **Bayesian Methods:** Explicitly model estimation uncertainty

## Key Insights

- Maximum Sharpe ratio achieved: 1.0203
- Portfolios show good diversification (HHI < 0.2)
- Strong positive risk-return tradeoff observed
