# IEEE Access Ready Paragraphs

## Experimental Setup (Markowitz Assumptions and Constraints)

The Markowitz Mean-Variance optimization framework was applied to generate 100000 
optimal portfolios. The analysis employs quadratic programming to solve the mean-variance optimization problem, 
subject to constraints including fully invested portfolios (weights sum to one) and long-only positions.

## Efficient Frontier Interpretation

The efficient frontier represents the set of optimal portfolios that maximize expected return for a given level 
of risk, measured as portfolio volatility. Portfolios on the frontier dominate all other portfolios in terms of 
risk-return efficiency, providing investors with the optimal tradeoff between risk and return.

## Risk-Return Tradeoff Findings

Analysis reveals a positive relationship between expected return and volatility, consistent with financial theory. 
Higher returns are associated with increased risk, as measured by portfolio variance. The Sharpe ratio, which 
measures risk-adjusted returns, varies across portfolios, with optimal portfolios achieving superior risk-adjusted performance.

## Distributional Violations and Tail Risk Implications

While Markowitz optimization assumes normally distributed returns, empirical analysis reveals deviations from normality, 
including negative skewness and excess kurtosis. These distributional violations suggest that variance-based risk measures 
may underestimate tail risk, highlighting the limitations of classical mean-variance optimization for extreme downside scenarios.

## Limitations as Classical Baseline

Markowitz Mean-Variance optimization serves as a classical baseline for portfolio optimization, but exhibits limitations 
including sensitivity to input parameters (expected returns and covariance matrix), assumption of normal returns, and 
focus on overall volatility rather than tail risk. These limitations motivate alternative approaches such as CVaR optimization 
and Black-Litterman models that address specific weaknesses of the classical framework.
