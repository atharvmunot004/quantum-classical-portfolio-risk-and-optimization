# IEEE Access Ready Paragraphs

## Experimental Setup

Four portfolio optimization methods were evaluated: Markowitz Mean-Variance, 
Black-Litterman, CVaR Optimization, and Risk Parity/ERC. Each method generated 
portfolios optimized under fully invested, long-only constraints. Metrics were 
harmonized across tools to ensure consistent comparison.

## Cross-Tool Findings

Analysis reveals significant variation in portfolio performance across optimization methods. 
Markowitz optimization provides baseline risk-return tradeoffs, while Black-Litterman 
incorporates investor views for improved return forecasts. CVaR optimization excels 
in tail-risk management, and Risk Parity provides superior diversification through equal 
risk contribution.

## Discussion

The choice of optimization method depends on investor objectives: return maximization 
(Markowitz, Black-Litterman), tail-risk management (CVaR), or risk diversification 
(Risk Parity). No single method dominates across all metrics, highlighting the importance 
of aligning optimization objectives with investment goals.

## Limitations

Comparison assumes consistent input data and evaluation methodology. Real-world 
performance may vary based on market conditions, parameter estimation, and implementation 
details. Runtime comparisons reflect algorithmic complexity but may vary with problem size 
and solver configuration.
