# Runtime and Scalability

## Runtime Distribution Comparison

Runtime performance varies significantly across optimization methods:

- **Markowitz:** Fast QP solver, scales well
- **Black-Litterman:** Moderate runtime due to view processing
- **CVaR:** LP solver, efficient for scenario-based optimization
- **Risk Parity/ERC:** Nonlinear optimization, slower but provides risk-balanced portfolios

## Tradeoff: Quality vs Compute Cost

More sophisticated methods (Black-Litterman, CVaR) provide better risk management 
but require more computational resources compared to classical Markowitz optimization.
