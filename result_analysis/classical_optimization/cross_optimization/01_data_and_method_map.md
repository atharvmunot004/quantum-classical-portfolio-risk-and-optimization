# Data and Method Map

## Optimization Methods Compared

### Markowitz Mean-Variance
- **Objective:** Maximize return for given risk (variance)
- **Constraints:** Fully invested, long-only
- **Solver:** Quadratic Programming (QP)

### Black-Litterman
- **Objective:** Incorporate investor views with market equilibrium
- **Constraints:** Fully invested, long-only
- **Solver:** Quadratic Programming (QP)

### CVaR Optimization
- **Objective:** Minimize Conditional Value at Risk
- **Constraints:** Fully invested, long-only
- **Solver:** Linear Programming (LP) via Rockafellar-Uryasev formulation

### Risk Parity/ERC
- **Objective:** Equalize risk contributions across assets
- **Constraints:** Fully invested, long-only
- **Solver:** Nonlinear optimization

## Evaluation Assumptions

- Confidence level filtering: Prefer 0.95, also compute for 0.99
- Horizon alignment: Primary horizon 1 day
- Metric harmonization: Canonical metric set with alias resolution
