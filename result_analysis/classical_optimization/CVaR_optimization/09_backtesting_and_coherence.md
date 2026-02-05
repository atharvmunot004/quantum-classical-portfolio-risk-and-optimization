# Backtesting and Coherence

## CVaR Coherence Properties

CVaR satisfies the coherence axioms for risk measures:

1. **Monotonicity**: If portfolio A has returns always greater than portfolio B, CVaR(A) ≤ CVaR(B)
2. **Translation Invariance**: CVaR(X + c) = CVaR(X) - c for constant c
3. **Positive Homogeneity**: CVaR(λX) = λCVaR(X) for λ ≥ 0
4. **Subadditivity**: CVaR(X + Y) ≤ CVaR(X) + CVaR(Y)

## Validation Results

- ✓ CVaR ≥ VaR: Verified