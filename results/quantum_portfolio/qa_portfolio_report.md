# Quantum Annealing Multi-Objective Portfolio Optimization Report

Generated: 2026-02-05 17:37:03

## Problem Formulation

### Objectives

1. **Expected Return Maximization**: Maximize portfolio expected return
2. **Tail Risk Minimization**: Minimize Conditional Value at Risk (CVaR)
3. **Diversification**: Minimize pairwise correlation penalty

### Constraints

- **Budget Constraint**: Fully invested (sum of weights = 1.0)
- **Cardinality Constraint**: 5-15 assets
- **Long-Only**: No short selling allowed

## QUBO Construction

The portfolio optimization problem is formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem:

```
minimize: x^T Q x + c
subject to: x ∈ {0, 1}^n
```

### Objective Weights
- Return weight: [0.5, 1.0]
- Risk weight: [1.0, 2.0]
- Diversification weight: [0.1, 0.5]

### Penalty Weights
- Budget penalty: 10.0
- Cardinality penalty: 8.0

## Quantum Annealing and Embedding

### Solver Configuration
- Backend: dwave_or_simulated_annealer
- Number of reads: 5000
- Annealing time: [20, 50] μs

- Average embedding size: nan qubits
- Average logical to physical qubit ratio: nan
- Average chain break fraction: nan

## Multi-Objective Tradeoffs

### Performance by Weight Configuration

 return_weight  risk_weight  diversification_weight  realized_return  realized_volatility  realized_cvar  sharpe_ratio
           0.5          1.0                     0.1         0.855726             0.150008      -0.283973       4.20085
           0.5          1.0                     0.5         0.855726             0.150008      -0.283973       4.20085
           0.5          2.0                     0.1         0.855726             0.150008      -0.283973       4.20085
           0.5          2.0                     0.5         0.855726             0.150008      -0.283973       4.20085
           1.0          1.0                     0.1         0.855726             0.150008      -0.283973       4.20085
           1.0          1.0                     0.5         0.855726             0.150008      -0.283973       4.20085
           1.0          2.0                     0.1         0.855726             0.150008      -0.283973       4.20085
           1.0          2.0                     0.5         0.855726             0.150008      -0.283973       4.20085


## Out-of-Sample Performance

### Overall Performance Metrics

- Mean Realized Return: 0.8557
- Mean Realized Volatility: 0.1500
- Mean Realized CVaR: -0.2840
- Mean Sharpe Ratio: 4.2009


## Benchmark Comparison

Benchmark comparison would require running classical optimization methods.
This section can be populated after running benchmark optimizations.

## Limitations and Scalability

### Current Limitations

1. **Asset Universe Size**: Limited by quantum annealer connectivity (currently ~40 assets)
2. **Embedding Overhead**: Logical to physical qubit ratio affects solution quality
3. **Chain Breaks**: May occur in D-Wave systems, affecting solution quality
4. **Computation Time**: Quantum annealing may be slower than classical methods for small problems

### Scalability Considerations

- Larger asset universes require more sophisticated embedding strategies
- Multi-objective weight sweeps increase computation time linearly
- Rolling window optimization scales with number of rebalancing dates

## Key Insights

- Best energy achieved: 348.8233
- Average energy gap: 0.0000
- Average Pareto front size: 1.0

### Recommendations

1. Experiment with different weight configurations to explore Pareto frontier
2. Monitor chain break fractions to assess solution quality
3. Compare with classical optimization methods for validation
4. Consider hybrid quantum-classical approaches for larger problems

## Summary

Total optimizations performed: 1504
Total portfolio weight records: 15040
