"""
QAE Portfolio CVaR Evaluation Package.

Implements Quantum Amplitude Estimation for portfolio-level CVaR evaluation
with precomputation and reuse across portfolios.
"""

from .main import run_qae_portfolio_evaluation

__all__ = ['run_qae_portfolio_evaluation']
