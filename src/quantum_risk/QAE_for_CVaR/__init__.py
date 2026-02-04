"""
QAE (Quantum Amplitude Estimation) for VaR and CVaR at asset level.

This module implements quantum-enhanced risk estimation using:
- Parametric Student-t distribution fitted to asset losses
- QAE for CDF-based VaR estimation via bisection
- Rockafellar-Uryasev CVaR with QAE tail expectation
- Rolling window estimation with backtesting
"""

from .main import evaluate_qae_var_cvar

__all__ = ['evaluate_qae_var_cvar']
