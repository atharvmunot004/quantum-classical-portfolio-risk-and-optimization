"""
Extreme Value Theory (EVT) - Peaks Over Threshold (POT) for VaR and CVaR evaluation.

This module implements EVT-POT methodology for estimating extreme risk measures
at the asset level using the Generalized Pareto Distribution (GPD) fitted to 
exceedances over a threshold with rolling window estimation.
"""

from .main import evaluate_evt_pot_var_cvar

__all__ = ['evaluate_evt_pot_var_cvar']

