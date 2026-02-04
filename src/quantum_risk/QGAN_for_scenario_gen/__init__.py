"""
QGAN for Scenario Generation - Asset-Level Evaluation

Quantum Generative Adversarial Networks for generating realistic risk scenarios
at the asset level using rolling-window training.
"""
from .main import evaluate_qgan_scenarios

__all__ = ['evaluate_qgan_scenarios']
