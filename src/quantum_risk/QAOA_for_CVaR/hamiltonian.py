"""
Hamiltonian construction for QAOA CVaR asset-level risk.

Builds Ising cost Hamiltonian from discretized loss objective and X mixer.
"""
import numpy as np
from typing import Optional

try:
    from qiskit.quantum_info import SparsePauliOp, Operator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from .loss_discretization import DiscretizationResult


def build_cost_hamiltonian_from_discretized_loss(
    discretization: DiscretizationResult,
    num_qubits: int,
) -> "SparsePauliOp":
    """
    Build diagonal Ising cost Hamiltonian from discretized loss levels.

    Each basis state |i> (i = 0..2^n-1) has energy = loss_levels[i].
    The Hamiltonian is diagonal in the computational basis.

    Args:
        discretization: Result from quantile_grid discretization
        num_qubits: Number of qubits (num_levels must equal 2^num_qubits)

    Returns:
        SparsePauliOp cost Hamiltonian
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for Hamiltonian construction")

    loss_levels = discretization.loss_levels
    n_levels = len(loss_levels)
    if n_levels != (1 << num_qubits):
        raise ValueError(
            f"num_levels {n_levels} must equal 2^num_qubits = {1 << num_qubits}"
        )

    diag = np.array(loss_levels, dtype=np.float64)
    op = Operator(np.diag(diag))
    cost_h = SparsePauliOp.from_operator(op)
    return cost_h.simplify()


def build_x_mixer(num_qubits: int) -> "SparsePauliOp":
    """
    Build standard transverse-field X mixer: sum_i X_i.

    For unconstrained binary search space (QAOA standard mixer).

    Args:
        num_qubits: Number of qubits

    Returns:
        SparsePauliOp mixer Hamiltonian
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for Hamiltonian construction")

    terms = []
    for i in range(num_qubits):
        pauli_str = "I" * i + "X" + "I" * (num_qubits - 1 - i)
        terms.append((pauli_str, 1.0))
    return SparsePauliOp.from_list(terms)
