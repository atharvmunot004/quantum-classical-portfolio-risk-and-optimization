"""
Quantum circuits for QAE-based CDF and tail expectation estimation.

Implements state preparation from analytic distribution and Iterative Amplitude Estimation
for CDF(L <= x) used in VaR bisection.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

try:
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import StatePreparation
    from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
    from qiskit_aer.primitives import Sampler
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


@dataclass
class QAEResult:
    """Result of QAE estimation."""

    estimation: float
    confidence_interval: Tuple[float, float]
    num_iterations: int
    circuit_depth: int
    num_shots: int


def _bin_index_for_threshold(
    bin_centers: np.ndarray,
    threshold: float
) -> int:
    """Return the largest bin index i such that bin_centers[i] <= threshold."""
    idx = np.searchsorted(bin_centers, threshold, side='right') - 1
    return max(0, min(idx, len(bin_centers) - 1))


def _add_cdf_oracle(
    qc: "QuantumCircuit",
    state_qubits: list,
    ancilla_qubit,
    good_state_indices: list
) -> None:
    """
    Add oracle that flips ancilla when state index is in good_state_indices.
    Uses multi-controlled X for each good state (scalable for small num_qubits).
    """
    num_qubits = len(state_qubits)
    n_states = 1 << num_qubits

    for idx in good_state_indices:
        if idx >= n_states:
            break
        # Multicontrolled X: when state equals |idx>, flip ancilla
        # ctrl_state: '1' means control is active when that qubit is 1
        ctrl_state = format(idx, f'0{num_qubits}b')
        qc.mcx(state_qubits, ancilla_qubit, ctrl_state=ctrl_state)


def create_cdf_estimation_circuit(
    probs: np.ndarray,
    bin_centers: np.ndarray,
    threshold: float
) -> Tuple[QuantumCircuit, int]:
    """
    Create state preparation + oracle for CDF(L <= threshold) estimation.

    Args:
        probs: Probability vector (length 2^n)
        bin_centers: Loss values at bin centers
        threshold: VaR candidate (loss threshold)

    Returns:
        (circuit, ancilla_qubit_index)
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for QAE. Install with: pip install qiskit qiskit-aer qiskit-algorithms")

    n = int(np.log2(len(probs)))
    if 2**n != len(probs):
        raise ValueError("probs length must be power of 2")

    k = _bin_index_for_threshold(bin_centers, threshold)
    good_indices = list(range(k + 1))

    # Amplitudes: sqrt(prob) for each state, pad to include ancilla
    # State: sum_i sqrt(p_i)|i>|0> -> oracle -> sum_{i good} sqrt(p_i)|i>|1> + sum_{i bad} sqrt(p_i)|i>|0>
    # We need state_prep that creates sum_i sqrt(p_i)|i> on n qubits, then we add ancilla and oracle
    amps = np.sqrt(probs).astype(complex)

    qr_state = QuantumRegister(n, 'q')
    qr_ancilla = QuantumRegister(1, 'a')
    qc = QuantumCircuit(qr_state, qr_ancilla)

    # State preparation on state qubits
    sp = StatePreparation(amps)
    qc.append(sp, qr_state[:])

    # Oracle: flip ancilla when state in good set
    _add_cdf_oracle(qc, list(qr_state), qr_ancilla[0], good_indices)

    return qc, n


def estimate_cdf_qae(
    probs: np.ndarray,
    bin_centers: np.ndarray,
    threshold: float,
    epsilon_target: float = 0.01,
    alpha: float = 0.05,
    shots: int = 2000,
    backend: str = 'aer_simulator',
    max_grover_power: int = 8
) -> QAEResult:
    """
    Estimate CDF(L <= threshold) using Iterative Amplitude Estimation.

    Args:
        probs: Discretized probability vector
        bin_centers: Loss values at bin centers
        threshold: Loss threshold
        epsilon_target: Target precision
        alpha: Confidence level (1-alpha for interval)
        shots: Number of shots per measurement
        backend: Simulator backend
        max_grover_power: Max Grover iterations

    Returns:
        QAEResult with estimation and confidence interval
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required")

    qc, ancilla_idx = create_cdf_estimation_circuit(probs, bin_centers, threshold)

    # EstimationProblem: we want to estimate P(measure ancilla = 1)
    # The state after our circuit is sum_{good} sqrt(p)|s>|1> + sum_{bad} sqrt(p)|s>|0>
    # So P(ancilla=1) = sum_{good} p = CDF
    problem = EstimationProblem(
        state_preparation=qc,
        objective_qubits=[ancilla_idx],
    )

    sampler = Sampler(
        backend=AerSimulator(),
        options={"shots": shots}
    )

    iae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon_target,
        alpha=alpha,
        sampler=sampler,
    )

    result = iae.estimate(problem)

    conf_interval = (
        result.confidence_interval[0] if hasattr(result, 'confidence_interval') else result.estimation - epsilon_target,
        result.confidence_interval[1] if hasattr(result, 'confidence_interval') else result.estimation + epsilon_target
    )

    num_iter = getattr(result, 'num_iterations', 0)
    circ = iae.construct_circuit(problem, k=1) if hasattr(iae, 'construct_circuit') else qc
    depth = circ.depth() if hasattr(circ, 'depth') else 0

    return QAEResult(
        estimation=float(result.estimation),
        confidence_interval=tuple(float(x) for x in conf_interval),
        num_iterations=num_iter,
        circuit_depth=depth,
        num_shots=shots
    )


def estimate_cdf_classical(
    probs: np.ndarray,
    bin_centers: np.ndarray,
    threshold: float
) -> float:
    """Classical fallback for CDF estimation."""
    k = _bin_index_for_threshold(bin_centers, threshold)
    return float(np.sum(probs[: k + 1]))
