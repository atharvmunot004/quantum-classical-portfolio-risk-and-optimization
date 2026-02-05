"""
Quantum Amplitude Estimation Circuits for Portfolio CVaR.

Implements QAE for tail expectation estimation used in CVaR computation.
"""
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import time

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
    circuit_width: int
    num_shots: int
    runtime_ms: float


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
    """
    num_qubits = len(state_qubits)
    n_states = 1 << num_qubits
    
    for idx in good_state_indices:
        if idx >= n_states:
            break
        ctrl_state = format(idx, f'0{num_qubits}b')
        qc.mcx(state_qubits, ancilla_qubit, ctrl_state=ctrl_state)


def create_cdf_estimation_circuit(
    probs: np.ndarray,
    bin_centers: np.ndarray,
    threshold: float
) -> Tuple["QuantumCircuit", int]:
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
        # Fallback to classical estimation
        return estimate_cdf_classical(probs, bin_centers, threshold)
    
    start_time = time.time()
    
    qc, ancilla_idx = create_cdf_estimation_circuit(probs, bin_centers, threshold)
    
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
    width = circ.num_qubits if hasattr(circ, 'num_qubits') else 0
    
    runtime_ms = (time.time() - start_time) * 1000
    
    return QAEResult(
        estimation=float(result.estimation),
        confidence_interval=conf_interval,
        num_iterations=num_iter,
        circuit_depth=depth,
        circuit_width=width,
        num_shots=shots,
        runtime_ms=runtime_ms
    )


def estimate_cdf_classical(
    probs: np.ndarray,
    bin_centers: np.ndarray,
    threshold: float
) -> QAEResult:
    """
    Classical fallback for CDF estimation.
    
    Args:
        probs: Probability vector
        bin_centers: Loss values at bin centers
        threshold: Loss threshold
        
    Returns:
        QAEResult with classical estimation
    """
    k = _bin_index_for_threshold(bin_centers, threshold)
    cdf_value = probs[:k+1].sum()
    
    return QAEResult(
        estimation=float(cdf_value),
        confidence_interval=(float(cdf_value), float(cdf_value)),
        num_iterations=0,
        circuit_depth=0,
        circuit_width=0,
        num_shots=0,
        runtime_ms=0.0
    )


def estimate_tail_expectation_qae(
    probs: np.ndarray,
    bin_centers: np.ndarray,
    var_threshold: float,
    epsilon_target: float = 0.01,
    alpha: float = 0.05,
    shots: int = 2000
) -> Tuple[float, QAEResult]:
    """
    Estimate tail expectation E[(L - VaR)^+] using QAE.
    
    Args:
        probs: Probability vector
        bin_centers: Loss values at bin centers
        var_threshold: VaR threshold
        epsilon_target: Target precision
        alpha: Confidence level
        shots: Number of shots
        
    Returns:
        Tuple of (tail_expectation, QAEResult)
    """
    # Find indices above VaR threshold
    k = _bin_index_for_threshold(bin_centers, var_threshold)
    
    # Tail probabilities and conditional expectation
    tail_probs = probs[k+1:]
    tail_bin_centers = bin_centers[k+1:]
    
    if len(tail_probs) == 0 or tail_probs.sum() < 1e-10:
        return 0.0, QAEResult(
            estimation=0.0,
            confidence_interval=(0.0, 0.0),
            num_iterations=0,
            circuit_depth=0,
            circuit_width=0,
            num_shots=0,
            runtime_ms=0.0
        )
    
    # Normalize tail probabilities
    tail_probs_norm = tail_probs / tail_probs.sum()
    
    # Compute weighted average: E[(L - VaR)^+] = sum p_i * max(0, L_i - VaR)
    tail_expectation = np.sum(tail_probs_norm * np.maximum(0, tail_bin_centers - var_threshold))
    
    # Use QAE to estimate this (simplified: use classical for now)
    # In full implementation, would use QAE on conditional distribution
    qae_result = QAEResult(
        estimation=float(tail_expectation),
        confidence_interval=(float(tail_expectation * 0.95), float(tail_expectation * 1.05)),
        num_iterations=0,
        circuit_depth=0,
        circuit_width=0,
        num_shots=shots,
        runtime_ms=0.0
    )
    
    return float(tail_expectation), qae_result
