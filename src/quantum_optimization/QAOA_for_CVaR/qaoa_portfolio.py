"""
QAOA Portfolio Optimization Module.

Implements QAOA for CVaR-based portfolio selection with CVaR-of-energy objective.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import warnings

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from scipy.optimize import minimize
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn("Qiskit not available. Using classical optimization fallback.")


@dataclass
class QAOAResult:
    """Result of QAOA optimization."""
    best_energy: float
    best_parameters: List[float]
    best_solution: np.ndarray
    samples: List[Tuple[np.ndarray, float]]
    nfev: int
    optimizer_success: bool
    circuit_depth: int
    circuit_width: int
    reps: int
    shots: int
    alpha: float
    total_time_ms: float


def qubo_to_ising(Q: np.ndarray, constant: float = 0.0):
    """
    Convert QUBO matrix to Ising Hamiltonian.
    
    QUBO: minimize x^T Q x + c
    Ising: minimize sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j
    
    Args:
        Q: QUBO matrix
        constant: Constant term
        
    Returns:
        Tuple of (SparsePauliOp, constant_offset)
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for QAOA")
    
    n = Q.shape[0]
    
    # Convert binary x_i to spin: Z_i = 1 - 2*x_i, so x_i = (1 - Z_i) / 2
    # x^T Q x = sum_i Q_ii x_i + sum_{i<j} 2*Q_ij x_i x_j
    # = sum_i Q_ii (1-Z_i)/2 + sum_{i<j} 2*Q_ij (1-Z_i)(1-Z_j)/4
    # = ... (simplified)
    
    # Linear terms: h_i
    h = np.zeros(n)
    for i in range(n):
        h[i] = Q[i, i] / 2
        for j in range(n):
            if i != j:
                h[i] += Q[i, j] / 4
    
    # Quadratic terms: J_{ij}
    J = {}
    for i in range(n):
        for j in range(i + 1, n):
            if abs(Q[i, j]) > 1e-10:
                J[(i, j)] = Q[i, j] / 4
    
    # Build SparsePauliOp
    pauli_list = []
    coeffs = []
    
    # Linear terms: h_i * Z_i
    for i in range(n):
        if abs(h[i]) > 1e-10:
            pauli_str = ['I'] * n
            pauli_str[i] = 'Z'
            pauli_list.append(''.join(pauli_str))
            coeffs.append(h[i])
    
    # Quadratic terms: J_{ij} * Z_i Z_j
    for (i, j), coeff in J.items():
        if abs(coeff) > 1e-10:
            pauli_str = ['I'] * n
            pauli_str[i] = 'Z'
            pauli_str[j] = 'Z'
            pauli_list.append(''.join(pauli_str))
            coeffs.append(coeff)
    
    if not pauli_list:
        # Identity operator with zero coefficient
        pauli_list = ['I' * n]
        coeffs = [0.0]
    
    hamiltonian = SparsePauliOp(pauli_list, coeffs)
    
    # Constant offset
    constant_offset = constant + np.sum(Q.diagonal()) / 4 + np.sum(Q) / 8
    
    return hamiltonian, constant_offset


def _compute_cvar_from_energies(energies: np.ndarray, alpha: float) -> float:
    """
    Compute CVaR from energy values.
    
    CVaR is the mean of the worst (1-alpha) fraction of energies.
    
    Args:
        energies: Array of energy values
        alpha: Confidence level
        
    Returns:
        CVaR value
    """
    if len(energies) == 0:
        return float('inf')
    
    sorted_energies = np.sort(energies)
    tail_size = int((1 - alpha) * len(sorted_energies))
    
    if tail_size == 0:
        tail_size = 1
    
    tail_energies = sorted_energies[-tail_size:]
    cvar = np.mean(tail_energies)
    
    return float(cvar)


def _bitstring_to_index(bitstring: str) -> int:
    """Convert bitstring to integer index."""
    return int(bitstring[::-1], 2)


def run_qaoa_optimization(
    Q: np.ndarray,
    constant: float = 0.0,
    reps: int = 1,
    shots: int = 5000,
    alpha: float = 0.95,
    maxiter: int = 250,
    tol: float = 1e-4,
    seed: int = 42,
    top_n: int = 200
) -> QAOAResult:
    """
    Run QAOA optimization with CVaR-of-energy objective.
    
    Args:
        Q: QUBO matrix
        constant: Constant term
        reps: Number of QAOA layers
        shots: Number of measurement shots
        alpha: CVaR confidence level
        maxiter: Maximum optimizer iterations
        tol: Optimizer tolerance
        seed: Random seed
        top_n: Number of top samples to return
        
    Returns:
        QAOAResult with optimization results
    """
    start_time = time.time()
    
    if not QISKIT_AVAILABLE:
        # Fallback to classical optimization
        return _classical_fallback(Q, constant, seed, top_n)
    
    n = Q.shape[0]
    
    # Convert QUBO to Ising
    cost_h, constant_offset = qubo_to_ising(Q, constant)
    
    # Mixer Hamiltonian: sum of X operators
    mixer_h = SparsePauliOp.from_list([('X' * n, 1.0)])
    
    # Create QAOA ansatz
    ansatz = QAOAAnsatz(cost_h, reps=reps, mixer_operator=mixer_h)
    n_params = ansatz.num_parameters
    
    # Backend
    backend = AerSimulator()
    qc_measure = ansatz.copy()
    qc_measure.measure_all()
    
    # Energy computation function
    def compute_energies(params: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        """Compute energies for given parameters."""
        # Use assign_parameters for newer Qiskit versions, fallback to bind_parameters
        try:
            bound = ansatz.assign_parameters(params)
        except AttributeError:
            # Fallback for older Qiskit versions
            bound = ansatz.bind_parameters(params)
        job = backend.run(bound, shots=shots, seed_simulator=seed)
        result = job.result()
        counts = result.get_counts()
        
        # Compute energies for each bitstring
        energies = []
        bitstring_energies = {}
        
        for bitstr, count in counts.items():
            # Convert bitstring to solution
            solution = np.array([int(b) for b in bitstr[::-1]])
            
            # Compute energy: x^T Q x + c
            energy = solution.T @ Q @ solution + constant
            
            energies.extend([energy] * count)
            bitstring_energies[bitstr] = energy
        
        return np.array(energies), bitstring_energies
    
    # CVaR objective function
    def objective(params: np.ndarray) -> float:
        """CVaR-of-energy objective."""
        energies, _ = compute_energies(params)
        cvar = _compute_cvar_from_energies(energies, alpha)
        return cvar
    
    # Optimize
    np.random.seed(seed)
    x0 = np.zeros(n_params)
    
    res = minimize(
        objective,
        x0,
        method="COBYLA",
        options={"maxiter": maxiter, "tol": tol, "disp": False},
    )
    
    # Get final samples
    final_energies, final_bitstring_energies = compute_energies(res.x)
    
    # Sort samples by energy
    sorted_samples = sorted(
        final_bitstring_energies.items(),
        key=lambda x: x[1]
    )[:top_n]
    
    samples = []
    for bitstr, energy in sorted_samples:
        solution = np.array([int(b) for b in bitstr[::-1]])
        samples.append((solution, energy))
    
    best_solution, best_energy = samples[0] if samples else (np.zeros(n), float('inf'))
    
    total_time_ms = (time.time() - start_time) * 1000
    
    return QAOAResult(
        best_energy=best_energy,
        best_parameters=res.x.tolist(),
        best_solution=best_solution,
        samples=samples,
        nfev=res.nfev,
        optimizer_success=bool(res.success),
        circuit_depth=ansatz.depth(),
        circuit_width=n,
        reps=reps,
        shots=shots,
        alpha=alpha,
        total_time_ms=total_time_ms
    )


def _classical_fallback(
    Q: np.ndarray,
    constant: float,
    seed: int,
    top_n: int
) -> QAOAResult:
    """
    Classical fallback when Qiskit is not available.
    
    Uses random search to find good solutions.
    """
    np.random.seed(seed)
    n = Q.shape[0]
    
    # Random search
    samples = []
    for _ in range(1000):
        solution = np.random.randint(0, 2, size=n)
        energy = solution.T @ Q @ solution + constant
        samples.append((solution, energy))
    
    # Sort by energy
    samples.sort(key=lambda x: x[1])
    samples = samples[:top_n]
    
    best_solution, best_energy = samples[0] if samples else (np.zeros(n), float('inf'))
    
    return QAOAResult(
        best_energy=best_energy,
        best_parameters=[],
        best_solution=best_solution,
        samples=samples,
        nfev=1000,
        optimizer_success=True,
        circuit_depth=0,
        circuit_width=n,
        reps=0,
        shots=0,
        alpha=0.95,
        total_time_ms=0.0
    )
