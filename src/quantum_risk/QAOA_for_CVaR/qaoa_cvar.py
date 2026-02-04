"""
QAOA with CVaR objective for asset-level tail-risk estimation.

Uses CVaR of measurement outcomes (mean of worst alpha fraction of energies)
as the optimization objective instead of expectation value.
"""
import numpy as np
import time
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from scipy.optimize import minimize
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from .hamiltonian import build_cost_hamiltonian_from_discretized_loss, build_x_mixer
from .loss_discretization import discretize_losses_quantile_grid, DiscretizationResult


@dataclass
class QAOACVaRResult:
    """Result of QAOA CVaR optimization."""

    cvar: float
    best_energy: float
    best_parameters: List[float]
    nfev: int
    optimizer_success: bool
    circuit_depth: int
    circuit_width: int
    reps: int
    shots: int
    alpha: float
    total_time_ms: float
    cvar_value_trace: Optional[List[float]] = None
    energy_trace: Optional[List[float]] = None


def _compute_cvar_from_energies(energies: np.ndarray, alpha: float) -> float:
    """
    CVaR_alpha = mean of worst (1-alpha) fraction of outcomes.

    For confidence level alpha (e.g. 0.95): take the worst 5% of outcomes.
    Sort ascending; worst are highest. Take last ceil((1-alpha)*n) and average.
    """
    if len(energies) == 0:
        return float(np.nan)
    n = len(energies)
    k = max(1, int(np.ceil((1 - alpha) * n)))
    sorted_e = np.sort(energies)
    worst = sorted_e[-k:]
    return float(np.mean(worst))


def _bitstring_to_index(bitstring: str) -> int:
    """Convert Qiskit bitstring (MSB first) to integer index."""
    return int(bitstring, 2)


def _compute_classical_cvar(losses: np.ndarray, alpha: float) -> float:
    """Classical empirical CVaR: mean of worst (1-alpha) fraction of losses."""
    n = len(losses)
    k = max(1, int(np.ceil((1 - alpha) * n)))
    sorted_l = np.sort(losses)
    worst = sorted_l[-k:]
    return float(np.mean(worst))


def _run_qaoa_cvar_single(
    losses: np.ndarray,
    num_qubits: int,
    reps: int,
    alpha: float,
    shots: int,
    maxiter: int,
    tol: float,
    seed: int,
    discretization_config: Optional[Dict] = None,
) -> Tuple[float, QAOACVaRResult]:
    """
    Run QAOA with CVaR objective for a single window of losses.

    Returns (cvar_estimate, result).
    When Qiskit is not available, falls back to classical empirical CVaR.
    """
    if not QISKIT_AVAILABLE:
        cvar_classical = _compute_classical_cvar(losses, alpha)
        return cvar_classical, QAOACVaRResult(
            cvar=cvar_classical,
            best_energy=float(np.mean(losses)),
            best_parameters=[],
            nfev=0,
            optimizer_success=True,
            circuit_depth=0,
            circuit_width=num_qubits,
            reps=reps,
            shots=0,
            alpha=alpha,
            total_time_ms=0.0,
        )

    discretization_config = discretization_config or {}
    start_total = time.time()

    disc = discretize_losses_quantile_grid(
        losses,
        num_levels=1 << num_qubits,
        clip_quantiles=tuple(discretization_config.get('clip_quantiles', [0.001, 0.999])),
        rescale_to_unit_interval=discretization_config.get('rescale_to_unit_interval', True),
    )
    cost_h = build_cost_hamiltonian_from_discretized_loss(disc, num_qubits)
    mixer_h = build_x_mixer(num_qubits)

    ansatz = QAOAAnsatz(cost_h, reps=reps, mixer_operator=mixer_h)
    n_params = ansatz.num_parameters

    loss_levels = disc.loss_levels
    backend = AerSimulator()
    qc_measure = ansatz.copy()
    qc_measure.measure_all()

    energy_trace: List[float] = []
    cvar_trace: List[float] = []

    def objective(params: np.ndarray) -> float:
        bound = qc_measure.assign_parameters(params)
        job = backend.run(bound, shots=shots, seed_simulator=seed)
        result = job.result()
        counts = result.get_counts()
        if not counts:
            return float(np.inf)

        energies = []
        for bitstr, count in counts.items():
            idx = _bitstring_to_index(bitstr)
            if idx < len(loss_levels):
                e = loss_levels[idx]
            else:
                e = loss_levels[-1]
            energies.extend([e] * count)
        energies = np.array(energies)
        cvar_val = _compute_cvar_from_energies(energies, alpha)
        cvar_trace.append(cvar_val)
        energy_trace.append(float(np.mean(energies)))
        return cvar_val

    np.random.seed(seed)
    x0 = np.zeros(n_params)
    res = minimize(
        objective,
        x0,
        method="COBYLA",
        options={"maxiter": maxiter, "tol": tol, "disp": False},
    )

    total_time_ms = (time.time() - start_total) * 1000

    final_cvar_unit = res.fun if res.success else float(np.nan)
    if disc.rescale_to_unit and not np.isnan(final_cvar_unit):
        span = disc.support_high - disc.support_low
        final_cvar = final_cvar_unit * span + disc.support_low
    else:
        final_cvar = final_cvar_unit
    best_energy = energy_trace[-1] if energy_trace else float(np.nan)

    qaoa_result = QAOACVaRResult(
        cvar=final_cvar,
        best_energy=best_energy,
        best_parameters=res.x.tolist(),
        nfev=res.nfev,
        optimizer_success=bool(res.success),
        circuit_depth=ansatz.depth(),
        circuit_width=num_qubits,
        reps=reps,
        shots=shots,
        alpha=alpha,
        total_time_ms=total_time_ms,
        cvar_value_trace=cvar_trace if cvar_trace else None,
        energy_trace=energy_trace if energy_trace else None,
    )
    return final_cvar, qaoa_result


def run_rolling_qaoa_cvar(
    losses: np.ndarray,
    window: int,
    confidence_level: float,
    step_size: int = 1,
    num_qubits: int = 8,
    reps: int = 1,
    shots: int = 4000,
    maxiter: int = 200,
    tol: float = 1e-4,
    seed: int = 42,
    discretization_config: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Compute rolling CVaR time series using QAOA with CVaR objective.

    Args:
        losses: 1D array of loss values
        window: Rolling window size
        confidence_level: e.g. 0.95, 0.99 (alpha = confidence_level for CVaR tail)
        step_size: Roll step
        num_qubits: Number of qubits for discretization
        reps: QAOA reps
        shots: Measurement shots per evaluation
        maxiter: Optimizer max iterations
        tol: Optimizer tolerance
        seed: Random seed
        discretization_config: Optional discretization config

    Returns:
        (cvar_series, best_energy_series, param_records)
    """
    n = len(losses)
    cvar_series = np.full(n, np.nan)
    best_energy_series = np.full(n, np.nan)
    param_records: List[Dict] = []

    for i in range(window - 1, n, step_size):
        window_losses = losses[i - window + 1 : i + 1]
        window_losses = window_losses[~np.isnan(window_losses)]
        if len(window_losses) < window // 2:
            continue
        try:
            cvar_val, qaoa_res = _run_qaoa_cvar_single(
                window_losses,
                num_qubits=num_qubits,
                reps=reps,
                alpha=confidence_level,
                shots=shots,
                maxiter=maxiter,
                tol=tol,
                seed=seed,
                discretization_config=discretization_config,
            )
            cvar_series[i] = cvar_val
            best_energy_series[i] = qaoa_res.best_energy
            param_records.append({
                "index": i,
                "best_energy": qaoa_res.best_energy,
                "best_parameters": qaoa_res.best_parameters,
                "optimizer_success": qaoa_res.optimizer_success,
                "nfev": qaoa_res.nfev,
                "total_time_ms": qaoa_res.total_time_ms,
                "circuit_depth": qaoa_res.circuit_depth,
                "circuit_width": qaoa_res.circuit_width,
            })
        except Exception:
            pass
    return cvar_series, best_energy_series, param_records
