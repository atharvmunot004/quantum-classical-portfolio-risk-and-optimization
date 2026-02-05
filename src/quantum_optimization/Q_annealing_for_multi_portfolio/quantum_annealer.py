"""
Quantum Annealing Solver Module.

Provides interface to quantum annealers (D-Wave) and simulated annealers.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import time
import warnings

# Try to import D-Wave libraries
try:
    from dwave.system import DWaveSampler, EmbeddingComposite
    from dimod import BinaryQuadraticModel
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    warnings.warn("D-Wave libraries not available. Using simulated annealing.")

# Try to import simulated annealing
try:
    from neal import SimulatedAnnealingSampler
    NEAL_AVAILABLE = True
except ImportError:
    NEAL_AVAILABLE = False
    warnings.warn("Neal (simulated annealing) not available. Using scipy-based solver.")


class QuantumAnnealer:
    """
    Quantum annealing solver for QUBO problems.
    
    Supports both D-Wave quantum annealers and simulated annealing fallback.
    """
    
    def __init__(
        self,
        backend: str = 'dwave_or_simulated_annealer',
        num_reads: int = 5000,
        annealing_time_us: int = 20,
        chain_strength: Optional[float] = None,
        auto_scale: bool = True,
        optimize_embedding: bool = True,
        embedding_retries: int = 5,
        random_seed: Optional[int] = None
    ):
        """
        Initialize quantum annealer.
        
        Args:
            backend: Backend type ('dwave', 'simulated', 'dwave_or_simulated_annealer')
            num_reads: Number of reads/samples
            annealing_time_us: Annealing time in microseconds
            chain_strength: Chain strength (None for auto)
            auto_scale: Whether to auto-scale QUBO
            optimize_embedding: Whether to optimize embedding
            embedding_retries: Number of embedding retries
            random_seed: Random seed for reproducibility
        """
        self.backend = backend
        self.num_reads = num_reads
        self.annealing_time_us = annealing_time_us
        self.chain_strength = chain_strength
        self.auto_scale = auto_scale
        self.optimize_embedding = optimize_embedding
        self.embedding_retries = embedding_retries
        self.random_seed = random_seed
        
        # Initialize sampler
        self.sampler = None
        self._initialize_sampler()
    
    def _initialize_sampler(self):
        """Initialize the appropriate sampler."""
        if self.backend == 'dwave' or self.backend == 'dwave_or_simulated_annealer':
            if DWAVE_AVAILABLE:
                try:
                    # Try to connect to D-Wave
                    dwave_sampler = DWaveSampler()
                    self.sampler = EmbeddingComposite(dwave_sampler)
                    self.backend_type = 'dwave'
                    return
                except Exception as e:
                    warnings.warn(f"Could not connect to D-Wave: {e}. Falling back to simulated annealing.")
        
        # Fall back to simulated annealing
        if NEAL_AVAILABLE:
            self.sampler = SimulatedAnnealingSampler()
            self.backend_type = 'simulated'
        else:
            # Use scipy-based simulated annealing
            self.sampler = None
            self.backend_type = 'scipy_simulated'
    
    def _compute_chain_strength(self, Q: np.ndarray) -> float:
        """
        Compute chain strength for embedding.
        
        Args:
            Q: QUBO matrix
            
        Returns:
            Chain strength value
        """
        if self.chain_strength is not None:
            return self.chain_strength
        
        # Auto-compute chain strength
        # Use maximum absolute value of Q matrix elements
        max_abs = np.abs(Q).max()
        chain_strength = max(1.0, max_abs * 1.5)
        
        return chain_strength
    
    def solve(
        self,
        Q: np.ndarray,
        constant: float = 0.0
    ) -> Dict:
        """
        Solve QUBO problem using quantum/simulated annealing.
        
        Args:
            Q: QUBO matrix (n x n)
            constant: Constant term in QUBO
            
        Returns:
            Dictionary with solution information:
            - samples: List of sample dictionaries
            - energies: Array of energy values
            - best_solution: Best binary solution
            - best_energy: Best energy value
            - num_occurrences: Number of times each solution was found
            - chain_break_fraction: Fraction of chain breaks (for D-Wave)
            - embedding_size: Size of embedding (for D-Wave)
            - logical_to_physical_ratio: Ratio of logical to physical qubits
            - runtime_ms: Runtime in milliseconds
        """
        start_time = time.time()
        
        n = Q.shape[0]
        
        # Build BinaryQuadraticModel if using D-Wave or Neal
        if self.backend_type in ['dwave', 'simulated'] and self.sampler is not None:
            # Convert QUBO to BQM format
            bqm = BinaryQuadraticModel.empty('BINARY')
            
            # Add linear terms (diagonal)
            for i in range(n):
                bqm.add_variable(i, Q[i, i])
            
            # Add quadratic terms (upper triangle)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(Q[i, j]) > 1e-10:
                        bqm.add_interaction(i, j, Q[i, j])
            
            # Solve
            if self.backend_type == 'dwave':
                # D-Wave specific parameters
                response = self.sampler.sample(
                    bqm,
                    num_reads=self.num_reads,
                    annealing_time=self.annealing_time_us,
                    chain_strength=self._compute_chain_strength(Q),
                    auto_scale=self.auto_scale,
                    return_embedding=self.optimize_embedding
                )
            else:
                # Simulated annealing
                response = self.sampler.sample(
                    bqm,
                    num_reads=self.num_reads,
                    seed=self.random_seed
                )
            
            # Extract results
            samples = []
            energies = []
            num_occurrences = []
            
            for record in response.record:
                sample = np.zeros(n, dtype=int)
                for i in range(n):
                    sample[i] = record.sample[i]
                samples.append(sample)
                energies.append(record.energy + constant)
                num_occurrences.append(record.num_occurrences)
            
            energies = np.array(energies)
            num_occurrences = np.array(num_occurrences)
            
            # Find best solution
            best_idx = np.argmin(energies)
            best_solution = samples[best_idx]
            best_energy = energies[best_idx]
            
            # Extract D-Wave specific metrics
            chain_break_fraction = None
            embedding_size = None
            logical_to_physical_ratio = None
            
            if self.backend_type == 'dwave' and hasattr(response, 'info'):
                info = response.info
                if 'embedding' in info:
                    embedding = info['embedding']
                    embedding_size = len(embedding)
                    logical_to_physical_ratio = n / embedding_size if embedding_size > 0 else None
                
                if 'chain_break_fraction' in info:
                    chain_break_fraction = info['chain_break_fraction']
        
        else:
            # Fallback: scipy-based simulated annealing
            samples, energies, num_occurrences = self._scipy_simulated_annealing(Q, constant)
            best_idx = np.argmin(energies)
            best_solution = samples[best_idx]
            best_energy = energies[best_idx]
            chain_break_fraction = None
            embedding_size = None
            logical_to_physical_ratio = None
        
        runtime_ms = (time.time() - start_time) * 1000
        
        return {
            'samples': samples,
            'energies': energies,
            'best_solution': best_solution,
            'best_energy': best_energy,
            'num_occurrences': num_occurrences,
            'chain_break_fraction': chain_break_fraction,
            'embedding_size': embedding_size,
            'logical_to_physical_ratio': logical_to_physical_ratio,
            'runtime_ms': runtime_ms,
            'backend_type': self.backend_type
        }
    
    def _scipy_simulated_annealing(
        self,
        Q: np.ndarray,
        constant: float
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """
        Simple simulated annealing using scipy (fallback).
        
        Args:
            Q: QUBO matrix
            constant: Constant term
            
        Returns:
            Tuple of (samples, energies, num_occurrences)
        """
        from scipy.optimize import minimize
        
        n = Q.shape[0]
        samples = []
        energies = []
        
        def qubo_energy(x):
            x_binary = (x > 0.5).astype(int)
            return x_binary.T @ Q @ x_binary + constant
        
        # Run multiple random starts
        np.random.seed(self.random_seed)
        
        for _ in range(self.num_reads):
            # Random initial guess
            x0 = np.random.random(n)
            
            # Minimize
            result = minimize(
                qubo_energy,
                x0,
                method='L-BFGS-B',
                bounds=[(0, 1)] * n,
                options={'maxiter': 100}
            )
            
            # Convert to binary
            solution = (result.x > 0.5).astype(int)
            energy = qubo_energy(solution)
            
            samples.append(solution)
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Count occurrences
        unique_samples = {}
        for i, sample in enumerate(samples):
            sample_tuple = tuple(sample)
            if sample_tuple not in unique_samples:
                unique_samples[sample_tuple] = []
            unique_samples[sample_tuple].append(i)
        
        num_occurrences = np.array([
            len(unique_samples[tuple(sample)]) for sample in samples
        ])
        
        return samples, energies, num_occurrences
