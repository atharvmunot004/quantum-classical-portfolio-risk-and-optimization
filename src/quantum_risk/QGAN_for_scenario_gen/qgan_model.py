"""
QGAN model components: Quantum Generator and Classical Discriminator.

Quantum Generator: Parametrized quantum circuit (PQC) that generates bitstrings
Classical Discriminator: Neural network that distinguishes real from generated data
"""
import numpy as np
from typing import Dict, Optional, Tuple
import warnings

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn("Qiskit not available. QGAN will use classical fallback.")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Discriminator will use numpy-based implementation.")


class QuantumGenerator:
    """
    Quantum Generator using a parametrized quantum circuit.
    
    The generator maps latent bitstrings to probability distributions over
    measurement outcomes, which are then mapped to continuous return values.
    """
    
    def __init__(
        self,
        num_qubits: int = 6,
        ansatz_layers: int = 3,
        entanglement: str = 'linear',
        rotation_gates: list = None,
        backend: str = 'aer_simulator',
        shots: int = 4096,
        seed: Optional[int] = None
    ):
        """
        Initialize quantum generator.
        
        Args:
            num_qubits: Number of qubits (determines output dimension)
            ansatz_layers: Number of ansatz layers
            entanglement: 'linear' or 'circular' entanglement pattern
            rotation_gates: List of rotation gates, e.g., ['ry', 'rz']
            backend: Qiskit backend name
            shots: Number of measurement shots
            seed: Random seed
        """
        self.use_classical_fallback = not QISKIT_AVAILABLE
        if self.use_classical_fallback:
            warnings.warn("Qiskit not available. Using classical fallback for generator.")
        
        self.num_qubits = num_qubits
        self.ansatz_layers = ansatz_layers
        self.entanglement = entanglement
        self.rotation_gates = rotation_gates or ['ry', 'rz']
        self.shots = shots
        self.seed = seed
        
        # Initialize backend
        if not self.use_classical_fallback:
            self.backend = AerSimulator()
        if seed is not None:
            np.random.seed(seed)
        
        # Create parameterized circuit
        self.params = None
        self.circuit = None
        if not self.use_classical_fallback:
            self._build_circuit()
        else:
            # Classical fallback: use random parameters
            num_params_per_layer = len(self.rotation_gates) * self.num_qubits
            if self.entanglement == 'linear':
                num_params_per_layer += (self.num_qubits - 1)
            else:
                num_params_per_layer += self.num_qubits
            total_params = self.ansatz_layers * num_params_per_layer
            self.params = [None] * total_params  # Placeholder
    
    def _build_circuit(self):
        """Build the hardware-efficient ansatz circuit."""
        if self.use_classical_fallback:
            return
        if not QISKIT_AVAILABLE:
            return
        self.circuit = QuantumCircuit(self.num_qubits)
        
        # Create parameters
        num_params_per_layer = len(self.rotation_gates) * self.num_qubits
        if self.entanglement == 'linear':
            num_params_per_layer += (self.num_qubits - 1)  # CNOT gates
        else:  # circular
            num_params_per_layer += self.num_qubits
        
        total_params = self.ansatz_layers * num_params_per_layer
        self.params = [Parameter(f'θ_{i}') for i in range(total_params)]
        
        param_idx = 0
        
        for layer in range(self.ansatz_layers):
            # Rotation gates
            for gate_type in self.rotation_gates:
                for qubit in range(self.num_qubits):
                    if gate_type == 'ry':
                        self.circuit.ry(self.params[param_idx], qubit)
                    elif gate_type == 'rz':
                        self.circuit.rz(self.params[param_idx], qubit)
                    elif gate_type == 'rx':
                        self.circuit.rx(self.params[param_idx], qubit)
                    param_idx += 1
            
            # Entanglement
            if self.entanglement == 'linear':
                for qubit in range(self.num_qubits - 1):
                    self.circuit.cx(qubit, qubit + 1)
            else:  # circular
                for qubit in range(self.num_qubits):
                    self.circuit.cx(qubit, (qubit + 1) % self.num_qubits)
        
        # Add measurement
        self.circuit.measure_all()
    
    def generate_samples(
        self,
        params: np.ndarray,
        num_samples: int = 1000
    ) -> np.ndarray:
        """
        Generate samples from the quantum generator.
        
        Args:
            params: Parameter values for the circuit
            num_samples: Number of samples to generate
            
        Returns:
            Array of bitstrings (integers 0 to 2^num_qubits - 1)
        """
        if self.use_classical_fallback:
            # Classical fallback: use parameter-dependent distribution
            # Simple approach: use params to create a biased distribution
            if len(params) > 0 and not np.all(np.isnan(params)):
                # Use first param to control distribution
                bias = np.tanh(params[0]) if not np.isnan(params[0]) else 0.0
                # Create a distribution biased by the parameter
                probs = np.ones(2**self.num_qubits) / (2**self.num_qubits)
                if abs(bias) > 0.01:
                    # Shift probability mass
                    center = int((2**self.num_qubits - 1) * (1 + bias) / 2)
                    center = np.clip(center, 0, 2**self.num_qubits - 1)
                    probs[center] *= (1 + abs(bias) * 10)
                    probs = probs / probs.sum()
                samples = np.random.choice(2**self.num_qubits, size=num_samples, p=probs)
            else:
                # Uniform random bitstrings
                samples = np.random.randint(0, 2**self.num_qubits, size=num_samples)
            return samples
        
        if not QISKIT_AVAILABLE or self.circuit is None:
            # Fallback if circuit wasn't built
            return np.random.randint(0, 2**self.num_qubits, size=num_samples)
        
        # Bind parameters
        bound_circuit = self.circuit.bind_parameters(params)
        
        # Execute circuit
        job = self.backend.run(bound_circuit, shots=num_samples, seed_simulator=self.seed)
        result = job.result()
        counts = result.get_counts()
        
        # Convert counts to samples
        samples = []
        for bitstring, count in counts.items():
            # Qiskit uses MSB first, convert to integer
            int_val = int(bitstring, 2)
            samples.extend([int_val] * count)
        
        # If we don't have enough samples, pad with random
        while len(samples) < num_samples:
            samples.append(np.random.randint(0, 2**self.num_qubits))
        
        return np.array(samples[:num_samples])
    
    def get_circuit_depth(self) -> int:
        """Get circuit depth."""
        if self.use_classical_fallback or self.circuit is None:
            return self.ansatz_layers * 2  # Approximate
        return self.circuit.depth()
    
    def get_num_parameters(self) -> int:
        """Get number of parameters."""
        if self.params is None:
            num_params_per_layer = len(self.rotation_gates) * self.num_qubits
            if self.entanglement == 'linear':
                num_params_per_layer += (self.num_qubits - 1)
            else:
                num_params_per_layer += self.num_qubits
            return self.ansatz_layers * num_params_per_layer
        return len(self.params) if self.params else 0


class ClassicalDiscriminator:
    """
    Classical neural network discriminator.
    
    Distinguishes between real and generated data samples.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_layers: list = None,
        activation: str = 'relu',
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
        seed: Optional[int] = None,
        use_gpu: bool = True
    ):
        """
        Initialize discriminator.
        
        Args:
            input_dim: Input dimension (1 for scalar returns)
            hidden_layers: List of hidden layer sizes, e.g., [64, 32]
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            dropout: Dropout rate
            label_smoothing: Label smoothing factor
            seed: Random seed
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [64, 32]
        self.activation = activation
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.seed = seed
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        
        if seed is not None:
            np.random.seed(seed)
            if TORCH_AVAILABLE:
                torch.manual_seed(seed)
        
        if TORCH_AVAILABLE:
            self._build_torch_model()
        else:
            self._build_numpy_model()
    
    def _build_torch_model(self):
        """Build PyTorch discriminator model with GPU support."""
        layers = []
        in_dim = self.input_dim
        
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        self.use_torch = True
        
        # GPU support
        self.device = None
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            if self.seed is not None:
                torch.cuda.manual_seed_all(self.seed)
        else:
            self.device = torch.device('cpu')
    
    def _build_numpy_model(self):
        """Build numpy-based discriminator (simplified)."""
        self.use_torch = False
        self.weights = []
        self.biases = []
        in_dim = self.input_dim
        
        for hidden_dim in self.hidden_layers:
            self.weights.append(np.random.randn(in_dim, hidden_dim) * 0.1)
            self.biases.append(np.zeros(hidden_dim))
            in_dim = hidden_dim
        
        self.weights.append(np.random.randn(in_dim, 1) * 0.1)
        self.biases.append(np.zeros(1))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through discriminator.
        
        Args:
            x: Input data (num_samples, input_dim)
            
        Returns:
            Probabilities (num_samples, 1) - probability of being real
        """
        if self.use_torch:
            x_tensor = torch.FloatTensor(x).to(self.device)
            with torch.no_grad():
                output = self.model(x_tensor).cpu().numpy()
            return output
        else:
            # Numpy implementation
            a = x
            for i, (w, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
                a = a @ w + b
                if self.activation == 'relu':
                    a = np.maximum(0, a)
                elif self.activation == 'tanh':
                    a = np.tanh(a)
                elif self.activation == 'sigmoid':
                    a = 1 / (1 + np.exp(-np.clip(a, -500, 500)))
            
            # Final layer
            a = a @ self.weights[-1] + self.biases[-1]
            a = 1 / (1 + np.exp(-np.clip(a, -500, 500)))
            return a
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict probability of being real."""
        return self.forward(x)
    
    def get_parameters(self) -> list:
        """Get model parameters for optimization."""
        if self.use_torch:
            return list(self.model.parameters())
        else:
            return self.weights + self.biases
    
    def set_parameters(self, params: list):
        """Set model parameters."""
        if self.use_torch:
            for param, new_val in zip(self.model.parameters(), params):
                param.data = torch.FloatTensor(new_val)
        else:
            # Split params into weights and biases
            idx = 0
            for i in range(len(self.hidden_layers)):
                dim_in = self.input_dim if i == 0 else self.hidden_layers[i-1]
                dim_out = self.hidden_layers[i]
                size = dim_in * dim_out
                self.weights[i] = params[idx:idx+size].reshape(dim_in, dim_out)
                idx += size
                self.biases[i] = params[idx:idx+self.hidden_layers[i]]
                idx += self.hidden_layers[i]
            
            # Final layer
            dim_in = self.hidden_layers[-1]
            size = dim_in
            self.weights[-1] = params[idx:idx+size].reshape(dim_in, 1)
            idx += size
            self.biases[-1] = params[idx:idx+1]
