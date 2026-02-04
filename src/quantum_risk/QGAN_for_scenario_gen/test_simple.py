"""Simple test to verify QGAN training works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from src.quantum_risk.QGAN_for_scenario_gen.qgan_training import QGANTrainer
from src.quantum_risk.QGAN_for_scenario_gen.qgan_model import QuantumGenerator, ClassicalDiscriminator
from src.quantum_risk.QGAN_for_scenario_gen.discretization import create_uniform_grid

print("Testing QGAN training...")
data = np.random.randn(300)
grid = create_uniform_grid(data, num_bins=64)
gen = QuantumGenerator(num_qubits=6, ansatz_layers=3, seed=42)
disc = ClassicalDiscriminator(input_dim=1, hidden_layers=[32, 16], seed=42)
trainer = QGANTrainer(gen, disc, grid, {
    'optimizer_generator': {'learning_rate': 0.02},
    'optimizer_discriminator': {'learning_rate': 0.01}
}, seed=42)
result = trainer.train(data, max_iterations=5, batch_size=50, verbose=True)
print(f'Training completed: loss={result["final_generator_loss"]:.4f}')
