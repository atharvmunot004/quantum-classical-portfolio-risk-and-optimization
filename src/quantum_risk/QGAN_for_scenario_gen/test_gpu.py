"""Test GPU support and metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.quantum_risk.QGAN_for_scenario_gen.qgan_model import ClassicalDiscriminator
import numpy as np

print("Testing GPU support...")
disc = ClassicalDiscriminator(input_dim=1, hidden_layers=[32, 16], use_gpu=True)
print(f'Device: {disc.device if hasattr(disc, "device") else "CPU"}')
x = np.random.randn(100, 1)
out = disc.forward(x)
print(f'Output shape: {out.shape}, Mean: {out.mean():.4f}')
print("GPU test passed!")
