"""
Fast test script for QGAN with reduced parameters.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.quantum_risk.QGAN_for_scenario_gen.main import evaluate_qgan_scenarios
import json

# Load config and modify for faster testing
config_path = Path(__file__).parent / "llm.json"
with open(config_path) as f:
    config = json.load(f)

# Reduce parameters for faster testing
config['qgan_settings']['optimization']['max_iterations'] = 20
config['qgan_settings']['scenario_generation']['num_scenarios_per_timestamp'] = 500
config['qgan_settings']['rolling']['estimation_windows'] = [252]
config['qgan_settings']['rolling']['step_size'] = 100  # Process every 100 days instead of 5

print("=" * 80)
print("FAST QGAN TEST - Reduced Parameters")
print("=" * 80)
print(f"Max iterations: {config['qgan_settings']['optimization']['max_iterations']}")
print(f"Scenarios per timestamp: {config['qgan_settings']['scenario_generation']['num_scenarios_per_timestamp']}")
print(f"Step size: {config['qgan_settings']['rolling']['step_size']}")
print("=" * 80)

risk_df, metrics_df, scenarios_df, ts_df = evaluate_qgan_scenarios(config_dict=config)

print(f"\n✅ Completed!")
print(f"Risk records: {len(risk_df)}")
print(f"Metrics records: {len(metrics_df)}")
print(f"Scenarios: {len(scenarios_df)}")
print(f"Time-sliced: {len(ts_df)}")
