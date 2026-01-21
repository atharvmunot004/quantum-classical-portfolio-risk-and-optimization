"""
Script to update the results schema JSON file with actual row counts and columns
from the existing varcov parquet files.
"""
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any


def read_parquet_info(file_path: Path) -> Dict[str, Any]:
    """
    Read parquet file and return row count and column names.
    
    Args:
        file_path: Path to parquet file
        
    Returns:
        Dictionary with 'rows' and 'columns' keys
    """
    if not file_path.exists():
        return {'rows': 0, 'columns': []}
    
    try:
        df = pd.read_parquet(file_path)
        return {
            'rows': len(df),
            'columns': list(df.columns)
        }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {'rows': 0, 'columns': []}


def update_results_scheme(
    schema_path: Path,
    project_root: Path
) -> None:
    """
    Update the results schema JSON with actual data from parquet files.
    
    Args:
        schema_path: Path to the schema JSON file
        project_root: Path to project root directory
    """
    # Load existing schema
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    
    # Update risk_series
    if 'risk_series' in schema:
        risk_series_path = project_root / schema['risk_series']['path']
        risk_info = read_parquet_info(risk_series_path)
        schema['risk_series']['rows'] = risk_info['rows']
        if risk_info['columns']:
            schema['risk_series']['columns'] = risk_info['columns']
        print(f"Updated risk_series: {risk_info['rows']} rows, {len(risk_info['columns'])} columns")
    
    # Update metrics
    if 'metrics' in schema:
        metrics_path = project_root / schema['metrics']['path']
        metrics_info = read_parquet_info(metrics_path)
        schema['metrics']['rows'] = metrics_info['rows']
        if metrics_info['columns']:
            schema['metrics']['columns'] = metrics_info['columns']
        print(f"Updated metrics: {metrics_info['rows']} rows, {len(metrics_info['columns'])} columns")
    
    # Update time_sliced_metrics
    if 'time_sliced_metrics' in schema:
        time_sliced_path = project_root / schema['time_sliced_metrics']['path']
        time_sliced_info = read_parquet_info(time_sliced_path)
        schema['time_sliced_metrics']['rows'] = time_sliced_info['rows']
        if time_sliced_info['columns']:
            schema['time_sliced_metrics']['columns'] = time_sliced_info['columns']
        print(f"Updated time_sliced_metrics: {time_sliced_info['rows']} rows, {len(time_sliced_info['columns'])} columns")
    
    # Update parameter_store
    if 'parameter_store' in schema:
        param_path = project_root / schema['parameter_store']['path']
        param_info = read_parquet_info(param_path)
        schema['parameter_store']['rows'] = param_info['rows']
        if param_info['columns']:
            schema['parameter_store']['columns'] = param_info['columns']
        print(f"Updated parameter_store: {param_info['rows']} rows, {len(param_info['columns'])} columns")
    
    # Save updated schema
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    
    print(f"\nSchema updated successfully: {schema_path}")


def main():
    """Main entry point."""
    # Get project root (4 levels up from this file)
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    
    # Schema file path
    schema_path = project_root / "results/classical_risk/varcov_asset_level_results_scheme.json"
    
    if not schema_path.exists():
        print(f"Schema file not found: {schema_path}")
        return
    
    print("Updating results schema...")
    print(f"Project root: {project_root}")
    print(f"Schema path: {schema_path}\n")
    
    update_results_scheme(schema_path, project_root)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

