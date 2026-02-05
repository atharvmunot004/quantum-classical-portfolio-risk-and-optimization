"""
Convert parquet files in cross_model directory to markdown tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent


def markdown_table(df: pd.DataFrame, float_format: str = "%.6f", max_rows: int = 1000) -> str:
    """Convert DataFrame to markdown table."""
    if df.empty:
        return "_No data available._"
    
    d = df.head(max_rows)
    cols = list(d.columns)
    
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    
    for _, row in d.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if np.isnan(v) or np.isinf(v):
                    v = ""
                else:
                    v = float_format % v
            cells.append(str(v)[:80])
        rows.append("| " + " | ".join(cells) + " |")
    
    return "\n".join([header, sep] + rows)


def convert_parquet_to_markdown(parquet_path: Path) -> None:
    """Convert a parquet file to markdown."""
    if not parquet_path.exists():
        print(f"Warning: File not found: {parquet_path}")
        return
    
    print(f"Reading {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)
    print(f"  Shape: {df.shape}")
    
    # Create markdown file
    md_path = parquet_path.with_suffix(".md")
    
    # Generate markdown content
    title = parquet_path.stem.replace("_", " ").title()
    content = [
        f"# {title}",
        "",
        f"**Source:** `{parquet_path.name}`",
        f"**Rows:** {len(df)}",
        f"**Columns:** {len(df.columns)}",
        "",
        "## Data Table",
        "",
        markdown_table(df, max_rows=1000),
        "",
    ]
    
    # Add column descriptions if available
    content.extend([
        "## Column Descriptions",
        "",
    ])
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        content.append(f"- **{col}** ({dtype}): {non_null} non-null values")
    
    md_path.write_text("\n".join(content), encoding="utf-8")
    print(f"  Written: {md_path}")


def main() -> int:
    """Main execution function."""
    parquet_files = [
        THIS_DIR / "cross_model_summary.parquet",
        THIS_DIR / "cross_model_rankings.parquet",
        THIS_DIR / "cross_model_time_sliced_summary.parquet",
    ]
    
    print("Converting parquet files to markdown...")
    for parquet_path in parquet_files:
        convert_parquet_to_markdown(parquet_path)
    
    print("\nConversion complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
