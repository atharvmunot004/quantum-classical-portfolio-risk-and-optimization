"""Convert parquet files to markdown tables (.md)."""
from pathlib import Path
import pandas as pd
import numpy as np

def df_to_markdown(df: pd.DataFrame, max_cell_len: int = 50) -> str:
    if df.empty:
        return "_No data._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                v = ""
            cells.append(str(v)[:max_cell_len])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def convert_parquet_to_md(parquet_path: Path, md_path: Path | None = None, title: str | None = None) -> None:
    df = pd.read_parquet(parquet_path)
    out = md_path or parquet_path.with_suffix(".md")
    title = title or out.stem.replace("_", " ").title()
    body = df_to_markdown(df)
    content = f"# {title}\n\n{body}\n"
    out.write_text(content, encoding="utf-8")
    print(f"Wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    base = Path(__file__).parent
    # GARCH
    convert_parquet_to_md(base / "garch" / "garch_asset_level_time_sliced_analysis.parquet")
    # EVT-POT
    for name in (
        "evt_asset_level_analysis_summary.parquet",
        "evt_asset_level_time_sliced_analysis.parquet",
        "evt_asset_level_top_configs.parquet",
    ):
        p = base / "evt_pot" / name
        if p.exists():
            convert_parquet_to_md(p)
