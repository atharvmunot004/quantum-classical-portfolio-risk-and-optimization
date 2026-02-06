"""
IEEE Access Result Analysis Generator.

Generates publication-ready summary tables (Markdown) and figures from quantum
optimization tool outputs using llm.json configuration.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

# Optional matplotlib for figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_config(config_path: Path) -> dict:
    """Load llm.json configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_data_path(project_root: Path, path_key: str, data_sources: dict) -> Path | None:
    """Resolve parquet path; try exact name then qa_* fallback in same dir."""
    raw = data_sources.get(path_key)
    if not raw:
        return None
    base = project_root / raw
    if base.exists():
        return base
    # Fallback: e.g. q_annealing_portfolio_metrics.parquet -> qa_portfolio_metrics.parquet
    parent = base.parent
    name = base.name
    if "q_annealing_" in name:
        fallback_name = name.replace("q_annealing_", "qa_")
        fallback = parent / fallback_name
        if fallback.exists():
            return fallback
    return base  # return original so caller sees missing file


def load_sources(project_root: Path, data_sources: dict) -> dict[str, pd.DataFrame]:
    """Load all parquet sources; missing files yield empty DataFrame."""
    out = {}
    for key, rel_path in data_sources.items():
        path = resolve_data_path(project_root, key, data_sources)
        if path is None:
            out[key] = pd.DataFrame()
            continue
        if not path.exists():
            out[key] = pd.DataFrame()
            continue
        try:
            out[key] = pd.read_parquet(path)
        except Exception:
            out[key] = pd.DataFrame()
    return out


def _agg_func(agg: str):
    if agg == "mean":
        return "mean"
    if agg == "std":
        return "std"
    if agg == "p95":
        return lambda x: x.quantile(0.95)
    return "mean"


def build_summary_table(
    df: pd.DataFrame,
    group_by: list[str],
    metrics: list[str],
    aggregation: list[str],
    rounding: dict,
    drop_nan_rows: bool,
    sort_by: str | None,
) -> pd.DataFrame:
    """Build one summary table with group_by and metric aggregations."""
    if df.empty:
        return pd.DataFrame()
    decimals = rounding.get("decimals", 4)
    available = [c for c in metrics if c in df.columns]
    if not available:
        return pd.DataFrame()
    group_cols = [c for c in group_by if c in df.columns]
    if not group_cols:
        spec = {m: [(a, _agg_func(a)) for a in aggregation] for m in available}
        agg_df = df[available].agg(spec)
        if isinstance(agg_df.columns, pd.MultiIndex):
            agg_df.columns = [f"{a}_{b}" for a, b in agg_df.columns]
        agg_df = agg_df.T
        agg_df.index.name = "metric"
        return agg_df.round(decimals)
    spec = {m: [(a, _agg_func(a)) for a in aggregation] for m in available}
    result = df.groupby(group_cols, as_index=True)[available].agg(spec)
    # Flatten column multiindex to metric_agg
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [f"{a}_{b}" for a, b in result.columns]
    if drop_nan_rows:
        result = result.dropna(how="all")
    if sort_by and sort_by in result.index.names:
        result = result.sort_index(level=sort_by)
    return result.round(decimals)


def table_to_markdown(df: pd.DataFrame, caption: str, style: str = "ieee_access") -> str:
    """Convert DataFrame to Markdown with caption."""
    if df.empty:
        return f"\n*Table: {caption}*\n\n(No data.)\n"
    try:
        md = df.to_markdown()
    except ImportError:
        md = df.to_string()
    return f"\n*Table: {caption}*\n\n{md}\n"


def write_tables(
    config: dict,
    sources: dict[str, pd.DataFrame],
    base_output_dir: Path,
    output_settings: dict,
) -> None:
    """Generate and write all summary tables."""
    tables_cfg = config.get("summary_tables", {})
    if not tables_cfg.get("enabled", True):
        return
    checks = config.get("paper_ready_checks", {})
    rounding = checks.get("rounding", {"decimals": 4})
    drop_nan = checks.get("drop_nan_rows", True)
    sort_by = checks.get("sort_tables_by")
    base_output_dir = Path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    for table_def in tables_cfg.get("tables", []):
        source_key = table_def.get("source")
        if not source_key or source_key not in sources:
            continue
        df = sources[source_key]
        if df.empty:
            continue
        out_df = build_summary_table(
            df,
            group_by=table_def.get("group_by", []),
            metrics=table_def.get("metrics", []),
            aggregation=table_def.get("aggregation", ["mean"]),
            rounding=rounding,
            drop_nan_rows=drop_nan,
            sort_by=sort_by,
        )
        if out_df.empty:
            continue
        caption = table_def.get("caption", "")
        output_file = table_def.get("output_file", "table.md")
        out_path = base_output_dir / output_file
        content = table_to_markdown(out_df, caption, output_settings.get("style", "ieee_access"))
        if out_path.suffix.lower() == ".md" and not content.strip().startswith("#"):
            content = f"# {table_def.get('name', 'Summary')}\n\n{content}"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Wrote table: {out_path}")


def _ensure_figures_dir(
    base_output_dir: Path, figures_subdir: str, project_root: Path | None = None
) -> Path:
    """Resolve figures directory. If figures_subdir contains path separators, resolve from project_root."""
    if project_root is not None and ("/" in figures_subdir or "\\" in figures_subdir):
        figures_dir = project_root / figures_subdir
    else:
        figures_dir = base_output_dir / figures_subdir
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def plot_pareto_front(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_by: str | None,
    out_path: Path,
    caption: str,
    dpi: int = 300,
) -> None:
    """Scatter: x vs y, optional color by column."""
    if df.empty or x not in df.columns or y not in df.columns:
        return
    df = df.dropna(subset=[x, y])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    if color_by and color_by in df.columns:
        for val in df[color_by].dropna().unique():
            sub = df[df[color_by] == val]
            ax.scatter(sub[x], sub[y], label=str(val), alpha=0.7, s=20)
        ax.legend(loc="best", fontsize=8)
    else:
        ax.scatter(df[x], df[y], alpha=0.7, s=20)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    ax.set_title(caption[:80] + ("..." if len(caption) > 80 else ""))
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_return_vs_cvar(
    df: pd.DataFrame,
    x: str,
    y: str,
    group_by: str | None,
    out_path: Path,
    caption: str,
    dpi: int = 300,
) -> None:
    """Scatter with grouping (same as pareto)."""
    plot_pareto_front(df, x, y, group_by, out_path, caption, dpi)


def plot_sharpe_comparison(
    df: pd.DataFrame,
    y: str,
    group_by: str | None,
    out_path: Path,
    caption: str,
    dpi: int = 300,
) -> None:
    """Boxplot of y by group_by."""
    if df.empty or y not in df.columns:
        return
    df = df.dropna(subset=[y])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    if group_by and group_by in df.columns:
        groups = df.groupby(group_by)[y].apply(list)
        positions = range(len(groups))
        bp = ax.boxplot(groups.values, positions=positions, patch_artist=True)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(k) for k in groups.index])
        ax.set_ylabel(y.replace("_", " ").title())
    else:
        ax.boxplot(df[y].dropna())
        ax.set_ylabel(y.replace("_", " ").title())
    ax.set_title(caption[:80] + ("..." if len(caption) > 80 else ""))
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_chain_breaks_histogram(
    df: pd.DataFrame,
    x: str,
    out_path: Path,
    caption: str,
    dpi: int = 300,
) -> None:
    """Histogram of x. If all NaN, draws a placeholder with a note."""
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    if df.empty or x not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(x.replace("_", " ").title())
        ax.set_ylabel("Count")
    else:
        ser = df[x].dropna()
        if ser.empty:
            ax.bar(0, 1, width=0.1, color="gray", alpha=0.7, edgecolor="black")
            ax.set_xticks([0])
            ax.set_xticklabels(["N/A"])
            ax.text(0.5, 0.95, "No chain-break data\n(simulated annealing)", ha="center", va="top",
                    transform=ax.transAxes, fontsize=10)
            ax.set_xlabel("Chain break fraction")
            ax.set_ylabel("Count")
        else:
            ax.hist(ser, bins=min(50, max(10, len(ser) // 5)), edgecolor="black", alpha=0.7)
            ax.set_xlabel(x.replace("_", " ").title())
            ax.set_ylabel("Count")
    ax.set_title(caption[:80] + ("..." if len(caption) > 80 else ""))
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_breakdown(
    df: pd.DataFrame,
    x: str,
    y: list[str],
    out_path: Path,
    caption: str,
    dpi: int = 300,
) -> None:
    """Stacked bar: x as categories, y as stacked values."""
    if df.empty or x not in df.columns:
        return
    y_cols = [c for c in y if c in df.columns]
    if not y_cols:
        return
    agg = df.groupby(x)[y_cols].sum()
    if agg.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = np.zeros(len(agg))
    for col in y_cols:
        ax.bar(agg.index.astype(str), agg[col], bottom=bottom, label=col.replace("_", " ").title())
        bottom += agg[col].values
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel("Time (ms)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(caption[:80] + ("..." if len(caption) > 80 else ""))
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_figures(
    config: dict,
    sources: dict[str, pd.DataFrame],
    base_output_dir: Path,
    output_settings: dict,
    project_root: Path | None = None,
) -> None:
    """Generate and write all figures."""
    if not HAS_MATPLOTLIB:
        print("  matplotlib not available; skipping figures.")
        return
    figures_cfg = config.get("figures", {})
    if not figures_cfg.get("enabled", True):
        return
    figures_subdir = output_settings.get("figures_subdir", "figures")
    figures_dir = _ensure_figures_dir(base_output_dir, figures_subdir, project_root)
    dpi = output_settings.get("dpi", 300)
    for plot_def in figures_cfg.get("plots", []):
        source_key = plot_def.get("source")
        plot_name = plot_def.get("name", "")
        # Resolve data: for pareto_front use performance_table fallback when pareto is empty
        df = pd.DataFrame()
        if source_key and source_key in sources:
            df = sources[source_key]
        if plot_name == "pareto_front" and df.empty and "performance_table" in sources:
            perf = sources["performance_table"]
            if not perf.empty and plot_def.get("x") in perf.columns and plot_def.get("y") in perf.columns:
                df = perf
        if not source_key:
            continue
        out_file = plot_def.get("output_file", "figure.png")
        out_path = Path(out_file)
        if not out_path.is_absolute():
            out_path = figures_dir / out_path.name
        else:
            out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        caption = plot_def.get("caption", "")
        plot_type = plot_def.get("type", "scatter")
        if plot_type == "scatter":
            x = plot_def.get("x")
            y = plot_def.get("y")
            if plot_name == "pareto_front":
                plot_pareto_front(
                    df, x, y,
                    plot_def.get("color_by"),
                    out_path, caption, dpi,
                )
            else:
                plot_return_vs_cvar(
                    df, x, y,
                    plot_def.get("group_by"),
                    out_path, caption, dpi,
                )
        elif plot_type == "boxplot":
            plot_sharpe_comparison(
                df,
                plot_def.get("y"),
                plot_def.get("group_by"),
                out_path, caption, dpi,
            )
        elif plot_type == "histogram":
            plot_chain_breaks_histogram(
                df,
                plot_def.get("x"),
                out_path, caption, dpi,
            )
        elif plot_type == "stacked_bar":
            plot_runtime_breakdown(
                df,
                plot_def.get("x"),
                plot_def.get("y", []),
                out_path, caption, dpi,
            )
        if out_path.exists():
            print(f"  Wrote figure: {out_path}")


def run_analysis(config_path: Path, project_root: Path | None = None) -> None:
    """Load config, load data, generate tables and figures."""
    if project_root is None:
        project_root = config_path.resolve().parent
        # If config is in result_analysis/.../Q_annealing/llm.json, project root is repo root
        for _ in range(6):
            if (project_root / "results").is_dir() and (project_root / "src").is_dir():
                break
            parent = project_root.parent
            if parent == project_root:
                break
            project_root = parent
    config = load_config(config_path)
    inputs = config.get("inputs", {})
    data_sources = inputs.get("data_sources", inputs)
    output_settings = config.get("output_settings", {})
    base_output_dir = Path(output_settings.get("base_output_dir", "."))
    if not base_output_dir.is_absolute():
        base_output_dir = project_root / base_output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading data sources...")
    sources = load_sources(project_root, data_sources)
    for k, v in sources.items():
        print(f"  {k}: {len(v)} rows")
    print("Generating summary tables...")
    write_tables(config, sources, base_output_dir, output_settings)
    print("Generating figures...")
    write_figures(config, sources, base_output_dir, output_settings, project_root=project_root)
    print("Done.")


def main() -> None:
    import sys
    repo_root = Path(__file__).resolve().parent
    for _ in range(5):
        if (repo_root / "results").is_dir():
            break
        repo_root = repo_root.parent
    config_path = repo_root / "result_analysis" / "quantum_optimization" / "Q_annealing" / "llm.json"
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    run_analysis(config_path, project_root=repo_root)


if __name__ == "__main__":
    main()
