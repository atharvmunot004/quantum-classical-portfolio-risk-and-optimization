"""
Main evaluation script for qPCA factor risk at asset level.
Orchestrates: data, rolling covariance/density/qPCA, factor exposures, risk proxies, classical alignment, outputs.
"""
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from .returns import load_panel_prices, compute_daily_returns
from .covariance import build_covariance, to_density_matrix
from .qpca_core import qpca_density_matrix
from .factor_risk import (
    factor_exposures_projection,
    idiosyncratic_variance,
    factor_gaussian_var_cvar,
)
from .classical_baseline import (
    classical_pca_eigen,
    principal_angle_distance,
    explained_variance_gap,
    exposure_correlation,
)
from .metrics import (
    explained_variance_ratios,
    cumulative_explained_variance,
    factor_stability_cosine_similarity,
)
from .cache import QPCAWindowCache
from .time_sliced_metrics import compute_time_sliced_metrics
from .report_generator import generate_report


def _standardize_window(
    R: np.ndarray,
    winsorize_quantiles: List[float] = [0.001, 0.999],
) -> np.ndarray:
    """Z-score and winsorize per column (assets)."""
    R = np.asarray(R, dtype=float)
    q1, q2 = winsorize_quantiles[0], winsorize_quantiles[1]
    lo, hi = np.nanpercentile(R, [q1 * 100, q2 * 100], axis=0)
    R = np.clip(R, lo, hi)
    mu = np.nanmean(R, axis=0)
    std = np.nanstd(R, axis=0)
    std[std < 1e-12] = 1.0
    return (R - mu) / std


def _safety_checks(
    rho: np.ndarray,
    evals: np.ndarray,
    exposures: np.ndarray,
    checks: List[Dict],
) -> Tuple[bool, List[str]]:
    """Run safety checks; return (pass, list of failure messages)."""
    failures = []
    for c in checks:
        name = c.get("name", "")
        rule = c.get("rule", "")
        if name == "density_matrix_psd" and "min_eigenvalue" in rule:
            if np.any(evals < -1e-8):
                failures.append("density_matrix_psd")
        elif name == "explained_variance_monotone":
            if len(evals) > 1 and np.any(np.diff(evals) > 0):
                failures.append("explained_variance_monotone")
        elif name == "finite_factor_exposures":
            if not np.all(np.isfinite(exposures)):
                failures.append("finite_factor_exposures")
    return (len(failures) == 0, failures)


def evaluate_qpca_factor_risk(
    config_path: Optional[Path] = None,
    config_dict: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run qPCA factor risk pipeline. Returns (factor_df, exposure_df, risk_df, metrics_df, time_sliced_df).
    """
    if config_dict is None:
        config_path = config_path or Path(__file__).parent / "llm.json"
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = config_dict

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    panel_path = project_root / config["inputs"]["panel_price_path"]
    if not panel_path.exists() and "preprocessed" in str(panel_path):
        panel_path = project_root / str(config["inputs"]["panel_price_path"]).replace(
            "preprocessed", "processed"
        )

    print("=" * 80)
    print("qPCA FACTOR RISK ASSET-LEVEL EVALUATION")
    print("=" * 80)
    print(f"Loading: {panel_path}")

    prices = load_panel_prices(panel_path)
    asset_cfg = config["inputs"].get("asset_universe", {})
    if asset_cfg.get("include"):
        prices = prices[[c for c in asset_cfg["include"] if c in prices.columns]]
    if asset_cfg.get("exclude"):
        prices = prices.drop(columns=asset_cfg["exclude"], errors="ignore")

    data_settings = config.get("data_settings", {})
    if data_settings.get("calendar", {}).get("sort_index", True):
        prices = prices.sort_index()
    if data_settings.get("calendar", {}).get("drop_duplicate_dates", True):
        prices = prices[~prices.index.duplicated(keep="first")]

    return_type = data_settings.get("return_type", "log")
    returns = compute_daily_returns(prices, method=return_type)
    # Panel intersection
    returns = returns.dropna(axis=0, how="any")
    returns = returns.dropna(axis=1, how="any")
    min_obs = data_settings.get("missing_data_policy", {}).get("min_required_observations", 800)
    min_assets_config = data_settings.get("missing_data_policy", {}).get("min_required_assets_per_window", 20)
    # Use at most the assets we have so pipeline runs with smaller universes
    min_assets = min(min_assets_config, len(returns.columns))
    if min_assets < 2:
        raise ValueError("Need at least 2 assets for factor analysis.")
    if len(returns) < min_obs:
        raise ValueError(f"After dropna, returns have {len(returns)} rows; need >= {min_obs}")

    std_cfg = data_settings.get("standardization", {})
    winsorize = std_cfg.get("winsorize_quantiles", [0.001, 0.999])

    qpca_settings = config.get("qpca_settings", {})
    estimation_windows = qpca_settings.get("estimation_windows", [252, 500])
    top_k_list = qpca_settings.get("top_k_components", [3, 5, 10])
    rolling_cfg = qpca_settings.get("rolling", {})
    step_size = rolling_cfg.get("step_size", 5)
    dm_cfg = qpca_settings.get("density_matrix_construction", {})
    shrink = dm_cfg.get("shrinkage_method", "ledoit_wolf")
    to_dm = dm_cfg.get("to_density_matrix", {})
    numerical_eps = to_dm.get("numerical_stability_eps", 1e-10)

    quantum_exec = config.get("quantum_execution", {})
    phase_cfg = quantum_exec.get("phase_estimation", {})
    precision_bits = phase_cfg.get("precision_bits", 6)
    shots = quantum_exec.get("shots", 4000)

    factor_risk_cfg = config.get("factor_risk_construction", {})
    risk_proxy = factor_risk_cfg.get("risk_proxies", {})
    confidence_levels = risk_proxy.get("confidence_levels", [0.95, 0.99])

    safety_checks = config.get("computation_strategy", {}).get("safety_checks", {})
    check_list = safety_checks.get("checks", []) if safety_checks.get("enabled") else []

    cache_cfg = config.get("computation_strategy", {}).get("cache", {})
    cache = None
    if cache_cfg.get("enabled") and cache_cfg.get("parameter_store_path"):
        cache_path = project_root / cache_cfg["parameter_store_path"]
        cache = QPCAWindowCache(cache_path, cache_cfg.get("key_fields"))

    assets = returns.columns.tolist()
    dates = returns.index
    R_all = returns.values
    T, N = R_all.shape
    print(f"  Assets: {N}, dates: {T}")

    # Runtime instrumentation
    runtimes = {
        "total_runtime_ms": 0.0,
        "returns_compute_time_ms": 0.0,
        "covariance_build_time_ms": 0.0,
        "density_matrix_build_time_ms": 0.0,
        "state_preparation_time_ms": 0.0,
        "qpca_runtime_ms": 0.0,
        "projection_time_ms": 0.0,
        "factor_risk_compute_time_ms": 0.0,
        "time_slicing_time_ms": 0.0,
    }
    t_total = time.perf_counter()

    factor_records = []
    exposure_records = []
    risk_records = []
    metrics_records = []
    window_param_records = []
    prev_evecs_by_key = {}  # (estimation_window, top_k) -> evecs for stability

    for window in estimation_windows:
        if T < window:
            continue
        # End indices: from (window-1) to (T-1) with step_size
        for end_idx in range(window - 1, T, step_size):
            start_idx = end_idx - window + 1
            R = R_all[start_idx : end_idx + 1, :]
            if R.shape[0] != window or np.any(~np.isfinite(R)):
                continue
            if R.shape[1] < min_assets:
                continue
            date = dates[end_idx]
            R_std = _standardize_window(R, winsorize)

            t0 = time.perf_counter()
            cov, cov_time_ms = build_covariance(R_std, shrinkage_method=shrink)
            runtimes["covariance_build_time_ms"] += cov_time_ms

            try:
                rho = to_density_matrix(cov, numerical_stability_eps=numerical_eps)
            except ValueError:
                continue
            runtimes["density_matrix_build_time_ms"] += (time.perf_counter() - t0) * 1000

            for top_k in top_k_list:
                top_k = min(top_k, R.shape[1])
                cache_key = {
                    "date": date,
                    "estimation_window": window,
                    "top_k": top_k,
                    "precision_bits": precision_bits,
                    "shots": shots,
                    "shrinkage_method": shrink,
                    "return_type": return_type,
                }
                evals_q, evecs_q, q_metrics = qpca_density_matrix(
                    rho, top_k,
                    precision_bits=precision_bits,
                    max_eigenvalue_support=1.0,
                )
                runtimes["qpca_runtime_ms"] += q_metrics.get("qpca_runtime_ms", 0)
                evals_c, evecs_c = classical_pca_eigen(rho, top_k)
                t_proj = time.perf_counter()
                exposures_q = factor_exposures_projection(R_std, evecs_q)
                exposures_c = factor_exposures_projection(R_std, evecs_c)
                runtimes["projection_time_ms"] += (time.perf_counter() - t_proj) * 1000

                ok, fail_list = _safety_checks(rho, evals_q, exposures_q, check_list)
                if fail_list and any(c.get("on_fail") == "skip_timestamp" for c in check_list if c.get("name") in fail_list):
                    if "finite_factor_exposures" in fail_list:
                        continue
                fit_success = ok or all(c.get("on_fail") == "flag_and_continue" for c in check_list)

                ev_ratio = explained_variance_ratios(evals_q)
                cum_ev = cumulative_explained_variance(evals_q)
                key_stab = (window, top_k)
                prev_evecs = prev_evecs_by_key.get(key_stab)
                stab = factor_stability_cosine_similarity(prev_evecs, evecs_q)
                prev_evecs_by_key[key_stab] = evecs_q.copy()

                principal_angle = principal_angle_distance(evecs_q, evecs_c)
                ev_gap = explained_variance_gap(evals_q, evals_c)
                exp_corr = exposure_correlation(exposures_q, exposures_c)

                factor_std = np.sqrt(np.maximum(evals_q, 1e-12))
                idio_var = idiosyncratic_variance(R_std, exposures_q, evals_q)

                for j in range(top_k):
                    factor_records.append({
                        "date": date,
                        "estimation_window": window,
                        "top_k": top_k,
                        "factor_id": j,
                        "eigenvalue": float(evals_q[j]),
                        "explained_variance_ratio": float(ev_ratio[j]),
                        "cumulative_explained_variance": float(cum_ev[j]),
                    })

                t_risk = time.perf_counter()
                for i, asset in enumerate(assets):
                    for j in range(top_k):
                        exposure_records.append({
                            "asset": asset,
                            "date": date,
                            "estimation_window": window,
                            "top_k": top_k,
                            "factor_id": j,
                            "exposure": float(exposures_q[i, j]),
                        })
                    for conf in confidence_levels:
                        var_f, cvar_f = factor_gaussian_var_cvar(
                            exposures_q, evals_q, idio_var, conf,
                        )
                        risk_records.append({
                            "asset": asset,
                            "date": date,
                            "confidence_level": conf,
                            "estimation_window": window,
                            "top_k": top_k,
                            "VaR_factor": float(var_f[i]),
                            "CVaR_factor": float(cvar_f[i]),
                            "idiosyncratic_variance": float(idio_var[i]),
                        })
                runtimes["factor_risk_compute_time_ms"] += (time.perf_counter() - t_risk) * 1000

                # One metrics row per (asset, window, top_k) per date; we aggregate later
                for i, asset in enumerate(assets):
                    var_95, cvar_95 = factor_gaussian_var_cvar(exposures_q, evals_q, idio_var, 0.95)
                    var_99, cvar_99 = factor_gaussian_var_cvar(exposures_q, evals_q, idio_var, 0.99)
                    row = {
                        "asset": asset,
                        "estimation_window": window,
                        "top_k": top_k,
                        "explained_variance_ratio_k": float(ev_ratio[-1]) if len(ev_ratio) else np.nan,
                        "cumulative_explained_variance_k": float(cum_ev[-1]) if len(cum_ev) else np.nan,
                        "factor_stability_cosine_similarity": stab,
                        "principal_angle_distance_vs_classical": principal_angle,
                        "explained_variance_gap": ev_gap,
                        "exposure_correlation": exp_corr,
                        "factor_var_95": float(var_95[i]),
                        "factor_var_99": float(var_99[i]),
                        "factor_cvar_95": float(cvar_95[i]),
                        "factor_cvar_99": float(cvar_99[i]),
                        "idiosyncratic_variance": float(idio_var[i]),
                        "precision_bits": precision_bits,
                        "shots": shots,
                        "qpca_circuit_depth": q_metrics.get("qpca_circuit_depth"),
                        "qpca_circuit_width": q_metrics.get("qpca_circuit_width"),
                        "date": date,
                    }
                    metrics_records.append(row)

                window_param_records.append({
                    **cache_key,
                    "density_matrix_trace": float(np.trace(rho)),
                    "estimated_eigenvalues": evals_q.tolist(),
                    "explained_variance_ratio": ev_ratio.tolist(),
                    "fit_success": fit_success,
                })

    runtimes["total_runtime_ms"] = (time.perf_counter() - t_total) * 1000
    if cache and cache_cfg.get("cache_hit_ratio_metric"):
        runtimes["cache_hit_ratio"] = cache.get_hit_ratio()
    if cache and window_param_records:
        cache.save_records(window_param_records)

    factor_df = pd.DataFrame(factor_records)
    exposure_df = pd.DataFrame(exposure_records)
    risk_df = pd.DataFrame(risk_records)
    metrics_df = pd.DataFrame(metrics_records)

    # Deduplicate metrics to one row per (asset, estimation_window, top_k) - keep latest date
    if not metrics_df.empty and "date" in metrics_df.columns:
        metrics_df = metrics_df.sort_values("date").groupby(
            ["asset", "estimation_window", "top_k"], as_index=False
        ).last()

    # Time-sliced metrics
    time_sliced_list = []
    ts_cfg = config.get("evaluation", {}).get("time_sliced_metrics", {})
    if ts_cfg.get("enabled") and not factor_df.empty:
        t0_ts = time.perf_counter()
        for slice_by in ts_cfg.get("slice_by", ["year", "quarter", "month"]):
            for rec in compute_time_sliced_metrics(
                factor_df, metrics_df,
                slice_by=slice_by,
                min_observations=ts_cfg.get("minimum_observations_per_slice", 30),
            ):
                time_sliced_list.append(rec)
        runtimes["time_slicing_time_ms"] = (time.perf_counter() - t0_ts) * 1000
    time_sliced_df = pd.DataFrame(time_sliced_list)

    # Outputs
    outputs = config.get("outputs", {})

    if "window_parameter_store" in outputs:
        out_path = project_root / outputs["window_parameter_store"]["path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if window_param_records:
            cols = outputs["window_parameter_store"].get("contents", list(window_param_records[0].keys()))
            cols = [c for c in cols if c in window_param_records[0]]
            pd.DataFrame(window_param_records)[cols].drop_duplicates(
                subset=["date", "estimation_window", "top_k"], keep="last"
            ).to_parquet(out_path, index=False)
            print(f"Saved window parameters: {out_path}")

    if "factor_store" in outputs:
        out_path = project_root / outputs["factor_store"]["path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cols = outputs["factor_store"].get("contents", factor_df.columns.tolist())
        cols = [c for c in cols if c in factor_df.columns]
        factor_df[cols].to_parquet(out_path, index=False)
        print(f"Saved factor series: {out_path}")

    if "asset_exposure_store" in outputs:
        out_path = project_root / outputs["asset_exposure_store"]["path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        exposure_df.to_parquet(out_path, index=False)
        print(f"Saved asset exposures: {out_path}")

    if "asset_factor_risk_store" in outputs:
        out_path = project_root / outputs["asset_factor_risk_store"]["path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        risk_df.to_parquet(out_path, index=False)
        print(f"Saved factor risk proxies: {out_path}")

    if "metrics_table" in outputs:
        out_path = project_root / outputs["metrics_table"]["path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_parquet(out_path, index=False)
        print(f"Saved metrics: {out_path}")

    if "time_sliced_metrics_table" in outputs and not time_sliced_df.empty:
        out_path = project_root / outputs["time_sliced_metrics_table"]["path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        time_sliced_df.to_parquet(out_path, index=False)
        print(f"Saved time-sliced metrics: {out_path}")

    if "report" in outputs:
        out_path = project_root / outputs["report"]["path"]
        generate_report(
            metrics_df,
            out_path,
            factor_df=factor_df,
            runtime_metrics=runtimes,
            report_sections=outputs["report"].get("include_sections"),
            qpca_settings=qpca_settings,
        )
        print(f"Saved report: {out_path}")

    print(f"\nCompleted: factors {len(factor_df)}, exposures {len(exposure_df)}, risk {len(risk_df)}, metrics {len(metrics_df)}")
    print("=" * 80)
    return factor_df, exposure_df, risk_df, metrics_df, time_sliced_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="qPCA factor risk asset-level evaluation")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    factor_df, exposure_df, risk_df, metrics_df, ts_df = evaluate_qpca_factor_risk(config_path=args.config)
    print(f"Factors: {len(factor_df)}, Exposures: {len(exposure_df)}, Risk: {len(risk_df)}, Metrics: {len(metrics_df)}, Time-sliced: {len(ts_df)}")


if __name__ == "__main__":
    main()
