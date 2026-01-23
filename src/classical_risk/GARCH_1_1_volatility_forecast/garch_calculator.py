"""GARCH model calculator for volatility forecasting and VaR/CVaR estimation.

CRITICAL IEEE-Access correctness fix (vs earlier versions):
- arch_model often benefits from fitting on returns scaled by 100 (percentage points)
  for numerical stability.
- If you fit on scaled data but later compare VaR/CVaR against unscaled returns, you will
  inflate risk measures and can get near-zero violations.

This implementation enforces a single, consistent policy:
- Internally fit GARCH on scaled_returns = returns * scale_factor
- Use rescale=False in arch_model to avoid hidden internal rescaling.
- Convert all volatility outputs back to original return units by dividing by scale_factor.
- Same for forecasts: forecast variance is in scaled units^2 => divide by scale_factor^2.

IEEE convention:
- VaR/CVaR returned as positive loss magnitudes.
"""

from __future__ import annotations

import pickle
import time
import warnings
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional torch acceleration for simple tensor ops (not GARCH fitting itself)
try:
    import torch

    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except Exception:
    TORCH_AVAILABLE = False
    DEVICE = None

# arch library
try:
    from arch import arch_model

    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False

# Suppress noisy warnings from arch/scipy optimizers
warnings.filterwarnings('ignore', message='.*optimizer returned code.*')
warnings.filterwarnings('ignore', message='.*Inequality constraints incompatible.*')
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
warnings.filterwarnings('ignore', category=UserWarning, module='arch')


def _to_tensor(data: Union[pd.Series, np.ndarray, Any], dtype=None) -> Any:
    if not TORCH_AVAILABLE:
        return np.asarray(data.values if isinstance(data, pd.Series) else data)

    if dtype is None:
        dtype = torch.float32

    if isinstance(data, torch.Tensor):
        return data.to(device=DEVICE, dtype=dtype)
    if isinstance(data, pd.Series):
        return torch.tensor(data.values, device=DEVICE, dtype=dtype)
    return torch.tensor(data, device=DEVICE, dtype=dtype)


def _to_numpy(x: Any) -> np.ndarray:
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, pd.Series):
        return x.values
    return np.asarray(x)


class GARCHParameterCache:
    """Cache keyed by (asset, estimation_window)."""

    def __init__(self, cache_path: Optional[Union[str, Path]] = None):
        self.cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache_hits = 0
        self.cache_misses = 0

        if self.cache_path and self.cache_path.exists():
            try:
                self.load_cache()
            except Exception as e:
                warnings.warn(f"Failed to load cache: {e}")

    def _key(self, asset: str, window: int) -> Tuple[str, int]:
        return (asset, int(window))

    def get(self, asset: str, estimation_window: int) -> Optional[Dict[str, Any]]:
        k = self._key(asset, estimation_window)
        if k in self.cache:
            self.cache_hits += 1
            return self.cache[k]
        self.cache_misses += 1
        return None

    def set(self, asset: str, estimation_window: int, parameters: Dict[str, Any]) -> None:
        self.cache[self._key(asset, estimation_window)] = dict(parameters)

    def get_hit_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    def clear(self) -> None:
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def save_cache(self) -> None:
        if not self.cache_path:
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        if self.cache_path.suffix == '.parquet':
            rows = []
            for (asset, window), val in self.cache.items():
                cv = val.get('conditional_volatility')
                if isinstance(cv, pd.Series) and len(cv) > 0:
                    for dt, v in cv.items():
                        rows.append({'asset': asset, 'estimation_window': window, 'date': dt, 'conditional_volatility': float(v)})

            if rows:
                pd.DataFrame(rows).to_parquet(self.cache_path, index=False)
        else:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.cache, f)

    def load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return

        if self.cache_path.suffix == '.parquet':
            df = pd.read_parquet(self.cache_path)
            for (asset, window), g in df.groupby(['asset', 'estimation_window']):
                s = pd.Series(g['conditional_volatility'].to_numpy(), index=pd.to_datetime(g['date']), name='conditional_volatility')
                self.cache[(asset, int(window))] = {'conditional_volatility': s.sort_index()}
        else:
            with open(self.cache_path, 'rb') as f:
                self.cache = pickle.load(f)


def fit_garch_model(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = 't',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    scale_factor: float = 100.0
) -> Tuple[Optional[Any], Optional[Any], float]:
    """Fit GARCH on scaled returns for numerical stability.

    Returns:
        (model, fit_result, scale_factor_used)
    """
    if not ARCH_AVAILABLE:
        raise ImportError("arch library is required. Install with: pip install arch")

    r = returns.dropna().astype(float)
    if len(r) < max(p, q) + 25:
        return None, None, float(scale_factor)

    # Scale returns explicitly and disable arch's internal rescale
    r_scaled = r * float(scale_factor)

    try:
        model = arch_model(
            r_scaled,
            vol=vol,
            p=p,
            q=q,
            dist=('studentst' if dist in ['t', 'student_t', 'studentst'] else dist),
            mean=mean,
            rescale=False
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit_result = model.fit(update_freq=0, disp='off')

        return model, fit_result, float(scale_factor)
    except Exception:
        return None, None, float(scale_factor)


def compute_conditional_volatility(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = 't',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True,
    scale_factor: float = 100.0
) -> pd.Series:
    """Conditional volatility (sigma_t) in ORIGINAL return units."""
    model, fit_result, sf = fit_garch_model(returns, p=p, q=q, dist=dist, mean=mean, vol=vol, scale_factor=scale_factor)

    if fit_result is None:
        if fallback_long_run_variance:
            s = float(returns.dropna().std())
            return pd.Series(s, index=returns.index, name='conditional_volatility')
        return pd.Series(np.nan, index=returns.index, name='conditional_volatility')

    # fit_result.conditional_volatility is in scaled units => divide by sf
    cv = pd.Series(fit_result.conditional_volatility, index=returns.dropna().index, name='conditional_volatility')
    cv = cv / sf
    return cv


def forecast_volatility(
    returns: pd.Series,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = 't',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True,
    scale_factor: float = 100.0,
    use_gpu_for_fallback: bool = False
) -> float:
    """Forecast sigma for horizon steps, returned in ORIGINAL return units.

    Note: This function returns the *horizon-scaled* sigma (multiplied by sqrt(horizon)).
    In the asset-level pipeline we typically forecast 1-step sigma and apply sqrt(h)
    in VaR/CVaR functions to avoid double scaling.
    """
    model, fit_result, sf = fit_garch_model(returns, p=p, q=q, dist=dist, mean=mean, vol=vol, scale_factor=scale_factor)

    if fit_result is None:
        if not fallback_long_run_variance:
            return float('nan')

        r = returns.dropna().astype(float)
        if use_gpu_for_fallback and TORCH_AVAILABLE and DEVICE is not None and DEVICE.type == 'cuda':
            rt = _to_tensor(r)
            sigma = float(torch.std(rt).item())
        else:
            sigma = float(r.std())
        return float(sigma * np.sqrt(int(horizon)))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = fit_result.forecast(horizon=int(horizon), reindex=False)

        var_f = float(f.variance.iloc[-1, -1])  # scaled units^2
        sigma_f_scaled = float(np.sqrt(max(var_f, 0.0)))
        sigma_f = sigma_f_scaled / sf
        return float(sigma_f * np.sqrt(int(horizon)))
    except Exception:
        if not fallback_long_run_variance:
            return float('nan')
        sigma = float(returns.dropna().std())
        return float(sigma * np.sqrt(int(horizon)))


def compute_rolling_volatility_forecast(
    returns: pd.Series,
    window: int,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = 't',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True,
    batch_size: int = 50,
    scale_factor: float = 100.0
) -> pd.Series:
    """Rolling volatility forecasts (sigma) in original return units."""
    if not ARCH_AVAILABLE:
        raise ImportError("arch library is required. Install with: pip install arch")

    r = returns.dropna().astype(float)
    min_window = max(int(window), max(p, q) + 25)

    out = []
    idx = []

    for i in range(min_window, len(r)):
        w = r.iloc[i - int(window): i]
        sig = forecast_volatility(
            w,
            horizon=int(horizon),
            p=p,
            q=q,
            dist=dist,
            mean=mean,
            vol=vol,
            fallback_long_run_variance=fallback_long_run_variance,
            scale_factor=scale_factor
        )
        out.append(sig)
        idx.append(r.index[i])

        if TORCH_AVAILABLE and DEVICE is not None and DEVICE.type == 'cuda' and (i - min_window) % int(batch_size) == 0:
            torch.cuda.empty_cache()

    return pd.Series(out, index=idx, name='volatility_forecast')


def var_from_volatility(
    volatility: Union[pd.Series, float],
    confidence_level: float = 0.95,
    horizon: int = 1,
    dist: str = 't',
    df: Optional[int] = None,
    use_gpu: bool = False
) -> Union[pd.Series, float]:
    """VaR from sigma: VaR = z_alpha * sigma * sqrt(h).

    Returned as POSITIVE loss magnitude.
    """
    from scipy import stats

    alpha = float(1.0 - confidence_level)

    if dist in ['t', 'student_t', 'studentst']:
        if df is None:
            df = 5
        z = float(stats.t.ppf(alpha, df=df))
    elif dist == 'normal':
        z = float(stats.norm.ppf(alpha))
    else:
        if df is None:
            df = 5
        z = float(stats.t.ppf(alpha, df=df))

    z_pos = -z
    hs = float(np.sqrt(int(horizon)))

    if isinstance(volatility, pd.Series):
        vals = volatility.astype(float) * z_pos * hs
        return pd.Series(vals.to_numpy(), index=volatility.index, name='VaR')

    return float(float(volatility) * z_pos * hs)


def cvar_from_volatility(
    volatility: Union[pd.Series, float],
    confidence_level: float = 0.95,
    horizon: int = 1,
    dist: str = 't',
    df: Optional[int] = None,
    use_gpu: bool = False
) -> Union[pd.Series, float]:
    """CVaR/ES from sigma: CVaR = ES_factor * sigma * sqrt(h).

    Returned as POSITIVE loss magnitude.
    """
    from scipy import stats

    alpha = float(1.0 - confidence_level)

    if dist in ['t', 'student_t', 'studentst']:
        if df is None:
            df = 5
        t_alpha = float(stats.t.ppf(alpha, df=df))
        f = float(stats.t.pdf(t_alpha, df=df))
        es = -((df + t_alpha ** 2) / (df - 1.0)) * (f / max(alpha, 1e-15))
    elif dist == 'normal':
        z = float(stats.norm.ppf(alpha))
        phi = float(stats.norm.pdf(z))
        es = -(phi / max(alpha, 1e-15))
    else:
        if df is None:
            df = 5
        t_alpha = float(stats.t.ppf(alpha, df=df))
        f = float(stats.t.pdf(t_alpha, df=df))
        es = -((df + t_alpha ** 2) / (df - 1.0)) * (f / max(alpha, 1e-15))

    hs = float(np.sqrt(int(horizon)))

    if isinstance(volatility, pd.Series):
        vals = volatility.astype(float) * float(es) * hs
        return pd.Series(vals.to_numpy(), index=volatility.index, name='CVaR')

    return float(float(volatility) * float(es) * hs)


def compute_rolling_var_from_garch(
    returns: pd.Series,
    window: int,
    confidence_level: float = 0.95,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = 't',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True,
    batch_size: int = 50,
    scale_factor: float = 100.0
) -> pd.Series:
    sig = compute_rolling_volatility_forecast(
        returns,
        window=window,
        horizon=1,  # keep 1-step sigma; apply sqrt(h) in VaR
        p=p,
        q=q,
        dist=dist,
        mean=mean,
        vol=vol,
        fallback_long_run_variance=fallback_long_run_variance,
        batch_size=batch_size,
        scale_factor=scale_factor
    )
    return var_from_volatility(sig, confidence_level=confidence_level, horizon=horizon, dist=dist)


def compute_rolling_cvar_from_garch(
    returns: pd.Series,
    window: int,
    confidence_level: float = 0.95,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = 't',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True,
    batch_size: int = 50,
    scale_factor: float = 100.0
) -> pd.Series:
    sig = compute_rolling_volatility_forecast(
        returns,
        window=window,
        horizon=1,  # keep 1-step sigma; apply sqrt(h) in CVaR
        p=p,
        q=q,
        dist=dist,
        mean=mean,
        vol=vol,
        fallback_long_run_variance=fallback_long_run_variance,
        batch_size=batch_size,
        scale_factor=scale_factor
    )
    return cvar_from_volatility(sig, confidence_level=confidence_level, horizon=horizon, dist=dist)


def compute_asset_level_conditional_volatility(
    asset_returns: pd.Series,
    estimation_window: int,
    p: int = 1,
    q: int = 1,
    dist: str = 't',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True,
    cache: Optional[GARCHParameterCache] = None,
    scale_factor: float = 100.0
) -> pd.Series:
    name = getattr(asset_returns, 'name', 'asset') or 'asset'
    if cache:
        c = cache.get(name, int(estimation_window))
        if c and isinstance(c.get('conditional_volatility'), pd.Series):
            return c['conditional_volatility'].copy()

    r = asset_returns.dropna().astype(float)
    min_window = max(int(estimation_window), max(p, q) + 25)

    vals, idx = [], []
    for i in range(min_window, len(r)):
        w = r.iloc[i - int(estimation_window): i]
        cv = compute_conditional_volatility(
            w,
            p=p,
            q=q,
            dist=dist,
            mean=mean,
            vol=vol,
            fallback_long_run_variance=fallback_long_run_variance,
            scale_factor=scale_factor
        )
        if len(cv) == 0:
            continue
        vals.append(float(cv.iloc[-1]))
        idx.append(r.index[i])

    s = pd.Series(vals, index=idx, name='conditional_volatility')
    if cache:
        cache.set(name, int(estimation_window), {'conditional_volatility': s})
    return s


def _fit_single_asset_window(args: Tuple[str, pd.Series, int, int, int, str, str, str, bool, Optional[GARCHParameterCache], float]) -> Tuple[Tuple[str, int], pd.Series]:
    asset, r, w, p, q, dist, mean, vol, fb, cache, sf = args
    try:
        s = compute_asset_level_conditional_volatility(r, w, p=p, q=q, dist=dist, mean=mean, vol=vol, fallback_long_run_variance=fb, cache=cache, scale_factor=sf)
        return (asset, int(w)), s
    except Exception:
        return (asset, int(w)), pd.Series(dtype=float)


def compute_all_asset_conditional_volatilities(
    daily_returns: pd.DataFrame,
    estimation_windows: List[int],
    p: int = 1,
    q: int = 1,
    dist: str = 't',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True,
    cache: Optional[GARCHParameterCache] = None,
    n_jobs: Optional[int] = None,
    scale_factor: float = 100.0
) -> Dict[Tuple[str, int], pd.Series]:
    tasks = []
    for a in daily_returns.columns:
        r = daily_returns[a].dropna()
        for w in estimation_windows:
            tasks.append((a, r, int(w), int(p), int(q), dist, mean, vol, bool(fallback_long_run_variance), cache, float(scale_factor)))

    if not tasks:
        return {}

    if not n_jobs or int(n_jobs) <= 1:
        out = {}
        for t in tasks:
            k, s = _fit_single_asset_window(t)
            out[k] = s
        return out

    out: Dict[Tuple[str, int], pd.Series] = {}
    try:
        with Pool(processes=int(n_jobs)) as pool:
            res = pool.map(_fit_single_asset_window, tasks)
        for k, s in res:
            out[k] = s
    except Exception as e:
        warnings.warn(f"Parallel computation failed ({e}); falling back to sequential")
        for t in tasks:
            k, s = _fit_single_asset_window(t)
            out[k] = s

    return out


def compute_portfolio_volatility_from_assets(
    portfolio_weights: pd.Series,
    asset_conditional_vols: Dict[str, pd.Series],
    covariance_matrix: Optional[pd.DataFrame] = None,
    use_dynamic_covariance: bool = False,
    use_gpu: bool = False
) -> pd.Series:
    """Compute portfolio volatility via variance aggregation from asset-level conditional variances."""
    common_assets = portfolio_weights.index.intersection(set(asset_conditional_vols.keys()))
    if len(common_assets) == 0:
        return pd.Series(dtype=float)

    w = portfolio_weights.loc[common_assets].astype(float)
    s = float(w.sum())
    if s == 0:
        return pd.Series(dtype=float)
    w = w / s

    vols = [asset_conditional_vols[a].dropna() for a in common_assets]
    if not vols:
        return pd.Series(dtype=float)

    dates = vols[0].index
    for v in vols[1:]:
        dates = dates.intersection(v.index)
    if len(dates) == 0:
        return pd.Series(dtype=float)

    aligned_vols = {}
    for asset in common_assets:
        aligned_vols[asset] = asset_conditional_vols[asset].loc[dates]

    portfolio_vols = []
    for date in dates:
        portfolio_var = 0.0
        for asset in common_assets:
            w_i = w[asset]
            sigma_i = aligned_vols[asset].loc[date]
            portfolio_var += (w_i ** 2) * (sigma_i ** 2)
        portfolio_vols.append(np.sqrt(portfolio_var))

    return pd.Series(portfolio_vols, index=dates, name='portfolio_volatility')


def compute_horizons(
    base_horizon: int = 1,
    scaled_horizons: Optional[List[int]] = None,
    scaling_rule: str = 'sqrt_time'
) -> List[int]:
    """Compute list of horizons from base horizon and scaled horizons."""
    horizons = [base_horizon]

    if scaled_horizons:
        for scale in scaled_horizons:
            if scaling_rule == 'sqrt_time':
                horizon = int(base_horizon * np.sqrt(scale))
            else:
                horizon = base_horizon * scale
            if horizon not in horizons:
                horizons.append(horizon)

    return sorted(horizons)


def compute_rolling_garch_asset_level(
    asset_returns: pd.Series,
    window: int,
    p: int = 1,
    q: int = 1,
    dist: str = 't',
    mean: str = 'Zero',
    vol: str = 'GARCH',
    fallback_long_run_variance: bool = True,
    cache: Optional[GARCHParameterCache] = None,
    asset_name: Optional[str] = None,
    scale_factor: float = 100.0
) -> Optional[Dict[str, Any]]:
    """Compute rolling GARCH conditional volatility and one-step-ahead forecasts for an asset."""
    asset_name = asset_name or (getattr(asset_returns, 'name', 'asset') or 'asset')

    if cache:
        cached = cache.get(asset_name, window)
        if cached:
            return cached

    r = asset_returns.dropna().astype(float)
    min_window = max(int(window), max(p, q) + 25)

    conditional_vols = []
    forecast_vols = []
    vol_dates = []
    garch_params_list = []

    for i in range(min_window, len(r)):
        w = r.iloc[i - int(window): i]
        model, fit_result, sf = fit_garch_model(w, p=p, q=q, dist=dist, mean=mean, vol=vol, scale_factor=scale_factor)

        if fit_result is None:
            if fallback_long_run_variance:
                fallback_vol = float(w.std())
                conditional_vols.append(fallback_vol)
                forecast_vols.append(fallback_vol)
                vol_dates.append(r.index[i])
                garch_params_list.append({
                    'omega': np.nan,
                    'alpha': np.nan,
                    'beta': np.nan,
                    'alpha_plus_beta': np.nan,
                    'unconditional_variance': fallback_vol ** 2,
                    'last_conditional_sigma': fallback_vol,
                    'fit_success': False,
                    'convergence_flag': False
                })
            continue

        try:
            cv_series = fit_result.conditional_volatility
            if len(cv_series) == 0:
                continue

            last_cv = float(cv_series.iloc[-1]) / sf  # scale back

            try:
                f = fit_result.forecast(horizon=1, reindex=False)
                var_f = float(f.variance.iloc[-1, -1])
                forecast_vol = float(np.sqrt(max(var_f, 0.0))) / sf  # scale back
            except Exception:
                forecast_vol = last_cv

            params = fit_result.params
            omega = params.get('omega', np.nan)
            alpha = params.get('alpha[1]', params.get('alpha', np.nan))
            beta = params.get('beta[1]', params.get('beta', np.nan))
            alpha_plus_beta = alpha + beta if not (np.isnan(alpha) or np.isnan(beta)) else np.nan

            if not np.isnan(omega) and not np.isnan(alpha_plus_beta) and alpha_plus_beta < 1:
                unconditional_var = (omega / (sf ** 2)) / (1 - alpha_plus_beta)  # omega is in scaled units^2
            else:
                unconditional_var = np.nan

            conditional_vols.append(last_cv)
            forecast_vols.append(forecast_vol)
            vol_dates.append(r.index[i])

            garch_params_list.append({
                'omega': float(omega / (sf ** 2)) if not np.isnan(omega) else np.nan,  # scale back
                'alpha': float(alpha) if not np.isnan(alpha) else np.nan,
                'beta': float(beta) if not np.isnan(beta) else np.nan,
                'alpha_plus_beta': float(alpha_plus_beta) if not np.isnan(alpha_plus_beta) else np.nan,
                'unconditional_variance': float(unconditional_var) if not np.isnan(unconditional_var) else np.nan,
                'last_conditional_sigma': last_cv,
                'fit_success': True,
                'convergence_flag': True,
                'loglikelihood': float(fit_result.loglikelihood) if hasattr(fit_result, 'loglikelihood') else np.nan,
                'aic': float(fit_result.aic) if hasattr(fit_result, 'aic') else np.nan,
                'bic': float(fit_result.bic) if hasattr(fit_result, 'bic') else np.nan
            })

        except Exception:
            if fallback_long_run_variance:
                fallback_vol = float(w.std())
                conditional_vols.append(fallback_vol)
                forecast_vols.append(fallback_vol)
                vol_dates.append(r.index[i])
                garch_params_list.append({
                    'omega': np.nan,
                    'alpha': np.nan,
                    'beta': np.nan,
                    'alpha_plus_beta': np.nan,
                    'unconditional_variance': fallback_vol ** 2,
                    'last_conditional_sigma': fallback_vol,
                    'fit_success': False,
                    'convergence_flag': False
                })

    if len(conditional_vols) == 0:
        return None

    conditional_vol_series = pd.Series(conditional_vols, index=vol_dates, name='conditional_volatility')
    forecast_vol_series = pd.Series(forecast_vols, index=vol_dates, name='forecast_volatility')
    parameters = garch_params_list[-1] if len(garch_params_list) > 0 else {}

    result = {
        'conditional_volatility': conditional_vol_series,
        'forecast_volatility': forecast_vol_series,
        'parameters': parameters
    }

    if cache:
        cache.set(asset_name, window, result)

    return result
