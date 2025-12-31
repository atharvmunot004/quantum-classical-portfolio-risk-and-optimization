"""
Main evaluation script for GARCH(1,1) Volatility Forecasting VaR and CVaR.

Orchestrates the entire VaR/CVaR evaluation pipeline with batched execution:
- Asset-level GARCH fitting (once per asset/window)
- Conditional volatility computation (cached and reused)
- Portfolio volatility projection from assets
- Batched portfolio processing with resume support
- Parquet-only output with schema JSON
- Markdown summary report
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
import gc

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*found in sys.modules.*')
warnings.filterwarnings('ignore', message='.*optimizer returned code.*')
warnings.filterwarnings('ignore', message='.*Inequality constraints incompatible.*')
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
warnings.filterwarnings('ignore', category=UserWarning, module='arch')

from .returns import (
    load_panel_prices,
    load_portfolio_weights,
    compute_daily_returns,
    compute_portfolio_returns
)
from .garch_calculator import (
    GARCHParameterCache,
    compute_all_asset_conditional_volatilities,
    compute_portfolio_volatility_from_assets,
    compute_horizons,
    var_from_volatility,
    cvar_from_volatility
)

# Check GPU availability
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available() if torch.cuda.is_available() else False
    if GPU_AVAILABLE:
        GPU_DEVICE = torch.device('cuda')
        GPU_COUNT = torch.cuda.device_count()
    else:
        GPU_DEVICE = None
        GPU_COUNT = 0
except ImportError:
    GPU_AVAILABLE = False
    GPU_DEVICE = None
    GPU_COUNT = 0
from .backtesting import compute_accuracy_metrics
from .metrics import (
    compute_tail_metrics,
    compute_cvar_tail_metrics,
    compute_structure_metrics,
    compute_distribution_metrics,
    compute_runtime_metrics
)
from .report_generator import generate_report


def _get_worker_count(config: Dict, default: Optional[int] = None) -> int:
    """
    Get number of workers based on configuration.
    
    Args:
        config: Configuration dictionary
        default: Default number of workers (None for auto)
        
    Returns:
        Number of workers to use
    """
    parallel_config = config.get('parallelization', {})
    
    if not parallel_config.get('enabled', True):
        return 1
    
    workers_config = parallel_config.get('workers', {})
    mode = workers_config.get('mode', 'auto')
    max_workers = workers_config.get('max_workers')
    cap_fraction = workers_config.get('cap_fraction_of_cpu', 0.8)
    
    if max_workers is not None:
        return max(1, int(max_workers))
    
    if mode == 'auto':
        cpu_count_val = cpu_count()
        return max(1, int(cpu_count_val * cap_fraction))
    
    if default is not None:
        return max(1, default)
    
    return max(1, cpu_count())


def _process_single_portfolio_worker(
    portfolio_data: Tuple[int, Union[int, str], pd.Series],
    daily_returns: pd.DataFrame,
    asset_conditional_vols: Dict[Tuple[str, int], pd.Series],
    covariance_matrix: pd.DataFrame,
    confidence_levels: List[float],
    horizons: List[int],
    estimation_windows: List[int],
    dist: str,
    use_dynamic_covariance: bool,
    use_gpu: bool = True
) -> List[Dict]:
    """
    Worker function to process a single portfolio (for within-batch parallelization).
    
    This function is designed to be stateless and read-only with respect to cached data.
    
    Args:
        portfolio_data: Tuple of (portfolio_idx, portfolio_id, portfolio_weights)
        daily_returns: DataFrame of daily returns
        asset_conditional_vols: Dictionary mapping (asset, window) to conditional volatility Series
        covariance_matrix: Covariance matrix for portfolio volatility
        confidence_levels: List of confidence levels
        horizons: List of horizons
        estimation_windows: List of estimation windows
        dist: Distribution assumption
        use_dynamic_covariance: Whether to use dynamic covariance
        
    Returns:
        List of result dictionaries for this portfolio
    """
    portfolio_idx, portfolio_id, portfolio_weights = portfolio_data
    results = []
    
    try:
        # Compute portfolio returns
        common_assets = daily_returns.columns.intersection(portfolio_weights.index)
        if len(common_assets) == 0:
            return results
        
        portfolio_returns = compute_portfolio_returns(
            daily_returns[common_assets],
            portfolio_weights[common_assets],
            align_assets=False
        )
        
        # Get covariance subset
        cov_subset = covariance_matrix.loc[common_assets, common_assets] if covariance_matrix is not None else None
        
        # Process each configuration
        for confidence_level in confidence_levels:
            for horizon in horizons:
                for window in estimation_windows:
                    try:
                        # Get asset-level conditional volatilities
                        asset_vols_dict = {}
                        for asset in common_assets:
                            key = (asset, window)
                            if key in asset_conditional_vols:
                                asset_vols_dict[asset] = asset_conditional_vols[key]
                        
                        if len(asset_vols_dict) == 0:
                            continue
                        
                        # Compute portfolio volatility (GPU-accelerated)
                        portfolio_vol = compute_portfolio_volatility_from_assets(
                            portfolio_weights[common_assets],
                            asset_vols_dict,
                            covariance_matrix=cov_subset,
                            use_dynamic_covariance=use_dynamic_covariance,
                            use_gpu=use_gpu
                        )
                        
                        if len(portfolio_vol) == 0:
                            continue
                        
                        # Scale by horizon
                        portfolio_vol_scaled = portfolio_vol * np.sqrt(horizon)
                        
                        # Convert to VaR and CVaR (GPU-accelerated)
                        rolling_var = var_from_volatility(
                            portfolio_vol_scaled,
                            confidence_level=confidence_level,
                            horizon=1,
                            dist=dist,
                            use_gpu=use_gpu
                        )
                        
                        rolling_cvar = cvar_from_volatility(
                            portfolio_vol_scaled,
                            confidence_level=confidence_level,
                            horizon=1,
                            dist=dist,
                            use_gpu=use_gpu
                        )
                        
                        # Align returns and VaR/CVaR
                        common_dates = portfolio_returns.index.intersection(rolling_var.index)
                        if len(common_dates) == 0:
                            continue
                        
                        aligned_returns = portfolio_returns.loc[common_dates]
                        aligned_var = rolling_var.loc[common_dates]
                        aligned_cvar = rolling_cvar.loc[common_dates]
                        
                        # Compute metrics (aggregated over entire time series)
                        accuracy_metrics = compute_accuracy_metrics(
                            aligned_returns,
                            aligned_var,
                            confidence_level=confidence_level
                        )
                        
                        tail_metrics = compute_tail_metrics(
                            aligned_returns,
                            aligned_var,
                            confidence_level=confidence_level
                        )
                        
                        cvar_tail_metrics = compute_cvar_tail_metrics(
                            aligned_returns,
                            aligned_cvar,
                            aligned_var,
                            confidence_level=confidence_level
                        )
                        
                        structure_metrics = compute_structure_metrics(
                            portfolio_weights[common_assets],
                            cov_subset
                        )
                        
                        distribution_metrics = compute_distribution_metrics(
                            aligned_returns
                        )
                        
                        # Create one row per date (as per schema primary_key)
                        # Backtesting metrics are the same for all dates in the same portfolio-configuration
                        for date in common_dates:
                            result = {
                                'portfolio_id': portfolio_id,
                                'date': date,
                                'confidence_level': confidence_level,
                                'horizon': horizon,
                                'estimation_window': window,
                                'var': float(aligned_var.loc[date]) if date in aligned_var.index else np.nan,
                                'cvar': float(aligned_cvar.loc[date]) if date in aligned_cvar.index else np.nan,
                                **accuracy_metrics,  # Same for all dates in this portfolio-configuration
                                **tail_metrics,
                                **cvar_tail_metrics,
                                **structure_metrics,
                                **distribution_metrics
                            }
                            
                            results.append(result)
                            
                    except Exception:
                        continue
                        
    except Exception:
        pass
    
    return results


def _load_preprocessing_cache(cache_path: Path) -> Optional[Dict]:
    """Load preprocessing cache if it exists."""
    if cache_path.exists():
        try:
            cache_df = pd.read_parquet(cache_path)
            return {
                'aligned_return_matrix': cache_df,
                'asset_index_mapping': dict(zip(cache_df.columns, range(len(cache_df.columns)))),
                'timestamp': cache_path.stat().st_mtime
            }
        except Exception:
            return None
    return None


def _save_preprocessing_cache(cache_path: Path, daily_returns: pd.DataFrame, metadata: Dict):
    """Save preprocessing cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    daily_returns.to_parquet(cache_path)


def _load_batch_progress(progress_path: Path) -> Dict:
    """Load batch progress tracker."""
    if progress_path.exists():
        try:
            with open(progress_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        'total_batches': 0,
        'completed_batches': 0,
        'last_successful_batch_id': -1,
        'timestamp': None
    }


def _save_batch_progress(progress_path: Path, progress: Dict):
    """Save batch progress tracker."""
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress['timestamp'] = datetime.now().isoformat()
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)


def _process_portfolio_batch(
    batch_portfolios: List[Tuple[int, Union[int, str], pd.Series]],
    daily_returns: pd.DataFrame,
    asset_conditional_vols: Dict[Tuple[str, int], pd.Series],
    covariance_matrix: pd.DataFrame,
    confidence_levels: List[float],
    horizons: List[int],
    estimation_windows: List[int],
    dist: str,
    use_dynamic_covariance: bool,
    n_jobs: int = 1,
    chunk_size: int = 250,
    enable_parallel: bool = True,
    use_gpu: bool = True
) -> List[Dict]:
    """
    Process a batch of portfolios with optional parallelization.
    
    Args:
        batch_portfolios: List of (portfolio_idx, portfolio_id, portfolio_weights) tuples
        daily_returns: DataFrame of daily returns
        asset_conditional_vols: Dictionary mapping (asset, window) to conditional volatility Series
        covariance_matrix: Covariance matrix for portfolio volatility
        confidence_levels: List of confidence levels
        horizons: List of horizons
        estimation_windows: List of estimation windows
        dist: Distribution assumption
        use_dynamic_covariance: Whether to use dynamic covariance
        n_jobs: Number of parallel workers (1 for sequential)
        chunk_size: Chunk size for parallel processing
        enable_parallel: Whether to enable parallelization
        
    Returns:
        List of result dictionaries
    """
    if not enable_parallel or n_jobs <= 1 or len(batch_portfolios) <= chunk_size:
        # Sequential processing
        results = []
        for portfolio_data in batch_portfolios:
            portfolio_results = _process_single_portfolio_worker(
                portfolio_data,
                daily_returns,
                asset_conditional_vols,
                covariance_matrix,
                confidence_levels,
                horizons,
                estimation_windows,
                dist,
                use_dynamic_covariance,
                use_gpu=use_gpu
            )
            results.extend(portfolio_results)
        return results
    
    # Parallel processing
    worker_func = partial(
        _process_single_portfolio_worker,
        daily_returns=daily_returns,
        asset_conditional_vols=asset_conditional_vols,
        covariance_matrix=covariance_matrix,
        confidence_levels=confidence_levels,
        horizons=horizons,
        estimation_windows=estimation_windows,
        dist=dist,
        use_dynamic_covariance=use_dynamic_covariance,
        use_gpu=use_gpu
    )
    
    results = []
    
    try:
        with Pool(processes=n_jobs) as pool:
            # Use imap for better memory management with chunking
            portfolio_results_iter = pool.imap(worker_func, batch_portfolios, chunksize=chunk_size)
            
            for portfolio_results in portfolio_results_iter:
                if portfolio_results:
                    results.extend(portfolio_results)
    except Exception as e:
        # Graceful degradation: fall back to sequential processing
        warnings.warn(f"Parallel processing failed: {e}. Falling back to sequential processing.")
        results = []
        for portfolio_data in batch_portfolios:
            try:
                portfolio_results = _process_single_portfolio_worker(
                    portfolio_data,
                    daily_returns,
                    asset_conditional_vols,
                    covariance_matrix,
                    confidence_levels,
                    horizons,
                    estimation_windows,
                    dist,
                    use_dynamic_covariance,
                    use_gpu=use_gpu
                )
                results.extend(portfolio_results)
            except Exception:
                continue
    
    return results


def _save_batch_results(
    results: List[Dict],
    metrics_path: Path,
    is_first_batch: bool
):
    """Save batch results to parquet (append mode)."""
    if len(results) == 0:
        return
    
    results_df = pd.DataFrame(results)
    
    if is_first_batch or not metrics_path.exists():
        results_df.to_parquet(metrics_path, index=False)
    else:
        try:
            existing_df = pd.read_parquet(metrics_path)
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            combined_df.to_parquet(metrics_path, index=False)
            del existing_df, combined_df
        except Exception:
            results_df.to_parquet(metrics_path, index=False)


def _generate_schema_json(schema_path: Path, config: Dict):
    """Generate schema JSON describing the parquet output."""
    schema = {
        'description': 'GARCH(1,1) VaR/CVaR evaluation results',
        'columns': {
            'portfolio_id': {
                'dtype': 'int64',
                'description': 'Portfolio identifier',
                'unit': None
            },
            'date': {
                'dtype': 'datetime64[ns]',
                'description': 'Evaluation date',
                'unit': None
            },
            'confidence_level': {
                'dtype': 'float64',
                'description': 'VaR/CVaR confidence level (e.g., 0.99 for 99%)',
                'unit': None
            },
            'horizon': {
                'dtype': 'int64',
                'description': 'Forecast horizon in days',
                'unit': 'days'
            },
            'estimation_window': {
                'dtype': 'int64',
                'description': 'GARCH estimation window size',
                'unit': 'days'
            },
            'var': {
                'dtype': 'float64',
                'description': 'Value at Risk',
                'unit': 'portfolio return units'
            },
            'cvar': {
                'dtype': 'float64',
                'description': 'Conditional Value at Risk',
                'unit': 'portfolio return units'
            },
            'hit_rate': {
                'dtype': 'float64',
                'description': 'Proportion of VaR violations',
                'unit': None
            },
            'violation_ratio': {
                'dtype': 'float64',
                'description': 'Actual violations / Expected violations',
                'unit': None
            },
            'kupiec_statistic': {
                'dtype': 'float64',
                'description': 'Kupiec unconditional coverage test statistic',
                'unit': None
            },
            'christoffersen_statistic': {
                'dtype': 'float64',
                'description': 'Christoffersen independence test statistic',
                'unit': None
            },
            'traffic_light_zone': {
                'dtype': 'object',
                'description': 'Basel traffic light zone (green/yellow/red)',
                'unit': None
            },
            'mean_exceedance': {
                'dtype': 'float64',
                'description': 'Mean exceedance beyond VaR',
                'unit': 'portfolio return units'
            },
            'max_exceedance': {
                'dtype': 'float64',
                'description': 'Maximum exceedance beyond VaR',
                'unit': 'portfolio return units'
            },
            'std_exceedance': {
                'dtype': 'float64',
                'description': 'Standard deviation of exceedances',
                'unit': 'portfolio return units'
            },
            'cvar_mean_exceedance': {
                'dtype': 'float64',
                'description': 'Mean exceedance beyond CVaR',
                'unit': 'portfolio return units'
            },
            'num_active_assets': {
                'dtype': 'int64',
                'description': 'Number of assets with non-zero weights',
                'unit': None
            },
            'hhi_concentration': {
                'dtype': 'float64',
                'description': 'Herfindahl-Hirschman Index (concentration measure)',
                'unit': None
            },
            'effective_number_of_assets': {
                'dtype': 'float64',
                'description': 'Effective number of assets (1/HHI)',
                'unit': None
            },
            'skewness': {
                'dtype': 'float64',
                'description': 'Skewness of portfolio returns',
                'unit': None
            },
            'kurtosis': {
                'dtype': 'float64',
                'description': 'Excess kurtosis of portfolio returns',
                'unit': None
            },
            'jarque_bera_p_value': {
                'dtype': 'float64',
                'description': 'Jarque-Bera normality test p-value',
                'unit': None
            },
            'runtime_per_portfolio_ms': {
                'dtype': 'float64',
                'description': 'Runtime per portfolio in milliseconds',
                'unit': 'milliseconds'
            },
            'garch_fitting_time_ms': {
                'dtype': 'float64',
                'description': 'GARCH fitting time in milliseconds',
                'unit': 'milliseconds'
            },
            'cache_hit_ratio': {
                'dtype': 'float64',
                'description': 'GARCH parameter cache hit ratio',
                'unit': None
            }
        },
        'primary_key': ['portfolio_id', 'date', 'confidence_level', 'horizon'],
        'metadata': {
            'task': config.get('task', 'garch_1_1_var_cvar_evaluation_batched'),
            'generated_at': datetime.now().isoformat(),
            'garch_settings': config.get('garch_settings', {})
        }
    }
    
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2, default=str)


def evaluate_garch_var_cvar(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict] = None,
    n_jobs: Optional[int] = None,
    max_portfolios: Optional[int] = None
) -> pd.DataFrame:
    """
    Main function to evaluate VaR and CVaR using GARCH(1,1) volatility forecasting.
    
    Implements batched execution with asset-level GARCH fitting and caching.
    Supports multi-level parallelization:
    - Asset-level GARCH fitting (parallel across asset/window combinations)
    - Within-batch portfolio processing (parallel across portfolios in a batch)
    - Optional batch-level parallelization with shard writing
    
    Args:
        config_path: Path to JSON configuration file
        config_dict: Configuration dictionary (if not loading from file)
        n_jobs: Number of parallel workers (None for auto-detection from config)
        max_portfolios: Maximum number of portfolios to process (None for all)
        
    Returns:
        DataFrame with all computed metrics
    """
    # Load configuration
    if config_dict is None:
        if config_path is None:
            config_path = Path(__file__).parent / "llm.json"
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = config_dict
    
    # Get project root
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    
    # Get paths
    panel_price_path = project_root / config['inputs']['panel_price_path']
    portfolio_weights_path = project_root / config['inputs']['portfolio_weights_path']
    
    # Adjust path if needed
    if not panel_price_path.exists() and 'preprocessed' in str(panel_price_path):
        panel_price_path = project_root / str(config['inputs']['panel_price_path']).replace('preprocessed', 'processed')
    
    print("=" * 80)
    print("GARCH(1,1) VOLATILITY FORECASTING FOR VAR/CVAR EVALUATION (BATCHED)")
    print("=" * 80)
    print(f"\nLoading data...")
    print(f"  Panel prices: {panel_price_path}")
    print(f"  Portfolio weights: {portfolio_weights_path}")
    
    # Load data
    prices = load_panel_prices(panel_price_path)
    portfolio_weights_df = load_portfolio_weights(portfolio_weights_path)
    
    data_period = f"{prices.index.min().strftime('%Y-%m-%d')} to {prices.index.max().strftime('%Y-%m-%d')}"
    
    print(f"\nLoaded:")
    print(f"  Prices: {len(prices)} dates, {len(prices.columns)} assets")
    print(f"  Portfolios: {len(portfolio_weights_df)} portfolios")
    print(f"  Data period: {data_period}")
    
    # Get GARCH settings
    garch_settings = config.get('garch_settings', {})
    return_type = garch_settings.get('return_type', 'log')
    confidence_levels = garch_settings.get('confidence_levels', [0.99])
    estimation_windows = garch_settings.get('estimation_windows', [500])
    p = garch_settings.get('p', 1)
    q = garch_settings.get('q', 1)
    dist = garch_settings.get('distribution', 'normal')
    mean_model_config = garch_settings.get('mean_model', {})
    use_mean_model = mean_model_config.get('enabled', False)
    mean = 'AR' if use_mean_model else 'Zero'
    vol = garch_settings.get('model_type', 'GARCH')
    fallback_long_run_variance = garch_settings.get('fallback_long_run_variance', True)
    
    # Get horizons
    horizons_config = garch_settings.get('horizons', {})
    if isinstance(horizons_config, dict):
        base_horizon = horizons_config.get('base_horizon', 1)
        scaled_horizons = horizons_config.get('scaled_horizons', [])
        scaling_rule = horizons_config.get('scaling_rule', 'sqrt_time')
        horizons = compute_horizons(base_horizon, scaled_horizons, scaling_rule)
    else:
        horizons = horizons_config if isinstance(horizons_config, list) else [1, 10]
    
    # Get batch settings
    batch_config = config.get('batch_execution', {})
    batch_size = batch_config.get('batch_size', 1000)
    resume_supported = batch_config.get('resume_supported', True)
    
    # Get parallelization settings
    parallel_config = config.get('parallelization', {})
    parallel_enabled = parallel_config.get('enabled', True)
    parallel_levels = parallel_config.get('levels', {})
    
    # Within-batch parallelization settings
    within_batch_config = parallel_levels.get('within_batch_level', {})
    within_batch_enabled = within_batch_config.get('enabled', True) and parallel_enabled
    
    # Chunking settings
    chunking_config = parallel_config.get('chunking', {})
    portfolio_chunk_size = chunking_config.get('portfolio_chunk_size', 250)
    
    # I/O safety settings
    io_safety_config = parallel_config.get('io_safety', {})
    use_shards = io_safety_config.get('write_strategy', 'append_per_batch') == 'write_per_batch_files_then_merge'
    
    # Get worker count
    n_workers = _get_worker_count(config, n_jobs)
    
    # GPU settings
    gpu_config = parallel_config.get('gpu', {})
    use_gpu = gpu_config.get('enabled', True) and GPU_AVAILABLE
    gpu_batch_size = gpu_config.get('batch_size', 1000)  # Number of portfolios to process in GPU batch
    
    # Asset-level parallelization settings (needed for print statement)
    asset_level_config = parallel_levels.get('asset_level_fitting', {})
    asset_level_parallel = asset_level_config.get('enabled', True) and parallel_enabled
    
    # Get cache settings
    cache_config = config.get('cache_strategy', {})
    cache_root = project_root / cache_config.get('cache_root', 'cache/')
    
    preprocessing_cache_path = cache_root / cache_config.get('preprocessing_cache', {}).get('path', 'preprocessing_state.parquet')
    garch_cache_path = cache_root / cache_config.get('garch_parameter_cache', {}).get('path', 'garch_parameters.parquet')
    conditional_vol_cache_path = cache_root / cache_config.get('conditional_volatility_cache', {}).get('path', 'garch_conditional_volatility.parquet')
    progress_path = cache_root / cache_config.get('batch_progress_tracker', {}).get('path', 'batch_progress.json')
    
    # Get output paths
    outputs = config.get('outputs', {})
    metrics_path = project_root / outputs.get('metrics_parquet', {}).get('path', 'results/classical_risk/garch_var_cvar_metrics.parquet')
    schema_path = project_root / outputs.get('parquet_schema_json', {}).get('path', 'results/classical_risk/garch_var_cvar_schema.json')
    report_path = project_root / outputs.get('summary_report', {}).get('path', 'results/classical_risk/garch_var_cvar_report.md')
    
    print(f"\nGARCH Settings:")
    print(f"  Model: GARCH({p},{q})")
    print(f"  Distribution: {dist}")
    print(f"  Mean model: {mean}")
    print(f"  Confidence levels: {confidence_levels}")
    print(f"  Horizons: {horizons} days")
    print(f"  Estimation windows: {estimation_windows} days")
    print(f"  Batch size: {batch_size:,} portfolios")
    print(f"\nParallelization Settings:")
    print(f"  Enabled: {parallel_enabled}")
    print(f"  Workers: {n_workers}")
    print(f"  Asset-level fitting: {'Parallel' if asset_level_parallel else 'Sequential'}")
    print(f"  Within-batch processing: {'Parallel' if within_batch_enabled else 'Sequential'}")
    print(f"  Portfolio chunk size: {portfolio_chunk_size}")
    print(f"  Shard writing: {'Enabled' if use_shards else 'Disabled'}")
    print(f"\nGPU Settings:")
    print(f"  GPU Available: {GPU_AVAILABLE}")
    if GPU_AVAILABLE:
        print(f"  GPU Device: {GPU_DEVICE}")
        print(f"  GPU Count: {GPU_COUNT}")
        try:
            import torch
            print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        except:
            pass
    print(f"  GPU Acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    if use_gpu:
        print(f"  GPU Batch Size: {gpu_batch_size}")
    
    # Compute daily returns
    print(f"\nComputing daily returns (method: {return_type})...")
    daily_returns = compute_daily_returns(prices, method=return_type)
    print(f"  Daily returns: {len(daily_returns)} dates")
    
    # Step 1: Compute asset-level conditional volatilities (once for all portfolios)
    print(f"\n{'='*80}")
    print("STEP 1: Computing Asset-Level Conditional Volatilities")
    print(f"{'='*80}")
    print(f"  Fitting GARCH models for each asset/window combination")
    print(f"  Conditional volatilities will be cached and reused across all batches")
    
    start_time_asset_level = time.time()
    
    # Initialize GARCH parameter cache
    garch_cache = GARCHParameterCache(garch_cache_path) if cache_config.get('garch_parameter_cache', {}).get('enabled', True) else None
    
    # Compute all asset conditional volatilities (with optional parallelization)
    asset_conditional_vols = compute_all_asset_conditional_volatilities(
        daily_returns,
        estimation_windows,
        p=p,
        q=q,
        dist=dist,
        mean=mean,
        vol=vol,
        fallback_long_run_variance=fallback_long_run_variance,
        cache=garch_cache,
        n_jobs=n_workers if asset_level_parallel else 1
    )
    
    asset_level_runtime = time.time() - start_time_asset_level
    
    print(f"  Completed in {asset_level_runtime/60:.2f} minutes ({asset_level_runtime:.2f} seconds)")
    print(f"  Computed conditional volatilities for {len(asset_conditional_vols)} asset/window combinations")
    if garch_cache:
        print(f"  Cache hit ratio: {garch_cache.get_hit_ratio():.2%}")
        garch_cache.save_cache()
    
    # Compute covariance matrix (once) - GPU-accelerated if available
    print(f"\nComputing covariance matrix for portfolio volatility projection...")
    if use_gpu and GPU_AVAILABLE:
        try:
            import torch
            # Convert returns to GPU tensor for faster computation
            returns_tensor = torch.tensor(daily_returns.values, device=GPU_DEVICE, dtype=torch.float32)
            # Compute covariance on GPU
            returns_centered = returns_tensor - returns_tensor.mean(dim=0, keepdim=True)
            cov_tensor = torch.mm(returns_centered.t(), returns_centered) / (returns_tensor.shape[0] - 1)
            covariance_matrix = pd.DataFrame(
                cov_tensor.cpu().numpy(),
                index=daily_returns.columns,
                columns=daily_returns.columns
            )
            print(f"  Covariance matrix: {covariance_matrix.shape} (computed on GPU)")
        except Exception as e:
            warnings.warn(f"GPU covariance computation failed: {e}. Falling back to CPU.")
            covariance_matrix = daily_returns.cov()
            print(f"  Covariance matrix: {covariance_matrix.shape} (computed on CPU)")
    else:
        covariance_matrix = daily_returns.cov()
        print(f"  Covariance matrix: {covariance_matrix.shape} (computed on CPU)")
    
    # Get portfolio volatility projection settings
    portfolio_projection_config = config.get('modules', {}).get('portfolio_volatility_projection', {})
    use_dynamic_covariance = portfolio_projection_config.get('use_dynamic_covariance', False)
    
    # Step 2: Process portfolios in batches
    print(f"\n{'='*80}")
    print("STEP 2: Processing Portfolios in Batches")
    print(f"{'='*80}")
    
    # Limit portfolios if specified
    num_portfolios_total = len(portfolio_weights_df)
    if max_portfolios is not None and max_portfolios > 0:
        num_portfolios = min(num_portfolios_total, max_portfolios)
        portfolio_weights_df = portfolio_weights_df.iloc[:num_portfolios]
        print(f"  Limiting to first {num_portfolios:,} portfolios (out of {num_portfolios_total:,} total)")
    else:
        num_portfolios = num_portfolios_total
        print(f"  Processing all {num_portfolios:,} portfolios")
    
    # Calculate batches
    num_batches = (num_portfolios + batch_size - 1) // batch_size
    print(f"  Total batches: {num_batches}")
    print(f"  Batch size: {batch_size:,} portfolios")
    
    # Load batch progress
    progress = _load_batch_progress(progress_path)
    start_batch = progress.get('last_successful_batch_id', -1) + 1 if resume_supported else 0
    
    if start_batch > 0:
        print(f"  Resuming from batch {start_batch} (last successful: {progress.get('last_successful_batch_id', -1)})")
    
    # Prepare portfolio data
    portfolio_data_list = [
        (idx, portfolio_id, portfolio_weights)
        for idx, (portfolio_id, portfolio_weights) in enumerate(portfolio_weights_df.iterrows())
    ]
    
    # Process batches
    start_time_total = time.time()
    all_runtimes = []
    
    for batch_idx in range(start_batch, num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_portfolios)
        
        batch_portfolios = portfolio_data_list[batch_start:batch_end]
        
        print(f"\n  Processing batch {batch_idx + 1}/{num_batches} (portfolios {batch_start:,}-{batch_end:,})...")
        batch_start_time = time.time()
        
        # Process batch (with optional within-batch parallelization and GPU acceleration)
        batch_results = _process_portfolio_batch(
            batch_portfolios,
            daily_returns,
            asset_conditional_vols,
            covariance_matrix,
            confidence_levels,
            horizons,
            estimation_windows,
            dist,
            use_dynamic_covariance,
            n_jobs=n_workers if within_batch_enabled else 1,
            chunk_size=portfolio_chunk_size,
            enable_parallel=within_batch_enabled,
            use_gpu=use_gpu
        )
        
        batch_runtime = time.time() - batch_start_time
        all_runtimes.append(batch_runtime)
        
        # Save batch results (either to shard or main file)
        if use_shards:
            # Write to shard file
            shard_path = project_root / outputs.get('optional_parallel_shards', {}).get('path_pattern', 
                'results/classical_risk/shards/garch_var_cvar_metrics_batch_{batch_id}.parquet').format(batch_id=batch_idx)
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            _save_batch_results(batch_results, shard_path, is_first_batch=True)
            print(f"    Saved to shard: {shard_path}")
        else:
            # Append to main file
            is_first_batch = (batch_idx == start_batch)
            _save_batch_results(batch_results, metrics_path, is_first_batch)
        
        # Update progress
        progress['total_batches'] = num_batches
        progress['completed_batches'] = batch_idx + 1
        progress['last_successful_batch_id'] = batch_idx
        _save_batch_progress(progress_path, progress)
        
        print(f"    Batch completed in {batch_runtime:.2f} seconds")
        print(f"    Results: {len(batch_results)} portfolio-configuration combinations")
        print(f"    Saved to: {metrics_path}")
        
        # Clear memory
        del batch_results
        gc.collect()
        
        # Clear GPU cache periodically
        if use_gpu and GPU_AVAILABLE:
            try:
                import torch
                if batch_idx % 10 == 0:  # Clear every 10 batches
                    torch.cuda.empty_cache()
            except:
                pass
    
    total_runtime = time.time() - start_time_total
    
    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"  Total runtime: {total_runtime/60:.2f} minutes ({total_runtime:.2f} seconds)")
    print(f"  Asset-level GARCH fitting: {asset_level_runtime/60:.2f} minutes")
    print(f"  Portfolio processing: {(total_runtime - asset_level_runtime)/60:.2f} minutes")
    
    # Merge shards if using shard writing
    if use_shards:
        print(f"\nMerging shard files...")
        shard_pattern = outputs.get('optional_parallel_shards', {}).get('path_pattern', 
            'results/classical_risk/shards/garch_var_cvar_metrics_batch_{batch_id}.parquet')
        shard_dir = project_root / shard_pattern.split('{')[0].rsplit('/', 1)[0]
        
        shard_files = sorted(shard_dir.glob('garch_var_cvar_metrics_batch_*.parquet'))
        if len(shard_files) > 0:
            shard_dfs = []
            for shard_file in shard_files:
                try:
                    shard_df = pd.read_parquet(shard_file)
                    shard_dfs.append(shard_df)
                except Exception as e:
                    warnings.warn(f"Failed to read shard {shard_file}: {e}")
            
            if len(shard_dfs) > 0:
                merged_df = pd.concat(shard_dfs, ignore_index=True)
                # Sort by primary key
                primary_key = outputs.get('metrics_parquet', {}).get('primary_key', ['portfolio_id', 'date', 'confidence_level', 'horizon'])
                merged_df = merged_df.sort_values(primary_key)
                merged_df.to_parquet(metrics_path, index=False)
                print(f"  Merged {len(shard_files)} shards into {metrics_path}")
                print(f"  Total rows: {len(merged_df)}")
    
    # Generate schema JSON
    print(f"\nGenerating schema JSON...")
    _generate_schema_json(schema_path, config)
    print(f"  Saved: {schema_path}")
    
    # Load final results and generate report
    print(f"\nGenerating summary report...")
    if metrics_path.exists():
        results_df = pd.read_parquet(metrics_path)
        
        # Add runtime metrics
        if len(all_runtimes) > 0:
            runtime_metrics = compute_runtime_metrics(all_runtimes)
            for key, value in runtime_metrics.items():
                results_df[key] = value
        
        # Generate report
        generate_report(
            results_df,
            report_path,
            garch_settings=garch_settings,
            report_sections=outputs.get('summary_report', {}).get('content', [])
        )
        print(f"  Saved: {report_path}")
        print(f"  Total results: {len(results_df)} rows")
    else:
        results_df = pd.DataFrame()
        print(f"  Warning: No results file found at {metrics_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return results_df


def main():
    """Main entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate GARCH(1,1) Volatility Forecasting VaR/CVaR for portfolios (batched)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration JSON file (default: llm.json in same directory)'
    )
    parser.add_argument(
        '--max-portfolios',
        type=int,
        default=None,
        help='Maximum number of portfolios to process (default: all)'
    )
    
    args = parser.parse_args()
    
    results_df = evaluate_garch_var_cvar(
        config_path=args.config,
        max_portfolios=args.max_portfolios
    )
    
    print(f"\nResults summary:")
    print(f"  Total rows: {len(results_df)}")
    print(f"  Columns: {len(results_df.columns)}")
    if len(results_df) > 0:
        print(f"\nFirst few rows:")
        print(results_df.head())


if __name__ == "__main__":
    main()
