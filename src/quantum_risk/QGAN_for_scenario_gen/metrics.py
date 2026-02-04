"""
Metrics computation for QGAN scenario generation evaluation.

Includes distribution fidelity, tail metrics, stylized facts, and quantum-specific metrics.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple
from scipy.stats import wasserstein_distance


def compute_wasserstein_distance(real: np.ndarray, generated: np.ndarray) -> float:
    """Compute Wasserstein distance between real and generated distributions."""
    try:
        return wasserstein_distance(real, generated)
    except:
        return np.nan


def compute_ks_statistic(real: np.ndarray, generated: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov statistic."""
    try:
        ks_stat, _ = stats.ks_2samp(real, generated)
        return ks_stat
    except:
        return np.nan


def compute_js_divergence(real: np.ndarray, generated: np.ndarray, bins: int = 50) -> float:
    """Compute Jensen-Shannon divergence."""
    try:
        # Create histograms
        min_val = min(real.min(), generated.min())
        max_val = max(real.max(), generated.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        hist_real, _ = np.histogram(real, bins=bin_edges, density=True)
        hist_gen, _ = np.histogram(generated, bins=bin_edges, density=True)
        
        # Normalize
        hist_real = hist_real / (hist_real.sum() + 1e-10)
        hist_gen = hist_gen / (hist_gen.sum() + 1e-10)
        
        # JS divergence
        js = jensenshannon(hist_real, hist_gen)
        return js
    except:
        return np.nan


def compute_moment_errors(real: np.ndarray, generated: np.ndarray) -> Dict[str, float]:
    """Compute errors in mean, variance, skewness, and kurtosis."""
    try:
        real_mean = np.mean(real)
        gen_mean = np.mean(generated)
        mean_error = abs(real_mean - gen_mean)
        
        real_var = np.var(real)
        gen_var = np.var(generated)
        var_error = abs(real_var - gen_var)
        
        real_skew = stats.skew(real)
        gen_skew = stats.skew(generated)
        skew_error = abs(real_skew - gen_skew)
        
        real_kurt = stats.kurtosis(real)
        gen_kurt = stats.kurtosis(generated)
        kurt_error = abs(real_kurt - gen_kurt)
        
        return {
            'moment_error_mean': mean_error,
            'moment_error_var': var_error,
            'moment_error_skew': skew_error,
            'moment_error_kurt': kurt_error
        }
    except:
        return {
            'moment_error_mean': np.nan,
            'moment_error_var': np.nan,
            'moment_error_skew': np.nan,
            'moment_error_kurt': np.nan
        }


def compute_distribution_fidelity_metrics(
    real_returns: np.ndarray,
    generated_returns: np.ndarray
) -> Dict[str, float]:
    """
    Compute distribution fidelity metrics.
    
    Args:
        real_returns: Real return data
        generated_returns: Generated return scenarios
        
    Returns:
        Dict with fidelity metrics
    """
    metrics = {}
    
    # Wasserstein distance
    metrics['wasserstein_distance'] = compute_wasserstein_distance(real_returns, generated_returns)
    
    # KS statistic
    metrics['ks_statistic'] = compute_ks_statistic(real_returns, generated_returns)
    
    # JS divergence
    metrics['js_divergence'] = compute_js_divergence(real_returns, generated_returns)
    
    # Moment errors
    moment_errors = compute_moment_errors(real_returns, generated_returns)
    metrics.update(moment_errors)
    
    return metrics


def compute_tail_metrics(
    real_losses: np.ndarray,
    generated_losses: np.ndarray,
    confidence_levels: List[float] = [0.95, 0.99]
) -> Dict[str, float]:
    """
    Compute tail risk metrics (VaR/CVaR errors).
    
    Args:
        real_losses: Real loss data
        generated_losses: Generated loss scenarios
        confidence_levels: Confidence levels for VaR/CVaR
        
    Returns:
        Dict with tail metrics
    """
    metrics = {}
    
    for conf in confidence_levels:
        alpha = 1 - conf
        
        # Real VaR/CVaR
        real_var = np.quantile(real_losses, conf)
        real_tail = real_losses[real_losses >= real_var]
        real_cvar = np.mean(real_tail) if len(real_tail) > 0 else real_var
        
        # Generated VaR/CVaR
        gen_var = np.quantile(generated_losses, conf)
        gen_tail = generated_losses[generated_losses >= gen_var]
        gen_cvar = np.mean(gen_tail) if len(gen_tail) > 0 else gen_var
        
        # Errors
        var_error = abs(real_var - gen_var)
        cvar_error = abs(real_cvar - gen_cvar)
        
        # Tail mass error
        real_tail_mass = np.mean(real_losses >= real_var)
        gen_tail_mass = np.mean(generated_losses >= gen_var)
        tail_mass_error = abs(real_tail_mass - gen_tail_mass)
        
        # Extreme quantile error
        extreme_quantile = 0.999
        real_extreme = np.quantile(real_losses, extreme_quantile)
        gen_extreme = np.quantile(generated_losses, extreme_quantile)
        extreme_quantile_error = abs(real_extreme - gen_extreme)
        
        conf_str = str(int(conf * 100))
        metrics[f'var_error_{conf_str}'] = var_error
        metrics[f'cvar_error_{conf_str}'] = cvar_error
        metrics[f'tail_mass_error_{conf_str}'] = tail_mass_error
        metrics[f'extreme_quantile_error_{conf_str}'] = extreme_quantile_error
    
    return metrics


def compute_stylized_facts_metrics(
    real_returns: np.ndarray,
    generated_returns: np.ndarray
) -> Dict[str, float]:
    """
    Compute stylized facts metrics (volatility clustering, leptokurtosis, etc.).
    
    Args:
        real_returns: Real return data
        generated_returns: Generated return scenarios
        
    Returns:
        Dict with stylized facts metrics
    """
    metrics = {}
    
    try:
        # Volatility clustering proxy (ACF of squared returns)
        real_squared = real_returns ** 2
        gen_squared = generated_returns ** 2
        
        # ACF at lag 1-5
        acf_real = []
        acf_gen = []
        for lag in range(1, 6):
            if len(real_squared) > lag:
                acf_real.append(np.corrcoef(real_squared[:-lag], real_squared[lag:])[0, 1])
            if len(gen_squared) > lag:
                acf_gen.append(np.corrcoef(gen_squared[:-lag], gen_squared[lag:])[0, 1])
        
        if len(acf_real) > 0 and len(acf_gen) > 0:
            metrics['acf_squared_returns_lag1_5'] = np.mean([abs(a - b) for a, b in zip(acf_real[:5], acf_gen[:5])])
        else:
            metrics['acf_squared_returns_lag1_5'] = np.nan
        
        # Leptokurtosis gap
        real_kurt = stats.kurtosis(real_returns)
        gen_kurt = stats.kurtosis(generated_returns)
        metrics['leptokurtosis_gap'] = abs(real_kurt - gen_kurt)
        
        # Downside skew preservation
        real_skew = stats.skew(real_returns)
        gen_skew = stats.skew(generated_returns)
        metrics['downside_skew_preservation'] = abs(real_skew - gen_skew)
        
        # Volatility clustering proxy (simplified)
        metrics['volatility_clustering_proxy'] = metrics.get('acf_squared_returns_lag1_5', np.nan)
        
    except Exception as e:
        metrics['acf_squared_returns_lag1_5'] = np.nan
        metrics['leptokurtosis_gap'] = np.nan
        metrics['downside_skew_preservation'] = np.nan
        metrics['volatility_clustering_proxy'] = np.nan
    
    return metrics


def compute_mode_collapse_score(generated_returns: np.ndarray, num_bins: int = 50) -> float:
    """
    Compute mode collapse score (ratio of unique bins).
    
    Args:
        generated_returns: Generated return scenarios
        num_bins: Number of bins for histogram
        
    Returns:
        Unique bin ratio (higher is better, indicates less mode collapse)
    """
    try:
        min_val = generated_returns.min()
        max_val = generated_returns.max()
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        hist, _ = np.histogram(generated_returns, bins=bin_edges)
        unique_bins = np.sum(hist > 0)
        unique_bin_ratio = unique_bins / num_bins
        return unique_bin_ratio
    except:
        return np.nan


def compute_quantum_specific_metrics(
    generator,
    training_result: Dict,
    qgan_settings: Dict
) -> Dict[str, float]:
    """
    Compute quantum-specific metrics.
    
    Args:
        generator: QuantumGenerator instance
        training_result: Training result dict with loss traces
        qgan_settings: QGAN settings dict
        
    Returns:
        Dict with quantum-specific metrics
    """
    metrics = {}
    
    try:
        # Circuit metrics
        metrics['generator_circuit_depth'] = float(generator.get_circuit_depth())
        metrics['generator_circuit_width'] = float(generator.num_qubits)
        metrics['num_qubits'] = float(generator.num_qubits)
        metrics['shots'] = float(qgan_settings.get('execution', {}).get('shots', 4096))
        
        # Loss traces (store as summary statistics)
        gen_losses = training_result.get('generator_losses', [])
        disc_losses = training_result.get('discriminator_losses', [])
        
        if len(gen_losses) > 0:
            metrics['generator_loss_final'] = float(gen_losses[-1])
            metrics['generator_loss_mean'] = float(np.mean(gen_losses))
            metrics['generator_loss_std'] = float(np.std(gen_losses))
            metrics['generator_loss_min'] = float(np.min(gen_losses))
            metrics['generator_loss_max'] = float(np.max(gen_losses))
            # Store full trace as list (will be serialized as string for parquet compatibility)
            trace = gen_losses[:100] if len(gen_losses) > 100 else gen_losses
            metrics['generator_loss_trace'] = str(trace)  # Convert to string for parquet storage
        else:
            metrics['generator_loss_final'] = np.nan
            metrics['generator_loss_mean'] = np.nan
            metrics['generator_loss_std'] = np.nan
            metrics['generator_loss_min'] = np.nan
            metrics['generator_loss_max'] = np.nan
            metrics['generator_loss_trace'] = '[]'
        
        if len(disc_losses) > 0:
            metrics['discriminator_loss_final'] = float(disc_losses[-1])
            metrics['discriminator_loss_mean'] = float(np.mean(disc_losses))
            metrics['discriminator_loss_std'] = float(np.std(disc_losses))
            metrics['discriminator_loss_min'] = float(np.min(disc_losses))
            metrics['discriminator_loss_max'] = float(np.max(disc_losses))
            trace = disc_losses[:100] if len(disc_losses) > 100 else disc_losses
            metrics['discriminator_loss_trace'] = str(trace)  # Convert to string for parquet storage
        else:
            metrics['discriminator_loss_final'] = np.nan
            metrics['discriminator_loss_mean'] = np.nan
            metrics['discriminator_loss_std'] = np.nan
            metrics['discriminator_loss_min'] = np.nan
            metrics['discriminator_loss_max'] = np.nan
            metrics['discriminator_loss_trace'] = '[]'
        
    except Exception as e:
        # Fill with NaN if computation fails
        metrics.update({
            'generator_circuit_depth': np.nan,
            'generator_circuit_width': np.nan,
            'num_qubits': np.nan,
            'shots': np.nan,
            'generator_loss_final': np.nan,
            'discriminator_loss_final': np.nan,
        })
    
    return metrics
