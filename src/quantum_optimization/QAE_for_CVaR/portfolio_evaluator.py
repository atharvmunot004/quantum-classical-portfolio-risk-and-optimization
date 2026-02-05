"""
Portfolio Evaluator Module for QAE CVaR.

Evaluates portfolios in batch using precomputed QAE artifacts.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from scipy import stats

from .distribution_fitting import (
    fit_multivariate_normal,
    discretize_portfolio_loss_distribution,
    DistributionParams
)
from .qae_circuits import (
    estimate_cdf_qae,
    estimate_tail_expectation_qae,
    QAEResult
)
from .precompute_registry import PrecomputeRegistry


class PortfolioEvaluator:
    """
    Evaluates portfolios using QAE-based CVaR estimation.
    
    Uses precomputed quantum artifacts per asset set and reuses them
    across portfolios sharing the same asset composition.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        registry: PrecomputeRegistry,
        num_state_qubits: int = 6,
        estimation_window: int = 252,
        epsilon_target: float = 0.01,
        confidence_alpha: float = 0.05,
        shots: int = 2000,
        use_shrinkage: bool = True
    ):
        """
        Initialize portfolio evaluator.
        
        Args:
            returns: DataFrame of returns
            registry: Precompute registry for caching
            num_state_qubits: Number of qubits for state encoding
            estimation_window: Window size for distribution estimation
            epsilon_target: Target precision for QAE
            confidence_alpha: Confidence level for QAE
            shots: Number of shots for QAE
            use_shrinkage: Whether to use shrinkage for covariance
        """
        self.returns = returns
        self.registry = registry
        self.num_state_qubits = num_state_qubits
        self.estimation_window = estimation_window
        self.epsilon_target = epsilon_target
        self.confidence_alpha = confidence_alpha
        self.shots = shots
        self.use_shrinkage = use_shrinkage
    
    def precompute_quantum_risk_per_asset_set(
        self,
        asset_sets: List[Tuple[str, ...]],
        confidence_levels: List[float]
    ) -> Dict:
        """
        Precompute quantum CVaR artifacts for each unique asset set.
        
        Args:
            asset_sets: List of asset set tuples
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary with precomputation results
        """
        results = {
            'distribution_params': {},
            'var_thresholds': {},
            'qae_tail_expectations': {},
            'qae_results': {}
        }
        
        unique_asset_sets = list(set(asset_sets))
        
        print(f"Precomputing quantum risk for {len(unique_asset_sets)} unique asset sets...")
        print(f"  Confidence levels: {confidence_levels}")
        print(f"  Total precomputations: {len(unique_asset_sets)} asset sets × {len(confidence_levels)} confidence levels = {len(unique_asset_sets) * len(confidence_levels)}")
        print()
        
        import sys
        import time
        
        for idx, asset_set in enumerate(unique_asset_sets, 1):
            start_time = time.time()
            print(f"  [{idx}/{len(unique_asset_sets)}] Processing asset set: {asset_set}")
            sys.stdout.flush()
            
            # Fit distribution
            try:
                dist_params = fit_multivariate_normal(
                    self.returns,
                    asset_set,
                    self.estimation_window,
                    use_shrinkage=self.use_shrinkage
                )
                
                # Store distribution params
                results['distribution_params'][asset_set] = {
                    'mean': dist_params.mean,
                    'cov': dist_params.cov,
                    'asset_set': asset_set
                }
                self.registry.store_distribution_params(asset_set, results['distribution_params'][asset_set])
                
                # For each confidence level, compute VaR and tail expectation
                for confidence_level in confidence_levels:
                    # Use equal weights for precomputation (representative portfolio)
                    weights = np.ones(len(asset_set)) / len(asset_set)
                    
                    # Discretize distribution
                    probs, bin_centers = discretize_portfolio_loss_distribution(
                        dist_params,
                        weights,
                        num_state_qubits=self.num_state_qubits
                    )
                    
                    # Find VaR using bisection on CDF
                    var_threshold = self._find_var_bisection(
                        probs,
                        bin_centers,
                        confidence_level
                    )
                    
                    results['var_thresholds'][(asset_set, confidence_level)] = var_threshold
                    self.registry.store_var_threshold(asset_set, confidence_level, var_threshold)
                    
                    # Compute tail expectation using QAE
                    tail_expectation, qae_result = estimate_tail_expectation_qae(
                        probs,
                        bin_centers,
                        var_threshold,
                        epsilon_target=self.epsilon_target,
                        alpha=self.confidence_alpha,
                        shots=self.shots
                    )
                    
                    results['qae_tail_expectations'][(asset_set, confidence_level)] = tail_expectation
                    results['qae_results'][(asset_set, confidence_level)] = qae_result
                    self.registry.store_qae_tail_expectation(asset_set, confidence_level, tail_expectation)
            
            except Exception as e:
                print(f"    Error processing asset set {asset_set}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            elapsed = time.time() - start_time
            print(f"    Completed in {elapsed:.2f}s")
            sys.stdout.flush()
            
            # Progress checkpoint every 50 asset sets
            if idx % 50 == 0:
                print(f"\n  [CHECKPOINT] Processed {idx}/{len(unique_asset_sets)} asset sets ({idx/len(unique_asset_sets)*100:.1f}%)")
                sys.stdout.flush()
        
        return results
    
    def _find_var_bisection(
        self,
        probs: np.ndarray,
        bin_centers: np.ndarray,
        confidence_level: float,
        tol: float = 0.002,
        max_iter: int = 25
    ) -> float:
        """
        Find VaR using bisection on CDF.
        
        Args:
            probs: Probability vector
            bin_centers: Loss values at bin centers
            confidence_level: Confidence level
            tol: Tolerance for bisection
            max_iter: Maximum iterations
            
        Returns:
            VaR threshold
        """
        target_prob = 1 - confidence_level
        
        # Binary search on bin_centers
        low_idx = 0
        high_idx = len(bin_centers) - 1
        
        for _ in range(max_iter):
            mid_idx = (low_idx + high_idx) // 2
            threshold = bin_centers[mid_idx]
            
            # Compute CDF at threshold
            cdf_value = probs[:mid_idx+1].sum()
            
            if abs(cdf_value - target_prob) < tol:
                return threshold
            
            if cdf_value < target_prob:
                low_idx = mid_idx + 1
            else:
                high_idx = mid_idx - 1
            
            if low_idx > high_idx:
                break
        
        # Return final threshold
        final_idx = (low_idx + high_idx) // 2
        return bin_centers[final_idx]
    
    def evaluate_portfolio_cvar(
        self,
        weights: pd.Series,
        confidence_level: float
    ) -> Dict:
        """
        Evaluate CVaR for a single portfolio using precomputed artifacts.
        
        Args:
            weights: Portfolio weights (Series with asset names as index)
            confidence_level: Confidence level
            
        Returns:
            Dictionary with CVaR and related metrics
        """
        # Get asset set
        asset_set = tuple(sorted(weights[weights > 1e-10].index))
        
        # Get precomputed distribution params
        dist_params_dict = self.registry.get_distribution_params(asset_set)
        if dist_params_dict is None:
            raise ValueError(f"Distribution params not found for asset set {asset_set}")
        
        # Reconstruct distribution params
        from .distribution_fitting import DistributionParams
        dist_params = DistributionParams(
            mean=dist_params_dict['mean'],
            cov=dist_params_dict['cov'],
            asset_set=asset_set,
            estimation_window=self.estimation_window
        )
        
        # Get weights aligned with asset set
        weights_aligned = weights[list(asset_set)].values
        
        # Discretize distribution with actual weights
        probs, bin_centers = discretize_portfolio_loss_distribution(
            dist_params,
            weights_aligned,
            num_state_qubits=self.num_state_qubits
        )
        
        # Find VaR
        var_threshold = self._find_var_bisection(
            probs,
            bin_centers,
            confidence_level
        )
        
        # Compute tail expectation
        tail_expectation, qae_result = estimate_tail_expectation_qae(
            probs,
            bin_centers,
            var_threshold,
            epsilon_target=self.epsilon_target,
            alpha=self.confidence_alpha,
            shots=self.shots
        )
        
        # CVaR = VaR + (1 / (1 - alpha)) * E[(L - VaR)^+]
        cvar = var_threshold + (1 / (1 - confidence_level)) * tail_expectation
        
        return {
            'var': var_threshold,
            'cvar': cvar,
            'tail_expectation': tail_expectation,
            'qae_point_estimate': qae_result.estimation,
            'qae_ci_width': qae_result.confidence_interval[1] - qae_result.confidence_interval[0],
            'num_grover_iterations': qae_result.num_iterations,
            'circuit_depth': qae_result.circuit_depth,
            'circuit_width': qae_result.circuit_width,
            'num_shots': qae_result.num_shots
        }
    
    def evaluate_portfolios_batch(
        self,
        portfolio_weights: pd.DataFrame,
        confidence_levels: List[float]
    ) -> pd.DataFrame:
        """
        Evaluate portfolios in batch.
        
        Args:
            portfolio_weights: DataFrame of portfolio weights (N portfolios x M assets)
            confidence_levels: List of confidence levels
            
        Returns:
            DataFrame with CVaR metrics for each portfolio
        """
        results = []
        
        import sys
        import time
        
        total_evaluations = len(portfolio_weights) * len(confidence_levels)
        print(f"Evaluating {len(portfolio_weights):,} portfolios with {len(confidence_levels)} confidence levels...")
        print(f"  Total evaluations: {total_evaluations:,}")
        print()
        sys.stdout.flush()
        
        eval_start_time = time.time()
        completed = 0
        
        for portfolio_idx, portfolio_id in enumerate(portfolio_weights.index, 1):
            weights = portfolio_weights.loc[portfolio_id]
            
            for conf_idx, confidence_level in enumerate(confidence_levels):
                try:
                    result = self.evaluate_portfolio_cvar(weights, confidence_level)
                    result['portfolio_id'] = portfolio_id
                    result['confidence_level'] = confidence_level
                    results.append(result)
                    completed += 1
                    
                    # Progress update every 1000 evaluations
                    if completed % 1000 == 0:
                        elapsed = time.time() - eval_start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        remaining = (total_evaluations - completed) / rate if rate > 0 else 0
                        print(f"  Progress: {completed:,}/{total_evaluations:,} ({completed/total_evaluations*100:.1f}%) | "
                              f"Rate: {rate:.1f} eval/s | ETA: {remaining/60:.1f} min")
                        sys.stdout.flush()
                        
                except Exception as e:
                    print(f"  Error evaluating portfolio {portfolio_id} at {confidence_level}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Portfolio-level checkpoint every 10000 portfolios
            if portfolio_idx % 10000 == 0:
                elapsed = time.time() - eval_start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total_evaluations - completed) / rate if rate > 0 else 0
                print(f"\n  [CHECKPOINT] Processed {portfolio_idx:,}/{len(portfolio_weights):,} portfolios "
                      f"({completed:,} evaluations) | ETA: {remaining/60:.1f} min\n")
                sys.stdout.flush()
        
        if not results:
            return pd.DataFrame()
        
        return pd.DataFrame(results).set_index('portfolio_id')
