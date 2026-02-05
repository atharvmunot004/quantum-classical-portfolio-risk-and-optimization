"""
Portfolio Optimizer Module for Quantum Annealing.

Handles rolling window optimization with multi-objective QUBO formulation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from pathlib import Path
import warnings

# Try to import GPU libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    cudf = None

from .qubo_builder import QUBOBuilder
from .quantum_annealer import QuantumAnnealer
from .returns import (
    compute_rolling_mean_return,
    compute_rolling_correlation_matrix,
    compute_cvar,
    load_precomputed_portfolios
)
from .metrics import (
    compute_portfolio_performance_metrics,
    compute_optimization_quality_metrics,
    compute_quantum_specific_metrics,
    compute_turnover
)


class PortfolioOptimizer:
    """
    Portfolio optimizer using quantum annealing.
    
    Performs rolling window optimization with multi-objective QUBO formulation.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        estimation_windows: List[int],
        rebalance_frequency: int = 21,
        return_weights: List[float] = [0.5, 1.0],
        risk_weights: List[float] = [1.0, 2.0],
        diversification_weights: List[float] = [0.1, 0.5],
        budget_penalty: float = 10.0,
        cardinality_penalty: float = 8.0,
        min_assets: int = 5,
        max_assets: int = 15,
        confidence_level: float = 0.95,
        quantum_settings: Optional[Dict] = None,
        warmup_policy: str = 'skip_until_window_full',
        precomputed_portfolios: Optional[pd.DataFrame] = None,
        precomputed_portfolios_path: Optional[str] = None,
        num_top_portfolios: int = 10
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            returns: DataFrame of returns
            estimation_windows: List of estimation window sizes
            rebalance_frequency: Rebalancing frequency in days
            return_weights: List of return weight values to sweep
            risk_weights: List of risk weight values to sweep
            diversification_weights: List of diversification weight values to sweep
            budget_penalty: Budget constraint penalty
            cardinality_penalty: Cardinality constraint penalty
            min_assets: Minimum number of assets
            max_assets: Maximum number of assets
            confidence_level: CVaR confidence level
            quantum_settings: Quantum annealing settings
            warmup_policy: Warmup policy ('skip_until_window_full')
            precomputed_portfolios: Optional DataFrame of precomputed portfolios
            precomputed_portfolios_path: Optional path to precomputed portfolios file
            num_top_portfolios: Number of top portfolios to select per configuration
        """
        self.returns = returns
        self.estimation_windows = estimation_windows
        self.rebalance_frequency = rebalance_frequency
        self.return_weights = return_weights
        self.risk_weights = risk_weights
        self.diversification_weights = diversification_weights
        self.budget_penalty = budget_penalty
        self.cardinality_penalty = cardinality_penalty
        self.min_assets = min_assets
        self.max_assets = max_assets
        self.confidence_level = confidence_level
        self.warmup_policy = warmup_policy
        self.num_top_portfolios = num_top_portfolios
        
        # Quantum settings
        if quantum_settings is None:
            quantum_settings = {}
        self.quantum_settings = quantum_settings
        
        # Load precomputed portfolios
        if precomputed_portfolios is not None:
            self.precomputed_portfolios = precomputed_portfolios
        elif precomputed_portfolios_path is not None:
            print(f"Loading precomputed portfolios from {precomputed_portfolios_path}...")
            self.precomputed_portfolios = load_precomputed_portfolios(
                precomputed_portfolios_path,
                asset_universe=returns.columns
            )
            print(f"  Loaded {len(self.precomputed_portfolios):,} precomputed portfolios")
            
            # Convert to GPU if available
            if CUDF_AVAILABLE and CUPY_AVAILABLE and cudf is not None:
                try:
                    print("  Converting portfolios to GPU (cuDF)...")
                    self.precomputed_portfolios_gpu = cudf.from_pandas(self.precomputed_portfolios)
                    print(f"  GPU conversion successful: {len(self.precomputed_portfolios_gpu):,} portfolios on GPU")
                except Exception as e:
                    print(f"  Warning: GPU conversion failed: {e}. Using CPU.")
                    self.precomputed_portfolios_gpu = None
            else:
                self.precomputed_portfolios_gpu = None
        else:
            self.precomputed_portfolios = None
            self.precomputed_portfolios_gpu = None
        
        # GPU settings
        self.use_gpu = CUPY_AVAILABLE and CUPY_AVAILABLE and (self.precomputed_portfolios_gpu is not None or CUDF_AVAILABLE)
        if self.use_gpu:
            print(f"  GPU acceleration: ENABLED (CuPy + cuDF)")
        else:
            if CUPY_AVAILABLE:
                print(f"  GPU acceleration: PARTIAL (CuPy available, cuDF not available)")
            else:
                print(f"  GPU acceleration: DISABLED (using CPU)")
        
        # Results storage
        self.portfolio_weights = []
        self.portfolio_performance = []
        self.optimization_metrics = []
        self.pareto_fronts = []
    
    def _generate_weight_configs(self) -> List[Dict]:
        """
        Generate weight configurations for multi-objective optimization.
        
        Returns:
            List of weight configuration dictionaries
        """
        configs = []
        
        # Generate all combinations
        for rw in self.return_weights:
            for rskw in self.risk_weights:
                for dw in self.diversification_weights:
                    configs.append({
                        'return_weight': rw,
                        'risk_weight': rskw,
                        'diversification_weight': dw
                    })
        
        return configs
    
    def _optimize_single_window(
        self,
        window_returns: pd.DataFrame,
        weight_config: Dict,
        estimation_window: int,
        date: pd.Timestamp
    ) -> Dict:
        """
        Optimize portfolio for a single window.
        
        If precomputed portfolios are available, evaluates and selects best ones.
        Otherwise, uses quantum annealing to generate new portfolios.
        
        Args:
            window_returns: Returns for estimation window
            weight_config: Weight configuration
            estimation_window: Estimation window size
            date: Current date
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Compute inputs
        expected_returns = compute_rolling_mean_return(
            window_returns,
            window=estimation_window,
            annualized=True
        )
        
        cvar = compute_cvar(
            window_returns,
            confidence_level=self.confidence_level,
            window=estimation_window
        )
        
        correlation_matrix = compute_rolling_correlation_matrix(
            window_returns,
            window=estimation_window
        )
        
        # Build QUBO
        qubo_builder = QUBOBuilder(
            expected_returns=expected_returns,
            cvar=cvar,
            correlation_matrix=correlation_matrix,
            return_weight=weight_config['return_weight'],
            risk_weight=weight_config['risk_weight'],
            diversification_weight=weight_config['diversification_weight'],
            budget_penalty=self.budget_penalty,
            cardinality_penalty=self.cardinality_penalty,
            min_assets=self.min_assets,
            max_assets=self.max_assets,
            normalization=True
        )
        
        Q, constant = qubo_builder.build_qubo()
        
        qubo_build_time = (time.time() - start_time) * 1000
        
        # If precomputed portfolios are available, evaluate them
        if self.precomputed_portfolios is not None:
            evaluation_start = time.time()
            best_portfolios = self._evaluate_precomputed_portfolios(
                qubo_builder, Q, constant, window_returns.columns
            )
            evaluation_time = (time.time() - evaluation_start) * 1000
            
            # Use the best portfolio
            best_row = best_portfolios.iloc[0]
            best_portfolio_id = best_row['portfolio_id']
            best_weights = best_row['weights']
            if isinstance(best_weights, pd.Series):
                # Ensure it's a Series with proper index
                best_weights = best_weights.reindex(window_returns.columns, fill_value=0.0)
            else:
                # Convert dict to Series
                best_weights = pd.Series(best_weights).reindex(window_returns.columns, fill_value=0.0)
            best_energy = best_row['energy']
            
            # Convert portfolio weights to binary solution for compatibility
            # Find assets with non-zero weights
            selected_assets = best_weights[best_weights > 1e-10].index
            best_solution = np.zeros(len(window_returns.columns), dtype=int)
            asset_to_idx = {asset: idx for idx, asset in enumerate(window_returns.columns)}
            for asset in selected_assets:
                if asset in asset_to_idx:
                    best_solution[asset_to_idx[asset]] = 1
            
            # Create mock annealing results for compatibility
            annealing_results = {
                'best_solution': best_solution,
                'best_energy': best_energy,
                'energies': best_portfolios['energy'].values[:self.num_top_portfolios],
                'samples': [best_solution] * min(self.num_top_portfolios, len(best_portfolios)),
                'num_occurrences': np.ones(min(self.num_top_portfolios, len(best_portfolios))),
                'chain_break_fraction': None,
                'embedding_size': None,
                'logical_to_physical_ratio': None,
                'backend_type': 'precomputed_evaluation'
            }
            
            annealing_time = evaluation_time
        else:
            # Solve with quantum annealing
            annealer_start = time.time()
            annealer = QuantumAnnealer(**self.quantum_settings)
            annealing_results = annealer.solve(Q, constant)
            annealing_time = (time.time() - annealer_start) * 1000
            
            # Convert solution to weights
            best_solution = annealing_results['best_solution']
            best_weights = qubo_builder.solution_to_weights(best_solution)
            best_energy = annealing_results['best_energy']
        
        # Compute metrics
        optimization_metrics = compute_optimization_quality_metrics(
            annealing_results,
            qubo_builder
        )
        
        quantum_metrics = compute_quantum_specific_metrics(annealing_results)
        quantum_metrics['annealing_time_us'] = self.quantum_settings.get('annealing_time_us', 20)
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            'date': date,
            'estimation_window': estimation_window,
            'weights': best_weights,
            'return_weight': weight_config['return_weight'],
            'risk_weight': weight_config['risk_weight'],
            'diversification_weight': weight_config['diversification_weight'],
            'optimization_metrics': optimization_metrics,
            'quantum_metrics': quantum_metrics,
            'runtime_ms': {
                'total_runtime_ms': total_time,
                'qubo_build_time_ms': qubo_build_time,
                'annealing_time_ms': annealing_time
            },
            'best_energy': best_energy,
            'best_solution': best_solution
        }
    
    def _evaluate_precomputed_portfolios(
        self,
        qubo_builder: QUBOBuilder,
        Q: np.ndarray,
        constant: float,
        asset_universe: pd.Index
    ) -> pd.DataFrame:
        """
        Evaluate precomputed portfolios using QUBO energy.
        
        Uses GPU acceleration if available for faster computation.
        
        Args:
            qubo_builder: QUBOBuilder instance
            Q: QUBO matrix
            constant: QUBO constant term
            asset_universe: Current asset universe
            
        Returns:
            DataFrame with top portfolios sorted by energy
        """
        start_time = time.time()
        
        # Align portfolios with current asset universe
        if self.use_gpu and self.precomputed_portfolios_gpu is not None:
            # Use GPU DataFrames
            portfolios_aligned = self.precomputed_portfolios_gpu.reindex(
                columns=asset_universe.tolist(),
                fill_value=0.0
            )
            print(f"    GPU: Aligned {len(portfolios_aligned):,} portfolios")
        else:
            # Use CPU DataFrames
            portfolios_aligned = self.precomputed_portfolios.reindex(
                columns=asset_universe,
                fill_value=0.0
            )
        
        # Filter portfolios that satisfy cardinality constraints
        num_assets = (portfolios_aligned > 1e-10).sum(axis=1)
        valid_mask = (num_assets >= self.min_assets) & (num_assets <= self.max_assets)
        valid_portfolios = portfolios_aligned[valid_mask]
        
        if len(valid_portfolios) == 0:
            # Fallback: use all portfolios
            valid_portfolios = portfolios_aligned
        
        print(f"    Evaluating {len(valid_portfolios):,} valid portfolios...")
        
        # Convert to GPU arrays if available
        if self.use_gpu and CUPY_AVAILABLE:
            try:
                # Convert QUBO matrix to GPU
                Q_gpu = cp.asarray(Q)
                constant_gpu = cp.asarray(constant)
                
                # Convert portfolios to GPU array (binary representation)
                # Get portfolio weights as numpy array
                if CUDF_AVAILABLE and cudf is not None and isinstance(valid_portfolios, cudf.DataFrame):
                    portfolio_weights_array = valid_portfolios.values
                else:
                    portfolio_weights_array = valid_portfolios.values
                
                # Convert to binary solutions on GPU
                portfolio_weights_gpu = cp.asarray(portfolio_weights_array)
                binary_solutions_gpu = (portfolio_weights_gpu > 1e-10).astype(cp.int32)
                
                # Vectorized energy computation: (solutions @ Q) @ solutions^T
                # For each solution s: energy = s^T @ Q @ s + constant
                # We can compute this as: (solutions @ Q) * solutions and sum
                energies_gpu = cp.sum((binary_solutions_gpu @ Q_gpu) * binary_solutions_gpu, axis=1) + constant_gpu
                
                # Get top portfolios
                top_indices_gpu = cp.argsort(energies_gpu)[:self.num_top_portfolios]
                top_energies = cp.asnumpy(energies_gpu[top_indices_gpu])
                top_indices = cp.asnumpy(top_indices_gpu)
                
                # Extract top portfolios
                if CUDF_AVAILABLE and cudf is not None and isinstance(valid_portfolios, cudf.DataFrame):
                    top_portfolios_gpu = valid_portfolios.iloc[top_indices]
                    top_portfolios = top_portfolios_gpu.to_pandas()
                else:
                    top_portfolios = valid_portfolios.iloc[top_indices]
                
                # Create results DataFrame
                portfolio_list = []
                for idx, (portfolio_id, weights_row) in enumerate(top_portfolios.iterrows()):
                    portfolio_list.append({
                        'portfolio_id': portfolio_id,
                        'weights': weights_row,
                        'energy': float(top_energies[idx]),
                        'num_assets': int((weights_row > 1e-10).sum())
                    })
                
                results_df = pd.DataFrame(portfolio_list)
                eval_time = time.time() - start_time
                print(f"    GPU evaluation complete: {len(results_df):,} top portfolios in {eval_time:.2f}s")
                
                return results_df
                
            except Exception as e:
                print(f"    Warning: GPU evaluation failed ({e}), falling back to CPU...")
                self.use_gpu = False
        
        # CPU fallback: vectorized evaluation
        print(f"    Using CPU evaluation...")
        portfolio_list = []
        
        # Pre-compute asset to index mapping
        asset_to_idx = {asset: idx for idx, asset in enumerate(asset_universe)}
        
        # Convert portfolios to binary solutions (vectorized)
        portfolio_weights_array = valid_portfolios.values
        binary_solutions = (portfolio_weights_array > 1e-10).astype(np.int32)
        
        # Vectorized energy computation
        energies = np.sum((binary_solutions @ Q) * binary_solutions, axis=1) + constant
        
        # Get top portfolios
        top_indices = np.argsort(energies)[:self.num_top_portfolios]
        top_energies = energies[top_indices]
        top_portfolios = valid_portfolios.iloc[top_indices]
        
        # Create results DataFrame
        for idx, (portfolio_id, weights_row) in enumerate(top_portfolios.iterrows()):
            portfolio_list.append({
                'portfolio_id': portfolio_id,
                'weights': weights_row,
                'energy': float(top_energies[idx]),
                'num_assets': int((weights_row > 1e-10).sum())
            })
        
        results_df = pd.DataFrame(portfolio_list)
        eval_time = time.time() - start_time
        print(f"    CPU evaluation complete: {len(results_df):,} top portfolios in {eval_time:.2f}s")
        
        return results_df
    
    def optimize(self) -> Dict:
        """
        Run rolling window optimization.
        
        Returns:
            Dictionary with optimization results
        """
        print("Starting quantum annealing portfolio optimization...")
        print(f"  Returns shape: {self.returns.shape}")
        print(f"  Estimation windows: {self.estimation_windows}")
        print(f"  Rebalance frequency: {self.rebalance_frequency} days")
        
        # Generate weight configurations
        weight_configs = self._generate_weight_configs()
        print(f"  Weight configurations: {len(weight_configs)}")
        
        # Determine rebalancing dates
        max_window = max(self.estimation_windows)
        start_date = self.returns.index[max_window]
        end_date = self.returns.index[-1]
        
        rebalance_dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date in self.returns.index:
                rebalance_dates.append(current_date)
            # Move to next rebalancing date
            current_idx = self.returns.index.get_loc(current_date)
            if current_idx + self.rebalance_frequency < len(self.returns):
                current_date = self.returns.index[current_idx + self.rebalance_frequency]
            else:
                break
        
        print(f"  Rebalancing dates: {len(rebalance_dates)}")
        
        # Optimize for each date and weight configuration
        all_results = []
        
        # Add progress tracking and periodic flushing
        import sys
        sys.stdout.flush()
        
        for date_idx, date in enumerate(rebalance_dates):
            print(f"\n{'='*80}")
            print(f"Processing date {date_idx + 1}/{len(rebalance_dates)}: {date.date()}")
            print(f"{'='*80}")
            sys.stdout.flush()
            
            date_start_time = time.time()
            
            for window in self.estimation_windows:
                print(f"\n  Estimation window: {window} days")
                sys.stdout.flush()
                
                # Get estimation window data
                date_idx_in_returns = self.returns.index.get_loc(date)
                if date_idx_in_returns < window:
                    if self.warmup_policy == 'skip_until_window_full':
                        continue
                
                window_start_idx = max(0, date_idx_in_returns - window + 1)
                window_returns = self.returns.iloc[window_start_idx:date_idx_in_returns + 1]
                
                if len(window_returns) < window:
                    continue
                
                for weight_idx, weight_config in enumerate(weight_configs):
                    config_start_time = time.time()
                    print(f"    Weight config {weight_idx + 1}/{len(weight_configs)}: "
                          f"return={weight_config['return_weight']:.2f}, "
                          f"risk={weight_config['risk_weight']:.2f}, "
                          f"div={weight_config['diversification_weight']:.2f}")
                    sys.stdout.flush()
                    
                    try:
                        result = self._optimize_single_window(
                            window_returns,
                            weight_config,
                            window,
                            date
                        )
                        all_results.append(result)
                        config_time = time.time() - config_start_time
                        print(f"      [OK] Optimization complete ({config_time:.2f}s)")
                        sys.stdout.flush()
                    except Exception as e:
                        config_time = time.time() - config_start_time
                        print(f"      [ERROR] Error after {config_time:.2f}s: {e}")
                        import traceback
                        traceback.print_exc()
                        sys.stdout.flush()
                        continue
            
            date_time = time.time() - date_start_time
            print(f"\n  Date {date_idx + 1} completed in {date_time:.2f}s. Total results: {len(all_results)}")
            sys.stdout.flush()
            
            # Periodic checkpoint every 10 dates
            if (date_idx + 1) % 10 == 0:
                print(f"\n  [CHECKPOINT] Processed {date_idx + 1} dates, {len(all_results)} optimizations")
                sys.stdout.flush()
        
        # Organize results
        print(f"\nOrganizing results from {len(all_results)} optimizations...")
        sys.stdout.flush()
        
        print("  Organizing weights...")
        sys.stdout.flush()
        weights_df = self._organize_weights(all_results)
        print(f"    Weights DataFrame: {len(weights_df)} rows")
        sys.stdout.flush()
        
        print("  Computing performance metrics...")
        sys.stdout.flush()
        performance_df = self._compute_performance(weights_df)
        print(f"    Performance DataFrame: {len(performance_df)} rows")
        sys.stdout.flush()
        
        print("  Organizing optimization metrics...")
        sys.stdout.flush()
        metrics_df = self._organize_metrics(all_results)
        print(f"    Metrics DataFrame: {len(metrics_df)} rows")
        sys.stdout.flush()
        
        return {
            'portfolio_weights': weights_df,
            'portfolio_performance': performance_df,
            'optimization_metrics': metrics_df,
            'num_optimizations': len(all_results)
        }
    
    def _organize_weights(self, results: List[Dict]) -> pd.DataFrame:
        """
        Organize portfolio weights into DataFrame.
        
        Args:
            results: List of optimization results
            
        Returns:
            DataFrame of portfolio weights
        """
        rows = []
        
        for result in results:
            date = result['date']
            weights = result['weights']
            
            for asset, weight in weights.items():
                rows.append({
                    'date': date,
                    'asset': asset,
                    'weight': weight,
                    'return_weight': result['return_weight'],
                    'risk_weight': result['risk_weight'],
                    'diversification_weight': result['diversification_weight'],
                    'estimation_window': result['estimation_window']
                })
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        if 'date' in df.columns:
            df = df.set_index('date')
        return df
    
    def _compute_performance(self, weights_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute portfolio performance from weights.
        
        Args:
            weights_df: DataFrame of portfolio weights
            
        Returns:
            DataFrame of portfolio performance
        """
        import sys
        
        if weights_df.empty:
            return pd.DataFrame()
        
        print(f"      Processing {len(weights_df)} weight records...")
        sys.stdout.flush()
        
        # Group by date and weight configuration to get portfolio returns
        performance_rows = []
        
        # Reset index if date is in index
        if 'date' in weights_df.index.names:
            weights_df = weights_df.reset_index()
        
        groups = list(weights_df.groupby(
            ['date', 'return_weight', 'risk_weight', 'diversification_weight', 'estimation_window']
        ))
        total_groups = len(groups)
        
        for idx, ((date, rw, rskw, dw, ew), group) in enumerate(groups):
            if idx % 100 == 0 and idx > 0:
                print(f"        Processed {idx}/{total_groups} groups...")
                sys.stdout.flush()
            
            # Get portfolio weights for this configuration
            weights_dict = dict(zip(group['asset'], group['weight']))
            weights_series = pd.Series(weights_dict)
            
            # Compute portfolio return for this date
            if date in self.returns.index:
                asset_returns = self.returns.loc[date]
                # Align weights with returns
                aligned_weights = weights_series.reindex(asset_returns.index, fill_value=0.0)
                portfolio_return = (aligned_weights * asset_returns).sum()
                
                performance_rows.append({
                    'date': date,
                    'portfolio_return': portfolio_return,
                    'return_weight': rw,
                    'risk_weight': rskw,
                    'diversification_weight': dw,
                    'estimation_window': ew
                })
        
        if not performance_rows:
            return pd.DataFrame()
        
        print(f"      Creating performance DataFrame from {len(performance_rows)} rows...")
        sys.stdout.flush()
        perf_df = pd.DataFrame(performance_rows)
        
        if perf_df.empty:
            return pd.DataFrame()
        
        # Set date as index for easier manipulation
        perf_df = perf_df.set_index('date')
        
        print(f"      Computing cumulative metrics for {len(perf_df.groupby(['return_weight', 'risk_weight', 'diversification_weight', 'estimation_window']))} configurations...")
        sys.stdout.flush()
        
        # Compute cumulative performance metrics for each configuration
        config_groups = list(perf_df.groupby(
            ['return_weight', 'risk_weight', 'diversification_weight', 'estimation_window']
        ))
        for idx, ((rw, rskw, dw, ew), group) in enumerate(config_groups):
            if idx % 5 == 0:
                print(f"        Computing metrics for config {idx + 1}/{len(config_groups)}...")
                sys.stdout.flush()
            
            # Get portfolio returns sorted by date
            portfolio_returns = group['portfolio_return'].sort_index()
            
            if len(portfolio_returns) > 0:
                # Compute cumulative metrics
                from .metrics import compute_portfolio_performance_metrics
                metrics = compute_portfolio_performance_metrics(portfolio_returns)
                
                # Add metrics to all rows in this group
                for key, value in metrics.items():
                    perf_df.loc[group.index, key] = value
        
        return perf_df
    
    def _organize_metrics(self, results: List[Dict]) -> pd.DataFrame:
        """
        Organize optimization metrics into DataFrame.
        
        Args:
            results: List of optimization results
            
        Returns:
            DataFrame of optimization metrics
        """
        rows = []
        
        for result in results:
            row = {
                'date': result['date'],
                'estimation_window': result['estimation_window'],
                'return_weight': result['return_weight'],
                'risk_weight': result['risk_weight'],
                'diversification_weight': result['diversification_weight'],
                **result['optimization_metrics'],
                **result['quantum_metrics'],
                **result['runtime_ms']
            }
            rows.append(row)
        
        if not rows:
            return pd.DataFrame()
        
        return pd.DataFrame(rows).set_index('date')
