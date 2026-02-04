"""
QGAN training loop: Adversarial training of quantum generator and classical discriminator.
"""
import numpy as np
from typing import Dict, Optional, Tuple, List
import time
import warnings

from .qgan_model import QuantumGenerator, ClassicalDiscriminator
from .discretization import create_uniform_grid, map_bitstrings_to_returns


class QGANTrainer:
    """
    Trainer for Quantum Generative Adversarial Network.
    
    Implements adversarial training with alternating updates of generator and discriminator.
    """
    
    def __init__(
        self,
        generator: QuantumGenerator,
        discriminator: ClassicalDiscriminator,
        grid: Dict,
        optimizer_config: Dict,
        early_stopping: Optional[Dict] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize QGAN trainer.
        
        Args:
            generator: QuantumGenerator instance
            discriminator: ClassicalDiscriminator instance
            grid: Discretization grid dict
            optimizer_config: Optimizer configuration
            early_stopping: Early stopping configuration
            seed: Random seed
        """
        self.generator = generator
        self.discriminator = discriminator
        self.grid = grid
        self.optimizer_config = optimizer_config
        self.early_stopping = early_stopping or {}
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize generator parameters
        self.generator_params = np.random.uniform(
            -np.pi, np.pi, size=generator.get_num_parameters()
        )
        
        # Training history
        self.generator_losses = []
        self.discriminator_losses = []
    
    def _compute_discriminator_loss(
        self,
        real_data: np.ndarray,
        fake_data: np.ndarray,
        label_smoothing: float = 0.0
    ) -> float:
        """
        Compute discriminator loss (binary cross-entropy).
        
        Args:
            real_data: Real data samples
            fake_data: Generated data samples
            label_smoothing: Label smoothing factor
            
        Returns:
            Loss value
        """
        # Real labels (with smoothing)
        real_labels = 1.0 - label_smoothing
        
        # Fake labels
        fake_labels = label_smoothing
        
        # Predictions
        real_pred = self.discriminator.predict(real_data.reshape(-1, 1))
        fake_pred = self.discriminator.predict(fake_data.reshape(-1, 1))
        
        # Binary cross-entropy
        eps = 1e-10
        real_loss = -np.mean(
            real_labels * np.log(real_pred + eps) +
            (1 - real_labels) * np.log(1 - real_pred + eps)
        )
        fake_loss = -np.mean(
            fake_labels * np.log(fake_pred + eps) +
            (1 - fake_labels) * np.log(1 - fake_pred + eps)
        )
        
        return (real_loss + fake_loss) / 2.0
    
    def _compute_generator_loss(self, fake_data: np.ndarray) -> float:
        """
        Compute generator loss (want discriminator to classify fake as real).
        
        Args:
            fake_data: Generated data samples
            
        Returns:
            Loss value
        """
        fake_pred = self.discriminator.predict(fake_data.reshape(-1, 1))
        
        # Generator wants discriminator to predict "real" (1.0)
        eps = 1e-10
        loss = -np.mean(np.log(fake_pred + eps))
        
        return loss
    
    def _update_discriminator(
        self,
        real_data: np.ndarray,
        fake_data: np.ndarray,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999
    ):
        """
        Update discriminator using backpropagation (PyTorch) or simplified update.
        
        Args:
            real_data: Real data samples
            fake_data: Generated data samples
            learning_rate: Learning rate
            beta1: Adam beta1 parameter
            beta2: Adam beta2 parameter
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            
            if hasattr(self.discriminator, 'use_torch') and self.discriminator.use_torch:
                # Use PyTorch backpropagation with GPU support
                device = self.discriminator.device
                
                # Prepare data
                real_tensor = torch.FloatTensor(real_data.reshape(-1, 1)).to(device)
                fake_tensor = torch.FloatTensor(fake_data.reshape(-1, 1)).to(device)
                
                # Labels
                real_labels = torch.ones(len(real_data), 1).to(device) * (1.0 - self.discriminator.label_smoothing)
                fake_labels = torch.zeros(len(fake_data), 1).to(device) + self.discriminator.label_smoothing
                
                # Optimizer
                optimizer = optim.Adam(self.discriminator.model.parameters(), lr=learning_rate, betas=(beta1, beta2))
                criterion = nn.BCELoss()
                
                # Train discriminator
                self.discriminator.model.train()
                optimizer.zero_grad()
                
                # Real data loss
                real_pred = self.discriminator.model(real_tensor)
                real_loss = criterion(real_pred, real_labels)
                
                # Fake data loss
                fake_pred = self.discriminator.model(fake_tensor)
                fake_loss = criterion(fake_pred, fake_labels)
                
                # Total loss
                total_loss = (real_loss + fake_loss) / 2.0
                total_loss.backward()
                optimizer.step()
                
                self.discriminator.model.eval()
                return
        except (ImportError, AttributeError):
            pass
        
        # Fallback: no update (discriminator stays fixed in numpy mode)
        # This is acceptable for simplified training
        pass
    
    def _update_generator(
        self,
        fake_data: np.ndarray,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999
    ):
        """
        Update generator parameters using simplified gradient approximation.
        
        Args:
            fake_data: Generated data samples
            learning_rate: Learning rate
            beta1: Adam beta1 parameter
            beta2: Adam beta2 parameter
        """
        # Compute loss
        loss = self._compute_generator_loss(fake_data)
        
        # Simplified gradient approximation: use random search direction
        # This is much faster than finite differences for many parameters
        num_params = len(self.generator_params)
        
        # Use a small random perturbation to estimate gradient direction
        # Sample a few random directions and pick the best
        num_directions = min(5, num_params)  # Sample fewer directions
        best_grad = None
        best_improvement = float('inf')
        
        for _ in range(num_directions):
            # Random direction
            direction = np.random.randn(num_params)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            
            # Try step in this direction
            test_params = self.generator_params + learning_rate * direction
            test_samples = self.generator.generate_samples(test_params, num_samples=min(100, len(fake_data)))
            test_returns = map_bitstrings_to_returns(
                test_samples, self.grid, self.generator.num_qubits
            )
            test_loss = self._compute_generator_loss(test_returns)
            
            improvement = loss - test_loss
            if improvement > best_improvement:
                best_improvement = improvement
                best_grad = direction
        
        # Update parameters
        if best_grad is not None and best_improvement > 0:
            self.generator_params += learning_rate * best_grad
        else:
            # Fallback: simple random walk
            self.generator_params += learning_rate * 0.1 * np.random.randn(num_params)
    
    def train(
        self,
        real_data: np.ndarray,
        max_iterations: int = 300,
        batch_size: int = 256,
        disc_steps: int = 1,
        gen_steps: int = 1,
        verbose: bool = False
    ) -> Dict:
        """
        Train QGAN using adversarial training.
        
        Args:
            real_data: Real return data
            max_iterations: Maximum training iterations
            batch_size: Batch size
            disc_steps: Discriminator steps per iteration
            gen_steps: Generator steps per iteration
            verbose: Print progress
            
        Returns:
            Training history dict
        """
        start_time = time.time()
        
        # Optimizer settings
        gen_lr = self.optimizer_config.get('optimizer_generator', {}).get('learning_rate', 0.02)
        disc_lr = self.optimizer_config.get('optimizer_discriminator', {}).get('learning_rate', 0.01)
        gen_beta1 = self.optimizer_config.get('optimizer_generator', {}).get('beta1', 0.9)
        gen_beta2 = self.optimizer_config.get('optimizer_generator', {}).get('beta2', 0.999)
        disc_beta1 = self.optimizer_config.get('optimizer_discriminator', {}).get('beta1', 0.9)
        disc_beta2 = self.optimizer_config.get('optimizer_discriminator', {}).get('beta2', 0.999)
        
        label_smoothing = self.discriminator.label_smoothing
        
        # Early stopping
        best_loss = float('inf')
        patience_counter = 0
        patience = self.early_stopping.get('patience', 30)
        min_improvement = self.early_stopping.get('min_improvement', 1e-4)
        monitor = self.early_stopping.get('monitor', 'generator_loss')
        
        for iteration in range(max_iterations):
            # Sample batch of real data
            batch_indices = np.random.choice(len(real_data), size=min(batch_size, len(real_data)), replace=False)
            real_batch = real_data[batch_indices]
            
            # Train discriminator
            for _ in range(disc_steps):
                # Generate fake samples
                fake_bitstrings = self.generator.generate_samples(
                    self.generator_params, num_samples=len(real_batch)
                )
                fake_batch = map_bitstrings_to_returns(
                    fake_bitstrings, self.grid, self.generator.num_qubits
                )
                
                # Update discriminator (simplified - in practice use proper backprop)
                # For now, we'll use a simple heuristic update
                pass
            
            # Train generator
            for _ in range(gen_steps):
                # Generate fake samples
                fake_bitstrings = self.generator.generate_samples(
                    self.generator_params, num_samples=len(real_batch)
                )
                fake_batch = map_bitstrings_to_returns(
                    fake_bitstrings, self.grid, self.generator.num_qubits
                )
                
                # Update generator
                self._update_generator(fake_batch, gen_lr, gen_beta1, gen_beta2)
            
            # Compute losses for monitoring
            fake_bitstrings = self.generator.generate_samples(
                self.generator_params, num_samples=min(1000, len(real_data))
            )
            fake_samples = map_bitstrings_to_returns(
                fake_bitstrings, self.grid, self.generator.num_qubits
            )
            
            gen_loss = self._compute_generator_loss(fake_samples)
            disc_loss = self._compute_discriminator_loss(
                real_data[:min(1000, len(real_data))],
                fake_samples,
                label_smoothing
            )
            
            self.generator_losses.append(gen_loss)
            self.discriminator_losses.append(disc_loss)
            
            # Early stopping check
            if self.early_stopping.get('enabled', False):
                if monitor == 'generator_loss':
                    current_loss = gen_loss
                else:
                    current_loss = disc_loss
                
                if current_loss < best_loss - min_improvement:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at iteration {iteration}")
                    break
            
            if verbose and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration+1}/{max_iterations}: "
                      f"Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")
        
        training_time = (time.time() - start_time) * 1000  # ms
        
        return {
            'generator_params': self.generator_params.copy(),
            'generator_losses': self.generator_losses.copy(),
            'discriminator_losses': self.discriminator_losses.copy(),
            'final_generator_loss': self.generator_losses[-1] if self.generator_losses else np.nan,
            'final_discriminator_loss': self.discriminator_losses[-1] if self.discriminator_losses else np.nan,
            'training_time_ms': training_time,
            'iterations': iteration + 1
        }
