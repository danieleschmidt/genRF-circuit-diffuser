"""
AI Models for RF Circuit Generation.

This module implements the generative AI models used for circuit topology
generation and parameter optimization:
- CycleGAN for topology generation
- DiffusionModel for parameter optimization
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    input_dim: int = 100
    hidden_dim: int = 256
    output_dim: int = 64
    num_layers: int = 4
    dropout: float = 0.1
    activation: str = "relu"
    batch_norm: bool = True
    device: str = "auto"


class CycleGAN(nn.Module):
    """
    Cycle-consistent GAN for RF circuit topology generation.
    
    Generates circuit topologies conditioned on design specifications.
    Uses cycle consistency to ensure topology validity and diversity.
    """
    
    def __init__(
        self,
        spec_dim: int = 9,
        latent_dim: int = 100,
        topology_dim: int = 64,
        hidden_dim: int = 256
    ):
        """
        Initialize CycleGAN for topology generation.
        
        Args:
            spec_dim: Dimension of specification conditioning vector
            latent_dim: Dimension of latent noise vector
            topology_dim: Dimension of topology representation
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.spec_dim = spec_dim
        self.latent_dim = latent_dim
        self.topology_dim = topology_dim
        self.hidden_dim = hidden_dim
        
        # Generator: spec + noise -> topology
        self.generator = self._build_generator()
        
        # Discriminator: topology -> real/fake
        self.discriminator = self._build_discriminator()
        
        # Cycle generator: topology -> reconstructed spec
        self.cycle_generator = self._build_cycle_generator()
        
        logger.info(f"CycleGAN initialized with spec_dim={spec_dim}, "
                   f"latent_dim={latent_dim}, topology_dim={topology_dim}")
    
    def _build_generator(self) -> nn.Module:
        """Build the main generator network."""
        input_dim = self.spec_dim + self.latent_dim
        
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(self.hidden_dim, self.topology_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def _build_discriminator(self) -> nn.Module:
        """Build the discriminator network."""
        return nn.Sequential(
            nn.Linear(self.topology_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def _build_cycle_generator(self) -> nn.Module:
        """Build the cycle consistency generator."""
        return nn.Sequential(
            nn.Linear(self.topology_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            nn.Linear(self.hidden_dim // 2, self.spec_dim)
        )
    
    def forward(self, specs: torch.Tensor, noise: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of CycleGAN."""
        batch_size = specs.size(0)
        
        # Generate topology from spec + noise
        gen_input = torch.cat([specs, noise], dim=1)
        topology = self.generator(gen_input)
        
        # Discriminator prediction
        d_pred = self.discriminator(topology)
        
        # Cycle consistency: topology -> reconstructed spec
        reconstructed_spec = self.cycle_generator(topology)
        
        return {
            'topology': topology,
            'discriminator_pred': d_pred,
            'reconstructed_spec': reconstructed_spec
        }
    
    def generate(self, specs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Generate topologies for given specifications."""
        self.eval()
        batch_size = specs.size(0)
        
        topologies = []
        for _ in range(num_samples):
            noise = torch.randn(batch_size, self.latent_dim, device=specs.device)
            with torch.no_grad():
                gen_input = torch.cat([specs, noise], dim=1)
                topology = self.generator(gen_input)
                topologies.append(topology)
        
        return torch.stack(topologies, dim=1)  # [batch, num_samples, topology_dim]
    
    def _build_discriminator(self) -> nn.Module:
        """Build the discriminator network."""
        return nn.Sequential(
            nn.Linear(self.topology_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def _build_cycle_generator(self) -> nn.Module:
        """Build the cycle consistency generator."""
        return nn.Sequential(
            nn.Linear(self.topology_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            nn.Linear(self.hidden_dim // 2, self.spec_dim),
            nn.Tanh()  # Match input range
        )


class DiffusionModel(nn.Module):
    """Denoising diffusion model for RF circuit parameter generation."""
    
    def __init__(
        self,
        param_dim: int = 32,
        condition_dim: int = 64,
        hidden_dim: int = 256,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        super().__init__()
        
        self.param_dim = param_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        
        # Beta schedule for noise
        self.register_buffer(
            'betas',
            torch.linspace(beta_start, beta_end, num_timesteps)
        )
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # U-Net style denoising network
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.input_projection = nn.Linear(param_dim + condition_dim, hidden_dim)
        
        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(3)
        ])
        
        # Middle
        self.middle = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GroupNorm(16, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Decoder  
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(3)
        ])
        
        self.output_projection = nn.Linear(hidden_dim, param_dim)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Predict noise to be removed from x at timestep t."""
        batch_size = x.size(0)
        
        # Time embedding
        t_emb = self.time_embedding(t.float().unsqueeze(-1))
        
        # Input projection
        h = self.input_projection(torch.cat([x, condition], dim=-1))
        h = h + t_emb
        
        # Encoder with skip connections
        skip_connections = []
        for layer in self.encoder:
            h = layer(h) + h  # Residual connection
            skip_connections.append(h)
        
        # Middle
        h = self.middle(h)
        
        # Decoder with skip connections
        for layer, skip in zip(self.decoder, reversed(skip_connections)):
            h = torch.cat([h, skip], dim=-1)
            h = layer(h)
        
        # Output
        noise_pred = self.output_projection(h)
        return noise_pred
    
    def add_noise(self, x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Add noise to clean parameters."""
        sqrt_alpha_cumprod = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[t]).sqrt()
        
        return sqrt_alpha_cumprod.view(-1, 1) * x + sqrt_one_minus_alpha_cumprod.view(-1, 1) * noise
    
    def sample(self, condition: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Sample parameters using DDPM sampling."""
        device = condition.device
        batch_size = condition.size(0)
        
        # Start from pure noise
        x = torch.randn(batch_size * num_samples, self.param_dim, device=device)
        condition_expanded = condition.repeat_interleave(num_samples, dim=0)
        
        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((batch_size * num_samples,), t, device=device)
            
            with torch.no_grad():
                noise_pred = self.forward(x, t_tensor, condition_expanded)
            
            # DDPM sampling step
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            # Compute x_{t-1}
            x = (x - beta / (1 - alpha_cumprod).sqrt() * noise_pred) / alpha.sqrt()
            
            # Add noise (except for final step)
            if t > 0:
                noise = torch.randn_like(x)
                x = x + (beta.sqrt() * noise)
        
        return x.view(batch_size, num_samples, self.param_dim)


class PhysicsInformedDiffusion(DiffusionModel):
    """Physics-informed diffusion model with RF constraints."""
    
    def __init__(self, *args, physics_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.physics_weight = physics_weight
        
    def physics_loss(self, params: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Compute physics-based loss terms."""
        # Extract circuit parameters (this would be domain-specific)
        # For demo: assume first few dims are R, L, C values
        R = params[:, 0].abs() + 1e-6  # Resistance > 0
        L = params[:, 1].abs() + 1e-9  # Inductance > 0  
        C = params[:, 2].abs() + 1e-12 # Capacitance > 0
        
        # Extract frequency from condition
        freq = condition[:, 0] * 1e11  # Scale to actual frequency
        omega = 2 * math.pi * freq
        
        # RF physics constraints
        # 1. Resonant frequency constraint
        f_res = 1 / (2 * math.pi * torch.sqrt(L * C))
        resonance_loss = F.mse_loss(f_res, freq)
        
        # 2. Quality factor constraint (should be reasonable)
        Q = torch.sqrt(L / C) / R
        Q_target = 10.0  # Target Q factor
        q_loss = F.mse_loss(Q, torch.full_like(Q, Q_target))
        
        # 3. Impedance matching (50 ohm)
        Z_char = torch.sqrt(L / C)
        impedance_loss = F.mse_loss(Z_char, torch.full_like(Z_char, 50.0))
        
        return resonance_loss + q_loss + impedance_loss
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward with physics loss."""
        noise_pred = super().forward(x, t, condition)
        
        # Compute physics loss on clean parameters (x_0 estimate)
        alpha_cumprod = self.alphas_cumprod[t[0]]  # Assume same timestep for batch
        x0_pred = (x - (1 - alpha_cumprod).sqrt() * noise_pred) / alpha_cumprod.sqrt()
        physics_loss = self.physics_loss(x0_pred, condition)
        
        return {
            'noise_pred': noise_pred,
            'physics_loss': physics_loss,
            'x0_pred': x0_pred
        }
    
    def sample(self, condition: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Sample parameters using physics-informed diffusion sampling."""
        device = condition.device
        batch_size = condition.size(0)
        
        # Start from pure noise
        x = torch.randn(batch_size * num_samples, self.param_dim, device=device)
        condition_expanded = condition.repeat_interleave(num_samples, dim=0)
        
        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((batch_size * num_samples,), t, device=device)
            
            with torch.no_grad():
                # Get predictions including physics loss
                output = self.forward(x, t_tensor, condition_expanded)
                noise_pred = output['noise_pred']
            
            # DDPM sampling step
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            x = (x - beta / (1 - alpha_cumprod).sqrt() * noise_pred) / alpha.sqrt()
            
            # Add noise (except for final step)
            if t > 0:
                noise = torch.randn_like(x)
                x = x + (beta.sqrt() * noise)
        
        return x.view(batch_size, num_samples, self.param_dim)
    
    def generate(
        self, 
        noise: torch.Tensor, 
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate topology from noise and conditioning.
        
        Args:
            noise: Random noise tensor [batch_size, latent_dim]
            condition: Conditioning vector [batch_size, spec_dim]
            
        Returns:
            Generated topology tensor [batch_size, topology_dim]
        """
        # Concatenate condition and noise
        input_tensor = torch.cat([condition, noise], dim=1)
        
        # Generate topology
        topology = self.generator(input_tensor)
        
        return topology
    
    def discriminate(self, topology: torch.Tensor) -> torch.Tensor:
        """
        Discriminate real vs fake topologies.
        
        Args:
            topology: Topology tensor [batch_size, topology_dim]
            
        Returns:
            Probability of being real [batch_size, 1]
        """
        return self.discriminator(topology)
    
    def cycle_forward(self, topology: torch.Tensor) -> torch.Tensor:
        """
        Cycle consistency forward pass.
        
        Args:
            topology: Topology tensor [batch_size, topology_dim]
            
        Returns:
            Reconstructed specification [batch_size, spec_dim]
        """
        return self.cycle_generator(topology)
    
    def forward(
        self, 
        noise: torch.Tensor, 
        condition: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for training.
        
        Args:
            noise: Random noise tensor
            condition: Conditioning vector
            
        Returns:
            Dictionary with generated topology, discriminator output, 
            and cycle consistency reconstruction
        """
        # Generate topology
        fake_topology = self.generate(noise, condition)
        
        # Discriminate
        fake_score = self.discriminate(fake_topology)
        
        # Cycle consistency
        reconstructed_spec = self.cycle_forward(fake_topology)
        
        return {
            'fake_topology': fake_topology,
            'fake_score': fake_score,
            'reconstructed_spec': reconstructed_spec
        }
        """
        Initialize diffusion model for parameter generation.
        
        Args:
            param_dim: Dimension of parameter vector
            condition_dim: Dimension of conditioning vector
            hidden_dim: Hidden layer dimension
            time_dim: Time embedding dimension
            num_timesteps: Number of diffusion timesteps
        """
        super().__init__()
        
        self.param_dim = param_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_timesteps = num_timesteps
        
        # U-Net architecture for denoising
        self.denoiser = self._build_denoiser()
        
        # Time embedding
        self.time_embedding = self._build_time_embedding()
        
        # Noise schedule
        self.register_buffer('betas', self._cosine_beta_schedule(num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        logger.info(f"DiffusionModel initialized with param_dim={param_dim}, "
                   f"timesteps={num_timesteps}")
    
    def _build_denoiser(self) -> nn.Module:
        """Build the denoising U-Net."""
        input_dim = self.param_dim + self.condition_dim + self.time_dim
        
        return nn.Sequential(
            # Encoder
            nn.Linear(input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(inplace=True),
            
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            
            # Bottleneck
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.SiLU(inplace=True),
            
            # Decoder
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(inplace=True),
            
            nn.Linear(self.hidden_dim, self.param_dim)
        )
    
    def _build_time_embedding(self) -> nn.Module:
        """Build sinusoidal time embedding."""
        return nn.Sequential(
            nn.Linear(1, self.time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.time_dim, self.time_dim)
        )
    
    def _cosine_beta_schedule(self, timesteps: int) -> torch.Tensor:
        """Create cosine noise schedule."""
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _get_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get sinusoidal time embeddings."""
        # Normalize timesteps to [0, 1]
        normalized_t = timesteps.float() / self.num_timesteps
        
        # Apply time embedding network
        time_emb = self.time_embedding(normalized_t.unsqueeze(-1))
        
        return time_emb
    
    def add_noise(
        self, 
        x_0: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to clean parameters according to diffusion schedule.
        
        Args:
            x_0: Clean parameter tensor [batch_size, param_dim]
            timesteps: Timestep tensor [batch_size]
            
        Returns:
            Tuple of (noisy_params, noise)
        """
        noise = torch.randn_like(x_0)
        
        # Get alpha values for timesteps
        alpha_cumprod = self.alphas_cumprod.gather(0, timesteps)
        alpha_cumprod = alpha_cumprod.view(-1, 1)
        
        # Add noise: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        x_t = torch.sqrt(alpha_cumprod) * x_0 + torch.sqrt(1 - alpha_cumprod) * noise
        
        return x_t, noise
    
    def denoise(
        self, 
        x_t: torch.Tensor, 
        timesteps: torch.Tensor, 
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Denoise parameters at given timesteps.
        
        Args:
            x_t: Noisy parameter tensor [batch_size, param_dim]
            timesteps: Timestep tensor [batch_size]
            condition: Conditioning tensor [batch_size, condition_dim]
            
        Returns:
            Predicted noise [batch_size, param_dim]
        """
        # Get time embeddings
        time_emb = self._get_time_embedding(timesteps)
        
        # Concatenate inputs
        denoiser_input = torch.cat([x_t, condition, time_emb], dim=1)
        
        # Predict noise
        predicted_noise = self.denoiser(denoiser_input)
        
        return predicted_noise
    
    def sample(
        self, 
        condition: torch.Tensor, 
        num_inference_steps: int = 50
    ) -> torch.Tensor:
        """
        Sample parameters using DDPM sampling.
        
        Args:
            condition: Conditioning tensor [batch_size, condition_dim]
            num_inference_steps: Number of denoising steps
            
        Returns:
            Generated parameters [batch_size, param_dim]
        """
        batch_size = condition.shape[0]
        device = condition.device
        
        # Start from pure noise
        x_t = torch.randn(batch_size, self.param_dim, device=device)
        
        # Create sampling timesteps
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_inference_steps,
            dtype=torch.long, device=device
        )
        
        # Denoise iteratively
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            with torch.no_grad():
                # Predict noise
                predicted_noise = self.denoise(x_t, t_batch, condition)
                
                # Get alpha values
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                
                if t > 0:
                    alpha_cumprod_prev = self.alphas_cumprod[t - 1]
                else:
                    alpha_cumprod_prev = torch.tensor(1.0, device=device)
                
                # Compute denoised sample
                pred_x_0 = (x_t - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
                
                # Compute previous sample
                if t > 0:
                    # Add noise for next step
                    noise = torch.randn_like(x_t)
                    x_t = (
                        torch.sqrt(alpha_cumprod_prev) * pred_x_0 +
                        torch.sqrt(1 - alpha_cumprod_prev - alpha * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)) * noise +
                        torch.sqrt(alpha * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)) * predicted_noise
                    )
                else:
                    x_t = pred_x_0
        
        return x_t
    
    def forward(
        self, 
        x_0: torch.Tensor, 
        condition: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            x_0: Clean parameter tensor
            condition: Conditioning tensor
            timesteps: Optional timesteps (random if None)
            
        Returns:
            Dictionary with loss components
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps if not provided
        if timesteps is None:
            timesteps = torch.randint(
                0, self.num_timesteps, (batch_size,), device=device
            )
        
        # Add noise
        x_t, noise = self.add_noise(x_0, timesteps)
        
        # Predict noise
        predicted_noise = self.denoise(x_t, timesteps, condition)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return {
            'loss': loss,
            'predicted_noise': predicted_noise,
            'target_noise': noise,
            'noisy_params': x_t
        }


class ModelTrainer:
    """Trainer for both CycleGAN and DiffusionModel."""
    
    def __init__(
        self,
        cycle_gan: CycleGAN,
        diffusion_model: DiffusionModel,
        device: torch.device,
        learning_rate: float = 1e-4
    ):
        """
        Initialize model trainer.
        
        Args:
            cycle_gan: CycleGAN model
            diffusion_model: Diffusion model
            device: Training device
            learning_rate: Learning rate for optimizers
        """
        self.cycle_gan = cycle_gan
        self.diffusion_model = diffusion_model
        self.device = device
        
        # Optimizers
        self.gen_optimizer = torch.optim.Adam(
            cycle_gan.generator.parameters(), lr=learning_rate
        )
        self.disc_optimizer = torch.optim.Adam(
            cycle_gan.discriminator.parameters(), lr=learning_rate
        )
        self.diffusion_optimizer = torch.optim.Adam(
            diffusion_model.parameters(), lr=learning_rate
        )
        
        logger.info("ModelTrainer initialized")
    
    def train_cycle_gan_step(
        self, 
        real_topology: torch.Tensor,
        spec_condition: torch.Tensor,
        noise: torch.Tensor
    ) -> Dict[str, float]:
        """Train CycleGAN for one step."""
        batch_size = real_topology.shape[0]
        
        # Train Generator
        self.gen_optimizer.zero_grad()
        
        gan_output = self.cycle_gan(noise, spec_condition)
        fake_topology = gan_output['fake_topology']
        fake_score = gan_output['fake_score']
        reconstructed_spec = gan_output['reconstructed_spec']
        
        # Generator losses
        gen_loss = F.binary_cross_entropy(
            fake_score, torch.ones_like(fake_score)
        )
        cycle_loss = F.mse_loss(reconstructed_spec, spec_condition)
        
        total_gen_loss = gen_loss + 10.0 * cycle_loss
        total_gen_loss.backward()
        self.gen_optimizer.step()
        
        # Train Discriminator
        self.disc_optimizer.zero_grad()
        
        real_score = self.cycle_gan.discriminate(real_topology)
        fake_score = self.cycle_gan.discriminate(fake_topology.detach())
        
        real_loss = F.binary_cross_entropy(
            real_score, torch.ones_like(real_score)
        )
        fake_loss = F.binary_cross_entropy(
            fake_score, torch.zeros_like(fake_score)
        )
        
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        self.disc_optimizer.step()
        
        return {
            'gen_loss': gen_loss.item(),
            'cycle_loss': cycle_loss.item(),
            'disc_loss': disc_loss.item()
        }
    
    def train_diffusion_step(
        self,
        clean_params: torch.Tensor,
        condition: torch.Tensor
    ) -> Dict[str, float]:
        """Train diffusion model for one step."""
        self.diffusion_optimizer.zero_grad()
        
        diffusion_output = self.diffusion_model(clean_params, condition)
        loss = diffusion_output['loss']
        
        loss.backward()
        self.diffusion_optimizer.step()
        
        return {'diffusion_loss': loss.item()}


def load_pretrained_models(checkpoint_path: str, device: torch.device) -> Tuple[CycleGAN, DiffusionModel]:
    """
    Load pre-trained models from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load models on
        
    Returns:
        Tuple of (CycleGAN, DiffusionModel)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize models
    cycle_gan = CycleGAN().to(device)
    diffusion_model = DiffusionModel().to(device)
    
    # Load state dicts
    cycle_gan.load_state_dict(checkpoint['cycle_gan_state_dict'])
    diffusion_model.load_state_dict(checkpoint['diffusion_model_state_dict'])
    
    logger.info(f"Models loaded from {checkpoint_path}")
    
    return cycle_gan, diffusion_model


def save_models(
    cycle_gan: CycleGAN,
    diffusion_model: DiffusionModel,
    checkpoint_path: str,
    epoch: int,
    losses: Dict[str, float]
) -> None:
    """
    Save model checkpoint.
    
    Args:
        cycle_gan: CycleGAN model
        diffusion_model: Diffusion model
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch
        losses: Training losses
    """
    checkpoint = {
        'cycle_gan_state_dict': cycle_gan.state_dict(),
        'diffusion_model_state_dict': diffusion_model.state_dict(),
        'epoch': epoch,
        'losses': losses
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Models saved to {checkpoint_path}")