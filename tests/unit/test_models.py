"""Tests for AI models."""

import pytest
import torch
import numpy as np

from genrf.core.models import CycleGAN, DiffusionModel, ModelTrainer


class TestCycleGAN:
    """Test CycleGAN topology generator."""
    
    def test_initialization(self):
        """Test CycleGAN initialization."""
        model = CycleGAN()
        assert model.spec_dim == 9
        assert model.latent_dim == 100
        assert model.topology_dim == 64
        assert model.hidden_dim == 256
    
    def test_generate(self):
        """Test topology generation."""
        model = CycleGAN()
        batch_size = 4
        
        noise = torch.randn(batch_size, model.latent_dim)
        condition = torch.randn(batch_size, model.spec_dim)
        
        topology = model.generate(noise, condition)
        
        assert topology.shape == (batch_size, model.topology_dim)
        assert torch.all(topology >= -1) and torch.all(topology <= 1)
    
    def test_discriminate(self):
        """Test topology discrimination."""
        model = CycleGAN()
        batch_size = 4
        
        topology = torch.randn(batch_size, model.topology_dim)
        score = model.discriminate(topology)
        
        assert score.shape == (batch_size, 1)
        assert torch.all(score >= 0) and torch.all(score <= 1)
    
    def test_cycle_forward(self):
        """Test cycle consistency."""
        model = CycleGAN()
        batch_size = 4
        
        topology = torch.randn(batch_size, model.topology_dim)
        reconstructed = model.cycle_forward(topology)
        
        assert reconstructed.shape == (batch_size, model.spec_dim)
    
    def test_forward(self):
        """Test full forward pass."""
        model = CycleGAN()
        batch_size = 4
        
        noise = torch.randn(batch_size, model.latent_dim)
        condition = torch.randn(batch_size, model.spec_dim)
        
        output = model.forward(noise, condition)
        
        assert 'fake_topology' in output
        assert 'fake_score' in output
        assert 'reconstructed_spec' in output
        
        assert output['fake_topology'].shape == (batch_size, model.topology_dim)
        assert output['fake_score'].shape == (batch_size, 1)
        assert output['reconstructed_spec'].shape == (batch_size, model.spec_dim)


class TestDiffusionModel:
    """Test diffusion parameter optimizer."""
    
    def test_initialization(self):
        """Test diffusion model initialization."""
        model = DiffusionModel()
        assert model.param_dim == 32
        assert model.condition_dim == 16
        assert model.hidden_dim == 256
        assert model.num_timesteps == 1000
    
    def test_add_noise(self):
        """Test noise addition."""
        model = DiffusionModel()
        batch_size = 4
        
        x_0 = torch.randn(batch_size, model.param_dim)
        timesteps = torch.randint(0, model.num_timesteps, (batch_size,))
        
        x_t, noise = model.add_noise(x_0, timesteps)
        
        assert x_t.shape == x_0.shape
        assert noise.shape == x_0.shape
    
    def test_denoise(self):
        """Test denoising prediction."""
        model = DiffusionModel()
        batch_size = 4
        
        x_t = torch.randn(batch_size, model.param_dim)
        timesteps = torch.randint(0, model.num_timesteps, (batch_size,))
        condition = torch.randn(batch_size, model.condition_dim)
        
        predicted_noise = model.denoise(x_t, timesteps, condition)
        
        assert predicted_noise.shape == (batch_size, model.param_dim)
    
    def test_sample(self):
        """Test parameter sampling."""
        model = DiffusionModel()
        batch_size = 4
        
        condition = torch.randn(batch_size, model.condition_dim)
        
        # Use fewer steps for faster testing
        samples = model.sample(condition, num_inference_steps=10)
        
        assert samples.shape == (batch_size, model.param_dim)
    
    def test_forward(self):
        """Test training forward pass."""
        model = DiffusionModel()
        batch_size = 4
        
        x_0 = torch.randn(batch_size, model.param_dim)
        condition = torch.randn(batch_size, model.condition_dim)
        
        output = model.forward(x_0, condition)
        
        assert 'loss' in output
        assert 'predicted_noise' in output
        assert 'target_noise' in output
        assert 'noisy_params' in output
        
        assert isinstance(output['loss'], torch.Tensor)
        assert output['predicted_noise'].shape == (batch_size, model.param_dim)


class TestModelTrainer:
    """Test model trainer."""
    
    def test_initialization(self):
        """Test trainer initialization."""
        device = torch.device('cpu')
        cycle_gan = CycleGAN()
        diffusion_model = DiffusionModel()
        
        trainer = ModelTrainer(cycle_gan, diffusion_model, device)
        
        assert trainer.cycle_gan is cycle_gan
        assert trainer.diffusion_model is diffusion_model
        assert trainer.device is device
    
    def test_train_cycle_gan_step(self):
        """Test CycleGAN training step."""
        device = torch.device('cpu')
        cycle_gan = CycleGAN()
        diffusion_model = DiffusionModel()
        trainer = ModelTrainer(cycle_gan, diffusion_model, device)
        
        batch_size = 4
        real_topology = torch.randn(batch_size, cycle_gan.topology_dim)
        spec_condition = torch.randn(batch_size, cycle_gan.spec_dim)
        noise = torch.randn(batch_size, cycle_gan.latent_dim)
        
        losses = trainer.train_cycle_gan_step(real_topology, spec_condition, noise)
        
        assert 'gen_loss' in losses
        assert 'cycle_loss' in losses
        assert 'disc_loss' in losses
        assert all(isinstance(loss, float) for loss in losses.values())
    
    def test_train_diffusion_step(self):
        """Test diffusion training step."""
        device = torch.device('cpu')
        cycle_gan = CycleGAN()
        diffusion_model = DiffusionModel()
        trainer = ModelTrainer(cycle_gan, diffusion_model, device)
        
        batch_size = 4
        clean_params = torch.randn(batch_size, diffusion_model.param_dim)
        condition = torch.randn(batch_size, diffusion_model.condition_dim)
        
        losses = trainer.train_diffusion_step(clean_params, condition)
        
        assert 'diffusion_loss' in losses
        assert isinstance(losses['diffusion_loss'], float)