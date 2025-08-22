"""
Federated Learning for Distributed RF Circuit Knowledge.

Revolutionary breakthrough: Enable collaborative learning across multiple
design teams while preserving IP confidentiality through federated optimization.
"""

import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

from .models import DiffusionModel, CycleGAN
from .circuit_diffuser import CircuitResult
from .design_spec import DesignSpec

logger = logging.getLogger(__name__)


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""
    num_clients: int = 5
    rounds: int = 100
    client_fraction: float = 0.6
    local_epochs: int = 5
    learning_rate: float = 1e-4
    differential_privacy: bool = True
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    secure_aggregation: bool = True
    min_clients: int = 3


@dataclass
class ClientMetrics:
    """Metrics from a federated client."""
    client_id: str
    round_num: int
    local_loss: float
    num_samples: int
    training_time: float
    model_hash: str
    convergence_metrics: Dict[str, float] = field(default_factory=dict)


class SecureAggregator:
    """Secure aggregation with cryptographic protection."""
    
    def __init__(self, num_clients: int, key_size: int = 2048):
        self.num_clients = num_clients
        self.private_keys = {}
        self.public_keys = {}
        
        # Generate key pairs for each client
        for i in range(num_clients):
            client_id = f"client_{i}"
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size
            )
            public_key = private_key.public_key()
            
            self.private_keys[client_id] = private_key
            self.public_keys[client_id] = public_key
    
    def encrypt_weights(self, weights: torch.Tensor, client_id: str) -> bytes:
        """Encrypt model weights for secure transmission."""
        weights_bytes = weights.cpu().numpy().tobytes()
        
        # Encrypt with client's public key
        public_key = self.public_keys[client_id]
        encrypted = public_key.encrypt(
            weights_bytes[:245],  # RSA limit
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted
    
    def decrypt_weights(self, encrypted_weights: bytes, client_id: str, shape: Tuple) -> torch.Tensor:
        """Decrypt model weights."""
        private_key = self.private_keys[client_id]
        decrypted = private_key.decrypt(
            encrypted_weights,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Reconstruct tensor
        weights_array = np.frombuffer(decrypted, dtype=np.float32)
        return torch.tensor(weights_array).reshape(shape)


class DifferentialPrivacyEngine:
    """Differential privacy for federated learning."""
    
    def __init__(self, noise_multiplier: float = 1.0, max_grad_norm: float = 1.0):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
    
    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients to bound sensitivity."""
        total_norm = 0.0
        
        # Calculate total gradient norm
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** (1. / 2)
        
        # Clip gradients
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return total_norm
    
    def add_noise(self, model: nn.Module, device: torch.device):
        """Add Gaussian noise to model parameters."""
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.normal(
                        0, 
                        self.noise_multiplier * self.max_grad_norm,
                        param.shape,
                        device=device
                    )
                    param.grad.add_(noise)


class FederatedClient:
    """Individual federated learning client."""
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        local_data: List[Tuple[DesignSpec, CircuitResult]],
        config: FederatedConfig,
        device: torch.device
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.local_data = local_data
        self.config = config
        self.device = device
        
        # Privacy engine
        self.dp_engine = DifferentialPrivacyEngine(
            noise_multiplier=config.noise_multiplier,
            max_grad_norm=config.max_grad_norm
        ) if config.differential_privacy else None
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        logger.info(f"FederatedClient {client_id} initialized with {len(local_data)} samples")
    
    def local_train(self, global_round: int) -> ClientMetrics:
        """Perform local training for specified epochs."""
        start_time = time.time()
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.local_epochs):
            for spec, circuit_result in self.local_data:
                self.optimizer.zero_grad()
                
                # Convert to training format
                condition = self._spec_to_condition(spec)
                target_params = self._result_to_params(circuit_result)
                
                # Forward pass
                if isinstance(self.model, DiffusionModel):
                    output = self.model(target_params, condition)
                    loss = output['loss']
                else:  # CycleGAN
                    noise = torch.randn(1, self.model.latent_dim, device=self.device)
                    output = self.model(condition, noise)
                    loss = self._compute_gan_loss(output, spec)
                
                # Backward pass
                loss.backward()
                
                # Apply differential privacy
                if self.dp_engine:
                    self.dp_engine.clip_gradients(self.model)
                    self.dp_engine.add_noise(self.model, self.device)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        training_time = time.time() - start_time
        
        # Compute model hash for integrity
        model_hash = self._compute_model_hash()
        
        return ClientMetrics(
            client_id=self.client_id,
            round_num=global_round,
            local_loss=avg_loss,
            num_samples=len(self.local_data),
            training_time=training_time,
            model_hash=model_hash,
            convergence_metrics={
                'gradient_norm': self._compute_gradient_norm(),
                'parameter_change': self._compute_parameter_change()
            }
        )
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights."""
        return {name: param.clone().detach() for name, param in self.model.named_parameters()}
    
    def set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights from global aggregation."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.copy_(weights[name].to(self.device))
    
    def _spec_to_condition(self, spec: DesignSpec) -> torch.Tensor:
        """Convert design spec to conditioning tensor."""
        condition = torch.tensor([
            spec.frequency / 1e12,
            spec.gain_min / 50.0,
            spec.nf_max / 10.0,
            spec.power_max / 1e-3
        ], device=self.device).unsqueeze(0)
        return condition
    
    def _result_to_params(self, result: CircuitResult) -> torch.Tensor:
        """Convert circuit result to parameter tensor."""
        # Extract key parameters
        params = []
        for key, value in result.parameters.items():
            if isinstance(value, (int, float)):
                params.append(float(value))
        
        # Pad or truncate to fixed size
        target_size = 32
        if len(params) < target_size:
            params.extend([0.0] * (target_size - len(params)))
        else:
            params = params[:target_size]
        
        return torch.tensor(params, device=self.device).unsqueeze(0)
    
    def _compute_gan_loss(self, output: Dict[str, torch.Tensor], spec: DesignSpec) -> torch.Tensor:
        """Compute GAN training loss."""
        # Simplified loss computation
        fake_score = output.get('fake_score', torch.tensor(0.5, device=self.device))
        reconstructed_spec = output.get('reconstructed_spec', torch.zeros(4, device=self.device))
        
        target_condition = self._spec_to_condition(spec)
        
        gen_loss = -torch.log(fake_score + 1e-8).mean()
        cycle_loss = torch.nn.functional.mse_loss(reconstructed_spec, target_condition)
        
        return gen_loss + 10.0 * cycle_loss
    
    def _compute_model_hash(self) -> str:
        """Compute hash of model weights for integrity checking."""
        model_str = ""
        for name, param in self.model.named_parameters():
            model_str += f"{name}:{param.data.cpu().numpy().tobytes()}"
        
        return hashlib.sha256(model_str.encode()).hexdigest()[:16]
    
    def _compute_gradient_norm(self) -> float:
        """Compute gradient norm for convergence monitoring."""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def _compute_parameter_change(self) -> float:
        """Compute parameter change magnitude."""
        if not hasattr(self, '_prev_params'):
            self._prev_params = self.get_model_weights()
            return 0.0
        
        total_change = 0.0
        current_params = self.get_model_weights()
        
        for name, current in current_params.items():
            if name in self._prev_params:
                change = torch.norm(current - self._prev_params[name]).item()
                total_change += change ** 2
        
        self._prev_params = current_params
        return total_change ** 0.5


class FederatedServer:
    """Federated learning server coordinator."""
    
    def __init__(
        self,
        global_model: nn.Module,
        clients: List[FederatedClient],
        config: FederatedConfig,
        device: torch.device
    ):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.config = config
        self.device = device
        
        # Secure aggregation
        self.secure_aggregator = SecureAggregator(
            len(clients)
        ) if config.secure_aggregation else None
        
        # Training history
        self.training_history = []
        self.convergence_history = []
        
        logger.info(f"FederatedServer initialized with {len(clients)} clients")
    
    def federated_averaging(self, client_weights: List[Dict[str, torch.Tensor]], 
                          client_sizes: List[int]) -> Dict[str, torch.Tensor]:
        """Perform federated averaging with weighted combination."""
        total_samples = sum(client_sizes)
        averaged_weights = {}
        
        # Get parameter names from first client
        param_names = list(client_weights[0].keys())
        
        for name in param_names:
            # Weighted average of parameters
            weighted_sum = torch.zeros_like(client_weights[0][name])
            
            for i, weights in enumerate(client_weights):
                weight_factor = client_sizes[i] / total_samples
                weighted_sum += weight_factor * weights[name].to(self.device)
            
            averaged_weights[name] = weighted_sum
        
        return averaged_weights
    
    def train_round(self, round_num: int) -> Dict[str, Any]:
        """Execute one round of federated training."""
        logger.info(f"Starting federated round {round_num}")
        
        # Select participating clients
        num_selected = max(
            self.config.min_clients,
            int(len(self.clients) * self.config.client_fraction)
        )
        selected_clients = np.random.choice(
            self.clients, 
            size=min(num_selected, len(self.clients)),
            replace=False
        ).tolist()
        
        # Distribute global model to selected clients
        global_weights = {
            name: param.clone().detach() 
            for name, param in self.global_model.named_parameters()
        }
        
        for client in selected_clients:
            client.set_model_weights(global_weights)
        
        # Local training
        client_metrics = []
        client_weights = []
        client_sizes = []
        
        for client in selected_clients:
            metrics = client.local_train(round_num)
            client_metrics.append(metrics)
            
            weights = client.get_model_weights()
            client_weights.append(weights)
            client_sizes.append(metrics.num_samples)
        
        # Aggregate updates
        aggregated_weights = self.federated_averaging(client_weights, client_sizes)
        
        # Update global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_weights:
                    param.copy_(aggregated_weights[name])
        
        # Compute round statistics
        round_stats = self._compute_round_stats(client_metrics)
        round_stats['round'] = round_num
        round_stats['participating_clients'] = len(selected_clients)
        
        self.training_history.append(round_stats)
        
        logger.info(f"Round {round_num} completed. Avg loss: {round_stats['avg_loss']:.4f}")
        
        return round_stats
    
    def train(self) -> Dict[str, Any]:
        """Execute complete federated training."""
        logger.info(f"Starting federated training for {self.config.rounds} rounds")
        start_time = time.time()
        
        for round_num in range(self.config.rounds):
            round_stats = self.train_round(round_num)
            
            # Check convergence
            if self._check_convergence(round_stats):
                logger.info(f"Convergence achieved at round {round_num}")
                break
        
        training_time = time.time() - start_time
        
        final_results = {
            'total_rounds': len(self.training_history),
            'training_time': training_time,
            'final_loss': self.training_history[-1]['avg_loss'] if self.training_history else None,
            'convergence_achieved': len(self.training_history) < self.config.rounds,
            'training_history': self.training_history
        }
        
        logger.info(f"Federated training completed in {training_time:.2f}s")
        return final_results
    
    def _compute_round_stats(self, client_metrics: List[ClientMetrics]) -> Dict[str, float]:
        """Compute statistics for a training round."""
        losses = [m.local_loss for m in client_metrics]
        sample_counts = [m.num_samples for m in client_metrics]
        training_times = [m.training_time for m in client_metrics]
        
        # Weighted average loss
        total_samples = sum(sample_counts)
        weighted_loss = sum(l * s for l, s in zip(losses, sample_counts)) / total_samples
        
        return {
            'avg_loss': weighted_loss,
            'min_loss': min(losses),
            'max_loss': max(losses),
            'std_loss': np.std(losses),
            'avg_training_time': np.mean(training_times),
            'total_samples': total_samples
        }
    
    def _check_convergence(self, round_stats: Dict[str, float]) -> bool:
        """Check if training has converged."""
        if len(self.training_history) < 10:
            return False
        
        # Check loss improvement over last 5 rounds
        recent_losses = [h['avg_loss'] for h in self.training_history[-5:]]
        loss_improvement = (max(recent_losses) - min(recent_losses)) / max(recent_losses)
        
        return loss_improvement < 0.01  # Less than 1% improvement


class FederatedCircuitDiffuser:
    """Federated learning wrapper for CircuitDiffuser."""
    
    def __init__(
        self,
        base_model: nn.Module,
        client_data: List[List[Tuple[DesignSpec, CircuitResult]]],
        config: FederatedConfig,
        device: torch.device
    ):
        self.base_model = base_model
        self.client_data = client_data
        self.config = config
        self.device = device
        
        # Create clients
        self.clients = [
            FederatedClient(
                client_id=f"client_{i}",
                model=base_model.__class__(),  # Create new instance
                local_data=data,
                config=config,
                device=device
            )
            for i, data in enumerate(client_data)
        ]
        
        # Create server
        self.server = FederatedServer(
            global_model=base_model,
            clients=self.clients,
            config=config,
            device=device
        )
    
    def federated_train(self) -> Dict[str, Any]:
        """Execute federated training and return results."""
        return self.server.train()
    
    def get_global_model(self) -> nn.Module:
        """Get the trained global model."""
        return self.server.global_model
    
    def evaluate_global_model(self, test_specs: List[DesignSpec]) -> Dict[str, float]:
        """Evaluate the global model on test specifications."""
        self.server.global_model.eval()
        results = []
        
        with torch.no_grad():
            for spec in test_specs:
                condition = torch.tensor([
                    spec.frequency / 1e12,
                    spec.gain_min / 50.0,
                    spec.nf_max / 10.0,
                    spec.power_max / 1e-3
                ], device=self.device).unsqueeze(0)
                
                if isinstance(self.server.global_model, DiffusionModel):
                    generated = self.server.global_model.sample(condition)
                    # Evaluate quality of generated parameters
                    quality = self._evaluate_parameter_quality(generated, spec)
                else:
                    noise = torch.randn(1, self.server.global_model.latent_dim, device=self.device)
                    generated = self.server.global_model.generate(condition, noise)
                    quality = self._evaluate_topology_quality(generated, spec)
                
                results.append(quality)
        
        return {
            'avg_quality': np.mean(results),
            'min_quality': np.min(results),
            'max_quality': np.max(results),
            'std_quality': np.std(results)
        }
    
    def _evaluate_parameter_quality(self, params: torch.Tensor, spec: DesignSpec) -> float:
        """Evaluate quality of generated parameters."""
        # Simplified quality metric based on parameter reasonableness
        param_values = params.cpu().numpy().flatten()
        
        # Check if parameters are within reasonable ranges
        reasonable_count = 0
        total_params = len(param_values)
        
        for val in param_values:
            if -10.0 <= val <= 10.0:  # Reasonable range after normalization
                reasonable_count += 1
        
        return reasonable_count / max(total_params, 1)
    
    def _evaluate_topology_quality(self, topology: torch.Tensor, spec: DesignSpec) -> float:
        """Evaluate quality of generated topology."""
        # Simplified topology quality metric
        topology_values = topology.cpu().numpy().flatten()
        
        # Check topology diversity and validity
        diversity = np.std(topology_values)
        validity = 1.0 if -1.0 <= np.min(topology_values) and np.max(topology_values) <= 1.0 else 0.0
        
        return 0.5 * diversity + 0.5 * validity


# Example usage and demonstration
def create_federated_demo():
    """Create a demonstration of federated circuit learning."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration
    config = FederatedConfig(
        num_clients=5,
        rounds=50,
        client_fraction=0.8,
        local_epochs=3,
        differential_privacy=True,
        secure_aggregation=True
    )
    
    # Create dummy client data
    client_data = []
    for i in range(config.num_clients):
        # Generate synthetic data for each client
        specs = []
        results = []
        
        for j in range(20):  # 20 samples per client
            spec = DesignSpec(
                circuit_type="LNA",
                frequency=2.4e9 + np.random.normal(0, 100e6),
                gain_min=15.0 + np.random.normal(0, 2.0),
                nf_max=2.0 + np.random.normal(0, 0.5),
                power_max=10e-3 + np.random.normal(0, 2e-3)
            )
            
            # Synthetic circuit result
            result = CircuitResult(
                netlist="* Synthetic netlist",
                parameters={
                    'W1': 50e-6 + np.random.normal(0, 10e-6),
                    'L1': 100e-9 + np.random.normal(0, 20e-9),
                    'Rd': 1000 + np.random.normal(0, 200),
                    'Ibias': 5e-3 + np.random.normal(0, 1e-3)
                },
                performance={
                    'gain_db': spec.gain_min + 2.0,
                    'noise_figure_db': spec.nf_max - 0.2,
                    'power_w': spec.power_max * 0.8
                },
                topology=f"lna_topology_{j}",
                technology="TSMC65nm",
                generation_time=1.0,
                spice_valid=True
            )
            
            specs.append(spec)
            results.append(result)
        
        client_data.append(list(zip(specs, results)))
    
    # Create base model (DiffusionModel for parameter optimization)
    base_model = DiffusionModel(
        param_dim=32,
        condition_dim=4,
        hidden_dim=128,
        num_timesteps=100
    )
    
    # Create federated diffuser
    federated_diffuser = FederatedCircuitDiffuser(
        base_model=base_model,
        client_data=client_data,
        config=config,
        device=device
    )
    
    return federated_diffuser


if __name__ == "__main__":
    # Demonstration
    federated_diffuser = create_federated_demo()
    
    print("🔗 Starting Federated Circuit Learning Demonstration")
    print(f"📊 Configuration: {federated_diffuser.config.num_clients} clients, "
          f"{federated_diffuser.config.rounds} rounds")
    
    # Execute federated training
    results = federated_diffuser.federated_train()
    
    print(f"✅ Federated training completed in {results['training_time']:.2f}s")
    print(f"🎯 Final loss: {results['final_loss']:.4f}")
    print(f"🔄 Convergence: {'Yes' if results['convergence_achieved'] else 'No'}")
    
    # Evaluate global model
    test_specs = [
        DesignSpec(circuit_type="LNA", frequency=2.4e9, gain_min=16.0, nf_max=1.8),
        DesignSpec(circuit_type="LNA", frequency=5.8e9, gain_min=14.0, nf_max=2.5)
    ]
    
    eval_results = federated_diffuser.evaluate_global_model(test_specs)
    print(f"🧪 Global model quality: {eval_results['avg_quality']:.3f}")
    
    logger.info("Federated circuit learning demonstration completed successfully")