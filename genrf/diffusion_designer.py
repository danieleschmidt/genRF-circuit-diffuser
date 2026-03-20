"""DiffusionDesigner: Gaussian noise + denoise loop that samples circuit parameters."""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from .topology import CircuitTopology


class DiffusionDesigner:
    """
    Simplified diffusion-based circuit parameter sampler.
    
    Forward process: x_t = x_0 + sigma(t) * eps
    Reverse process: iterative denoising toward low-loss circuits.
    """

    def __init__(
        self,
        n_steps: int = 50,
        sigma_max: float = 1.0,
        sigma_min: float = 0.01,
        seed: int = 42,
    ):
        self.n_steps = n_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rng = np.random.default_rng(seed)

        # Noise schedule: linear from sigma_max → sigma_min
        self.sigmas = np.linspace(sigma_max, sigma_min, n_steps)

    def forward(self, params: np.ndarray, t: int) -> np.ndarray:
        """Add noise at step t."""
        noise = self.rng.standard_normal(params.shape)
        return params + self.sigmas[t] * noise

    def denoise_step(
        self,
        x_t: np.ndarray,
        t: int,
        score_fn: Callable[[np.ndarray], np.ndarray],
        step_size: float = 0.1,
    ) -> np.ndarray:
        """
        One denoising step using Langevin MCMC update.
        
        x_{t-1} = x_t + step_size * score(x_t) + sqrt(2*step_size) * eps
        """
        score = score_fn(x_t)
        noise = self.rng.standard_normal(x_t.shape)
        sigma = self.sigmas[t]
        x_next = x_t + step_size * sigma * score + np.sqrt(2 * step_size) * sigma * noise
        return x_next

    def sample(
        self,
        template: CircuitTopology,
        score_fn: Callable[[np.ndarray], np.ndarray],
        n_samples: int = 1,
    ) -> List[np.ndarray]:
        """
        Sample circuit parameters via reverse diffusion.
        
        Args:
            template: base topology to parameterize
            score_fn: gradient of log p(x) — should push toward good circuits
            n_samples: number of circuit samples to generate
        
        Returns:
            List of parameter vectors
        """
        n_params = len(template.components)
        # Start from noise
        x = self.rng.standard_normal((n_samples, n_params)) * self.sigma_max

        # Reverse diffusion
        for t in range(self.n_steps - 1, -1, -1):
            for i in range(n_samples):
                x[i] = self.denoise_step(x[i], t, score_fn)
            # Project to positive values (physical constraint)
            x = np.abs(x) + 1e-15

        return [x[i] for i in range(n_samples)]

    def sample_random(self, template: CircuitTopology, n_samples: int = 1) -> List[np.ndarray]:
        """Sample with a trivial identity score (pure noise → clipped random)."""
        def trivial_score(x):
            return -x  # mean-reverting score
        return self.sample(template, trivial_score, n_samples)
