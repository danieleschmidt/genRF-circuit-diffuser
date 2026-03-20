"""Diffusion model for circuit parameter generation."""

import numpy as np
from typing import List, Dict


class DiffusionDesigner:
    """
    Simplified diffusion model for generating circuit parameter vectors.
    Uses a linear beta noise schedule and score-based denoising.
    """

    def __init__(
        self,
        n_params: int = 20,
        n_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.n_params = n_params
        self.n_steps = n_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Linear beta schedule
        self.betas = np.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)  # cumulative product

        # Precompute sqrt terms for noise schedule
        self.sqrt_alpha_bars = np.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = np.sqrt(1.0 - self.alpha_bars)

    @property
    def noise_schedule(self) -> np.ndarray:
        """Return the beta noise schedule."""
        return self.betas.copy()

    def add_noise(self, params: np.ndarray, t: int) -> np.ndarray:
        """Add noise at diffusion step t using the forward process."""
        t = int(np.clip(t, 0, self.n_steps - 1))
        sqrt_ab = self.sqrt_alpha_bars[t]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bars[t]
        noise = np.random.randn(*params.shape)
        return sqrt_ab * params + sqrt_one_minus_ab * noise

    def _spec_embedding(self, spec: Dict) -> np.ndarray:
        """Encode spec dict into a fixed-size condition vector."""
        # Simple linear embedding of spec values
        bw = float(spec.get("bandwidth_hz", 1e6))
        gain = float(spec.get("gain_db", 0.0))
        topo_map = {"lowpass": 0.0, "highpass": 0.5, "bandpass": 1.0}
        topo_val = topo_map.get(str(spec.get("topology", "lowpass")), 0.0)

        # Normalize to reasonable range
        bw_norm = np.log10(max(bw, 1.0)) / 9.0   # 1Hz to 1GHz -> [0,1]
        gain_norm = (gain + 60.0) / 120.0          # -60dB to +60dB -> [0,1]

        embed = np.zeros(self.n_params, dtype=np.float64)
        embed[0] = bw_norm
        embed[1] = gain_norm
        embed[2] = topo_val
        # Fill rest with a simple pattern derived from spec
        for i in range(3, self.n_params):
            embed[i] = (bw_norm * np.sin(i * 0.3) + gain_norm * np.cos(i * 0.2)) * 0.1
        return embed

    def score_fn(self, params: np.ndarray, t: int, spec_condition: Dict) -> np.ndarray:
        """
        Score function (gradient of log p(x_t | spec)).
        Simplified: returns direction toward spec-conditioned mean.
        """
        t = int(np.clip(t, 0, self.n_steps - 1))
        spec_embed = self._spec_embedding(spec_condition)

        # Score = -(x - mu) / sigma^2 (Gaussian score)
        sigma2 = max(1.0 - self.alpha_bars[t], 1e-8)
        score = -(params - spec_embed) / sigma2
        return score

    def design(self, spec: Dict, n_circuits: int = 3) -> List[np.ndarray]:
        """
        Generate n_circuits parameter vectors conditioned on spec.
        Uses reverse diffusion (denoising) process.
        """
        spec_embed = self._spec_embedding(spec)
        results = []

        for _ in range(n_circuits):
            # Start from pure noise
            x = np.random.randn(self.n_params).astype(np.float64)

            # Reverse diffusion: denoise from t=T to t=0
            for t in reversed(range(self.n_steps)):
                alpha_t = self.alphas[t]
                alpha_bar_t = self.alpha_bars[t]
                beta_t = self.betas[t]

                # Score-based denoising step
                score = self.score_fn(x, t, spec)
                # Langevin-style update
                x = (1.0 / np.sqrt(alpha_t)) * (x + beta_t * score)

                # Add noise for t > 0 (stochastic sampling)
                if t > 0:
                    noise = np.random.randn(self.n_params)
                    x = x + np.sqrt(beta_t) * noise * 0.1

            # Blend result toward spec embedding for conditioning
            alpha_blend = 0.6
            x = alpha_blend * spec_embed + (1.0 - alpha_blend) * x
            # Ensure positive values for physical parameters
            x = np.abs(x)
            results.append(x)

        return results

    def param_to_values(self, params: np.ndarray) -> Dict:
        """
        Convert a parameter vector to physical R/L/C values.
        Maps groups of params to component values with physical scaling.
        """
        values = {}
        # First third: R values (Ohms, scaled to 1-10k range)
        n_r = self.n_params // 3
        r_vals = np.abs(params[:n_r]) * 1000.0 + 1.0  # 1-1001 Ohm range
        values["R"] = r_vals.tolist()

        # Middle third: C values (Farads, scaled to pF-nF range)
        n_c = self.n_params // 3
        c_vals = np.abs(params[n_r:n_r + n_c]) * 1e-9 + 1e-12  # pF to nF range
        values["C"] = c_vals.tolist()

        # Last part: L values (Henries, scaled to nH-uH range)
        l_vals = np.abs(params[n_r + n_c:]) * 1e-6 + 1e-9  # nH to uH range
        values["L"] = l_vals.tolist()

        return values
