"""PerformanceEvaluator: bandwidth, gain, impedance match."""

import numpy as np
from typing import Dict, Optional, Tuple

from .topology import CircuitTopology
from .spice_sim import SPICESimulator


class PerformanceEvaluator:
    """Evaluate RF circuit performance metrics from AC simulation results."""

    def __init__(self, target_freq: float = 1e9, z0: float = 50.0):
        """
        Args:
            target_freq: center/target frequency in Hz
            z0: reference impedance for S-parameters (Ohms)
        """
        self.target_freq = target_freq
        self.z0 = z0
        self.simulator = SPICESimulator()

    def evaluate(self, topo: CircuitTopology) -> Dict[str, float]:
        """Compute bandwidth, gain, impedance match for a circuit."""
        try:
            result = self.simulator.simulate(topo)
        except Exception:
            return {"bandwidth": 0.0, "gain_db": -100.0, "impedance_match": 0.0, "score": 0.0}

        freqs = result["frequencies"]
        H = result["H"]
        mag_db = result["magnitude_db"]

        gain_db = self._gain_at_freq(freqs, mag_db, self.target_freq)
        bw = self._bandwidth(freqs, mag_db)
        imp_match = self._impedance_match(freqs, H, self.target_freq)
        score = self._composite_score(gain_db, bw, imp_match)

        return {
            "bandwidth": float(bw),
            "gain_db": float(gain_db),
            "impedance_match": float(imp_match),
            "score": float(score),
        }

    def _gain_at_freq(self, freqs: np.ndarray, mag_db: np.ndarray, f: float) -> float:
        """Get gain at the nearest frequency point."""
        idx = np.argmin(np.abs(freqs - f))
        return float(mag_db[idx])

    def _bandwidth(self, freqs: np.ndarray, mag_db: np.ndarray) -> float:
        """Compute -3dB bandwidth."""
        peak = np.max(mag_db)
        threshold = peak - 3.0
        above = freqs[mag_db >= threshold]
        if len(above) < 2:
            return float(freqs[-1] - freqs[0]) / 1e3 if len(above) < 1 else 0.0
        return float(above[-1] - above[0])

    def _impedance_match(self, freqs: np.ndarray, H: np.ndarray, f: float) -> float:
        """
        Estimate impedance match quality at target frequency.
        Returns value in [0, 1] where 1 = perfect match.
        """
        idx = np.argmin(np.abs(freqs - f))
        z_out = np.abs(H[idx]) * self.z0
        # Reflection coefficient magnitude
        gamma = np.abs((z_out - self.z0) / (z_out + self.z0 + 1e-10))
        return float(np.clip(1.0 - gamma, 0, 1))

    def _composite_score(self, gain_db: float, bandwidth: float, imp_match: float) -> float:
        """Composite quality score (higher = better)."""
        # Normalize gain: 0 dB is ideal for a filter passband
        gain_norm = np.clip(1.0 - abs(gain_db) / 60.0, 0, 1)
        # Normalize bandwidth: reward wider bandwidth
        bw_norm = np.clip(bandwidth / 1e9, 0, 1)
        return float(0.4 * gain_norm + 0.3 * bw_norm + 0.3 * imp_match)

    def batch_evaluate(self, topologies) -> list:
        return [self.evaluate(t) for t in topologies]
