"""Performance evaluator for RF circuit topologies."""

import numpy as np
from typing import Dict
from .spice_simulator import SPICESimulator
from .topology import CircuitTopology


class PerformanceEvaluator:
    """Evaluates circuit performance metrics: bandwidth, gain, Q-factor, stability."""

    def __init__(self, simulator: SPICESimulator):
        self.simulator = simulator

    def bandwidth_from_transfer(self, H: np.ndarray, freqs: np.ndarray) -> float:
        """Compute -3dB bandwidth from transfer function H and frequency array."""
        mag = np.abs(H)
        if np.max(mag) == 0:
            return 0.0

        peak_mag = np.max(mag)
        threshold = peak_mag / np.sqrt(2.0)  # -3dB point

        above = mag >= threshold
        if not np.any(above):
            return 0.0

        # Find first and last frequency above -3dB threshold
        indices = np.where(above)[0]
        f_low = freqs[indices[0]]
        f_high = freqs[indices[-1]]
        return float(f_high - f_low)

    def gain_db_from_transfer(self, H: np.ndarray) -> float:
        """Compute peak gain in dB from transfer function."""
        mag = np.abs(H)
        peak = np.max(mag)
        if peak <= 0:
            return -np.inf
        return float(20.0 * np.log10(peak))

    def evaluate(self, topology: CircuitTopology) -> Dict:
        """
        Evaluate circuit performance metrics.
        Returns bandwidth_hz, gain_db, resonant_freq, q_factor, is_stable.
        """
        freqs = self.simulator.freq_points
        H = self.simulator.transfer_function(topology)

        bw = self.bandwidth_from_transfer(H, freqs)
        gain = self.gain_db_from_transfer(H)

        # Resonant frequency: frequency of peak magnitude
        mag = np.abs(H)
        peak_idx = int(np.argmax(mag))
        resonant_freq = float(freqs[peak_idx])

        # Q factor: resonant_freq / bandwidth
        q_factor = resonant_freq / bw if bw > 0 else 0.0

        # Stability check: capacitors cause issues at DC (f=0), but we start at 1kHz
        # Check no component causes instability (infinite impedance for C at DC)
        is_stable = self._check_stability(topology)

        return {
            "bandwidth_hz": bw,
            "gain_db": gain,
            "resonant_freq": resonant_freq,
            "q_factor": q_factor,
            "is_stable": is_stable,
        }

    def _check_stability(self, topology: CircuitTopology) -> bool:
        """Check that no component causes infinite impedance issues."""
        components = topology.get_components()
        for comp in components:
            # Capacitor with extremely small value causes issues
            if comp["type"] == "C" and comp["value"] < 1e-20:
                return False
            # Inductor with extremely large value causes issues at high freq
            if comp["type"] == "L" and comp["value"] > 1e6:
                return False
        return True

    def compare(self, topology: CircuitTopology, target_spec: Dict) -> Dict:
        """Compare circuit performance against target spec."""
        actual = self.evaluate(topology)

        target_bw = float(target_spec.get("bandwidth_hz", 1e6))
        target_gain = float(target_spec.get("gain_db", 0.0))

        bw_error = abs(actual["bandwidth_hz"] - target_bw) / max(target_bw, 1.0)
        gain_error = abs(actual["gain_db"] - target_gain)

        return {
            "target": {
                "bandwidth_hz": target_bw,
                "gain_db": target_gain,
            },
            "actual": actual,
            "bandwidth_error_pct": bw_error * 100.0,
            "gain_error_db": gain_error,
            "score": 1.0 / (1.0 + bw_error + gain_error * 0.1),
        }
