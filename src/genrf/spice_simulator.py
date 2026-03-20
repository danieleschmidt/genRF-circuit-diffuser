"""Simplified SPICE-like AC simulator using impedance calculations."""

import numpy as np
from typing import Optional, Dict
from .topology import CircuitTopology


FREQ_POINTS = np.logspace(3, 9, 50)  # 1kHz to 1GHz


class SPICESimulator:
    """AC analysis simulator using impedance calculations."""

    def __init__(self):
        self.freq_points = FREQ_POINTS.copy()

    def impedance_R(self, R: float, freq: np.ndarray) -> np.ndarray:
        """Resistor impedance: constant R (real)."""
        return np.full(len(freq), R, dtype=np.complex128)

    def impedance_C(self, C: float, freq: np.ndarray) -> np.ndarray:
        """Capacitor impedance: 1 / (j * 2*pi*f*C)."""
        omega = 2.0 * np.pi * freq
        with np.errstate(divide='ignore', invalid='ignore'):
            z = np.where(omega * C != 0, 1.0 / (1j * omega * C), np.inf + 0j)
        return z.astype(np.complex128)

    def impedance_L(self, L: float, freq: np.ndarray) -> np.ndarray:
        """Inductor impedance: j * 2*pi*f*L."""
        omega = 2.0 * np.pi * freq
        return (1j * omega * L).astype(np.complex128)

    def _component_impedance(self, comp: dict, freq: np.ndarray) -> np.ndarray:
        """Get impedance of a single component."""
        t = comp["type"]
        v = comp["value"]
        if t == "R":
            return self.impedance_R(v, freq)
        elif t == "C":
            return self.impedance_C(v, freq)
        elif t == "L":
            return self.impedance_L(v, freq)
        elif t == "V":
            # Ideal voltage source: zero impedance
            return np.zeros(len(freq), dtype=np.complex128)
        else:
            # GND or unknown: treat as large resistor
            return np.full(len(freq), 1e12, dtype=np.complex128)

    def simulate(self, topology: CircuitTopology, freq_points: Optional[np.ndarray] = None) -> Dict:
        """
        Run AC simulation. Returns frequencies, total series impedance, and phase.
        Simplified model: compute total series impedance of all RLC branches.
        """
        if freq_points is None:
            freq_points = self.freq_points

        components = topology.get_components()
        if not components:
            z_total = np.ones(len(freq_points), dtype=np.complex128)
        else:
            # Sum all component impedances in series (simplified model)
            z_total = np.zeros(len(freq_points), dtype=np.complex128)
            for comp in components:
                if comp["type"] not in ("V", "GND"):
                    z_total += self._component_impedance(comp, freq_points)

            # If no passive components, use unit impedance
            if np.allclose(z_total, 0):
                z_total = np.ones(len(freq_points), dtype=np.complex128)

        phase = np.angle(z_total, deg=True)
        return {
            "frequencies": freq_points.copy(),
            "impedance": z_total,
            "phase": phase,
        }

    def transfer_function(
        self,
        topology: CircuitTopology,
        input_node: int = 0,
        output_node: int = 1,
        freq_points: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute voltage gain vs frequency (simplified voltage divider model).
        Splits components into input-side and output-side based on node connectivity.
        """
        if freq_points is None:
            freq_points = self.freq_points

        components = topology.get_components()
        passive = [c for c in components if c["type"] not in ("V", "GND")]

        if not passive:
            return np.ones(len(freq_points), dtype=np.complex128)

        # Voltage divider: Z_out / (Z_in + Z_out)
        # Components connected to output_node go to Z_out, rest to Z_in
        z_in = np.zeros(len(freq_points), dtype=np.complex128)
        z_out = np.zeros(len(freq_points), dtype=np.complex128)

        for comp in passive:
            z_comp = self._component_impedance(comp, freq_points)
            if comp["node1"] == output_node or comp["node2"] == output_node:
                z_out += z_comp
            else:
                z_in += z_comp

        # Avoid division by zero
        denom = z_in + z_out
        denom = np.where(np.abs(denom) < 1e-30, 1e-30 + 0j, denom)

        with np.errstate(divide='ignore', invalid='ignore'):
            H = np.where(np.abs(z_out) > 0, z_out / denom, np.ones(len(freq_points), dtype=np.complex128))

        return H.astype(np.complex128)
