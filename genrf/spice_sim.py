"""SPICESimulator: simplified AC analysis using impedance calculation."""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .topology import CircuitTopology, Component


class SPICESimulator:
    """
    Simplified AC frequency-domain analysis using impedance methods.
    
    Computes voltage transfer function H(f) for a simple ladder topology:
      IN → L → OUT → C ∥ R_load → GND
    """

    def __init__(self, freq_points: int = 200):
        self.freq_points = freq_points

    def _component_impedance(self, comp: Component, omega: np.ndarray) -> np.ndarray:
        """Compute complex impedance of a component at angular frequencies omega."""
        if comp.ctype == "R":
            return np.full_like(omega, comp.value, dtype=complex)
        elif comp.ctype == "L":
            return 1j * omega * comp.value
        elif comp.ctype == "C":
            return 1.0 / (1j * omega * comp.value + 1e-30)
        elif comp.ctype == "transistor":
            # Simplified: treat as a transconductance current source (impedance 1/gm)
            return np.full_like(omega, 1.0 / (comp.value + 1e-30), dtype=complex)
        return np.ones_like(omega, dtype=complex)

    def ac_analysis(
        self,
        topo: CircuitTopology,
        f_start: float = 1e3,
        f_stop: float = 10e9,
    ) -> Dict[str, np.ndarray]:
        """
        Run AC analysis: compute H(f) = Vout/Vin.
        
        For a series-shunt ladder: H = Z_shunt / (Z_series + Z_shunt)
        """
        freqs = np.logspace(np.log10(f_start), np.log10(f_stop), self.freq_points)
        omega = 2 * np.pi * freqs

        resistors = topo.get_components_by_type("R")
        inductors = topo.get_components_by_type("L")
        capacitors = topo.get_components_by_type("C")
        transistors = topo.get_components_by_type("transistor")

        # Series impedance: sum of inductors in series path
        Z_series = np.zeros_like(omega, dtype=complex)
        for ind in inductors:
            Z_series += self._component_impedance(ind, omega)

        # Shunt impedance: capacitors || resistors in parallel at output
        Z_shunt = np.zeros_like(omega, dtype=complex)
        Y_shunt = np.zeros_like(omega, dtype=complex)
        for cap in capacitors:
            Z_cap = self._component_impedance(cap, omega)
            Y_shunt += 1.0 / (Z_cap + 1e-30)
        for res in resistors:
            Z_res = self._component_impedance(res, omega)
            Y_shunt += 1.0 / (Z_res + 1e-30)

        # Add transistor gain effect
        for tr in transistors:
            Y_shunt += tr.value  # gm adds to shunt admittance

        Z_shunt = np.where(np.abs(Y_shunt) > 1e-30, 1.0 / Y_shunt, 1e6 + 0j)

        # Transfer function H = Z_shunt / (Z_series + Z_shunt)
        denom = Z_series + Z_shunt
        H = Z_shunt / (denom + 1e-30)

        magnitude_db = 20 * np.log10(np.abs(H) + 1e-15)
        phase_deg = np.degrees(np.angle(H))

        return {
            "frequencies": freqs,
            "H": H,
            "magnitude_db": magnitude_db,
            "phase_deg": phase_deg,
        }

    def simulate(self, topo: CircuitTopology, **kwargs) -> Dict[str, np.ndarray]:
        """Convenience wrapper for ac_analysis."""
        return self.ac_analysis(topo, **kwargs)
