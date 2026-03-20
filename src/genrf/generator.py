"""RF Circuit Generator: spec-driven synthesis using diffusion models."""

import numpy as np
from typing import List, Dict

from .topology import CircuitTopology
from .spice_simulator import SPICESimulator
from .diffusion_designer import DiffusionDesigner
from .performance_evaluator import PerformanceEvaluator


class RFCircuitGenerator:
    """High-level interface for spec-driven RF circuit synthesis."""

    def __init__(self):
        self.simulator = SPICESimulator()
        self.designer = DiffusionDesigner(n_params=40, n_steps=50)
        self.evaluator = PerformanceEvaluator(self.simulator)

    def generate(self, spec: Dict, n_candidates: int = 5) -> List[CircuitTopology]:
        """
        Generate candidate circuits conditioned on spec.
        Uses DiffusionDesigner to generate param vectors, converts to topologies,
        evaluates performance, and returns sorted by spec match.
        """
        param_vectors = self.designer.design(spec, n_circuits=n_candidates)

        topologies = []
        for vec in param_vectors:
            topo = CircuitTopology.from_vector(vec, n_nodes=4)
            # Ensure topology has a voltage source (required for validate())
            has_source = any(c["type"] == "V" for c in topo.get_components())
            if not has_source:
                topo.add_component("V", 0, 3, 1.0)
            topologies.append(topo)

        # Score each topology against the spec
        scored = []
        for topo in topologies:
            try:
                comparison = self.evaluator.compare(topo, spec)
                score = comparison["score"]
            except Exception:
                score = 0.0
            scored.append((score, topo))

        # Sort by score descending (best match first)
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored]

    def best_match(self, spec: Dict) -> CircuitTopology:
        """Return the single best matching circuit for the given spec."""
        candidates = self.generate(spec, n_candidates=5)
        return candidates[0]

    def design_lowpass(self, cutoff_hz: float, impedance_ohm: float = 50.0) -> CircuitTopology:
        """
        Design a lowpass filter with given cutoff frequency and characteristic impedance.
        Uses L-C ladder topology: series L into shunt C.
        """
        topo = CircuitTopology(n_nodes=4)

        # RC lowpass: R in series, C to ground
        # Cutoff: f_c = 1 / (2*pi*R*C) => C = 1 / (2*pi*f_c*R)
        R = impedance_ohm
        C = 1.0 / (2.0 * np.pi * cutoff_hz * R)

        # L-C ladder for better rolloff
        # Series inductor: Z_L = R at cutoff => L = R / (2*pi*f_c)
        L = R / (2.0 * np.pi * cutoff_hz)

        topo.add_component("V", 0, 3, 1.0)   # Voltage source input
        topo.add_component("R", 0, 1, R)       # Input resistor
        topo.add_component("L", 1, 2, L)       # Series inductor
        topo.add_component("C", 2, 3, C)       # Shunt capacitor to ground
        topo.add_component("R", 2, 3, R)       # Load resistor

        return topo

    def design_bandpass(self, center_hz: float, bandwidth_hz: float) -> CircuitTopology:
        """
        Design a bandpass filter centered at center_hz with given bandwidth.
        Uses series RLC topology.
        """
        topo = CircuitTopology(n_nodes=4)

        # Q factor
        Q = center_hz / max(bandwidth_hz, 1.0)
        omega_0 = 2.0 * np.pi * center_hz
        R = 50.0  # characteristic impedance

        # Series RLC: omega_0 = 1/sqrt(LC), Q = omega_0*L/R
        L = Q * R / omega_0
        C = 1.0 / (omega_0 ** 2 * L)

        topo.add_component("V", 0, 3, 1.0)   # Voltage source input
        topo.add_component("R", 0, 1, R)       # Source resistance
        topo.add_component("L", 1, 2, L)       # Series inductor
        topo.add_component("C", 2, 3, C)       # Series capacitor
        topo.add_component("R", 2, 3, R)       # Load resistor

        return topo
