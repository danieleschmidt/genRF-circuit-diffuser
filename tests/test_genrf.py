"""Tests for genRF-circuit-diffuser."""

import pytest
import numpy as np

from genrf.topology import CircuitTopology
from genrf.spice_simulator import SPICESimulator, FREQ_POINTS
from genrf.diffusion_designer import DiffusionDesigner
from genrf.performance_evaluator import PerformanceEvaluator
from genrf.generator import RFCircuitGenerator


# --- CircuitTopology Tests ---

def test_topology_add_component():
    topo = CircuitTopology(n_nodes=4)
    cid = topo.add_component("R", 0, 1, 1000.0)
    assert cid == 0
    cid2 = topo.add_component("C", 1, 2, 1e-9)
    assert cid2 == 1
    components = topo.get_components()
    assert len(components) == 2
    assert components[0]["type"] == "R"
    assert components[0]["value"] == 1000.0
    assert components[1]["type"] == "C"


def test_topology_netlist():
    topo = CircuitTopology(n_nodes=3)
    topo.add_component("R", 0, 1, 50.0)
    topo.add_component("C", 1, 2, 1e-12)
    topo.add_component("V", 0, 2, 1.0)
    netlist = topo.get_netlist()
    assert "R0 0 1" in netlist
    assert "C1 1 2" in netlist
    assert "V2 0 2" in netlist
    assert ".END" in netlist


def test_topology_validate():
    topo = CircuitTopology(n_nodes=4)
    # No components: invalid
    assert topo.validate() is False

    # Add components but no source: invalid
    topo.add_component("R", 0, 1, 100.0)
    topo.add_component("C", 1, 2, 1e-9)
    topo.add_component("L", 2, 3, 1e-6)
    assert topo.validate() is False

    # Add voltage source: still need all nodes connected
    topo.add_component("V", 0, 3, 1.0)
    assert topo.validate() is True


def test_topology_vector_roundtrip():
    topo = CircuitTopology(n_nodes=4)
    topo.add_component("R", 0, 1, 500.0)
    topo.add_component("C", 1, 2, 1e-9)
    topo.add_component("L", 2, 3, 1e-6)

    vec = topo.component_vector()
    assert isinstance(vec, np.ndarray)
    assert len(vec) == 40  # MAX_COMPONENTS * PARAMS_PER_COMPONENT

    # Reconstruct and check we get a valid topology
    topo2 = CircuitTopology.from_vector(vec, n_nodes=4)
    comps = topo2.get_components()
    assert len(comps) > 0


# --- SPICESimulator Tests ---

def test_impedance_resistor():
    sim = SPICESimulator()
    freqs = np.array([1e3, 1e6, 1e9])
    Z = sim.impedance_R(100.0, freqs)
    assert Z.dtype == np.complex128
    assert len(Z) == 3
    assert np.allclose(Z.real, 100.0)
    assert np.allclose(Z.imag, 0.0)


def test_impedance_capacitor():
    sim = SPICESimulator()
    freqs = np.array([1e6])
    C = 1e-9  # 1 nF
    Z = sim.impedance_C(C, freqs)
    assert Z.dtype == np.complex128
    # |Z_C| = 1 / (2*pi*f*C) ≈ 159 Ohm at 1MHz
    expected_mag = 1.0 / (2 * np.pi * 1e6 * 1e-9)
    assert abs(abs(Z[0]) - expected_mag) < 1.0


def test_impedance_inductor():
    sim = SPICESimulator()
    freqs = np.array([1e6])
    L = 1e-6  # 1 uH
    Z = sim.impedance_L(L, freqs)
    assert Z.dtype == np.complex128
    # |Z_L| = 2*pi*f*L ≈ 6.28 Ohm at 1MHz
    expected_mag = 2 * np.pi * 1e6 * 1e-6
    assert abs(abs(Z[0]) - expected_mag) < 0.01


def test_spice_simulate_basic():
    sim = SPICESimulator()
    topo = CircuitTopology(n_nodes=4)
    topo.add_component("R", 0, 1, 50.0)
    topo.add_component("C", 1, 2, 1e-9)
    topo.add_component("L", 2, 3, 1e-6)
    topo.add_component("V", 0, 3, 1.0)

    result = sim.simulate(topo)
    assert "frequencies" in result
    assert "impedance" in result
    assert "phase" in result
    assert len(result["frequencies"]) == 50
    assert len(result["impedance"]) == 50
    assert result["impedance"].dtype == np.complex128


def test_transfer_function_shape():
    sim = SPICESimulator()
    topo = CircuitTopology(n_nodes=4)
    topo.add_component("R", 0, 1, 50.0)
    topo.add_component("C", 1, 2, 1e-9)
    topo.add_component("V", 0, 3, 1.0)

    H = sim.transfer_function(topo)
    assert isinstance(H, np.ndarray)
    assert len(H) == len(FREQ_POINTS)
    assert H.dtype == np.complex128
    # All magnitudes should be finite
    assert np.all(np.isfinite(np.abs(H)))


# --- DiffusionDesigner Tests ---

def test_diffusion_noise_schedule():
    designer = DiffusionDesigner(n_params=20, n_steps=50, beta_start=0.0001, beta_end=0.02)
    betas = designer.noise_schedule
    assert len(betas) == 50
    assert betas[0] == pytest.approx(0.0001, rel=1e-5)
    assert betas[-1] == pytest.approx(0.02, rel=1e-5)
    # Monotonically increasing
    assert np.all(np.diff(betas) >= 0)


def test_diffusion_add_noise():
    np.random.seed(42)
    designer = DiffusionDesigner(n_params=20, n_steps=50)
    params = np.ones(20)
    noisy = designer.add_noise(params, t=0)
    assert noisy.shape == params.shape
    assert not np.allclose(noisy, params)  # Should have noise added

    # At t=n_steps-1 (max noise), result should differ significantly
    very_noisy = designer.add_noise(params, t=49)
    assert not np.allclose(very_noisy, params)


def test_diffusion_design_returns_vectors():
    np.random.seed(0)
    designer = DiffusionDesigner(n_params=20, n_steps=10)
    spec = {"bandwidth_hz": 1e6, "gain_db": 0.0, "topology": "lowpass"}
    results = designer.design(spec, n_circuits=3)
    assert len(results) == 3
    for vec in results:
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (20,)
        assert np.all(np.isfinite(vec))


# --- PerformanceEvaluator Tests ---

def _make_rc_lowpass(cutoff_hz=1e6):
    """Helper: simple RC lowpass filter."""
    topo = CircuitTopology(n_nodes=4)
    R = 50.0
    C = 1.0 / (2.0 * np.pi * cutoff_hz * R)
    topo.add_component("V", 0, 3, 1.0)
    topo.add_component("R", 0, 1, R)
    topo.add_component("C", 1, 3, C)
    topo.add_component("R", 1, 2, R)
    topo.add_component("R", 2, 3, R)
    return topo


def test_performance_evaluator_bandwidth():
    sim = SPICESimulator()
    evaluator = PerformanceEvaluator(sim)
    topo = _make_rc_lowpass(cutoff_hz=1e6)
    result = evaluator.evaluate(topo)
    assert "bandwidth_hz" in result
    assert result["bandwidth_hz"] >= 0.0
    assert np.isfinite(result["bandwidth_hz"])


def test_performance_evaluator_gain():
    sim = SPICESimulator()
    evaluator = PerformanceEvaluator(sim)
    topo = _make_rc_lowpass()
    result = evaluator.evaluate(topo)
    assert "gain_db" in result
    assert np.isfinite(result["gain_db"])
    assert "resonant_freq" in result
    assert "q_factor" in result
    assert "is_stable" in result
    assert isinstance(result["is_stable"], bool)


# --- RFCircuitGenerator Tests ---

def test_generator_lowpass_design():
    gen = RFCircuitGenerator()
    circuit = gen.design_lowpass(cutoff_hz=1e6, impedance_ohm=50.0)
    assert isinstance(circuit, CircuitTopology)
    components = circuit.get_components()
    assert len(components) > 0
    # Should have at least an R and a C
    types = {c["type"] for c in components}
    assert "R" in types or "C" in types
    # Netlist should be valid
    netlist = circuit.get_netlist()
    assert ".END" in netlist


def test_generator_best_match():
    np.random.seed(42)
    gen = RFCircuitGenerator()
    spec = {"bandwidth_hz": 1e6, "gain_db": 0.0, "topology": "lowpass"}
    circuit = gen.best_match(spec)
    assert isinstance(circuit, CircuitTopology)
    components = circuit.get_components()
    assert len(components) > 0
    netlist = circuit.get_netlist()
    assert "* genRF Circuit Netlist" in netlist
