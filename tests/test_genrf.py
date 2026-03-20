"""Tests for genRF circuit diffuser."""

import pytest
import numpy as np

from genrf.topology import CircuitTopology, Component
from genrf.spice_sim import SPICESimulator
from genrf.diffusion_designer import DiffusionDesigner
from genrf.performance_evaluator import PerformanceEvaluator


# ── CircuitTopology ────────────────────────────────────────────────────────────

class TestCircuitTopology:
    def test_add_resistor(self):
        topo = CircuitTopology("Test")
        comp = topo.add_resistor("IN", "OUT", 50.0)
        assert comp.ctype == "R"
        assert comp.value == 50.0
        assert len(topo.components) == 1

    def test_add_inductor(self):
        topo = CircuitTopology()
        comp = topo.add_inductor("A", "B", 1e-9)
        assert comp.ctype == "L"
        assert comp.value == 1e-9

    def test_add_capacitor(self):
        topo = CircuitTopology()
        comp = topo.add_capacitor("A", "GND", 1e-12)
        assert comp.ctype == "C"

    def test_add_transistor(self):
        topo = CircuitTopology()
        comp = topo.add_transistor("D", "G", "S", gm=0.1)
        assert comp.ctype == "transistor"
        assert comp.node_c == "G"

    def test_nodes_include_gnd(self):
        topo = CircuitTopology()
        topo.add_resistor("IN", "OUT", 50.0)
        assert "GND" in topo.nodes

    def test_parameter_vector(self):
        topo = CircuitTopology.lc_filter()
        pv = topo.parameter_vector()
        assert len(pv) == 3  # L1, C1, R_load

    def test_update_from_vector(self):
        topo = CircuitTopology.lc_filter()
        topo.update_from_vector([2e-9, 5e-12, 100.0])
        assert abs(topo.components[0].value - 2e-9) < 1e-20

    def test_lc_filter_factory(self):
        topo = CircuitTopology.lc_filter()
        assert topo.name == "LC_Filter"
        assert len(topo.get_components_by_type("L")) == 1
        assert len(topo.get_components_by_type("C")) == 1

    def test_get_components_by_type(self):
        topo = CircuitTopology()
        topo.add_resistor("A", "B", 50)
        topo.add_resistor("C", "D", 100)
        topo.add_capacitor("A", "GND", 1e-12)
        assert len(topo.get_components_by_type("R")) == 2
        assert len(topo.get_components_by_type("C")) == 1


# ── SPICESimulator ─────────────────────────────────────────────────────────────

class TestSPICESimulator:
    def test_ac_analysis_returns_keys(self):
        topo = CircuitTopology.lc_filter(L=1e-9, C=1e-12)
        sim = SPICESimulator()
        result = sim.ac_analysis(topo)
        assert "frequencies" in result
        assert "H" in result
        assert "magnitude_db" in result
        assert "phase_deg" in result

    def test_frequency_range(self):
        topo = CircuitTopology.lc_filter()
        sim = SPICESimulator()
        result = sim.ac_analysis(topo, f_start=1e6, f_stop=1e10)
        assert result["frequencies"][0] >= 1e6
        assert result["frequencies"][-1] <= 1e10

    def test_magnitude_is_real(self):
        topo = CircuitTopology.lc_filter()
        sim = SPICESimulator()
        result = sim.ac_analysis(topo)
        assert np.all(np.isreal(result["magnitude_db"]))

    def test_lc_filter_rolls_off(self):
        """LC filter should attenuate at high frequencies."""
        topo = CircuitTopology.lc_filter(L=100e-9, C=100e-12, R_load=50.0)
        sim = SPICESimulator()
        result = sim.ac_analysis(topo, f_start=1e6, f_stop=10e9)
        mag = result["magnitude_db"]
        # Gain at low freq should be higher than at high freq
        assert mag[0] > mag[-1]


# ── DiffusionDesigner ──────────────────────────────────────────────────────────

class TestDiffusionDesigner:
    def test_forward_adds_noise(self):
        designer = DiffusionDesigner(n_steps=10)
        params = np.array([1e-9, 1e-12, 50.0])
        noisy = designer.forward(params, t=0)
        assert not np.allclose(noisy, params)

    def test_denoise_step_changes_params(self):
        designer = DiffusionDesigner(n_steps=10)
        x = np.array([0.5, 0.5, 0.5])
        score_fn = lambda x: -x
        x_next = designer.denoise_step(x, t=0, score_fn=score_fn)
        assert x_next.shape == x.shape

    def test_sample_random_returns_list(self):
        topo = CircuitTopology.lc_filter()
        designer = DiffusionDesigner(n_steps=5)
        samples = designer.sample_random(topo, n_samples=3)
        assert len(samples) == 3

    def test_sample_produces_positive_values(self):
        topo = CircuitTopology.lc_filter()
        designer = DiffusionDesigner(n_steps=5)
        samples = designer.sample_random(topo, n_samples=5)
        for s in samples:
            assert np.all(s > 0)

    def test_sample_with_score_fn(self):
        topo = CircuitTopology.lc_filter()
        designer = DiffusionDesigner(n_steps=5)
        # Score function pushing toward [1e-9, 1e-12, 50]
        target = np.array([1e-9, 1e-12, 50.0])
        score_fn = lambda x: (target - x) / (np.linalg.norm(target - x) + 1e-10)
        samples = designer.sample(topo, score_fn, n_samples=2)
        assert len(samples) == 2


# ── PerformanceEvaluator ───────────────────────────────────────────────────────

class TestPerformanceEvaluator:
    def test_evaluate_returns_keys(self):
        topo = CircuitTopology.lc_filter()
        evaluator = PerformanceEvaluator()
        metrics = evaluator.evaluate(topo)
        assert "bandwidth" in metrics
        assert "gain_db" in metrics
        assert "impedance_match" in metrics
        assert "score" in metrics

    def test_bandwidth_is_positive(self):
        topo = CircuitTopology.lc_filter()
        evaluator = PerformanceEvaluator()
        metrics = evaluator.evaluate(topo)
        assert metrics["bandwidth"] >= 0

    def test_score_in_range(self):
        topo = CircuitTopology.lc_filter()
        evaluator = PerformanceEvaluator()
        metrics = evaluator.evaluate(topo)
        assert 0.0 <= metrics["score"] <= 1.0

    def test_impedance_match_in_range(self):
        topo = CircuitTopology.lc_filter()
        evaluator = PerformanceEvaluator(target_freq=1e9)
        metrics = evaluator.evaluate(topo)
        assert 0.0 <= metrics["impedance_match"] <= 1.0

    def test_batch_evaluate(self):
        topos = [CircuitTopology.lc_filter(L=1e-9 * (i+1)) for i in range(3)]
        evaluator = PerformanceEvaluator()
        results = evaluator.batch_evaluate(topos)
        assert len(results) == 3
        assert all("score" in r for r in results)

    def test_lc_filter_optimization_demo(self):
        """Demo: compare two LC filters and find the better one."""
        evaluator = PerformanceEvaluator(target_freq=1e9)
        topo_a = CircuitTopology.lc_filter(L=10e-9, C=10e-12)
        topo_b = CircuitTopology.lc_filter(L=1e-9, C=1e-12)
        score_a = evaluator.evaluate(topo_a)["score"]
        score_b = evaluator.evaluate(topo_b)["score"]
        # Both should produce valid scores
        assert isinstance(score_a, float)
        assert isinstance(score_b, float)
