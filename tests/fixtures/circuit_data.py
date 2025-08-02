"""
Test fixtures for circuit data and specifications.
"""
import pytest
from typing import Dict, Any, List


@pytest.fixture
def lna_spec() -> Dict[str, Any]:
    """Sample LNA specification for testing."""
    return {
        "circuit_type": "LNA",
        "frequency": 2.4e9,  # 2.4 GHz
        "gain_min": 15,      # 15 dB minimum
        "nf_max": 1.5,       # 1.5 dB max noise figure
        "power_max": 10e-3,  # 10 mW max power
        "technology": "TSMC65nm",
        "supply_voltage": 1.2,
        "input_impedance": 50,
        "output_impedance": 50
    }


@pytest.fixture
def mixer_spec() -> Dict[str, Any]:
    """Sample mixer specification for testing."""
    return {
        "circuit_type": "Mixer",
        "rf_frequency": 2.4e9,
        "lo_frequency": 2.2e9,
        "if_frequency": 200e6,
        "conversion_gain_min": 10,
        "iip3_min": 5,
        "technology": "TSMC65nm",
        "supply_voltage": 1.2,
        "power_max": 20e-3
    }


@pytest.fixture
def vco_spec() -> Dict[str, Any]:
    """Sample VCO specification for testing."""
    return {
        "circuit_type": "VCO",
        "center_frequency": 5.8e9,
        "tuning_range": 500e6,
        "phase_noise_1mhz": -110,  # dBc/Hz @ 1MHz offset
        "power_max": 15e-3,
        "technology": "TSMC65nm",
        "supply_voltage": 1.2,
        "output_impedance": 50
    }


@pytest.fixture
def pa_spec() -> Dict[str, Any]:
    """Sample power amplifier specification for testing."""
    return {
        "circuit_type": "PA",
        "frequency": 2.4e9,
        "output_power": 1.0,  # 1W
        "pae_min": 35,        # 35% minimum PAE
        "gain_min": 20,       # 20 dB minimum
        "technology": "GF22nm",
        "supply_voltage": 3.3,
        "load_impedance": 50
    }


@pytest.fixture
def filter_spec() -> Dict[str, Any]:
    """Sample filter specification for testing."""
    return {
        "circuit_type": "Filter",
        "filter_type": "lowpass",
        "cutoff_frequency": 1e9,
        "insertion_loss_max": 1.0,  # 1 dB max
        "rejection_min": 40,         # 40 dB min stopband rejection
        "technology": "TSMC65nm",
        "impedance": 50
    }


@pytest.fixture
def circuit_specs() -> List[Dict[str, Any]]:
    """Collection of all circuit specifications for batch testing."""
    return [
        {
            "circuit_type": "LNA",
            "frequency": 2.4e9,
            "gain_min": 15,
            "nf_max": 1.5,
            "power_max": 10e-3,
            "technology": "TSMC65nm"
        },
        {
            "circuit_type": "Mixer",
            "rf_frequency": 5.8e9,
            "conversion_gain_min": 8,
            "technology": "GF22nm"
        },
        {
            "circuit_type": "VCO",
            "center_frequency": 10e9,
            "tuning_range": 1e9,
            "technology": "TSMC65nm"
        }
    ]


@pytest.fixture
def sample_netlist() -> str:
    """Sample SPICE netlist for testing."""
    return """
* Sample LNA Circuit
.param vdd=1.2
.param w_cs=50u
.param l_cs=65n
.param w_load=20u
.param l_load=65n

Vdd vdd 0 DC {vdd}
Vin in 0 AC 1 SIN(0 1m 2.4G)

* Input matching
Lin in in_matched 1n
Cin in_matched gnd 1p

* Common source amplifier
M1 out in_matched 0 0 nmos w={w_cs} l={l_cs}
M2 out vdd vdd vdd pmos w={w_load} l={l_load}

* Output matching
Cout out out_matched 1p
Lout out_matched 0 1n

.model nmos nmos (level=54 version=4.4)
.model pmos pmos (level=54 version=4.4)

.ac dec 100 1G 100G
.noise v(out_matched) Vin dec 100 1G 100G

.end
"""


@pytest.fixture
def technology_params() -> Dict[str, Dict[str, Any]]:
    """Technology-specific parameters for testing."""
    return {
        "TSMC65nm": {
            "min_channel_length": 65e-9,
            "min_channel_width": 120e-9,
            "max_vdd": 1.2,
            "substrate_resistivity": 10,
            "metal_layers": 9,
            "top_metal_thickness": 3.0e-6
        },
        "GF22nm": {
            "min_channel_length": 22e-9,
            "min_channel_width": 60e-9,
            "max_vdd": 0.8,
            "substrate_resistivity": 15,
            "metal_layers": 11,
            "top_metal_thickness": 2.0e-6
        },
        "SKY130": {
            "min_channel_length": 130e-9,
            "min_channel_width": 420e-9,
            "max_vdd": 1.8,
            "substrate_resistivity": 5,
            "metal_layers": 5,
            "top_metal_thickness": 1.26e-6
        }
    }


@pytest.fixture
def simulation_results() -> Dict[str, Any]:
    """Sample simulation results for testing."""
    return {
        "frequency": [1e9, 2e9, 5e9, 10e9],
        "s21_mag": [18.2, 18.5, 17.8, 16.2],  # dB
        "s21_phase": [-25, -45, -90, -135],    # degrees
        "s11_mag": [-15.2, -18.5, -12.8, -8.2],  # dB
        "noise_figure": [0.8, 0.9, 1.2, 1.8],    # dB
        "power_consumption": 8.5e-3,  # W
        "iip3": 12.5,  # dBm
        "convergence": True,
        "simulation_time": 45.2  # seconds
    }


@pytest.fixture
def mock_spice_engine():
    """Mock SPICE engine for testing without actual simulation."""
    class MockSpiceEngine:
        def __init__(self):
            self.netlist = None
            self.results = {
                "s21_mag": 18.0,
                "s11_mag": -15.0,
                "noise_figure": 1.0,
                "power": 10e-3,
                "converged": True
            }
        
        def simulate(self, netlist: str) -> Dict[str, Any]:
            self.netlist = netlist
            return self.results
        
        def set_results(self, results: Dict[str, Any]):
            self.results.update(results)
    
    return MockSpiceEngine()