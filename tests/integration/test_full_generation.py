"""Integration test for full circuit generation pipeline."""

import pytest
import torch
import tempfile
from pathlib import Path

from genrf import CircuitDiffuser, DesignSpec, TechnologyFile


class TestFullGeneration:
    """Test complete circuit generation pipeline."""
    
    def test_lna_generation(self):
        """Test LNA circuit generation."""
        # Create design specification
        spec = DesignSpec(
            circuit_type="LNA",
            frequency=2.4e9,
            gain_min=15,
            nf_max=1.5,
            power_max=10e-3,
            technology="TSMC65nm"
        )
        
        # Initialize diffuser (without checkpoint for testing)
        diffuser = CircuitDiffuser(
            checkpoint=None,
            spice_engine="ngspice",
            device="cpu",
            verbose=False
        )
        
        # Generate circuit with minimal parameters for testing
        result = diffuser.generate(
            spec,
            n_candidates=2,
            optimization_steps=3,
            validate_spice=False  # Skip SPICE validation for unit test
        )
        
        # Check result
        assert result is not None
        assert result.topology.startswith("LNA_topology_")
        assert result.technology == "default"
        assert result.generation_time > 0
        assert len(result.parameters) > 0
        assert len(result.performance) > 0
        assert result.netlist is not None
        
        # Check performance metrics
        assert 'gain_db' in result.performance
        assert 'noise_figure_db' in result.performance
        assert 'power_w' in result.performance
    
    def test_mixer_generation(self):
        """Test mixer circuit generation."""
        spec = DesignSpec(
            circuit_type="Mixer",
            frequency=5.8e9,
            gain_min=10,
            nf_max=8.0,
            power_max=20e-3,
            technology="TSMC28nm"
        )
        
        diffuser = CircuitDiffuser(
            checkpoint=None,
            spice_engine="xyce",
            device="cpu",
            verbose=False
        )
        
        result = diffuser.generate(
            spec,
            n_candidates=2,
            optimization_steps=3,
            validate_spice=False
        )
        
        assert result.topology.startswith("Mixer_topology_")
        assert result.gain >= 0  # Should have some gain
        assert result.power > 0  # Should consume some power
    
    def test_export_functionality(self):
        """Test circuit export to various formats."""
        spec = DesignSpec(
            circuit_type="LNA",
            frequency=1e9,
            gain_min=10,
            nf_max=2.0,
            power_max=5e-3
        )
        
        diffuser = CircuitDiffuser(checkpoint=None, verbose=False)
        result = diffuser.generate(
            spec,
            n_candidates=1,
            optimization_steps=1,
            validate_spice=False
        )
        
        # Test export to temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test SKILL export
            skill_file = temp_path / "test.il"
            result.export_skill(skill_file)
            assert skill_file.exists()
            
            # Test Verilog-A export  
            verilog_file = temp_path / "test.va"
            result.export_verilog_a(verilog_file)
            assert verilog_file.exists()
    
    def test_performance_estimation(self):
        """Test performance estimation without SPICE."""
        spec = DesignSpec(
            circuit_type="LNA",
            frequency=2.4e9,
            gain_min=15,
            nf_max=1.5,
            power_max=10e-3
        )
        
        diffuser = CircuitDiffuser(checkpoint=None, verbose=False)
        
        # Create a simple topology for testing
        topology = {
            'name': 'test_topology',
            'type': 'LNA',
            'components': diffuser._get_default_components('LNA'),
            'connections': diffuser._get_default_connections('LNA'),
            'id': 0
        }
        
        parameters = {
            'M1_w': 50e-6,
            'M1_l': 100e-9,
            'R1_r': 1000,
            'C1_c': 1e-12,
            'C2_c': 1e-12,
            'L1_l': 1e-9
        }
        
        # Test performance estimation
        performance = diffuser._estimate_performance(topology, parameters, spec)
        
        assert 'gain_db' in performance
        assert 'noise_figure_db' in performance
        assert 'power_w' in performance
        assert 's11_db' in performance
        assert 'bandwidth_hz' in performance
        
        # Check reasonable values
        assert 0 <= performance['gain_db'] <= 50
        assert 0.5 <= performance['noise_figure_db'] <= 20
        assert 0 < performance['power_w'] <= 1
    
    def test_figure_of_merit_calculation(self):
        """Test FoM calculation."""
        spec = DesignSpec(
            circuit_type="LNA",
            frequency=2.4e9,
            gain_min=15,
            nf_max=1.5,
            power_max=10e-3
        )
        
        diffuser = CircuitDiffuser(checkpoint=None, verbose=False)
        
        # Test valid performance
        good_performance = {
            'gain_db': 20.0,
            'noise_figure_db': 1.2,
            'power_w': 8e-3
        }
        
        fom = diffuser._calculate_fom(good_performance, spec)
        assert fom > 0  # Should be positive for valid design
        
        # Test invalid performance (doesn't meet specs)
        bad_performance = {
            'gain_db': 10.0,  # Too low
            'noise_figure_db': 2.0,  # Too high
            'power_w': 15e-3  # Too high
        }
        
        fom = diffuser._calculate_fom(bad_performance, spec)
        assert fom == float('-inf')  # Should be invalid
    
    def test_netlist_generation(self):
        """Test SPICE netlist generation."""
        diffuser = CircuitDiffuser(checkpoint=None, verbose=False)
        
        topology = {
            'name': 'test_lna',
            'components': diffuser._get_default_components('LNA'),
            'connections': diffuser._get_default_connections('LNA')
        }
        
        parameters = {
            'M1_w': 50e-6,
            'M1_l': 100e-9,
            'R1_r': 1000,
            'C1_c': 1e-12,
            'C2_c': 1e-12,
            'L1_l': 1e-9
        }
        
        netlist = diffuser._generate_netlist(topology, parameters)
        
        assert isinstance(netlist, str)
        assert 'test_lna netlist' in netlist
        assert 'Generated by GenRF CircuitDiffuser' in netlist
        assert '.param vdd=1.2' in netlist
        assert '.end' in netlist
        
        # Check component lines are present
        assert 'M1' in netlist  # Transistor
        assert 'R1' in netlist  # Resistor
        assert 'C1' in netlist  # Capacitor
        assert 'L1' in netlist  # Inductor