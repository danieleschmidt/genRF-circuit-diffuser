#!/usr/bin/env python3
"""
Generation 1 Validation Tests

Comprehensive test suite to validate Generation 1 (MAKE IT WORK) functionality.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add the current directory to Python path for imports  
sys.path.insert(0, str(Path(__file__).parent))

def activate_venv():
    """Activate the virtual environment if it exists."""
    venv_path = Path(__file__).parent / "genrf_env"
    if venv_path.exists():
        site_packages = venv_path / "lib" / "python3.12" / "site-packages"
        if site_packages.exists():
            sys.path.insert(0, str(site_packages))

activate_venv()

def test_basic_import():
    """Test 1: Basic package import functionality."""
    print("Test 1: Basic Package Import")
    try:
        import genrf
        assert genrf.__version__ == "0.1.0"
        print("✅ Package import successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_design_spec_creation():
    """Test 2: Design specification creation and validation."""
    print("\nTest 2: Design Specification Creation")
    try:
        from genrf import DesignSpec
        
        # Test LNA spec
        lna_spec = DesignSpec(
            circuit_type="LNA",
            frequency=5.8e9,
            name="Test_LNA_5G8"
        )
        assert lna_spec.circuit_type == "LNA"
        assert lna_spec.frequency == 5.8e9
        
        # Test Mixer spec
        mixer_spec = DesignSpec(
            circuit_type="Mixer", 
            frequency=28e9,
            name="Test_Mixer_28G"
        )
        assert mixer_spec.circuit_type == "Mixer"
        assert mixer_spec.frequency == 28e9
        
        # Test VCO spec
        vco_spec = DesignSpec(
            circuit_type="VCO",
            frequency=10e9,
            name="Test_VCO_10G"
        )
        assert vco_spec.circuit_type == "VCO"
        
        print("✅ Design specification creation successful")
        return True
    except Exception as e:
        print(f"❌ Design spec creation failed: {e}")
        return False

def test_circuit_diffuser_initialization():
    """Test 3: CircuitDiffuser initialization."""
    print("\nTest 3: CircuitDiffuser Initialization")
    try:
        from genrf import CircuitDiffuser
        
        diffuser = CircuitDiffuser(
            device="cpu",
            verbose=False  # Reduce noise during testing
        )
        
        assert diffuser.device.type == "cpu"
        assert hasattr(diffuser, 'generate_simple')
        
        print("✅ CircuitDiffuser initialization successful")
        return True
    except Exception as e:
        print(f"❌ CircuitDiffuser initialization failed: {e}")
        return False

def test_lna_generation():
    """Test 4: LNA circuit generation."""
    print("\nTest 4: LNA Circuit Generation")
    try:
        from genrf import DesignSpec, CircuitDiffuser
        
        spec = DesignSpec(
            circuit_type="LNA",
            frequency=2.4e9,
            name="Test_LNA"
        )
        
        diffuser = CircuitDiffuser(device="cpu", verbose=False)
        result = diffuser.generate_simple(spec)
        
        # Validate result structure
        assert hasattr(result, 'netlist')
        assert hasattr(result, 'parameters') 
        assert hasattr(result, 'performance')
        assert hasattr(result, 'topology')
        
        # Validate content
        assert len(result.netlist) > 100  # Should have substantial netlist
        assert len(result.parameters) > 0  # Should have parameters
        assert result.topology == "cascode_lna"
        assert result.spice_valid == True
        
        # Validate performance metrics
        assert 'gain_db' in result.performance
        assert 'noise_figure_db' in result.performance
        assert 'power_w' in result.performance
        
        # Validate ranges (reasonable RF values)
        assert 5 < result.gain < 25  # Reasonable gain range
        assert 0.5 < result.nf < 3.0  # Reasonable NF range 
        assert 0.001 < result.power < 0.1  # Reasonable power range
        
        print(f"✅ LNA generation successful: {result.gain:.1f}dB gain, {result.nf:.2f}dB NF")
        return True
    except Exception as e:
        print(f"❌ LNA generation failed: {e}")
        return False

def test_mixer_generation():
    """Test 5: Mixer circuit generation."""
    print("\nTest 5: Mixer Circuit Generation")
    try:
        from genrf import DesignSpec, CircuitDiffuser
        
        spec = DesignSpec(
            circuit_type="Mixer",
            frequency=5.8e9,
            name="Test_Mixer"
        )
        
        diffuser = CircuitDiffuser(device="cpu", verbose=False)
        result = diffuser.generate_simple(spec)
        
        # Validate mixer-specific attributes
        assert result.topology == "gilbert_mixer"
        assert 'conversion_gain_db' in result.performance
        assert 'iip3_dbm' in result.performance
        
        # Check mixer-specific parameters
        assert 'rf_transistor_width' in result.parameters
        assert 'lo_transistor_width' in result.parameters
        
        print(f"✅ Mixer generation successful: {result.topology}")
        return True
    except Exception as e:
        print(f"❌ Mixer generation failed: {e}")
        return False

def test_vco_generation():
    """Test 6: VCO circuit generation.""" 
    print("\nTest 6: VCO Circuit Generation")
    try:
        from genrf import DesignSpec, CircuitDiffuser
        
        spec = DesignSpec(
            circuit_type="VCO",
            frequency=10e9,
            name="Test_VCO"
        )
        
        diffuser = CircuitDiffuser(device="cpu", verbose=False)
        result = diffuser.generate_simple(spec)
        
        # Validate VCO-specific attributes
        assert result.topology == "lc_vco"
        assert 'phase_noise_dbc_hz' in result.performance
        assert 'tuning_range_percent' in result.performance
        
        # Check VCO-specific parameters
        assert 'tail_current' in result.parameters
        assert 'tank_inductance' in result.parameters
        
        print(f"✅ VCO generation successful: {result.topology}")
        return True
    except Exception as e:
        print(f"❌ VCO generation failed: {e}")
        return False

def test_export_functionality():
    """Test 7: Circuit export functionality."""
    print("\nTest 7: Export Functionality")
    try:
        from genrf import DesignSpec, CircuitDiffuser
        
        spec = DesignSpec(circuit_type="LNA", frequency=2.4e9)
        diffuser = CircuitDiffuser(device="cpu", verbose=False)
        result = diffuser.generate_simple(spec)
        
        # Create test output directory
        output_dir = Path("gen1_validation_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Test netlist export
        netlist_file = output_dir / "test_lna.spice"
        with open(netlist_file, 'w') as f:
            f.write(result.netlist)
        assert netlist_file.exists()
        assert netlist_file.stat().st_size > 0
        
        # Test JSON export
        json_file = output_dir / "test_lna.json"
        export_data = {
            "parameters": result.parameters,
            "performance": result.performance,
            "topology": result.topology
        }
        with open(json_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        assert json_file.exists()
        
        # Validate exported JSON
        with open(json_file, 'r') as f:
            loaded_data = json.load(f)
        assert 'parameters' in loaded_data
        assert 'performance' in loaded_data
        
        print("✅ Export functionality successful")
        return True
    except Exception as e:
        print(f"❌ Export functionality failed: {e}")
        return False

def test_multiple_circuits():
    """Test 8: Multiple circuit generation."""
    print("\nTest 8: Multiple Circuit Generation")
    try:
        from genrf import DesignSpec, CircuitDiffuser
        
        diffuser = CircuitDiffuser(device="cpu", verbose=False)
        results = []
        
        # Generate multiple circuit types
        circuit_specs = [
            ("LNA", 2.4e9),
            ("Mixer", 5.8e9), 
            ("VCO", 10e9),
            ("PA", 28e9)  # Test generic circuit
        ]
        
        for circuit_type, freq in circuit_specs:
            spec = DesignSpec(circuit_type=circuit_type, frequency=freq)
            result = diffuser.generate_simple(spec)
            results.append((circuit_type, result))
            
            # Validate each result
            assert len(result.netlist) > 50
            assert len(result.parameters) > 0
            assert result.spice_valid
        
        # Ensure we got different topologies
        topologies = [result.topology for _, result in results]
        assert len(set(topologies)) > 1  # Should have at least 2 different topologies
        
        print(f"✅ Generated {len(results)} different circuits successfully")
        return True
    except Exception as e:
        print(f"❌ Multiple circuit generation failed: {e}")
        return False

def test_performance_metrics():
    """Test 9: Performance metrics validation."""
    print("\nTest 9: Performance Metrics Validation")
    try:
        from genrf import DesignSpec, CircuitDiffuser
        
        diffuser = CircuitDiffuser(device="cpu", verbose=False)
        
        # Generate LNA and check metrics
        lna_spec = DesignSpec(circuit_type="LNA", frequency=2.4e9)
        lna_result = diffuser.generate_simple(lna_spec)
        
        # Validate LNA metrics
        assert lna_result.gain > 0  # Should have positive gain
        assert lna_result.nf > 0    # Should have positive NF
        assert lna_result.power > 0 # Should consume power
        
        # Check property access
        assert hasattr(lna_result, 'gain')
        assert hasattr(lna_result, 'nf')
        assert hasattr(lna_result, 'power')
        
        # Generate VCO and check different metrics
        vco_spec = DesignSpec(circuit_type="VCO", frequency=10e9)
        vco_result = diffuser.generate_simple(vco_spec)
        
        assert 'phase_noise_dbc_hz' in vco_result.performance
        assert vco_result.performance['phase_noise_dbc_hz'] < -50  # Should be negative
        
        print("✅ Performance metrics validation successful")
        return True
    except Exception as e:
        print(f"❌ Performance metrics validation failed: {e}")
        return False

def test_generation_speed():
    """Test 10: Generation speed performance."""
    print("\nTest 10: Generation Speed Performance")
    try:
        from genrf import DesignSpec, CircuitDiffuser
        
        diffuser = CircuitDiffuser(device="cpu", verbose=False)
        spec = DesignSpec(circuit_type="LNA", frequency=2.4e9)
        
        # Measure generation time
        start_time = time.time()
        result = diffuser.generate_simple(spec)
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # Should be very fast for Generation 1
        assert generation_time < 1.0  # Should complete in under 1 second
        
        # Test multiple generations
        total_start = time.time()
        for i in range(5):
            result = diffuser.generate_simple(spec)
        total_time = time.time() - total_start
        
        avg_time = total_time / 5
        assert avg_time < 0.5  # Average should be under 0.5 seconds
        
        print(f"✅ Generation speed test passed: {avg_time:.3f}s average")
        return True
    except Exception as e:
        print(f"❌ Generation speed test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🧪 GenRF Generation 1 Validation Test Suite")
    print("=" * 55)
    
    tests = [
        test_basic_import,
        test_design_spec_creation,
        test_circuit_diffuser_initialization,
        test_lna_generation,
        test_mixer_generation,
        test_vco_generation,
        test_export_functionality,
        test_multiple_circuits,
        test_performance_metrics,
        test_generation_speed
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All Generation 1 tests passed! Ready for Generation 2.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review and fix before continuing.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)