#!/usr/bin/env python3
"""
Basic GenRF demonstration without heavy dependencies.

This demo shows the core concepts and architecture of GenRF
without requiring PyTorch or other ML libraries.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_design_spec():
    """Test design specification functionality."""
    print("🔧 Testing DesignSpec...")
    
    try:
        from genrf.core.design_spec import DesignSpec, CommonSpecs
        
        # Create a WiFi LNA specification
        spec = CommonSpecs.wifi_lna()
        print(f"✅ Created WiFi LNA spec: {spec.name}")
        print(f"   Frequency: {spec.frequency/1e9:.2f} GHz")
        print(f"   Gain: {spec.gain_min}-{spec.gain_max} dB")
        print(f"   NF: ≤{spec.nf_max} dB")
        print(f"   Power: ≤{spec.power_max*1000:.1f} mW")
        
        # Validate specification
        warnings = spec.check_feasibility()
        if warnings:
            print(f"⚠️  Feasibility warnings: {warnings}")
        else:
            print("✅ Specification is feasible")
            
        return True
        
    except Exception as e:
        print(f"❌ DesignSpec test failed: {e}")
        return False


def test_technology_file():
    """Test technology file functionality."""
    print("\n🏭 Testing TechnologyFile...")
    
    try:
        from genrf.core.technology import TechnologyFile
        
        # Create TSMC 65nm technology
        tech = TechnologyFile.tsmc65nm()
        print(f"✅ Created technology: {tech.name}")
        print(f"   Process: {tech.foundry} {tech.process_node}")
        print(f"   Supply: {tech.supply_voltage_nominal}V")
        print(f"   Devices: {list(tech.device_models.keys())}")
        
        # Test device validation
        valid_nmos = tech.validate_device_size('nmos', 10e-6, 100e-9)
        print(f"✅ NMOS validation (10μm/100nm): {'PASS' if valid_nmos else 'FAIL'}")
        
        return True
        
    except Exception as e:
        print(f"❌ TechnologyFile test failed: {e}")
        return False


def test_validation():
    """Test input validation."""
    print("\n🛡️  Testing Validation...")
    
    try:
        from genrf.core.validation import InputValidator
        from genrf.core.design_spec import CommonSpecs
        
        validator = InputValidator()
        spec = CommonSpecs.wifi_lna()
        
        # Validate design spec
        warnings = validator.validate_design_spec(spec)
        print(f"✅ Design spec validation: {len(warnings)} warnings")
        
        # Test parameter validation
        params = {
            'M1_w': 10e-6,
            'M1_l': 100e-9,
            'R1_r': 1000.0,
            'C1_c': 1e-12
        }
        
        param_warnings = validator.validate_parameters(params)
        print(f"✅ Parameter validation: {len(param_warnings)} warnings")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False


def test_security():
    """Test security features."""
    print("\n🔒 Testing Security...")
    
    try:
        from genrf.core.security import SecurityManager, SecurityConfig
        
        # Create security manager
        config = SecurityConfig()
        security = SecurityManager(config)
        
        print(f"✅ Security manager initialized")
        print(f"   Max file size: {config.max_file_size_mb}MB")
        print(f"   Rate limit: {config.max_requests_per_minute}/min")
        
        # Test rate limiting
        security.check_rate_limit("test_user")
        print("✅ Rate limit check passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        return False


def test_monitoring():
    """Test monitoring functionality."""
    print("\n📊 Testing Monitoring...")
    
    try:
        from genrf.core.monitoring import SystemMonitor
        
        monitor = SystemMonitor(monitoring_interval=60.0)
        
        # Get system metrics
        metrics = monitor.get_system_metrics()
        print(f"✅ System metrics collected")
        print(f"   CPU: {metrics.cpu_percent:.1f}%")
        print(f"   Memory: {metrics.memory_percent:.1f}% ({metrics.memory_used_gb:.1f}/{metrics.memory_total_gb:.1f} GB)")
        print(f"   Disk: {metrics.disk_percent:.1f}% ({metrics.disk_used_gb:.1f}/{metrics.disk_total_gb:.1f} GB)")
        
        # Get health status
        health = monitor.get_health_status()
        print(f"✅ Health status: {health.overall_status}")
        print(f"   Components: {health.components}")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")
        return False


def test_caching():
    """Test caching functionality."""
    print("\n💾 Testing Caching...")
    
    try:
        from genrf.core.cache import LRUCache, get_cache_stats
        
        # Test LRU cache
        cache = LRUCache(max_size=100, max_memory_mb=10.0)
        
        # Store some test data
        cache.put("test_key_1", "test_value_1")
        cache.put("test_key_2", {"data": "complex_value"})
        
        # Retrieve data
        value1 = cache.get("test_key_1")
        value2 = cache.get("test_key_2")
        
        print(f"✅ Cache operations successful")
        print(f"   Retrieved: {value1}, {value2}")
        
        # Get cache statistics
        stats = cache.get_stats()
        print(f"✅ Cache stats: {stats['size']} items, {stats['hit_rate_percent']:.1f}% hit rate")
        
        return True
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        return False


def main():
    """Run all basic tests."""
    print("🚀 GenRF Basic Demonstration")
    print("=" * 50)
    
    tests = [
        test_design_spec,
        test_technology_file,
        test_validation,
        test_security,
        test_monitoring,
        test_caching
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total} tests ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All basic tests passed! GenRF core functionality is working.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)