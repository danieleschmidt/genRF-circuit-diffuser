#!/usr/bin/env python3
"""
Lightweight GenRF Demo - Demonstrates core functionality without heavy dependencies
Part of Generation 2: MAKE IT ROBUST implementation
"""

import json
import time
import logging
import traceback
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Configure logging for robustness
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('genrf_robust.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RobustDesignSpec:
    """Robust design specification with comprehensive validation"""
    circuit_type: str
    frequency: float
    gain_min: float
    nf_max: float
    power_max: float
    technology: str
    
    def __post_init__(self):
        """Validate design specification parameters"""
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Comprehensive parameter validation"""
        try:
            # Validate circuit type
            valid_types = ["LNA", "Mixer", "VCO", "PA", "Filter"]
            if self.circuit_type not in valid_types:
                raise ValueError(f"Invalid circuit type: {self.circuit_type}. Must be one of {valid_types}")
            
            # Validate frequency range
            if not (0.1e9 <= self.frequency <= 300e9):
                raise ValueError(f"Frequency {self.frequency/1e9:.1f} GHz out of valid range (0.1-300 GHz)")
            
            # Validate gain
            if not (0 <= self.gain_min <= 60):
                raise ValueError(f"Gain {self.gain_min} dB out of valid range (0-60 dB)")
            
            # Validate noise figure
            if not (0.1 <= self.nf_max <= 20):
                raise ValueError(f"Noise figure {self.nf_max} dB out of valid range (0.1-20 dB)")
            
            # Validate power
            if not (1e-6 <= self.power_max <= 1.0):
                raise ValueError(f"Power {self.power_max*1000:.1f} mW out of valid range (0.001-1000 mW)")
            
            logger.info(f"Design spec validation passed for {self.circuit_type} at {self.frequency/1e9:.1f} GHz")
            
        except Exception as e:
            logger.error(f"Design spec validation failed: {e}")
            raise

@dataclass
class RobustCircuitResult:
    """Robust circuit result with comprehensive metadata"""
    design_spec: RobustDesignSpec
    topology: str
    components: Dict[str, Any]
    performance: Dict[str, float]
    generation_time: float
    validation_status: str
    security_hash: str
    
    def export_json(self, filepath: str) -> None:
        """Export result to JSON with error handling"""
        try:
            data = asdict(self)
            # Convert design_spec to dict for JSON serialization
            data['design_spec'] = asdict(self.design_spec)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Circuit result exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise

class RobustCircuitGenerator:
    """Robust circuit generator with comprehensive error handling and security"""
    
    def __init__(self, security_enabled: bool = True):
        self.security_enabled = security_enabled
        self.generation_count = 0
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        logger.info("RobustCircuitGenerator initialized with security enabled")
    
    def _generate_security_hash(self, spec: RobustDesignSpec) -> str:
        """Generate security hash for circuit validation"""
        import hashlib
        spec_str = f"{spec.circuit_type}_{spec.frequency}_{spec.gain_min}_{spec.nf_max}_{spec.power_max}_{spec.technology}"
        return hashlib.sha256(spec_str.encode()).hexdigest()[:16]
    
    def _validate_security(self, spec: RobustDesignSpec) -> bool:
        """Comprehensive security validation"""
        try:
            # Check for malicious parameters
            if any(isinstance(getattr(spec, attr), str) and len(getattr(spec, attr)) > 100 
                   for attr in ['circuit_type', 'technology']):
                raise SecurityError("Parameter length exceeds security limits")
            
            # Check for injection patterns
            dangerous_patterns = ['<script>', 'DROP TABLE', 'SELECT *', '../../']
            for attr in ['circuit_type', 'technology']:
                value = getattr(spec, attr)
                if isinstance(value, str) and any(pattern in value.upper() for pattern in dangerous_patterns):
                    raise SecurityError(f"Dangerous pattern detected in {attr}")
            
            logger.info("Security validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return False
    
    def _simulate_robust_generation(self, spec: RobustDesignSpec) -> Dict[str, Any]:
        """Simulate robust circuit generation with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Generation attempt {attempt + 1}/{max_retries}")
                
                # Simulate generation process with potential failures
                if attempt == 0 and spec.frequency > 100e9:
                    raise GenerationError("High frequency generation requires specialized models")
                
                # Simulate successful generation
                topology = self._select_optimal_topology(spec)
                components = self._generate_components(spec, topology)
                performance = self._simulate_performance(spec, components)
                
                logger.info(f"Generation successful on attempt {attempt + 1}")
                return {
                    'topology': topology,
                    'components': components,
                    'performance': performance
                }
                
            except GenerationError as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                
        raise GenerationError("All generation attempts failed")
    
    def _select_optimal_topology(self, spec: RobustDesignSpec) -> str:
        """Select optimal topology based on specification"""
        topology_map = {
            "LNA": {
                (0.1e9, 6e9): "Common Source",
                (6e9, 40e9): "Cascode",
                (40e9, 300e9): "Common Gate"
            },
            "Mixer": {
                (0.1e9, 20e9): "Gilbert Cell",
                (20e9, 100e9): "Passive Mixer",
                (100e9, 300e9): "Sub-harmonic Mixer"
            },
            "VCO": {
                (0.1e9, 10e9): "LC Oscillator",
                (10e9, 50e9): "Ring Oscillator",
                (50e9, 300e9): "Push-push Oscillator"
            },
            "PA": {
                (0.1e9, 6e9): "Class AB",
                (6e9, 40e9): "Class F",
                (40e9, 300e9): "Class E"
            },
            "Filter": {
                (0.1e9, 20e9): "Butterworth",
                (20e9, 100e9): "Chebyshev",
                (100e9, 300e9): "Transmission Line"
            }
        }
        
        freq = spec.frequency
        for freq_range, topology in topology_map[spec.circuit_type].items():
            if freq_range[0] <= freq <= freq_range[1]:
                return topology
        
        return "Generic Topology"
    
    def _generate_components(self, spec: RobustDesignSpec, topology: str) -> Dict[str, Any]:
        """Generate component values with robust algorithms"""
        components = {}
        
        # Technology-dependent parameters
        tech_params = {
            "TSMC65nm": {"Vdd": 1.2, "ft": 200e9, "min_L": 65e-9},
            "TSMC28nm": {"Vdd": 1.0, "ft": 300e9, "min_L": 28e-9},
            "GaN": {"Vdd": 28.0, "ft": 100e9, "min_L": 150e-9},
            "SiGe": {"Vdd": 3.3, "ft": 500e9, "min_L": 130e-9}
        }
        
        tech = tech_params.get(spec.technology, tech_params["TSMC65nm"])
        
        # Generate robust component values
        if spec.circuit_type == "LNA":
            components = {
                "input_transistor": {
                    "width": max(10e-6, spec.frequency / 1e9 * 2e-6),
                    "length": max(tech["min_L"], tech["min_L"] * 2),
                    "fingers": min(16, max(1, int(spec.gain_min / 5)))
                },
                "load_inductor": {
                    "value": 1 / (2 * 3.14159 * spec.frequency) * 1e-9,
                    "q_factor": max(10, 50 - spec.frequency / 1e9)
                },
                "source_degeneration": {
                    "inductor": 0.1e-9,
                    "resistor": 50.0 if spec.frequency < 10e9 else 25.0
                }
            }
        
        # Add robustness parameters
        components["robustness"] = {
            "process_corners": ["TT", "FF", "SS", "FS", "SF"],
            "temperature_range": [-40, 125],
            "supply_tolerance": 0.1,
            "yield_target": 0.95
        }
        
        return components
    
    def _simulate_performance(self, spec: RobustDesignSpec, components: Dict[str, Any]) -> Dict[str, float]:
        """Simulate circuit performance with statistical modeling"""
        import random
        random.seed(42)  # Reproducible results
        
        # Base performance calculations
        base_gain = spec.gain_min + random.uniform(0, 5)
        base_nf = spec.nf_max * (0.7 + random.uniform(0, 0.2))
        base_power = spec.power_max * (0.6 + random.uniform(0, 0.3))
        
        # Technology-dependent improvements
        tech_factor = {
            "TSMC65nm": 1.0,
            "TSMC28nm": 1.15,
            "GaN": 1.3,
            "SiGe": 1.25
        }.get(spec.technology, 1.0)
        
        # Frequency-dependent adjustments
        freq_factor = max(0.8, 1.2 - spec.frequency / 100e9)
        
        performance = {
            "gain_db": base_gain * tech_factor * freq_factor,
            "noise_figure_db": base_nf / tech_factor,
            "power_consumption_w": base_power,
            "input_return_loss_db": -15 - random.uniform(0, 10),
            "output_return_loss_db": -12 - random.uniform(0, 8),
            "stability_factor": 1.2 + random.uniform(0, 0.5),
            "bandwidth_hz": spec.frequency * 0.1,
            "yield_estimate": 0.90 + random.uniform(0, 0.08)
        }
        
        return performance
    
    def generate_circuit(self, spec: RobustDesignSpec) -> RobustCircuitResult:
        """Generate circuit with comprehensive robustness measures"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting robust circuit generation for {spec.circuit_type}")
            
            # Security validation
            if self.security_enabled and not self._validate_security(spec):
                raise SecurityError("Security validation failed")
            
            # Generate security hash
            security_hash = self._generate_security_hash(spec)
            
            # Robust generation with retry logic
            generation_result = self._simulate_robust_generation(spec)
            
            # Create robust result
            generation_time = time.time() - start_time
            self.generation_count += 1
            
            result = RobustCircuitResult(
                design_spec=spec,
                topology=generation_result['topology'],
                components=generation_result['components'],
                performance=generation_result['performance'],
                generation_time=generation_time,
                validation_status="PASSED",
                security_hash=security_hash
            )
            
            # Cache result for performance
            cache_file = self.cache_dir / f"circuit_{security_hash}.json"
            result.export_json(str(cache_file))
            
            logger.info(f"Circuit generation completed successfully in {generation_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Circuit generation failed: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Return error result for robustness
            return RobustCircuitResult(
                design_spec=spec,
                topology="ERROR",
                components={},
                performance={},
                generation_time=time.time() - start_time,
                validation_status=f"FAILED: {str(e)}",
                security_hash=""
            )

# Custom exceptions for better error handling
class GenRFRobustError(Exception):
    """Base exception for GenRF robust operations"""
    pass

class SecurityError(GenRFRobustError):
    """Security validation error"""
    pass

class GenerationError(GenRFRobustError):
    """Circuit generation error"""
    pass

def run_robust_demonstration():
    """Run comprehensive robust demonstration"""
    print("🛡️  GenRF Robust Circuit Generation Demonstration")
    print("=" * 60)
    
    try:
        # Initialize robust generator
        generator = RobustCircuitGenerator(security_enabled=True)
        
        # Test cases with different complexities
        test_cases = [
            {
                "name": "2.4 GHz LNA",
                "spec": RobustDesignSpec(
                    circuit_type="LNA",
                    frequency=2.4e9,
                    gain_min=15,
                    nf_max=1.5,
                    power_max=10e-3,
                    technology="TSMC65nm"
                )
            },
            {
                "name": "28 GHz Mixer",
                "spec": RobustDesignSpec(
                    circuit_type="Mixer",
                    frequency=28e9,
                    gain_min=10,
                    nf_max=8.0,
                    power_max=50e-3,
                    technology="TSMC28nm"
                )
            },
            {
                "name": "60 GHz PA",
                "spec": RobustDesignSpec(
                    circuit_type="PA",
                    frequency=60e9,
                    gain_min=20,
                    nf_max=5.0,
                    power_max=100e-3,
                    technology="GaN"
                )
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔧 Test Case {i}: {test_case['name']}")
            print("-" * 40)
            
            try:
                result = generator.generate_circuit(test_case['spec'])
                
                if result.validation_status == "PASSED":
                    print(f"✅ Generation successful!")
                    print(f"   Topology: {result.topology}")
                    print(f"   Gain: {result.performance.get('gain_db', 0):.1f} dB")
                    print(f"   Noise Figure: {result.performance.get('noise_figure_db', 0):.2f} dB")
                    print(f"   Power: {result.performance.get('power_consumption_w', 0)*1000:.1f} mW")
                    print(f"   Generation Time: {result.generation_time:.3f}s")
                    print(f"   Security Hash: {result.security_hash}")
                else:
                    print(f"❌ Generation failed: {result.validation_status}")
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ Test case failed: {e}")
        
        # Generate summary report
        print(f"\n📊 Robust Generation Summary")
        print("=" * 60)
        print(f"Total test cases: {len(test_cases)}")
        print(f"Successful generations: {sum(1 for r in results if r.validation_status == 'PASSED')}")
        print(f"Average generation time: {sum(r.generation_time for r in results) / len(results):.3f}s")
        print(f"Total circuits in cache: {generator.generation_count}")
        
        # Test error handling and security
        print(f"\n🔒 Security and Error Handling Tests")
        print("-" * 40)
        
        # Test invalid parameters
        try:
            invalid_spec = RobustDesignSpec(
                circuit_type="INVALID",
                frequency=2.4e9,
                gain_min=15,
                nf_max=1.5,
                power_max=10e-3,
                technology="TSMC65nm"
            )
        except ValueError as e:
            print(f"✅ Invalid parameter detection working: {e}")
        
        # Test security validation
        try:
            malicious_spec = RobustDesignSpec(
                circuit_type="LNA<script>alert('hack')</script>",
                frequency=2.4e9,
                gain_min=15,
                nf_max=1.5,
                power_max=10e-3,
                technology="TSMC65nm"
            )
            generator.generate_circuit(malicious_spec)
        except (SecurityError, ValueError) as e:
            print(f"✅ Security validation working: {e}")
        
        print(f"\n🎉 Robust demonstration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        logger.error(f"Demonstration failed: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = run_robust_demonstration()
    exit(0 if success else 1)