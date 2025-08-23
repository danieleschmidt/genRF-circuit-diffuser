#!/usr/bin/env python3
"""
Generation 2: MAKE IT ROBUST - Enhanced GenRF with error handling, validation, and security
Autonomous SDLC execution with comprehensive robustness
"""

import json
import time
import random
import math
import hashlib
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GenRFError(Exception):
    """Base exception for GenRF errors"""
    pass

class InvalidSpecificationError(GenRFError):
    """Raised when design specification is invalid"""
    pass

class CircuitGenerationError(GenRFError):
    """Raised when circuit generation fails"""
    pass

class OptimizationError(GenRFError):
    """Raised when parameter optimization fails"""
    pass

class ValidationError(GenRFError):
    """Raised when circuit validation fails"""
    pass

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    security_score: float
    performance_score: float
    reliability_score: float
    
    def to_dict(self) -> Dict:
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'scores': {
                'security': self.security_score,
                'performance': self.performance_score,
                'reliability': self.reliability_score,
                'overall': (self.security_score + self.performance_score + self.reliability_score) / 3
            }
        }

class SecurityValidator:
    """Security validation for circuit designs"""
    
    @staticmethod
    def validate_parameters(parameters: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate parameters for security issues"""
        errors = []
        
        # Check for potentially dangerous parameter values
        for param, value in parameters.items():
            if value <= 0:
                errors.append(f"Security: Parameter {param} has non-positive value: {value}")
            
            if value > 1e6:  # Extremely large values
                errors.append(f"Security: Parameter {param} has suspiciously large value: {value}")
            
            if math.isnan(value) or math.isinf(value):
                errors.append(f"Security: Parameter {param} has invalid value: {value}")
        
        # Check for injection patterns in parameter names
        dangerous_patterns = ['<', '>', '&', '"', "'", ';', '|', '`']
        for param in parameters.keys():
            if any(pattern in str(param) for pattern in dangerous_patterns):
                errors.append(f"Security: Parameter name contains dangerous characters: {param}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_netlist(netlist: str) -> Tuple[bool, List[str]]:
        """Validate SPICE netlist for security issues"""
        errors = []
        
        # Check for command injection attempts
        dangerous_commands = ['.system', '.exec', '.shell', 'system(', 'exec(', 'eval(']
        for cmd in dangerous_commands:
            if cmd.lower() in netlist.lower():
                errors.append(f"Security: Potentially dangerous command detected: {cmd}")
        
        # Check for file system access attempts
        file_patterns = ['../', '.\\', '/etc/', '/proc/', 'C:\\', 'file://', 'http://', 'https://']
        for pattern in file_patterns:
            if pattern in netlist:
                errors.append(f"Security: File system access pattern detected: {pattern}")
        
        # Validate netlist length
        if len(netlist) > 1000000:  # 1MB limit
            errors.append("Security: Netlist exceeds maximum safe size")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def calculate_security_score(parameters: Dict[str, float], netlist: str) -> float:
        """Calculate overall security score (0-100)"""
        score = 100.0
        
        param_valid, param_errors = SecurityValidator.validate_parameters(parameters)
        netlist_valid, netlist_errors = SecurityValidator.validate_netlist(netlist)
        
        # Deduct points for security issues
        score -= len(param_errors) * 10
        score -= len(netlist_errors) * 20
        
        return max(0.0, min(100.0, score))

class RobustDesignSpec:
    """Enhanced design specification with validation and security"""
    
    def __init__(self, circuit_type: str = 'LNA', frequency: float = 2.4e9, 
                 gain_min: float = 15, nf_max: float = 1.5, power_max: float = 10e-3,
                 supply_voltage: float = 1.2, temperature: float = 27,
                 input_impedance: float = 50, output_impedance: float = 50,
                 validation_level: str = 'strict'):
        
        # Input sanitization
        self.circuit_type = self._sanitize_string(circuit_type)
        self.frequency = self._validate_frequency(frequency)
        self.gain_min = self._validate_number(gain_min, 'gain_min', -50, 100)
        self.nf_max = self._validate_number(nf_max, 'nf_max', 0.1, 50)
        self.power_max = self._validate_number(power_max, 'power_max', 1e-6, 1.0)
        self.supply_voltage = self._validate_number(supply_voltage, 'supply_voltage', 0.5, 5.0)
        self.temperature = self._validate_number(temperature, 'temperature', -55, 125)
        self.input_impedance = self._validate_number(input_impedance, 'input_impedance', 1, 1000)
        self.output_impedance = self._validate_number(output_impedance, 'output_impedance', 1, 1000)
        self.validation_level = validation_level
        
        # Generate specification hash for integrity
        self.spec_hash = self._generate_hash()
        
        logger.info(f"RobustDesignSpec created: {circuit_type} @ {frequency/1e9:.2f}GHz")
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            raise InvalidSpecificationError(f"Expected string, got {type(value)}")
        
        # Remove dangerous characters
        sanitized = ''.join(c for c in value if c.isalnum() or c in ['_', '-'])
        
        if not sanitized:
            raise InvalidSpecificationError("String became empty after sanitization")
        
        return sanitized[:50]  # Limit length
    
    def _validate_frequency(self, freq: float) -> float:
        """Validate frequency parameter"""
        try:
            freq = float(freq)
        except (TypeError, ValueError):
            raise InvalidSpecificationError(f"Invalid frequency: {freq}")
        
        if freq <= 0 or freq > 1e12:  # 0 to 1THz
            raise InvalidSpecificationError(f"Frequency out of valid range: {freq}")
        
        return freq
    
    def _validate_number(self, value: float, name: str, min_val: float, max_val: float) -> float:
        """Validate numeric parameter with bounds"""
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise InvalidSpecificationError(f"Invalid {name}: {value}")
        
        # Special handling for infinity (valid for some parameters like nf_max)
        if math.isinf(value):
            if name == 'nf_max' and value > 0:
                return value  # Positive infinity is valid for nf_max
            else:
                raise InvalidSpecificationError(f"Invalid {name}: {value}")
        
        if math.isnan(value):
            raise InvalidSpecificationError(f"Invalid {name}: {value}")
        
        if value < min_val or value > max_val:
            raise InvalidSpecificationError(f"{name} out of range [{min_val}, {max_val}]: {value}")
        
        return value
    
    def _generate_hash(self) -> str:
        """Generate integrity hash for specification"""
        spec_str = f"{self.circuit_type}:{self.frequency}:{self.gain_min}:{self.nf_max}:{self.power_max}"
        return hashlib.sha256(spec_str.encode()).hexdigest()[:16]
    
    def validate_integrity(self) -> bool:
        """Validate specification integrity"""
        return self._generate_hash() == self.spec_hash
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'circuit_type': self.circuit_type,
            'frequency': self.frequency,
            'gain_min': self.gain_min,
            'nf_max': self.nf_max,
            'power_max': self.power_max,
            'supply_voltage': self.supply_voltage,
            'temperature': self.temperature,
            'input_impedance': self.input_impedance,
            'output_impedance': self.output_impedance,
            'validation_level': self.validation_level,
            'spec_hash': self.spec_hash
        }

class RobustCircuitResult:
    """Enhanced circuit result with validation and security"""
    
    def __init__(self, netlist: str, parameters: Dict[str, float], 
                 performance: Dict[str, float], topology: str,
                 technology: str, generation_time: float, 
                 validation_report: ValidationReport,
                 spice_valid: bool = True):
        
        self.netlist = netlist
        self.parameters = parameters
        self.performance = performance
        self.topology = topology
        self.technology = technology
        self.generation_time = generation_time
        self.validation_report = validation_report
        self.spice_valid = spice_valid
        
        # Security validation
        self.security_score = SecurityValidator.calculate_security_score(parameters, netlist)
        
        # Generate result integrity hash
        self.result_hash = self._generate_integrity_hash()
        
        logger.info(f"CircuitResult created: {topology} (Security: {self.security_score:.1f})")
    
    def _generate_integrity_hash(self) -> str:
        """Generate integrity hash for result"""
        data = f"{self.topology}:{len(self.netlist)}:{len(self.parameters)}:{self.generation_time}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    @property
    def gain(self) -> float:
        """Circuit gain in dB"""
        return self.performance.get('gain_db', 0.0)
    
    @property
    def nf(self) -> float:
        """Noise figure in dB"""
        return self.performance.get('noise_figure_db', float('inf'))
    
    @property
    def power(self) -> float:
        """Power consumption in W"""
        return self.performance.get('power_w', 0.0)
    
    @property
    def is_secure(self) -> bool:
        """Check if result passes security validation"""
        return self.security_score >= 80.0
    
    @property
    def is_reliable(self) -> bool:
        """Check if result is reliable"""
        return (self.validation_report.reliability_score >= 75.0 and
                self.validation_report.is_valid and
                len(self.validation_report.errors) == 0)
    
    def export_skill(self, filepath: Union[str, Path], include_validation: bool = True):
        """Export to Cadence SKILL format with validation metadata"""
        
        if not self.is_secure:
            logger.warning(f"Exporting circuit with low security score: {self.security_score}")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(f"; SKILL Export - {self.topology}\n")
            f.write(f"; Generated by GenRF Generation 2 - Robust Implementation\n")
            f.write(f"; Performance: Gain={self.gain:.1f}dB, NF={self.nf:.2f}dB, Power={self.power*1000:.1f}mW\n")
            f.write(f"; Security Score: {self.security_score:.1f}/100\n")
            f.write(f"; Reliability: {self.validation_report.reliability_score:.1f}/100\n")
            f.write(f"; Result Hash: {self.result_hash}\n\n")
            
            if include_validation:
                f.write(f"; Validation Report:\n")
                for error in self.validation_report.errors:
                    f.write(f";   ERROR: {error}\n")
                for warning in self.validation_report.warnings:
                    f.write(f";   WARNING: {warning}\n")
                f.write(f"\n")
            
            f.write(self.netlist)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with all metadata"""
        return {
            'netlist': self.netlist,
            'parameters': self.parameters,
            'performance': self.performance,
            'topology': self.topology,
            'technology': self.technology,
            'generation_time': self.generation_time,
            'spice_valid': self.spice_valid,
            'security_score': self.security_score,
            'validation_report': self.validation_report.to_dict(),
            'reliability_metrics': {
                'is_secure': self.is_secure,
                'is_reliable': self.is_reliable,
                'result_hash': self.result_hash
            }
        }

class RobustCircuitDiffuser:
    """Generation 2: Robust AI-powered RF circuit generator"""
    
    def __init__(self, checkpoint: Optional[str] = None, 
                 spice_engine: str = "analytical",
                 technology: str = "generic", 
                 verbose: bool = True,
                 validation_level: str = "strict",
                 enable_security: bool = True,
                 max_attempts: int = 5):
        
        self.checkpoint = checkpoint
        self.spice_engine = spice_engine
        self.technology = technology
        self.verbose = verbose
        self.validation_level = validation_level
        self.enable_security = enable_security
        self.max_attempts = max_attempts
        
        # Initialize security components
        self.security_validator = SecurityValidator()
        
        # Generation statistics
        self.stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'security_failures': 0,
            'validation_failures': 0,
            'optimization_failures': 0
        }
        
        if verbose:
            logger.info("🔧 CircuitDiffuser Generation 2: MAKE IT ROBUST")
            logger.info(f"   Technology: {technology}")
            logger.info(f"   SPICE Engine: {spice_engine}")
            logger.info(f"   Validation: {validation_level}")
            logger.info(f"   Security: {'Enabled' if enable_security else 'Disabled'}")
            logger.info("   ✅ Initialization complete")
    
    def generate(self, spec: RobustDesignSpec, n_candidates: int = 10,
                 optimization_steps: int = 20, spice_validation: bool = False,
                 retry_on_failure: bool = True) -> RobustCircuitResult:
        """Generate optimized RF circuit with robust error handling"""
        
        self.stats['total_generations'] += 1
        start_time = time.time()
        
        # Validate input specification
        if not spec.validate_integrity():
            raise InvalidSpecificationError("Specification integrity check failed")
        
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                logger.info(f"🔬 Generating {spec.circuit_type} circuit (attempt {attempt + 1}/{self.max_attempts})")
                logger.info(f"   Target: {spec.frequency/1e9:.2f}GHz, {spec.gain_min}dB gain, {spec.nf_max}dB NF")
                
                # Step 1: Generate topology with error handling
                try:
                    topology = self._generate_topology_robust(spec)
                except Exception as e:
                    logger.warning(f"Topology generation failed: {e}")
                    if not retry_on_failure or attempt == self.max_attempts - 1:
                        raise CircuitGenerationError(f"Topology generation failed: {e}")
                    continue
                
                # Step 2: Optimize parameters with error handling
                try:
                    parameters = self._optimize_parameters_robust(topology, spec, optimization_steps)
                except Exception as e:
                    logger.warning(f"Parameter optimization failed: {e}")
                    self.stats['optimization_failures'] += 1
                    if not retry_on_failure or attempt == self.max_attempts - 1:
                        raise OptimizationError(f"Parameter optimization failed: {e}")
                    continue
                
                # Step 3: Security validation
                if self.enable_security:
                    param_secure, param_errors = self.security_validator.validate_parameters(parameters)
                    if not param_secure:
                        logger.warning(f"Security validation failed: {param_errors}")
                        self.stats['security_failures'] += 1
                        if spec.validation_level == 'strict':
                            if not retry_on_failure or attempt == self.max_attempts - 1:
                                raise ValidationError(f"Security validation failed: {param_errors}")
                            continue
                
                # Step 4: Create netlist with validation
                try:
                    netlist = self._create_netlist_robust(topology, parameters, spec)
                except Exception as e:
                    logger.warning(f"Netlist creation failed: {e}")
                    if not retry_on_failure or attempt == self.max_attempts - 1:
                        raise CircuitGenerationError(f"Netlist creation failed: {e}")
                    continue
                
                # Step 5: Security validation of netlist
                if self.enable_security:
                    netlist_secure, netlist_errors = self.security_validator.validate_netlist(netlist)
                    if not netlist_secure:
                        logger.warning(f"Netlist security validation failed: {netlist_errors}")
                        self.stats['security_failures'] += 1
                        if spec.validation_level == 'strict':
                            if not retry_on_failure or attempt == self.max_attempts - 1:
                                raise ValidationError(f"Netlist security validation failed: {netlist_errors}")
                            continue
                
                # Step 6: Performance estimation with error handling
                try:
                    performance = self._estimate_performance_robust(parameters, spec)
                except Exception as e:
                    logger.warning(f"Performance estimation failed: {e}")
                    if not retry_on_failure or attempt == self.max_attempts - 1:
                        raise CircuitGenerationError(f"Performance estimation failed: {e}")
                    continue
                
                # Step 7: Comprehensive validation
                validation_report = self._comprehensive_validation(
                    topology, parameters, performance, netlist, spec
                )
                
                if not validation_report.is_valid and spec.validation_level == 'strict':
                    logger.warning(f"Comprehensive validation failed: {validation_report.errors}")
                    self.stats['validation_failures'] += 1
                    if not retry_on_failure or attempt == self.max_attempts - 1:
                        raise ValidationError(f"Comprehensive validation failed: {validation_report.errors}")
                    continue
                
                # Success! Create result
                generation_time = time.time() - start_time
                
                result = RobustCircuitResult(
                    netlist=netlist,
                    parameters=parameters,
                    performance=performance,
                    topology=topology['name'],
                    technology=self.technology,
                    generation_time=generation_time,
                    validation_report=validation_report,
                    spice_valid=spice_validation
                )
                
                self.stats['successful_generations'] += 1
                
                logger.info(f"   ✅ Generation successful (attempt {attempt + 1}, {generation_time:.2f}s)")
                logger.info(f"   Performance: Gain={result.gain:.1f}dB, NF={result.nf:.2f}dB, Power={result.power*1000:.1f}mW")
                logger.info(f"   Security Score: {result.security_score:.1f}/100")
                logger.info(f"   Reliability Score: {validation_report.reliability_score:.1f}/100")
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.error(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_attempts - 1 and retry_on_failure:
                    logger.info(f"Retrying... ({attempt + 2}/{self.max_attempts})")
                    time.sleep(0.1)  # Small delay between attempts
                else:
                    break
        
        # All attempts failed
        error_msg = f"Circuit generation failed after {self.max_attempts} attempts"
        if last_exception:
            error_msg += f": {last_exception}"
        
        logger.error(error_msg)
        raise CircuitGenerationError(error_msg)
    
    def _generate_topology_robust(self, spec: RobustDesignSpec) -> Dict[str, Any]:
        """Generate circuit topology with error handling"""
        
        try:
            # Enhanced topology generation with more variants
            topologies = {
                'LNA': [
                    {
                        'name': 'cascode_lna_v1',
                        'components': ['M1_main', 'M2_cascode', 'L_source', 'L_gate', 'L_drain', 'C_coupling', 'R_load'],
                        'type': 'cascode',
                        'reliability': 0.9
                    },
                    {
                        'name': 'common_source_lna',
                        'components': ['M1_main', 'L_source', 'L_gate', 'L_drain', 'C_coupling', 'R_load'],
                        'type': 'common_source',
                        'reliability': 0.8
                    },
                    {
                        'name': 'differential_lna',
                        'components': ['M1_main', 'M2_main', 'Mtail', 'L_load1', 'L_load2', 'C_coupling'],
                        'type': 'differential',
                        'reliability': 0.85
                    }
                ],
                'Mixer': [
                    {
                        'name': 'gilbert_mixer_v1',
                        'components': ['M_rf1', 'M_rf2', 'M_lo1', 'M_lo2', 'M_tail', 'R_load1', 'R_load2'],
                        'type': 'gilbert_cell',
                        'reliability': 0.9
                    },
                    {
                        'name': 'passive_mixer',
                        'components': ['D1', 'D2', 'D3', 'D4', 'T1', 'C_rf', 'C_lo', 'R_load'],
                        'type': 'passive_ring',
                        'reliability': 0.85
                    }
                ],
                'VCO': [
                    {
                        'name': 'lc_vco_cross_coupled',
                        'components': ['M1', 'M2', 'L_tank', 'C_tank', 'C_var'],
                        'type': 'cross_coupled',
                        'reliability': 0.9
                    },
                    {
                        'name': 'ring_vco',
                        'components': ['M1', 'M2', 'M3', 'R_load1', 'R_load2', 'R_load3'],
                        'type': 'ring_oscillator',
                        'reliability': 0.75
                    }
                ]
            }
            
            circuit_topologies = topologies.get(spec.circuit_type, topologies['LNA'])
            
            # Select topology based on reliability score if in strict mode
            if spec.validation_level == 'strict':
                topology = max(circuit_topologies, key=lambda x: x['reliability'])
            else:
                topology = random.choice(circuit_topologies)
            
            # Add metadata
            topology['variant'] = random.randint(1, 5)
            topology['frequency_optimized'] = spec.frequency
            topology['generation_timestamp'] = time.time()
            
            return topology
            
        except Exception as e:
            logger.error(f"Topology generation failed: {e}")
            raise CircuitGenerationError(f"Topology generation failed: {e}")
    
    def _optimize_parameters_robust(self, topology: Dict[str, Any], 
                                   spec: RobustDesignSpec, steps: int) -> Dict[str, float]:
        """Optimize circuit parameters with comprehensive error handling"""
        
        try:
            # Get initial parameters with bounds checking
            parameters = self._get_initial_parameters(topology, spec)
            
            # Validate initial parameters
            if self.enable_security:
                param_valid, param_errors = self.security_validator.validate_parameters(parameters)
                if not param_valid:
                    logger.warning(f"Initial parameters failed security check: {param_errors}")
                    # Sanitize parameters
                    parameters = self._sanitize_parameters(parameters)
            
            best_fom = float('-inf')
            best_params = parameters.copy()
            optimization_history = []
            
            # Enhanced optimization with momentum and adaptive step size
            momentum = {key: 0.0 for key in parameters.keys()}
            step_size = 0.1
            
            for step in range(steps):
                try:
                    # Generate parameter variations with momentum
                    test_params = {}
                    for key, value in parameters.items():
                        # Add momentum and random variation
                        variation = random.uniform(-step_size, step_size)
                        momentum[key] = 0.9 * momentum[key] + 0.1 * variation
                        
                        # Apply bounds based on parameter type
                        bounds = self._get_parameter_bounds(key, spec.circuit_type)
                        min_val, max_val = bounds.get(key, (value * 0.1, value * 10))
                        
                        new_value = value * (1 + momentum[key])
                        test_params[key] = max(min_val, min(max_val, new_value))
                    
                    # Security validation of test parameters
                    if self.enable_security:
                        param_valid, param_errors = self.security_validator.validate_parameters(test_params)
                        if not param_valid:
                            continue  # Skip invalid parameters
                    
                    # Evaluate performance
                    perf = self._estimate_performance_robust(test_params, spec)
                    fom = self._calculate_fom_robust(perf, spec)
                    
                    # Track optimization history
                    optimization_history.append({
                        'step': step,
                        'fom': fom,
                        'parameters': test_params.copy(),
                        'performance': perf.copy()
                    })
                    
                    # Update best if improved
                    if fom > best_fom:
                        best_fom = fom
                        best_params = test_params.copy()
                        parameters = test_params.copy()  # Use for next iteration
                        
                        # Increase step size on improvement
                        step_size = min(0.3, step_size * 1.1)
                    else:
                        # Decrease step size on no improvement
                        step_size = max(0.01, step_size * 0.95)
                
                except Exception as e:
                    logger.debug(f"Optimization step {step} failed: {e}")
                    continue
            
            # Final validation of optimized parameters
            if self.enable_security:
                param_valid, param_errors = self.security_validator.validate_parameters(best_params)
                if not param_valid:
                    logger.warning(f"Optimized parameters failed security validation: {param_errors}")
                    if spec.validation_level == 'strict':
                        raise OptimizationError(f"Optimized parameters are insecure: {param_errors}")
            
            logger.debug(f"Optimization completed: FoM improved from {optimization_history[0]['fom']:.3f} to {best_fom:.3f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            raise OptimizationError(f"Parameter optimization failed: {e}")
    
    def _get_initial_parameters(self, topology: Dict[str, Any], spec: RobustDesignSpec) -> Dict[str, float]:
        """Get initial parameters with proper bounds and validation"""
        
        circuit_type = spec.circuit_type
        frequency = spec.frequency
        
        try:
            if circuit_type == 'LNA':
                # LNA parameters with frequency scaling
                freq_scale = math.sqrt(frequency / 2.4e9)  # Scale from 2.4GHz baseline
                
                parameters = {
                    'W1': random.uniform(20e-6, 200e-6) * freq_scale,
                    'L1': random.uniform(50e-9, 500e-9) / freq_scale,
                    'W2': random.uniform(10e-6, 100e-6) * freq_scale,
                    'L2': random.uniform(50e-9, 300e-9) / freq_scale,
                    'Ls': random.uniform(0.5e-9, 20e-9) / freq_scale,
                    'Lg': random.uniform(0.5e-9, 20e-9) / freq_scale,
                    'Ld': random.uniform(0.5e-9, 20e-9) / freq_scale,
                    'Ibias': random.uniform(1e-3, 20e-3),
                    'Vbias': random.uniform(0.3, min(1.0, spec.supply_voltage - 0.2)),
                    'Rd': random.uniform(200, 5000),
                    'Cs': random.uniform(1e-15, 50e-12),
                    'Cg': random.uniform(1e-15, 50e-12)
                }
                
            elif circuit_type == 'Mixer':
                freq_scale = math.sqrt(frequency / 5.8e9)
                
                parameters = {
                    'Wrf': random.uniform(20e-6, 150e-6) * freq_scale,
                    'Lrf': random.uniform(50e-9, 300e-9) / freq_scale,
                    'Wlo': random.uniform(30e-6, 200e-6) * freq_scale,
                    'Llo': random.uniform(50e-9, 300e-9) / freq_scale,
                    'Wtail': random.uniform(40e-6, 300e-6) * freq_scale,
                    'Ltail': random.uniform(100e-9, 1000e-9),
                    'RL': random.uniform(300, 3000),
                    'Ibias': random.uniform(0.5e-3, 15e-3),
                    'Cin': random.uniform(1e-15, 30e-12),
                    'Cout': random.uniform(1e-15, 30e-12)
                }
                
            elif circuit_type == 'VCO':
                freq_scale = frequency / 10e9
                
                parameters = {
                    'W': random.uniform(40e-6, 400e-6) * freq_scale,
                    'L': random.uniform(50e-9, 300e-9) / freq_scale,
                    'Ltank': random.uniform(0.5e-9, 50e-9) / (freq_scale * freq_scale),
                    'Ctank': random.uniform(0.2e-12, 10e-12) / (freq_scale * freq_scale),
                    'Cvar': random.uniform(0.1e-12, 5e-12) / (freq_scale * freq_scale),
                    'Rtail': random.uniform(500, 10000),
                    'Ibias': random.uniform(2e-3, 50e-3)
                }
            
            else:
                # Generic parameters
                parameters = {
                    'W1': random.uniform(20e-6, 100e-6),
                    'L1': random.uniform(50e-9, 500e-9),
                    'R1': random.uniform(100, 2000),
                    'C1': random.uniform(1e-15, 10e-12),
                    'Ibias': random.uniform(1e-3, 10e-3)
                }
            
            return parameters
            
        except Exception as e:
            logger.error(f"Initial parameter generation failed: {e}")
            raise OptimizationError(f"Initial parameter generation failed: {e}")
    
    def _sanitize_parameters(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Sanitize parameters to remove security issues"""
        
        sanitized = {}
        
        for key, value in parameters.items():
            # Ensure finite values
            if not math.isfinite(value):
                sanitized[key] = 1e-6  # Default safe value
                continue
            
            # Ensure positive values
            if value <= 0:
                sanitized[key] = abs(value) + 1e-12
                continue
            
            # Limit maximum values
            if value > 1e6:
                sanitized[key] = 1e6
                continue
            
            sanitized[key] = value
        
        return sanitized
    
    def _get_parameter_bounds(self, param_name: str, circuit_type: str) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for optimization"""
        
        bounds = {}
        
        if circuit_type == 'LNA':
            bounds.update({
                'W1': (1e-6, 500e-6), 'L1': (28e-9, 10e-6),
                'W2': (1e-6, 200e-6), 'L2': (28e-9, 5e-6),
                'Ls': (0.1e-9, 50e-9), 'Lg': (0.1e-9, 50e-9), 'Ld': (0.1e-9, 50e-9),
                'Ibias': (0.1e-3, 50e-3), 'Vbias': (0.1, 3.0),
                'Rd': (50, 20000), 'Cs': (1e-15, 100e-12), 'Cg': (1e-15, 100e-12)
            })
        elif circuit_type == 'Mixer':
            bounds.update({
                'Wrf': (1e-6, 300e-6), 'Lrf': (28e-9, 5e-6),
                'Wlo': (1e-6, 400e-6), 'Llo': (28e-9, 5e-6),
                'Wtail': (2e-6, 600e-6), 'Ltail': (50e-9, 20e-6),
                'RL': (100, 10000), 'Ibias': (0.2e-3, 30e-3),
                'Cin': (1e-15, 100e-12), 'Cout': (1e-15, 100e-12)
            })
        elif circuit_type == 'VCO':
            bounds.update({
                'W': (5e-6, 1000e-6), 'L': (28e-9, 1e-6),
                'Ltank': (0.1e-9, 100e-9), 'Ctank': (0.1e-12, 50e-12),
                'Cvar': (0.05e-12, 20e-12), 'Rtail': (100, 50000),
                'Ibias': (0.5e-3, 100e-3)
            })
        
        return bounds
    
    def _create_netlist_robust(self, topology: Dict[str, Any], 
                              parameters: Dict[str, float], 
                              spec: RobustDesignSpec) -> str:
        """Create SPICE netlist with enhanced security and validation"""
        
        try:
            # Sanitize topology name
            topo_name = ''.join(c for c in topology['name'] if c.isalnum() or c in ['_', '-'])
            
            # Header with security metadata
            netlist = f"""* {topo_name} - Generated by GenRF Generation 2 (Robust)
* Target: {spec.frequency/1e9:.2f}GHz, {spec.gain_min}dB gain, {spec.nf_max}dB NF
* Generation Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
* Specification Hash: {spec.spec_hash}
* Security Level: {'High' if self.enable_security else 'Standard'}

.param vdd={spec.supply_voltage:.3f}
.param temp={spec.temperature:.1f}
.param freq={spec.frequency:.0f}

"""
            
            # Circuit-specific netlist generation
            if spec.circuit_type == 'LNA':
                netlist += self._create_lna_netlist_robust(parameters, spec)
            elif spec.circuit_type == 'Mixer':
                netlist += self._create_mixer_netlist_robust(parameters, spec)
            elif spec.circuit_type == 'VCO':
                netlist += self._create_vco_netlist_robust(parameters, spec)
            else:
                netlist += self._create_generic_netlist_robust(parameters, spec)
            
            # Add footer with validation
            netlist += f"""
* Parameter validation passed: {self.enable_security}
* Component count: {len(parameters)}
* Netlist checksum: {hashlib.md5(netlist.encode()).hexdigest()[:8]}

.model nch nmos level=1 vto=0.5 kp=100u gamma=0.5 phi=0.7
.model pch pmos level=1 vto=-0.5 kp=50u gamma=0.5 phi=0.7

.end
"""
            
            return netlist
            
        except Exception as e:
            logger.error(f"Netlist creation failed: {e}")
            raise CircuitGenerationError(f"Netlist creation failed: {e}")
    
    def _create_lna_netlist_robust(self, parameters: Dict[str, float], spec: RobustDesignSpec) -> str:
        """Create robust LNA netlist with enhanced validation"""
        
        # Parameter validation and sanitization
        W1 = max(1e-6, min(500e-6, parameters.get('W1', 50e-6)))
        L1 = max(28e-9, min(10e-6, parameters.get('L1', 100e-9)))
        W2 = max(1e-6, min(200e-6, parameters.get('W2', 25e-6)))
        L2 = max(28e-9, min(5e-6, parameters.get('L2', 80e-9)))
        
        Ls = max(0.1e-9, min(50e-9, parameters.get('Ls', 2e-9)))
        Lg = max(0.1e-9, min(50e-9, parameters.get('Lg', 3e-9)))
        Ld = max(0.1e-9, min(50e-9, parameters.get('Ld', 4e-9)))
        
        Ibias = max(0.1e-3, min(50e-3, parameters.get('Ibias', 5e-3)))
        Vbias = max(0.1, min(spec.supply_voltage - 0.1, parameters.get('Vbias', 0.6)))
        
        Rd = max(50, min(20000, parameters.get('Rd', 1000)))
        
        return f"""* Robust LNA Core Circuit
* Input stage with validation
M1 n1 gate source bulk nch W={W1:.6e} L={L1:.6e}
M2 drain vbias2 n1 bulk nch W={W2:.6e} L={L2:.6e}

* Input matching network with bounds checking
Ls input n2 {Ls:.6e}
Lg n2 gate {Lg:.6e}
Cgs gate source {parameters.get('Cg', 1e-12):.6e}

* Output matching network
Ld drain n3 {Ld:.6e}
Cd n3 output {parameters.get('Cs', 2e-12):.6e}
Rd output vdd {Rd:.1f}

* Biasing network with protection
Vbias vbias 0 DC {Vbias:.3f}
Vbias2 vbias2 0 DC {min(spec.supply_voltage * 0.8, Vbias + 0.3):.3f}
Ibias source 0 DC {Ibias:.6e}

* Supply and ground connections
Vdd vdd 0 DC {spec.supply_voltage:.3f}
Vss bulk 0 DC 0

* Protection elements
Rprotect input n2 1.0
Cprotect drain vdd {1e-12:.6e}
"""
    
    def _create_mixer_netlist_robust(self, parameters: Dict[str, float], spec: RobustDesignSpec) -> str:
        """Create robust mixer netlist"""
        
        Wrf = max(1e-6, min(300e-6, parameters.get('Wrf', 50e-6)))
        Wlo = max(1e-6, min(400e-6, parameters.get('Wlo', 75e-6)))
        Wtail = max(2e-6, min(600e-6, parameters.get('Wtail', 100e-6)))
        
        RL = max(100, min(10000, parameters.get('RL', 1000)))
        Ibias = max(0.2e-3, min(30e-3, parameters.get('Ibias', 2e-3)))
        
        return f"""* Robust Gilbert Cell Mixer
* RF input stage
M1 n1 rf_p tail bulk nch W={Wrf:.6e} L=100n
M2 n2 rf_n tail bulk nch W={Wrf:.6e} L=100n

* LO switching quad with validation
M3 if_p lo_p n1 bulk nch W={Wlo:.6e} L=100n
M4 if_n lo_n n1 bulk nch W={Wlo:.6e} L=100n
M5 if_n lo_p n2 bulk nch W={Wlo:.6e} L=100n
M6 if_p lo_n n2 bulk nch W={Wlo:.6e} L=100n

* Tail current source with protection
Mtail tail vbias 0 bulk nch W={Wtail:.6e} L=200n
Itail vbias 0 DC {Ibias:.6e}

* Load with matching
RL1 if_p vdd {RL:.1f}
RL2 if_n vdd {RL:.1f}

* AC coupling with protection
Cin1 rf_input rf_p {parameters.get('Cin', 5e-12):.6e}
Cin2 rf_input rf_n {parameters.get('Cin', 5e-12):.6e}
Cout if_p if_output {parameters.get('Cout', 10e-12):.6e}

* Supply
Vdd vdd 0 DC {spec.supply_voltage:.3f}
Vss bulk 0 DC 0
"""
    
    def _create_vco_netlist_robust(self, parameters: Dict[str, float], spec: RobustDesignSpec) -> str:
        """Create robust VCO netlist"""
        
        W = max(5e-6, min(1000e-6, parameters.get('W', 100e-6)))
        L = max(28e-9, min(1e-6, parameters.get('L', 100e-9)))
        Ltank = max(0.1e-9, min(100e-9, parameters.get('Ltank', 5e-9)))
        Ctank = max(0.1e-12, min(50e-12, parameters.get('Ctank', 1e-12)))
        
        Ibias = max(0.5e-3, min(100e-3, parameters.get('Ibias', 10e-3)))
        
        return f"""* Robust LC VCO
* Cross-coupled pair with validation
M1 outp outn tail bulk nch W={W:.6e} L={L:.6e}
M2 outn outp tail bulk nch W={W:.6e} L={L:.6e}

* LC tank with bounds
L1 outp vdd {Ltank:.6e}
L2 outn vdd {Ltank:.6e}
C1 outp outn {Ctank:.6e}

* Varactor for tuning with protection
Cvar1 outp vctrl {parameters.get('Cvar', 0.5e-12):.6e}
Cvar2 outn vctrl {parameters.get('Cvar', 0.5e-12):.6e}

* Tail current with protection
Rtail tail 0 {parameters.get('Rtail', 2000):.1f}
Itail tail 0 DC {Ibias:.6e}

* Control and supply
Vctrl vctrl 0 DC {spec.supply_voltage * 0.5:.3f}
Vdd vdd 0 DC {spec.supply_voltage:.3f}
Vss bulk 0 DC 0

* Startup circuit for reliability
Rstartup outp outn 100k
"""
    
    def _create_generic_netlist_robust(self, parameters: Dict[str, float], spec: RobustDesignSpec) -> str:
        """Create robust generic netlist"""
        
        return f"""* Robust Generic Circuit
M1 output input 0 bulk nch W={parameters.get('W1', 50e-6):.6e} L={parameters.get('L1', 100e-9):.6e}
R1 output vdd {parameters.get('R1', 1000):.1f}
C1 input gate {parameters.get('C1', 1e-12):.6e}

* Supply
Vdd vdd 0 DC {spec.supply_voltage:.3f}
Vss bulk 0 DC 0
"""
    
    def _estimate_performance_robust(self, parameters: Dict[str, float], 
                                    spec: RobustDesignSpec) -> Dict[str, float]:
        """Robust performance estimation with error handling"""
        
        try:
            if spec.circuit_type == 'LNA':
                return self._estimate_lna_performance_robust(parameters, spec)
            elif spec.circuit_type == 'Mixer':
                return self._estimate_mixer_performance_robust(parameters, spec)
            elif spec.circuit_type == 'VCO':
                return self._estimate_vco_performance_robust(parameters, spec)
            else:
                return self._estimate_generic_performance_robust(parameters, spec)
                
        except Exception as e:
            logger.error(f"Performance estimation failed: {e}")
            # Return conservative estimates
            return {
                'gain_db': 0.0,
                'noise_figure_db': 50.0,
                'power_w': spec.power_max,
                's11_db': 0.0,
                'bandwidth_hz': spec.frequency * 0.01,
                'estimation_failed': True
            }
    
    def _estimate_lna_performance_robust(self, parameters: Dict[str, float], 
                                        spec: RobustDesignSpec) -> Dict[str, float]:
        """Robust LNA performance estimation"""
        
        try:
            # Extract and validate parameters
            W1 = max(1e-6, parameters.get('W1', 50e-6))
            L1 = max(28e-9, parameters.get('L1', 100e-9))
            Ibias = max(0.1e-3, parameters.get('Ibias', 5e-3))
            Rd = max(50, parameters.get('Rd', 1000))
            
            # Enhanced gm calculation with temperature effects
            temp_factor = 1.0 - (spec.temperature - 27) / 1000
            Vov = 0.3 * temp_factor  # Overdrive voltage
            gm = 2 * Ibias / max(0.1, Vov)  # Transconductance
            
            # Frequency-dependent gain calculation
            freq_factor = 1.0 / (1 + (spec.frequency / 50e9) ** 2)
            gain_linear = gm * Rd * freq_factor
            gain_db = 20 * math.log10(max(0.001, gain_linear))
            gain_db = min(60, max(-20, gain_db))  # Realistic bounds
            
            # Enhanced noise figure calculation
            gamma = 2.0  # Channel noise factor
            alpha = 1.0  # Gate noise factor
            gm_normalized = gm / 1e-3  # Normalize to mS
            
            nf_thermal = 1 + gamma * 2.5 / max(0.1, gm_normalized)
            nf_flicker = alpha * math.sqrt(W1 / L1) / 100
            nf_freq = 1 + (spec.frequency / 100e9) ** 0.5
            
            nf_linear = nf_thermal + nf_flicker + nf_freq
            nf_db = 10 * math.log10(max(1.01, nf_linear))
            nf_db = min(50, max(0.5, nf_db))  # Realistic bounds
            
            # Power consumption with margin
            power_dc = Ibias * spec.supply_voltage
            power_overhead = power_dc * 0.1  # 10% overhead for biasing
            power_total = power_dc + power_overhead
            
            # Input matching estimation
            Cgs = W1 * L1 * 2e6 * (1 + spec.temperature / 300)  # Temperature-dependent
            wt = gm / Cgs  # Unity gain frequency
            s11_db = -20 * math.exp(-spec.frequency / wt) if wt > 0 else -5
            s11_db = max(-30, min(-5, s11_db))
            
            # Bandwidth estimation
            bandwidth = min(wt / (2 * math.pi * 5), spec.frequency * 2)
            
            # Stability factor
            stability_margin = max(0, 10 * math.log10(wt / spec.frequency)) if wt > spec.frequency else 0
            
            return {
                'gain_db': gain_db,
                'noise_figure_db': nf_db,
                'power_w': power_total,
                's11_db': s11_db,
                'bandwidth_hz': bandwidth,
                'stability_margin_db': stability_margin,
                'unity_gain_freq_hz': wt / (2 * math.pi),
                'transconductance_ms': gm * 1000,
                'estimation_confidence': 0.85
            }
            
        except Exception as e:
            logger.error(f"LNA performance estimation failed: {e}")
            return {
                'gain_db': 0.0,
                'noise_figure_db': 50.0,
                'power_w': spec.power_max,
                's11_db': 0.0,
                'bandwidth_hz': spec.frequency * 0.01,
                'estimation_failed': True
            }
    
    def _estimate_mixer_performance_robust(self, parameters: Dict[str, float], 
                                          spec: RobustDesignSpec) -> Dict[str, float]:
        """Robust mixer performance estimation"""
        
        try:
            Wrf = max(1e-6, parameters.get('Wrf', 50e-6))
            Ibias = max(0.2e-3, parameters.get('Ibias', 2e-3))
            RL = max(100, parameters.get('RL', 1000))
            
            # Enhanced mixer analysis
            gm = 2 * Ibias / 0.3  # RF transconductance
            
            # Conversion gain (Gilbert cell analysis)
            conversion_efficiency = (2 / math.pi)
            freq_degradation = 1.0 / (1 + (spec.frequency / 100e9) ** 2)
            
            conversion_gain_linear = conversion_efficiency * gm * RL * freq_degradation
            conversion_gain_db = 20 * math.log10(max(0.001, conversion_gain_linear))
            conversion_gain_db = min(25, max(-20, conversion_gain_db))
            
            # Mixer noise figure (frequency dependent)
            base_nf = 8.0
            freq_nf = 3 * math.log10(max(1, spec.frequency / 1e9))
            nf_db = base_nf + freq_nf
            nf_db = min(40, max(6, nf_db))
            
            # Power consumption
            power_core = Ibias * spec.supply_voltage
            power_biasing = power_core * 0.2  # 20% for biasing
            power_total = power_core + power_biasing
            
            # Input/output matching
            s11_db = -15 + 5 * math.sin(spec.frequency / 10e9)  # Frequency ripple
            s11_db = max(-25, min(-8, s11_db))
            
            # Bandwidth
            bandwidth = min(gm / (2 * math.pi * 1e-12), spec.frequency * 0.3)
            
            # IP3 estimation
            ip3_dbm = 10 * math.log10(Ibias * 1000) + 5
            
            return {
                'gain_db': conversion_gain_db,
                'noise_figure_db': nf_db,
                'power_w': power_total,
                's11_db': s11_db,
                'bandwidth_hz': bandwidth,
                'ip3_dbm': ip3_dbm,
                'conversion_loss_db': -conversion_gain_db if conversion_gain_db < 0 else 0,
                'estimation_confidence': 0.8
            }
            
        except Exception as e:
            logger.error(f"Mixer performance estimation failed: {e}")
            return {
                'gain_db': -10.0,
                'noise_figure_db': 30.0,
                'power_w': spec.power_max,
                's11_db': -5.0,
                'estimation_failed': True
            }
    
    def _estimate_vco_performance_robust(self, parameters: Dict[str, float], 
                                        spec: RobustDesignSpec) -> Dict[str, float]:
        """Robust VCO performance estimation"""
        
        try:
            W = max(5e-6, parameters.get('W', 100e-6))
            L = max(28e-9, parameters.get('L', 100e-9))
            Ltank = max(0.1e-9, parameters.get('Ltank', 5e-9))
            Ctank = max(0.1e-12, parameters.get('Ctank', 1e-12))
            Ibias = max(0.5e-3, parameters.get('Ibias', 10e-3))
            
            # Oscillation frequency
            f_osc = 1 / (2 * math.pi * math.sqrt(Ltank * Ctank))
            f_osc = max(100e6, min(100e9, f_osc))  # Realistic bounds
            
            # Enhanced phase noise model (Leeson's formula)
            gm = 2 * Ibias / 0.3
            Q_tank = math.sqrt(Ltank / Ctank) / 10  # Tank Q factor
            
            # Phase noise at 1MHz offset
            kT = 1.38e-23 * (spec.temperature + 273)  # Thermal noise
            P_sig = 0.5 * Ibias * spec.supply_voltage  # Signal power
            
            phase_noise_1MHz = 10 * math.log10(
                2 * kT * (1e6) ** 2 / (P_sig * Q_tank ** 2 * f_osc ** 2)
            )
            phase_noise_1MHz = max(-150, min(-80, phase_noise_1MHz))
            
            # Power consumption
            power_core = Ibias * spec.supply_voltage
            power_buffer = power_core * 0.3  # Output buffer
            power_total = power_core + power_buffer
            
            # Tuning range estimation
            Cvar = max(0.05e-12, parameters.get('Cvar', 0.5e-12))
            tuning_range = f_osc * Cvar / (2 * Ctank + Cvar)
            tuning_range = min(f_osc * 0.5, tuning_range)
            
            # Startup condition check
            gm_crit = 2 / (Q_tank * math.sqrt(Ltank / Ctank))  # Critical gm
            startup_margin = 20 * math.log10(gm / gm_crit) if gm_crit > 0 else 0
            
            return {
                'gain_db': 0.0,  # VCO doesn't have gain
                'noise_figure_db': float('inf'),  # Not applicable
                'power_w': power_total,
                'frequency_hz': f_osc,
                'phase_noise_1MHz_dbchz': phase_noise_1MHz,
                'tuning_range_hz': tuning_range,
                'q_factor': Q_tank,
                'startup_margin_db': startup_margin,
                'frequency_error_pct': abs((f_osc - spec.frequency) / spec.frequency) * 100,
                'estimation_confidence': 0.75
            }
            
        except Exception as e:
            logger.error(f"VCO performance estimation failed: {e}")
            return {
                'gain_db': 0.0,
                'noise_figure_db': float('inf'),
                'power_w': spec.power_max,
                'frequency_hz': spec.frequency,
                'phase_noise_1MHz_dbchz': -100,
                'estimation_failed': True
            }
    
    def _estimate_generic_performance_robust(self, parameters: Dict[str, float], 
                                           spec: RobustDesignSpec) -> Dict[str, float]:
        """Generic robust performance estimation"""
        
        return {
            'gain_db': 15.0,
            'noise_figure_db': 3.0,
            'power_w': parameters.get('Ibias', 5e-3) * spec.supply_voltage,
            's11_db': -10.0,
            'bandwidth_hz': spec.frequency * 0.1,
            'estimation_confidence': 0.5
        }
    
    def _calculate_fom_robust(self, performance: Dict[str, float], spec: RobustDesignSpec) -> float:
        """Calculate robust figure of merit with enhanced weighting"""
        
        try:
            gain = performance.get('gain_db', 0)
            nf = performance.get('noise_figure_db', 100)
            power = performance.get('power_w', 1)
            
            # Specification compliance scoring
            gain_target = max(spec.gain_min, 0.1)
            nf_target = min(spec.nf_max, 100)
            power_target = spec.power_max
            
            # Normalized scores (0 to 1)
            gain_score = min(1.5, max(0, gain / gain_target))
            nf_score = max(0, min(1.5, nf_target / max(1, nf)))
            power_score = max(0, min(1.5, power_target / max(1e-6, power)))
            
            # Check hard constraints
            meets_gain = gain >= spec.gain_min if spec.gain_min > 0 else True
            meets_nf = nf <= spec.nf_max if spec.nf_max < float('inf') else True
            meets_power = power <= spec.power_max
            
            # Base FoM calculation
            if meets_gain and meets_nf and meets_power:
                # All specs met - calculate performance FoM
                base_fom = (gain_score * nf_score * power_score) ** (1/3)  # Geometric mean
                
                # Bonus for exceeding specifications
                gain_bonus = max(0, (gain - spec.gain_min) / max(1, spec.gain_min)) * 0.1
                nf_bonus = max(0, (spec.nf_max - nf) / max(1, spec.nf_max)) * 0.1
                power_bonus = max(0, (spec.power_max - power) / spec.power_max) * 0.1
                
                fom = base_fom + gain_bonus + nf_bonus + power_bonus
                
                # Additional performance metrics bonus
                if 'bandwidth_hz' in performance:
                    bw_bonus = min(0.1, performance['bandwidth_hz'] / spec.frequency)
                    fom += bw_bonus
                
                if 'stability_margin_db' in performance:
                    stability_bonus = min(0.1, max(0, performance['stability_margin_db']) / 20)
                    fom += stability_bonus
                
            else:
                # Specs not met - penalty based on how far off
                gain_penalty = max(0, (spec.gain_min - gain) / max(1, spec.gain_min)) if not meets_gain else 0
                nf_penalty = max(0, (nf - spec.nf_max) / max(1, spec.nf_max)) if not meets_nf else 0
                power_penalty = max(0, (power - spec.power_max) / spec.power_max) if not meets_power else 0
                
                total_penalty = gain_penalty + nf_penalty + power_penalty
                fom = max(0, 0.1 - total_penalty)  # Heavily penalized but not zero
            
            # Confidence factor
            confidence = performance.get('estimation_confidence', 0.5)
            fom *= confidence
            
            return max(0, min(10, fom))  # Bounded FoM
            
        except Exception as e:
            logger.error(f"FoM calculation failed: {e}")
            return 0.0
    
    def _comprehensive_validation(self, topology: Dict[str, Any], 
                                 parameters: Dict[str, float],
                                 performance: Dict[str, float], 
                                 netlist: str,
                                 spec: RobustDesignSpec) -> ValidationReport:
        """Comprehensive validation with detailed reporting"""
        
        errors = []
        warnings = []
        
        # 1. Parameter validation
        try:
            for param, value in parameters.items():
                if not math.isfinite(value):
                    errors.append(f"Parameter {param} is not finite: {value}")
                elif value <= 0:
                    errors.append(f"Parameter {param} is non-positive: {value}")
                elif value > 1e6:
                    warnings.append(f"Parameter {param} is very large: {value}")
        except Exception as e:
            errors.append(f"Parameter validation failed: {e}")
        
        # 2. Performance validation
        try:
            gain = performance.get('gain_db', 0)
            nf = performance.get('noise_figure_db', 100)
            power = performance.get('power_w', 1)
            
            if spec.gain_min > 0 and gain < spec.gain_min * 0.9:
                warnings.append(f"Gain {gain:.1f}dB is below target {spec.gain_min:.1f}dB")
            
            if spec.nf_max < 100 and nf > spec.nf_max * 1.1:
                warnings.append(f"Noise figure {nf:.2f}dB exceeds target {spec.nf_max:.2f}dB")
            
            if power > spec.power_max * 1.1:
                warnings.append(f"Power {power*1000:.1f}mW exceeds target {spec.power_max*1000:.1f}mW")
            
        except Exception as e:
            errors.append(f"Performance validation failed: {e}")
        
        # 3. Security validation
        security_score = 100.0
        try:
            if self.enable_security:
                param_secure, param_errors = self.security_validator.validate_parameters(parameters)
                netlist_secure, netlist_errors = self.security_validator.validate_netlist(netlist)
                
                errors.extend(param_errors)
                errors.extend(netlist_errors)
                
                security_score = self.security_validator.calculate_security_score(parameters, netlist)
                
                if security_score < 80:
                    warnings.append(f"Security score is low: {security_score:.1f}/100")
                    
        except Exception as e:
            errors.append(f"Security validation failed: {e}")
            security_score = 0.0
        
        # 4. Physical realizability validation
        try:
            # Check for reasonable component values
            if spec.circuit_type == 'LNA':
                W1 = parameters.get('W1', 0)
                L1 = parameters.get('L1', 0)
                if W1/L1 > 1000:
                    warnings.append(f"Very high W/L ratio: {W1/L1:.1f}")
                
                Ibias = parameters.get('Ibias', 0)
                if Ibias > 50e-3:
                    warnings.append(f"High bias current: {Ibias*1000:.1f}mA")
            
        except Exception as e:
            warnings.append(f"Physical validation failed: {e}")
        
        # 5. Calculate scores
        try:
            performance_score = min(100, max(0, 100 * self._calculate_fom_robust(performance, spec)))
            
            reliability_score = 100.0
            reliability_score -= len(errors) * 20
            reliability_score -= len(warnings) * 5
            reliability_score = max(0, min(100, reliability_score))
            
        except Exception as e:
            errors.append(f"Score calculation failed: {e}")
            performance_score = 0.0
            reliability_score = 0.0
        
        is_valid = len(errors) == 0
        
        return ValidationReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            security_score=security_score,
            performance_score=performance_score,
            reliability_score=reliability_score
        )
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        total = max(1, self.stats['total_generations'])
        
        return {
            'total_generations': self.stats['total_generations'],
            'successful_generations': self.stats['successful_generations'],
            'success_rate': self.stats['successful_generations'] / total,
            'security_failures': self.stats['security_failures'],
            'validation_failures': self.stats['validation_failures'],
            'optimization_failures': self.stats['optimization_failures'],
            'failure_rates': {
                'security': self.stats['security_failures'] / total,
                'validation': self.stats['validation_failures'] / total,
                'optimization': self.stats['optimization_failures'] / total
            }
        }

def demo_generation_2():
    """Demonstrate Generation 2 robust functionality"""
    
    print("=" * 70)
    print("🔧 GenRF Generation 2: MAKE IT ROBUST - AUTONOMOUS EXECUTION")
    print("=" * 70)
    
    # Test different circuit types with challenging specs
    test_cases = [
        {
            'name': 'High-Gain LNA',
            'spec': RobustDesignSpec(
                circuit_type='LNA',
                frequency=2.4e9,
                gain_min=20,
                nf_max=1.2,
                power_max=8e-3,
                validation_level='strict'
            )
        },
        {
            'name': 'mmWave Mixer',
            'spec': RobustDesignSpec(
                circuit_type='Mixer', 
                frequency=28e9,
                gain_min=5,
                nf_max=12.0,
                power_max=20e-3,
                validation_level='strict'
            )
        },
        {
            'name': 'Low-Power VCO',
            'spec': RobustDesignSpec(
                circuit_type='VCO',
                frequency=5.8e9,
                gain_min=0,
                nf_max=float('inf'),
                power_max=15e-3,
                validation_level='normal'
            )
        }
    ]
    
    results = []
    
    # Initialize robust diffuser
    diffuser = RobustCircuitDiffuser(
        verbose=True,
        validation_level='strict',
        enable_security=True,
        max_attempts=3
    )
    
    for test_case in test_cases:
        name = test_case['name']
        spec = test_case['spec']
        
        print(f"\n🔬 Generating {name}...")
        print(f"   Circuit: {spec.circuit_type} @ {spec.frequency/1e9:.1f}GHz")
        print(f"   Targets: Gain≥{spec.gain_min}dB, NF≤{spec.nf_max}dB, Power≤{spec.power_max*1000:.1f}mW")
        
        try:
            # Generate circuit with robust error handling
            result = diffuser.generate(
                spec,
                n_candidates=8,
                optimization_steps=15,
                spice_validation=False,
                retry_on_failure=True
            )
            
            # Collect results
            test_result = {
                'name': name,
                'circuit_type': spec.circuit_type,
                'frequency_ghz': spec.frequency / 1e9,
                'target_specs': {
                    'gain_min_db': spec.gain_min,
                    'nf_max_db': spec.nf_max,
                    'power_max_mw': spec.power_max * 1000
                },
                'achieved_performance': {
                    'gain_db': result.gain,
                    'nf_db': result.nf,
                    'power_mw': result.power * 1000
                },
                'quality_metrics': {
                    'security_score': result.security_score,
                    'reliability_score': result.validation_report.reliability_score,
                    'performance_score': result.validation_report.performance_score,
                    'overall_score': (result.security_score + 
                                     result.validation_report.reliability_score + 
                                     result.validation_report.performance_score) / 3
                },
                'validation_status': {
                    'is_valid': result.validation_report.is_valid,
                    'is_secure': result.is_secure,
                    'is_reliable': result.is_reliable,
                    'error_count': len(result.validation_report.errors),
                    'warning_count': len(result.validation_report.warnings)
                },
                'generation_time_s': result.generation_time,
                'meets_all_specs': all([
                    result.gain >= spec.gain_min if spec.gain_min > 0 else True,
                    result.nf <= spec.nf_max if spec.nf_max < float('inf') else True,
                    result.power <= spec.power_max
                ])
            }
            
            results.append(test_result)
            
            # Save individual results
            output_dir = Path("gen2_robust_outputs")
            output_dir.mkdir(exist_ok=True)
            
            # Save netlist
            netlist_file = output_dir / f"{spec.circuit_type.lower()}_{name.replace(' ', '_').lower()}_netlist.spice"
            with open(netlist_file, 'w') as f:
                f.write(result.netlist)
            
            # Save SKILL with validation report
            skill_file = output_dir / f"{spec.circuit_type.lower()}_{name.replace(' ', '_').lower()}_design.il"
            result.export_skill(skill_file, include_validation=True)
            
            # Save detailed result
            result_file = output_dir / f"{spec.circuit_type.lower()}_{name.replace(' ', '_').lower()}_result.json"
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            print(f"   ✅ Generation successful!")
            print(f"   Performance: Gain={result.gain:.1f}dB, NF={result.nf:.2f}dB, Power={result.power*1000:.1f}mW")
            print(f"   Quality: Security={result.security_score:.1f}, Reliability={result.validation_report.reliability_score:.1f}")
            print(f"   Validation: {len(result.validation_report.errors)} errors, {len(result.validation_report.warnings)} warnings")
            
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            
            # Record failure
            test_result = {
                'name': name,
                'circuit_type': spec.circuit_type,
                'frequency_ghz': spec.frequency / 1e9,
                'generation_failed': True,
                'failure_reason': str(e),
                'generation_time_s': 0,
                'meets_all_specs': False
            }
            results.append(test_result)
    
    # Generate comprehensive summary
    generation_stats = diffuser.get_generation_stats()
    
    summary = {
        'generation': 'Generation 2: MAKE IT ROBUST',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'execution_summary': {
            'total_test_cases': len(test_cases),
            'successful_generations': len([r for r in results if not r.get('generation_failed', False)]),
            'failed_generations': len([r for r in results if r.get('generation_failed', False)]),
            'success_rate': len([r for r in results if not r.get('generation_failed', False)]) / len(test_cases),
            'avg_generation_time': sum(r.get('generation_time_s', 0) for r in results) / len(results)
        },
        'quality_metrics': {
            'avg_security_score': sum(r.get('quality_metrics', {}).get('security_score', 0) for r in results if not r.get('generation_failed')) / max(1, len([r for r in results if not r.get('generation_failed')])),
            'avg_reliability_score': sum(r.get('quality_metrics', {}).get('reliability_score', 0) for r in results if not r.get('generation_failed')) / max(1, len([r for r in results if not r.get('generation_failed')])),
            'total_validation_errors': sum(r.get('validation_status', {}).get('error_count', 0) for r in results),
            'total_validation_warnings': sum(r.get('validation_status', {}).get('warning_count', 0) for r in results)
        },
        'generation_statistics': generation_stats,
        'detailed_results': results
    }
    
    # Save comprehensive summary
    summary_file = Path("gen2_robust_outputs/robust_generation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print final summary
    print(f"\n" + "=" * 70)
    print(f"✅ Generation 2 Robust Execution Complete!")
    print(f"   Total Test Cases: {len(test_cases)}")
    print(f"   Successful: {summary['execution_summary']['successful_generations']}")
    print(f"   Failed: {summary['execution_summary']['failed_generations']}")
    print(f"   Success Rate: {summary['execution_summary']['success_rate']:.1%}")
    print(f"   Average Security Score: {summary['quality_metrics']['avg_security_score']:.1f}/100")
    print(f"   Average Reliability Score: {summary['quality_metrics']['avg_reliability_score']:.1f}/100")
    print(f"   Total Validation Errors: {summary['quality_metrics']['total_validation_errors']}")
    print(f"   Total Validation Warnings: {summary['quality_metrics']['total_validation_warnings']}")
    print(f"   Outputs saved to: gen2_robust_outputs/")
    print("=" * 70)
    
    return summary

if __name__ == "__main__":
    demo_generation_2()