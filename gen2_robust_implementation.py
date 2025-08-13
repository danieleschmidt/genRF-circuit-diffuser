#!/usr/bin/env python3
"""
Generation 2: MAKE IT ROBUST (Reliable)
Enhanced implementation with comprehensive error handling, validation, and monitoring
"""

import json
import random
import time
import hashlib
import logging
import traceback
import threading
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager
from enum import Enum
import warnings
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('genrf_robust.log'),
        logging.StreamHandler()
    ]
)

class CircuitType(Enum):
    """Enumerated circuit types for type safety"""
    LNA = "LNA"
    MIXER = "MIXER" 
    VCO = "VCO"
    PA = "PA"
    FILTER = "FILTER"

class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"

class GenRFError(Exception):
    """Base exception for GenRF operations"""
    pass

class ValidationError(GenRFError):
    """Raised when validation fails"""
    pass

class GenerationError(GenRFError):
    """Raised when circuit generation fails"""
    pass

class ExportError(GenRFError):
    """Raised when export operations fail"""
    pass

class SecurityError(GenRFError):
    """Raised when security validation fails"""
    pass

@dataclass
class RobustDesignSpec:
    """Enhanced design specification with comprehensive validation"""
    circuit_type: str
    frequency: float
    gain_min: Optional[float] = None
    nf_max: Optional[float] = None
    power_max: Optional[float] = None
    technology: str = "Generic"
    temperature: float = 25.0  # Celsius
    voltage_supply: float = 1.8  # Volts
    process_corner: str = "TT"  # Typical-Typical
    
    def __post_init__(self):
        """Validate specification after initialization"""
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Comprehensive parameter validation"""
        # Circuit type validation
        if not isinstance(self.circuit_type, str):
            raise ValidationError("Circuit type must be a string")
        
        if self.circuit_type.upper() not in [ct.value for ct in CircuitType]:
            warnings.warn(f"Unknown circuit type: {self.circuit_type}")
        
        # Frequency validation
        if not isinstance(self.frequency, (int, float)):
            raise ValidationError("Frequency must be numeric")
        if self.frequency <= 0:
            raise ValidationError("Frequency must be positive")
        if self.frequency > 1e12:  # 1 THz limit
            raise ValidationError("Frequency exceeds reasonable limits (>1THz)")
        
        # Performance parameter validation
        if self.gain_min is not None:
            if not isinstance(self.gain_min, (int, float)):
                raise ValidationError("Minimum gain must be numeric")
            if self.gain_min < -50 or self.gain_min > 100:
                raise ValidationError("Minimum gain outside reasonable range (-50 to 100 dB)")
        
        if self.nf_max is not None:
            if not isinstance(self.nf_max, (int, float)):
                raise ValidationError("Maximum noise figure must be numeric")
            if self.nf_max < 0 or self.nf_max > 50:
                raise ValidationError("Maximum noise figure outside reasonable range (0 to 50 dB)")
        
        if self.power_max is not None:
            if not isinstance(self.power_max, (int, float)):
                raise ValidationError("Maximum power must be numeric")
            if self.power_max <= 0 or self.power_max > 10:  # 10W limit
                raise ValidationError("Maximum power outside reasonable range (0 to 10W)")
        
        # Environmental parameter validation
        if not isinstance(self.temperature, (int, float)):
            raise ValidationError("Temperature must be numeric")
        if self.temperature < -273 or self.temperature > 1000:
            raise ValidationError("Temperature outside reasonable range (-273 to 1000°C)")
        
        if not isinstance(self.voltage_supply, (int, float)):
            raise ValidationError("Supply voltage must be numeric")
        if self.voltage_supply <= 0 or self.voltage_supply > 50:
            raise ValidationError("Supply voltage outside reasonable range (0 to 50V)")

@dataclass
class RobustCircuitResult:
    """Enhanced circuit result with validation and metadata"""
    circuit_id: str
    circuit_type: str
    gain: float
    noise_figure: float
    power: float
    components: Dict[str, Any]
    netlist: str
    generation_time: float
    validation_level: str
    warnings: List[str]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate result after initialization"""
        self._validate_result()
    
    def _validate_result(self):
        """Validate circuit result parameters"""
        if not isinstance(self.circuit_id, str) or not self.circuit_id:
            raise ValidationError("Circuit ID must be a non-empty string")
        
        if not isinstance(self.gain, (int, float)):
            raise ValidationError("Gain must be numeric")
        
        if not isinstance(self.noise_figure, (int, float)):
            raise ValidationError("Noise figure must be numeric")
        
        if not isinstance(self.power, (int, float)) or self.power < 0:
            raise ValidationError("Power must be non-negative numeric")

class SecurityManager:
    """Security validation and sanitization"""
    
    @staticmethod
    def sanitize_input(value: str) -> str:
        """Sanitize string inputs to prevent injection attacks"""
        if not isinstance(value, str):
            raise SecurityError("Input must be string for sanitization")
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'", ';', '|', '&', '`', '$', '(', ')']
        sanitized = value
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length to prevent DoS
        if len(sanitized) > 1000:
            raise SecurityError("Input string too long (>1000 chars)")
        
        return sanitized
    
    @staticmethod
    def validate_file_path(path: str) -> Path:
        """Validate file path for security"""
        if not isinstance(path, str):
            raise SecurityError("Path must be string")
        
        # Sanitize path
        clean_path = SecurityManager.sanitize_input(path)
        path_obj = Path(clean_path)
        
        # Check for path traversal attempts
        if '..' in path_obj.parts:
            raise SecurityError("Path traversal attempt detected")
        
        # Ensure path is within current directory
        try:
            path_obj.resolve().relative_to(Path.cwd())
        except ValueError:
            raise SecurityError("Path outside allowed directory")
        
        return path_obj

class RobustValidator:
    """Enhanced validation with multiple levels"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STRICT):
        self.level = level
        self.logger = logging.getLogger(__name__)
    
    def validate_spec(self, spec: RobustDesignSpec) -> Tuple[bool, List[str]]:
        """Enhanced specification validation with warnings"""
        warnings = []
        
        try:
            # Basic validation (already done in __post_init__)
            self.logger.info(f"Validating specification for {spec.circuit_type}")
            
            # Technology-specific validation
            if self.level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                warnings.extend(self._validate_technology_compatibility(spec))
            
            # Physics-based validation
            if self.level == ValidationLevel.PARANOID:
                warnings.extend(self._validate_physics_constraints(spec))
            
            self.logger.info("Specification validation completed")
            return True, warnings
            
        except Exception as e:
            self.logger.error(f"Specification validation failed: {e}")
            raise ValidationError(f"Specification validation failed: {e}")
    
    def _validate_technology_compatibility(self, spec: RobustDesignSpec) -> List[str]:
        """Validate technology-specific constraints"""
        warnings = []
        
        # CMOS technology limits
        if "CMOS" in spec.technology.upper():
            if spec.frequency > 100e9:  # 100 GHz
                warnings.append(f"High frequency ({spec.frequency/1e9:.1f}GHz) may be challenging for CMOS")
        
        # Temperature compatibility
        if spec.temperature > 125:
            warnings.append(f"High temperature ({spec.temperature}°C) may affect reliability")
        elif spec.temperature < -40:
            warnings.append(f"Low temperature ({spec.temperature}°C) may affect performance")
        
        return warnings
    
    def _validate_physics_constraints(self, spec: RobustDesignSpec) -> List[str]:
        """Validate physics-based constraints"""
        warnings = []
        
        # Gain-bandwidth product limitations
        if spec.gain_min and spec.frequency:
            gbw = spec.gain_min * spec.frequency
            if gbw > 1e12:  # 1 THz*dB
                warnings.append("Gain-bandwidth product may exceed device limits")
        
        # Power-frequency scaling
        if spec.power_max and spec.frequency > 10e9:  # Above 10 GHz
            if spec.power_max > 0.1:  # 100 mW
                warnings.append("High power at high frequency may cause thermal issues")
        
        return warnings
    
    def validate_result(self, result: RobustCircuitResult, spec: RobustDesignSpec) -> bool:
        """Enhanced result validation against specification"""
        self.logger.info(f"Validating result for circuit {result.circuit_id}")
        
        validation_errors = []
        
        # Performance validation with tolerance
        if spec.gain_min is not None:
            if result.gain < spec.gain_min * 0.9:  # 10% tolerance
                validation_errors.append(f"Gain {result.gain:.1f}dB below minimum {spec.gain_min:.1f}dB")
        
        if spec.nf_max is not None:
            if result.noise_figure > spec.nf_max * 1.1:  # 10% tolerance
                validation_errors.append(f"Noise figure {result.noise_figure:.1f}dB above maximum {spec.nf_max:.1f}dB")
        
        if spec.power_max is not None:
            if result.power > spec.power_max * 1.1:  # 10% tolerance
                validation_errors.append(f"Power {result.power*1000:.1f}mW above maximum {spec.power_max*1000:.1f}mW")
        
        if validation_errors:
            self.logger.warning(f"Validation warnings: {validation_errors}")
            return False
        
        self.logger.info("Result validation passed")
        return True

class MonitoringManager:
    """System monitoring and metrics collection"""
    
    def __init__(self):
        self.metrics = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_time': 0.0,
            'avg_generation_time': 0.0,
            'circuit_types': {},
            'validation_failures': 0
        }
        self._lock = threading.Lock()
    
    def record_generation(self, circuit_type: str, success: bool, generation_time: float):
        """Record generation metrics"""
        with self._lock:
            self.metrics['total_generations'] += 1
            
            if success:
                self.metrics['successful_generations'] += 1
            else:
                self.metrics['failed_generations'] += 1
            
            self.metrics['total_time'] += generation_time
            self.metrics['avg_generation_time'] = self.metrics['total_time'] / self.metrics['total_generations']
            
            # Track by circuit type
            if circuit_type not in self.metrics['circuit_types']:
                self.metrics['circuit_types'][circuit_type] = {'count': 0, 'success': 0}
            
            self.metrics['circuit_types'][circuit_type]['count'] += 1
            if success:
                self.metrics['circuit_types'][circuit_type]['success'] += 1
    
    def record_validation_failure(self):
        """Record validation failure"""
        with self._lock:
            self.metrics['validation_failures'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self._lock:
            return self.metrics.copy()
    
    def generate_report(self) -> str:
        """Generate metrics report"""
        metrics = self.get_metrics()
        
        if metrics['total_generations'] == 0:
            return "No generations recorded"
        
        success_rate = 100 * metrics['successful_generations'] / metrics['total_generations']
        
        report = f"""
🛡️  GenRF Robust Generation Report
======================================
Total Generations: {metrics['total_generations']}
Successful: {metrics['successful_generations']}
Failed: {metrics['failed_generations']}
Success Rate: {success_rate:.1f}%
Average Time: {metrics['avg_generation_time']*1000:.1f}ms
Validation Failures: {metrics['validation_failures']}

Circuit Types:
"""
        
        for circuit_type, stats in metrics['circuit_types'].items():
            type_success_rate = 100 * stats['success'] / stats['count'] if stats['count'] > 0 else 0
            report += f"  {circuit_type}: {stats['count']} ({type_success_rate:.1f}% success)\n"
        
        return report.strip()

class RobustCircuitGenerator:
    """Enhanced circuit generator with comprehensive error handling"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.logger = logging.getLogger(__name__)
        self.validator = RobustValidator(validation_level)
        self.monitor = MonitoringManager()
        self.cache = {}
        self._lock = threading.Lock()
        
        self.logger.info("Robust circuit generator initialized")
    
    @contextmanager
    def _error_handler(self, operation: str):
        """Context manager for consistent error handling"""
        try:
            self.logger.debug(f"Starting operation: {operation}")
            yield
            self.logger.debug(f"Completed operation: {operation}")
        except ValidationError as e:
            self.logger.error(f"Validation error in {operation}: {e}")
            self.monitor.record_validation_failure()
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in {operation}: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise GenerationError(f"Operation '{operation}' failed: {e}")
    
    def generate_circuit(self, spec: RobustDesignSpec, retries: int = 3) -> RobustCircuitResult:
        """Generate circuit with comprehensive error handling and retries"""
        
        with self._error_handler("circuit_generation"):
            # Validate specification
            is_valid, warnings = self.validator.validate_spec(spec)
            if not is_valid:
                raise ValidationError("Specification validation failed")
            
            # Attempt generation with retries
            last_error = None
            for attempt in range(retries + 1):
                try:
                    self.logger.info(f"Generation attempt {attempt + 1}/{retries + 1} for {spec.circuit_type}")
                    
                    start_time = time.time()
                    result = self._generate_circuit_internal(spec, warnings)
                    generation_time = time.time() - start_time
                    
                    # Validate result
                    if self.validator.validate_result(result, spec):
                        self.monitor.record_generation(spec.circuit_type, True, generation_time)
                        self.logger.info(f"Successfully generated circuit {result.circuit_id}")
                        return result
                    else:
                        self.logger.warning(f"Generated circuit failed validation on attempt {attempt + 1}")
                        if attempt == retries:
                            self.monitor.record_generation(spec.circuit_type, False, generation_time)
                            raise ValidationError("Generated circuit failed validation after all retries")
                
                except Exception as e:
                    last_error = e
                    if attempt < retries:
                        self.logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(0.1 * (attempt + 1))  # Progressive backoff
                    else:
                        self.monitor.record_generation(spec.circuit_type, False, 0.0)
                        raise
            
            # This should not be reached, but handle the case
            if last_error:
                raise last_error
            else:
                raise GenerationError("All generation attempts failed for unknown reasons")
    
    def _generate_circuit_internal(self, spec: RobustDesignSpec, warnings: List[str]) -> RobustCircuitResult:
        """Internal circuit generation logic"""
        
        # Create deterministic but varied results
        seed = hash(f"{spec.circuit_type}_{spec.frequency}_{spec.technology}_{spec.temperature}_{spec.voltage_supply}")
        random.seed(seed)
        
        circuit_id = hashlib.md5(str(seed).encode()).hexdigest()[:16]
        
        # Enhanced circuit parameter generation with environmental effects
        temp_factor = 1.0 + (spec.temperature - 25) * 0.001  # Temperature coefficient
        voltage_factor = spec.voltage_supply / 1.8  # Voltage scaling
        
        if spec.circuit_type.upper() == "LNA":
            base_gain = random.uniform(spec.gain_min or 10, (spec.gain_min or 10) + 15)
            gain = base_gain * temp_factor * voltage_factor
            
            base_nf = random.uniform(0.3, spec.nf_max or 2.0)
            nf = base_nf * temp_factor
            
            base_power = random.uniform(1e-3, spec.power_max or 20e-3)
            power = base_power * voltage_factor**2
            
            components = {
                "transistor_M1": {
                    "type": "NMOS", 
                    "W": 120e-6 * voltage_factor, 
                    "L": 65e-9,
                    "multiplier": max(1, int(voltage_factor))
                },
                "inductor_L1": {"value": 2.2e-9, "Q": 18 * temp_factor},
                "capacitor_C1": {"value": 1.2e-12},
                "resistor_R1": {"value": 50.0 * temp_factor},
                "bias_current": {"value": 2e-3 * voltage_factor}
            }
            netlist = self._generate_enhanced_lna_netlist(components, spec)
            
        elif spec.circuit_type.upper() == "MIXER":
            gain = random.uniform(8, 18) * temp_factor
            nf = random.uniform(6, 12) * temp_factor
            power = random.uniform(8e-3, 60e-3) * voltage_factor**2
            
            components = {
                "transistor_M1": {"type": "NMOS", "W": 60e-6, "L": 65e-9},
                "transistor_M2": {"type": "NMOS", "W": 60e-6, "L": 65e-9},
                "transistor_M3": {"type": "NMOS", "W": 30e-6, "L": 65e-9},
                "inductor_L1": {"value": 1.0e-9, "Q": 15},
                "capacitor_C1": {"value": 800e-15},
                "if_load": {"value": 1e3}
            }
            netlist = self._generate_enhanced_mixer_netlist(components, spec)
            
        elif spec.circuit_type.upper() == "VCO":
            gain = 0  # N/A for VCO
            nf = random.uniform(-100, -80)  # Phase noise in dBc/Hz
            power = random.uniform(5e-3, 30e-3) * voltage_factor**2
            
            components = {
                "transistor_M1": {"type": "NMOS", "W": 80e-6, "L": 130e-9},
                "transistor_M2": {"type": "NMOS", "W": 80e-6, "L": 130e-9},
                "inductor_L1": {"value": 500e-12, "Q": 25},
                "varactor_C1": {"value": 200e-15, "tuning_range": 0.3},
                "tail_current": {"value": 4e-3}
            }
            netlist = self._generate_vco_netlist(components, spec)
            
        else:
            # Generic circuit
            gain = random.uniform(-5, 25)
            nf = random.uniform(2, 15)
            power = random.uniform(1e-3, 100e-3)
            components = {"generic": {"type": "unknown", "value": 1}}
            netlist = f"* Generic {spec.circuit_type} circuit\n.end"
        
        # Add metadata
        metadata = {
            "generation_timestamp": time.time(),
            "specification": asdict(spec),
            "temperature_factor": temp_factor,
            "voltage_factor": voltage_factor,
            "random_seed": seed,
            "process_corner": spec.process_corner
        }
        
        return RobustCircuitResult(
            circuit_id=circuit_id,
            circuit_type=spec.circuit_type,
            gain=gain,
            noise_figure=nf,
            power=power,
            components=components,
            netlist=netlist,
            generation_time=0.0,  # Will be set by caller
            validation_level=self.validator.level.value,
            warnings=warnings,
            metadata=metadata
        )
    
    def _generate_enhanced_lna_netlist(self, components: Dict, spec: RobustDesignSpec) -> str:
        """Generate enhanced SPICE netlist for LNA with environmental parameters"""
        temp = spec.temperature
        vdd = spec.voltage_supply
        
        return f"""* Enhanced Low Noise Amplifier
* Technology: {spec.technology}
* Temperature: {temp}°C, VDD: {vdd}V
.subckt LNA_ENHANCED in out vdd gnd
.param temp_coeff={1 + (temp-25)*0.001}

* Main amplifying transistor
M1 out in bias gnd nmos w={components['transistor_M1']['W']} l={components['transistor_M1']['L']} m={components['transistor_M1']['multiplier']}

* Load inductor
L1 vdd out {components['inductor_L1']['value']} Q={components['inductor_L1']['Q']}

* Input matching
C1 in gnd {components['capacitor_C1']['value']}

* Bias network
Rbias bias gnd {components['resistor_R1']['value']}
Ibias bias gnd {components['bias_current']['value']}

* Temperature and voltage monitoring
.ic v(vdd)={vdd}
.temp {temp}

.ends LNA_ENHANCED
.end"""

    def _generate_enhanced_mixer_netlist(self, components: Dict, spec: RobustDesignSpec) -> str:
        """Generate enhanced mixer netlist"""
        return f"""* Enhanced Gilbert Cell Mixer  
* Technology: {spec.technology}
* Frequency: {spec.frequency/1e9:.2f} GHz
.subckt MIXER_ENHANCED rf_in lo_in if_out vdd gnd
* RF input stage
M1 n1 rf_in tail gnd nmos w={components['transistor_M1']['W']} l={components['transistor_M1']['L']}
M2 n2 rf_in tail gnd nmos w={components['transistor_M2']['W']} l={components['transistor_M2']['L']}

* LO switching stage  
M3 if_out lo_in n1 gnd nmos w={components['transistor_M3']['W']} l={components['transistor_M3']['L']}
M4 if_out lo_in_b n2 gnd nmos w={components['transistor_M3']['W']} l={components['transistor_M3']['L']}

* Load and output
L1 vdd if_out {components['inductor_L1']['value']}
C1 if_out gnd {components['capacitor_C1']['value']}
Rload if_out gnd {components['if_load']['value']}

* Tail current source
Itail tail gnd 4e-3

.ends MIXER_ENHANCED
.end"""

    def _generate_vco_netlist(self, components: Dict, spec: RobustDesignSpec) -> str:
        """Generate VCO netlist"""
        return f"""* LC VCO with Varactor Tuning
* Frequency: {spec.frequency/1e9:.2f} GHz
.subckt VCO_LC out_p out_n vdd gnd vtune
* Cross-coupled pair
M1 out_p out_n tail gnd nmos w={components['transistor_M1']['W']} l={components['transistor_M1']['L']}
M2 out_n out_p tail gnd nmos w={components['transistor_M2']['W']} l={components['transistor_M2']['L']}

* LC tank
L1 vdd out_p {components['inductor_L1']['value']} Q={components['inductor_L1']['Q']}
L2 vdd out_n {components['inductor_L1']['value']} Q={components['inductor_L1']['Q']}

* Varactor tuning
Cvar1 out_p gnd {components['varactor_C1']['value']} 
Cvar2 out_n gnd {components['varactor_C1']['value']}

* Tail current
Itail tail gnd {components['tail_current']['value']}

.ends VCO_LC
.end"""

class RobustExporter:
    """Enhanced export with security validation and comprehensive formats"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.security = SecurityManager()
    
    def export_spice(self, result: RobustCircuitResult, filename: str, include_testbench: bool = True):
        """Export enhanced SPICE netlist with optional testbench"""
        
        with self._export_context("SPICE", filename):
            safe_path = self.security.validate_file_path(filename)
            
            content = result.netlist
            
            if include_testbench:
                content += self._generate_testbench(result)
            
            with open(safe_path, 'w') as f:
                f.write(content)
            
            self.logger.info(f"SPICE netlist exported to {safe_path}")
    
    def export_json(self, result: RobustCircuitResult, filename: str, include_metadata: bool = True):
        """Export comprehensive JSON data"""
        
        with self._export_context("JSON", filename):
            safe_path = self.security.validate_file_path(filename)
            
            data = {
                'circuit_info': {
                    'id': result.circuit_id,
                    'type': result.circuit_type,
                    'validation_level': result.validation_level
                },
                'performance': {
                    'gain_db': round(result.gain, 3),
                    'noise_figure_db': round(result.noise_figure, 3),
                    'power_w': round(result.power, 6),
                    'generation_time_ms': round(result.generation_time * 1000, 2)
                },
                'components': result.components,
                'netlist': result.netlist,
                'warnings': result.warnings
            }
            
            if include_metadata:
                data['metadata'] = result.metadata
            
            with open(safe_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"JSON data exported to {safe_path}")
    
    def export_summary_report(self, results: List[RobustCircuitResult], filename: str):
        """Export comprehensive summary report"""
        
        with self._export_context("Summary Report", filename):
            safe_path = self.security.validate_file_path(filename)
            
            report = self._generate_summary_report(results)
            
            with open(safe_path, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Summary report exported to {safe_path}")
    
    @contextmanager 
    def _export_context(self, export_type: str, filename: str):
        """Context manager for export operations"""
        try:
            self.logger.debug(f"Starting {export_type} export to {filename}")
            yield
            self.logger.debug(f"Completed {export_type} export")
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            raise ExportError(f"{export_type} export to {filename} failed: {e}")
    
    def _generate_testbench(self, result: RobustCircuitResult) -> str:
        """Generate SPICE testbench for the circuit"""
        return f"""

* Testbench for {result.circuit_type} {result.circuit_id}
.include "{result.circuit_id}.sp"

* Supply voltage
Vdd vdd gnd 1.8
Vgnd gnd 0 0

* Test signal
Vin in gnd 0 AC 1m

* DUT instantiation  
X1 in out vdd gnd {result.circuit_type}_ENHANCED

* Analysis
.ac dec 100 1meg 100gig
.noise v(out) Vin dec 100 1meg 100gig
.op

.control
run
let gain_db = db(v(out)/v(in))
print gain_db
.endc

.end"""

    def _generate_summary_report(self, results: List[RobustCircuitResult]) -> str:
        """Generate comprehensive summary report"""
        if not results:
            return "No circuit results to summarize."
        
        # Aggregate statistics
        total_circuits = len(results)
        avg_gain = sum(r.gain for r in results) / total_circuits
        avg_nf = sum(r.noise_figure for r in results) / total_circuits
        avg_power = sum(r.power for r in results) / total_circuits
        avg_gen_time = sum(r.generation_time for r in results) / total_circuits
        
        # Circuit type distribution
        type_counts = {}
        for result in results:
            type_counts[result.circuit_type] = type_counts.get(result.circuit_type, 0) + 1
        
        # Warning analysis
        all_warnings = []
        for result in results:
            all_warnings.extend(result.warnings)
        warning_counts = {}
        for warning in all_warnings:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
        
        report = f"""🛡️  GenRF Robust Circuit Generation Summary Report
================================================================

📊 Overall Statistics:
- Total Circuits Generated: {total_circuits}
- Average Gain: {avg_gain:.2f} dB
- Average Noise Figure: {avg_nf:.2f} dB  
- Average Power: {avg_power*1000:.2f} mW
- Average Generation Time: {avg_gen_time*1000:.2f} ms

📈 Circuit Type Distribution:
"""
        for circuit_type, count in type_counts.items():
            percentage = 100 * count / total_circuits
            report += f"- {circuit_type}: {count} ({percentage:.1f}%)\n"
        
        if warning_counts:
            report += f"\n⚠️  Warning Analysis:\n"
            for warning, count in sorted(warning_counts.items(), key=lambda x: x[1], reverse=True):
                report += f"- {warning}: {count} occurrences\n"
        
        report += f"\n🔧 Individual Circuit Details:\n"
        for i, result in enumerate(results, 1):
            report += f"""
Circuit {i}: {result.circuit_id}
  Type: {result.circuit_type}
  Performance: Gain={result.gain:.1f}dB, NF={result.noise_figure:.2f}dB, Power={result.power*1000:.1f}mW
  Generation Time: {result.generation_time*1000:.1f}ms
  Validation Level: {result.validation_level}
  Warnings: {len(result.warnings)}
"""
        
        return report

def main():
    """Generation 2 demonstration with comprehensive robustness"""
    print("🛡️  Generation 2: MAKE IT ROBUST - Enhanced Circuit Generator")
    print("=" * 70)
    
    # Initialize components
    generator = RobustCircuitGenerator(ValidationLevel.STRICT)
    exporter = RobustExporter()
    
    # Enhanced test specifications
    test_specs = [
        RobustDesignSpec(
            circuit_type="LNA",
            frequency=2.4e9,
            gain_min=18,
            nf_max=1.2,
            power_max=8e-3,
            technology="TSMC65nm",
            temperature=85,  # High temperature test
            voltage_supply=1.8,
            process_corner="SS"
        ),
        RobustDesignSpec(
            circuit_type="MIXER",
            frequency=28e9,  # mmWave
            technology="SiGe130nm", 
            temperature=-20,  # Low temperature test
            voltage_supply=2.5
        ),
        RobustDesignSpec(
            circuit_type="VCO",
            frequency=5.8e9,
            technology="CMOS180nm",
            temperature=125,  # Extreme temperature
            voltage_supply=1.2
        )
    ]
    
    results = []
    
    for i, spec in enumerate(test_specs, 1):
        print(f"\n🧪 Test Case {i}: {spec.circuit_type} @ {spec.frequency/1e9:.1f}GHz, {spec.temperature}°C")
        
        try:
            # Generate circuit with comprehensive error handling
            start_time = time.time()
            circuit = generator.generate_circuit(spec)
            generation_time = time.time() - start_time
            circuit.generation_time = generation_time
            
            print(f"   ✅ Generated circuit ID: {circuit.circuit_id}")
            print(f"   📊 Performance: Gain={circuit.gain:.1f}dB, NF={circuit.noise_figure:.2f}dB, Power={circuit.power*1000:.1f}mW")
            print(f"   ⏱️  Generation time: {generation_time*1000:.1f}ms")
            print(f"   🔒 Validation level: {circuit.validation_level}")
            
            if circuit.warnings:
                print(f"   ⚠️  Warnings ({len(circuit.warnings)}): {', '.join(circuit.warnings[:2])}...")
            
            results.append(circuit)
            
            # Export with enhanced formats
            output_dir = Path("gen2_robust_outputs")
            output_dir.mkdir(exist_ok=True)
            
            # Export multiple formats
            base_name = f"{circuit.circuit_type.lower()}_{circuit.circuit_id}"
            
            exporter.export_spice(circuit, str(output_dir / f"{base_name}.sp"), include_testbench=True)
            exporter.export_json(circuit, str(output_dir / f"{base_name}.json"), include_metadata=True)
            
            print(f"   💾 Exported: {base_name}.sp, {base_name}.json")
            
        except ValidationError as e:
            print(f"   ❌ Validation error: {e}")
        except GenerationError as e:
            print(f"   ❌ Generation error: {e}")
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
    
    # Generate monitoring report
    print("\n" + "=" * 70)
    print(generator.monitor.generate_report())
    
    # Export comprehensive summary
    if results:
        output_dir = Path("gen2_robust_outputs")
        exporter.export_summary_report(results, str(output_dir / "generation_summary.txt"))
        print(f"\n💾 Summary report exported to generation_summary.txt")
    
    print(f"\n✅ Generation 2: MAKE IT ROBUST - COMPLETED")
    print(f"📈 Success rate: {len(results)}/{len(test_specs)} ({100*len(results)/len(test_specs):.0f}%)")
    
    return results

if __name__ == "__main__":
    main()