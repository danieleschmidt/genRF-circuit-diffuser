"""
Input validation and sanitization for GenRF.

This module provides comprehensive validation for all inputs to ensure
robustness and security of the circuit generation pipeline.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path

from .exceptions import ValidationError
from .design_spec import DesignSpec
from .technology import TechnologyFile

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Comprehensive input validator for GenRF components.
    
    Validates and sanitizes all inputs to prevent errors and security issues.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator.
        
        Args:
            strict_mode: Enable strict validation (recommended for production)
        """
        self.strict_mode = strict_mode
        
        # Define validation rules
        self.frequency_range = (1e6, 1e12)  # 1 MHz to 1 THz
        self.voltage_range = (0.1, 10.0)    # 0.1V to 10V
        self.temperature_range = (-273.15, 200.0)  # Physical limits
        self.power_range = (1e-9, 10.0)     # 1 nW to 10W
        self.gain_range = (-50.0, 100.0)    # -50 to 100 dB
        self.nf_range = (0.1, 50.0)         # 0.1 to 50 dB
        
        # File path security patterns
        self.dangerous_patterns = [
            r'\.\./',    # Directory traversal
            r'~/',       # Home directory access
            r'/etc/',    # System directories
            r'/proc/',   # Process filesystem
            r'/sys/',    # System filesystem
            r'\\',       # Windows path separators
        ]
        
        logger.info(f"InputValidator initialized (strict_mode={strict_mode})")
    
    def validate_design_spec(self, spec: DesignSpec) -> List[str]:
        """
        Validate design specification.
        
        Args:
            spec: Design specification to validate
            
        Returns:
            List of validation warnings (empty if all valid)
            
        Raises:
            ValidationError: If critical validation fails
        """
        warnings = []
        
        try:
            # Validate circuit type
            self._validate_circuit_type(spec.circuit_type)
            
            # Validate frequencies
            self._validate_frequency(spec.frequency, "frequency")
            
            if spec.frequency_min is not None:
                self._validate_frequency(spec.frequency_min, "frequency_min")
            
            if spec.frequency_max is not None:
                self._validate_frequency(spec.frequency_max, "frequency_max")
            
            if spec.bandwidth is not None:
                self._validate_frequency(spec.bandwidth, "bandwidth")
            
            # Validate electrical parameters
            self._validate_voltage(spec.supply_voltage, "supply_voltage")
            self._validate_temperature(spec.temperature, "temperature")
            
            # Validate performance requirements
            if not self.gain_range[0] <= spec.gain_min <= self.gain_range[1]:
                raise ValidationError(f"gain_min {spec.gain_min} outside valid range {self.gain_range}")
            
            if not self.gain_range[0] <= spec.gain_max <= self.gain_range[1]:
                raise ValidationError(f"gain_max {spec.gain_max} outside valid range {self.gain_range}")
            
            if spec.gain_min > spec.gain_max:
                raise ValidationError("gain_min cannot be greater than gain_max")
            
            if not self.nf_range[0] <= spec.nf_max <= self.nf_range[1]:
                raise ValidationError(f"nf_max {spec.nf_max} outside valid range {self.nf_range}")
            
            if not self.power_range[0] <= spec.power_max <= self.power_range[1]:
                raise ValidationError(f"power_max {spec.power_max} outside valid range {self.power_range}")
            
            # Validate impedances
            if spec.input_impedance <= 0 or spec.input_impedance > 1000:
                raise ValidationError(f"input_impedance {spec.input_impedance} outside reasonable range (0, 1000)")
            
            if spec.output_impedance <= 0 or spec.output_impedance > 1000:
                raise ValidationError(f"output_impedance {spec.output_impedance} outside reasonable range (0, 1000)")
            
            # Check for feasibility warnings
            feasibility_warnings = spec.check_feasibility()
            warnings.extend(feasibility_warnings)
            
            # Technology-specific validation
            if hasattr(spec, 'technology') and spec.technology:
                tech_warnings = self._validate_technology_compatibility(spec)
                warnings.extend(tech_warnings)
            
            logger.info(f"Design specification validation passed with {len(warnings)} warnings")
            
        except Exception as e:
            logger.error(f"Design specification validation failed: {e}")
            raise ValidationError(f"Design specification validation failed: {e}")
        
        return warnings
    
    def validate_parameters(self, parameters: Dict[str, float], technology: Optional[TechnologyFile] = None) -> List[str]:
        """
        Validate circuit parameters.
        
        Args:
            parameters: Dictionary of parameter name -> value
            technology: Optional technology file for constraint checking
            
        Returns:
            List of validation warnings
            
        Raises:
            ValidationError: If critical validation fails
        """
        warnings = []
        
        for param_name, param_value in parameters.items():
            try:
                # Basic type and value checks
                if not isinstance(param_value, (int, float)):
                    raise ValidationError(f"Parameter {param_name} must be numeric, got {type(param_value)}")
                
                if not np.isfinite(param_value):
                    raise ValidationError(f"Parameter {param_name} must be finite, got {param_value}")
                
                if param_value < 0:
                    raise ValidationError(f"Parameter {param_name} cannot be negative, got {param_value}")
                
                # Parameter-specific validation
                if '_w' in param_name:  # Width parameter
                    if param_value < 1e-9 or param_value > 1e-3:
                        warnings.append(f"Width parameter {param_name}={param_value:.2e} outside typical range (1nm-1mm)")
                
                elif '_l' in param_name:  # Length parameter
                    if param_value < 1e-9 or param_value > 1e-3:
                        warnings.append(f"Length parameter {param_name}={param_value:.2e} outside typical range (1nm-1mm)")
                
                elif '_r' in param_name:  # Resistance parameter
                    if param_value < 0.1 or param_value > 1e9:
                        warnings.append(f"Resistance parameter {param_name}={param_value:.2e} outside typical range (0.1Ω-1GΩ)")
                
                elif '_c' in param_name:  # Capacitance parameter
                    if param_value < 1e-18 or param_value > 1e-6:
                        warnings.append(f"Capacitance parameter {param_name}={param_value:.2e} outside typical range (1aF-1μF)")
                
                elif '_l' in param_name and '_w' not in param_name:  # Inductance parameter
                    if param_value < 1e-12 or param_value > 1e-3:
                        warnings.append(f"Inductance parameter {param_name}={param_value:.2e} outside typical range (1pH-1mH)")
                
                # Technology-specific validation
                if technology is not None:
                    tech_warnings = self._validate_parameter_against_technology(param_name, param_value, technology)
                    warnings.extend(tech_warnings)
                
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(f"Unexpected error validating parameter {param_name}: {e}")
        
        logger.info(f"Parameter validation completed with {len(warnings)} warnings")
        return warnings
    
    def validate_file_path(self, filepath: Union[str, Path], check_exists: bool = True) -> Path:
        """
        Validate and sanitize file path.
        
        Args:
            filepath: File path to validate
            check_exists: Whether to check if file exists
            
        Returns:
            Validated and sanitized Path object
            
        Raises:
            ValidationError: If path is invalid or dangerous
        """
        try:
            # Convert to Path object
            path = Path(filepath)
            
            # Security checks
            path_str = str(path)
            for pattern in self.dangerous_patterns:
                if re.search(pattern, path_str):
                    raise ValidationError(f"Potentially dangerous path pattern detected: {pattern}")
            
            # Check for extremely long paths
            if len(path_str) > 4096:
                raise ValidationError("File path too long (>4096 characters)")
            
            # Resolve path to prevent symlink attacks
            try:
                resolved_path = path.resolve()
            except (OSError, RuntimeError) as e:
                raise ValidationError(f"Cannot resolve path: {e}")
            
            # Check existence if required
            if check_exists and not resolved_path.exists():
                raise ValidationError(f"File does not exist: {resolved_path}")
            
            return resolved_path
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Path validation failed: {e}")
    
    def sanitize_string(self, text: str, max_length: int = 1000, allow_special: bool = False) -> str:
        """
        Sanitize string input.
        
        Args:
            text: Text to sanitize
            max_length: Maximum allowed length
            allow_special: Whether to allow special characters
            
        Returns:
            Sanitized string
            
        Raises:
            ValidationError: If string is invalid
        """
        if not isinstance(text, str):
            raise ValidationError(f"Expected string, got {type(text)}")
        
        if len(text) > max_length:
            raise ValidationError(f"String too long: {len(text)} > {max_length}")
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        
        if not allow_special:
            # Keep only alphanumeric, spaces, and safe punctuation
            sanitized = re.sub(r'[^a-zA-Z0-9\s\-_\.\(\)\[\]{}:,;]', '', sanitized)
        
        return sanitized.strip()
    
    def validate_model_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate AI model checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint metadata
            
        Raises:
            ValidationError: If checkpoint is invalid
        """
        try:
            import torch
            
            path = self.validate_file_path(checkpoint_path, check_exists=True)
            
            # Try to load checkpoint metadata
            try:
                checkpoint = torch.load(path, map_location='cpu')
            except Exception as e:
                raise ValidationError(f"Cannot load checkpoint: {e}")
            
            # Validate checkpoint structure
            required_keys = ['model_state_dict', 'model_config']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise ValidationError(f"Checkpoint missing required keys: {missing_keys}")
            
            # Extract metadata
            metadata = {
                'path': str(path),
                'epoch': checkpoint.get('epoch', 0),
                'model_config': checkpoint.get('model_config', {}),
                'file_size': path.stat().st_size
            }
            
            logger.info(f"Checkpoint validation passed: {path}")
            return metadata
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Checkpoint validation failed: {e}")
    
    def _validate_circuit_type(self, circuit_type: str):
        """Validate circuit type."""
        valid_types = ['LNA', 'Mixer', 'VCO', 'PA', 'Filter', 'Balun', 'Coupler', 'Switch', 'Attenuator']
        
        if not isinstance(circuit_type, str):
            raise ValidationError(f"Circuit type must be string, got {type(circuit_type)}")
        
        if circuit_type not in valid_types:
            raise ValidationError(f"Invalid circuit type '{circuit_type}'. Valid types: {valid_types}")
    
    def _validate_frequency(self, frequency: float, field_name: str):
        """Validate frequency parameter."""
        if not isinstance(frequency, (int, float)):
            raise ValidationError(f"{field_name} must be numeric, got {type(frequency)}")
        
        if not np.isfinite(frequency):
            raise ValidationError(f"{field_name} must be finite, got {frequency}")
        
        if not self.frequency_range[0] <= frequency <= self.frequency_range[1]:
            raise ValidationError(f"{field_name} {frequency} outside valid range {self.frequency_range}")
    
    def _validate_voltage(self, voltage: float, field_name: str):
        """Validate voltage parameter."""
        if not isinstance(voltage, (int, float)):
            raise ValidationError(f"{field_name} must be numeric, got {type(voltage)}")
        
        if not np.isfinite(voltage):
            raise ValidationError(f"{field_name} must be finite, got {voltage}")
        
        if not self.voltage_range[0] <= voltage <= self.voltage_range[1]:
            raise ValidationError(f"{field_name} {voltage} outside valid range {self.voltage_range}")
    
    def _validate_temperature(self, temperature: float, field_name: str):
        """Validate temperature parameter."""
        if not isinstance(temperature, (int, float)):
            raise ValidationError(f"{field_name} must be numeric, got {type(temperature)}")
        
        if not np.isfinite(temperature):
            raise ValidationError(f"{field_name} must be finite, got {temperature}")
        
        if not self.temperature_range[0] <= temperature <= self.temperature_range[1]:
            raise ValidationError(f"{field_name} {temperature}°C outside valid range {self.temperature_range}")
    
    def _validate_technology_compatibility(self, spec: DesignSpec) -> List[str]:
        """Check technology compatibility with specification."""
        warnings = []
        
        # Check frequency vs technology capability
        if spec.frequency > 100e9 and '65nm' in spec.technology:
            warnings.append("65nm technology may not be optimal for >100 GHz operation")
        
        if spec.frequency < 1e9 and '7nm' in spec.technology:
            warnings.append("Advanced process nodes may be overkill for <1 GHz operation")
        
        # Check power vs technology
        if spec.power_max > 1.0 and 'nm' in spec.technology:
            warnings.append("High power requirements may need special consideration for advanced nodes")
        
        return warnings
    
    def _validate_parameter_against_technology(
        self, 
        param_name: str, 
        param_value: float, 
        technology: TechnologyFile
    ) -> List[str]:
        """Validate parameter against technology constraints."""
        warnings = []
        
        try:
            if '_w' in param_name and '_l' in param_name:
                # Extract device type from parameter name
                device_type = param_name.split('_')[0]
                
                if '_w' in param_name:
                    # Width parameter
                    device_model = technology.get_device_model('nmos')  # Default to nmos
                    if device_model and not technology.validate_device_size('nmos', param_value, device_model.length_min):
                        warnings.append(f"Width {param_name}={param_value:.2e} may violate technology constraints")
                
            elif '_r' in param_name:
                # Resistor parameter
                resistor_model = technology.get_passive_model('resistor')
                if resistor_model and not technology.validate_passive_value('resistor', param_value):
                    warnings.append(f"Resistance {param_name}={param_value:.2e} outside technology range")
            
            elif '_c' in param_name:
                # Capacitor parameter
                cap_model = technology.get_passive_model('capacitor')
                if cap_model and not technology.validate_passive_value('capacitor', param_value):
                    warnings.append(f"Capacitance {param_name}={param_value:.2e} outside technology range")
        
        except Exception as e:
            logger.debug(f"Technology validation warning for {param_name}: {e}")
        
        return warnings


# Global validator instance
default_validator = InputValidator()


def validate_design_spec(spec: DesignSpec) -> List[str]:
    """Convenience function for design spec validation."""
    return default_validator.validate_design_spec(spec)


def validate_parameters(parameters: Dict[str, float], technology: Optional[TechnologyFile] = None) -> List[str]:
    """Convenience function for parameter validation."""
    return default_validator.validate_parameters(parameters, technology)


def validate_file_path(filepath: Union[str, Path], check_exists: bool = True) -> Path:
    """Convenience function for file path validation."""
    return default_validator.validate_file_path(filepath, check_exists)


def sanitize_string(text: str, max_length: int = 1000, allow_special: bool = False) -> str:
    """Convenience function for string sanitization."""
    return default_validator.sanitize_string(text, max_length, allow_special)