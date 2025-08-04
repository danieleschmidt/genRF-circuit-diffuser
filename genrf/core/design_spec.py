"""
Design Specification classes for RF circuit generation.

This module defines the DesignSpec class for specifying circuit requirements
and constraints for the generation process.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import yaml
import json


@dataclass
class DesignSpec:
    """
    Design specification for RF circuit generation.
    
    Defines the requirements, constraints, and objectives for circuit synthesis.
    """
    
    # Circuit identification
    circuit_type: str  # 'LNA', 'Mixer', 'VCO', 'PA', 'Filter'
    name: Optional[str] = None
    description: Optional[str] = None
    
    # Operating conditions
    frequency: float = 2.4e9  # Operating frequency in Hz
    temperature: float = 27.0  # Temperature in Celsius
    supply_voltage: float = 1.2  # Supply voltage in V
    
    # Performance requirements
    gain_min: float = 10.0  # Minimum gain in dB
    gain_max: float = 50.0  # Maximum gain in dB
    nf_max: float = 3.0     # Maximum noise figure in dB
    power_max: float = 50e-3  # Maximum power consumption in W
    
    # Matching requirements
    input_impedance: float = 50.0   # Input impedance in Ohms
    output_impedance: float = 50.0  # Output impedance in Ohms
    s11_max: float = -10.0          # Maximum input return loss in dB
    s22_max: float = -10.0          # Maximum output return loss in dB
    
    # Bandwidth specifications
    bandwidth: Optional[float] = None  # 3dB bandwidth in Hz
    frequency_min: Optional[float] = None  # Minimum frequency in Hz
    frequency_max: Optional[float] = None  # Maximum frequency in Hz
    
    # Linearity specifications
    ip3_min: Optional[float] = None  # Minimum IIP3 in dBm
    p1db_min: Optional[float] = None  # Minimum P1dB in dBm
    
    # Technology constraints
    technology: str = "generic"  # Technology node
    area_max: Optional[float] = None  # Maximum area in mm²
    
    # Additional constraints
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        'gain': 1.0,
        'noise_figure': 1.0,
        'power': 1.0,
        'area': 0.5,
        'linearity': 0.5
    })
    
    def __post_init__(self):
        """Validate and normalize specification after initialization."""
        self._validate_circuit_type()
        self._set_defaults_by_type()
        self._validate_frequency_specs()
        self._validate_performance_specs()
    
    def _validate_circuit_type(self):
        """Validate circuit type and set name if not provided."""
        valid_types = ['LNA', 'Mixer', 'VCO', 'PA', 'Filter', 'Balun', 'Coupler']
        
        if self.circuit_type not in valid_types:
            raise ValueError(f"Invalid circuit_type '{self.circuit_type}'. "
                           f"Must be one of: {valid_types}")
        
        if self.name is None:
            self.name = f"{self.circuit_type}_{self.frequency/1e9:.1f}GHz"
    
    def _set_defaults_by_type(self):
        """Set type-specific default values."""
        defaults = {
            'LNA': {
                'gain_min': 15.0, 'gain_max': 25.0,
                'nf_max': 2.0, 'power_max': 20e-3
            },
            'Mixer': {
                'gain_min': 8.0, 'gain_max': 15.0,
                'nf_max': 8.0, 'power_max': 30e-3
            },
            'VCO': {
                'gain_min': 0.0, 'gain_max': 5.0,
                'nf_max': float('inf'), 'power_max': 50e-3
            },
            'PA': {
                'gain_min': 10.0, 'gain_max': 30.0,
                'nf_max': 10.0, 'power_max': 1.0
            },
            'Filter': {
                'gain_min': -3.0, 'gain_max': 0.0,
                'nf_max': 1.0, 'power_max': 0.0
            }
        }
        
        if self.circuit_type in defaults:
            circuit_defaults = defaults[self.circuit_type]
            
            # Only update if current values are at class defaults
            if self.gain_min == 10.0:  # Class default
                self.gain_min = circuit_defaults['gain_min']
            if self.gain_max == 50.0:  # Class default
                self.gain_max = circuit_defaults['gain_max']
            if self.nf_max == 3.0:  # Class default
                self.nf_max = circuit_defaults['nf_max']
            if self.power_max == 50e-3:  # Class default
                self.power_max = circuit_defaults['power_max']
    
    def _validate_frequency_specs(self):
        """Validate frequency specifications."""
        if self.frequency <= 0:
            raise ValueError("Frequency must be positive")
        
        if self.frequency_min is not None and self.frequency_max is not None:
            if self.frequency_min >= self.frequency_max:
                raise ValueError("frequency_min must be less than frequency_max")
            
            if not (self.frequency_min <= self.frequency <= self.frequency_max):
                raise ValueError("Operating frequency must be within frequency range")
        
        if self.bandwidth is not None:
            if self.bandwidth <= 0:
                raise ValueError("Bandwidth must be positive")
            
            if self.bandwidth > self.frequency:
                raise ValueError("Bandwidth cannot exceed operating frequency")
    
    def _validate_performance_specs(self):
        """Validate performance specifications."""
        if self.gain_min > self.gain_max:
            raise ValueError("gain_min must be less than or equal to gain_max")
        
        if self.nf_max <= 0:
            raise ValueError("Maximum noise figure must be positive")
        
        if self.power_max <= 0:
            raise ValueError("Maximum power must be positive")
        
        if self.supply_voltage <= 0:
            raise ValueError("Supply voltage must be positive")
        
        if self.input_impedance <= 0 or self.output_impedance <= 0:
            raise ValueError("Impedances must be positive")
    
    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> 'DesignSpec':
        """Load design specification from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> 'DesignSpec':
        """Load design specification from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(**data)
    
    def to_yaml(self, filepath: Union[str, Path]) -> None:
        """Save design specification to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def to_json(self, filepath: Union[str, Path], indent: int = 2) -> None:
        """Save design specification to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to dictionary."""
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                data[key] = value
            elif isinstance(value, dict):
                data[key] = value.copy()
            elif isinstance(value, list):
                data[key] = value.copy()
        
        return data
    
    def get_frequency_range(self) -> tuple[float, float]:
        """Get effective frequency range for the specification."""
        if self.frequency_min is not None and self.frequency_max is not None:
            return (self.frequency_min, self.frequency_max)
        elif self.bandwidth is not None:
            f_low = self.frequency - self.bandwidth / 2
            f_high = self.frequency + self.bandwidth / 2
            return (max(0, f_low), f_high)
        else:
            # Default to ±10% around center frequency
            delta = self.frequency * 0.1
            return (self.frequency - delta, self.frequency + delta)
    
    def is_broadband(self) -> bool:
        """Check if this is a broadband specification."""
        f_low, f_high = self.get_frequency_range()
        fractional_bw = (f_high - f_low) / self.frequency
        return fractional_bw > 0.1  # >10% fractional bandwidth
    
    def get_wavelength(self) -> float:
        """Get wavelength at operating frequency (assuming c=3e8 m/s)."""
        return 3e8 / self.frequency
    
    def check_feasibility(self) -> List[str]:
        """Check specification feasibility and return list of warnings."""
        warnings = []
        
        # Check gain vs noise figure tradeoff
        if self.circuit_type == 'LNA':
            if self.gain_min > 20 and self.nf_max < 1.0:
                warnings.append("High gain with very low noise figure may be challenging")
        
        # Check power vs performance tradeoffs
        if self.power_max < 1e-3 and self.gain_min > 15:
            warnings.append("Low power budget may limit achievable gain")
        
        # Check frequency-dependent limitations
        if self.frequency > 100e9:
            warnings.append("mmWave frequencies may have limited technology support")
        
        # Check linearity requirements
        if self.ip3_min is not None and self.ip3_min > 10:
            if self.power_max < 100e-3:
                warnings.append("High linearity with low power may be difficult")
        
        return warnings
    
    def __str__(self) -> str:
        """String representation of the design specification."""
        lines = [
            f"DesignSpec: {self.name}",
            f"  Type: {self.circuit_type}",
            f"  Frequency: {self.frequency/1e9:.2f} GHz",
            f"  Gain: {self.gain_min}-{self.gain_max} dB",
            f"  NF: ≤{self.nf_max} dB",
            f"  Power: ≤{self.power_max*1000:.1f} mW",
            f"  Technology: {self.technology}"
        ]
        
        if self.description:
            lines.insert(1, f"  Description: {self.description}")
        
        return "\n".join(lines)


# Predefined specifications for common circuits
class CommonSpecs:
    """Collection of common RF circuit specifications."""
    
    @staticmethod
    def wifi_lna() -> DesignSpec:
        """2.4 GHz WiFi LNA specification."""
        return DesignSpec(
            circuit_type="LNA",
            name="WiFi_LNA_2G4",
            description="Low noise amplifier for 2.4 GHz WiFi",
            frequency=2.4e9,
            gain_min=15.0,
            gain_max=20.0,
            nf_max=1.5,
            power_max=10e-3,
            bandwidth=100e6,
            technology="TSMC65nm"
        )
    
    @staticmethod
    def cellular_lna() -> DesignSpec:
        """Cellular LNA specification."""
        return DesignSpec(
            circuit_type="LNA",
            name="Cellular_LNA_1G8",
            description="Low noise amplifier for cellular band",
            frequency=1.8e9,
            gain_min=18.0,
            gain_max=22.0,
            nf_max=1.2,
            power_max=15e-3,
            bandwidth=200e6,
            ip3_min=0.0,
            technology="TSMC65nm"
        )
    
    @staticmethod
    def bluetooth_mixer() -> DesignSpec:
        """Bluetooth mixer specification."""
        return DesignSpec(
            circuit_type="Mixer",
            name="Bluetooth_Mixer_2G4",
            description="Down-conversion mixer for Bluetooth",
            frequency=2.4e9,
            gain_min=8.0,
            gain_max=12.0,
            nf_max=8.0,
            power_max=20e-3,
            technology="TSMC65nm"
        )
    
    @staticmethod
    def mmwave_pa() -> DesignSpec:
        """mmWave power amplifier specification."""
        return DesignSpec(
            circuit_type="PA",
            name="mmWave_PA_28G",
            description="Power amplifier for 28 GHz mmWave",
            frequency=28e9,
            gain_min=15.0,
            gain_max=25.0,
            nf_max=6.0,
            power_max=500e-3,
            p1db_min=15.0,
            technology="TSMC28nm"
        )
    
    @staticmethod
    def ism_vco() -> DesignSpec:
        """ISM band VCO specification."""
        return DesignSpec(
            circuit_type="VCO",
            name="ISM_VCO_2G4",
            description="Voltage controlled oscillator for ISM band",
            frequency=2.4e9,
            gain_min=0.0,
            gain_max=5.0,
            nf_max=float('inf'),
            power_max=30e-3,
            frequency_min=2.35e9,
            frequency_max=2.45e9,
            technology="TSMC65nm"
        )