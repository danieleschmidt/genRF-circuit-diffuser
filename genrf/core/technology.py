"""
Technology file management for RF circuit generation.

This module defines the TechnologyFile class for managing PDK-specific
constraints, device models, and design rules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import yaml
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class DeviceModel:
    """Device model parameters for a specific technology."""
    
    name: str
    device_type: str  # 'nmos', 'pmos', 'bjt', 'diode', etc.
    model_file: Optional[str] = None
    
    # Physical constraints
    width_min: float = 1e-6      # Minimum width in meters
    width_max: float = 1000e-6   # Maximum width in meters
    length_min: float = 28e-9    # Minimum length in meters
    length_max: float = 10e-6    # Maximum length in meters
    
    # Electrical parameters
    threshold_voltage: float = 0.4  # Threshold voltage in V
    oxide_thickness: float = 2e-9   # Gate oxide thickness in m
    mobility: float = 400e-4        # Mobility in m²/V·s
    
    # Process parameters
    process_corners: List[str] = field(default_factory=lambda: ['TT', 'FF', 'SS', 'FS', 'SF'])
    temperature_range: Tuple[float, float] = (-40.0, 125.0)  # Temperature range in Celsius
    
    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate device model parameters."""
        if self.width_min >= self.width_max:
            raise ValueError("width_min must be less than width_max")
        if self.length_min >= self.length_max:
            raise ValueError("length_min must be less than length_max")


@dataclass
class PassiveModel:
    """Passive component model parameters."""
    
    name: str
    component_type: str  # 'resistor', 'capacitor', 'inductor'
    
    # Value constraints
    value_min: float = 1e-15     # Minimum component value
    value_max: float = 1e6       # Maximum component value
    
    # Physical constraints
    area_factor: float = 1.0     # Area scaling factor
    quality_factor: float = 10.0 # Quality factor
    
    # Parasitic parameters
    parasitic_r: float = 0.0     # Series resistance
    parasitic_c: float = 0.0     # Parallel capacitance
    parasitic_l: float = 0.0     # Series inductance
    
    # Model parameters
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class DesignRules:
    """Design rules and constraints for a technology."""
    
    # Minimum feature sizes
    min_width: float = 100e-9       # Minimum feature width
    min_spacing: float = 100e-9     # Minimum spacing
    min_via_size: float = 80e-9     # Minimum via size
    
    # Metal layer constraints
    metal_layers: int = 6           # Number of metal layers
    metal_thickness: List[float] = field(default_factory=lambda: [0.13e-6, 0.14e-6, 0.28e-6, 0.28e-6, 0.8e-6, 3.0e-6])
    metal_resistivity: List[float] = field(default_factory=lambda: [2.7e-8] * 6)
    
    # Well and substrate
    well_spacing: float = 1e-6      # Minimum well spacing
    substrate_resistivity: float = 10.0  # Substrate resistivity in Ω·cm
    
    # Matching constraints
    matching_tolerance: float = 0.01  # Matching tolerance (1%)
    thermal_gradient: float = 1e-6   # Thermal gradient effect
    
    # High-frequency constraints
    skin_depth_factor: float = 1.0   # Skin depth scaling
    dielectric_constant: float = 3.9 # Dielectric constant
    loss_tangent: float = 0.02       # Loss tangent


@dataclass
class TechnologyFile:
    """
    Technology file containing PDK-specific information for circuit generation.
    
    Manages device models, design rules, and constraints for a specific
    semiconductor process technology.
    """
    
    name: str
    process_node: str              # e.g., "65nm", "28nm", "7nm"
    foundry: str                   # e.g., "TSMC", "GlobalFoundries", "Intel"
    version: str = "1.0"
    
    # Operating conditions
    supply_voltage_nominal: float = 1.2    # Nominal supply voltage
    supply_voltage_range: Tuple[float, float] = (1.08, 1.32)  # Supply voltage range
    temperature_nominal: float = 27.0      # Nominal temperature
    
    # Device models
    device_models: Dict[str, DeviceModel] = field(default_factory=dict)
    passive_models: Dict[str, PassiveModel] = field(default_factory=dict)
    
    # Design rules
    design_rules: DesignRules = field(default_factory=DesignRules)
    
    # Model files
    model_library_path: Optional[str] = None
    corner_files: Dict[str, str] = field(default_factory=dict)
    
    # Additional constraints
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set up default models if none provided."""
        if not self.device_models:
            self._create_default_device_models()
        
        if not self.passive_models:
            self._create_default_passive_models()
    
    def _create_default_device_models(self):
        """Create default device models for common devices."""
        # Default NMOS model
        self.device_models['nmos'] = DeviceModel(
            name='nch',
            device_type='nmos',
            width_min=240e-9,
            width_max=1000e-6,
            length_min=float(self.process_node.rstrip('nm')) * 1e-9,
            length_max=10e-6,
            threshold_voltage=0.4,
            mobility=400e-4
        )
        
        # Default PMOS model
        self.device_models['pmos'] = DeviceModel(
            name='pch',
            device_type='pmos',
            width_min=240e-9,
            width_max=1000e-6,
            length_min=float(self.process_node.rstrip('nm')) * 1e-9,
            length_max=10e-6,
            threshold_voltage=-0.4,
            mobility=150e-4
        )
    
    def _create_default_passive_models(self):
        """Create default passive component models."""
        # Resistor model
        self.passive_models['resistor'] = PassiveModel(
            name='res',
            component_type='resistor',
            value_min=1.0,
            value_max=1e6,
            quality_factor=100.0
        )
        
        # Capacitor model (MIM cap)
        self.passive_models['capacitor'] = PassiveModel(
            name='mimcap',
            component_type='capacitor',
            value_min=1e-15,
            value_max=100e-12,
            quality_factor=50.0
        )
        
        # Inductor model
        self.passive_models['inductor'] = PassiveModel(
            name='ind',
            component_type='inductor',
            value_min=100e-12,
            value_max=100e-9,
            quality_factor=20.0
        )
    
    @classmethod
    def default(cls) -> 'TechnologyFile':
        """Create a default generic technology file."""
        return cls(
            name="Generic",
            process_node="65nm",
            foundry="Generic",
            supply_voltage_nominal=1.2
        )
    
    @classmethod
    def tsmc65nm(cls) -> 'TechnologyFile':
        """Create TSMC 65nm technology file."""
        tech = cls(
            name="TSMC65nm",
            process_node="65nm",
            foundry="TSMC",
            supply_voltage_nominal=1.2,
            supply_voltage_range=(1.08, 1.32)
        )
        
        # Update device models for 65nm
        tech.device_models['nmos'].length_min = 60e-9
        tech.device_models['pmos'].length_min = 60e-9
        
        return tech
    
    @classmethod
    def tsmc28nm(cls) -> 'TechnologyFile':
        """Create TSMC 28nm technology file."""
        tech = cls(
            name="TSMC28nm",
            process_node="28nm",
            foundry="TSMC",
            supply_voltage_nominal=1.0,
            supply_voltage_range=(0.9, 1.1)
        )
        
        # Update device models for 28nm
        tech.device_models['nmos'].length_min = 28e-9
        tech.device_models['nmos'].threshold_voltage = 0.35
        tech.device_models['pmos'].length_min = 28e-9
        tech.device_models['pmos'].threshold_voltage = -0.35
        
        # Update design rules
        tech.design_rules.min_width = 28e-9
        tech.design_rules.min_spacing = 56e-9
        
        return tech
    
    @classmethod
    def gf22fdx(cls) -> 'TechnologyFile':
        """Create GlobalFoundries 22FDX technology file."""
        tech = cls(
            name="GF22FDX",
            process_node="22nm",
            foundry="GlobalFoundries",
            supply_voltage_nominal=0.8,
            supply_voltage_range=(0.72, 0.88)
        )
        
        # Update for 22FDX specifics
        tech.device_models['nmos'].length_min = 22e-9
        tech.device_models['nmos'].threshold_voltage = 0.3
        tech.device_models['pmos'].length_min = 22e-9
        tech.device_models['pmos'].threshold_voltage = -0.3
        
        # FD-SOI specific parameters
        tech.constraints['body_bias_range'] = (-2.0, 2.0)  # Body bias range for FD-SOI
        tech.constraints['is_fdsoi'] = True
        
        return tech
    
    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> 'TechnologyFile':
        """Load technology file from YAML."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert nested dictionaries to dataclass objects
        if 'device_models' in data:
            device_models = {}
            for name, model_data in data['device_models'].items():
                device_models[name] = DeviceModel(**model_data)
            data['device_models'] = device_models
        
        if 'passive_models' in data:
            passive_models = {}
            for name, model_data in data['passive_models'].items():
                passive_models[name] = PassiveModel(**model_data)
            data['passive_models'] = passive_models
        
        if 'design_rules' in data:
            data['design_rules'] = DesignRules(**data['design_rules'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> 'TechnologyFile':
        """Load technology file from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert nested dictionaries (similar to YAML)
        if 'device_models' in data:
            device_models = {}
            for name, model_data in data['device_models'].items():
                device_models[name] = DeviceModel(**model_data)
            data['device_models'] = device_models
        
        if 'passive_models' in data:
            passive_models = {}
            for name, model_data in data['passive_models'].items():
                passive_models[name] = PassiveModel(**model_data)
            data['passive_models'] = passive_models
        
        if 'design_rules' in data:
            data['design_rules'] = DesignRules(**data['design_rules'])
        
        return cls(**data)
    
    def to_yaml(self, filepath: Union[str, Path]) -> None:
        """Save technology file to YAML."""
        data = self._to_dict()
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
    
    def to_json(self, filepath: Union[str, Path], indent: int = 2) -> None:
        """Save technology file to JSON."""
        data = self._to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert technology file to dictionary for serialization."""
        data = {
            'name': self.name,
            'process_node': self.process_node,
            'foundry': self.foundry,
            'version': self.version,
            'supply_voltage_nominal': self.supply_voltage_nominal,
            'supply_voltage_range': self.supply_voltage_range,
            'temperature_nominal': self.temperature_nominal,
            'model_library_path': self.model_library_path,
            'corner_files': self.corner_files.copy(),
            'constraints': self.constraints.copy()
        }
        
        # Convert device models
        data['device_models'] = {}
        for name, model in self.device_models.items():
            data['device_models'][name] = {
                'name': model.name,
                'device_type': model.device_type,
                'model_file': model.model_file,
                'width_min': model.width_min,
                'width_max': model.width_max,
                'length_min': model.length_min,
                'length_max': model.length_max,
                'threshold_voltage': model.threshold_voltage,
                'oxide_thickness': model.oxide_thickness,
                'mobility': model.mobility,
                'process_corners': model.process_corners.copy(),
                'temperature_range': model.temperature_range,
                'model_params': model.model_params.copy()
            }
        
        # Convert passive models
        data['passive_models'] = {}
        for name, model in self.passive_models.items():
            data['passive_models'][name] = {
                'name': model.name,
                'component_type': model.component_type,
                'value_min': model.value_min,
                'value_max': model.value_max,
                'area_factor': model.area_factor,
                'quality_factor': model.quality_factor,
                'parasitic_r': model.parasitic_r,
                'parasitic_c': model.parasitic_c,
                'parasitic_l': model.parasitic_l,
                'model_params': model.model_params.copy()
            }
        
        # Convert design rules
        data['design_rules'] = {
            'min_width': self.design_rules.min_width,
            'min_spacing': self.design_rules.min_spacing,
            'min_via_size': self.design_rules.min_via_size,
            'metal_layers': self.design_rules.metal_layers,
            'metal_thickness': self.design_rules.metal_thickness.copy(),
            'metal_resistivity': self.design_rules.metal_resistivity.copy(),
            'well_spacing': self.design_rules.well_spacing,
            'substrate_resistivity': self.design_rules.substrate_resistivity,
            'matching_tolerance': self.design_rules.matching_tolerance,
            'thermal_gradient': self.design_rules.thermal_gradient,
            'skin_depth_factor': self.design_rules.skin_depth_factor,
            'dielectric_constant': self.design_rules.dielectric_constant,
            'loss_tangent': self.design_rules.loss_tangent
        }
        
        return data
    
    def get_device_model(self, device_type: str) -> Optional[DeviceModel]:
        """Get device model by type."""
        return self.device_models.get(device_type)
    
    def get_passive_model(self, component_type: str) -> Optional[PassiveModel]:
        """Get passive component model by type."""
        return self.passive_models.get(component_type)
    
    def validate_device_size(self, device_type: str, width: float, length: float) -> bool:
        """Validate device dimensions against design rules."""
        model = self.get_device_model(device_type)
        if model is None:
            logger.warning(f"No model found for device type: {device_type}")
            return False
        
        if not (model.width_min <= width <= model.width_max):
            return False
        
        if not (model.length_min <= length <= model.length_max):
            return False
        
        return True
    
    def validate_passive_value(self, component_type: str, value: float) -> bool:
        """Validate passive component value against constraints."""
        model = self.get_passive_model(component_type)
        if model is None:
            logger.warning(f"No model found for component type: {component_type}")
            return False
        
        return model.value_min <= value <= model.value_max
    
    def get_frequency_range(self) -> Tuple[float, float]:
        """Get recommended frequency range for this technology."""
        # Estimate based on process node
        node_nm = float(self.process_node.rstrip('nm'))
        
        # Rough estimation: smaller nodes support higher frequencies
        f_max = 300e9 / node_nm  # Very rough approximation
        f_min = 100e6  # Minimum practical RF frequency
        
        return (f_min, f_max)
    
    def estimate_area(self, components: Dict[str, Dict[str, float]]) -> float:
        """Estimate total circuit area in mm²."""
        total_area = 0.0
        
        for comp_name, params in components.items():
            if 'width' in params and 'length' in params:
                # Active device area
                area = params['width'] * params['length']
                total_area += area
            elif 'value' in params:
                # Passive component area (rough estimation)
                total_area += 100e-12  # 100 μm² per passive
        
        # Add routing overhead (factor of 3-5)
        total_area *= 4.0
        
        # Convert to mm²
        return total_area * 1e6
    
    @property
    def model_file(self) -> str:
        """Get the model file path for SPICE simulation."""
        if self.model_library_path:
            return self.model_library_path
        
        # Generate default model file name based on technology
        return f"{self.foundry.lower()}_{self.process_node}_models.lib"
    
    @property
    def nmos_model(self) -> str:
        """Get the NMOS model name for SPICE netlists."""
        nmos = self.get_device_model('nmos')
        return nmos.name if nmos else 'nch'
    
    @property
    def pmos_model(self) -> str:
        """Get the PMOS model name for SPICE netlists."""
        pmos = self.get_device_model('pmos')
        return pmos.name if pmos else 'pch'
    
    def get_default(self) -> 'TechnologyFile':
        """Compatibility method - returns default technology."""
        return TechnologyFile.default()
    
    def __str__(self) -> str:
        """String representation of technology file."""
        lines = [
            f"TechnologyFile: {self.name}",
            f"  Process: {self.foundry} {self.process_node}",
            f"  Supply: {self.supply_voltage_nominal}V",
            f"  Devices: {list(self.device_models.keys())}",
            f"  Passives: {list(self.passive_models.keys())}"
        ]
        
        return "\n".join(lines)