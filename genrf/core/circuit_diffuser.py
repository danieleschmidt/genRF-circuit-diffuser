"""
CircuitDiffuser - Main class for RF circuit generation using AI models.

This module implements the core CircuitDiffuser class that orchestrates
the circuit generation pipeline using cycle-consistent GANs and diffusion models.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import warnings

import torch
import numpy as np
from dataclasses import dataclass

from .design_spec import DesignSpec
from .technology import TechnologyFile
from .models import CycleGAN, DiffusionModel
from .simulation import SPICEEngine
from .optimization import BayesianOptimizer
from .export import CodeExporter

logger = logging.getLogger(__name__)


@dataclass
class CircuitResult:
    """Result of circuit generation with performance metrics."""
    netlist: str
    parameters: Dict[str, float]
    performance: Dict[str, float]
    topology: str
    technology: str
    generation_time: float
    spice_valid: bool
    
    @property
    def gain(self) -> float:
        """Circuit gain in dB."""
        return self.performance.get('gain_db', 0.0)
    
    @property
    def nf(self) -> float:
        """Noise figure in dB.""" 
        return self.performance.get('noise_figure_db', float('inf'))
    
    @property
    def power(self) -> float:
        """Power consumption in W."""
        return self.performance.get('power_w', 0.0)
    
    def export_skill(self, filepath: Union[str, Path]) -> None:
        """Export circuit to Cadence SKILL format."""
        exporter = CodeExporter()
        exporter.export_skill(self, filepath)
    
    def export_verilog_a(self, filepath: Union[str, Path]) -> None:
        """Export circuit to Verilog-A format."""
        exporter = CodeExporter()
        exporter.export_verilog_a(self, filepath)


class CircuitDiffuser:
    """
    Main class for AI-powered RF circuit generation.
    
    Combines cycle-consistent GANs for topology generation with diffusion models
    for parameter optimization, using SPICE-in-the-loop validation.
    """
    
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        spice_engine: str = "ngspice",
        technology: Optional[TechnologyFile] = None,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize CircuitDiffuser.
        
        Args:
            checkpoint: Path to pre-trained model checkpoint
            spice_engine: SPICE simulator to use ('ngspice', 'xyce', 'spectre')
            technology: Technology file for PDK-specific constraints
            device: Device to run models on ('cpu', 'cuda', 'auto')
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self._setup_logging()
        
        # Device selection
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing CircuitDiffuser on device: {self.device}")
        
        # Initialize components
        self.technology = technology or TechnologyFile.default()
        self.spice_engine = SPICEEngine(engine=spice_engine)
        
        # Load or initialize models
        self._load_models(checkpoint)
        
        # Initialize optimizer
        self.optimizer = BayesianOptimizer()
        
        logger.info("CircuitDiffuser initialized successfully")
    
    def _setup_logging(self) -> None:
        """Configure logging for the circuit diffuser."""
        if self.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def _load_models(self, checkpoint: Optional[str]) -> None:
        """Load pre-trained models or initialize new ones."""
        try:
            if checkpoint and Path(checkpoint).exists():
                logger.info(f"Loading models from checkpoint: {checkpoint}")
                checkpoint_data = torch.load(checkpoint, map_location=self.device)
                
                self.topology_model = CycleGAN().to(self.device)
                self.topology_model.load_state_dict(checkpoint_data['topology_model'])
                
                self.diffusion_model = DiffusionModel().to(self.device)
                self.diffusion_model.load_state_dict(checkpoint_data['diffusion_model'])
                
                logger.info("Models loaded successfully from checkpoint")
            else:
                if checkpoint:
                    logger.warning(f"Checkpoint not found: {checkpoint}. Using default models.")
                
                logger.info("Initializing default models")
                self.topology_model = CycleGAN().to(self.device)
                self.diffusion_model = DiffusionModel().to(self.device)
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.info("Falling back to default model initialization")
            self.topology_model = CycleGAN().to(self.device)
            self.diffusion_model = DiffusionModel().to(self.device)
    
    def generate(
        self,
        spec: DesignSpec,
        n_candidates: int = 10,
        optimization_steps: int = 20,
        validate_spice: bool = True
    ) -> CircuitResult:
        """
        Generate an optimized RF circuit based on design specification.
        
        Args:
            spec: Design specification with requirements and constraints
            n_candidates: Number of candidate topologies to generate
            optimization_steps: Number of optimization iterations
            validate_spice: Whether to validate with SPICE simulation
            
        Returns:
            CircuitResult with optimized circuit and performance metrics
        """
        start_time = time.time()
        logger.info(f"Starting circuit generation for {spec.circuit_type}")
        
        try:
            # Step 1: Generate topology candidates
            logger.info(f"Generating {n_candidates} topology candidates")
            topologies = self._generate_topologies(spec, n_candidates)
            
            # Step 2: Optimize parameters for each topology
            best_result = None
            best_performance = float('-inf')
            
            for i, topology in enumerate(topologies):
                logger.info(f"Optimizing topology {i+1}/{len(topologies)}")
                
                # Generate initial parameters
                parameters = self._generate_parameters(topology, spec)
                
                # Optimize with Bayesian optimization
                optimized_params, performance = self._optimize_parameters(
                    topology, parameters, spec, optimization_steps
                )
                
                # SPICE validation
                spice_valid = True
                if validate_spice:
                    performance, spice_valid = self._validate_spice(
                        topology, optimized_params, spec
                    )
                
                # Calculate figure of merit
                fom = self._calculate_fom(performance, spec)
                
                if fom > best_performance and spice_valid:
                    best_performance = fom
                    best_result = CircuitResult(
                        netlist=self._generate_netlist(topology, optimized_params),
                        parameters=optimized_params,
                        performance=performance,
                        topology=topology['name'],
                        technology=self.technology.name,
                        generation_time=time.time() - start_time,
                        spice_valid=spice_valid
                    )
            
            if best_result is None:
                raise RuntimeError("No valid circuit found meeting specifications")
            
            logger.info(f"Circuit generation complete. "
                       f"Best performance FoM: {best_performance:.2f}")
            
            return best_result
            
        except Exception as e:
            logger.error(f"Circuit generation failed: {e}")
            raise
    
    def _generate_topologies(self, spec: DesignSpec, n_candidates: int) -> List[Dict]:
        """Generate topology candidates using CycleGAN."""
        logger.debug("Generating circuit topologies")
        
        # Create conditioning vector from spec
        condition = self._spec_to_condition(spec)
        
        topologies = []
        with torch.no_grad():
            for i in range(n_candidates):
                # Generate random noise
                noise = torch.randn(1, 100, device=self.device)
                
                # Generate topology
                topology_tensor = self.topology_model.generate(noise, condition)
                topology = self._tensor_to_topology(topology_tensor, spec.circuit_type)
                topologies.append(topology)
        
        return topologies
    
    def _generate_parameters(self, topology: Dict, spec: DesignSpec) -> Dict[str, float]:
        """Generate component parameters using diffusion model."""
        logger.debug("Generating component parameters")
        
        # Create conditioning from topology and spec
        condition = self._create_parameter_condition(topology, spec)
        
        with torch.no_grad():
            # Generate parameters using diffusion sampling
            params_tensor = self.diffusion_model.sample(condition)
            parameters = self._tensor_to_parameters(params_tensor, topology)
        
        return parameters
    
    def _optimize_parameters(
        self,
        topology: Dict,
        initial_params: Dict[str, float],
        spec: DesignSpec,
        steps: int
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """Optimize parameters using Bayesian optimization."""
        logger.debug(f"Running Bayesian optimization for {steps} steps")
        
        def objective_function(params: Dict[str, float]) -> float:
            # Quick performance estimation without full SPICE
            perf = self._estimate_performance(topology, params, spec)
            return self._calculate_fom(perf, spec)
        
        # Run optimization
        optimized_params = self.optimizer.optimize(
            objective_function, 
            initial_params, 
            n_iterations=steps
        )
        
        # Get final performance
        performance = self._estimate_performance(topology, optimized_params, spec)
        
        return optimized_params, performance
    
    def _validate_spice(
        self, 
        topology: Dict, 
        parameters: Dict[str, float], 
        spec: DesignSpec
    ) -> tuple[Dict[str, float], bool]:
        """Validate circuit with SPICE simulation."""
        logger.debug("Running SPICE validation")
        
        try:
            # Generate netlist
            netlist = self._generate_netlist(topology, parameters)
            
            # Run SPICE simulation
            results = self.spice_engine.simulate(netlist, spec)
            
            # Extract performance metrics
            performance = {
                'gain_db': results.get('gain', 0.0),
                'noise_figure_db': results.get('nf', float('inf')),
                'power_w': results.get('power', 0.0),
                's11_db': results.get('s11', 0.0),
                'bandwidth_hz': results.get('bandwidth', 0.0)
            }
            
            return performance, True
            
        except Exception as e:
            logger.warning(f"SPICE validation failed: {e}")
            # Fall back to estimated performance
            performance = self._estimate_performance(topology, parameters, spec)
            return performance, False
    
    def _spec_to_condition(self, spec: DesignSpec) -> torch.Tensor:
        """Convert design specification to model conditioning vector."""
        condition_data = [
            spec.frequency / 1e12,  # Normalize to THz
            spec.gain_min / 50.0,   # Normalize gain
            spec.nf_max / 10.0,     # Normalize NF
            spec.power_max / 1e-3,  # Normalize to mW
        ]
        
        # Add circuit type encoding
        circuit_types = ['LNA', 'Mixer', 'VCO', 'PA', 'Filter']
        type_encoding = [1.0 if ct == spec.circuit_type else 0.0 for ct in circuit_types]
        condition_data.extend(type_encoding)
        
        return torch.tensor(condition_data, device=self.device).float()
    
    def _tensor_to_topology(self, tensor: torch.Tensor, circuit_type: str) -> Dict:
        """Convert model output tensor to circuit topology representation."""
        # This is a simplified conversion - real implementation would be more complex
        topology_id = int(tensor.argmax().item()) % 10
        
        return {
            'name': f"{circuit_type}_topology_{topology_id}",
            'type': circuit_type,
            'components': self._get_default_components(circuit_type),
            'connections': self._get_default_connections(circuit_type),
            'id': topology_id
        }
    
    def _get_default_components(self, circuit_type: str) -> List[Dict]:
        """Get default component list for circuit type."""
        components = {
            'LNA': [
                {'name': 'M1', 'type': 'nmos', 'nodes': ['drain', 'gate', 'source', 'bulk']},
                {'name': 'R1', 'type': 'resistor', 'nodes': ['vdd', 'drain']},
                {'name': 'C1', 'type': 'capacitor', 'nodes': ['input', 'gate']},
                {'name': 'C2', 'type': 'capacitor', 'nodes': ['drain', 'output']},
                {'name': 'L1', 'type': 'inductor', 'nodes': ['source', 'gnd']}
            ],
            'Mixer': [
                {'name': 'M1', 'type': 'nmos', 'nodes': ['d1', 'rf_in', 's1', 'bulk']},
                {'name': 'M2', 'type': 'nmos', 'nodes': ['d2', 'lo_p', 's1', 'bulk']},
                {'name': 'M3', 'type': 'nmos', 'nodes': ['d3', 'lo_n', 's1', 'bulk']},
                {'name': 'R1', 'type': 'resistor', 'nodes': ['vdd', 'd1']},
                {'name': 'R2', 'type': 'resistor', 'nodes': ['vdd', 'd2']}
            ]
        }
        return components.get(circuit_type, components['LNA'])
    
    def _get_default_connections(self, circuit_type: str) -> List[Dict]:
        """Get default connections for circuit type."""
        return [
            {'from': 'input', 'to': 'gate', 'via': 'C1'},
            {'from': 'drain', 'to': 'output', 'via': 'C2'},
            {'from': 'vdd', 'to': 'drain', 'via': 'R1'}
        ]
    
    def _tensor_to_parameters(self, tensor: torch.Tensor, topology: Dict) -> Dict[str, float]:
        """Convert parameter tensor to component values."""
        # Extract parameter values from tensor
        param_values = tensor.cpu().numpy().flatten()
        
        parameters = {}
        param_idx = 0
        
        for component in topology['components']:
            comp_name = component['name']
            comp_type = component['type']
            
            if comp_type == 'nmos':
                parameters[f"{comp_name}_w"] = max(1e-6, param_values[param_idx] * 100e-6)
                parameters[f"{comp_name}_l"] = max(28e-9, param_values[param_idx + 1] * 1e-6)
                param_idx += 2
            elif comp_type == 'resistor':
                parameters[f"{comp_name}_r"] = max(1.0, param_values[param_idx] * 10000)
                param_idx += 1
            elif comp_type == 'capacitor':
                parameters[f"{comp_name}_c"] = max(1e-15, param_values[param_idx] * 10e-12)
                param_idx += 1
            elif comp_type == 'inductor':
                parameters[f"{comp_name}_l"] = max(1e-12, param_values[param_idx] * 10e-9)
                param_idx += 1
        
        return parameters
    
    def _create_parameter_condition(self, topology: Dict, spec: DesignSpec) -> torch.Tensor:
        """Create conditioning for parameter generation."""
        condition_data = []
        
        # Add spec information
        condition_data.extend([
            spec.frequency / 1e12,
            spec.gain_min / 50.0,
            spec.nf_max / 10.0,
            spec.power_max / 1e-3
        ])
        
        # Add topology information
        condition_data.append(float(topology['id']) / 10.0)
        condition_data.append(len(topology['components']) / 10.0)
        
        return torch.tensor(condition_data, device=self.device).float()
    
    def _estimate_performance(
        self, 
        topology: Dict, 
        parameters: Dict[str, float], 
        spec: DesignSpec
    ) -> Dict[str, float]:
        """Estimate circuit performance without full SPICE simulation."""
        # Simplified performance estimation based on component values
        # Real implementation would use more sophisticated models
        
        # Get key component values
        total_current = 0.0
        total_gain = 1.0
        total_noise = 1.0
        
        for comp in topology['components']:
            if comp['type'] == 'nmos':
                w = parameters.get(f"{comp['name']}_w", 10e-6)
                l = parameters.get(f"{comp['name']}_l", 100e-9)
                gm = w / l * 1e-3  # Simplified gm estimation
                total_gain *= (1 + gm * 1000)  # Simplified gain
                total_current += gm * 0.7  # Simplified current
                total_noise *= (1 + 1/gm)  # Simplified noise
        
        # Convert to performance metrics
        gain_db = min(50.0, 20 * np.log10(max(1.0, total_gain)))
        nf_db = max(0.5, 10 * np.log10(total_noise))
        power_w = total_current * 1.2  # Assume 1.2V supply
        
        return {
            'gain_db': gain_db,
            'noise_figure_db': nf_db,
            'power_w': power_w,
            's11_db': -15.0,  # Simplified
            'bandwidth_hz': spec.frequency * 0.1  # Simplified
        }
    
    def _calculate_fom(self, performance: Dict[str, float], spec: DesignSpec) -> float:
        """Calculate figure of merit for circuit performance."""
        gain = performance.get('gain_db', 0.0)
        nf = performance.get('noise_figure_db', float('inf'))
        power = performance.get('power_w', float('inf'))
        
        # Check if specs are met
        gain_ok = gain >= spec.gain_min
        nf_ok = nf <= spec.nf_max
        power_ok = power <= spec.power_max
        
        if not (gain_ok and nf_ok and power_ok):
            return float('-inf')  # Invalid solution
        
        # Calculate FoM: maximize gain, minimize noise and power
        fom = gain / (power * 1000 * max(1.0, nf - 1.0))
        return fom
    
    def _generate_netlist(self, topology: Dict, parameters: Dict[str, float]) -> str:
        """Generate SPICE netlist from topology and parameters."""
        netlist_lines = [
            f"* {topology['name']} netlist",
            f"* Generated by GenRF CircuitDiffuser",
            "",
            ".param vdd=1.2",
            ".param temp=27",
            ""
        ]
        
        # Add components
        for component in topology['components']:
            comp_name = component['name']
            comp_type = component['type']
            nodes = ' '.join(component['nodes'])
            
            if comp_type == 'nmos':
                w = parameters.get(f"{comp_name}_w", 10e-6)
                l = parameters.get(f"{comp_name}_l", 100e-9)
                netlist_lines.append(f"{comp_name} {nodes} nch w={w:.3e} l={l:.3e}")
            elif comp_type == 'resistor':
                r = parameters.get(f"{comp_name}_r", 1000)
                netlist_lines.append(f"{comp_name} {nodes} {r:.3e}")
            elif comp_type == 'capacitor':
                c = parameters.get(f"{comp_name}_c", 1e-12)
                netlist_lines.append(f"{comp_name} {nodes} {c:.3e}")
            elif comp_type == 'inductor':
                l_val = parameters.get(f"{comp_name}_l", 1e-9)
                netlist_lines.append(f"{comp_name} {nodes} {l_val:.3e}")
        
        netlist_lines.extend([
            "",
            ".model nch nmos level=1",
            ".end"
        ])
        
        return '\n'.join(netlist_lines)