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
        
        # Initialize AI models
        self.cycle_gan = CycleGAN().to(self.device)
        self.diffusion_model = DiffusionModel().to(self.device)
        
        # Load pre-trained weights if available
        if checkpoint and Path(checkpoint).exists():
            self._load_checkpoint(checkpoint)
        else:
            logger.warning("No checkpoint provided or file not found. Using randomly initialized models.")
        
        # Initialize SPICE engine
        self.spice_engine = SPICEEngine(engine=spice_engine)
        
        # Initialize Bayesian optimizer
        self.optimizer = BayesianOptimizer()
        
        # Set technology file
        if technology is None:
            self.technology = TechnologyFile.get_default()
        else:
            self.technology = technology
        
        # Initialize code exporter
        self.exporter = CodeExporter()
        
        logger.info("CircuitDiffuser initialization complete")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load pre-trained model weights from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'cycle_gan_state_dict' in checkpoint:
                self.cycle_gan.load_state_dict(checkpoint['cycle_gan_state_dict'])
                logger.info("Loaded CycleGAN weights")
            
            if 'diffusion_model_state_dict' in checkpoint:
                self.diffusion_model.load_state_dict(checkpoint['diffusion_model_state_dict'])
                logger.info("Loaded DiffusionModel weights")
            
            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Checkpoint loading failed: {e}")
    
    def generate(
        self,
        spec: DesignSpec,
        n_candidates: int = 100,
        optimization_steps: int = 50,
        spice_validation: bool = True
    ) -> CircuitResult:
        """
        Generate optimized RF circuit for given specification.
        
        Args:
            spec: Design specification defining requirements
            n_candidates: Number of topology candidates to generate
            optimization_steps: Number of parameter optimization iterations
            spice_validation: Whether to validate using SPICE simulation
            
        Returns:
            CircuitResult containing optimized circuit and performance metrics
        """
        start_time = time.time()
        
        logger.info(f"Generating {spec.circuit_type} circuit")
        logger.info(f"Target: {spec.gain_min}dB gain, {spec.nf_max}dB NF, {spec.power_max*1000:.1f}mW")
        
        try:
            # Step 1: Generate topology candidates using CycleGAN
            logger.info("Step 1: Generating topology candidates...")
            topologies = self._generate_topologies(spec, n_candidates)
            
            # Step 2: Optimize parameters for each topology
            logger.info("Step 2: Optimizing circuit parameters...")
            candidates = []
            
            for i, topology in enumerate(topologies[:min(10, len(topologies))]):  # Limit for speed
                logger.debug(f"Optimizing candidate {i+1}")
                
                # Generate parameters using diffusion model
                optimized_params = self._optimize_parameters(topology, spec, optimization_steps)
                
                # Create netlist
                netlist = self._create_netlist(topology, optimized_params, spec)
                
                candidates.append({
                    'topology': topology,
                    'parameters': optimized_params,
                    'netlist': netlist,
                    'index': i
                })
            
            # Step 3: Validate and rank candidates
            if spice_validation:
                logger.info("Step 3: SPICE validation and ranking...")
                candidates = self._validate_candidates(candidates, spec)
            else:
                logger.info("Step 3: Analytical ranking (SPICE disabled)...")
                candidates = self._rank_candidates_analytical(candidates, spec)
            
            # Step 4: Select best candidate
            if not candidates:
                raise RuntimeError("No valid circuit candidates generated")
            
            best_candidate = candidates[0]
            
            # Step 5: Final performance evaluation
            logger.info("Step 4: Final performance evaluation...")
            
            if spice_validation:
                try:
                    performance = self.spice_engine.simulate(best_candidate['netlist'], spec)
                    spice_valid = True
                except Exception as e:
                    logger.warning(f"SPICE simulation failed: {e}")
                    performance = self._estimate_performance(best_candidate, spec)
                    spice_valid = False
            else:
                performance = self._estimate_performance(best_candidate, spec)
                spice_valid = False
            
            # Create result object
            generation_time = time.time() - start_time
            
            result = CircuitResult(
                netlist=best_candidate['netlist'],
                parameters=best_candidate['parameters'],
                performance=performance,
                topology=f"topology_{best_candidate['index']}",
                technology=self.technology.name,
                generation_time=generation_time,
                spice_valid=spice_valid
            )
            
            logger.info(f"Circuit generation completed in {generation_time:.2f}s")
            logger.info(f"Final performance: Gain={result.performance.get('gain', 0):.1f}dB, "
                       f"NF={result.performance.get('nf', 0):.2f}dB, "
                       f"Power={result.performance.get('power', 0)*1000:.1f}mW")
            
            return result
            
        except Exception as e:
            logger.error(f"Circuit generation failed: {e}")
            raise
    
    def _generate_topologies(self, spec: DesignSpec, n_candidates: int) -> List[torch.Tensor]:
        """Generate circuit topology candidates using CycleGAN."""
        self.cycle_gan.eval()
        
        # Convert specification to conditioning vector
        spec_vector = self._spec_to_vector(spec)
        spec_tensor = torch.tensor(spec_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Generate multiple topology candidates
        topologies = []
        
        with torch.no_grad():
            # Generate in batches to avoid memory issues
            batch_size = min(20, n_candidates)
            n_batches = (n_candidates + batch_size - 1) // batch_size
            
            for batch in range(n_batches):
                current_batch_size = min(batch_size, n_candidates - batch * batch_size)
                
                # Generate noise for this batch
                noise = torch.randn(current_batch_size, self.cycle_gan.latent_dim, device=self.device)
                spec_batch = spec_tensor.repeat(current_batch_size, 1)
                
                # Generate topologies
                batch_topologies = self.cycle_gan.generate(noise, spec_batch)
                topologies.extend([batch_topologies[i] for i in range(current_batch_size)])
        
        logger.debug(f"Generated {len(topologies)} topology candidates")
        return topologies
    
    def _spec_to_vector(self, spec: DesignSpec) -> List[float]:
        """Convert design specification to normalized conditioning vector."""
        # Normalize all specification parameters to [-1, 1] range
        vector = [
            np.tanh(np.log10(spec.frequency / 1e9)),  # Log-normalized frequency
            np.tanh(spec.gain_min / 25.0),  # Normalized gain
            np.tanh(spec.nf_max / 5.0),  # Normalized noise figure
            np.tanh(np.log10(spec.power_max * 1000)),  # Log-normalized power (mW)
            np.tanh(spec.supply_voltage / 2.0),  # Normalized supply voltage
            np.tanh(spec.temperature / 50.0),  # Normalized temperature
            np.tanh(spec.input_impedance / 100.0),  # Normalized input impedance
            np.tanh(spec.output_impedance / 100.0),  # Normalized output impedance
            1.0 if spec.circuit_type == 'LNA' else -1.0  # Circuit type encoding
        ]
        
        return vector
    
    def _optimize_parameters(
        self, 
        topology: torch.Tensor, 
        spec: DesignSpec, 
        optimization_steps: int
    ) -> Dict[str, float]:
        """Optimize circuit parameters using diffusion model and Bayesian optimization."""
        
        # Generate initial parameter candidates using diffusion model
        self.diffusion_model.eval()
        
        with torch.no_grad():
            # Use topology as conditioning for parameter generation
            condition = topology.unsqueeze(0)
            param_samples = self.diffusion_model.sample(condition, num_inference_steps=20)
        
        # Convert tensor to parameter dictionary
        initial_params = self._tensor_to_parameters(param_samples[0], spec)
        
        # Define objective function for Bayesian optimization
        def objective_function(params: Dict[str, float]) -> float:
            return self._evaluate_parameter_quality(params, topology, spec)
        
        # Get parameter bounds for optimization
        param_bounds = self._get_parameter_bounds(spec)
        
        try:
            # Run Bayesian optimization
            optimization_result = self.optimizer.optimize(
                objective_function,
                param_bounds,
                n_iterations=min(optimization_steps, 20)  # Limit for speed
            )
            
            return optimization_result.best_parameters
            
        except Exception as e:
            logger.warning(f"Parameter optimization failed: {e}. Using diffusion model result.")
            return initial_params
    
    def _tensor_to_parameters(self, param_tensor: torch.Tensor, spec: DesignSpec) -> Dict[str, float]:
        """Convert parameter tensor to named circuit parameters."""
        params = param_tensor.cpu().numpy()
        
        # Circuit-specific parameter mapping
        if spec.circuit_type == 'LNA':
            return self._tensor_to_lna_parameters(params)
        elif spec.circuit_type == 'Mixer':
            return self._tensor_to_mixer_parameters(params)
        elif spec.circuit_type == 'VCO':
            return self._tensor_to_vco_parameters(params)
        elif spec.circuit_type == 'PA':
            return self._tensor_to_pa_parameters(params)
        else:
            return self._tensor_to_generic_parameters(params)
    
    def _tensor_to_lna_parameters(self, params: np.ndarray) -> Dict[str, float]:
        """Convert tensor to LNA-specific parameters."""
        return {
            'W1': max(1e-6, abs(params[0]) * 200e-6),   # Main transistor width
            'L1': max(28e-9, abs(params[1]) * 2e-6),    # Main transistor length
            'W2': max(1e-6, abs(params[2]) * 100e-6),   # Cascode transistor width
            'L2': max(28e-9, abs(params[3]) * 1e-6),    # Cascode transistor length
            'Ls': max(1e-12, abs(params[4]) * 20e-9),   # Source inductance
            'Lg': max(1e-12, abs(params[5]) * 20e-9),   # Gate inductance
            'Ld': max(1e-12, abs(params[6]) * 20e-9),   # Drain inductance
            'Cs': max(1e-15, abs(params[7]) * 50e-12),  # Source capacitance
            'Cg': max(1e-15, abs(params[8]) * 50e-12),  # Gate capacitance
            'Cd': max(1e-15, abs(params[9]) * 50e-12),  # Drain capacitance
            'Rd': max(10.0, abs(params[10]) * 5000),    # Drain resistance
            'Rs': max(10.0, abs(params[11]) * 1000),    # Source resistance
            'Ibias': max(100e-6, abs(params[12]) * 20e-3), # Bias current
            'Vbias': max(0.1, abs(params[13]) * 2.0)    # Bias voltage
        }
    
    def _tensor_to_mixer_parameters(self, params: np.ndarray) -> Dict[str, float]:
        """Convert tensor to mixer-specific parameters."""
        return {
            'Wrf': max(1e-6, abs(params[0]) * 100e-6),   # RF transistor width
            'Lrf': max(28e-9, abs(params[1]) * 1e-6),    # RF transistor length
            'Wlo': max(1e-6, abs(params[2]) * 150e-6),   # LO transistor width
            'Llo': max(28e-9, abs(params[3]) * 1e-6),    # LO transistor length
            'Wtail': max(2e-6, abs(params[4]) * 200e-6), # Tail transistor width
            'Ltail': max(28e-9, abs(params[5]) * 2e-6),  # Tail transistor length
            'RL': max(100.0, abs(params[6]) * 5000),     # Load resistance
            'Ibias': max(0.5e-3, abs(params[7]) * 10e-3), # Bias current
            'Cin': max(1e-15, abs(params[8]) * 20e-12),  # Input capacitance
            'Cout': max(1e-15, abs(params[9]) * 20e-12)  # Output capacitance
        }
    
    def _tensor_to_vco_parameters(self, params: np.ndarray) -> Dict[str, float]:
        """Convert tensor to VCO-specific parameters."""
        return {
            'W': max(5e-6, abs(params[0]) * 300e-6),     # Transistor width
            'L': max(28e-9, abs(params[1]) * 500e-9),    # Transistor length
            'Ltank': max(1e-12, abs(params[2]) * 50e-9), # Tank inductance
            'Ctank': max(1e-15, abs(params[3]) * 10e-12), # Tank capacitance
            'Cvar': max(1e-15, abs(params[4]) * 5e-12),  # Varactor capacitance
            'Rtail': max(1000, abs(params[5]) * 10000),  # Tail resistance
            'Ibias': max(1e-3, abs(params[6]) * 20e-3)   # Bias current
        }
    
    def _tensor_to_pa_parameters(self, params: np.ndarray) -> Dict[str, float]:
        """Convert tensor to power amplifier parameters."""
        return {
            'W1': max(10e-6, abs(params[0]) * 1000e-6),  # Driver stage width
            'L1': max(28e-9, abs(params[1]) * 1e-6),     # Driver stage length
            'W2': max(50e-6, abs(params[2]) * 2000e-6),  # Output stage width
            'L2': max(28e-9, abs(params[3]) * 1e-6),     # Output stage length
            'Lmatch': max(1e-12, abs(params[4]) * 20e-9), # Matching inductance
            'Cmatch': max(1e-15, abs(params[5]) * 50e-12), # Matching capacitance
            'Rbias': max(1000, abs(params[6]) * 50000),  # Bias resistance
            'Ibias': max(5e-3, abs(params[7]) * 100e-3)  # Bias current
        }
    
    def _tensor_to_generic_parameters(self, params: np.ndarray) -> Dict[str, float]:
        """Convert tensor to generic circuit parameters."""
        return {f'param_{i}': float(params[i % len(params)]) for i in range(8)}
    
    def _get_parameter_bounds(self, spec: DesignSpec) -> Dict[str, tuple]:
        """Get optimization bounds for circuit parameters."""
        if spec.circuit_type == 'LNA':
            return {
                'W1': (1e-6, 500e-6), 'L1': (28e-9, 10e-6),
                'W2': (1e-6, 200e-6), 'L2': (28e-9, 5e-6),
                'Ls': (1e-12, 50e-9), 'Lg': (1e-12, 50e-9), 'Ld': (1e-12, 50e-9),
                'Cs': (1e-15, 100e-12), 'Cg': (1e-15, 100e-12), 'Cd': (1e-15, 100e-12),
                'Rd': (50, 10000), 'Rs': (10, 2000),
                'Ibias': (0.1e-3, 50e-3), 'Vbias': (0.1, 3.0)
            }
        elif spec.circuit_type == 'Mixer':
            return {
                'Wrf': (1e-6, 200e-6), 'Lrf': (28e-9, 5e-6),
                'Wlo': (1e-6, 300e-6), 'Llo': (28e-9, 5e-6),
                'Wtail': (2e-6, 400e-6), 'Ltail': (28e-9, 10e-6),
                'RL': (100, 10000), 'Ibias': (0.5e-3, 20e-3),
                'Cin': (1e-15, 50e-12), 'Cout': (1e-15, 50e-12)
            }
        else:
            # Generic bounds
            return {f'param_{i}': (-10.0, 10.0) for i in range(8)}
    
    def _evaluate_parameter_quality(
        self, 
        params: Dict[str, float], 
        topology: torch.Tensor, 
        spec: DesignSpec
    ) -> float:
        """Evaluate parameter set quality for optimization objective."""
        try:
            # Create netlist for these parameters
            netlist = self._create_netlist(topology, params, spec)
            
            # Quick convergence check
            if not self.spice_engine.check_convergence(netlist):
                return -1000.0  # Heavy penalty for non-convergent circuits
            
            # Estimate performance
            performance = self._estimate_performance({'parameters': params}, spec)
            
            # Calculate multi-objective score
            gain_score = min(performance.get('gain', 0) / max(spec.gain_min, 1), 3.0)
            nf_score = max(0, 3.0 - performance.get('nf', 20) / max(spec.nf_max, 1))
            power_score = max(0, 3.0 - performance.get('power', 1) / max(spec.power_max, 1e-6))
            
            # Weighted combination
            objective = 0.4 * gain_score + 0.3 * nf_score + 0.3 * power_score
            
            return objective
            
        except Exception as e:
            logger.debug(f"Parameter evaluation failed: {e}")
            return -1000.0
    
    def _create_netlist(
        self, 
        topology: torch.Tensor, 
        params: Dict[str, float], 
        spec: DesignSpec
    ) -> str:
        """Create SPICE netlist from topology and parameters."""
        
        if spec.circuit_type == 'LNA':
            return self._create_lna_netlist(params, spec)
        elif spec.circuit_type == 'Mixer':
            return self._create_mixer_netlist(params, spec)
        elif spec.circuit_type == 'VCO':
            return self._create_vco_netlist(params, spec)
        elif spec.circuit_type == 'PA':
            return self._create_pa_netlist(params, spec)
        else:
            return self._create_generic_netlist(params, spec)
    
    def _create_lna_netlist(self, params: Dict[str, float], spec: DesignSpec) -> str:
        """Create LNA SPICE netlist."""
        return f'''* Low Noise Amplifier - Generated by GenRF
* Target: {spec.frequency/1e9:.2f} GHz, {spec.gain_min} dB gain, {spec.nf_max} dB NF

.include '{self.technology.model_file}'

* Input matching network
Ls input n1 {params['Ls']:.3e}
Cgs n1 gate {params['Cg']:.3e}
Lg gate n2 {params['Lg']:.3e}
Rs n2 vbias {params['Rs']:.1f}

* Main LNA core (cascode topology)
M1 n3 n2 source bulk {self.technology.nmos_model} W={params['W1']:.3e} L={params['L1']:.3e}
M2 drain n4 n3 bulk {self.technology.nmos_model} W={params['W2']:.3e} L={params['L2']:.3e}

* Biasing
Vbias vbias 0 DC {params['Vbias']:.3f}
Ibias source 0 DC {params['Ibias']:.3e}
Vgate n4 0 DC {spec.supply_voltage * 0.7:.3f}

* Output matching network
Ld drain n5 {params['Ld']:.3e}
Cd n5 output {params['Cd']:.3e}
Rd output vdd {params['Rd']:.1f}

* Supply
Vdd vdd 0 DC {spec.supply_voltage:.3f}

* Load
RL output 0 {spec.output_impedance:.1f}

.end'''
    
    def _create_mixer_netlist(self, params: Dict[str, float], spec: DesignSpec) -> str:
        """Create mixer SPICE netlist."""
        return f'''* Gilbert Cell Mixer - Generated by GenRF
.include '{self.technology.model_file}'

* RF input stage
M1 n1 rf_p tail bulk {self.technology.nmos_model} W={params['Wrf']:.3e} L={params['Lrf']:.3e}
M2 n2 rf_n tail bulk {self.technology.nmos_model} W={params['Wrf']:.3e} L={params['Lrf']:.3e}

* LO switching quad
M3 if_p lo_p n1 bulk {self.technology.nmos_model} W={params['Wlo']:.3e} L={params['Llo']:.3e}
M4 if_n lo_n n1 bulk {self.technology.nmos_model} W={params['Wlo']:.3e} L={params['Llo']:.3e}
M5 if_n lo_p n2 bulk {self.technology.nmos_model} W={params['Wlo']:.3e} L={params['Llo']:.3e}
M6 if_p lo_n n2 bulk {self.technology.nmos_model} W={params['Wlo']:.3e} L={params['Llo']:.3e}

* Tail current source
Mtail tail vbias 0 bulk {self.technology.nmos_model} W={params['Wtail']:.3e} L={params['Ltail']:.3e}
Itail vbias 0 DC {params['Ibias']:.3e}

* Load resistors
RL1 if_p vdd {params['RL']:.1f}
RL2 if_n vdd {params['RL']:.1f}

* AC coupling
Cin1 rf_input rf_p {params['Cin']:.3e}
Cin2 rf_input rf_n {params['Cin']:.3e}
Cout1 if_p if_output {params['Cout']:.3e}

* Supply
Vdd vdd 0 DC {spec.supply_voltage:.3f}

.end'''
    
    def _create_vco_netlist(self, params: Dict[str, float], spec: DesignSpec) -> str:
        """Create VCO SPICE netlist."""
        return f'''* LC VCO - Generated by GenRF
.include '{self.technology.model_file}'

* Cross-coupled pair
M1 outp outn tail bulk {self.technology.nmos_model} W={params['W']:.3e} L={params['L']:.3e}
M2 outn outp tail bulk {self.technology.nmos_model} W={params['W']:.3e} L={params['L']:.3e}

* LC tank
L1 outp vdd {params['Ltank']:.3e}
L2 outn vdd {params['Ltank']:.3e}
C1 outp outn {params['Ctank']:.3e}

* Varactor for tuning
Cvar1 outp vctrl {params['Cvar']:.3e}
Cvar2 outn vctrl {params['Cvar']:.3e}

* Tail current
Rtail tail 0 {params['Rtail']:.1f}
Itail tail 0 DC {params['Ibias']:.3e}

* Control voltage
Vctrl vctrl 0 DC {spec.supply_voltage * 0.5:.3f}

* Supply
Vdd vdd 0 DC {spec.supply_voltage:.3f}

.end'''
    
    def _create_pa_netlist(self, params: Dict[str, float], spec: DesignSpec) -> str:
        """Create power amplifier SPICE netlist."""
        return f'''* Power Amplifier - Generated by GenRF
.include '{self.technology.model_file}'

* Driver stage
M1 n1 input 0 bulk {self.technology.nmos_model} W={params['W1']:.3e} L={params['L1']:.3e}
Rbias1 input vbias1 {params['Rbias']:.1f}

* Output stage
M2 output n1 0 bulk {self.technology.nmos_model} W={params['W2']:.3e} L={params['L2']:.3e}

* Matching network
Lmatch output n2 {params['Lmatch']:.3e}
Cmatch n2 0 {params['Cmatch']:.3e}

* Biasing
Vbias1 vbias1 0 DC {spec.supply_voltage * 0.4:.3f}
Ibias1 n1 vdd DC {params['Ibias']:.3e}

* Supply
Vdd vdd 0 DC {spec.supply_voltage:.3f}

* Load
RL output 0 {spec.output_impedance:.1f}

.end'''
    
    def _create_generic_netlist(self, params: Dict[str, float], spec: DesignSpec) -> str:
        """Create generic circuit netlist."""
        return f'''* Generic Circuit - Generated by GenRF
.include '{self.technology.model_file}'

* Simple amplifier
M1 output input 0 bulk {self.technology.nmos_model} W=50u L=100n
R1 output vdd 1k

* Supply
Vdd vdd 0 DC {spec.supply_voltage:.3f}

.end'''
    
    def _validate_candidates(
        self, 
        candidates: List[Dict[str, Any]], 
        spec: DesignSpec
    ) -> List[Dict[str, Any]]:
        """Validate candidates using SPICE simulation and rank by performance."""
        valid_candidates = []
        
        logger.info(f"Validating {len(candidates)} candidates with SPICE...")
        
        for i, candidate in enumerate(candidates):
            logger.debug(f"Validating candidate {i+1}/{len(candidates)}")
            
            try:
                # Check convergence first
                if not self.spice_engine.check_convergence(candidate['netlist']):
                    logger.debug(f"Candidate {i} failed convergence")
                    continue
                
                # Run full SPICE simulation
                performance = self.spice_engine.simulate(candidate['netlist'], spec)
                
                # Calculate performance score
                score = self._calculate_performance_score(performance, spec)
                
                candidate['performance'] = performance
                candidate['score'] = score
                candidate['spice_valid'] = True
                
                valid_candidates.append(candidate)
                
                logger.debug(f"Candidate {i}: score={score:.3f}, "
                           f"gain={performance.get('gain', 0):.1f}dB")
                
            except Exception as e:
                logger.debug(f"Candidate {i} SPICE validation failed: {e}")
                continue
        
        # Sort by score (descending)
        valid_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        logger.info(f"Successfully validated {len(valid_candidates)} candidates")
        
        # Return at least one candidate (best effort)
        return valid_candidates if valid_candidates else candidates[:1]
    
    def _rank_candidates_analytical(
        self, 
        candidates: List[Dict[str, Any]], 
        spec: DesignSpec
    ) -> List[Dict[str, Any]]:
        """Rank candidates using analytical performance estimation."""
        
        for candidate in candidates:
            try:
                performance = self._estimate_performance(candidate, spec)
                score = self._calculate_performance_score(performance, spec)
                
                candidate['performance'] = performance
                candidate['score'] = score
                candidate['spice_valid'] = False
                
            except Exception as e:
                logger.debug(f"Analytical ranking failed for candidate: {e}")
                candidate['score'] = 0
        
        # Sort by score
        candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return candidates
    
    def _estimate_performance(self, candidate: Dict[str, Any], spec: DesignSpec) -> Dict[str, float]:
        """Estimate circuit performance using analytical models."""
        params = candidate['parameters']
        
        if spec.circuit_type == 'LNA':
            return self._estimate_lna_performance(params, spec)
        elif spec.circuit_type == 'Mixer':
            return self._estimate_mixer_performance(params, spec)
        elif spec.circuit_type == 'VCO':
            return self._estimate_vco_performance(params, spec)
        elif spec.circuit_type == 'PA':
            return self._estimate_pa_performance(params, spec)
        else:
            return self._estimate_generic_performance(params, spec)
    
    def _estimate_lna_performance(self, params: Dict[str, float], spec: DesignSpec) -> Dict[str, float]:
        """Estimate LNA performance using simplified RF models."""
        
        # Extract key parameters
        W1 = params.get('W1', 50e-6)
        L1 = params.get('L1', 100e-9)
        Ibias = params.get('Ibias', 5e-3)
        Rd = params.get('Rd', 1000)
        
        # Simplified gm calculation
        gm = 2 * Ibias / 0.3  # Assuming Vov = 0.3V
        
        # Gain estimation
        gain_linear = gm * Rd
        gain_db = 20 * np.log10(max(gain_linear, 1e-3))
        
        # Noise figure estimation (simplified)
        gamma = 2.0  # Channel noise factor
        alpha = 1.0  # Gate noise factor
        nf_linear = 1 + gamma * 2.5 / gm + alpha * (W1 / L1)
        nf_db = 10 * np.log10(nf_linear)
        
        # Power consumption
        power = Ibias * spec.supply_voltage
        
        # Input matching (simplified)
        Cgs = W1 * L1 * 2e6  # Simplified Cgs model
        wt = gm / Cgs  # Unity gain frequency
        s11_db = -20 if wt > 2 * np.pi * spec.frequency else -5
        
        # Bandwidth estimation
        bandwidth = wt / (2 * np.pi * 5)  # Rough BW estimate
        
        return {
            'gain': gain_db,
            'nf': nf_db,
            'power': power,
            's11': s11_db,
            'bandwidth': bandwidth
        }
    
    def _estimate_mixer_performance(self, params: Dict[str, float], spec: DesignSpec) -> Dict[str, float]:
        """Estimate mixer performance."""
        
        Wrf = params.get('Wrf', 50e-6)
        Ibias = params.get('Ibias', 2e-3)
        RL = params.get('RL', 1000)
        
        # Transconductance
        gm = 2 * Ibias / 0.3
        
        # Conversion gain (Gilbert cell)
        conversion_gain_linear = (2 / np.pi) * gm * RL
        conversion_gain_db = 20 * np.log10(max(conversion_gain_linear, 1e-3))
        
        # Noise figure (mixer)
        nf_db = 10 + 20 * np.log10(spec.frequency / 1e9)  # Frequency dependent
        
        # Power
        power = Ibias * spec.supply_voltage
        
        return {
            'gain': conversion_gain_db,
            'nf': nf_db,
            'power': power,
            's11': -10.0,
            'bandwidth': spec.frequency * 0.1
        }
    
    def _estimate_vco_performance(self, params: Dict[str, float], spec: DesignSpec) -> Dict[str, float]:
        """Estimate VCO performance."""
        
        W = params.get('W', 100e-6)
        L = params.get('L', 100e-9)
        Ltank = params.get('Ltank', 5e-9)
        Ctank = params.get('Ctank', 1e-12)
        Ibias = params.get('Ibias', 10e-3)
        
        # Oscillation frequency
        f_osc = 1 / (2 * np.pi * np.sqrt(Ltank * Ctank))
        
        # Phase noise estimation (Leeson's model simplified)
        gm = 2 * Ibias / 0.3
        Q = np.sqrt(Ltank / Ctank) / 10  # Tank Q
        phase_noise = -120 - 20 * np.log10(Q) - 10 * np.log10(gm)
        
        # Power
        power = Ibias * spec.supply_voltage
        
        return {
            'gain': 0.0,  # VCO doesn't have gain
            'nf': float('inf'),  # Not applicable
            'power': power,
            'frequency': f_osc,
            'phase_noise': phase_noise,
            'tuning_range': f_osc * 0.1
        }
    
    def _estimate_pa_performance(self, params: Dict[str, float], spec: DesignSpec) -> Dict[str, float]:
        """Estimate power amplifier performance."""
        
        W2 = params.get('W2', 500e-6)
        Ibias = params.get('Ibias', 50e-3)
        
        # Power gain
        gm = 2 * Ibias / 0.3
        gain_db = 20 + 10 * np.log10(W2 / 100e-6)  # Size dependent
        
        # Output power (1dB compression)
        p1db_dbm = 10 * np.log10(Ibias * spec.supply_voltage * 1000) - 10
        
        # Efficiency
        efficiency = min(0.6, Ibias * 0.1)
        
        # Power consumption
        power = Ibias * spec.supply_voltage
        
        return {
            'gain': gain_db,
            'nf': 5.0,  # Typical PA NF
            'power': power,
            'p1db': p1db_dbm,
            'efficiency': efficiency,
            's11': -8.0
        }
    
    def _estimate_generic_performance(self, params: Dict[str, float], spec: DesignSpec) -> Dict[str, float]:
        """Generic performance estimation."""
        return {
            'gain': 15.0,
            'nf': 3.0,
            'power': 0.01,
            's11': -10.0,
            'bandwidth': spec.frequency * 0.1
        }
    
    def _calculate_performance_score(self, performance: Dict[str, float], spec: DesignSpec) -> float:
        """Calculate overall performance score for ranking."""
        
        # Normalize performance metrics
        gain_score = min(performance.get('gain', 0) / max(spec.gain_min, 1), 3.0)
        nf_score = max(0, 3.0 - performance.get('nf', 20) / max(spec.nf_max, 1))
        power_score = max(0, 3.0 - performance.get('power', 1) / max(spec.power_max, 1e-6))
        
        # Input matching score
        s11 = performance.get('s11', 0)
        s11_score = max(0, 2.0 + s11 / 10.0)  # Better matching = higher score
        
        # Weighted combination
        total_score = (
            0.35 * gain_score +
            0.25 * nf_score +
            0.25 * power_score +
            0.15 * s11_score
        )
        
        return total_score
        
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