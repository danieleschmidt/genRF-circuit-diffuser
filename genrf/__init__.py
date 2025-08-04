"""
GenRF Circuit Diffuser - Generative AI for RF Circuit Design

A toolkit for generating analog and RF circuits using cycle-consistent GANs
and diffusion models with SPICE-in-the-loop optimization.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

# Core imports for easy access  
from .core import (
    CircuitDiffuser, CircuitResult,
    DesignSpec, CommonSpecs,
    TechnologyFile, DeviceModel, PassiveModel, DesignRules,
    SPICEEngine, SPICEError,
    BayesianOptimizer, OptimizationResult, ParetoFrontOptimizer,
    CodeExporter,
    CycleGAN, DiffusionModel
)

# Make main classes available at package level
__all__ = [
    "CircuitDiffuser",
    "CircuitResult",
    "DesignSpec", 
    "CommonSpecs",
    "TechnologyFile",
    "DeviceModel",
    "PassiveModel",
    "DesignRules",
    "SPICEEngine",
    "SPICEError", 
    "BayesianOptimizer",
    "OptimizationResult",
    "ParetoFrontOptimizer",
    "CodeExporter",
    "CycleGAN",
    "DiffusionModel"
]