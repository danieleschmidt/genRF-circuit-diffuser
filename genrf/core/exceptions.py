"""
Custom exceptions for GenRF circuit diffuser.

This module defines custom exception classes for better error handling
and debugging in the circuit generation pipeline.
"""

from typing import Optional, Dict, Any


class GenRFError(Exception):
    """Base exception class for GenRF."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ModelError(GenRFError):
    """Exception raised for AI model related errors."""
    
    def __init__(self, message: str, model_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="MODEL_ERROR", **kwargs)
        if model_type:
            self.details['model_type'] = model_type


class SPICEError(GenRFError):
    """Exception raised for SPICE simulation errors."""
    
    def __init__(self, message: str, simulation_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="SPICE_ERROR", **kwargs)
        if simulation_type:
            self.details['simulation_type'] = simulation_type


class OptimizationError(GenRFError):
    """Exception raised for optimization errors."""
    
    def __init__(self, message: str, optimizer_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="OPTIMIZATION_ERROR", **kwargs)
        if optimizer_type:
            self.details['optimizer_type'] = optimizer_type


class ValidationError(GenRFError):
    """Exception raised for input validation errors."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        if field_name:
            self.details['field_name'] = field_name


class TechnologyError(GenRFError):
    """Exception raised for technology file related errors."""
    
    def __init__(self, message: str, technology: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="TECHNOLOGY_ERROR", **kwargs)
        if technology:
            self.details['technology'] = technology


class ExportError(GenRFError):
    """Exception raised for code export errors."""
    
    def __init__(self, message: str, export_format: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="EXPORT_ERROR", **kwargs)
        if export_format:
            self.details['export_format'] = export_format


class ConvergenceError(GenRFError):
    """Exception raised when optimization doesn't converge."""
    
    def __init__(self, message: str, iterations: Optional[int] = None, **kwargs):
        super().__init__(message, error_code="CONVERGENCE_ERROR", **kwargs)
        if iterations is not None:
            self.details['iterations'] = iterations


class ResourceError(GenRFError):
    """Exception raised for resource related errors (memory, compute, etc.)."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="RESOURCE_ERROR", **kwargs)
        if resource_type:
            self.details['resource_type'] = resource_type


class ConfigurationError(GenRFError):
    """Exception raised for configuration errors."""
    
    def __init__(self, message: str, config_section: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        if config_section:
            self.details['config_section'] = config_section


class SecurityError(GenRFError):
    """Exception raised for security related errors."""
    
    def __init__(self, message: str, security_issue: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="SECURITY_ERROR", **kwargs)
        if security_issue:
            self.details['security_issue'] = security_issue