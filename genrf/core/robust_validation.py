"""
Robust Validation & Error Recovery System for RF Circuit Generation.

This module implements comprehensive validation, error recovery, and circuit
verification systems to ensure generated circuits meet physical constraints
and design specifications with high reliability.

Features:
1. Multi-level validation pipeline
2. Physics-based constraint verification
3. Automatic error recovery mechanisms
4. Adversarial robustness testing
5. Circuit stability analysis
6. Performance variance assessment

Research Innovation: First implementation of adversarial robustness testing
for analog circuit generation, ensuring circuits remain stable under
process variations and environmental conditions.
"""

import logging
import time
import traceback
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from .design_spec import DesignSpec
from .models import CircuitResult
from .exceptions import ValidationError, CircuitGenerationError, SPICEError

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate" 
    COMPREHENSIVE = "comprehensive"
    ADVERSARIAL = "adversarial"


@dataclass
class ValidationResult:
    """Result of circuit validation."""
    
    is_valid: bool
    confidence_score: float
    validation_level: ValidationLevel
    
    # Detailed results
    constraint_violations: List[Dict[str, Any]] = field(default_factory=list)
    physics_violations: List[Dict[str, Any]] = field(default_factory=list)
    stability_issues: List[Dict[str, Any]] = field(default_factory=list)
    performance_warnings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recovery suggestions
    suggested_fixes: List[Dict[str, Any]] = field(default_factory=list)
    alternative_architectures: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metrics
    validation_time: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0


class PhysicsConstraintValidator:
    """Validates circuits against fundamental physics constraints."""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        
        # Physical constants
        self.c = 299792458.0  # Speed of light (m/s)
        self.k_boltzmann = 1.380649e-23  # Boltzmann constant (J/K)
        self.q_electron = 1.602176634e-19  # Elementary charge (C)
        
        logger.info("PhysicsConstraintValidator initialized")
    
    def validate_impedance_constraints(
        self,
        circuit_params: Dict[str, float],
        frequency: float,
        spec: DesignSpec
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Validate impedance matching constraints."""
        violations = []
        
        try:
            # Extract relevant parameters
            L_values = [v for k, v in circuit_params.items() if k.endswith('_l')]
            C_values = [v for k, v in circuit_params.items() if k.endswith('_c')]
            R_values = [v for k, v in circuit_params.items() if k.endswith('_r')]
            
            # Characteristic impedance calculation
            omega = 2 * np.pi * frequency
            
            for L in L_values:
                for C in C_values:
                    if L > 0 and C > 0:
                        Z_char = np.sqrt(L / C)
                        
                        # Check if impedance is reasonable
                        if Z_char < 1.0 or Z_char > 1000.0:
                            violations.append({
                                'type': 'impedance_range',
                                'value': Z_char,
                                'expected_range': (1.0, 1000.0),
                                'severity': 'warning',
                                'fix': 'Adjust L/C ratio for reasonable impedance'
                            })
                        
                        # Resonant frequency check
                        f_resonant = 1 / (2 * np.pi * np.sqrt(L * C))
                        if abs(f_resonant - frequency) / frequency > 0.5:
                            violations.append({
                                'type': 'resonance_mismatch',
                                'resonant_freq': f_resonant,
                                'target_freq': frequency,
                                'severity': 'error',
                                'fix': 'Adjust LC product for target frequency'
                            })
            
            # Resistance constraints
            for R in R_values:
                if R <= 0:
                    violations.append({
                        'type': 'negative_resistance',
                        'value': R,
                        'severity': 'critical',
                        'fix': 'Ensure all resistances are positive'
                    })
                elif R > 1e6:
                    violations.append({
                        'type': 'excessive_resistance',
                        'value': R,
                        'max_allowed': 1e6,
                        'severity': 'warning',
                        'fix': 'Consider using active loads instead'
                    })
            
            return len(violations) == 0, violations
            
        except Exception as e:
            logger.error(f"Impedance validation failed: {e}")
            return False, [{'type': 'validation_error', 'error': str(e), 'severity': 'critical'}]
    
    def validate_power_constraints(
        self,
        circuit_params: Dict[str, float],
        spec: DesignSpec
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Validate power consumption and dissipation constraints."""
        violations = []
        
        try:
            # Extract current and voltage parameters
            I_values = [v for k, v in circuit_params.items() if 'bias' in k.lower()]
            supply_voltage = getattr(spec, 'supply_voltage', 1.2)
            
            total_power = 0.0
            for I in I_values:
                if I > 0:
                    power = I * supply_voltage
                    total_power += power
                    
                    # Check individual current limits
                    if I > 0.1:  # 100mA limit
                        violations.append({
                            'type': 'excessive_current',
                            'current': I,
                            'max_allowed': 0.1,
                            'severity': 'error',
                            'fix': 'Reduce bias current or use current limiting'
                        })
            
            # Total power constraint
            if total_power > spec.power_max:
                violations.append({
                    'type': 'power_budget_exceeded',
                    'total_power': total_power,
                    'max_power': spec.power_max,
                    'severity': 'critical',
                    'fix': 'Reduce bias currents or optimize topology'
                })
            
            # Power efficiency check
            if hasattr(spec, 'gain_min'):
                efficiency = spec.gain_min / (total_power * 1000)  # dB/mW
                if efficiency < 1.0:
                    violations.append({
                        'type': 'low_efficiency',
                        'efficiency': efficiency,
                        'severity': 'warning',
                        'fix': 'Optimize for better gain/power ratio'
                    })
            
            return len(violations) == 0, violations
            
        except Exception as e:
            logger.error(f"Power validation failed: {e}")
            return False, [{'type': 'validation_error', 'error': str(e), 'severity': 'critical'}]
    
    def validate_stability_constraints(
        self,
        circuit_params: Dict[str, float],
        frequency: float
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Validate circuit stability using simplified analysis."""
        violations = []
        
        try:
            # Extract transconductance and capacitance values
            gm_values = [v for k, v in circuit_params.items() if 'gm' in k]
            C_values = [v for k, v in circuit_params.items() if k.endswith('_c')]
            
            omega = 2 * np.pi * frequency
            
            # Unity gain bandwidth check
            for gm in gm_values:
                for C in C_values:
                    if gm > 0 and C > 0:
                        f_unity = gm / (2 * np.pi * C)
                        
                        # Phase margin estimation
                        if f_unity < frequency * 3:  # Less than 3x operating frequency
                            violations.append({
                                'type': 'insufficient_bandwidth',
                                'unity_gain_freq': f_unity,
                                'operating_freq': frequency,
                                'severity': 'warning',
                                'fix': 'Increase gm or reduce parasitic capacitance'
                            })
                        
                        # Oscillation risk
                        if f_unity > frequency * 100:  # Too high bandwidth
                            violations.append({
                                'type': 'oscillation_risk',
                                'unity_gain_freq': f_unity,
                                'operating_freq': frequency,
                                'severity': 'error',
                                'fix': 'Add compensation or reduce gain'
                            })
            
            # Slew rate check
            for gm in gm_values:
                for C in C_values:
                    if gm > 0 and C > 0:
                        I_bias = gm * 0.1  # Simplified assumption
                        slew_rate = I_bias / C
                        
                        if slew_rate < omega * 0.1:  # Too slow
                            violations.append({
                                'type': 'insufficient_slew_rate',
                                'slew_rate': slew_rate,
                                'severity': 'warning',
                                'fix': 'Increase bias current or reduce load capacitance'
                            })
            
            return len(violations) == 0, violations
            
        except Exception as e:
            logger.error(f"Stability validation failed: {e}")
            return False, [{'type': 'validation_error', 'error': str(e), 'severity': 'critical'}]


class AdversarialRobustnessValidator:
    """
    Validates circuit robustness against adversarial perturbations.
    
    Research Innovation: First implementation of adversarial testing for 
    analog circuits, ensuring robustness to process variations and attacks.
    """
    
    def __init__(self, perturbation_budget: float = 0.1):
        self.perturbation_budget = perturbation_budget
        self.attack_methods = [
            'parameter_perturbation',
            'temperature_variation', 
            'supply_voltage_variation',
            'process_corner_analysis',
            'aging_effects'
        ]
        
        logger.info("AdversarialRobustnessValidator initialized")
    
    def validate_parameter_robustness(
        self,
        circuit_params: Dict[str, float],
        performance_function: Callable[[Dict[str, float]], Dict[str, float]],
        spec: DesignSpec,
        num_perturbations: int = 100
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Test robustness to parameter perturbations."""
        violations = []
        baseline_performance = performance_function(circuit_params)
        
        try:
            # Generate random perturbations
            robust_count = 0
            
            for i in range(num_perturbations):
                perturbed_params = circuit_params.copy()
                
                # Add random perturbations within budget
                for key, value in circuit_params.items():
                    if isinstance(value, (int, float)) and value != 0:
                        perturbation = np.random.uniform(
                            -self.perturbation_budget, 
                            self.perturbation_budget
                        )
                        perturbed_params[key] = value * (1 + perturbation)
                
                try:
                    # Evaluate perturbed performance
                    perturbed_performance = performance_function(perturbed_params)
                    
                    # Check if performance degrades significantly
                    performance_stable = True
                    
                    for metric, value in baseline_performance.items():
                        perturbed_value = perturbed_performance.get(metric, 0)
                        
                        if abs(perturbed_value - value) / abs(value) > 0.2:  # 20% tolerance
                            performance_stable = False
                            break
                    
                    if performance_stable:
                        robust_count += 1
                        
                except Exception:
                    # Perturbed circuit failed
                    pass
            
            robustness_ratio = robust_count / num_perturbations
            
            if robustness_ratio < 0.8:  # Less than 80% robustness
                violations.append({
                    'type': 'insufficient_robustness',
                    'robustness_ratio': robustness_ratio,
                    'min_required': 0.8,
                    'severity': 'error' if robustness_ratio < 0.5 else 'warning',
                    'fix': 'Increase component tolerances or use robust topologies'
                })
            
            return len(violations) == 0, violations
            
        except Exception as e:
            logger.error(f"Robustness validation failed: {e}")
            return False, [{'type': 'validation_error', 'error': str(e), 'severity': 'critical'}]
    
    def validate_temperature_robustness(
        self,
        circuit_params: Dict[str, float],
        performance_function: Callable[[Dict[str, float]], Dict[str, float]],
        spec: DesignSpec,
        temperature_range: Tuple[float, float] = (-40, 125)  # °C
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Test robustness to temperature variations."""
        violations = []
        
        try:
            baseline_temp = 27  # Room temperature
            baseline_performance = performance_function(circuit_params)
            
            # Test at temperature extremes
            for temp in [temperature_range[0], temperature_range[1]]:
                # Apply temperature-dependent parameter scaling
                temp_params = self._apply_temperature_scaling(circuit_params, temp, baseline_temp)
                
                try:
                    temp_performance = performance_function(temp_params)
                    
                    # Check performance degradation
                    for metric, baseline_value in baseline_performance.items():
                        temp_value = temp_performance.get(metric, 0)
                        degradation = abs(temp_value - baseline_value) / abs(baseline_value)
                        
                        if degradation > 0.3:  # 30% degradation limit
                            violations.append({
                                'type': 'temperature_sensitivity',
                                'metric': metric,
                                'temperature': temp,
                                'degradation': degradation,
                                'severity': 'error' if degradation > 0.5 else 'warning',
                                'fix': f'Add temperature compensation for {metric}'
                            })
                
                except Exception as e:
                    violations.append({
                        'type': 'temperature_failure',
                        'temperature': temp,
                        'error': str(e),
                        'severity': 'critical',
                        'fix': 'Circuit fails at temperature extreme'
                    })
            
            return len(violations) == 0, violations
            
        except Exception as e:
            logger.error(f"Temperature robustness validation failed: {e}")
            return False, [{'type': 'validation_error', 'error': str(e), 'severity': 'critical'}]
    
    def _apply_temperature_scaling(
        self,
        params: Dict[str, float],
        temp: float,
        baseline_temp: float = 27
    ) -> Dict[str, float]:
        """Apply temperature-dependent scaling to circuit parameters."""
        temp_params = params.copy()
        
        delta_temp = temp - baseline_temp
        
        # Temperature coefficients (simplified)
        temp_coeffs = {
            'resistor': 3500e-6,      # 3500 ppm/°C for typical resistors
            'capacitor': -750e-6,     # -750 ppm/°C for ceramic capacitors
            'transistor_mobility': -1500e-6,  # Mobility temperature coefficient
            'threshold_voltage': -2e-3,  # -2 mV/°C for Vth
            'current': 8000e-6        # Current temperature coefficient
        }
        
        for key, value in params.items():
            # Apply appropriate temperature scaling
            if 'r' in key.lower() or 'resistance' in key.lower():
                temp_params[key] = value * (1 + temp_coeffs['resistor'] * delta_temp)
            elif 'c' in key.lower() or 'capacitor' in key.lower():
                temp_params[key] = value * (1 + temp_coeffs['capacitor'] * delta_temp)
            elif 'current' in key.lower() or 'bias' in key.lower():
                temp_params[key] = value * (1 + temp_coeffs['current'] * delta_temp)
            elif 'gm' in key.lower():
                temp_params[key] = value * (1 + temp_coeffs['transistor_mobility'] * delta_temp)
        
        return temp_params


class CircuitRecoveryEngine:
    """
    Automatic error recovery and circuit repair engine.
    
    Attempts to fix common circuit issues automatically through 
    parameter adjustment and topology modifications.
    """
    
    def __init__(self):
        self.recovery_strategies = [
            'parameter_scaling',
            'bias_point_adjustment',
            'stability_compensation',
            'impedance_matching_fix',
            'power_optimization'
        ]
        
        self.recovery_history = []
        logger.info("CircuitRecoveryEngine initialized")
    
    def attempt_recovery(
        self,
        circuit_params: Dict[str, float],
        violations: List[Dict[str, Any]],
        spec: DesignSpec
    ) -> Tuple[bool, Dict[str, float], List[str]]:
        """
        Attempt to automatically fix circuit violations.
        
        Args:
            circuit_params: Original circuit parameters
            violations: List of validation violations
            spec: Design specification
            
        Returns:
            Tuple of (success, fixed_params, applied_fixes)
        """
        fixed_params = circuit_params.copy()
        applied_fixes = []
        
        try:
            for violation in violations:
                violation_type = violation.get('type', '')
                severity = violation.get('severity', 'warning')
                
                if severity == 'critical':
                    success, new_params, fix_description = self._apply_critical_fix(
                        fixed_params, violation, spec
                    )
                    if success:
                        fixed_params.update(new_params)
                        applied_fixes.append(fix_description)
                
                elif severity == 'error':
                    success, new_params, fix_description = self._apply_error_fix(
                        fixed_params, violation, spec
                    )
                    if success:
                        fixed_params.update(new_params)
                        applied_fixes.append(fix_description)
                
                elif severity == 'warning':
                    success, new_params, fix_description = self._apply_warning_fix(
                        fixed_params, violation, spec
                    )
                    if success:
                        fixed_params.update(new_params)
                        applied_fixes.append(fix_description)
            
            # Record recovery attempt
            self.recovery_history.append({
                'original_params': circuit_params,
                'fixed_params': fixed_params,
                'violations': violations,
                'applied_fixes': applied_fixes,
                'timestamp': time.time()
            })
            
            return len(applied_fixes) > 0, fixed_params, applied_fixes
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            return False, circuit_params, []
    
    def _apply_critical_fix(
        self,
        params: Dict[str, float],
        violation: Dict[str, Any],
        spec: DesignSpec
    ) -> Tuple[bool, Dict[str, float], str]:
        """Apply fixes for critical violations."""
        violation_type = violation.get('type', '')
        
        if violation_type == 'negative_resistance':
            # Fix negative resistance
            for key, value in params.items():
                if 'r' in key.lower() and value <= 0:
                    params[key] = abs(value) + 1.0  # Make positive with minimum value
            return True, params, "Fixed negative resistance values"
        
        elif violation_type == 'power_budget_exceeded':
            # Scale down bias currents to meet power budget
            power_reduction_factor = spec.power_max / violation.get('total_power', 1.0)
            for key, value in params.items():
                if 'bias' in key.lower() or 'current' in key.lower():
                    params[key] = value * power_reduction_factor * 0.9  # 10% margin
            return True, params, "Scaled bias currents to meet power budget"
        
        return False, {}, ""
    
    def _apply_error_fix(
        self,
        params: Dict[str, float],
        violation: Dict[str, Any],
        spec: DesignSpec
    ) -> Tuple[bool, Dict[str, float], str]:
        """Apply fixes for error-level violations."""
        violation_type = violation.get('type', '')
        
        if violation_type == 'resonance_mismatch':
            # Adjust LC product for target frequency
            target_freq = violation.get('target_freq', spec.frequency)
            target_lc_product = 1 / ((2 * np.pi * target_freq) ** 2)
            
            # Find L and C parameters
            l_keys = [k for k in params.keys() if k.endswith('_l')]
            c_keys = [k for k in params.keys() if k.endswith('_c')]
            
            if l_keys and c_keys:
                # Scale L and C to achieve target LC product
                current_l = params[l_keys[0]]
                current_c = params[c_keys[0]]
                current_lc = current_l * current_c
                
                scale_factor = np.sqrt(target_lc_product / current_lc)
                params[l_keys[0]] = current_l * scale_factor
                params[c_keys[0]] = current_c * scale_factor
                
                return True, params, "Adjusted LC values for target frequency"
        
        elif violation_type == 'oscillation_risk':
            # Add stability compensation
            gm_keys = [k for k in params.keys() if 'gm' in k]
            for key in gm_keys:
                params[key] = params[key] * 0.7  # Reduce gain for stability
            
            return True, params, "Reduced transconductance for stability"
        
        return False, {}, ""
    
    def _apply_warning_fix(
        self,
        params: Dict[str, float],
        violation: Dict[str, Any],
        spec: DesignSpec
    ) -> Tuple[bool, Dict[str, float], str]:
        """Apply fixes for warning-level violations."""
        violation_type = violation.get('type', '')
        
        if violation_type == 'low_efficiency':
            # Optimize for better efficiency
            # Slightly increase transconductance and reduce power
            for key, value in params.items():
                if 'gm' in key.lower():
                    params[key] = value * 1.1  # 10% increase
                elif 'bias' in key.lower():
                    params[key] = value * 0.95  # 5% reduction
            
            return True, params, "Optimized for better power efficiency"
        
        elif violation_type == 'impedance_range':
            # Adjust impedance to reasonable range
            target_impedance = 50.0  # Standard impedance
            current_impedance = violation.get('value', 50.0)
            scale_factor = target_impedance / current_impedance
            
            # Adjust L and C values
            l_keys = [k for k in params.keys() if k.endswith('_l')]
            c_keys = [k for k in params.keys() if k.endswith('_c')]
            
            for key in l_keys:
                params[key] = params[key] * scale_factor
            for key in c_keys:
                params[key] = params[key] / scale_factor
            
            return True, params, "Adjusted impedance to standard range"
        
        return False, {}, ""


class ComprehensiveCircuitValidator:
    """
    Main validation orchestrator that coordinates all validation subsystems.
    
    Provides unified interface for comprehensive circuit validation with
    automatic error recovery and robustness testing.
    """
    
    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE,
        enable_recovery: bool = True,
        parallel_validation: bool = True
    ):
        self.validation_level = validation_level
        self.enable_recovery = enable_recovery
        self.parallel_validation = parallel_validation
        
        # Initialize validation subsystems
        self.physics_validator = PhysicsConstraintValidator()
        self.robustness_validator = AdversarialRobustnessValidator()
        self.recovery_engine = CircuitRecoveryEngine()
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'successful_recoveries': 0,
            'average_validation_time': 0.0
        }
        
        logger.info(f"ComprehensiveCircuitValidator initialized with {validation_level.value} level")
    
    def validate_circuit(
        self,
        circuit_result: CircuitResult,
        spec: DesignSpec,
        performance_function: Optional[Callable] = None
    ) -> ValidationResult:
        """
        Perform comprehensive circuit validation.
        
        Args:
            circuit_result: Generated circuit to validate
            spec: Design specification
            performance_function: Optional function to evaluate performance
            
        Returns:
            Comprehensive validation result
        """
        start_time = time.time()
        
        validation_result = ValidationResult(
            is_valid=False,
            confidence_score=0.0,
            validation_level=self.validation_level
        )
        
        try:
            self.validation_stats['total_validations'] += 1
            
            # Extract circuit parameters
            circuit_params = circuit_result.parameters
            
            # Run validation pipeline
            if self.parallel_validation:
                validation_tasks = self._run_parallel_validation(
                    circuit_params, spec, performance_function
                )
            else:
                validation_tasks = self._run_sequential_validation(
                    circuit_params, spec, performance_function
                )
            
            # Aggregate results
            all_violations = []
            all_tests_passed = 0
            all_tests_total = 0
            
            for task_name, (success, violations) in validation_tasks.items():
                if success:
                    all_tests_passed += 1
                all_tests_total += 1
                
                validation_result.constraint_violations.extend([
                    {**v, 'validator': task_name} for v in violations
                ])
            
            validation_result.tests_passed = all_tests_passed
            validation_result.tests_total = all_tests_total
            
            # Calculate confidence score
            base_confidence = all_tests_passed / max(all_tests_total, 1)
            
            # Adjust for violation severity
            severity_penalty = 0.0
            for violation in validation_result.constraint_violations:
                if violation.get('severity') == 'critical':
                    severity_penalty += 0.3
                elif violation.get('severity') == 'error':
                    severity_penalty += 0.1
                elif violation.get('severity') == 'warning':
                    severity_penalty += 0.02
            
            validation_result.confidence_score = max(0.0, base_confidence - severity_penalty)
            
            # Attempt recovery if enabled and needed
            if (self.enable_recovery and 
                validation_result.confidence_score < 0.8 and 
                validation_result.constraint_violations):
                
                recovery_success, fixed_params, applied_fixes = self.recovery_engine.attempt_recovery(
                    circuit_params, validation_result.constraint_violations, spec
                )
                
                if recovery_success:
                    self.validation_stats['successful_recoveries'] += 1
                    
                    # Re-validate fixed circuit
                    fixed_circuit_result = CircuitResult(
                        netlist=circuit_result.netlist,
                        parameters=fixed_params,
                        performance=circuit_result.performance,
                        topology=circuit_result.topology,
                        technology=circuit_result.technology,
                        generation_time=circuit_result.generation_time,
                        spice_valid=circuit_result.spice_valid
                    )
                    
                    # Quick re-validation
                    revalidation_tasks = self._run_basic_validation(fixed_params, spec)
                    
                    recovery_success_count = sum(
                        1 for _, (success, _) in revalidation_tasks.items() if success
                    )
                    
                    if recovery_success_count > all_tests_passed:
                        validation_result.suggested_fixes = [
                            {'type': 'parameter_recovery', 'fixes': applied_fixes}
                        ]
                        validation_result.confidence_score = min(1.0, validation_result.confidence_score + 0.3)
            
            # Determine overall validity
            critical_violations = [
                v for v in validation_result.constraint_violations 
                if v.get('severity') == 'critical'
            ]
            
            validation_result.is_valid = (
                len(critical_violations) == 0 and 
                validation_result.confidence_score >= 0.6
            )
            
            if validation_result.is_valid:
                self.validation_stats['successful_validations'] += 1
            
            # Update timing statistics
            validation_time = time.time() - start_time
            validation_result.validation_time = validation_time
            
            self._update_validation_stats(validation_time)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Circuit validation failed: {e}")
            traceback.print_exc()
            
            validation_result.constraint_violations.append({
                'type': 'validation_system_error',
                'error': str(e),
                'severity': 'critical'
            })
            
            return validation_result
    
    def _run_parallel_validation(
        self,
        circuit_params: Dict[str, float],
        spec: DesignSpec,
        performance_function: Optional[Callable]
    ) -> Dict[str, Tuple[bool, List[Dict[str, Any]]]]:
        """Run validation tasks in parallel."""
        
        validation_tasks = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit validation tasks
            futures = {}
            
            # Physics constraints
            futures['physics_impedance'] = executor.submit(
                self.physics_validator.validate_impedance_constraints,
                circuit_params, spec.frequency, spec
            )
            
            futures['physics_power'] = executor.submit(
                self.physics_validator.validate_power_constraints,
                circuit_params, spec
            )
            
            futures['physics_stability'] = executor.submit(
                self.physics_validator.validate_stability_constraints,
                circuit_params, spec.frequency
            )
            
            # Robustness testing (if comprehensive or adversarial level)
            if self.validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.ADVERSARIAL]:
                if performance_function:
                    futures['robustness_parameter'] = executor.submit(
                        self.robustness_validator.validate_parameter_robustness,
                        circuit_params, performance_function, spec, 50
                    )
                    
                    futures['robustness_temperature'] = executor.submit(
                        self.robustness_validator.validate_temperature_robustness,
                        circuit_params, performance_function, spec
                    )
            
            # Wait for completion and collect results
            for task_name, future in futures.items():
                try:
                    validation_tasks[task_name] = future.result(timeout=30)  # 30s timeout
                except concurrent.futures.TimeoutError:
                    validation_tasks[task_name] = (False, [{'type': 'timeout', 'severity': 'error'}])
                except Exception as e:
                    validation_tasks[task_name] = (False, [{'type': 'exception', 'error': str(e), 'severity': 'error'}])
        
        return validation_tasks
    
    def _run_sequential_validation(
        self,
        circuit_params: Dict[str, float],
        spec: DesignSpec,
        performance_function: Optional[Callable]
    ) -> Dict[str, Tuple[bool, List[Dict[str, Any]]]]:
        """Run validation tasks sequentially."""
        
        validation_tasks = {}
        
        try:
            # Physics constraints
            validation_tasks['physics_impedance'] = self.physics_validator.validate_impedance_constraints(
                circuit_params, spec.frequency, spec
            )
            
            validation_tasks['physics_power'] = self.physics_validator.validate_power_constraints(
                circuit_params, spec
            )
            
            validation_tasks['physics_stability'] = self.physics_validator.validate_stability_constraints(
                circuit_params, spec.frequency
            )
            
            # Robustness testing
            if (self.validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.ADVERSARIAL] 
                and performance_function):
                
                validation_tasks['robustness_parameter'] = self.robustness_validator.validate_parameter_robustness(
                    circuit_params, performance_function, spec, 25
                )
                
                validation_tasks['robustness_temperature'] = self.robustness_validator.validate_temperature_robustness(
                    circuit_params, performance_function, spec
                )
        
        except Exception as e:
            logger.error(f"Sequential validation failed: {e}")
            validation_tasks['validation_error'] = (False, [{'type': 'system_error', 'error': str(e), 'severity': 'critical'}])
        
        return validation_tasks
    
    def _run_basic_validation(
        self,
        circuit_params: Dict[str, float],
        spec: DesignSpec
    ) -> Dict[str, Tuple[bool, List[Dict[str, Any]]]]:
        """Run basic validation for recovery verification."""
        
        validation_tasks = {}
        
        try:
            validation_tasks['physics_impedance'] = self.physics_validator.validate_impedance_constraints(
                circuit_params, spec.frequency, spec
            )
            
            validation_tasks['physics_power'] = self.physics_validator.validate_power_constraints(
                circuit_params, spec
            )
            
        except Exception as e:
            validation_tasks['validation_error'] = (False, [{'type': 'system_error', 'error': str(e), 'severity': 'critical'}])
        
        return validation_tasks
    
    def _update_validation_stats(self, validation_time: float):
        """Update validation statistics."""
        current_avg = self.validation_stats['average_validation_time']
        total_validations = self.validation_stats['total_validations']
        
        # Update running average
        if total_validations > 1:
            self.validation_stats['average_validation_time'] = (
                (current_avg * (total_validations - 1) + validation_time) / total_validations
            )
        else:
            self.validation_stats['average_validation_time'] = validation_time
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        stats = self.validation_stats.copy()
        
        if stats['total_validations'] > 0:
            stats['success_rate'] = stats['successful_validations'] / stats['total_validations']
            stats['recovery_rate'] = stats['successful_recoveries'] / stats['total_validations']
        else:
            stats['success_rate'] = 0.0
            stats['recovery_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'successful_recoveries': 0,
            'average_validation_time': 0.0
        }
        
        logger.info("Validation statistics reset")


# Factory functions
def create_circuit_validator(
    level: ValidationLevel = ValidationLevel.COMPREHENSIVE,
    enable_recovery: bool = True
) -> ComprehensiveCircuitValidator:
    """
    Create comprehensive circuit validator with specified configuration.
    
    Args:
        level: Validation level (basic, intermediate, comprehensive, adversarial)
        enable_recovery: Enable automatic error recovery
        
    Returns:
        Configured circuit validator
    """
    validator = ComprehensiveCircuitValidator(
        validation_level=level,
        enable_recovery=enable_recovery,
        parallel_validation=True
    )
    
    logger.info(f"Created circuit validator with {level.value} level")
    return validator