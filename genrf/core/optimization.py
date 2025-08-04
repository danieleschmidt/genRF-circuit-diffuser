"""
Bayesian optimization for circuit parameter tuning.

This module implements multi-objective Bayesian optimization for RF circuit
parameter optimization with Pareto front exploration.
"""

import numpy as np
from typing import Dict, List, Callable, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of Bayesian optimization."""
    best_parameters: Dict[str, float]
    best_objective: float
    optimization_history: List[Dict[str, Any]]
    n_iterations: int
    converged: bool


class AcquisitionFunction:
    """Base class for acquisition functions."""
    
    def __init__(self, model, xi: float = 0.01):
        self.model = model
        self.xi = xi  # Exploration parameter
    
    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function."""
    
    def __call__(self, x: np.ndarray, f_best: float) -> np.ndarray:
        """
        Calculate Expected Improvement.
        
        Args:
            x: Input parameters
            f_best: Current best objective value
            
        Returns:
            Expected improvement values
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        try:
            mu, sigma = self.model.predict(x, return_std=True)
            
            # Handle numerical issues
            sigma = np.maximum(sigma, 1e-9)
            
            # Calculate improvement
            improvement = mu - f_best - self.xi
            Z = improvement / sigma
            
            # Expected improvement
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            
            return ei.flatten()
            
        except Exception as e:
            logger.warning(f"Error in Expected Improvement calculation: {e}")
            return np.zeros(x.shape[0])


class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound acquisition function."""
    
    def __init__(self, model, kappa: float = 2.576):
        super().__init__(model)
        self.kappa = kappa  # Confidence parameter (2.576 for 99% confidence)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate Upper Confidence Bound.
        
        Args:
            x: Input parameters
            
        Returns:
            UCB values
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        try:
            mu, sigma = self.model.predict(x, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            
            ucb = mu + self.kappa * sigma
            return ucb.flatten()
            
        except Exception as e:
            logger.warning(f"Error in UCB calculation: {e}")
            return np.zeros(x.shape[0])


class GaussianProcessSurrogate:
    """
    Simplified Gaussian Process surrogate model.
    
    This is a basic implementation. For production use, consider using
    scikit-learn's GaussianProcessRegressor or GPy.
    """
    
    def __init__(self, kernel_type: str = 'rbf', noise_level: float = 1e-6):
        self.kernel_type = kernel_type
        self.noise_level = noise_level
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
        
        # Kernel hyperparameters
        self.length_scale = 1.0
        self.signal_variance = 1.0
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the Gaussian Process to training data."""
        self.X_train = np.atleast_2d(X)
        self.y_train = np.atleast_1d(y)
        self.is_fitted = True
        
        # Simple hyperparameter estimation
        if len(self.y_train) > 1:
            self.signal_variance = np.var(self.y_train)
            
            # Estimate length scale from data spread
            if self.X_train.shape[0] > 1:
                distances = []
                for i in range(self.X_train.shape[0]):
                    for j in range(i+1, self.X_train.shape[0]):
                        dist = np.linalg.norm(self.X_train[i] - self.X_train[j])
                        distances.append(dist)
                
                if distances:
                    self.length_scale = np.median(distances)
    
    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between X1 and X2."""
        if self.kernel_type == 'rbf':
            # RBF kernel
            sqdist = np.sum(X1**2, axis=1, keepdims=True) + \
                    np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return self.signal_variance * np.exp(-0.5 * sqdist / self.length_scale**2)
        else:
            raise NotImplementedError(f"Kernel {self.kernel_type} not implemented")
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions with the Gaussian Process."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.atleast_2d(X)
        
        if self.X_train.shape[0] == 0:
            # No training data
            mean = np.zeros(X.shape[0])
            std = np.ones(X.shape[0])
        elif self.X_train.shape[0] == 1:
            # Single training point
            mean = np.full(X.shape[0], self.y_train[0])
            std = np.ones(X.shape[0]) * np.sqrt(self.signal_variance)
        else:
            # Full GP prediction
            try:
                K = self._kernel(self.X_train, self.X_train)
                K += self.noise_level * np.eye(K.shape[0])  # Add noise
                
                K_star = self._kernel(self.X_train, X)
                K_star_star = self._kernel(X, X)
                
                # Solve for mean
                L = np.linalg.cholesky(K + 1e-6 * np.eye(K.shape[0]))
                alpha = np.linalg.solve(L, np.linalg.solve(L.T, self.y_train))
                mean = K_star.T @ alpha
                
                if return_std:
                    # Solve for variance
                    v = np.linalg.solve(L, K_star)
                    var = np.diag(K_star_star) - np.sum(v**2, axis=0)
                    var = np.maximum(var, 1e-9)  # Ensure positive variance
                    std = np.sqrt(var)
                
            except np.linalg.LinAlgError as e:
                logger.warning(f"Numerical issues in GP prediction: {e}")
                # Fall back to simple prediction
                mean = np.full(X.shape[0], np.mean(self.y_train))
                std = np.ones(X.shape[0]) * np.std(self.y_train) if len(self.y_train) > 1 else np.ones(X.shape[0])
        
        if return_std:
            return mean, std
        else:
            return mean


class BayesianOptimizer:
    """
    Bayesian optimizer for circuit parameter optimization.
    
    Uses Gaussian Process regression with acquisition functions to efficiently
    explore the parameter space and find optimal circuit configurations.
    """
    
    def __init__(
        self,
        acquisition_type: str = 'ei',
        n_initial_samples: int = 5,
        max_iterations: int = 50,
        convergence_threshold: float = 1e-6,
        acquisition_kappa: float = 2.576,
        xi: float = 0.01
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            acquisition_type: Type of acquisition function ('ei', 'ucb')
            n_initial_samples: Number of initial random samples
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold for objective
            acquisition_kappa: UCB exploration parameter
            xi: Expected improvement exploration parameter
        """
        self.acquisition_type = acquisition_type
        self.n_initial_samples = n_initial_samples
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Initialize surrogate model
        self.surrogate = GaussianProcessSurrogate()
        
        # Initialize acquisition function
        if acquisition_type == 'ei':
            self.acquisition_func = ExpectedImprovement(self.surrogate, xi=xi)
        elif acquisition_type == 'ucb':
            self.acquisition_func = UpperConfidenceBound(self.surrogate, kappa=acquisition_kappa)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_type}")
        
        # Optimization history
        self.history = []
        
        logger.info(f"Initialized BayesianOptimizer with {acquisition_type} acquisition")
    
    def optimize(
        self,
        objective_func: Callable[[Dict[str, float]], float],
        parameter_bounds: Dict[str, Tuple[float, float]],
        n_iterations: Optional[int] = None
    ) -> OptimizationResult:
        """
        Optimize objective function using Bayesian optimization.
        
        Args:
            objective_func: Function to maximize (circuit performance metric)
            parameter_bounds: Dictionary mapping parameter names to (min, max) bounds
            n_iterations: Number of optimization iterations (overrides default)
            
        Returns:
            OptimizationResult with best parameters and optimization history
        """
        if n_iterations is None:
            n_iterations = self.max_iterations
        
        # Extract parameter names and bounds
        param_names = list(parameter_bounds.keys())
        bounds = np.array([parameter_bounds[name] for name in param_names])
        
        logger.info(f"Starting Bayesian optimization with {len(param_names)} parameters")
        
        # Initialize with random samples
        X_samples = []
        y_samples = []
        
        logger.info(f"Generating {self.n_initial_samples} initial samples")
        for i in range(self.n_initial_samples):
            # Random sample within bounds
            x = np.random.uniform(bounds[:, 0], bounds[:, 1])
            params = dict(zip(param_names, x))
            
            try:
                y = objective_func(params)
                X_samples.append(x)
                y_samples.append(y)
                
                self.history.append({
                    'iteration': i,
                    'parameters': params.copy(),
                    'objective': y,
                    'type': 'initial'
                })
                
                logger.debug(f"Initial sample {i+1}: objective = {y:.4f}")
                
            except Exception as e:
                logger.warning(f"Error evaluating initial sample {i+1}: {e}")
                # Continue with other samples
        
        if not X_samples:
            raise RuntimeError("No valid initial samples obtained")
        
        X_samples = np.array(X_samples)
        y_samples = np.array(y_samples)
        
        # Main optimization loop
        best_y = np.max(y_samples)
        best_x = X_samples[np.argmax(y_samples)]
        
        converged = False
        
        for iteration in range(n_iterations):
            logger.debug(f"Bayesian optimization iteration {iteration + 1}/{n_iterations}")
            
            # Fit surrogate model
            self.surrogate.fit(X_samples, y_samples)
            
            # Optimize acquisition function
            try:
                next_x = self._optimize_acquisition(bounds, best_y)
                next_params = dict(zip(param_names, next_x))
                
                # Evaluate objective at new point
                next_y = objective_func(next_params)
                
                # Update samples
                X_samples = np.vstack([X_samples, next_x])
                y_samples = np.append(y_samples, next_y)
                
                # Update best
                if next_y > best_y:
                    improvement = next_y - best_y
                    best_y = next_y
                    best_x = next_x
                    
                    logger.info(f"New best found at iteration {iteration + 1}: "
                              f"objective = {best_y:.4f} (improvement: {improvement:.4f})")
                    
                    # Check convergence
                    if improvement < self.convergence_threshold:
                        logger.info(f"Converged after {iteration + 1} iterations")
                        converged = True
                        break
                
                self.history.append({
                    'iteration': len(self.history),
                    'parameters': next_params.copy(),
                    'objective': next_y,
                    'type': 'bayesian'
                })
                
            except Exception as e:
                logger.warning(f"Error in Bayesian optimization iteration {iteration + 1}: {e}")
                # Try random sample as fallback
                try:
                    next_x = np.random.uniform(bounds[:, 0], bounds[:, 1])
                    next_params = dict(zip(param_names, next_x))
                    next_y = objective_func(next_params)
                    
                    X_samples = np.vstack([X_samples, next_x])
                    y_samples = np.append(y_samples, next_y)
                    
                    if next_y > best_y:
                        best_y = next_y
                        best_x = next_x
                        
                    self.history.append({
                        'iteration': len(self.history),
                        'parameters': next_params.copy(),
                        'objective': next_y,
                        'type': 'random_fallback'
                    })
                    
                except Exception as e2:
                    logger.error(f"Even random fallback failed: {e2}")
                    break
        
        # Return results
        best_params = dict(zip(param_names, best_x))
        
        logger.info(f"Optimization completed. Best objective: {best_y:.4f}")
        
        return OptimizationResult(
            best_parameters=best_params,
            best_objective=best_y,
            optimization_history=self.history.copy(),
            n_iterations=len(self.history),
            converged=converged
        )
    
    def _optimize_acquisition(self, bounds: np.ndarray, f_best: float) -> np.ndarray:
        """Optimize acquisition function to find next sampling point."""
        
        def acquisition_objective(x):
            """Objective for acquisition optimization (minimize negative acquisition)."""
            try:
                if self.acquisition_type == 'ei':
                    return -self.acquisition_func(x, f_best)
                else:  # UCB
                    return -self.acquisition_func(x)
            except Exception as e:
                logger.debug(f"Error in acquisition function: {e}")
                return 0.0
        
        # Multiple random starts for global optimization
        n_starts = min(10, bounds.shape[0] * 2)
        best_x = None
        best_acq = float('inf')
        
        for _ in range(n_starts):
            # Random starting point
            x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
            
            try:
                # Local optimization
                result = minimize(
                    acquisition_objective,
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.success and result.fun < best_acq:
                    best_acq = result.fun
                    best_x = result.x
                    
            except Exception as e:
                logger.debug(f"Acquisition optimization failed: {e}")
                continue
        
        # Fallback to random point if optimization failed
        if best_x is None:
            logger.warning("Acquisition optimization failed, using random point")
            best_x = np.random.uniform(bounds[:, 0], bounds[:, 1])
        
        return best_x
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        if not self.history:
            return {}
        
        objectives = [entry['objective'] for entry in self.history]
        
        summary = {
            'n_evaluations': len(self.history),
            'best_objective': max(objectives),
            'worst_objective': min(objectives),
            'mean_objective': np.mean(objectives),
            'std_objective': np.std(objectives),
            'improvement_over_initial': max(objectives) - max(objectives[:self.n_initial_samples]) if len(objectives) > self.n_initial_samples else 0.0
        }
        
        return summary
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """Plot optimization convergence (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return
        
        if not self.history:
            logger.warning("No optimization history to plot")
            return
        
        iterations = [entry['iteration'] for entry in self.history]
        objectives = [entry['objective'] for entry in self.history]
        
        # Calculate running best
        running_best = []
        best_so_far = -float('inf')
        for obj in objectives:
            if obj > best_so_far:
                best_so_far = obj
            running_best.append(best_so_far)
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(iterations, objectives, 'bo-', alpha=0.7, label='Objective values')
        plt.plot(iterations, running_best, 'r-', linewidth=2, label='Best so far')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('Optimization Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        improvements = np.diff([0] + running_best)
        plt.semilogy(iterations[1:], np.maximum(improvements[1:], 1e-10), 'g-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Improvement (log scale)')
        plt.title('Improvement per Iteration')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {save_path}")
        else:
            plt.show()


# Multi-objective optimization extensions
class ParetoFrontOptimizer:
    """Multi-objective optimizer for Pareto front exploration."""
    
    def __init__(self, objectives: List[str], weights: Optional[Dict[str, float]] = None):
        """
        Initialize multi-objective optimizer.
        
        Args:
            objectives: List of objective names
            weights: Optional weights for scalarization
        """
        self.objectives = objectives
        self.weights = weights or {obj: 1.0 for obj in objectives}
        
        # Pareto front storage
        self.pareto_front = []
        
    def is_dominated(self, point1: Dict[str, float], point2: Dict[str, float]) -> bool:
        """Check if point1 is dominated by point2."""
        better_in_all = True
        strictly_better_in_some = False
        
        for obj in self.objectives:
            if point1[obj] < point2[obj]:
                better_in_all = False
            elif point1[obj] > point2[obj]:
                strictly_better_in_some = True
        
        return better_in_all and strictly_better_in_some
    
    def update_pareto_front(self, new_point: Dict[str, float]):
        """Update Pareto front with new point."""
        # Remove dominated points
        self.pareto_front = [
            point for point in self.pareto_front
            if not self.is_dominated(point, new_point)
        ]
        
        # Add new point if not dominated
        if not any(self.is_dominated(new_point, point) for point in self.pareto_front):
            self.pareto_front.append(new_point.copy())
    
    def scalarize_objectives(self, objectives: Dict[str, float]) -> float:
        """Convert multi-objective to single objective using weighted sum."""
        return sum(self.weights[obj] * objectives[obj] for obj in self.objectives)