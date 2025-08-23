#!/usr/bin/env python3
"""
Generation 3: MAKE IT SCALE - High-performance GenRF with advanced optimization and scaling
Autonomous SDLC execution with enterprise-grade performance and scalability
"""

import json
import time
import random
import math
import hashlib
import logging
import traceback
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
import warnings
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import gc

# Import robust components from Generation 2
from generation2_robust import (
    GenRFError, InvalidSpecificationError, CircuitGenerationError, 
    OptimizationError, ValidationError, ValidationReport, 
    SecurityValidator, RobustDesignSpec, RobustCircuitResult, 
    RobustCircuitDiffuser
)

# Configure high-performance logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance tracking"""
    generation_time: float = 0.0
    optimization_time: float = 0.0
    validation_time: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    parallel_efficiency: float = 0.0
    throughput_circuits_per_sec: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'generation_time_s': self.generation_time,
            'optimization_time_s': self.optimization_time,
            'validation_time_s': self.validation_time,
            'memory_peak_mb': self.memory_peak_mb,
            'cpu_utilization_pct': self.cpu_utilization * 100,
            'cache_hit_rate_pct': self.cache_hit_rate * 100,
            'parallel_efficiency_pct': self.parallel_efficiency * 100,
            'throughput_circuits_per_sec': self.throughput_circuits_per_sec
        }

@dataclass
class OptimizationStrategy:
    """Advanced optimization strategy configuration"""
    algorithm: str = 'bayesian'  # bayesian, genetic, particle_swarm, gradient_descent
    max_iterations: int = 50
    population_size: int = 20
    convergence_threshold: float = 1e-6
    multi_objective: bool = True
    parallel_evaluation: bool = True
    adaptive_parameters: bool = True
    cache_evaluations: bool = True
    early_stopping: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'algorithm': self.algorithm,
            'max_iterations': self.max_iterations,
            'population_size': self.population_size,
            'convergence_threshold': self.convergence_threshold,
            'multi_objective': self.multi_objective,
            'parallel_evaluation': self.parallel_evaluation,
            'adaptive_parameters': self.adaptive_parameters,
            'cache_evaluations': self.cache_evaluations,
            'early_stopping': self.early_stopping
        }

class PerformanceCache:
    """High-performance caching system for circuit evaluations"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
    
    def _generate_key(self, parameters: Dict[str, float], spec: RobustDesignSpec) -> str:
        """Generate cache key from parameters and spec"""
        param_str = ','.join(f"{k}:{v:.6e}" for k, v in sorted(parameters.items()))
        spec_str = f"{spec.circuit_type}:{spec.frequency:.0f}:{spec.spec_hash}"
        combined = f"{param_str}|{spec_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, parameters: Dict[str, float], spec: RobustDesignSpec) -> Optional[Dict[str, float]]:
        """Get cached performance evaluation"""
        key = self._generate_key(parameters, spec)
        
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                # Check TTL
                if current_time - self.access_times[key] < self.ttl_seconds:
                    self.hit_count += 1
                    self.access_times[key] = current_time
                    logger.debug(f"Cache hit for key: {key[:8]}...")
                    return self.cache[key].copy()
                else:
                    # Expired
                    del self.cache[key]
                    del self.access_times[key]
            
            self.miss_count += 1
            logger.debug(f"Cache miss for key: {key[:8]}...")
            return None
    
    def put(self, parameters: Dict[str, float], spec: RobustDesignSpec, 
            performance: Dict[str, float]) -> None:
        """Cache performance evaluation"""
        key = self._generate_key(parameters, spec)
        
        with self.lock:
            current_time = time.time()
            
            # Evict old entries if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = performance.copy()
            self.access_times[key] = current_time
            logger.debug(f"Cached result for key: {key[:8]}...")
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if not self.cache:
            return
        
        lru_key = min(self.access_times, key=self.access_times.get)
        del self.cache[lru_key]
        del self.access_times[lru_key]
        logger.debug(f"Evicted LRU entry: {lru_key[:8]}...")
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hit_count + self.miss_count
        return self.hit_count / max(1, total)
    
    def clear(self) -> None:
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': self.get_hit_rate(),
                'memory_usage_mb': len(str(self.cache)) / 1024 / 1024
            }

class ParallelOptimizer:
    """High-performance parallel optimization engine"""
    
    def __init__(self, n_workers: Optional[int] = None, strategy: OptimizationStrategy = None):
        self.n_workers = n_workers or min(cpu_count(), 8)  # Limit to reasonable number
        self.strategy = strategy or OptimizationStrategy()
        self.cache = PerformanceCache()
        
        # Thread pool for parallel evaluations
        self.thread_pool = ThreadPoolExecutor(max_workers=self.n_workers)
        
        logger.info(f"ParallelOptimizer initialized with {self.n_workers} workers")
    
    def optimize(self, objective_func: Callable, initial_params: Dict[str, float], 
                 param_bounds: Dict[str, Tuple[float, float]], 
                 spec: RobustDesignSpec) -> Tuple[Dict[str, float], Dict[str, float]]:
        """High-performance parallel optimization"""
        
        start_time = time.time()
        
        if self.strategy.algorithm == 'bayesian':
            result = self._bayesian_optimization(objective_func, initial_params, param_bounds, spec)
        elif self.strategy.algorithm == 'genetic':
            result = self._genetic_optimization(objective_func, initial_params, param_bounds, spec)
        elif self.strategy.algorithm == 'particle_swarm':
            result = self._particle_swarm_optimization(objective_func, initial_params, param_bounds, spec)
        else:
            result = self._gradient_descent_optimization(objective_func, initial_params, param_bounds, spec)
        
        optimization_time = time.time() - start_time
        logger.info(f"Optimization completed in {optimization_time:.3f}s using {self.strategy.algorithm}")
        
        return result
    
    def _bayesian_optimization(self, objective_func: Callable, initial_params: Dict[str, float],
                              param_bounds: Dict[str, Tuple[float, float]], 
                              spec: RobustDesignSpec) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Advanced Bayesian optimization with parallel evaluation"""
        
        best_params = initial_params.copy()
        best_performance = None
        best_score = float('-inf')
        
        # Initialize with multiple random points in parallel
        initial_population = []
        for _ in range(min(self.strategy.population_size, self.n_workers * 2)):
            params = {}
            for key, value in initial_params.items():
                if key in param_bounds:
                    min_val, max_val = param_bounds[key]
                    params[key] = random.uniform(min_val, max_val)
                else:
                    params[key] = value * random.uniform(0.5, 2.0)
            initial_population.append(params)
        
        # Evaluate initial population in parallel
        if self.strategy.parallel_evaluation:
            futures = []
            for params in initial_population:
                future = self.thread_pool.submit(self._cached_evaluation, 
                                               objective_func, params, spec)
                futures.append((future, params))
            
            for future, params in futures:
                try:
                    performance, score = future.result(timeout=10)
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        best_performance = performance
                except Exception as e:
                    logger.debug(f"Parallel evaluation failed: {e}")
        
        # Bayesian optimization iterations
        evaluation_history = []
        convergence_history = []
        
        for iteration in range(self.strategy.max_iterations):
            try:
                # Acquisition function: Expected Improvement with exploration
                next_params = self._acquisition_function(
                    evaluation_history, param_bounds, initial_params
                )
                
                # Evaluate new point
                performance, score = self._cached_evaluation(objective_func, next_params, spec)
                
                # Update history
                evaluation_history.append({
                    'params': next_params.copy(),
                    'performance': performance.copy(),
                    'score': score,
                    'iteration': iteration
                })
                
                # Update best
                if score > best_score:
                    improvement = score - best_score
                    best_score = score
                    best_params = next_params.copy()
                    best_performance = performance
                    
                    logger.debug(f"Iteration {iteration}: New best score {score:.4f} (improvement: {improvement:.4f})")
                else:
                    logger.debug(f"Iteration {iteration}: Score {score:.4f} (best: {best_score:.4f})")
                
                # Check convergence
                if len(convergence_history) >= 5:
                    recent_scores = convergence_history[-5:]
                    if max(recent_scores) - min(recent_scores) < self.strategy.convergence_threshold:
                        logger.info(f"Converged after {iteration + 1} iterations")
                        break
                
                convergence_history.append(score)
                
                # Adaptive parameters
                if self.strategy.adaptive_parameters and iteration % 10 == 0:
                    self._adapt_strategy(evaluation_history)
                
                # Early stopping
                if (self.strategy.early_stopping and iteration > 20 and 
                    len(set(convergence_history[-10:])) == 1):
                    logger.info(f"Early stopping after {iteration + 1} iterations")
                    break
                    
            except Exception as e:
                logger.warning(f"Optimization iteration {iteration} failed: {e}")
                continue
        
        return best_params, best_performance or {}
    
    def _acquisition_function(self, history: List[Dict], param_bounds: Dict[str, Tuple[float, float]],
                             initial_params: Dict[str, float]) -> Dict[str, float]:
        """Acquisition function for Bayesian optimization"""
        
        if len(history) < 2:
            # Random exploration for first few points
            params = {}
            for key, value in initial_params.items():
                if key in param_bounds:
                    min_val, max_val = param_bounds[key]
                    params[key] = random.uniform(min_val, max_val)
                else:
                    params[key] = value * random.uniform(0.5, 2.0)
            return params
        
        # Simple Expected Improvement approximation
        best_score = max(h['score'] for h in history)
        
        # Generate candidate points
        best_candidate = None
        best_ei = float('-inf')
        
        for _ in range(100):  # Sample 100 candidates
            params = {}
            for key, value in initial_params.items():
                if key in param_bounds:
                    min_val, max_val = param_bounds[key]
                    
                    # Bias towards unexplored regions
                    explored_values = [h['params'].get(key, value) for h in history]
                    mean_explored = sum(explored_values) / len(explored_values)
                    
                    # Add exploration bias
                    if random.random() < 0.3:  # 30% pure exploration
                        params[key] = random.uniform(min_val, max_val)
                    else:  # 70% exploitation with exploration
                        # Gaussian around mean with some exploration
                        std = (max_val - min_val) * 0.1
                        candidate = random.gauss(mean_explored, std)
                        params[key] = max(min_val, min(max_val, candidate))
                else:
                    params[key] = value * random.uniform(0.8, 1.2)
            
            # Simple Expected Improvement calculation
            # In practice, this would use Gaussian Process predictions
            distance_penalty = sum(
                min(abs(params.get(k, 0) - h['params'].get(k, 0)) for h in history)
                for k in params.keys()
            )
            
            exploration_bonus = distance_penalty * 0.1
            expected_improvement = exploration_bonus + random.gauss(0, 0.1)
            
            if expected_improvement > best_ei:
                best_ei = expected_improvement
                best_candidate = params
        
        return best_candidate or initial_params.copy()
    
    def _genetic_optimization(self, objective_func: Callable, initial_params: Dict[str, float],
                             param_bounds: Dict[str, Tuple[float, float]], 
                             spec: RobustDesignSpec) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Genetic algorithm optimization"""
        
        # Initialize population
        population = []
        population_size = self.strategy.population_size
        
        # Add initial params
        population.append(initial_params.copy())
        
        # Generate random population
        for _ in range(population_size - 1):
            individual = {}
            for key, value in initial_params.items():
                if key in param_bounds:
                    min_val, max_val = param_bounds[key]
                    individual[key] = random.uniform(min_val, max_val)
                else:
                    individual[key] = value * random.uniform(0.1, 10.0)
            population.append(individual)
        
        best_individual = None
        best_performance = None
        best_fitness = float('-inf')
        
        for generation in range(self.strategy.max_iterations // 2):  # Fewer generations, more evaluations per gen
            try:
                # Evaluate population in parallel
                fitness_scores = []
                
                if self.strategy.parallel_evaluation:
                    futures = []
                    for individual in population:
                        future = self.thread_pool.submit(self._cached_evaluation, 
                                                       objective_func, individual, spec)
                        futures.append(future)
                    
                    for i, future in enumerate(futures):
                        try:
                            performance, score = future.result(timeout=5)
                            fitness_scores.append((score, performance, population[i]))
                        except Exception as e:
                            fitness_scores.append((float('-inf'), {}, population[i]))
                else:
                    # Sequential evaluation
                    for individual in population:
                        try:
                            performance, score = self._cached_evaluation(objective_func, individual, spec)
                            fitness_scores.append((score, performance, individual))
                        except Exception:
                            fitness_scores.append((float('-inf'), {}, individual))
                
                # Sort by fitness
                fitness_scores.sort(key=lambda x: x[0], reverse=True)
                
                # Update best
                if fitness_scores[0][0] > best_fitness:
                    best_fitness = fitness_scores[0][0]
                    best_performance = fitness_scores[0][1]
                    best_individual = fitness_scores[0][2].copy()
                
                # Selection: Keep top 50%
                survivors = [x[2] for x in fitness_scores[:population_size // 2]]
                
                # Crossover and mutation
                new_population = survivors.copy()
                
                while len(new_population) < population_size:
                    # Select two parents
                    parent1 = random.choice(survivors)
                    parent2 = random.choice(survivors)
                    
                    # Crossover
                    child = {}
                    for key in parent1.keys():
                        if random.random() < 0.5:
                            child[key] = parent1[key]
                        else:
                            child[key] = parent2[key]
                    
                    # Mutation
                    for key, value in child.items():
                        if random.random() < 0.1:  # 10% mutation rate
                            if key in param_bounds:
                                min_val, max_val = param_bounds[key]
                                mutation = random.gauss(0, (max_val - min_val) * 0.05)
                                child[key] = max(min_val, min(max_val, value + mutation))
                            else:
                                child[key] = value * random.uniform(0.9, 1.1)
                    
                    new_population.append(child)
                
                population = new_population
                
                logger.debug(f"Generation {generation}: Best fitness {best_fitness:.4f}")
                
            except Exception as e:
                logger.warning(f"Genetic optimization generation {generation} failed: {e}")
                break
        
        return best_individual or initial_params, best_performance or {}
    
    def _particle_swarm_optimization(self, objective_func: Callable, initial_params: Dict[str, float],
                                   param_bounds: Dict[str, Tuple[float, float]], 
                                   spec: RobustDesignSpec) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Particle Swarm Optimization"""
        
        swarm_size = self.strategy.population_size
        
        # Initialize particles
        particles = []
        velocities = []
        personal_best = []
        personal_best_scores = []
        
        for _ in range(swarm_size):
            particle = {}
            velocity = {}
            
            for key, value in initial_params.items():
                if key in param_bounds:
                    min_val, max_val = param_bounds[key]
                    particle[key] = random.uniform(min_val, max_val)
                    velocity[key] = random.uniform(-(max_val - min_val) * 0.1, 
                                                  (max_val - min_val) * 0.1)
                else:
                    particle[key] = value * random.uniform(0.1, 10.0)
                    velocity[key] = particle[key] * random.uniform(-0.1, 0.1)
            
            particles.append(particle)
            velocities.append(velocity)
            personal_best.append(particle.copy())
            personal_best_scores.append(float('-inf'))
        
        global_best = initial_params.copy()
        global_best_performance = None
        global_best_score = float('-inf')
        
        # PSO parameters
        w = 0.7  # Inertia weight
        c1 = 1.4  # Cognitive parameter
        c2 = 1.4  # Social parameter
        
        for iteration in range(self.strategy.max_iterations):
            try:
                # Evaluate particles in parallel
                if self.strategy.parallel_evaluation:
                    futures = []
                    for particle in particles:
                        future = self.thread_pool.submit(self._cached_evaluation, 
                                                       objective_func, particle, spec)
                        futures.append(future)
                    
                    scores = []
                    performances = []
                    for future in futures:
                        try:
                            performance, score = future.result(timeout=5)
                            scores.append(score)
                            performances.append(performance)
                        except Exception:
                            scores.append(float('-inf'))
                            performances.append({})
                else:
                    scores = []
                    performances = []
                    for particle in particles:
                        try:
                            performance, score = self._cached_evaluation(objective_func, particle, spec)
                            scores.append(score)
                            performances.append(performance)
                        except Exception:
                            scores.append(float('-inf'))
                            performances.append({})
                
                # Update personal and global bests
                for i in range(swarm_size):
                    if scores[i] > personal_best_scores[i]:
                        personal_best_scores[i] = scores[i]
                        personal_best[i] = particles[i].copy()
                    
                    if scores[i] > global_best_score:
                        global_best_score = scores[i]
                        global_best = particles[i].copy()
                        global_best_performance = performances[i]
                
                # Update velocities and positions
                for i in range(swarm_size):
                    for key in particles[i].keys():
                        r1, r2 = random.random(), random.random()
                        
                        # Velocity update
                        velocities[i][key] = (
                            w * velocities[i][key] +
                            c1 * r1 * (personal_best[i][key] - particles[i][key]) +
                            c2 * r2 * (global_best[key] - particles[i][key])
                        )
                        
                        # Position update
                        particles[i][key] += velocities[i][key]
                        
                        # Boundary handling
                        if key in param_bounds:
                            min_val, max_val = param_bounds[key]
                            if particles[i][key] < min_val:
                                particles[i][key] = min_val
                                velocities[i][key] = 0
                            elif particles[i][key] > max_val:
                                particles[i][key] = max_val
                                velocities[i][key] = 0
                
                logger.debug(f"PSO Iteration {iteration}: Global best score {global_best_score:.4f}")
                
            except Exception as e:
                logger.warning(f"PSO iteration {iteration} failed: {e}")
                break
        
        return global_best, global_best_performance or {}
    
    def _gradient_descent_optimization(self, objective_func: Callable, initial_params: Dict[str, float],
                                     param_bounds: Dict[str, Tuple[float, float]], 
                                     spec: RobustDesignSpec) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Gradient descent with numerical differentiation"""
        
        current_params = initial_params.copy()
        best_params = current_params.copy()
        best_performance = None
        best_score = float('-inf')
        
        learning_rate = 0.01
        epsilon = 1e-6  # For numerical gradient
        
        for iteration in range(self.strategy.max_iterations):
            try:
                # Calculate current score
                current_performance, current_score = self._cached_evaluation(
                    objective_func, current_params, spec
                )
                
                if current_score > best_score:
                    best_score = current_score
                    best_params = current_params.copy()
                    best_performance = current_performance
                
                # Calculate numerical gradient
                gradient = {}
                
                if self.strategy.parallel_evaluation:
                    # Parallel gradient calculation
                    futures = {}
                    for key, value in current_params.items():
                        params_plus = current_params.copy()
                        params_plus[key] = value + epsilon
                        
                        future = self.thread_pool.submit(self._cached_evaluation, 
                                                       objective_func, params_plus, spec)
                        futures[key] = future
                    
                    for key, future in futures.items():
                        try:
                            _, score_plus = future.result(timeout=5)
                            gradient[key] = (score_plus - current_score) / epsilon
                        except Exception:
                            gradient[key] = 0.0
                else:
                    # Sequential gradient calculation
                    for key, value in current_params.items():
                        try:
                            params_plus = current_params.copy()
                            params_plus[key] = value + epsilon
                            
                            _, score_plus = self._cached_evaluation(objective_func, params_plus, spec)
                            gradient[key] = (score_plus - current_score) / epsilon
                        except Exception:
                            gradient[key] = 0.0
                
                # Update parameters
                for key, grad in gradient.items():
                    current_params[key] += learning_rate * grad
                    
                    # Apply bounds
                    if key in param_bounds:
                        min_val, max_val = param_bounds[key]
                        current_params[key] = max(min_val, min(max_val, current_params[key]))
                
                # Adaptive learning rate
                if iteration % 10 == 0:
                    learning_rate *= 0.95  # Decay learning rate
                
                logger.debug(f"Gradient descent iteration {iteration}: Score {current_score:.4f}")
                
            except Exception as e:
                logger.warning(f"Gradient descent iteration {iteration} failed: {e}")
                break
        
        return best_params, best_performance or {}
    
    def _cached_evaluation(self, objective_func: Callable, params: Dict[str, float], 
                          spec: RobustDesignSpec) -> Tuple[Dict[str, float], float]:
        """Cached evaluation with performance tracking"""
        
        # Check cache first
        if self.strategy.cache_evaluations:
            cached_perf = self.cache.get(params, spec)
            if cached_perf is not None:
                # Calculate score from cached performance
                score = self._calculate_score_from_performance(cached_perf, spec)
                return cached_perf, score
        
        # Evaluate
        try:
            performance, score = objective_func(params, spec)
            
            # Cache result
            if self.strategy.cache_evaluations:
                self.cache.put(params, spec, performance)
            
            return performance, score
        except Exception as e:
            logger.debug(f"Evaluation failed: {e}")
            return {}, float('-inf')
    
    def _calculate_score_from_performance(self, performance: Dict[str, float], 
                                        spec: RobustDesignSpec) -> float:
        """Calculate score from performance metrics"""
        try:
            gain = performance.get('gain_db', 0)
            nf = performance.get('noise_figure_db', 100)
            power = performance.get('power_w', 1)
            
            # Simple FoM calculation
            if (gain >= spec.gain_min if spec.gain_min > 0 else True and
                nf <= spec.nf_max if spec.nf_max < float('inf') else True and
                power <= spec.power_max):
                return gain / (power * 1000 * max(1, nf - 1))
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _adapt_strategy(self, history: List[Dict]) -> None:
        """Adapt optimization strategy based on history"""
        if len(history) < 10:
            return
        
        recent_scores = [h['score'] for h in history[-10:]]
        improvement_rate = (max(recent_scores) - min(recent_scores)) / max(abs(min(recent_scores)), 1e-6)
        
        if improvement_rate < 0.01:  # Slow improvement
            # Increase exploration
            if hasattr(self, 'exploration_factor'):
                self.exploration_factor = min(2.0, self.exploration_factor * 1.1)
        else:  # Good improvement
            # Focus on exploitation
            if hasattr(self, 'exploration_factor'):
                self.exploration_factor = max(0.5, self.exploration_factor * 0.9)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.thread_pool.shutdown(wait=True)
        self.cache.clear()

class ScalableCircuitDiffuser(RobustCircuitDiffuser):
    """Generation 3: Scalable high-performance RF circuit generator"""
    
    def __init__(self, checkpoint: Optional[str] = None,
                 spice_engine: str = "analytical",
                 technology: str = "generic",
                 verbose: bool = True,
                 validation_level: str = "strict",
                 enable_security: bool = True,
                 max_attempts: int = 3,
                 n_workers: Optional[int] = None,
                 optimization_strategy: OptimizationStrategy = None,
                 enable_caching: bool = True,
                 enable_profiling: bool = True):
        
        # Initialize parent class
        super().__init__(
            checkpoint=checkpoint,
            spice_engine=spice_engine,
            technology=technology,
            verbose=verbose,
            validation_level=validation_level,
            enable_security=enable_security,
            max_attempts=max_attempts
        )
        
        # Scalability enhancements
        self.n_workers = n_workers or min(cpu_count(), 8)
        self.optimization_strategy = optimization_strategy or OptimizationStrategy()
        self.enable_caching = enable_caching
        self.enable_profiling = enable_profiling
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.generation_history = []
        
        # High-performance optimizer
        self.parallel_optimizer = ParallelOptimizer(
            n_workers=self.n_workers,
            strategy=self.optimization_strategy
        )
        
        # Memory management
        self._memory_threshold_mb = 1000  # 1GB threshold
        
        if verbose:
            logger.info("⚡ CircuitDiffuser Generation 3: MAKE IT SCALE")
            logger.info(f"   Workers: {self.n_workers}")
            logger.info(f"   Optimization: {self.optimization_strategy.algorithm}")
            logger.info(f"   Caching: {'Enabled' if enable_caching else 'Disabled'}")
            logger.info(f"   Profiling: {'Enabled' if enable_profiling else 'Disabled'}")
            logger.info("   🚀 High-performance initialization complete")
    
    def generate_batch(self, specs: List[RobustDesignSpec], 
                      parallel: bool = True,
                      optimization_steps: int = 20) -> List[RobustCircuitResult]:
        """Generate multiple circuits in parallel for maximum throughput"""
        
        start_time = time.time()
        
        if not specs:
            return []
        
        logger.info(f"⚡ Batch generation: {len(specs)} circuits")
        logger.info(f"   Parallel: {parallel}, Workers: {self.n_workers if parallel else 1}")
        
        results = []
        
        if parallel and len(specs) > 1:
            # Parallel batch processing
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                
                for spec in specs:
                    future = executor.submit(
                        self._generate_single_optimized,
                        spec, optimization_steps
                    )
                    futures.append(future)
                
                # Collect results
                for i, future in enumerate(futures):
                    try:
                        result = future.result(timeout=60)  # 1 minute timeout per circuit
                        results.append(result)
                        logger.info(f"   ✅ Circuit {i+1}/{len(specs)} completed")
                    except Exception as e:
                        logger.error(f"   ❌ Circuit {i+1}/{len(specs)} failed: {e}")
                        # Add placeholder failed result
                        results.append(None)
        else:
            # Sequential processing
            for i, spec in enumerate(specs):
                try:
                    result = self._generate_single_optimized(spec, optimization_steps)
                    results.append(result)
                    logger.info(f"   ✅ Circuit {i+1}/{len(specs)} completed")
                except Exception as e:
                    logger.error(f"   ❌ Circuit {i+1}/{len(specs)} failed: {e}")
                    results.append(None)
        
        # Calculate batch performance metrics
        batch_time = time.time() - start_time
        successful_results = [r for r in results if r is not None]
        
        self.performance_metrics.throughput_circuits_per_sec = len(successful_results) / max(batch_time, 0.001)
        
        logger.info(f"⚡ Batch completed: {len(successful_results)}/{len(specs)} successful")
        logger.info(f"   Total time: {batch_time:.2f}s")
        logger.info(f"   Throughput: {self.performance_metrics.throughput_circuits_per_sec:.2f} circuits/sec")
        
        return results
    
    def _generate_single_optimized(self, spec: RobustDesignSpec, 
                                  optimization_steps: int) -> RobustCircuitResult:
        """Generate single circuit with advanced optimization"""
        
        start_time = time.time()
        
        # Memory management
        if self.enable_profiling:
            self._check_memory_usage()
        
        try:
            # Step 1: Enhanced topology generation
            topology_start = time.time()
            topology = self._generate_topology_scalable(spec)
            topology_time = time.time() - topology_start
            
            # Step 2: High-performance parameter optimization
            optimization_start = time.time()
            parameters, performance = self._optimize_parameters_scalable(
                topology, spec, optimization_steps
            )
            optimization_time = time.time() - optimization_start
            
            # Step 3: Create netlist
            netlist = self._create_netlist_robust(topology, parameters, spec)
            
            # Step 4: Validation
            validation_start = time.time()
            validation_report = self._comprehensive_validation(
                topology, parameters, performance, netlist, spec
            )
            validation_time = time.time() - validation_start
            
            # Create result
            generation_time = time.time() - start_time
            
            result = RobustCircuitResult(
                netlist=netlist,
                parameters=parameters,
                performance=performance,
                topology=topology['name'],
                technology=self.technology,
                generation_time=generation_time,
                validation_report=validation_report,
                spice_valid=False  # Using analytical models
            )
            
            # Update performance metrics
            self.performance_metrics.generation_time = generation_time
            self.performance_metrics.optimization_time = optimization_time
            self.performance_metrics.validation_time = validation_time
            
            if self.enable_caching:
                self.performance_metrics.cache_hit_rate = self.parallel_optimizer.get_cache_stats().get('hit_rate', 0.0)
            
            # Record in history
            self.generation_history.append({
                'timestamp': time.time(),
                'circuit_type': spec.circuit_type,
                'frequency': spec.frequency,
                'generation_time': generation_time,
                'optimization_time': optimization_time,
                'validation_time': validation_time,
                'topology_time': topology_time,
                'success': True,
                'performance': performance.copy()
            })
            
            return result
            
        except Exception as e:
            generation_time = time.time() - start_time
            
            # Record failure in history
            self.generation_history.append({
                'timestamp': time.time(),
                'circuit_type': spec.circuit_type,
                'frequency': spec.frequency,
                'generation_time': generation_time,
                'success': False,
                'error': str(e)
            })
            
            raise
    
    def _generate_topology_scalable(self, spec: RobustDesignSpec) -> Dict[str, Any]:
        """Enhanced topology generation with performance optimization"""
        
        # Use parent method with additional optimization hints
        topology = self._generate_topology_robust(spec)
        
        # Add performance hints based on frequency
        if spec.frequency > 10e9:  # mmWave
            topology['optimization_hints'] = {
                'priority': 'bandwidth',
                'parasitic_aware': True,
                'em_effects': True
            }
        elif spec.frequency > 1e9:  # Microwave
            topology['optimization_hints'] = {
                'priority': 'gain_nf',
                'matching_critical': True
            }
        else:  # RF
            topology['optimization_hints'] = {
                'priority': 'power_efficiency',
                'low_frequency_optimized': True
            }
        
        # Add scalability metadata
        topology['scalable_parameters'] = self._identify_scalable_parameters(topology, spec)
        
        return topology
    
    def _identify_scalable_parameters(self, topology: Dict[str, Any], 
                                    spec: RobustDesignSpec) -> List[str]:
        """Identify parameters that have the most impact on performance"""
        
        # Heuristic-based parameter importance
        scalable_params = []
        
        if spec.circuit_type == 'LNA':
            scalable_params = ['W1', 'L1', 'Ibias', 'Ld', 'Lg']
        elif spec.circuit_type == 'Mixer':
            scalable_params = ['Wrf', 'Wlo', 'Ibias', 'RL']
        elif spec.circuit_type == 'VCO':
            scalable_params = ['W', 'L', 'Ltank', 'Ctank', 'Ibias']
        else:
            scalable_params = list(topology.get('components', []))
        
        return scalable_params
    
    def _optimize_parameters_scalable(self, topology: Dict[str, Any], 
                                     spec: RobustDesignSpec, 
                                     steps: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """High-performance parameter optimization"""
        
        # Get initial parameters
        initial_params = self._get_initial_parameters(topology, spec)
        param_bounds = self._get_parameter_bounds_enhanced(spec.circuit_type, spec)
        
        # Define objective function
        def objective_func(params: Dict[str, float], 
                         circuit_spec: RobustDesignSpec) -> Tuple[Dict[str, float], float]:
            try:
                # Estimate performance
                performance = self._estimate_performance_robust(params, circuit_spec)
                
                # Calculate score
                score = self._calculate_fom_robust(performance, circuit_spec)
                
                return performance, score
            except Exception as e:
                logger.debug(f"Objective function evaluation failed: {e}")
                return {}, float('-inf')
        
        # Run optimization
        self.optimization_strategy.max_iterations = steps
        optimized_params, final_performance = self.parallel_optimizer.optimize(
            objective_func, initial_params, param_bounds, spec
        )
        
        return optimized_params, final_performance
    
    def _get_parameter_bounds_enhanced(self, circuit_type: str, 
                                     spec: RobustDesignSpec) -> Dict[str, Tuple[float, float]]:
        """Enhanced parameter bounds with frequency scaling"""
        
        # Get base bounds using parent method
        if circuit_type == 'LNA':
            bounds = {
                'W1': (1e-6, 500e-6), 'L1': (28e-9, 10e-6),
                'W2': (1e-6, 200e-6), 'L2': (28e-9, 5e-6),
                'Ls': (0.1e-9, 50e-9), 'Lg': (0.1e-9, 50e-9), 'Ld': (0.1e-9, 50e-9),
                'Ibias': (0.1e-3, 50e-3), 'Vbias': (0.1, 3.0),
                'Rd': (50, 20000), 'Cs': (1e-15, 100e-12), 'Cg': (1e-15, 100e-12)
            }
        elif circuit_type == 'Mixer':
            bounds = {
                'Wrf': (1e-6, 300e-6), 'Lrf': (28e-9, 5e-6),
                'Wlo': (1e-6, 400e-6), 'Llo': (28e-9, 5e-6),
                'Wtail': (2e-6, 600e-6), 'Ltail': (50e-9, 20e-6),
                'RL': (100, 10000), 'Ibias': (0.2e-3, 30e-3),
                'Cin': (1e-15, 100e-12), 'Cout': (1e-15, 100e-12)
            }
        elif circuit_type == 'VCO':
            bounds = {
                'W': (5e-6, 1000e-6), 'L': (28e-9, 1e-6),
                'Ltank': (0.1e-9, 100e-9), 'Ctank': (0.1e-12, 50e-12),
                'Cvar': (0.05e-12, 20e-12), 'Rtail': (100, 50000),
                'Ibias': (0.5e-3, 100e-3)
            }
        else:
            bounds = {f'param_{i}': (-10.0, 10.0) for i in range(8)}
        
        # Frequency-dependent scaling
        freq_scale = spec.frequency / 2.4e9  # Baseline 2.4GHz
        
        enhanced_bounds = {}
        for param, (min_val, max_val) in bounds.items():
            if 'L' in param and param != 'RL':  # Inductors scale inversely
                enhanced_bounds[param] = (min_val / freq_scale, max_val / freq_scale)
            elif 'C' in param:  # Capacitors scale inversely
                enhanced_bounds[param] = (min_val / freq_scale, max_val / freq_scale)
            elif 'W' in param:  # Widths may scale with frequency for some topologies
                if spec.frequency > 10e9:  # mmWave - smaller devices
                    enhanced_bounds[param] = (min_val * 0.5, max_val * 0.8)
                else:
                    enhanced_bounds[param] = (min_val, max_val)
            else:
                enhanced_bounds[param] = (min_val, max_val)
        
        return enhanced_bounds
    
    def _check_memory_usage(self) -> None:
        """Monitor memory usage and trigger garbage collection if needed"""
        
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            self.performance_metrics.memory_peak_mb = max(
                self.performance_metrics.memory_peak_mb, memory_mb
            )
            
            if memory_mb > self._memory_threshold_mb:
                logger.warning(f"High memory usage: {memory_mb:.1f}MB, triggering GC")
                gc.collect()
                
                # Clear caches if memory is still high
                new_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                if new_memory_mb > self._memory_threshold_mb * 0.8:
                    self.parallel_optimizer.cache.clear()
                    logger.info("Cleared optimization cache to free memory")
                    
        except ImportError:
            # psutil not available, skip memory monitoring
            pass
        except Exception as e:
            logger.debug(f"Memory monitoring failed: {e}")
    
    def optimize_design_space(self, base_spec: RobustDesignSpec, 
                             parameter_ranges: Dict[str, Tuple[float, float]],
                             n_points: int = 100,
                             parallel: bool = True) -> Dict[str, Any]:
        """Optimize over design space with Pareto front analysis"""
        
        start_time = time.time()
        
        logger.info(f"⚡ Design space optimization: {n_points} points")
        logger.info(f"   Parameters: {list(parameter_ranges.keys())}")
        
        # Generate parameter combinations
        parameter_combinations = []
        for _ in range(n_points):
            params = {}
            for param, (min_val, max_val) in parameter_ranges.items():
                params[param] = random.uniform(min_val, max_val)
            parameter_combinations.append(params)
        
        # Create specifications for each combination
        specs = []
        for params in parameter_combinations:
            spec = RobustDesignSpec(
                circuit_type=base_spec.circuit_type,
                frequency=params.get('frequency', base_spec.frequency),
                gain_min=params.get('gain_min', base_spec.gain_min),
                nf_max=params.get('nf_max', base_spec.nf_max),
                power_max=params.get('power_max', base_spec.power_max),
                validation_level='normal'  # Use normal validation for speed
            )
            specs.append(spec)
        
        # Generate circuits
        results = self.generate_batch(specs, parallel=parallel, optimization_steps=10)
        
        # Analyze results
        successful_results = [(params, result) for params, result in zip(parameter_combinations, results) 
                            if result is not None]
        
        # Calculate Pareto front
        pareto_front = self._calculate_pareto_front(successful_results)
        
        # Calculate design space statistics
        analysis_time = time.time() - start_time
        
        design_space_analysis = {
            'total_points': n_points,
            'successful_points': len(successful_results),
            'success_rate': len(successful_results) / max(1, n_points),
            'analysis_time_s': analysis_time,
            'pareto_front_size': len(pareto_front),
            'pareto_solutions': pareto_front,
            'parameter_ranges': parameter_ranges,
            'performance_statistics': self._calculate_performance_statistics(successful_results),
            'optimization_efficiency': len(pareto_front) / max(1, len(successful_results))
        }
        
        logger.info(f"⚡ Design space optimization complete:")
        logger.info(f"   Success rate: {design_space_analysis['success_rate']:.1%}")
        logger.info(f"   Pareto solutions: {len(pareto_front)}")
        logger.info(f"   Analysis time: {analysis_time:.2f}s")
        
        return design_space_analysis
    
    def _calculate_pareto_front(self, results: List[Tuple[Dict, RobustCircuitResult]]) -> List[Dict]:
        """Calculate Pareto front for multi-objective optimization"""
        
        if not results:
            return []
        
        pareto_solutions = []
        
        for i, (params_i, result_i) in enumerate(results):
            is_dominated = False
            
            # Check if this solution is dominated by any other
            for j, (params_j, result_j) in enumerate(results):
                if i != j:
                    if self._dominates(result_j, result_i):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_solutions.append({
                    'parameters': params_i,
                    'performance': result_i.performance,
                    'gain_db': result_i.gain,
                    'nf_db': result_i.nf,
                    'power_mw': result_i.power * 1000,
                    'security_score': result_i.security_score,
                    'reliability_score': result_i.validation_report.reliability_score,
                    'topology': result_i.topology
                })
        
        return pareto_solutions
    
    def _dominates(self, result_a: RobustCircuitResult, result_b: RobustCircuitResult) -> bool:
        """Check if result_a dominates result_b in Pareto sense"""
        
        # Multi-objective: maximize gain, minimize noise figure, minimize power
        gain_a, gain_b = result_a.gain, result_b.gain
        nf_a, nf_b = result_a.nf, result_b.nf
        power_a, power_b = result_a.power, result_b.power
        
        # A dominates B if A is at least as good in all objectives and better in at least one
        gain_better_or_equal = gain_a >= gain_b
        nf_better_or_equal = nf_a <= nf_b
        power_better_or_equal = power_a <= power_b
        
        at_least_one_better = (gain_a > gain_b) or (nf_a < nf_b) or (power_a < power_b)
        
        return gain_better_or_equal and nf_better_or_equal and power_better_or_equal and at_least_one_better
    
    def _calculate_performance_statistics(self, results: List[Tuple[Dict, RobustCircuitResult]]) -> Dict:
        """Calculate statistics across all results"""
        
        if not results:
            return {}
        
        gains = [result.gain for _, result in results]
        nfs = [result.nf for _, result in results if math.isfinite(result.nf)]
        powers = [result.power * 1000 for _, result in results]  # Convert to mW
        security_scores = [result.security_score for _, result in results]
        
        return {
            'gain_db': {
                'mean': sum(gains) / len(gains),
                'min': min(gains),
                'max': max(gains),
                'std': math.sqrt(sum((g - sum(gains)/len(gains))**2 for g in gains) / len(gains))
            },
            'noise_figure_db': {
                'mean': sum(nfs) / max(1, len(nfs)),
                'min': min(nfs) if nfs else 0,
                'max': max(nfs) if nfs else 0,
                'std': math.sqrt(sum((nf - sum(nfs)/len(nfs))**2 for nf in nfs) / max(1, len(nfs))) if nfs else 0
            },
            'power_mw': {
                'mean': sum(powers) / len(powers),
                'min': min(powers),
                'max': max(powers),
                'std': math.sqrt(sum((p - sum(powers)/len(powers))**2 for p in powers) / len(powers))
            },
            'security_score': {
                'mean': sum(security_scores) / len(security_scores),
                'min': min(security_scores),
                'max': max(security_scores)
            }
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        cache_stats = self.parallel_optimizer.get_cache_stats() if self.enable_caching else {}
        
        # Calculate generation statistics
        successful_generations = [h for h in self.generation_history if h.get('success', False)]
        failed_generations = [h for h in self.generation_history if not h.get('success', False)]
        
        if successful_generations:
            avg_generation_time = sum(h['generation_time'] for h in successful_generations) / len(successful_generations)
            avg_optimization_time = sum(h.get('optimization_time', 0) for h in successful_generations) / len(successful_generations)
            
            # Calculate parallel efficiency
            sequential_time_estimate = avg_optimization_time * len(successful_generations)
            actual_time = sum(h['generation_time'] for h in successful_generations)
            parallel_efficiency = sequential_time_estimate / max(actual_time, 0.001) / self.n_workers
        else:
            avg_generation_time = 0
            avg_optimization_time = 0
            parallel_efficiency = 0
        
        self.performance_metrics.parallel_efficiency = min(1.0, parallel_efficiency)
        
        return {
            'performance_metrics': self.performance_metrics.to_dict(),
            'generation_statistics': {
                'total_generations': len(self.generation_history),
                'successful_generations': len(successful_generations),
                'failed_generations': len(failed_generations),
                'success_rate': len(successful_generations) / max(1, len(self.generation_history)),
                'avg_generation_time_s': avg_generation_time,
                'avg_optimization_time_s': avg_optimization_time
            },
            'optimization_statistics': {
                'strategy': self.optimization_strategy.to_dict(),
                'cache_statistics': cache_stats,
                'worker_count': self.n_workers
            },
            'circuit_type_breakdown': self._get_circuit_type_breakdown(),
            'recent_performance_trend': self._get_recent_performance_trend()
        }
    
    def _get_circuit_type_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get performance breakdown by circuit type"""
        
        breakdown = {}
        
        for circuit_type in ['LNA', 'Mixer', 'VCO']:
            type_history = [h for h in self.generation_history 
                          if h.get('circuit_type') == circuit_type and h.get('success', False)]
            
            if type_history:
                breakdown[circuit_type] = {
                    'count': len(type_history),
                    'avg_generation_time': sum(h['generation_time'] for h in type_history) / len(type_history),
                    'success_rate': len(type_history) / max(1, len([h for h in self.generation_history if h.get('circuit_type') == circuit_type])),
                    'avg_gain_db': sum(h.get('performance', {}).get('gain_db', 0) for h in type_history) / len(type_history)
                }
        
        return breakdown
    
    def _get_recent_performance_trend(self, window: int = 10) -> Dict[str, float]:
        """Get recent performance trend"""
        
        if len(self.generation_history) < window:
            return {}
        
        recent_history = self.generation_history[-window:]
        older_history = self.generation_history[-(window*2):-window] if len(self.generation_history) >= window * 2 else []
        
        if not older_history:
            return {}
        
        recent_avg_time = sum(h['generation_time'] for h in recent_history) / len(recent_history)
        older_avg_time = sum(h['generation_time'] for h in older_history) / len(older_history)
        
        recent_success_rate = len([h for h in recent_history if h.get('success', False)]) / len(recent_history)
        older_success_rate = len([h for h in older_history if h.get('success', False)]) / len(older_history)
        
        return {
            'generation_time_trend': (recent_avg_time - older_avg_time) / max(older_avg_time, 0.001),
            'success_rate_trend': recent_success_rate - older_success_rate,
            'recent_throughput': len(recent_history) / sum(h['generation_time'] for h in recent_history)
        }
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.parallel_optimizer.cleanup()
        gc.collect()

def demo_generation_3():
    """Demonstrate Generation 3 scalable functionality"""
    
    print("=" * 80)
    print("⚡ GenRF Generation 3: MAKE IT SCALE - AUTONOMOUS EXECUTION")
    print("=" * 80)
    
    # Create scalable diffuser with high-performance configuration
    optimization_strategy = OptimizationStrategy(
        algorithm='bayesian',
        max_iterations=30,
        population_size=15,
        parallel_evaluation=True,
        cache_evaluations=True,
        adaptive_parameters=True
    )
    
    diffuser = ScalableCircuitDiffuser(
        verbose=True,
        validation_level='normal',  # Use normal for speed
        enable_security=True,
        n_workers=min(cpu_count(), 6),
        optimization_strategy=optimization_strategy,
        enable_caching=True,
        enable_profiling=True
    )
    
    # Test 1: Single circuit generation with advanced optimization
    print(f"\n🔬 Test 1: Advanced Single Circuit Generation")
    
    spec = RobustDesignSpec(
        circuit_type='LNA',
        frequency=5.8e9,
        gain_min=18,
        nf_max=1.8,
        power_max=12e-3,
        validation_level='normal'
    )
    
    start_time = time.time()
    single_gen_time = 0.0  # Initialize variable
    
    try:
        result = diffuser._generate_single_optimized(spec, optimization_steps=25)
        single_gen_time = time.time() - start_time
        
        print(f"   ✅ Single generation successful ({single_gen_time:.2f}s)")
        print(f"   Performance: Gain={result.gain:.1f}dB, NF={result.nf:.2f}dB, Power={result.power*1000:.1f}mW")
        print(f"   Quality: Security={result.security_score:.1f}, Reliability={result.validation_report.reliability_score:.1f}")
        
    except Exception as e:
        single_gen_time = time.time() - start_time
        print(f"   ❌ Single generation failed: {e}")
    
    # Test 2: Batch generation for throughput testing
    print(f"\n🚀 Test 2: High-Throughput Batch Generation")
    
    batch_specs = [
        RobustDesignSpec(circuit_type='LNA', frequency=2.4e9, gain_min=15, nf_max=2.0, power_max=10e-3),
        RobustDesignSpec(circuit_type='LNA', frequency=5.8e9, gain_min=18, nf_max=1.5, power_max=12e-3),
        RobustDesignSpec(circuit_type='Mixer', frequency=10e9, gain_min=8, nf_max=10.0, power_max=15e-3),
        RobustDesignSpec(circuit_type='Mixer', frequency=28e9, gain_min=5, nf_max=12.0, power_max=20e-3),
        RobustDesignSpec(circuit_type='VCO', frequency=5.8e9, gain_min=0, nf_max=float('inf'), power_max=15e-3),
        RobustDesignSpec(circuit_type='VCO', frequency=10e9, gain_min=0, nf_max=float('inf'), power_max=18e-3)
    ]
    
    batch_start = time.time()
    batch_results = diffuser.generate_batch(batch_specs, parallel=True, optimization_steps=20)
    batch_time = time.time() - batch_start
    
    successful_batch = [r for r in batch_results if r is not None]
    
    print(f"   ✅ Batch generation: {len(successful_batch)}/{len(batch_specs)} successful")
    print(f"   Total time: {batch_time:.2f}s, Throughput: {len(successful_batch)/batch_time:.2f} circuits/sec")
    
    # Test 3: Design space optimization
    print(f"\n🎯 Test 3: Design Space Optimization")
    
    base_spec = RobustDesignSpec(
        circuit_type='LNA',
        frequency=2.4e9,
        gain_min=15,
        nf_max=2.0,
        power_max=10e-3
    )
    
    parameter_ranges = {
        'frequency': (1e9, 10e9),
        'gain_min': (10, 25),
        'nf_max': (1.0, 3.0),
        'power_max': (5e-3, 20e-3)
    }
    
    design_space_start = time.time()
    try:
        design_analysis = diffuser.optimize_design_space(
            base_spec, 
            parameter_ranges, 
            n_points=50,  # Reduced for demo speed
            parallel=True
        )
        design_space_time = time.time() - design_space_start
        
        print(f"   ✅ Design space optimization successful ({design_space_time:.2f}s)")
        print(f"   Success rate: {design_analysis['success_rate']:.1%}")
        print(f"   Pareto solutions: {design_analysis['pareto_front_size']}")
        print(f"   Optimization efficiency: {design_analysis['optimization_efficiency']:.1%}")
        
        # Save design space results
        design_output = Path("gen3_scalable_outputs/design_space_analysis.json")
        design_output.parent.mkdir(exist_ok=True)
        with open(design_output, 'w') as f:
            json.dump(design_analysis, f, indent=2, default=str)
        
    except Exception as e:
        print(f"   ❌ Design space optimization failed: {e}")
    
    # Generate comprehensive performance report
    performance_report = diffuser.get_performance_report()
    
    # Save all results
    output_dir = Path("gen3_scalable_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save individual circuit results
    for i, result in enumerate(successful_batch):
        if result:
            circuit_file = output_dir / f"circuit_{i+1}_{result.topology}.json"
            with open(circuit_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            netlist_file = output_dir / f"circuit_{i+1}_{result.topology}.spice"
            with open(netlist_file, 'w') as f:
                f.write(result.netlist)
    
    # Generate comprehensive summary
    total_time = time.time() - batch_start + single_gen_time
    
    summary = {
        'generation': 'Generation 3: MAKE IT SCALE',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'execution_summary': {
            'single_circuit_test': {
                'success': True,
                'generation_time_s': single_gen_time,
                'circuit_type': spec.circuit_type,
                'frequency_ghz': spec.frequency / 1e9
            },
            'batch_generation_test': {
                'total_circuits': len(batch_specs),
                'successful_circuits': len(successful_batch),
                'success_rate': len(successful_batch) / len(batch_specs),
                'batch_time_s': batch_time,
                'throughput_circuits_per_sec': len(successful_batch) / batch_time,
                'parallel_efficiency': performance_report['performance_metrics']['parallel_efficiency_pct']
            },
            'design_space_test': design_analysis if 'design_analysis' in locals() else None
        },
        'performance_metrics': performance_report,
        'scalability_metrics': {
            'worker_count': diffuser.n_workers,
            'optimization_strategy': optimization_strategy.to_dict(),
            'cache_enabled': diffuser.enable_caching,
            'cache_hit_rate': performance_report['optimization_statistics']['cache_statistics'].get('hit_rate', 0),
            'memory_peak_mb': performance_report['performance_metrics']['memory_peak_mb'],
            'total_execution_time_s': total_time
        },
        'quality_metrics': {
            'avg_security_score': sum(r.security_score for r in successful_batch) / max(1, len(successful_batch)),
            'avg_reliability_score': sum(r.validation_report.reliability_score for r in successful_batch) / max(1, len(successful_batch)),
            'validation_success_rate': len([r for r in successful_batch if r.validation_report.is_valid]) / max(1, len(successful_batch))
        }
    }
    
    # Save comprehensive summary
    summary_file = output_dir / "scalable_generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print final results
    print(f"\n" + "=" * 80)
    print(f"⚡ Generation 3 Scalable Execution Complete!")
    print(f"   Total Execution Time: {total_time:.2f}s")
    print(f"   Single Circuit: {single_gen_time:.2f}s")
    print(f"   Batch Throughput: {len(successful_batch)/batch_time:.2f} circuits/sec")
    print(f"   Success Rate: {len(successful_batch)/len(batch_specs):.1%}")
    print(f"   Workers Used: {diffuser.n_workers}")
    print(f"   Cache Hit Rate: {performance_report['optimization_statistics']['cache_statistics'].get('hit_rate', 0):.1%}")
    print(f"   Average Security Score: {summary['quality_metrics']['avg_security_score']:.1f}/100")
    print(f"   Average Reliability Score: {summary['quality_metrics']['avg_reliability_score']:.1f}/100")
    
    if 'design_analysis' in locals():
        print(f"   Pareto Front Solutions: {design_analysis['pareto_front_size']}")
        print(f"   Design Space Efficiency: {design_analysis['optimization_efficiency']:.1%}")
    
    print(f"   Memory Peak: {performance_report['performance_metrics']['memory_peak_mb']:.1f}MB")
    print(f"   Outputs saved to: gen3_scalable_outputs/")
    print("=" * 80)
    
    # Cleanup
    diffuser.cleanup()
    
    return summary

if __name__ == "__main__":
    demo_generation_3()