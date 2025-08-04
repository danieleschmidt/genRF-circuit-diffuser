"""
Performance optimization utilities for GenRF.

This module provides tools for profiling, performance monitoring,
and optimization of the circuit generation pipeline.
"""

import time
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Callable, Tuple, Union, Iterator
import functools
import cProfile
import pstats
import io
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import logging
import psutil
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .cache import model_cache, simulation_cache
from .logging_config import performance_logger

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    operation: str
    duration_seconds: float
    memory_usage_mb: float
    cpu_percent: float
    success: bool = True
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation': self.operation,
            'duration_seconds': self.duration_seconds,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_percent': self.cpu_percent,
            'success': self.success,
            'additional_metrics': self.additional_metrics,
            'timestamp': self.timestamp.isoformat()
        }


class PerformanceProfiler:
    """
    Performance profiler for GenRF operations.
    
    Provides detailed profiling information including execution time,
    memory usage, and CPU utilization.
    """
    
    def __init__(self, enable_memory_profiling: bool = True):
        """Initialize performance profiler."""
        self.enable_memory_profiling = enable_memory_profiling
        self.profiles = []
        self.max_profiles = 1000
        
        logger.info("PerformanceProfiler initialized")
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        start_time = time.time()
        start_memory = self._get_memory_usage() if self.enable_memory_profiling else 0.0
        process = psutil.Process() if self.enable_memory_profiling else None
        
        success = True
        additional_metrics = {}
        
        try:
            if process:
                process.cpu_percent()  # Initialize CPU monitoring
            
            yield additional_metrics
            
        except Exception as e:
            success = False
            additional_metrics['error'] = str(e)
            raise
            
        finally:
            duration = time.time() - start_time
            end_memory = self._get_memory_usage() if self.enable_memory_profiling else 0.0
            memory_delta = max(0, end_memory - start_memory)
            cpu_percent = process.cpu_percent() if process else 0.0
            
            metrics = PerformanceMetrics(
                operation=operation_name,
                duration_seconds=duration,
                memory_usage_mb=memory_delta,
                cpu_percent=cpu_percent,
                success=success,
                additional_metrics=additional_metrics
            )
            
            self._record_metrics(metrics)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.profiles.append(metrics)
        
        # Limit stored profiles
        if len(self.profiles) > self.max_profiles:
            self.profiles.pop(0)
        
        # Log metrics
        performance_logger.logger.info(
            f"Performance: {metrics.operation} - {metrics.duration_seconds:.3f}s",
            extra={'structured_data': metrics.to_dict()}
        )
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        operation_profiles = [p for p in self.profiles if p.operation == operation_name]
        
        if not operation_profiles:
            return {'error': f'No profiles found for operation: {operation_name}'}
        
        durations = [p.duration_seconds for p in operation_profiles if p.success]
        memory_usage = [p.memory_usage_mb for p in operation_profiles if p.success]
        
        if not durations:
            return {'error': 'No successful operations found'}
        
        return {
            'operation': operation_name,
            'count': len(operation_profiles),
            'success_count': len(durations),
            'success_rate': len(durations) / len(operation_profiles) * 100,
            'duration_stats': {
                'mean': np.mean(durations),
                'std': np.std(durations),
                'min': np.min(durations),
                'max': np.max(durations),
                'p50': np.percentile(durations, 50),
                'p95': np.percentile(durations, 95),
                'p99': np.percentile(durations, 99)
            },
            'memory_stats': {
                'mean': np.mean(memory_usage),
                'std': np.std(memory_usage),
                'min': np.min(memory_usage),
                'max': np.max(memory_usage)
            } if memory_usage else {}
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        if not self.profiles:
            return {'error': 'No performance data available'}
        
        operations = set(p.operation for p in self.profiles)
        summary = {
            'total_operations': len(self.profiles),
            'unique_operations': len(operations),
            'success_rate': len([p for p in self.profiles if p.success]) / len(self.profiles) * 100,
            'operations': {}
        }
        
        for operation in operations:
            summary['operations'][operation] = self.get_operation_stats(operation)
        
        return summary


def profile_function(operation_name: Optional[str] = None):
    """Decorator for profiling functions."""
    def decorator(func: Callable) -> Callable:
        name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profiler.profile_operation(name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class BatchProcessor:
    """
    Efficient batch processing for circuit generation tasks.
    
    Processes multiple circuits in parallel using threading or multiprocessing
    based on the task characteristics.
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False,
        batch_size: int = 10
    ):
        """
        Initialize batch processor.
        
        Args:
            max_workers: Maximum number of workers (default: CPU count)
            use_processes: Use processes instead of threads
            batch_size: Size of processing batches
        """
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) * 2)
        self.use_processes = use_processes
        self.batch_size = batch_size
        
        logger.info(f"BatchProcessor initialized: {self.max_workers} workers, "
                   f"{'processes' if use_processes else 'threads'}")
    
    def process_batch(
        self,
        items: List[Any],
        processor_func: Callable[[Any], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Tuple[Any, Optional[Exception]]]:
        """
        Process items in batch with parallel execution.
        
        Args:
            items: Items to process
            processor_func: Function to process each item
            progress_callback: Optional progress callback
            
        Returns:
            List of (result, exception) tuples
        """
        results = []
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self._safe_process, processor_func, item): i
                for i, item in enumerate(items)
            }
            
            completed = 0
            for future in as_completed(future_to_item):
                item_index = future_to_item[future]
                result, exception = future.result()
                
                results.append((item_index, result, exception))
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, len(items))
        
        # Sort results by original order
        results.sort(key=lambda x: x[0])
        return [(result, exception) for _, result, exception in results]
    
    def _safe_process(self, processor_func: Callable, item: Any) -> Tuple[Any, Optional[Exception]]:
        """Safely process item with exception handling."""
        try:
            result = processor_func(item)
            return result, None
        except Exception as e:
            logger.error(f"Error processing item: {e}")
            return None, e
    
    def process_circuits_parallel(
        self,
        specs: List[Any],
        generator_func: Callable[[Any], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Tuple[Any, Optional[Exception]]]:
        """Process multiple circuit specifications in parallel."""
        logger.info(f"Processing {len(specs)} circuits in parallel")
        
        with profiler.profile_operation("batch_circuit_generation") as metrics:
            results = self.process_batch(specs, generator_func, progress_callback)
            
            success_count = sum(1 for _, exception in results if exception is None)
            metrics['success_count'] = success_count
            metrics['total_count'] = len(specs)
            metrics['success_rate'] = success_count / len(specs) * 100
        
        return results


class ModelOptimizer:
    """Optimization utilities for AI models."""
    
    def __init__(self):
        """Initialize model optimizer."""
        self.optimizations_applied = []
        logger.info("ModelOptimizer initialized")
    
    def optimize_model_for_inference(self, model: Any) -> Any:
        """Optimize model for faster inference."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping model optimization")
            return model
        
        optimized_model = model
        
        try:
            # Set to evaluation mode
            if hasattr(model, 'eval'):
                optimized_model.eval()
                self.optimizations_applied.append('eval_mode')
            
            # Disable gradient computation
            if hasattr(model, 'requires_grad_'):
                for param in model.parameters():
                    param.requires_grad_(False)
                self.optimizations_applied.append('no_grad')
            
            # Try TorchScript optimization
            if hasattr(torch, 'jit') and hasattr(model, 'forward'):
                try:
                    # Create dummy input for tracing
                    dummy_input = torch.randn(1, 100)  # Adjust based on model
                    traced_model = torch.jit.trace(model, dummy_input)
                    optimized_model = traced_model
                    self.optimizations_applied.append('torchscript_trace')
                except Exception as e:
                    logger.debug(f"TorchScript optimization failed: {e}")
            
            # Apply inference-specific optimizations
            if torch.cuda.is_available():
                try:
                    optimized_model = optimized_model.cuda()
                    self.optimizations_applied.append('cuda')
                except Exception as e:
                    logger.debug(f"CUDA optimization failed: {e}")
            
            logger.info(f"Model optimizations applied: {self.optimizations_applied}")
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model
        
        return optimized_model
    
    def benchmark_model(
        self,
        model: Any,
        input_shapes: List[Tuple[int, ...]],
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """Benchmark model performance."""
        if not TORCH_AVAILABLE:
            return {'error': 'PyTorch not available'}
        
        results = {}
        
        for i, input_shape in enumerate(input_shapes):
            shape_key = f"input_shape_{i}"
            
            try:
                # Create test input
                test_input = torch.randn(*input_shape)
                if torch.cuda.is_available() and hasattr(model, 'cuda'):
                    test_input = test_input.cuda()
                
                # Warmup
                with torch.no_grad():
                    for _ in range(10):
                        _ = model(test_input)
                
                # Benchmark
                times = []
                with torch.no_grad():
                    for _ in range(num_iterations):
                        start_time = time.time()
                        _ = model(test_input)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        times.append(time.time() - start_time)
                
                results[shape_key] = {
                    'input_shape': input_shape,
                    'mean_time_ms': np.mean(times) * 1000,
                    'std_time_ms': np.std(times) * 1000,
                    'min_time_ms': np.min(times) * 1000,
                    'max_time_ms': np.max(times) * 1000,
                    'throughput_per_sec': 1.0 / np.mean(times)
                }
                
            except Exception as e:
                results[shape_key] = {'error': str(e)}
        
        return results


class ResourceManager:
    """Manages computational resources for optimal performance."""
    
    def __init__(self):
        """Initialize resource manager."""
        self.cpu_count = mp.cpu_count()
        self.available_memory_gb = psutil.virtual_memory().total / 1024**3
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        
        logger.info(f"ResourceManager initialized: {self.cpu_count} CPUs, "
                   f"{self.available_memory_gb:.1f}GB RAM, {self.gpu_count} GPUs")
    
    def get_optimal_batch_size(
        self,
        task_type: str,
        memory_per_item_mb: float = 100.0,
        target_memory_usage: float = 0.8
    ) -> int:
        """Calculate optimal batch size based on available resources."""
        # Calculate based on memory constraints
        available_memory_mb = self.available_memory_gb * 1024 * target_memory_usage
        memory_based_batch_size = int(available_memory_mb / memory_per_item_mb)
        
        # Calculate based on CPU cores
        cpu_based_batch_size = self.cpu_count * 2
        
        # Use conservative estimate
        optimal_batch_size = min(memory_based_batch_size, cpu_based_batch_size, 100)
        
        logger.info(f"Optimal batch size for {task_type}: {optimal_batch_size}")
        return max(1, optimal_batch_size)
    
    def get_recommended_workers(self, task_type: str) -> int:
        """Get recommended number of workers for task type."""
        if task_type in ['model_inference', 'cpu_intensive']:
            return min(self.cpu_count, 8)  # CPU-bound tasks
        elif task_type in ['io_bound', 'simulation']:
            return min(self.cpu_count * 2, 16)  # I/O-bound tasks
        else:
            return min(self.cpu_count, 4)  # Conservative default
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        memory = psutil.virtual_memory()
        
        status = {
            'total_gb': memory.total / 1024**3,
            'available_gb': memory.available / 1024**3,
            'used_percent': memory.percent,
            'free_gb': memory.free / 1024**3
        }
        
        if self.gpu_available:
            gpu_status = []
            for i in range(self.gpu_count):
                try:
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    gpu_status.append({
                        'device_id': i,
                        'allocated_gb': memory_allocated,
                        'reserved_gb': memory_reserved
                    })
                except Exception as e:
                    gpu_status.append({'device_id': i, 'error': str(e)})
            
            status['gpu'] = gpu_status
        
        return status


# Global instances
profiler = PerformanceProfiler()
batch_processor = BatchProcessor()
model_optimizer = ModelOptimizer()
resource_manager = ResourceManager()


def get_performance_summary() -> Dict[str, Any]:
    """Get overall performance summary."""
    return {
        'profiler': profiler.get_summary(),
        'cache_stats': {
            'model_cache': model_cache.get_stats(),
            'simulation_cache': simulation_cache.memory_cache.get_stats()
        },
        'resource_status': resource_manager.get_memory_status()
    }


def optimize_for_production():
    """Apply production optimizations."""
    logger.info("Applying production optimizations...")
    
    # Clear any cached data that might not be needed
    import gc
    gc.collect()
    
    # Set optimal batch sizes based on available resources
    optimal_batch_size = resource_manager.get_optimal_batch_size('circuit_generation')
    batch_processor.batch_size = optimal_batch_size
    
    logger.info("Production optimizations applied")