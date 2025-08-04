"""
Concurrent processing and resource pooling for GenRF.

This module provides advanced concurrency primitives, resource pooling,
and distributed processing capabilities for scalable circuit generation.
"""

import asyncio
import threading
import multiprocessing as mp
import queue
import time
import weakref
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Awaitable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import psutil
from contextlib import contextmanager, asynccontextmanager
import functools

from .exceptions import ResourceError
from .monitoring import system_monitor
from .performance import profiler

logger = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    """Statistics for worker processes/threads."""
    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_time_seconds: float = 0.0
    average_time_seconds: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)
    memory_usage_mb: float = 0.0
    
    def update_completion(self, duration: float, success: bool = True):
        """Update statistics after task completion."""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        
        self.total_time_seconds += duration
        total_tasks = self.tasks_completed + self.tasks_failed
        self.average_time_seconds = self.total_time_seconds / max(1, total_tasks)
        self.last_activity = datetime.now()


class ResourcePool:
    """
    Generic resource pool for managing expensive-to-create resources.
    
    Supports GPU contexts, model instances, simulation engines, etc.
    """
    
    def __init__(
        self,
        resource_factory: Callable[[], Any],
        max_size: int = 10,
        min_size: int = 1,
        idle_timeout: float = 300.0,  # 5 minutes
        health_check: Optional[Callable[[Any], bool]] = None
    ):
        """
        Initialize resource pool.
        
        Args:
            resource_factory: Function to create new resources
            max_size: Maximum number of resources in pool
            min_size: Minimum number of resources to maintain
            idle_timeout: Timeout for idle resources (seconds)
            health_check: Optional health check function
        """
        self.resource_factory = resource_factory
        self.max_size = max_size
        self.min_size = min_size
        self.idle_timeout = idle_timeout
        self.health_check = health_check
        
        self._pool = queue.Queue()
        self._active_resources = weakref.WeakSet()
        self._resource_metadata = {}
        self._lock = threading.RLock()
        self._shutdown = False
        
        # Statistics
        self._created_count = 0
        self._destroyed_count = 0
        self._checkout_count = 0
        self._checkin_count = 0
        
        # Initialize minimum resources
        self._initialize_pool()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"ResourcePool initialized: max_size={max_size}, min_size={min_size}")
    
    def _initialize_pool(self):
        """Initialize pool with minimum resources."""
        for _ in range(self.min_size):
            try:
                resource = self._create_resource()
                self._pool.put(resource)
            except Exception as e:
                logger.error(f"Failed to initialize resource: {e}")
    
    def _create_resource(self) -> Any:
        """Create a new resource."""
        try:
            resource = self.resource_factory()
            resource_id = id(resource)
            
            self._resource_metadata[resource_id] = {
                'created_at': datetime.now(),
                'last_used': datetime.now(),
                'use_count': 0
            }
            
            self._active_resources.add(resource)
            self._created_count += 1
            
            logger.debug(f"Created new resource: {resource_id}")
            return resource
            
        except Exception as e:
            logger.error(f"Failed to create resource: {e}")
            raise ResourceError(f"Resource creation failed: {e}")
    
    def _destroy_resource(self, resource: Any):
        """Destroy a resource."""
        try:
            resource_id = id(resource)
            
            # Call cleanup method if available
            if hasattr(resource, 'close'):
                resource.close()
            elif hasattr(resource, 'cleanup'):
                resource.cleanup()
            
            # Remove from tracking
            if resource_id in self._resource_metadata:
                del self._resource_metadata[resource_id]
            
            self._destroyed_count += 1
            logger.debug(f"Destroyed resource: {resource_id}")
            
        except Exception as e:
            logger.error(f"Error destroying resource: {e}")
    
    @contextmanager
    def get_resource(self, timeout: float = 30.0):
        """
        Get resource from pool with automatic return.
        
        Args:
            timeout: Timeout for acquiring resource
            
        Yields:
            Resource instance
        """
        resource = None
        try:
            resource = self.checkout_resource(timeout)
            yield resource
        finally:
            if resource is not None:
                self.checkin_resource(resource)
    
    def checkout_resource(self, timeout: float = 30.0) -> Any:
        """
        Checkout resource from pool.
        
        Args:
            timeout: Timeout for acquiring resource
            
        Returns:
            Resource instance
            
        Raises:
            ResourceError: If no resource available within timeout
        """
        if self._shutdown:
            raise ResourceError("Resource pool is shutdown")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try to get existing resource
                resource = self._pool.get_nowait()
                
                # Health check
                if self.health_check and not self.health_check(resource):
                    self._destroy_resource(resource)
                    continue
                
                # Update metadata
                resource_id = id(resource)
                if resource_id in self._resource_metadata:
                    self._resource_metadata[resource_id]['last_used'] = datetime.now()
                    self._resource_metadata[resource_id]['use_count'] += 1
                
                self._checkout_count += 1
                logger.debug(f"Checked out resource: {resource_id}")
                return resource
                
            except queue.Empty:
                # No resources available, try to create new one
                with self._lock:
                    if len(self._active_resources) < self.max_size:
                        try:
                            resource = self._create_resource()
                            self._checkout_count += 1
                            return resource
                        except Exception as e:
                            logger.warning(f"Failed to create resource: {e}")
                
                # Wait a bit before retrying
                time.sleep(0.1)
        
        raise ResourceError(f"No resource available within {timeout} seconds")
    
    def checkin_resource(self, resource: Any):
        """
        Return resource to pool.
        
        Args:
            resource: Resource to return
        """
        if self._shutdown:
            self._destroy_resource(resource)
            return
        
        try:
            resource_id = id(resource)
            
            # Health check before returning
            if self.health_check and not self.health_check(resource):
                self._destroy_resource(resource)
                return
            
            # Return to pool
            self._pool.put_nowait(resource)
            self._checkin_count += 1
            
            logger.debug(f"Checked in resource: {resource_id}")
            
        except queue.Full:
            # Pool is full, destroy resource
            self._destroy_resource(resource)
        except Exception as e:
            logger.error(f"Error checking in resource: {e}")
            self._destroy_resource(resource)
    
    def _cleanup_loop(self):
        """Background cleanup of idle resources."""
        while not self._shutdown:
            try:
                self._cleanup_idle_resources()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def _cleanup_idle_resources(self):
        """Remove idle resources beyond minimum."""
        if self._pool.qsize() <= self.min_size:
            return
        
        current_time = datetime.now()
        resources_to_destroy = []
        
        # Check for idle resources
        temp_resources = []
        while True:
            try:
                resource = self._pool.get_nowait()
                resource_id = id(resource)
                
                metadata = self._resource_metadata.get(resource_id, {})
                last_used = metadata.get('last_used', current_time)
                
                if (current_time - last_used).total_seconds() > self.idle_timeout:
                    resources_to_destroy.append(resource)
                else:
                    temp_resources.append(resource)
                    
            except queue.Empty:
                break
        
        # Return non-idle resources to pool
        for resource in temp_resources:
            try:
                self._pool.put_nowait(resource)
            except queue.Full:
                resources_to_destroy.append(resource)
        
        # Destroy idle resources (keeping minimum)
        pool_size = len(temp_resources)
        for resource in resources_to_destroy:
            if pool_size > self.min_size:
                self._destroy_resource(resource)
                pool_size -= 1
            else:
                try:
                    self._pool.put_nowait(resource)
                except queue.Full:
                    self._destroy_resource(resource)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'pool_size': self._pool.qsize(),
            'active_resources': len(self._active_resources),
            'max_size': self.max_size,
            'min_size': self.min_size,
            'created_count': self._created_count,
            'destroyed_count': self._destroyed_count,
            'checkout_count': self._checkout_count,
            'checkin_count': self._checkin_count,
            'utilization_percent': (
                (len(self._active_resources) - self._pool.qsize()) / max(1, len(self._active_resources)) * 100
            )
        }
    
    def shutdown(self):
        """Shutdown resource pool."""
        self._shutdown = True
        
        # Destroy all resources
        while True:
            try:
                resource = self._pool.get_nowait()
                self._destroy_resource(resource)
            except queue.Empty:
                break
        
        logger.info("ResourcePool shutdown complete")


class WorkerPool:
    """
    Advanced worker pool with load balancing and health monitoring.
    
    Supports both thread and process workers with automatic scaling
    and health monitoring.
    """
    
    def __init__(
        self,
        worker_type: str = "thread",  # "thread" or "process"
        min_workers: int = 2,
        max_workers: int = 10,
        queue_size: int = 1000,
        worker_timeout: float = 300.0,
        auto_scale: bool = True
    ):
        """
        Initialize worker pool.
        
        Args:
            worker_type: Type of workers ("thread" or "process")
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            queue_size: Maximum queue size
            worker_timeout: Worker idle timeout
            auto_scale: Enable automatic scaling
        """
        self.worker_type = worker_type
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.worker_timeout = worker_timeout
        self.auto_scale = auto_scale
        
        # Task queue
        self.task_queue = queue.Queue(maxsize=queue_size)
        self.result_futures = {}
        
        # Worker management
        self.workers = {}
        self.worker_stats = {}
        self.next_worker_id = 0
        self._shutdown = False
        self._lock = threading.RLock()
        
        # Statistics
        self.tasks_submitted = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        
        # Initialize workers
        self._start_initial_workers()
        
        # Start management thread
        self._management_thread = threading.Thread(target=self._management_loop, daemon=True)
        self._management_thread.start()
        
        logger.info(f"WorkerPool initialized: {worker_type}, {min_workers}-{max_workers} workers")
    
    def _start_initial_workers(self):
        """Start initial set of workers."""
        for _ in range(self.min_workers):
            self._start_worker()
    
    def _start_worker(self) -> str:
        """Start a new worker."""
        with self._lock:
            worker_id = f"{self.worker_type}_worker_{self.next_worker_id}"
            self.next_worker_id += 1
            
            if self.worker_type == "thread":
                worker = threading.Thread(
                    target=self._worker_loop,
                    args=(worker_id,),
                    daemon=True
                )
            else:  # process
                worker = mp.Process(
                    target=self._worker_loop,
                    args=(worker_id,)
                )
            
            worker.start()
            self.workers[worker_id] = worker
            self.worker_stats[worker_id] = WorkerStats(worker_id)
            
            logger.debug(f"Started worker: {worker_id}")
            return worker_id
    
    def _worker_loop(self, worker_id: str):
        """Main worker loop."""
        logger.debug(f"Worker {worker_id} started")
        
        while not self._shutdown:
            try:
                # Get task with timeout
                try:
                    task_data = self.task_queue.get(timeout=self.worker_timeout)
                except queue.Empty:
                    # Timeout reached, check if we should shutdown
                    if self._should_shutdown_worker(worker_id):
                        break
                    continue
                
                if task_data is None:  # Shutdown signal
                    break
                
                task_id, func, args, kwargs = task_data
                
                # Execute task
                start_time = time.time()
                success = True
                result = None
                error = None
                
                try:
                    with profiler.profile_operation(f"worker_task_{func.__name__}"):
                        result = func(*args, **kwargs)
                except Exception as e:
                    success = False
                    error = e
                    logger.error(f"Worker {worker_id} task failed: {e}")
                
                duration = time.time() - start_time
                
                # Update statistics
                self.worker_stats[worker_id].update_completion(duration, success)
                
                # Store result
                if task_id in self.result_futures:
                    future = self.result_futures[task_id]
                    if success:
                        future.set_result(result)
                    else:
                        future.set_exception(error)
                
                self.task_queue.task_done()
                
                if success:
                    self.tasks_completed += 1
                else:
                    self.tasks_failed += 1
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.debug(f"Worker {worker_id} stopped")
    
    def _should_shutdown_worker(self, worker_id: str) -> bool:
        """Check if worker should be shutdown due to scaling."""
        if not self.auto_scale:
            return False
        
        with self._lock:
            active_workers = len([w for w in self.workers.values() if w.is_alive()])
            return active_workers > self.min_workers
    
    def _management_loop(self):
        """Management loop for auto-scaling and health monitoring."""
        while not self._shutdown:
            try:
                self._auto_scale_workers()
                self._cleanup_dead_workers()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in management loop: {e}")
    
    def _auto_scale_workers(self):
        """Auto-scale workers based on queue size and load."""
        if not self.auto_scale:
            return
        
        with self._lock:
            queue_size = self.task_queue.qsize()
            active_workers = len([w for w in self.workers.values() if w.is_alive()])
            
            # Scale up if queue is getting full
            if queue_size > active_workers * 2 and active_workers < self.max_workers:
                self._start_worker()
                logger.info(f"Scaled up to {active_workers + 1} workers")
            
            # Scale down if queue is empty and workers are idle
            elif queue_size == 0 and active_workers > self.min_workers:
                # Check for idle workers
                current_time = datetime.now()
                for worker_id, stats in self.worker_stats.items():
                    if ((current_time - stats.last_activity).total_seconds() > self.worker_timeout and
                        active_workers > self.min_workers):
                        # Signal worker to shutdown
                        self.task_queue.put(None)
                        break
    
    def _cleanup_dead_workers(self):
        """Clean up dead workers."""
        with self._lock:
            dead_workers = []
            for worker_id, worker in self.workers.items():
                if not worker.is_alive():
                    dead_workers.append(worker_id)
            
            for worker_id in dead_workers:
                del self.workers[worker_id]
                if worker_id in self.worker_stats:
                    del self.worker_stats[worker_id]
                logger.debug(f"Cleaned up dead worker: {worker_id}")
    
    def submit(self, func: Callable, *args, **kwargs) -> Future:
        """
        Submit task to worker pool.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Future object for result
        """
        if self._shutdown:
            raise RuntimeError("Worker pool is shutdown")
        
        task_id = f"task_{self.tasks_submitted}"
        self.tasks_submitted += 1
        
        # Create future for result
        future = Future()
        self.result_futures[task_id] = future
        
        # Submit task
        task_data = (task_id, func, args, kwargs)
        
        try:
            self.task_queue.put(task_data, timeout=5.0)
        except queue.Full:
            # Clean up
            del self.result_futures[task_id]
            raise ResourceError("Task queue is full")
        
        return future
    
    def map(self, func: Callable, iterable: List[Any], timeout: Optional[float] = None) -> List[Any]:
        """
        Map function over iterable using worker pool.
        
        Args:
            func: Function to apply
            iterable: Items to process
            timeout: Optional timeout for all tasks
            
        Returns:
            List of results
        """
        # Submit all tasks
        futures = [self.submit(func, item) for item in iterable]
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed in map: {e}")
                results.append(None)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        with self._lock:
            active_workers = len([w for w in self.workers.values() if w.is_alive()])
            
            return {
                'worker_type': self.worker_type,
                'active_workers': active_workers,
                'min_workers': self.min_workers,
                'max_workers': self.max_workers,
                'queue_size': self.task_queue.qsize(),
                'tasks_submitted': self.tasks_submitted,
                'tasks_completed': self.tasks_completed,
                'tasks_failed': self.tasks_failed,
                'success_rate': (
                    self.tasks_completed / max(1, self.tasks_completed + self.tasks_failed) * 100
                ),
                'worker_stats': {
                    worker_id: {
                        'tasks_completed': stats.tasks_completed,
                        'tasks_failed': stats.tasks_failed,
                        'average_time_seconds': stats.average_time_seconds,
                        'last_activity': stats.last_activity.isoformat()
                    }
                    for worker_id, stats in self.worker_stats.items()
                }
            }
    
    def shutdown(self, wait: bool = True):
        """Shutdown worker pool."""
        self._shutdown = True
        
        # Signal all workers to shutdown
        for _ in self.workers:
            try:
                self.task_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
        
        if wait:
            # Wait for workers to finish
            for worker in self.workers.values():
                if hasattr(worker, 'join'):
                    worker.join(timeout=30.0)
        
        logger.info("WorkerPool shutdown complete")


# Global instances for common use cases
_model_resource_pool = None
_simulation_resource_pool = None
_default_worker_pool = None


def get_model_resource_pool() -> ResourcePool:
    """Get global model resource pool."""
    global _model_resource_pool
    if _model_resource_pool is None:
        def create_model_context():
            # Placeholder for model context creation
            return {'type': 'model_context', 'created_at': datetime.now()}
        
        _model_resource_pool = ResourcePool(
            resource_factory=create_model_context,
            max_size=4,
            min_size=1,
            health_check=lambda x: x is not None
        )
    
    return _model_resource_pool


def get_simulation_resource_pool() -> ResourcePool:
    """Get global simulation resource pool."""
    global _simulation_resource_pool
    if _simulation_resource_pool is None:
        def create_simulation_context():
            # Placeholder for simulation context creation
            return {'type': 'simulation_context', 'created_at': datetime.now()}
        
        _simulation_resource_pool = ResourcePool(
            resource_factory=create_simulation_context,
            max_size=8,
            min_size=2,
            health_check=lambda x: x is not None
        )
    
    return _simulation_resource_pool


def get_default_worker_pool() -> WorkerPool:
    """Get global worker pool."""
    global _default_worker_pool
    if _default_worker_pool is None:
        cpu_count = mp.cpu_count()
        _default_worker_pool = WorkerPool(
            worker_type="thread",
            min_workers=max(2, cpu_count // 2),
            max_workers=min(16, cpu_count * 2),
            auto_scale=True
        )
    
    return _default_worker_pool


def shutdown_all_pools():
    """Shutdown all global resource pools."""
    global _model_resource_pool, _simulation_resource_pool, _default_worker_pool
    
    if _model_resource_pool:
        _model_resource_pool.shutdown()
        _model_resource_pool = None
    
    if _simulation_resource_pool:
        _simulation_resource_pool.shutdown()
        _simulation_resource_pool = None
    
    if _default_worker_pool:
        _default_worker_pool.shutdown()
        _default_worker_pool = None
    
    logger.info("All resource pools shutdown")