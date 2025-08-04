"""
System monitoring and health checks for GenRF.

This module provides comprehensive monitoring of system resources,
model performance, and operational health metrics.
"""

import psutil
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .logging_config import health_logger

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    gpu_metrics: Optional[Dict[str, Any]] = None
    process_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_gb': self.memory_used_gb,
            'memory_total_gb': self.memory_total_gb,
            'disk_percent': self.disk_percent,
            'disk_used_gb': self.disk_used_gb,
            'disk_total_gb': self.disk_total_gb,
            'gpu_metrics': self.gpu_metrics,
            'process_count': self.process_count
        }


@dataclass
class ModelMetrics:
    """AI model performance metrics."""
    timestamp: datetime
    model_name: str
    inference_time_ms: float
    memory_usage_mb: float
    batch_size: int = 1
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name,
            'inference_time_ms': self.inference_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'batch_size': self.batch_size,
            'success': self.success,
            'error_message': self.error_message
        }


@dataclass
class HealthStatus:
    """Overall system health status."""
    timestamp: datetime
    overall_status: str  # 'healthy', 'warning', 'critical'
    components: Dict[str, str] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    metrics: Optional[SystemMetrics] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_status': self.overall_status,
            'components': self.components,
            'alerts': self.alerts,
            'metrics': self.metrics.to_dict() if self.metrics else None
        }


class SystemMonitor:
    """
    System resource monitor for GenRF.
    
    Monitors CPU, memory, disk, GPU usage and provides health checks.
    """
    
    def __init__(
        self,
        monitoring_interval: float = 30.0,
        alert_thresholds: Optional[Dict[str, float]] = None,
        enable_gpu_monitoring: bool = True
    ):
        """
        Initialize system monitor.
        
        Args:
            monitoring_interval: Monitoring interval in seconds
            alert_thresholds: Resource usage alert thresholds
            enable_gpu_monitoring: Enable GPU monitoring if available
        """
        self.monitoring_interval = monitoring_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring and TORCH_AVAILABLE
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'gpu_memory_percent': 90.0
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
        self.max_history_size = 1000
        
        # Health check callbacks
        self.health_check_callbacks = []
        
        logger.info("SystemMonitor initialized", extra={
            'structured_data': {
                'monitoring_interval': monitoring_interval,
                'gpu_monitoring': self.enable_gpu_monitoring,
                'alert_thresholds': self.alert_thresholds
            }
        })
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Process count
            process_count = len(psutil.pids())
            
            # GPU metrics
            gpu_metrics = None
            if self.enable_gpu_monitoring:
                gpu_metrics = self._get_gpu_metrics()
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                gpu_metrics=gpu_metrics,
                process_count=process_count
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            # Return minimal metrics
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                memory_total_gb=0.0,
                disk_percent=0.0,
                disk_used_gb=0.0,
                disk_total_gb=0.0
            )
    
    def _get_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Get GPU metrics using PyTorch."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None
        
        try:
            gpu_count = torch.cuda.device_count()
            gpu_metrics = {
                'device_count': gpu_count,
                'devices': []
            }
            
            for i in range(gpu_count):
                device_props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)    # GB
                memory_total = device_props.total_memory / (1024**3)           # GB
                
                device_info = {
                    'device_id': i,
                    'name': device_props.name,
                    'memory_allocated_gb': memory_allocated,
                    'memory_reserved_gb': memory_reserved,
                    'memory_total_gb': memory_total,
                    'memory_percent': (memory_reserved / memory_total) * 100 if memory_total > 0 else 0,
                    'compute_capability': f"{device_props.major}.{device_props.minor}"
                }
                
                gpu_metrics['devices'].append(device_info)
            
            return gpu_metrics
            
        except Exception as e:
            logger.debug(f"Error collecting GPU metrics: {e}")
            return None
    
    def get_health_status(self) -> HealthStatus:
        """Get overall system health status."""
        metrics = self.get_system_metrics()
        
        # Component health checks
        components = {}
        alerts = []
        overall_status = "healthy"
        
        # CPU health
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            components['cpu'] = 'warning' if metrics.cpu_percent < 95 else 'critical'
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            if components['cpu'] == 'critical':
                overall_status = 'critical'
            elif overall_status == 'healthy':
                overall_status = 'warning'
        else:
            components['cpu'] = 'healthy'
        
        # Memory health
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            components['memory'] = 'warning' if metrics.memory_percent < 95 else 'critical'
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
            if components['memory'] == 'critical':
                overall_status = 'critical'
            elif overall_status == 'healthy':
                overall_status = 'warning'
        else:
            components['memory'] = 'healthy'
        
        # Disk health
        if metrics.disk_percent > self.alert_thresholds['disk_percent']:
            components['disk'] = 'warning' if metrics.disk_percent < 98 else 'critical'
            alerts.append(f"High disk usage: {metrics.disk_percent:.1f}%")
            if components['disk'] == 'critical':
                overall_status = 'critical'
            elif overall_status == 'healthy':
                overall_status = 'warning'
        else:
            components['disk'] = 'healthy'
        
        # GPU health
        if metrics.gpu_metrics:
            gpu_healthy = True
            for device in metrics.gpu_metrics['devices']:
                gpu_memory_percent = device['memory_percent']
                if gpu_memory_percent > self.alert_thresholds['gpu_memory_percent']:
                    gpu_healthy = False
                    alerts.append(f"High GPU memory usage on device {device['device_id']}: {gpu_memory_percent:.1f}%")
            
            components['gpu'] = 'healthy' if gpu_healthy else 'warning'
            if not gpu_healthy and overall_status == 'healthy':
                overall_status = 'warning'
        else:
            components['gpu'] = 'unavailable'
        
        # Run custom health checks
        for callback in self.health_check_callbacks:
            try:
                component_name, status, message = callback()
                components[component_name] = status
                if message and status != 'healthy':
                    alerts.append(message)
                
                if status == 'critical':
                    overall_status = 'critical'
                elif status == 'warning' and overall_status == 'healthy':
                    overall_status = 'warning'
                    
            except Exception as e:
                logger.error(f"Health check callback failed: {e}")
                components['custom_check'] = 'error'
                alerts.append(f"Health check error: {e}")
        
        health_status = HealthStatus(
            timestamp=datetime.now(),
            overall_status=overall_status,
            components=components,
            alerts=alerts,
            metrics=metrics
        )
        
        # Log health status
        health_logger.log_system_status(
            component='system',
            status=overall_status,
            details={
                'components': components,
                'alerts': alerts,
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'disk_percent': metrics.disk_percent
            }
        )
        
        return health_status
    
    def start_monitoring(self):
        """Start continuous monitoring in background thread."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self.get_system_metrics()
                
                # Store metrics in history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)
                
                # Log resource usage
                health_logger.log_resource_usage(
                    cpu_percent=metrics.cpu_percent,
                    memory_percent=metrics.memory_percent,
                    disk_percent=metrics.disk_percent,
                    gpu_memory_percent=self._get_max_gpu_memory_percent(metrics)
                )
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(min(self.monitoring_interval, 60.0))  # Don't spam errors
    
    def _get_max_gpu_memory_percent(self, metrics: SystemMetrics) -> Optional[float]:
        """Get maximum GPU memory usage percentage."""
        if not metrics.gpu_metrics or not metrics.gpu_metrics['devices']:
            return None
        
        return max(device['memory_percent'] for device in metrics.gpu_metrics['devices'])
    
    def add_health_check_callback(self, callback: Callable[[], tuple[str, str, str]]):
        """
        Add custom health check callback.
        
        Args:
            callback: Function that returns (component_name, status, message)
                     status should be 'healthy', 'warning', 'critical', or 'error'
        """
        self.health_check_callbacks.append(callback)
        logger.info(f"Added health check callback: {callback.__name__}")
    
    def get_metrics_history(self, duration_minutes: int = 60) -> List[SystemMetrics]:
        """Get metrics history for specified duration."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def export_metrics(self, filepath: Path, duration_minutes: int = 60):
        """Export metrics to JSON file."""
        metrics = self.get_metrics_history(duration_minutes)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'duration_minutes': duration_minutes,
            'metrics_count': len(metrics),
            'metrics': [m.to_dict() for m in metrics]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(metrics)} metrics to {filepath}")


class ModelMonitor:
    """Monitor AI model performance and resource usage."""
    
    def __init__(self, max_history_size: int = 1000):
        """Initialize model monitor."""
        self.max_history_size = max_history_size
        self.model_metrics = []
        
        logger.info("ModelMonitor initialized")
    
    def record_inference(
        self,
        model_name: str,
        inference_time_ms: float,
        memory_usage_mb: float,
        batch_size: int = 1,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Record model inference metrics."""
        metrics = ModelMetrics(
            timestamp=datetime.now(),
            model_name=model_name,
            inference_time_ms=inference_time_ms,
            memory_usage_mb=memory_usage_mb,
            batch_size=batch_size,
            success=success,
            error_message=error_message
        )
        
        self.model_metrics.append(metrics)
        if len(self.model_metrics) > self.max_history_size:
            self.model_metrics.pop(0)
        
        # Log metrics
        logger.info(
            f"Model inference: {model_name}",
            extra={'structured_data': metrics.to_dict()}
        )
    
    def get_model_stats(self, model_name: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get statistics for a specific model."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        model_metrics = [
            m for m in self.model_metrics 
            if m.model_name == model_name and m.timestamp >= cutoff_time
        ]
        
        if not model_metrics:
            return {'error': 'No metrics found for model'}
        
        inference_times = [m.inference_time_ms for m in model_metrics if m.success]
        memory_usages = [m.memory_usage_mb for m in model_metrics if m.success]
        
        success_count = sum(1 for m in model_metrics if m.success)
        total_count = len(model_metrics)
        
        stats = {
            'model_name': model_name,
            'duration_minutes': duration_minutes,
            'total_inferences': total_count,
            'successful_inferences': success_count,
            'success_rate': (success_count / total_count) * 100 if total_count > 0 else 0,
            'inference_time_stats': {
                'mean_ms': sum(inference_times) / len(inference_times) if inference_times else 0,
                'min_ms': min(inference_times) if inference_times else 0,
                'max_ms': max(inference_times) if inference_times else 0,
            },
            'memory_usage_stats': {
                'mean_mb': sum(memory_usages) / len(memory_usages) if memory_usages else 0,
                'min_mb': min(memory_usages) if memory_usages else 0,
                'max_mb': max(memory_usages) if memory_usages else 0,
            }
        }
        
        return stats


# Global monitor instances
system_monitor = SystemMonitor()
model_monitor = ModelMonitor()


def get_system_health() -> HealthStatus:
    """Convenience function to get system health."""
    return system_monitor.get_health_status()


def start_monitoring():
    """Start system monitoring."""
    system_monitor.start_monitoring()


def stop_monitoring():
    """Stop system monitoring."""
    system_monitor.stop_monitoring()