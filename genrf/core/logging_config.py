"""
Centralized logging configuration for GenRF.

This module provides structured logging with correlation IDs, performance
metrics, and configurable output formats.
"""

import logging
import logging.handlers
import sys
import os
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager
import threading

# Thread-local storage for correlation IDs
_local = threading.local()


class CorrelationFormatter(logging.Formatter):
    """Custom formatter that includes correlation IDs and structured data."""
    
    def __init__(self, include_correlation: bool = True, json_format: bool = False):
        self.include_correlation = include_correlation
        self.json_format = json_format
        
        if json_format:
            super().__init__()
        else:
            fmt = "%(asctime)s - %(name)s - %(levelname)s"
            if include_correlation:
                fmt += " - [%(correlation_id)s]"
            fmt += " - %(message)s"
            super().__init__(fmt)
    
    def format(self, record):
        # Add correlation ID if available
        if self.include_correlation:
            record.correlation_id = getattr(_local, 'correlation_id', 'no-correlation')
        
        # Add structured data if available
        structured_data = getattr(record, 'structured_data', {})
        
        if self.json_format:
            # JSON format for structured logging
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
            }
            
            if self.include_correlation:
                log_entry['correlation_id'] = getattr(record, 'correlation_id', 'no-correlation')
            
            if structured_data:
                log_entry['data'] = structured_data
            
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
            
            return json.dumps(log_entry)
        else:
            # Standard text format
            formatted = super().format(record)
            
            # Append structured data if present
            if structured_data:
                formatted += f" | Data: {json.dumps(structured_data)}"
            
            return formatted


class PerformanceLogger:
    """Logger for performance metrics and timing information."""
    
    def __init__(self, logger_name: str = 'genrf.performance'):
        self.logger = logging.getLogger(logger_name)
        self.timers = {}
    
    def start_timer(self, operation: str) -> str:
        """Start a performance timer."""
        timer_id = f"{operation}_{uuid.uuid4().hex[:8]}"
        self.timers[timer_id] = {
            'operation': operation,
            'start_time': time.time(),
            'correlation_id': getattr(_local, 'correlation_id', 'no-correlation')
        }
        
        self.logger.info(
            f"Started operation: {operation}",
            extra={'structured_data': {'operation': operation, 'timer_id': timer_id}}
        )
        
        return timer_id
    
    def end_timer(self, timer_id: str, success: bool = True, additional_data: Optional[Dict[str, Any]] = None):
        """End a performance timer and log results."""
        if timer_id not in self.timers:
            self.logger.warning(f"Timer {timer_id} not found")
            return
        
        timer_info = self.timers.pop(timer_id)
        duration = time.time() - timer_info['start_time']
        
        log_data = {
            'operation': timer_info['operation'],
            'duration_seconds': duration,
            'success': success,
            'timer_id': timer_id
        }
        
        if additional_data:
            log_data.update(additional_data)
        
        level = logging.INFO if success else logging.WARNING
        message = f"Completed operation: {timer_info['operation']} in {duration:.3f}s"
        
        self.logger.log(level, message, extra={'structured_data': log_data})
    
    @contextmanager
    def time_operation(self, operation: str, additional_data: Optional[Dict[str, Any]] = None):
        """Context manager for timing operations."""
        timer_id = self.start_timer(operation)
        success = True
        
        try:
            yield
        except Exception as e:
            success = False
            if additional_data is None:
                additional_data = {}
            additional_data['error'] = str(e)
            raise
        finally:
            self.end_timer(timer_id, success, additional_data)


class AuditLogger:
    """Logger for audit trail and security events."""
    
    def __init__(self, logger_name: str = 'genrf.audit'):
        self.logger = logging.getLogger(logger_name)
    
    def log_circuit_generation(
        self,
        spec_name: str,
        circuit_type: str,
        user_id: Optional[str] = None,
        success: bool = True,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Log circuit generation event."""
        log_data = {
            'event_type': 'circuit_generation',
            'spec_name': spec_name,
            'circuit_type': circuit_type,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        if user_id:
            log_data['user_id'] = user_id
        
        if additional_data:
            log_data.update(additional_data)
        
        level = logging.INFO if success else logging.ERROR
        message = f"Circuit generation: {spec_name} ({circuit_type}) - {'SUCCESS' if success else 'FAILED'}"
        
        self.logger.log(level, message, extra={'structured_data': log_data})
    
    def log_file_access(
        self,
        filepath: str,
        operation: str,
        success: bool = True,
        user_id: Optional[str] = None
    ):
        """Log file access event."""
        log_data = {
            'event_type': 'file_access',
            'filepath': filepath,
            'operation': operation,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        if user_id:
            log_data['user_id'] = user_id
        
        level = logging.INFO if success else logging.WARNING
        message = f"File {operation}: {filepath} - {'SUCCESS' if success else 'FAILED'}"
        
        self.logger.log(level, message, extra={'structured_data': log_data})


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    json_format: bool = False,
    console_output: bool = True,
    correlation_tracking: bool = True
) -> Dict[str, logging.Logger]:
    """
    Setup centralized logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        json_format: Use JSON format for structured logging
        console_output: Enable console output
        correlation_tracking: Enable correlation ID tracking
        
    Returns:
        Dictionary of configured loggers
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = CorrelationFormatter(
        include_correlation=correlation_tracking,
        json_format=json_format
    )
    
    # Configure root logger
    root_logger = logging.getLogger('genrf')
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Create specialized loggers
    loggers = {
        'main': root_logger,
        'performance': logging.getLogger('genrf.performance'),
        'audit': logging.getLogger('genrf.audit'),
        'security': logging.getLogger('genrf.security'),
        'models': logging.getLogger('genrf.models'),
        'simulation': logging.getLogger('genrf.simulation'),
        'optimization': logging.getLogger('genrf.optimization')
    }
    
    # Set levels for specialized loggers
    for logger in loggers.values():
        logger.setLevel(level)
    
    # Log startup message
    root_logger.info("GenRF logging system initialized", extra={
        'structured_data': {
            'level': logging.getLevelName(level),
            'log_file': str(log_file) if log_file else None,
            'json_format': json_format,
            'correlation_tracking': correlation_tracking
        }
    })
    
    return loggers


def get_correlation_id() -> str:
    """Get current correlation ID."""
    return getattr(_local, 'correlation_id', 'no-correlation')


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID for current thread."""
    if correlation_id is None:
        correlation_id = uuid.uuid4().hex[:12]
    
    _local.correlation_id = correlation_id
    return correlation_id


def clear_correlation_id():
    """Clear correlation ID for current thread."""
    if hasattr(_local, 'correlation_id'):
        delattr(_local, 'correlation_id')


@contextmanager
def correlation_context(correlation_id: Optional[str] = None):
    """Context manager for correlation ID."""
    old_id = getattr(_local, 'correlation_id', None)
    
    try:
        set_correlation_id(correlation_id)
        yield get_correlation_id()
    finally:
        if old_id is not None:
            _local.correlation_id = old_id
        else:
            clear_correlation_id()


# Global logger instances
performance_logger = PerformanceLogger()
audit_logger = AuditLogger()


def get_logger(name: str) -> logging.Logger:
    """Get logger with GenRF prefix."""
    if not name.startswith('genrf.'):
        name = f'genrf.{name}'
    return logging.getLogger(name)


# Health check logging
class HealthCheckLogger:
    """Logger for system health checks."""
    
    def __init__(self, logger_name: str = 'genrf.health'):
        self.logger = logging.getLogger(logger_name)
    
    def log_system_status(
        self,
        component: str,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log system component status."""
        log_data = {
            'component': component,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        if details:
            log_data['details'] = details
        
        level = logging.INFO if status == 'healthy' else logging.WARNING
        message = f"Health check: {component} - {status}"
        
        self.logger.log(level, message, extra={'structured_data': log_data})
    
    def log_resource_usage(
        self,
        cpu_percent: float,
        memory_percent: float,
        disk_percent: float,
        gpu_memory_percent: Optional[float] = None
    ):
        """Log resource usage metrics."""
        log_data = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'disk_percent': disk_percent,
            'timestamp': datetime.now().isoformat()
        }
        
        if gpu_memory_percent is not None:
            log_data['gpu_memory_percent'] = gpu_memory_percent
        
        # Determine alert level based on usage
        max_usage = max(cpu_percent, memory_percent, disk_percent)
        if max_usage > 90:
            level = logging.ERROR
        elif max_usage > 80:
            level = logging.WARNING
        else:
            level = logging.INFO
        
        message = f"Resource usage - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Disk: {disk_percent:.1f}%"
        
        self.logger.log(level, message, extra={'structured_data': log_data})


health_logger = HealthCheckLogger()


# Initialize default logging if not already configured
if not logging.getLogger('genrf').handlers:
    # Basic configuration for development
    log_level = os.getenv('GENRF_LOG_LEVEL', 'INFO')
    log_file = os.getenv('GENRF_LOG_FILE')
    json_logging = os.getenv('GENRF_JSON_LOGS', 'false').lower() == 'true'
    
    setup_logging(
        level=log_level,
        log_file=log_file,
        json_format=json_logging,
        console_output=True,
        correlation_tracking=True
    )