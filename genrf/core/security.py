"""
Security measures and input sanitization for GenRF.

This module provides security controls, access logging, and protection
against common vulnerabilities in the circuit generation pipeline.
"""

import hashlib
import secrets
import time
import os
import logging
from typing import Dict, List, Optional, Any, Set, Callable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import re

from .exceptions import SecurityError, ValidationError
from .logging_config import audit_logger

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    
    # File access controls
    allowed_file_extensions: Set[str] = field(default_factory=lambda: {
        '.yaml', '.yml', '.json', '.cir', '.sp', '.net', '.il', '.va', '.py', '.m'
    })
    
    blocked_directories: Set[str] = field(default_factory=lambda: {
        '/etc', '/proc', '/sys', '/dev', '/root', '/bin', '/sbin', '/usr/bin', '/usr/sbin'
    })
    
    max_file_size_mb: float = 100.0  # Maximum file size in MB
    max_files_per_operation: int = 1000  # Maximum files per operation
    
    # Rate limiting
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    
    # Model security
    verify_model_checksums: bool = True
    allowed_model_formats: Set[str] = field(default_factory=lambda: {'.pt', '.pth', '.ckpt'})
    
    # Code generation security
    sanitize_generated_code: bool = True
    max_code_length: int = 1000000  # 1MB max generated code
    
    # Audit logging
    log_all_file_access: bool = True
    log_parameter_access: bool = False  # May contain sensitive data
    
    # Resource limits
    max_memory_usage_mb: float = 8192.0  # 8GB max memory
    max_execution_time_seconds: float = 3600.0  # 1 hour max execution


class RateLimiter:
    """Rate limiter for API requests and operations."""
    
    def __init__(self, max_per_minute: int = 60, max_per_hour: int = 1000):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        
        self.requests_minute = []
        self.requests_hour = []
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str = "default") -> tuple[bool, str]:
        """
        Check if request is allowed based on rate limits.
        
        Args:
            identifier: Request identifier (IP, user ID, etc.)
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        with self.lock:
            now = time.time()
            
            # Clean old requests
            self.requests_minute = [t for t in self.requests_minute if now - t < 60]
            self.requests_hour = [t for t in self.requests_hour if now - t < 3600]
            
            # Check limits
            if len(self.requests_minute) >= self.max_per_minute:
                return False, f"Rate limit exceeded: {self.max_per_minute} requests per minute"
            
            if len(self.requests_hour) >= self.max_per_hour:
                return False, f"Rate limit exceeded: {self.max_per_hour} requests per hour"
            
            # Record request
            self.requests_minute.append(now)
            self.requests_hour.append(now)
            
            return True, "OK"


class FileAccessController:
    """Controls and validates file access operations."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.blocked_patterns = [
            r'\.\./',           # Directory traversal
            r'~/',              # Home directory
            r'/etc/',           # System directories
            r'/proc/',          # Process filesystem
            r'/sys/',           # System filesystem
            r'\\\\',            # Windows UNC paths
            r'[<>:"|?*]',       # Invalid filename characters
        ]
    
    def validate_file_path(self, filepath: Path, operation: str = "read") -> None:
        """
        Validate file path for security.
        
        Args:
            filepath: Path to validate
            operation: Operation type ("read", "write", "execute")
            
        Raises:
            SecurityError: If path is invalid or dangerous
        """
        try:
            # Convert to absolute path to resolve any relative components
            abs_path = filepath.resolve()
            path_str = str(abs_path)
            
            # Check for dangerous patterns
            for pattern in self.blocked_patterns:
                if re.search(pattern, path_str, re.IGNORECASE):
                    raise SecurityError(f"Dangerous path pattern detected: {pattern}")
            
            # Check blocked directories
            for blocked_dir in self.config.blocked_directories:
                if path_str.startswith(blocked_dir):
                    raise SecurityError(f"Access to blocked directory: {blocked_dir}")
            
            # Check file extension for read operations
            if operation in ["read", "write"] and filepath.suffix:
                if filepath.suffix.lower() not in self.config.allowed_file_extensions:
                    raise SecurityError(f"File extension not allowed: {filepath.suffix}")
            
            # Check file size for read operations
            if operation == "read" and filepath.exists():
                file_size_mb = filepath.stat().st_size / (1024 * 1024)
                if file_size_mb > self.config.max_file_size_mb:
                    raise SecurityError(f"File too large: {file_size_mb:.1f}MB > {self.config.max_file_size_mb}MB")
            
            # Log file access
            if self.config.log_all_file_access:
                audit_logger.log_file_access(
                    filepath=str(abs_path),
                    operation=operation,
                    success=True
                )
            
        except SecurityError:
            # Log security violation
            if self.config.log_all_file_access:
                audit_logger.log_file_access(
                    filepath=str(filepath),
                    operation=operation,
                    success=False
                )
            raise
        except Exception as e:
            raise SecurityError(f"File validation failed: {e}")
    
    def safe_file_read(self, filepath: Path, max_size_mb: Optional[float] = None) -> str:
        """Safely read file with security checks."""
        self.validate_file_path(filepath, "read")
        
        # Use configured max size if not specified
        max_size = max_size_mb or self.config.max_file_size_mb
        max_bytes = int(max_size * 1024 * 1024)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Read in chunks to avoid memory issues
                content = ""
                bytes_read = 0
                
                while True:
                    chunk = f.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    
                    bytes_read += len(chunk.encode('utf-8'))
                    if bytes_read > max_bytes:
                        raise SecurityError(f"File exceeds maximum size: {max_size}MB")
                    
                    content += chunk
                
                return content
                
        except UnicodeDecodeError:
            raise SecurityError("File contains invalid UTF-8 characters")
        except PermissionError:
            raise SecurityError("Permission denied reading file")
        except Exception as e:
            raise SecurityError(f"Error reading file: {e}")
    
    def safe_file_write(self, filepath: Path, content: str, max_size_mb: Optional[float] = None) -> None:
        """Safely write file with security checks."""
        self.validate_file_path(filepath, "write")
        
        # Check content size
        max_size = max_size_mb or self.config.max_file_size_mb
        content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
        
        if content_size_mb > max_size:
            raise SecurityError(f"Content too large: {content_size_mb:.1f}MB > {max_size}MB")
        
        try:
            # Create parent directories if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
                
        except PermissionError:
            raise SecurityError("Permission denied writing file")
        except Exception as e:
            raise SecurityError(f"Error writing file: {e}")


class CodeSanitizer:
    """Sanitizes generated code for security."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
        # Dangerous code patterns to detect
        self.dangerous_patterns = [
            r'import\s+os',
            r'import\s+subprocess',
            r'import\s+sys',
            r'__import__',
            r'eval\s*\(',
            r'exec\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
        ]
        
        # Safe patterns that are allowed
        self.safe_patterns = [
            r'import\s+(math|numpy|scipy|matplotlib)',
            r'from\s+(math|numpy|scipy|matplotlib)',
        ]
    
    def sanitize_code(self, code: str, language: str = "python") -> str:
        """
        Sanitize generated code.
        
        Args:
            code: Code to sanitize
            language: Programming language
            
        Returns:
            Sanitized code
            
        Raises:
            SecurityError: If dangerous patterns are detected
        """
        if not self.config.sanitize_generated_code:
            return code
        
        # Check code size
        if len(code) > self.config.max_code_length:
            raise SecurityError(f"Generated code too long: {len(code)} > {self.config.max_code_length}")
        
        # Language-specific sanitization
        if language.lower() == "python":
            return self._sanitize_python_code(code)
        elif language.lower() in ["skill", "verilog", "spice"]:
            return self._sanitize_hdl_code(code)
        else:
            return self._sanitize_generic_code(code)
    
    def _sanitize_python_code(self, code: str) -> str:
        """Sanitize Python code."""
        lines = code.split('\n')
        sanitized_lines = []
        
        for line_num, line in enumerate(lines, 1):
            # Check for dangerous patterns
            line_lower = line.lower().strip()
            
            # Skip empty lines and comments
            if not line_lower or line_lower.startswith('#'):
                sanitized_lines.append(line)
                continue
            
            # Check if line matches safe patterns
            is_safe = any(re.search(pattern, line, re.IGNORECASE) for pattern in self.safe_patterns)
            
            if not is_safe:
                # Check for dangerous patterns
                for pattern in self.dangerous_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        logger.warning(f"Dangerous pattern detected in line {line_num}: {pattern}")
                        # Comment out dangerous line
                        line = f"# SANITIZED: {line}"
                        break
            
            sanitized_lines.append(line)
        
        return '\n'.join(sanitized_lines)
    
    def _sanitize_hdl_code(self, code: str) -> str:
        """Sanitize hardware description language code."""
        # For HDL languages, mainly check for system commands
        dangerous_hdl_patterns = [
            r'\$system',
            r'\$readmem',
            r'\$writemem',
            r'`include\s+"[^"]*\.\./[^"]*"',  # Include with path traversal
        ]
        
        for pattern in dangerous_hdl_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                raise SecurityError(f"Dangerous HDL pattern detected: {pattern}")
        
        return code
    
    def _sanitize_generic_code(self, code: str) -> str:
        """Generic code sanitization."""
        # Remove or comment out potentially dangerous commands
        dangerous_generic = [
            r'system\s*\(',
            r'shell\s*\(',
            r'cmd\s*\(',
            r'execute\s*\(',
        ]
        
        for pattern in dangerous_generic:
            code = re.sub(pattern, '// SANITIZED: \\g<0>', code, flags=re.IGNORECASE)
        
        return code


class ModelSecurityValidator:
    """Validates AI model files for security."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.known_checksums = {}  # In production, load from secure storage
    
    def validate_model_file(self, model_path: Path) -> Dict[str, Any]:
        """
        Validate AI model file for security.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Validation results
            
        Raises:
            SecurityError: If model is invalid or dangerous
        """
        try:
            # Check file extension
            if model_path.suffix not in self.config.allowed_model_formats:
                raise SecurityError(f"Model format not allowed: {model_path.suffix}")
            
            # Check file size
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_memory_usage_mb:
                raise SecurityError(f"Model file too large: {file_size_mb:.1f}MB")
            
            # Calculate checksum
            checksum = self._calculate_file_checksum(model_path)
            
            validation_result = {
                'file_path': str(model_path),
                'file_size_mb': file_size_mb,
                'checksum': checksum,
                'validated': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # Verify checksum if enabled
            if self.config.verify_model_checksums:
                expected_checksum = self.known_checksums.get(model_path.name)
                if expected_checksum and expected_checksum != checksum:
                    raise SecurityError(f"Model checksum mismatch: expected {expected_checksum}, got {checksum}")
                
                validation_result['checksum_verified'] = expected_checksum is not None
            
            logger.info(f"Model validation passed: {model_path}")
            return validation_result
            
        except SecurityError:
            raise
        except Exception as e:
            raise SecurityError(f"Model validation failed: {e}")
    
    def _calculate_file_checksum(self, filepath: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()
        
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def register_known_model(self, model_name: str, checksum: str):
        """Register a known good model checksum."""
        self.known_checksums[model_name] = checksum
        logger.info(f"Registered known model: {model_name}")


class SecurityManager:
    """Main security manager for GenRF."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize security manager."""
        self.config = config or SecurityConfig()
        
        self.rate_limiter = RateLimiter(
            max_per_minute=self.config.max_requests_per_minute,
            max_per_hour=self.config.max_requests_per_hour
        )
        
        self.file_controller = FileAccessController(self.config)
        self.code_sanitizer = CodeSanitizer(self.config)
        self.model_validator = ModelSecurityValidator(self.config)
        
        # Security event tracking
        self.security_events = []
        self.max_events = 10000
        
        logger.info("SecurityManager initialized", extra={
            'structured_data': {
                'rate_limit_per_minute': self.config.max_requests_per_minute,
                'rate_limit_per_hour': self.config.max_requests_per_hour,
                'file_size_limit_mb': self.config.max_file_size_mb
            }
        })
    
    def check_rate_limit(self, identifier: str = "default") -> None:
        """Check rate limit and raise exception if exceeded."""
        allowed, reason = self.rate_limiter.is_allowed(identifier)
        if not allowed:
            self._log_security_event("rate_limit_exceeded", {
                'identifier': identifier,
                'reason': reason
            })
            raise SecurityError(f"Rate limit exceeded: {reason}")
    
    def validate_and_read_file(self, filepath: Path) -> str:
        """Validate and safely read file."""
        return self.file_controller.safe_file_read(filepath)
    
    def validate_and_write_file(self, filepath: Path, content: str) -> None:
        """Validate and safely write file."""
        # Sanitize content if it's code
        if filepath.suffix in ['.py', '.m', '.il', '.va']:
            language = {
                '.py': 'python',
                '.m': 'matlab',
                '.il': 'skill',
                '.va': 'verilog'
            }.get(filepath.suffix, 'generic')
            
            content = self.code_sanitizer.sanitize_code(content, language)
        
        self.file_controller.safe_file_write(filepath, content)
    
    def validate_model_file(self, model_path: Path) -> Dict[str, Any]:
        """Validate AI model file."""
        return self.model_validator.validate_model_file(model_path)
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }
        
        self.security_events.append(event)
        if len(self.security_events) > self.max_events:
            self.security_events.pop(0)
        
        logger.warning(f"Security event: {event_type}", extra={'structured_data': event})
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security event summary."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_iso = cutoff_time.isoformat()
        
        recent_events = [e for e in self.security_events if e['timestamp'] >= cutoff_iso]
        
        event_counts = {}
        for event in recent_events:
            event_type = event['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            'hours': hours,
            'total_events': len(recent_events),
            'event_types': event_counts,
            'recent_events': recent_events[-10:]  # Last 10 events
        }


# Global security manager instance
security_manager = SecurityManager()


def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    return security_manager


def configure_security(config: SecurityConfig) -> None:
    """Configure global security settings."""
    global security_manager
    security_manager = SecurityManager(config)
    logger.info("Security configuration updated")