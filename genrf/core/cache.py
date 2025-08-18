"""
Intelligent caching system for GenRF.

This module provides multi-level caching for models, simulation results,
and optimization data to improve performance and reduce computation time.
"""

import hashlib
import pickle
import json
import time
import threading
import weakref
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import OrderedDict
import sqlite3

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    ttl_seconds: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl_seconds
    
    def touch(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache implementation.
    
    Provides memory-efficient caching with automatic eviction based on
    access patterns and memory limits.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 512.0,
        default_ttl_seconds: Optional[float] = None
    ):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl_seconds: Default time-to-live for entries
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.default_ttl = default_ttl_seconds
        
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._total_size_bytes = 0
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.info(f"LRUCache initialized: max_size={max_size}, max_memory={max_memory_mb}MB")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return default
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self._misses += 1
                return default
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """Put value in cache."""
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except (pickle.PicklingError, TypeError):
                # Fallback size estimation
                size_bytes = len(str(value)) * 2  # Rough estimate
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Check if single item exceeds memory limit
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Cache item too large: {size_bytes / 1024 / 1024:.1f}MB > {self.max_memory_bytes / 1024 / 1024:.1f}MB")
                return
            
            # Evict entries to make room
            self._evict_to_fit(size_bytes)
            
            # Create and add entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.now(),
                ttl_seconds=ttl_seconds or self.default_ttl,
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._total_size_bytes += size_bytes
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._total_size_bytes = 0
            logger.info("Cache cleared")
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update size tracking."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._total_size_bytes -= entry.size_bytes
    
    def _evict_to_fit(self, new_size_bytes: int) -> None:
        """Evict entries to fit new item."""
        # Evict by size constraint
        while (self._total_size_bytes + new_size_bytes > self.max_memory_bytes and 
               len(self._cache) > 0):
            # Remove least recently used
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
            self._evictions += 1
        
        # Evict by count constraint
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
            self._evictions += 1
        
        # Evict expired entries
        expired_keys = []
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
            self._evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_mb': self._total_size_bytes / 1024 / 1024,
                'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate_percent': hit_rate
            }
    
    def get_items_info(self) -> List[Dict[str, Any]]:
        """Get information about cached items."""
        with self._lock:
            items = []
            for key, entry in self._cache.items():
                items.append({
                    'key': key,
                    'size_mb': entry.size_bytes / 1024 / 1024,
                    'access_count': entry.access_count,
                    'age_seconds': (datetime.now() - entry.timestamp).total_seconds(),
                    'expires_in_seconds': (
                        entry.ttl_seconds - (datetime.now() - entry.timestamp).total_seconds()
                        if entry.ttl_seconds else None
                    )
                })
            return items


class PersistentCache:
    """
    Persistent cache using SQLite for long-term storage.
    
    Stores computation results that are expensive to recalculate
    across application restarts.
    """
    
    def __init__(self, db_path: Path, table_name: str = "cache"):
        """Initialize persistent cache."""
        self.db_path = db_path
        self.table_name = table_name
        
        # Create database directory
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
        
        logger.info(f"PersistentCache initialized: {db_path}")
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL,
                    ttl_seconds REAL,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER
                )
            """)
            
            # Create indexes
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_timestamp ON {self.table_name}(timestamp)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_ttl ON {self.table_name}(ttl_seconds)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from persistent cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    f"SELECT value, timestamp, ttl_seconds FROM {self.table_name} WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row is None:
                    return None
                
                value_blob, timestamp, ttl_seconds = row
                
                # Check expiration
                if ttl_seconds and (time.time() - timestamp) > ttl_seconds:
                    # Remove expired entry
                    conn.execute(f"DELETE FROM {self.table_name} WHERE key = ?", (key,))
                    return None
                
                # Update access count
                conn.execute(
                    f"UPDATE {self.table_name} SET access_count = access_count + 1 WHERE key = ?",
                    (key,)
                )
                
                # Deserialize value
                return pickle.loads(value_blob)
                
        except Exception as e:
            logger.error(f"Error reading from persistent cache: {e}")
            return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Put value in persistent cache."""
        try:
            # Serialize value
            value_blob = pickle.dumps(value)
            size_bytes = len(value_blob)
            timestamp = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    f"""INSERT OR REPLACE INTO {self.table_name} 
                        (key, value, timestamp, ttl_seconds, access_count, size_bytes)
                        VALUES (?, ?, ?, ?, 0, ?)""",
                    (key, value_blob, timestamp, ttl_seconds, size_bytes)
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing to persistent cache: {e}")
            return False
    
    def remove(self, key: str) -> bool:
        """Remove entry from persistent cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"DELETE FROM {self.table_name} WHERE key = ?", (key,))
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error removing from persistent cache: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        try:
            current_time = time.time()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    f"""DELETE FROM {self.table_name} 
                        WHERE ttl_seconds IS NOT NULL 
                        AND (timestamp + ttl_seconds) < ?""",
                    (current_time,)
                )
                removed_count = cursor.rowcount
                
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired cache entries")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total entries
                cursor = conn.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                total_entries = cursor.fetchone()[0]
                
                # Total size
                cursor = conn.execute(f"SELECT SUM(size_bytes) FROM {self.table_name}")
                total_size = cursor.fetchone()[0] or 0
                
                # Expired entries
                current_time = time.time()
                cursor = conn.execute(
                    f"""SELECT COUNT(*) FROM {self.table_name} 
                        WHERE ttl_seconds IS NOT NULL 
                        AND (timestamp + ttl_seconds) < ?""",
                    (current_time,)
                )
                expired_entries = cursor.fetchone()[0]
                
                return {
                    'total_entries': total_entries,
                    'total_size_mb': total_size / 1024 / 1024,
                    'expired_entries': expired_entries,
                    'db_size_mb': self.db_path.stat().st_size / 1024 / 1024 if self.db_path.exists() else 0
                }
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}


class ModelCache:
    """Specialized cache for AI model inference results."""
    
    def __init__(self, memory_cache_mb: float = 256.0, persistent_cache: Optional[PersistentCache] = None):
        """Initialize model cache."""
        self.memory_cache = LRUCache(
            max_size=100,
            max_memory_mb=memory_cache_mb,
            default_ttl_seconds=3600  # 1 hour default
        )
        self.persistent_cache = persistent_cache
        
        logger.info("ModelCache initialized")
    
    def _make_key(self, model_name: str, inputs: Any, params: Dict[str, Any]) -> str:
        """Create cache key from inputs."""
        # Create deterministic hash of inputs
        key_data = {
            'model_name': model_name,
            'inputs': self._serialize_inputs(inputs),
            'params': params
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def _serialize_inputs(self, inputs: Any) -> str:
        """Serialize inputs for hashing."""
        if isinstance(inputs, np.ndarray):
            return f"array_{inputs.shape}_{inputs.dtype}_{hashlib.md5(inputs.tobytes()).hexdigest()[:8]}"
        elif isinstance(inputs, (list, tuple)):
            return str([self._serialize_inputs(item) for item in inputs])
        elif isinstance(inputs, dict):
            return str({k: self._serialize_inputs(v) for k, v in inputs.items()})
        else:
            return str(inputs)
    
    def get(self, model_name: str, inputs: Any, params: Dict[str, Any] = None) -> Optional[Any]:
        """Get cached model result."""
        if params is None:
            params = {}
        
        key = self._make_key(model_name, inputs, params)
        
        # Try memory cache first
        result = self.memory_cache.get(key)
        if result is not None:
            logger.debug(f"Model cache hit (memory): {model_name}")
            return result
        
        # Try persistent cache
        if self.persistent_cache:
            result = self.persistent_cache.get(key)
            if result is not None:
                # Promote to memory cache
                self.memory_cache.put(key, result, ttl_seconds=1800)  # 30 min in memory
                logger.debug(f"Model cache hit (persistent): {model_name}")
                return result
        
        logger.debug(f"Model cache miss: {model_name}")
        return None
    
    def put(self, model_name: str, inputs: Any, result: Any, params: Dict[str, Any] = None) -> None:
        """Cache model result."""
        if params is None:
            params = {}
        
        key = self._make_key(model_name, inputs, params)
        
        # Store in memory cache
        self.memory_cache.put(key, result, ttl_seconds=1800)  # 30 minutes
        
        # Store in persistent cache for expensive computations
        if self.persistent_cache:
            self.persistent_cache.put(key, result, ttl_seconds=86400)  # 24 hours
        
        logger.debug(f"Cached model result: {model_name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'memory_cache': self.memory_cache.get_stats()
        }
        
        if self.persistent_cache:
            stats['persistent_cache'] = self.persistent_cache.get_stats()
        
        return stats


class SimulationCache:
    """Cache for SPICE simulation results."""
    
    def __init__(self, persistent_cache: Optional[PersistentCache] = None):
        """Initialize simulation cache."""
        self.persistent_cache = persistent_cache
        
        # Use longer TTL for simulation results as they're expensive
        self.memory_cache = LRUCache(
            max_size=500,
            max_memory_mb=128.0,
            default_ttl_seconds=7200  # 2 hours
        )
        
        logger.info("SimulationCache initialized")
    
    def _make_key(self, netlist: str, analyses: List[str], temperature: float = 27.0) -> str:
        """Create cache key for simulation."""
        # Hash netlist content and parameters
        key_data = {
            'netlist_hash': hashlib.md5(netlist.encode()).hexdigest(),
            'analyses': sorted(analyses),
            'temperature': temperature
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def get(self, netlist: str, analyses: List[str], temperature: float = 27.0) -> Optional[Dict[str, Any]]:
        """Get cached simulation result."""
        key = self._make_key(netlist, analyses, temperature)
        
        # Try memory cache first
        result = self.memory_cache.get(key)
        if result is not None:
            logger.debug("Simulation cache hit (memory)")
            return result
        
        # Try persistent cache
        if self.persistent_cache:
            result = self.persistent_cache.get(key)
            if result is not None:
                # Promote to memory cache
                self.memory_cache.put(key, result)
                logger.debug("Simulation cache hit (persistent)")
                return result
        
        logger.debug("Simulation cache miss")
        return None
    
    def put(self, netlist: str, analyses: List[str], result: Dict[str, Any], temperature: float = 27.0) -> None:
        """Cache simulation result."""
        key = self._make_key(netlist, analyses, temperature)
        
        # Store in memory cache
        self.memory_cache.put(key, result)
        
        # Store in persistent cache
        if self.persistent_cache:
            # Use longer TTL for persistent storage
            self.persistent_cache.put(key, result, ttl_seconds=604800)  # 1 week
        
        logger.debug("Cached simulation result")


# Global cache instances
_cache_dir = Path.home() / ".genrf" / "cache"
_persistent_cache = PersistentCache(_cache_dir / "main.db")

model_cache = ModelCache(
    memory_cache_mb=512.0,
    persistent_cache=_persistent_cache
)

simulation_cache = SimulationCache(
    persistent_cache=_persistent_cache
)


def get_cache_stats() -> Dict[str, Any]:
    """Get overall cache statistics."""
    return {
        'model_cache': model_cache.get_stats(),
        'simulation_cache': simulation_cache.memory_cache.get_stats(),
        'persistent_cache': _persistent_cache.get_stats()
    }


def cleanup_caches() -> Dict[str, int]:
    """Clean up expired cache entries."""
    results = {}
    
    # Cleanup persistent cache
    results['persistent_expired'] = _persistent_cache.cleanup_expired()
    
    return results


def clear_all_caches() -> None:
    """Clear all caches."""
    model_cache.memory_cache.clear()
    simulation_cache.memory_cache.clear()
    logger.info("All caches cleared")


class ResultCache:
    """
    Simple cache for circuit generation results.
    
    Provides caching interface compatible with CircuitDiffuser.
    """
    
    def __init__(self, max_size: int = 100):
        """Initialize result cache."""
        self.cache = LRUCache(
            max_size=max_size,
            max_memory_mb=64.0,
            default_ttl_seconds=3600  # 1 hour
        )
    
    def get(self, spec) -> Optional[Any]:
        """Get cached result for design specification."""
        # Create key from spec
        key = self._spec_to_key(spec)
        return self.cache.get(key)
    
    def put(self, spec, result: Any) -> None:
        """Store result in cache."""
        key = self._spec_to_key(spec)
        self.cache.put(key, result)
    
    def _spec_to_key(self, spec) -> str:
        """Convert specification to cache key."""
        try:
            # Convert spec to dictionary
            if hasattr(spec, '__dict__'):
                spec_dict = spec.__dict__.copy()
            else:
                spec_dict = dict(spec)
            
            # Remove non-essential fields
            spec_dict.pop('name', None)
            spec_dict.pop('description', None)
            
            # Create deterministic key
            key_string = json.dumps(spec_dict, sort_keys=True, default=str)
            return hashlib.sha256(key_string.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.warning(f"Failed to create cache key: {e}")
            return str(hash(str(spec)))
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()