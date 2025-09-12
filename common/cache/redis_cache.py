"""Redis-based caching layer for improved performance."""

import json
import pickle
import logging
import hashlib
import time
from typing import Any, Optional, Dict, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """Serialization formats for cache data."""
    JSON = "json"
    PICKLE = "pickle"
    PARQUET = "parquet"


@dataclass
class CacheConfig:
    """Configuration for Redis cache."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    max_connections: int = 50
    
    # Cache behavior
    default_ttl: int = 3600  # 1 hour
    key_prefix: str = "trading_platform"
    compression_enabled: bool = True
    compression_threshold: int = 1024  # bytes
    
    # Serialization
    default_format: SerializationFormat = SerializationFormat.JSON
    enable_pandas_optimization: bool = True


class RedisCache:
    """High-performance Redis cache with multiple serialization formats."""
    
    def __init__(self, config: CacheConfig = None):
        """Initialize Redis cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self._client = None
        self._connection_pool = None
        
        # Statistics
        self._hit_count = 0
        self._miss_count = 0
        self._error_count = 0
        self._total_get_time = 0.0
        self._total_set_time = 0.0
        
    def connect(self) -> None:
        """Connect to Redis server."""
        try:
            import redis
            
            # Create connection pool
            self._connection_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                max_connections=self.config.max_connections,
                decode_responses=False  # We handle encoding ourselves
            )
            
            # Create Redis client
            self._client = redis.Redis(connection_pool=self._connection_pool)
            
            # Test connection
            self._client.ping()
            
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            
        except ImportError:
            logger.error("Redis package not installed. Install with: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from Redis server."""
        if self._connection_pool:
            self._connection_pool.disconnect()
            self._connection_pool = None
        self._client = None
        logger.info("Disconnected from Redis")
    
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        if not self._client:
            return False
        
        try:
            self._client.ping()
            return True
        except Exception:
            return False
    
    def get(
        self, 
        key: str, 
        default: Any = None,
        format: Optional[SerializationFormat] = None
    ) -> Any:
        """Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            format: Serialization format override
            
        Returns:
            Cached value or default
        """
        if not self._ensure_connected():
            return default
        
        start_time = time.time()
        
        try:
            full_key = self._make_key(key)
            data = self._client.get(full_key)
            
            if data is None:
                self._miss_count += 1
                return default
            
            # Deserialize data
            value = self._deserialize(data, format)
            self._hit_count += 1
            
            execution_time = time.time() - start_time
            self._total_get_time += execution_time
            
            logger.debug(f"Cache hit for key '{key}' in {execution_time:.4f}s")
            return value
            
        except Exception as e:
            self._error_count += 1
            logger.warning(f"Cache get error for key '{key}': {e}")
            return default
        finally:
            self._total_get_time += time.time() - start_time
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        format: Optional[SerializationFormat] = None
    ) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            format: Serialization format override
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_connected():
            return False
        
        start_time = time.time()
        
        try:
            full_key = self._make_key(key)
            effective_ttl = ttl or self.config.default_ttl
            
            # Serialize data
            data = self._serialize(value, format)
            
            # Set with TTL
            result = self._client.setex(full_key, effective_ttl, data)
            
            execution_time = time.time() - start_time
            self._total_set_time += execution_time
            
            logger.debug(f"Cache set for key '{key}' with TTL {effective_ttl}s in {execution_time:.4f}s")
            return bool(result)
            
        except Exception as e:
            self._error_count += 1
            logger.warning(f"Cache set error for key '{key}': {e}")
            return False
        finally:
            self._total_set_time += time.time() - start_time
    
    def delete(self, key: str) -> bool:
        """Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted, False otherwise
        """
        if not self._ensure_connected():
            return False
        
        try:
            full_key = self._make_key(key)
            result = self._client.delete(full_key)
            logger.debug(f"Cache delete for key '{key}': {bool(result)}")
            return bool(result)
            
        except Exception as e:
            self._error_count += 1
            logger.warning(f"Cache delete error for key '{key}': {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        if not self._ensure_connected():
            return False
        
        try:
            full_key = self._make_key(key)
            return bool(self._client.exists(full_key))
            
        except Exception as e:
            self._error_count += 1
            logger.warning(f"Cache exists error for key '{key}': {e}")
            return False
    
    def get_ttl(self, key: str) -> int:
        """Get TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        if not self._ensure_connected():
            return -2
        
        try:
            full_key = self._make_key(key)
            return self._client.ttl(full_key)
            
        except Exception as e:
            self._error_count += 1
            logger.warning(f"Cache TTL error for key '{key}': {e}")
            return -2
    
    def extend_ttl(self, key: str, additional_seconds: int) -> bool:
        """Extend TTL for a key.
        
        Args:
            key: Cache key
            additional_seconds: Additional seconds to add to TTL
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_connected():
            return False
        
        try:
            full_key = self._make_key(key)
            current_ttl = self._client.ttl(full_key)
            
            if current_ttl > 0:
                new_ttl = current_ttl + additional_seconds
                return bool(self._client.expire(full_key, new_ttl))
            
            return False
            
        except Exception as e:
            self._error_count += 1
            logger.warning(f"Cache extend TTL error for key '{key}': {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern.
        
        Args:
            pattern: Key pattern (supports wildcards)
            
        Returns:
            Number of keys deleted
        """
        if not self._ensure_connected():
            return 0
        
        try:
            full_pattern = self._make_key(pattern)
            keys = self._client.keys(full_pattern)
            
            if keys:
                deleted = self._client.delete(*keys)
                logger.info(f"Cleared {deleted} keys matching pattern '{pattern}'")
                return deleted
            
            return 0
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Cache clear pattern error for pattern '{pattern}': {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total_requests) if total_requests > 0 else 0.0
        
        avg_get_time = (self._total_get_time / max(self._hit_count + self._miss_count, 1))
        avg_set_time = (self._total_set_time / max(self._hit_count, 1))
        
        stats = {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "errors": self._error_count,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "average_get_time": avg_get_time,
            "average_set_time": avg_set_time,
            "connected": self.is_connected()
        }
        
        # Add Redis server info if connected
        if self.is_connected():
            try:
                info = self._client.info()
                stats["redis_info"] = {
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0)
                }
            except Exception as e:
                logger.debug(f"Could not get Redis info: {e}")
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._hit_count = 0
        self._miss_count = 0
        self._error_count = 0
        self._total_get_time = 0.0
        self._total_set_time = 0.0
        logger.info("Cache statistics reset")
    
    def cached_call(
        self,
        key: str,
        func: Callable,
        ttl: Optional[int] = None,
        format: Optional[SerializationFormat] = None,
        *args,
        **kwargs
    ) -> Any:
        """Cache the result of a function call.
        
        Args:
            key: Cache key
            func: Function to call if cache miss
            ttl: Cache TTL
            format: Serialization format
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result (from cache or fresh call)
        """
        # Try to get from cache first
        result = self.get(key, format=format)
        
        if result is not None:
            return result
        
        # Cache miss - call function
        try:
            result = func(*args, **kwargs)
            
            # Cache the result
            if result is not None:
                self.set(key, result, ttl=ttl, format=format)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in cached function call: {e}")
            raise
    
    def _ensure_connected(self) -> bool:
        """Ensure Redis connection is available."""
        if not self._client:
            try:
                self.connect()
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}")
                return False
        
        return self.is_connected()
    
    def _make_key(self, key: str) -> str:
        """Create full cache key with prefix.
        
        Args:
            key: Base key
            
        Returns:
            Full cache key
        """
        return f"{self.config.key_prefix}:{key}"
    
    def _serialize(self, value: Any, format: Optional[SerializationFormat] = None) -> bytes:
        """Serialize value for storage.
        
        Args:
            value: Value to serialize
            format: Serialization format
            
        Returns:
            Serialized bytes
        """
        effective_format = format or self._determine_best_format(value)
        
        try:
            if effective_format == SerializationFormat.JSON:
                serialized = json.dumps(value, default=str).encode('utf-8')
                
            elif effective_format == SerializationFormat.PICKLE:
                serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                
            elif effective_format == SerializationFormat.PARQUET and isinstance(value, pd.DataFrame):
                import io
                buffer = io.BytesIO()
                value.to_parquet(buffer, engine='pyarrow', compression='snappy')
                serialized = buffer.getvalue()
                
            else:
                # Fallback to pickle
                serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Apply compression if enabled and data is large enough
            if (self.config.compression_enabled and 
                len(serialized) > self.config.compression_threshold):
                serialized = self._compress(serialized)
                # Add compression marker
                serialized = b'COMPRESSED:' + serialized
            
            # Add format marker
            format_marker = f"FORMAT:{effective_format.value}:".encode('utf-8')
            return format_marker + serialized
            
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
    
    def _deserialize(self, data: bytes, format: Optional[SerializationFormat] = None) -> Any:
        """Deserialize data from storage.
        
        Args:
            data: Serialized data
            format: Expected format (for validation)
            
        Returns:
            Deserialized value
        """
        try:
            # Extract format marker
            if data.startswith(b'FORMAT:'):
                marker_end = data.find(b':', 7)  # Find second colon
                if marker_end != -1:
                    format_str = data[7:marker_end].decode('utf-8')
                    data = data[marker_end + 1:]
                    detected_format = SerializationFormat(format_str)
                else:
                    detected_format = self.config.default_format
            else:
                detected_format = self.config.default_format
            
            # Check for compression
            if data.startswith(b'COMPRESSED:'):
                data = self._decompress(data[11:])  # Remove marker
            
            # Deserialize based on format
            if detected_format == SerializationFormat.JSON:
                return json.loads(data.decode('utf-8'))
                
            elif detected_format == SerializationFormat.PICKLE:
                return pickle.loads(data)
                
            elif detected_format == SerializationFormat.PARQUET:
                import io
                buffer = io.BytesIO(data)
                return pd.read_parquet(buffer, engine='pyarrow')
                
            else:
                # Fallback to pickle
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise
    
    def _determine_best_format(self, value: Any) -> SerializationFormat:
        """Determine best serialization format for value.
        
        Args:
            value: Value to serialize
            
        Returns:
            Best serialization format
        """
        if self.config.enable_pandas_optimization and isinstance(value, pd.DataFrame):
            return SerializationFormat.PARQUET
        
        # Try JSON first (most portable)
        try:
            json.dumps(value, default=str)
            return SerializationFormat.JSON
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return SerializationFormat.PICKLE
    
    def _compress(self, data: bytes) -> bytes:
        """Compress data using zlib.
        
        Args:
            data: Data to compress
            
        Returns:
            Compressed data
        """
        import zlib
        return zlib.compress(data, level=6)
    
    def _decompress(self, data: bytes) -> bytes:
        """Decompress data using zlib.
        
        Args:
            data: Compressed data
            
        Returns:
            Decompressed data
        """
        import zlib
        return zlib.decompress(data)


# Cache key utilities
def make_query_cache_key(
    table_name: str,
    filters: List[Tuple[str, str, Any]] = None,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> str:
    """Create cache key for query results.
    
    Args:
        table_name: Name of the table/dataset
        filters: Query filters
        columns: Selected columns
        limit: Row limit
        
    Returns:
        Cache key string
    """
    key_parts = [f"query:{table_name}"]
    
    if filters:
        filter_str = ",".join(f"{col}:{op}:{val}" for col, op, val in filters)
        filter_hash = hashlib.md5(filter_str.encode()).hexdigest()[:8]
        key_parts.append(f"filters:{filter_hash}")
    
    if columns:
        cols_hash = hashlib.md5(",".join(sorted(columns)).encode()).hexdigest()[:8]
        key_parts.append(f"cols:{cols_hash}")
    
    if limit:
        key_parts.append(f"limit:{limit}")
    
    return ":".join(key_parts)


# Global cache instance
_global_cache: Optional[RedisCache] = None


def get_cache(config: CacheConfig = None) -> RedisCache:
    """Get or create global cache instance.
    
    Args:
        config: Cache configuration
        
    Returns:
        Global cache instance
    """
    global _global_cache
    
    if _global_cache is None:
        _global_cache = RedisCache(config)
        try:
            _global_cache.connect()
        except Exception as e:
            logger.warning(f"Could not connect to Redis cache: {e}")
    
    return _global_cache


def shutdown_cache() -> None:
    """Shutdown global cache instance."""
    global _global_cache
    
    if _global_cache is not None:
        _global_cache.disconnect()
        _global_cache = None