"""Caching functionality for repositories."""

import time
import hashlib
import logging
from functools import wraps
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CachingMixin:
    """Adds in-memory caching to repository operations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[str, tuple[Any, float]] = {}
        self.cache_enabled = True
        self.default_ttl_seconds = 300  # 5 minutes
        self.max_cache_size = 1000  # Maximum number of cached items
        
    def _cache_key(self, method_name: str, *args, **kwargs) -> str:
        """Generate cache key for method call."""
        # Convert arguments to a hashable string
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        key_data = f"{method_name}:{args_str}:{kwargs_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_time: float, ttl_seconds: int) -> bool:
        """Check if cached item is still valid."""
        return time.time() - cached_time < ttl_seconds
    
    def _cleanup_cache(self) -> None:
        """Remove expired items and enforce size limits."""
        current_time = time.time()
        
        # Remove expired items
        expired_keys = [
            key for key, (_, cached_time) in self._cache.items()
            if current_time - cached_time >= self.default_ttl_seconds
        ]
        
        for key in expired_keys:
            self._cache.pop(key, None)
        
        # Enforce size limit by removing oldest items
        if len(self._cache) > self.max_cache_size:
            # Sort by cached time and remove oldest
            sorted_items = sorted(
                self._cache.items(), 
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            
            items_to_remove = len(self._cache) - self.max_cache_size
            for key, _ in sorted_items[:items_to_remove]:
                self._cache.pop(key, None)
    
    def cached(self, ttl_seconds: Optional[int] = None):
        """Decorator to cache method results.
        
        Args:
            ttl_seconds: Time to live in seconds. Uses default if None.
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                if not self.cache_enabled:
                    return func(self, *args, **kwargs)
                
                # Use provided TTL or default
                effective_ttl = ttl_seconds or self.default_ttl_seconds
                cache_key = self._cache_key(func.__name__, *args, **kwargs)
                
                # Check cache
                if cache_key in self._cache:
                    cached_result, cached_time = self._cache[cache_key]
                    if self._is_cache_valid(cached_time, effective_ttl):
                        logger.debug(f"Cache hit for {func.__name__} with key {cache_key[:8]}...")
                        return cached_result
                    else:
                        # Remove expired item
                        del self._cache[cache_key]
                
                # Execute function and cache result
                logger.debug(f"Cache miss for {func.__name__}, executing...")
                result = func(self, *args, **kwargs)
                
                # Cache the result
                self._cache[cache_key] = (result, time.time())
                
                # Periodic cleanup
                if len(self._cache) % 100 == 0:  # Cleanup every 100 operations
                    self._cleanup_cache()
                
                return result
            
            return wrapper
        return decorator
    
    def clear_cache(self, pattern: Optional[str] = None) -> None:
        """Clear cache entries.
        
        Args:
            pattern: If provided, only clear keys containing this pattern
        """
        if pattern is None:
            self._cache.clear()
            logger.info("Cleared entire repository cache")
        else:
            keys_to_remove = [
                key for key in self._cache.keys()
                if pattern in key
            ]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Cleared {len(keys_to_remove)} cache entries matching pattern '{pattern}'")
    
    def cache_info(self) -> Dict[str, Any]:
        """Get cache information and statistics."""
        current_time = time.time()
        
        # Count valid vs expired entries
        valid_count = 0
        expired_count = 0
        
        for _, cached_time in self._cache.values():
            if self._is_cache_valid(cached_time, self.default_ttl_seconds):
                valid_count += 1
            else:
                expired_count += 1
        
        return {
            "enabled": self.cache_enabled,
            "total_entries": len(self._cache),
            "valid_entries": valid_count,
            "expired_entries": expired_count,
            "max_size": self.max_cache_size,
            "default_ttl_seconds": self.default_ttl_seconds,
            "hit_ratio": getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
        }
    
    def is_cached(self, method_name: str, *args, **kwargs) -> bool:
        """Check if specific method call result is cached and valid.
        
        Args:
            method_name: Name of the method
            *args: Method arguments
            **kwargs: Method keyword arguments
            
        Returns:
            True if cached and valid, False otherwise
        """
        if not self.cache_enabled:
            return False
        
        cache_key = self._cache_key(method_name, *args, **kwargs)
        
        if cache_key in self._cache:
            _, cached_time = self._cache[cache_key]
            return self._is_cache_valid(cached_time, self.default_ttl_seconds)
        
        return False
    
    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable caching.
        
        Args:
            enabled: True to enable caching, False to disable
        """
        self.cache_enabled = enabled
        if not enabled:
            self.clear_cache()
        logger.info(f"Repository caching {'enabled' if enabled else 'disabled'}")
    
    def configure_cache(
        self,
        default_ttl_seconds: Optional[int] = None,
        max_cache_size: Optional[int] = None
    ) -> None:
        """Configure cache parameters.
        
        Args:
            default_ttl_seconds: Default TTL for cache entries
            max_cache_size: Maximum number of cache entries
        """
        if default_ttl_seconds is not None:
            self.default_ttl_seconds = default_ttl_seconds
        
        if max_cache_size is not None:
            self.max_cache_size = max_cache_size
            # Clean up if current size exceeds new limit
            if len(self._cache) > max_cache_size:
                self._cleanup_cache()
        
        logger.info(
            f"Cache configured: TTL={self.default_ttl_seconds}s, "
            f"Max size={self.max_cache_size}"
        )