"""Common decorators for the trading platform."""

import time
import logging
import functools
from typing import Callable, Any, Dict, Optional, Type, Union
from datetime import datetime, timedelta

from .exceptions import APIException, ConfigurationException


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_failure: Optional[Callable] = None
):
    """Retry decorator with exponential backoff."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        if on_failure:
                            on_failure(func.__name__, attempt, e)
                        raise
                    
                    logging.warning(
                        f"Attempt {attempt} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None  # Should never reach here
        
        return wrapper
    return decorator


def rate_limit(calls_per_second: float):
    """Rate limiting decorator."""
    min_interval = 1.0 / calls_per_second
    last_called = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_key = f"{func.__module__}.{func.__name__}"
            now = time.time()
            
            if func_key in last_called:
                elapsed = now - last_called[func_key]
                if elapsed < min_interval:
                    sleep_time = min_interval - elapsed
                    time.sleep(sleep_time)
            
            last_called[func_key] = time.time()
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def cache_result(ttl_seconds: int = 300):
    """Simple in-memory cache decorator."""
    cache = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            now = time.time()
            
            # Check if we have a valid cached result
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if now - timestamp < ttl_seconds:
                    return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[cache_key] = (result, now)
            
            # Clean up expired entries periodically
            if len(cache) > 1000:  # Arbitrary limit
                expired_keys = [
                    key for key, (_, timestamp) in cache.items()
                    if now - timestamp >= ttl_seconds
                ]
                for key in expired_keys:
                    cache.pop(key, None)
            
            return result
        
        return wrapper
    return decorator


def log_execution_time(logger: Optional[logging.Logger] = None):
    """Log function execution time."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(
                    f"{func.__name__} completed in {execution_time:.3f}s"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"{func.__name__} failed after {execution_time:.3f}s: {e}"
                )
                raise
        
        return wrapper
    return decorator


def validate_config(required_fields: Dict[str, Type]):
    """Validate configuration parameters decorator."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if first argument is a config object
            if args and hasattr(args[0], '__dict__'):
                config = args[0]
                for field_name, field_type in required_fields.items():
                    if not hasattr(config, field_name):
                        raise ConfigurationException(
                            f"Missing required configuration field: {field_name}"
                        )
                    
                    field_value = getattr(config, field_name)
                    if not isinstance(field_value, field_type):
                        raise ConfigurationException(
                            f"Configuration field {field_name} must be of type {field_type.__name__}, "
                            f"got {type(field_value).__name__}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def handle_api_exceptions(
    default_return: Any = None,
    reraise: bool = True,
    log_errors: bool = True
):
    """Handle and transform API exceptions."""
    logger = logging.getLogger(__name__)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except APIException:
                # Re-raise API exceptions as-is
                raise
            except Exception as e:
                if log_errors:
                    logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                
                if reraise:
                    # Wrap in APIException
                    raise APIException(f"Error in {func.__name__}: {str(e)}") from e
                else:
                    return default_return
        
        return wrapper
    return decorator


def deprecated(reason: str, version: Optional[str] = None):
    """Mark function as deprecated."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import warnings
            
            message = f"{func.__name__} is deprecated"
            if version:
                message += f" since version {version}"
            if reason:
                message += f": {reason}"
            
            warnings.warn(
                message,
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def singleton(cls: Type) -> Type:
    """Singleton decorator for classes."""
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


class circuit_breaker:
    """Circuit breaker decorator for external service calls."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise APIException(
                        f"Circuit breaker OPEN for {func.__name__}. "
                        f"Service unavailable for {self.timeout_seconds}s."
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout_seconds
    
    def _on_success(self):
        """Reset circuit breaker on successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failure and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"