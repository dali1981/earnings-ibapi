"""
Specialized decorators for IB API calls with retry, circuit breaker, and monitoring.
Provides high-level abstractions for reliable financial trading operations.
"""
import time
import asyncio
import functools
import logging
from typing import Optional, Callable, Any, Dict, Union, Tuple, Type
from dataclasses import dataclass

from .retry import RetryManager, RetryConfig, RetryStrategy, JitterType
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, get_circuit_breaker
from .exceptions import (
    BaseTradingException, IBConnectionException, RateLimitException, 
    MarketDataTimeoutException, classify_ib_error, is_recoverable_error
)


@dataclass
class IBApiCallConfig:
    """Configuration for IB API call decorators."""
    
    # Retry configuration
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    retry_multiplier: float = 2.0
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: JitterType = JitterType.EQUAL
    
    # Circuit breaker configuration
    enable_circuit_breaker: bool = True
    circuit_breaker_name: Optional[str] = None
    failure_threshold: int = 5
    circuit_timeout: float = 60.0
    
    # Timeout configuration
    operation_timeout: Optional[float] = None
    
    # Rate limiting
    rate_limit_delay: Optional[float] = None
    
    # Logging
    log_attempts: bool = True
    log_performance: bool = True
    
    # Custom error handling
    custom_error_handler: Optional[Callable[[Exception], Exception]] = None
    ignore_errors: Tuple[Type[Exception], ...] = ()


class PerformanceMonitor:
    """Monitor performance metrics for API calls."""
    
    def __init__(self):
        self.call_times = {}
        self.call_counts = {}
        self.error_counts = {}
        self.logger = logging.getLogger(__name__)
    
    def record_call(self, function_name: str, duration: float, success: bool, 
                   exception: Optional[Exception] = None):
        """Record API call metrics."""
        
        # Track timing
        if function_name not in self.call_times:
            self.call_times[function_name] = []
        self.call_times[function_name].append(duration)
        
        # Track counts
        self.call_counts[function_name] = self.call_counts.get(function_name, 0) + 1
        
        # Track errors
        if not success and exception:
            error_type = type(exception).__name__
            error_key = f"{function_name}.{error_type}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def get_stats(self, function_name: str) -> Dict[str, Any]:
        """Get performance statistics for a function."""
        times = self.call_times.get(function_name, [])
        if not times:
            return {}
        
        return {
            'call_count': self.call_counts.get(function_name, 0),
            'avg_duration': sum(times) / len(times),
            'min_duration': min(times),
            'max_duration': max(times),
            'total_duration': sum(times),
            'error_rate': self._get_error_rate(function_name)
        }
    
    def _get_error_rate(self, function_name: str) -> float:
        """Calculate error rate for a function."""
        total_calls = self.call_counts.get(function_name, 0)
        if total_calls == 0:
            return 0.0
        
        error_count = sum(
            count for key, count in self.error_counts.items()
            if key.startswith(f"{function_name}.")
        )
        
        return error_count / total_calls


# Global performance monitor
_performance_monitor = PerformanceMonitor()


def ib_api_call(
    operation: str = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    enable_circuit_breaker: bool = True,
    timeout: Optional[float] = None,
    rate_limit_delay: Optional[float] = None,
    log_performance: bool = True,
    custom_error_handler: Optional[Callable[[Exception], Exception]] = None
):
    """
    Decorator for IB API calls with comprehensive error handling and monitoring.
    
    Args:
        operation: Description of the operation for logging
        max_retries: Maximum retry attempts
        base_delay: Base delay for exponential backoff
        max_delay: Maximum delay between retries
        enable_circuit_breaker: Whether to use circuit breaker
        timeout: Operation timeout in seconds
        rate_limit_delay: Delay to add for rate limiting
        log_performance: Whether to log performance metrics
        custom_error_handler: Custom error transformation function
    """
    
    def decorator(func: Callable) -> Callable:
        operation_name = operation or func.__name__
        circuit_breaker_name = f"ib_api.{operation_name}"
        
        # Setup retry configuration
        retry_config = RetryConfig(
            max_attempts=max_retries + 1,
            base_delay=base_delay,
            max_delay=max_delay,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter=JitterType.EQUAL,
            retry_on=(BaseTradingException, ConnectionError, TimeoutError),
            stop_on=(),
            retry_condition=is_recoverable_error,
            log_attempts=True
        )
        
        # Setup circuit breaker
        circuit_breaker = None
        if enable_circuit_breaker:
            cb_config = CircuitBreakerConfig(
                failure_threshold=5,
                timeout_duration=60.0,
                success_threshold=2
            )
            circuit_breaker = get_circuit_breaker(circuit_breaker_name, cb_config)
        
        retry_manager = RetryManager(retry_config)
        logger = logging.getLogger(f"{__name__}.{operation_name}")
        
        def execute_with_monitoring(execute_func: Callable) -> Any:
            """Execute function with performance monitoring."""
            start_time = time.time()
            exception = None
            
            try:
                result = execute_func()
                
                if log_performance:
                    duration = time.time() - start_time
                    _performance_monitor.record_call(operation_name, duration, True)
                    
                    if duration > 1.0:  # Log slow operations
                        logger.info(
                            f"IB API call '{operation_name}' completed in {duration:.2f}s",
                            extra={'operation': operation_name, 'duration': duration}
                        )
                
                return result
                
            except Exception as e:
                exception = e
                duration = time.time() - start_time
                
                # Transform error if custom handler provided
                if custom_error_handler:
                    try:
                        e = custom_error_handler(e)
                    except Exception:
                        pass  # Use original exception if handler fails
                
                # Record performance metrics
                if log_performance:
                    _performance_monitor.record_call(operation_name, duration, False, e)
                
                # Log error
                logger.error(
                    f"IB API call '{operation_name}' failed: {e}",
                    extra={
                        'operation': operation_name,
                        'duration': duration,
                        'exception_type': type(e).__name__,
                        'recoverable': is_recoverable_error(e)
                    }
                )
                
                raise
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Apply rate limiting
                if rate_limit_delay:
                    await asyncio.sleep(rate_limit_delay)
                
                async def execute_call():
                    if timeout:
                        return await asyncio.wait_for(func(*args, **kwargs), timeout)
                    else:
                        return await func(*args, **kwargs)
                
                # Apply circuit breaker if enabled
                if circuit_breaker:
                    return await execute_with_monitoring(
                        lambda: circuit_breaker.call_async(execute_call)
                    )
                else:
                    return await execute_with_monitoring(
                        lambda: retry_manager.execute_async(execute_call)
                    )
                    
            return async_wrapper
            
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Apply rate limiting
                if rate_limit_delay:
                    time.sleep(rate_limit_delay)
                
                def execute_call():
                    return func(*args, **kwargs)
                
                # Apply circuit breaker if enabled
                if circuit_breaker:
                    return execute_with_monitoring(
                        lambda: circuit_breaker.call(execute_call)
                    )
                else:
                    return execute_with_monitoring(
                        lambda: retry_manager.execute(execute_call)
                    )
                    
            return sync_wrapper
    
    return decorator


def market_data_request(
    symbol: Optional[str] = None,
    timeout: float = 30.0,
    max_retries: int = 2
):
    """
    Specialized decorator for market data requests.
    
    Args:
        symbol: Trading symbol for context
        timeout: Request timeout
        max_retries: Maximum retry attempts
    """
    
    def custom_error_handler(exception: Exception) -> Exception:
        """Transform IB API errors to trading exceptions."""
        if hasattr(exception, 'errorCode') and hasattr(exception, 'errorString'):
            return classify_ib_error(exception.errorCode, exception.errorString)
        
        if symbol and "timeout" in str(exception).lower():
            return MarketDataTimeoutException(symbol, timeout)
        
        return exception
    
    return ib_api_call(
        operation=f"market_data_{symbol}" if symbol else "market_data",
        max_retries=max_retries,
        base_delay=0.5,
        max_delay=5.0,
        timeout=timeout,
        rate_limit_delay=0.1,  # Small delay to avoid rate limits
        custom_error_handler=custom_error_handler
    )


def connection_operation(
    operation: str,
    critical: bool = True
):
    """
    Decorator for IB connection operations.
    
    Args:
        operation: Operation description
        critical: Whether operation is critical (affects retry behavior)
    """
    
    max_retries = 5 if critical else 2
    base_delay = 2.0 if critical else 1.0
    
    def custom_error_handler(exception: Exception) -> Exception:
        """Transform connection errors."""
        if isinstance(exception, ConnectionError):
            return IBConnectionException(str(exception), context={'operation': operation})
        return exception
    
    return ib_api_call(
        operation=f"connection_{operation}",
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=60.0,
        enable_circuit_breaker=True,
        custom_error_handler=custom_error_handler
    )


def historical_data_request(
    symbol: str,
    duration: str,
    bar_size: str,
    timeout: float = 60.0
):
    """
    Decorator for historical data requests.
    
    Args:
        symbol: Trading symbol
        duration: Data duration (e.g., "1 D", "5 D")
        bar_size: Bar size (e.g., "1 min", "1 day")
        timeout: Request timeout
    """
    
    operation = f"historical_data_{symbol}_{duration}_{bar_size}"
    
    return ib_api_call(
        operation=operation,
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        timeout=timeout,
        rate_limit_delay=0.2  # Historical data has stricter rate limits
    )


def contract_details_request(max_retries: int = 2):
    """
    Decorator for contract details requests.
    
    Args:
        max_retries: Maximum retry attempts
    """
    
    return ib_api_call(
        operation="contract_details",
        max_retries=max_retries,
        base_delay=0.5,
        max_delay=10.0,
        timeout=15.0,
        enable_circuit_breaker=False  # Contract details failures don't need circuit breaker
    )


def option_chain_request(
    symbol: str,
    timeout: float = 45.0
):
    """
    Decorator for option chain requests.
    
    Args:
        symbol: Underlying symbol
        timeout: Request timeout
    """
    
    return ib_api_call(
        operation=f"option_chain_{symbol}",
        max_retries=2,
        base_delay=1.0,
        max_delay=15.0,
        timeout=timeout,
        rate_limit_delay=0.3  # Option chains are expensive
    )


def resilient_operation(
    operation: str,
    max_retries: int = 5,
    circuit_breaker: bool = True
):
    """
    Decorator for critical operations requiring high resilience.
    
    Args:
        operation: Operation description
        max_retries: Maximum retry attempts
        circuit_breaker: Whether to enable circuit breaker
    """
    
    return ib_api_call(
        operation=operation,
        max_retries=max_retries,
        base_delay=2.0,
        max_delay=60.0,
        enable_circuit_breaker=circuit_breaker,
        log_performance=True
    )


# Utility functions for monitoring

def get_api_performance_stats() -> Dict[str, Dict[str, Any]]:
    """Get performance statistics for all monitored API calls."""
    stats = {}
    for function_name in _performance_monitor.call_times.keys():
        stats[function_name] = _performance_monitor.get_stats(function_name)
    return stats


def reset_performance_stats():
    """Reset all performance statistics."""
    global _performance_monitor
    _performance_monitor = PerformanceMonitor()


def log_circuit_breaker_status():
    """Log status of all circuit breakers."""
    from .circuit_breaker import _circuit_breakers
    
    logger = logging.getLogger(__name__)
    for name, breaker in _circuit_breakers.items():
        metrics = breaker.get_metrics()
        logger.info(
            f"Circuit breaker '{name}': {metrics['state']} "
            f"(failures: {metrics['failure_count']})",
            extra=metrics
        )