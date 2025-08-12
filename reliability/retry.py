"""
Exponential backoff retry mechanism for financial trading operations.
Provides configurable retry logic with jitter, maximum attempts, and custom stop conditions.
"""
import asyncio
import random
import time
import logging
from typing import Optional, Callable, Any, Union, Type, Tuple
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum

from .exceptions import (
    BaseTradingException, RateLimitException, CircuitBreakerException,
    is_recoverable_error, get_error_severity, ErrorSeverity
)


class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear" 
    FIXED = "fixed"
    FIBONACCI = "fibonacci"


class JitterType(Enum):
    """Types of jitter to apply to retry delays."""
    NONE = "none"
    FULL = "full"  # Random between 0 and calculated delay
    EQUAL = "equal"  # Add random amount up to calculated delay
    DECORRELATED = "decorrelated"  # AWS-style decorrelated jitter


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    # Basic retry settings
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    
    # Exponential backoff settings
    multiplier: float = 2.0
    jitter: JitterType = JitterType.EQUAL
    
    # Conditional retry settings
    retry_on: Tuple[Type[Exception], ...] = field(default_factory=lambda: (Exception,))
    stop_on: Tuple[Type[Exception], ...] = field(default_factory=tuple)
    
    # Custom conditions
    retry_condition: Optional[Callable[[Exception], bool]] = None
    
    # Logging
    log_attempts: bool = True
    log_level: int = logging.INFO


class RetryStatistics:
    """Statistics tracking for retry operations."""
    
    def __init__(self):
        self.total_attempts = 0
        self.successful_retries = 0
        self.failed_retries = 0
        self.total_delay = 0.0
        self.last_exception: Optional[Exception] = None
        
    def record_attempt(self, attempt: int, delay: float, exception: Optional[Exception] = None):
        """Record a retry attempt."""
        self.total_attempts = max(self.total_attempts, attempt)
        self.total_delay += delay
        if exception:
            self.last_exception = exception
            
    def record_success(self):
        """Record successful completion after retries."""
        if self.total_attempts > 1:
            self.successful_retries += 1
            
    def record_failure(self):
        """Record final failure after all retries."""
        self.failed_retries += 1


class RetryManager:
    """Manager for retry operations with exponential backoff."""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(__name__)
        self.statistics = RetryStatistics()
        
    def calculate_delay(self, attempt: int, base_delay: float = None) -> float:
        """Calculate delay for given attempt number."""
        base_delay = base_delay or self.config.base_delay
        
        if self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = base_delay * (self.config.multiplier ** (attempt - 1))
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = base_delay * attempt
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = base_delay * self._fibonacci(attempt)
        else:  # FIXED
            delay = base_delay
            
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Apply jitter
        delay = self._apply_jitter(delay, attempt)
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate Fibonacci number for retry delays."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    def _apply_jitter(self, delay: float, attempt: int) -> float:
        """Apply jitter to delay value."""
        if self.config.jitter == JitterType.NONE:
            return delay
        elif self.config.jitter == JitterType.FULL:
            return random.uniform(0, delay)
        elif self.config.jitter == JitterType.EQUAL:
            return delay + random.uniform(0, delay)
        elif self.config.jitter == JitterType.DECORRELATED:
            # AWS-style decorrelated jitter
            return random.uniform(self.config.base_delay, delay * 3)
        return delay
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if operation should be retried."""
        
        # Check maximum attempts
        if attempt >= self.config.max_attempts:
            return False
            
        # Check stop conditions
        if any(isinstance(exception, exc_type) for exc_type in self.config.stop_on):
            return False
            
        # Check circuit breaker
        if isinstance(exception, CircuitBreakerException):
            return False
            
        # Check custom retry condition
        if self.config.retry_condition:
            return self.config.retry_condition(exception)
            
        # Check retry conditions
        if not any(isinstance(exception, exc_type) for exc_type in self.config.retry_on):
            return False
            
        # Use custom logic for trading exceptions
        if isinstance(exception, BaseTradingException):
            return exception.recoverable
            
        # Default to is_recoverable_error function
        return is_recoverable_error(exception)
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                
                if attempt > 1:
                    self.statistics.record_success()
                    if self.config.log_attempts:
                        self.logger.info(f"Operation succeeded on attempt {attempt}")
                        
                return result
                
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(e, attempt):
                    break
                    
                if attempt < self.config.max_attempts:
                    delay = self.calculate_delay(attempt)
                    self.statistics.record_attempt(attempt, delay, e)
                    
                    if self.config.log_attempts:
                        severity = get_error_severity(e)
                        log_level = logging.ERROR if severity == ErrorSeverity.CRITICAL else self.config.log_level
                        
                        self.logger.log(
                            log_level,
                            f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s",
                            extra={
                                'attempt': attempt,
                                'delay': delay,
                                'exception_type': type(e).__name__,
                                'recoverable': is_recoverable_error(e)
                            }
                        )
                    
                    time.sleep(delay)
        
        # All attempts exhausted
        self.statistics.record_failure()
        if self.config.log_attempts:
            self.logger.error(
                f"All {self.config.max_attempts} attempts failed. Final exception: {last_exception}",
                extra={'total_attempts': self.config.max_attempts}
            )
        
        raise last_exception
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    
                if attempt > 1:
                    self.statistics.record_success()
                    if self.config.log_attempts:
                        self.logger.info(f"Async operation succeeded on attempt {attempt}")
                        
                return result
                
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(e, attempt):
                    break
                    
                if attempt < self.config.max_attempts:
                    delay = self.calculate_delay(attempt)
                    self.statistics.record_attempt(attempt, delay, e)
                    
                    if self.config.log_attempts:
                        self.logger.log(
                            self.config.log_level,
                            f"Async attempt {attempt} failed: {e}. Retrying in {delay:.2f}s",
                            extra={
                                'attempt': attempt,
                                'delay': delay,
                                'exception_type': type(e).__name__
                            }
                        )
                    
                    await asyncio.sleep(delay)
        
        # All attempts exhausted
        self.statistics.record_failure()
        raise last_exception


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
    jitter: JitterType = JitterType.EQUAL,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    stop_on: Tuple[Type[Exception], ...] = (),
    retry_condition: Optional[Callable[[Exception], bool]] = None,
    log_attempts: bool = True
):
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds
        multiplier: Exponential backoff multiplier
        jitter: Type of jitter to apply
        strategy: Retry strategy (exponential, linear, etc.)
        retry_on: Exception types to retry on
        stop_on: Exception types to stop retrying on
        retry_condition: Custom function to determine if retry should occur
        log_attempts: Whether to log retry attempts
    """
    
    def decorator(func: Callable) -> Callable:
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            multiplier=multiplier,
            jitter=jitter,
            strategy=strategy,
            retry_on=retry_on,
            stop_on=stop_on,
            retry_condition=retry_condition,
            log_attempts=log_attempts
        )
        
        retry_manager = RetryManager(config)
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await retry_manager.execute_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return retry_manager.execute(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator


# Specialized retry decorators for common scenarios
def ib_api_retry(
    max_attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 30.0
):
    """Specialized retry decorator for IB API calls."""
    return retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=(BaseTradingException, ConnectionError, TimeoutError),
        stop_on=(CircuitBreakerException,),
        retry_condition=lambda e: is_recoverable_error(e)
    )


def market_data_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0
):
    """Specialized retry decorator for market data requests."""
    return retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=RetryStrategy.LINEAR,
        retry_on=(BaseTradingException, TimeoutError),
        retry_condition=lambda e: not isinstance(e, RateLimitException) or is_recoverable_error(e)
    )


def data_operation_retry(
    max_attempts: int = 2,
    base_delay: float = 0.5
):
    """Specialized retry decorator for data operations."""
    return retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=RetryStrategy.FIXED,
        jitter=JitterType.NONE,
        retry_on=(IOError, OSError),
        log_attempts=False
    )