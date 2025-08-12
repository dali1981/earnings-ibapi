"""
Circuit breaker pattern implementation for financial trading applications.
Provides failure detection, automatic recovery, and system protection.
"""
import time
import threading
import logging
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from collections import deque, defaultdict

from .exceptions import (
    CircuitBreakerException, BaseTradingException, 
    get_error_severity, ErrorSeverity
)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    
    # Failure thresholds
    failure_threshold: int = 5  # Number of failures to open circuit
    success_threshold: int = 2  # Number of successes to close circuit from half-open
    
    # Time windows
    timeout_duration: float = 60.0  # Seconds to stay open
    monitoring_window: float = 300.0  # Window for failure rate calculation
    
    # Failure detection
    failure_rate_threshold: float = 0.5  # Percentage of failures to trigger
    minimum_requests: int = 10  # Minimum requests before applying failure rate
    
    # Recovery settings
    recovery_timeout: float = 30.0  # Timeout for half-open test requests
    max_half_open_requests: int = 1  # Max concurrent half-open requests
    
    # Custom conditions
    failure_condition: Optional[Callable[[Exception], bool]] = None
    
    # Logging and monitoring
    log_state_changes: bool = True
    collect_metrics: bool = True


class CircuitBreakerMetrics:
    """Metrics collection for circuit breaker."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.requests = deque(maxlen=window_size)
        self.lock = threading.Lock()
        
        # Counters
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_timeouts = 0
        self.total_rejections = 0
        
        # State tracking
        self.state_changes = []
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        
    def record_request(self, success: bool, response_time: float = 0.0, 
                      exception: Optional[Exception] = None):
        """Record a request attempt."""
        timestamp = time.time()
        
        with self.lock:
            request_data = {
                'timestamp': timestamp,
                'success': success,
                'response_time': response_time,
                'exception_type': type(exception).__name__ if exception else None
            }
            
            self.requests.append(request_data)
            self.total_requests += 1
            
            if success:
                self.total_successes += 1
                self.last_success_time = timestamp
            else:
                self.total_failures += 1
                self.last_failure_time = timestamp
    
    def record_rejection(self):
        """Record a request rejection due to open circuit."""
        with self.lock:
            self.total_rejections += 1
    
    def record_state_change(self, old_state: CircuitBreakerState, 
                           new_state: CircuitBreakerState):
        """Record circuit breaker state change."""
        timestamp = time.time()
        with self.lock:
            self.state_changes.append({
                'timestamp': timestamp,
                'old_state': old_state.value,
                'new_state': new_state.value
            })
    
    def get_failure_rate(self, window_seconds: float = 300.0) -> float:
        """Calculate failure rate within time window."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        with self.lock:
            recent_requests = [r for r in self.requests if r['timestamp'] >= cutoff_time]
            
            if not recent_requests:
                return 0.0
                
            failures = sum(1 for r in recent_requests if not r['success'])
            return failures / len(recent_requests)
    
    def get_request_count(self, window_seconds: float = 300.0) -> int:
        """Get request count within time window."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        with self.lock:
            return len([r for r in self.requests if r['timestamp'] >= cutoff_time])


class CircuitBreaker:
    """Circuit breaker implementation with failure detection and recovery."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # State management
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.next_attempt_time: Optional[float] = None
        self.half_open_requests = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Metrics
        self.metrics = CircuitBreakerMetrics() if self.config.collect_metrics else None
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened."""
        current_time = time.time()
        
        # Check failure count threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Check failure rate if we have enough requests
        if self.metrics:
            request_count = self.metrics.get_request_count(self.config.monitoring_window)
            if request_count >= self.config.minimum_requests:
                failure_rate = self.metrics.get_failure_rate(self.config.monitoring_window)
                if failure_rate >= self.config.failure_rate_threshold:
                    return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Determine if circuit should attempt to reset."""
        if self.state != CircuitBreakerState.OPEN:
            return False
            
        if self.next_attempt_time is None:
            return False
            
        return time.time() >= self.next_attempt_time
    
    def _transition_state(self, new_state: CircuitBreakerState):
        """Transition to new circuit breaker state."""
        old_state = self.state
        self.state = new_state
        current_time = time.time()
        
        if new_state == CircuitBreakerState.OPEN:
            self.next_attempt_time = current_time + self.config.timeout_duration
            
        elif new_state == CircuitBreakerState.HALF_OPEN:
            self.half_open_requests = 0
            self.success_count = 0
            
        elif new_state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
            self.next_attempt_time = None
            
        # Log state change
        if self.config.log_state_changes and old_state != new_state:
            self.logger.warning(
                f"Circuit breaker '{self.name}' state: {old_state.value} -> {new_state.value}",
                extra={
                    'circuit_breaker': self.name,
                    'old_state': old_state.value,
                    'new_state': new_state.value,
                    'failure_count': self.failure_count
                }
            )
            
        # Record metrics
        if self.metrics:
            self.metrics.record_state_change(old_state, new_state)
    
    def _is_failure(self, exception: Exception) -> bool:
        """Determine if exception should count as a failure."""
        
        # Use custom failure condition if provided
        if self.config.failure_condition:
            return self.config.failure_condition(exception)
        
        # Default failure logic
        if isinstance(exception, BaseTradingException):
            # Don't count low-severity or non-recoverable errors as failures
            severity = get_error_severity(exception)
            if severity == ErrorSeverity.LOW:
                return False
            # Validation errors shouldn't open circuit
            if not exception.recoverable:
                return False
        
        # Connection and timeout errors are failures
        if isinstance(exception, (ConnectionError, TimeoutError)):
            return True
            
        return True
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        
        with self.lock:
            # Check if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                if not self._should_attempt_reset():
                    if self.metrics:
                        self.metrics.record_rejection()
                    raise CircuitBreakerException(
                        service=self.name,
                        failure_count=self.failure_count
                    )
                else:
                    self._transition_state(CircuitBreakerState.HALF_OPEN)
            
            # Limit half-open requests
            elif self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_requests >= self.config.max_half_open_requests:
                    if self.metrics:
                        self.metrics.record_rejection()
                    raise CircuitBreakerException(
                        service=self.name,
                        failure_count=self.failure_count
                    )
                self.half_open_requests += 1
        
        # Execute the function
        start_time = time.time()
        exception = None
        
        try:
            result = func(*args, **kwargs)
            response_time = time.time() - start_time
            
            # Handle success
            with self.lock:
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.config.success_threshold:
                        self._transition_state(CircuitBreakerState.CLOSED)
                    else:
                        self.half_open_requests -= 1
                else:
                    # Reset failure count on success
                    self.failure_count = 0
            
            # Record metrics
            if self.metrics:
                self.metrics.record_request(True, response_time)
            
            return result
            
        except Exception as e:
            exception = e
            response_time = time.time() - start_time
            
            # Handle failure
            with self.lock:
                if self._is_failure(e):
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.state == CircuitBreakerState.HALF_OPEN:
                        # Failure during half-open test
                        self._transition_state(CircuitBreakerState.OPEN)
                    elif self.state == CircuitBreakerState.CLOSED:
                        # Check if we should open circuit
                        if self._should_open_circuit():
                            self._transition_state(CircuitBreakerState.OPEN)
                
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.half_open_requests -= 1
            
            # Record metrics
            if self.metrics:
                self.metrics.record_request(False, response_time, e)
            
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function through circuit breaker."""
        # Note: This is a simplified async version
        # In practice, you might want to use asyncio.timeout for better async handling
        
        import asyncio
        
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if not self._should_attempt_reset():
                    if self.metrics:
                        self.metrics.record_rejection()
                    raise CircuitBreakerException(
                        service=self.name,
                        failure_count=self.failure_count
                    )
                else:
                    self._transition_state(CircuitBreakerState.HALF_OPEN)
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_requests >= self.config.max_half_open_requests:
                    if self.metrics:
                        self.metrics.record_rejection()
                    raise CircuitBreakerException(
                        service=self.name,
                        failure_count=self.failure_count
                    )
                self.half_open_requests += 1
        
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            response_time = time.time() - start_time
            
            with self.lock:
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.config.success_threshold:
                        self._transition_state(CircuitBreakerState.CLOSED)
                    else:
                        self.half_open_requests -= 1
                else:
                    self.failure_count = 0
            
            if self.metrics:
                self.metrics.record_request(True, response_time)
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            
            with self.lock:
                if self._is_failure(e):
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.state == CircuitBreakerState.HALF_OPEN:
                        self._transition_state(CircuitBreakerState.OPEN)
                    elif self.state == CircuitBreakerState.CLOSED:
                        if self._should_open_circuit():
                            self._transition_state(CircuitBreakerState.OPEN)
                
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.half_open_requests -= 1
            
            if self.metrics:
                self.metrics.record_request(False, response_time, e)
            
            raise
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self.lock:
            return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'next_attempt_time': self.next_attempt_time,
                'metrics': {
                    'total_requests': self.metrics.total_requests,
                    'total_failures': self.metrics.total_failures,
                    'total_successes': self.metrics.total_successes,
                    'total_rejections': self.metrics.total_rejections,
                    'current_failure_rate': self.metrics.get_failure_rate(),
                } if self.metrics else None
            }
    
    def force_open(self):
        """Force circuit breaker to open state."""
        with self.lock:
            self._transition_state(CircuitBreakerState.OPEN)
    
    def force_close(self):
        """Force circuit breaker to closed state."""
        with self.lock:
            self._transition_state(CircuitBreakerState.CLOSED)


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout_duration: float = 60.0,
    success_threshold: int = 2,
    failure_condition: Optional[Callable[[Exception], bool]] = None
):
    """
    Decorator to apply circuit breaker pattern to functions.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Number of failures to open circuit
        timeout_duration: Seconds to keep circuit open
        success_threshold: Number of successes to close circuit
        failure_condition: Custom function to determine if exception is a failure
    """
    
    def decorator(func: Callable) -> Callable:
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            timeout_duration=timeout_duration,
            success_threshold=success_threshold,
            failure_condition=failure_condition
        )
        
        breaker = get_circuit_breaker(name, config)
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await breaker.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return breaker.call(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator