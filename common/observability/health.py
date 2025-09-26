"""
Health checks and circuit breaker monitoring for the trading platform.
Provides comprehensive system health monitoring and automatic recovery.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import time

import sentry_sdk
from app_settings.settings import get_settings
from common.observability.alerts import trigger_custom_alert, AlertSeverity


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing, requests blocked
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    response_time_ms: float
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_healthy(self) -> bool:
        """Check if result indicates healthy status."""
        return self.status == HealthStatus.HEALTHY


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    state: CircuitBreakerState
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    next_attempt_time: Optional[datetime]
    total_requests: int
    
    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failure_count / self.total_requests
    
    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        return 1.0 - self.failure_rate


class HealthCheck(ABC):
    """Base class for health checks."""
    
    def __init__(self, name: str, timeout_seconds: float = 5.0):
        self.name = name
        self.timeout_seconds = timeout_seconds
    
    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Perform the health check."""
        pass
    
    async def run_check(self) -> HealthCheckResult:
        """Run the health check with timeout and error handling."""
        start_time = time.time()
        
        try:
            # Use asyncio timeout to prevent hanging
            result = await asyncio.wait_for(
                self.check(),
                timeout=self.timeout_seconds
            )
            
            # Update response time
            response_time = (time.time() - start_time) * 1000
            result.response_time_ms = response_time
            
            return result
            
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message=f"Health check timed out after {self.timeout_seconds}s"
            )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message=f"Health check failed: {str(e)}",
                details={"error_type": type(e).__name__, "error": str(e)}
            )


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity."""
    
    def __init__(self, connection_factory: Callable, timeout_seconds: float = 5.0):
        super().__init__("database", timeout_seconds)
        self.connection_factory = connection_factory
    
    async def check(self) -> HealthCheckResult:
        """Check database connectivity."""
        try:
            # Test database connection
            conn = self.connection_factory()
            
            # Perform a simple query
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[0] == 1:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    response_time_ms=0.0,  # Will be set by run_check
                    message="Database connection successful"
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0.0,
                    message="Database query returned unexpected result"
                )
                
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0.0,
                message=f"Database connection failed: {str(e)}",
                details={"error": str(e)}
            )


class ExternalAPIHealthCheck(HealthCheck):
    """Health check for external API endpoints."""
    
    def __init__(self, name: str, url: str, expected_status: int = 200, timeout_seconds: float = 10.0):
        super().__init__(name, timeout_seconds)
        self.url = url
        self.expected_status = expected_status
    
    async def check(self) -> HealthCheckResult:
        """Check external API health."""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(self.url)
                
                if response.status_code == self.expected_status:
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.HEALTHY,
                        response_time_ms=0.0,
                        message=f"API responded with expected status {self.expected_status}",
                        details={
                            "status_code": response.status_code,
                            "url": self.url
                        }
                    )
                else:
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.DEGRADED,
                        response_time_ms=0.0,
                        message=f"API responded with unexpected status {response.status_code}",
                        details={
                            "status_code": response.status_code,
                            "expected_status": self.expected_status,
                            "url": self.url
                        }
                    )
                    
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0.0,
                message=f"API health check failed: {str(e)}",
                details={"error": str(e), "url": self.url}
            )


class IBAPIHealthCheck(HealthCheck):
    """Health check specifically for Interactive Brokers API."""
    
    def __init__(self, ib_client, timeout_seconds: float = 15.0):
        super().__init__("ib_api", timeout_seconds)
        self.ib_client = ib_client
    
    async def check(self) -> HealthCheckResult:
        """Check IB API connectivity and responsiveness."""
        try:
            # Check if client is connected
            if not hasattr(self.ib_client, 'isConnected') or not self.ib_client.isConnected():
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0.0,
                    message="IB API client is not connected"
                )
            
            # Test API responsiveness with a simple request
            # This would depend on your IB API wrapper implementation
            start_time = time.time()
            
            # Example: Request current time from IB (lightweight operation)
            if hasattr(self.ib_client, 'reqCurrentTime'):
                self.ib_client.reqCurrentTime()
                
                # Wait for response (implementation-specific)
                await asyncio.sleep(0.5)  # Give time for response
                
                response_time = (time.time() - start_time) * 1000
                
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    response_time_ms=response_time,
                    message="IB API is responsive"
                )
            
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    response_time_ms=0.0,
                    message="IB API client is connected"
                )
                
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0.0,
                message=f"IB API health check failed: {str(e)}",
                details={"error": str(e)}
            )


class DiskSpaceHealthCheck(HealthCheck):
    """Health check for disk space availability."""
    
    def __init__(self, path: str, min_free_gb: float = 1.0, timeout_seconds: float = 5.0):
        super().__init__("disk_space", timeout_seconds)
        self.path = path
        self.min_free_gb = min_free_gb
    
    async def check(self) -> HealthCheckResult:
        """Check available disk space."""
        try:
            import shutil
            
            total, used, free = shutil.disk_usage(self.path)
            free_gb = free / (1024 ** 3)  # Convert bytes to GB
            total_gb = total / (1024 ** 3)
            used_gb = used / (1024 ** 3)
            
            if free_gb >= self.min_free_gb:
                status = HealthStatus.HEALTHY
                message = f"Sufficient disk space available: {free_gb:.2f}GB free"
            elif free_gb >= self.min_free_gb * 0.5:
                status = HealthStatus.DEGRADED
                message = f"Low disk space warning: {free_gb:.2f}GB free"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk space: {free_gb:.2f}GB free"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                response_time_ms=0.0,
                message=message,
                details={
                    "path": self.path,
                    "free_gb": free_gb,
                    "total_gb": total_gb,
                    "used_gb": used_gb,
                    "min_required_gb": self.min_free_gb
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0.0,
                message=f"Disk space check failed: {str(e)}",
                details={"error": str(e), "path": self.path}
            )


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for external service calls.
    Prevents cascading failures by temporarily blocking requests to failing services.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 60,
        expected_exception: type = Exception,
        fallback: Optional[Callable] = None
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.expected_exception = expected_exception
        self.fallback = fallback
        
        # State tracking
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_success_time: Optional[datetime] = None
        self._next_attempt_time: Optional[datetime] = None
        self._total_requests = 0
    
    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state
    
    @property
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        return CircuitBreakerStats(
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=self._last_failure_time,
            last_success_time=self._last_success_time,
            next_attempt_time=self._next_attempt_time,
            total_requests=self._total_requests
        )
    
    def _record_success(self):
        """Record a successful operation."""
        self._success_count += 1
        self._last_success_time = datetime.now()
        
        if self._state == CircuitBreakerState.HALF_OPEN:
            # If we were testing, close the circuit
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0  # Reset failure count
            logger.info(f"Circuit breaker {self.name} closed after successful test")
    
    def _record_failure(self):
        """Record a failed operation."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        
        if self._failure_count >= self.failure_threshold:
            if self._state == CircuitBreakerState.CLOSED:
                # Open the circuit
                self._state = CircuitBreakerState.OPEN
                self._next_attempt_time = datetime.now() + timedelta(seconds=self.recovery_timeout_seconds)
                logger.warning(f"Circuit breaker {self.name} opened after {self._failure_count} failures")
                
                # Send alert
                asyncio.create_task(trigger_custom_alert(
                    title=f"Circuit Breaker Opened: {self.name}",
                    message=f"Circuit breaker for {self.name} has opened after {self._failure_count} consecutive failures. "
                           f"Requests will be blocked for {self.recovery_timeout_seconds} seconds.",
                    severity=AlertSeverity.HIGH,
                    tags={"circuit_breaker": self.name, "state": "open"}
                ))
            
            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Test failed, go back to open
                self._state = CircuitBreakerState.OPEN
                self._next_attempt_time = datetime.now() + timedelta(seconds=self.recovery_timeout_seconds)
                logger.warning(f"Circuit breaker {self.name} test failed, remaining open")
    
    def _can_attempt_reset(self) -> bool:
        """Check if we can attempt to reset the circuit breaker."""
        if self._state != CircuitBreakerState.OPEN:
            return False
        
        if self._next_attempt_time is None:
            return False
        
        return datetime.now() >= self._next_attempt_time
    
    async def call(self, func: Callable, *args, **kwargs):
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result or fallback result if circuit is open
            
        Raises:
            CircuitBreakerOpenError: If circuit is open and no fallback provided
        """
        self._total_requests += 1
        
        # Check circuit state
        if self._state == CircuitBreakerState.OPEN:
            if self._can_attempt_reset():
                # Try to close the circuit
                self._state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker {self.name} attempting recovery test")
            else:
                # Circuit is still open, use fallback or raise error
                if self.fallback:
                    logger.debug(f"Circuit breaker {self.name} is open, using fallback")
                    return await self.fallback(*args, **kwargs) if asyncio.iscoroutinefunction(self.fallback) else self.fallback(*args, **kwargs)
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
        
        # Execute the function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._record_success()
            return result
            
        except self.expected_exception as e:
            self._record_failure()
            
            # Capture error for observability
            with sentry_sdk.configure_scope() as scope:
                scope.set_tag("circuit_breaker", self.name)
                scope.set_tag("circuit_breaker_state", self._state.value)
                scope.set_context("circuit_breaker", {
                    "name": self.name,
                    "failure_count": self._failure_count,
                    "total_requests": self._total_requests,
                    "state": self._state.value
                })
                sentry_sdk.capture_exception(e)
            
            raise


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class HealthMonitor:
    """
    Comprehensive health monitoring system for the trading platform.
    Manages health checks and circuit breakers.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.health_checks: Dict[str, HealthCheck] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        
        self._setup_default_checks()
    
    def _setup_default_checks(self):
        """Setup default health checks."""
        # Disk space check
        self.add_health_check(DiskSpaceHealthCheck(
            path=str(self.settings.paths.data_root),
            min_free_gb=5.0  # Require at least 5GB free
        ))
        
        # Add more default checks as needed
    
    def add_health_check(self, check: HealthCheck) -> None:
        """Add a health check."""
        self.health_checks[check.name] = check
        logger.info(f"Added health check: {check.name}")
    
    def remove_health_check(self, name: str) -> None:
        """Remove a health check."""
        if name in self.health_checks:
            del self.health_checks[name]
            if name in self.last_results:
                del self.last_results[name]
            logger.info(f"Removed health check: {name}")
    
    def add_circuit_breaker(self, breaker: CircuitBreaker) -> None:
        """Add a circuit breaker."""
        self.circuit_breakers[breaker.name] = breaker
        logger.info(f"Added circuit breaker: {breaker.name}")
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks concurrently."""
        if not self.health_checks:
            return {}
        
        # Run all checks concurrently
        tasks = []
        check_names = []
        
        for name, check in self.health_checks.items():
            tasks.append(check.run_check())
            check_names.append(name)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        check_results = {}
        for i, result in enumerate(results):
            name = check_names[i]
            
            if isinstance(result, Exception):
                # Handle unexpected errors
                check_results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0.0,
                    message=f"Health check crashed: {str(result)}",
                    details={"error": str(result)}
                )
            else:
                check_results[name] = result
            
            # Cache the result
            self.last_results[name] = check_results[name]
        
        # Log unhealthy checks
        unhealthy_checks = [
            name for name, result in check_results.items() 
            if result.status == HealthStatus.UNHEALTHY
        ]
        
        if unhealthy_checks:
            logger.warning(f"Unhealthy checks detected: {', '.join(unhealthy_checks)}")
        
        return check_results
    
    def get_overall_status(self, results: Dict[str, HealthCheckResult] = None) -> HealthStatus:
        """
        Determine overall system health status.
        
        Args:
            results: Optional health check results, uses cached if not provided
            
        Returns:
            Overall health status
        """
        if results is None:
            results = self.last_results
        
        if not results:
            return HealthStatus.UNKNOWN
        
        # Count statuses
        status_counts = {status: 0 for status in HealthStatus}
        for result in results.values():
            status_counts[result.status] += 1
        
        # Determine overall status
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            return HealthStatus.DEGRADED
        elif status_counts[HealthStatus.HEALTHY] > 0:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        overall_status = self.get_overall_status()
        
        # Health check summary
        healthy_count = len([r for r in self.last_results.values() if r.status == HealthStatus.HEALTHY])
        degraded_count = len([r for r in self.last_results.values() if r.status == HealthStatus.DEGRADED])
        unhealthy_count = len([r for r in self.last_results.values() if r.status == HealthStatus.UNHEALTHY])
        
        # Circuit breaker summary
        circuit_breaker_summary = {}
        for name, breaker in self.circuit_breakers.items():
            stats = breaker.stats
            circuit_breaker_summary[name] = {
                "state": stats.state.value,
                "failure_count": stats.failure_count,
                "success_count": stats.success_count,
                "failure_rate": stats.failure_rate,
                "total_requests": stats.total_requests
            }
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "health_checks": {
                "total": len(self.health_checks),
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count
            },
            "circuit_breakers": circuit_breaker_summary,
            "details": {
                name: {
                    "status": result.status.value,
                    "response_time_ms": result.response_time_ms,
                    "message": result.message,
                    "timestamp": result.timestamp.isoformat()
                }
                for name, result in self.last_results.items()
            }
        }


# Global health monitor instance
_health_monitor = None


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor