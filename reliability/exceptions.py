"""
Custom exception hierarchy for financial trading application.
Provides structured error handling for IB API interactions, data operations, and system failures.
"""
from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for trading operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors in trading operations."""
    CONNECTION = "connection"
    AUTHENTICATION = "authentication"
    DATA = "data"
    MARKET = "market"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    SYSTEM = "system"


class BaseTradingException(Exception):
    """Base exception for all trading-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.recoverable = recoverable
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "exception_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "category": self.category.value,
            "recoverable": self.recoverable,
            "context": self.context
        }


# =============================================================================
# Connection-related Exceptions
# =============================================================================

class IBConnectionException(BaseTradingException):
    """IB API connection failures."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONNECTION)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class IBConnectionTimeoutException(IBConnectionException):
    """IB API connection timeout."""
    
    def __init__(self, timeout_seconds: float, **kwargs):
        message = f"IB connection timeout after {timeout_seconds} seconds"
        kwargs.setdefault('context', {}).update({'timeout_seconds': timeout_seconds})
        super().__init__(message, **kwargs)


class IBDisconnectedException(IBConnectionException):
    """IB API unexpected disconnection."""
    
    def __init__(self, message: str = "IB API disconnected unexpectedly", **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        super().__init__(message, **kwargs)


class IBAuthenticationException(BaseTradingException):
    """IB API authentication failures."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.AUTHENTICATION)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recoverable', False)
        super().__init__(message, **kwargs)


# =============================================================================
# Rate Limiting and Throttling Exceptions
# =============================================================================

class RateLimitException(BaseTradingException):
    """Rate limiting errors."""
    
    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.RATE_LIMIT)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        if retry_after:
            kwargs.setdefault('context', {}).update({'retry_after': retry_after})
        super().__init__(message, **kwargs)


class IBRateLimitException(RateLimitException):
    """IB API rate limiting."""
    
    def __init__(self, message: str = "IB API rate limit exceeded", **kwargs):
        super().__init__(message, **kwargs)


class RequestThrottledException(RateLimitException):
    """Request throttling by internal rate limiter."""
    
    def __init__(self, calls_per_second: float, **kwargs):
        message = f"Request throttled: exceeds {calls_per_second} calls/sec limit"
        kwargs.setdefault('context', {}).update({'calls_per_second': calls_per_second})
        super().__init__(message, **kwargs)


# =============================================================================
# Data-related Exceptions
# =============================================================================

class DataException(BaseTradingException):
    """Base class for data-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DATA)
        super().__init__(message, **kwargs)


class DataValidationException(DataException):
    """Data validation failures."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.VALIDATION)
        if field:
            kwargs.setdefault('context', {}).update({'field': field, 'value': value})
        super().__init__(message, **kwargs)


class DataNotFoundException(DataException):
    """Requested data not found."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.LOW)
        super().__init__(message, **kwargs)


class DataCorruptionException(DataException):
    """Data corruption detected."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recoverable', False)
        super().__init__(message, **kwargs)


class SchemaValidationException(DataValidationException):
    """Schema validation failures."""
    
    def __init__(self, message: str, expected_schema: Optional[str] = None, **kwargs):
        if expected_schema:
            kwargs.setdefault('context', {}).update({'expected_schema': expected_schema})
        super().__init__(message, **kwargs)


# =============================================================================
# Market Data Exceptions
# =============================================================================

class MarketDataException(BaseTradingException):
    """Market data related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.MARKET)
        super().__init__(message, **kwargs)


class MarketDataTimeoutException(MarketDataException):
    """Market data request timeout."""
    
    def __init__(self, symbol: str, timeout_seconds: float, **kwargs):
        message = f"Market data timeout for {symbol} after {timeout_seconds}s"
        kwargs.setdefault('category', ErrorCategory.TIMEOUT)
        kwargs.setdefault('context', {}).update({
            'symbol': symbol,
            'timeout_seconds': timeout_seconds
        })
        super().__init__(message, **kwargs)


class MarketDataUnavailableException(MarketDataException):
    """Market data not available."""
    
    def __init__(self, symbol: str, reason: Optional[str] = None, **kwargs):
        message = f"Market data unavailable for {symbol}"
        if reason:
            message += f": {reason}"
        kwargs.setdefault('context', {}).update({'symbol': symbol, 'reason': reason})
        super().__init__(message, **kwargs)


class InvalidSymbolException(MarketDataException):
    """Invalid trading symbol."""
    
    def __init__(self, symbol: str, **kwargs):
        message = f"Invalid symbol: {symbol}"
        kwargs.setdefault('severity', ErrorSeverity.LOW)
        kwargs.setdefault('recoverable', False)
        kwargs.setdefault('context', {}).update({'symbol': symbol})
        super().__init__(message, **kwargs)


# =============================================================================
# Timeout Exceptions
# =============================================================================

class TimeoutException(BaseTradingException):
    """Generic timeout exception."""
    
    def __init__(self, operation: str, timeout_seconds: float, **kwargs):
        message = f"Timeout in {operation} after {timeout_seconds} seconds"
        kwargs.setdefault('category', ErrorCategory.TIMEOUT)
        kwargs.setdefault('context', {}).update({
            'operation': operation,
            'timeout_seconds': timeout_seconds
        })
        super().__init__(message, **kwargs)


class RequestTimeoutException(TimeoutException):
    """Request-specific timeout."""
    
    def __init__(self, request_id: int, timeout_seconds: float, **kwargs):
        operation = f"request {request_id}"
        kwargs.setdefault('context', {}).update({'request_id': request_id})
        super().__init__(operation, timeout_seconds, **kwargs)


# =============================================================================
# System and Resource Exceptions
# =============================================================================

class SystemResourceException(BaseTradingException):
    """System resource exhaustion."""
    
    def __init__(self, resource: str, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SYSTEM)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('context', {}).update({'resource': resource})
        super().__init__(message, **kwargs)


class MemoryException(SystemResourceException):
    """Memory exhaustion."""
    
    def __init__(self, message: str = "Memory limit exceeded", **kwargs):
        super().__init__("memory", message, **kwargs)


class DiskSpaceException(SystemResourceException):
    """Disk space exhaustion."""
    
    def __init__(self, path: str, **kwargs):
        message = f"Insufficient disk space at {path}"
        kwargs.setdefault('context', {}).update({'path': path})
        super().__init__("disk", message, **kwargs)


# =============================================================================
# Circuit Breaker Exceptions
# =============================================================================

class CircuitBreakerException(BaseTradingException):
    """Circuit breaker is open."""
    
    def __init__(self, service: str, failure_count: int, **kwargs):
        message = f"Circuit breaker open for {service} (failures: {failure_count})"
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('context', {}).update({
            'service': service,
            'failure_count': failure_count
        })
        super().__init__(message, **kwargs)


# =============================================================================
# Repository and Storage Exceptions
# =============================================================================

class RepositoryException(BaseTradingException):
    """Repository operation failures."""
    
    def __init__(self, operation: str, repository: str, **kwargs):
        message = f"Repository {operation} failed in {repository}"
        kwargs.setdefault('context', {}).update({
            'operation': operation,
            'repository': repository
        })
        super().__init__(message, **kwargs)


class StorageException(BaseTradingException):
    """Storage system failures."""
    
    def __init__(self, message: str, storage_type: str = "unknown", **kwargs):
        kwargs.setdefault('category', ErrorCategory.SYSTEM)
        kwargs.setdefault('context', {}).update({'storage_type': storage_type})
        super().__init__(message, **kwargs)


# =============================================================================
# Exception Classification Utilities
# =============================================================================

def classify_ib_error(error_code: int, message: str) -> BaseTradingException:
    """Classify IB API errors into appropriate exception types."""
    
    # Connection errors
    if error_code in [502, 504, 1100, 1101, 1102]:
        return IBConnectionException(message, error_code=str(error_code))
    
    # Authentication errors
    elif error_code in [1101, 1102]:
        return IBAuthenticationException(message, error_code=str(error_code))
    
    # Rate limiting
    elif error_code in [162, 166]:
        return IBRateLimitException(message, error_code=str(error_code))
    
    # Market data errors
    elif error_code in [200, 354, 420]:
        return MarketDataException(message, error_code=str(error_code))
    
    # Timeout errors
    elif "timeout" in message.lower():
        return TimeoutException("IB API operation", 30.0, 
                               error_code=str(error_code), context={'ib_message': message})
    
    # Default to generic IB exception
    else:
        return BaseTradingException(message, error_code=str(error_code),
                                  category=ErrorCategory.SYSTEM)


def is_recoverable_error(exception: Exception) -> bool:
    """Determine if an error is recoverable for retry logic."""
    if isinstance(exception, BaseTradingException):
        return exception.recoverable
    
    # Network-related exceptions are typically recoverable
    if isinstance(exception, (ConnectionError, TimeoutError)):
        return True
    
    # Authentication and validation errors are not recoverable
    if "authentication" in str(exception).lower() or "unauthorized" in str(exception).lower():
        return False
    
    # Default to recoverable for unknown exceptions
    return True


def get_error_severity(exception: Exception) -> ErrorSeverity:
    """Get error severity for logging and alerting."""
    if isinstance(exception, BaseTradingException):
        return exception.severity
    
    # Map standard exceptions to severity levels
    if isinstance(exception, (ConnectionError, TimeoutError)):
        return ErrorSeverity.HIGH
    elif isinstance(exception, (ValueError, TypeError)):
        return ErrorSeverity.MEDIUM
    else:
        return ErrorSeverity.LOW