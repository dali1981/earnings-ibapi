"""
Comprehensive reliability system for financial trading applications.

This package provides:
- Custom exception hierarchy for trading operations
- Exponential backoff retry mechanisms with jitter
- Circuit breaker pattern for fault tolerance
- Specialized decorators for IB API calls
- Structured logging with correlation context
- Performance monitoring and metrics

Usage:
    from reliability import (
        ib_api_call, market_data_request, 
        get_trading_logger, log_context,
        BaseTradingException, RateLimitException
    )
    
    @market_data_request(symbol="AAPL", timeout=30.0)
    def get_market_data():
        # Your IB API call here
        pass
    
    logger = get_trading_logger(__name__)
    with log_context(operation="data_fetch", symbol="AAPL"):
        logger.info("Starting market data fetch")
"""

# Core exceptions
from .exceptions import (
    # Base exceptions
    BaseTradingException,
    ErrorSeverity,
    ErrorCategory,
    
    # Connection exceptions
    IBConnectionException,
    IBConnectionTimeoutException,
    IBDisconnectedException,
    IBAuthenticationException,
    
    # Rate limiting exceptions
    RateLimitException,
    IBRateLimitException,
    RequestThrottledException,
    
    # Data exceptions
    DataException,
    DataValidationException,
    DataNotFoundException,
    DataCorruptionException,
    SchemaValidationException,
    
    # Market data exceptions
    MarketDataException,
    MarketDataTimeoutException,
    MarketDataUnavailableException,
    InvalidSymbolException,
    
    # Timeout exceptions
    TimeoutException,
    RequestTimeoutException,
    
    # System exceptions
    SystemResourceException,
    MemoryException,
    DiskSpaceException,
    
    # Circuit breaker exceptions
    CircuitBreakerException,
    
    # Repository exceptions
    RepositoryException,
    StorageException,
    
    # Utility functions
    classify_ib_error,
    is_recoverable_error,
    get_error_severity
)

# Retry mechanisms
from .retry import (
    # Core classes
    RetryManager,
    RetryConfig,
    RetryStrategy,
    JitterType,
    RetryStatistics,
    
    # Decorators
    retry,
    ib_api_retry,
    market_data_retry,
    data_operation_retry
)

# Circuit breaker
from .circuit_breaker import (
    # Core classes
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerMetrics,
    
    # Decorator and utilities
    circuit_breaker,
    get_circuit_breaker
)

# Specialized decorators
from .decorators import (
    # Main decorators
    ib_api_call,
    market_data_request,
    connection_operation,
    historical_data_request,
    contract_details_request,
    option_chain_request,
    resilient_operation,
    
    # Monitoring utilities
    get_api_performance_stats,
    reset_performance_stats,
    log_circuit_breaker_status
)

# Structured logging
from .logging import (
    # Core classes
    LogContext,
    StructuredFormatter,
    TradingLoggerAdapter,
    
    # Context management
    log_context,
    get_current_context,
    set_current_context,
    clear_current_context,
    
    # Logger utilities
    get_trading_logger,
    configure_trading_logging,
    performance_log,
    
    # Specialized logging functions
    log_ib_api_call,
    log_ib_api_response,
    log_data_operation,
    log_retry_attempt,
    log_circuit_breaker_event
)

# Version info
__version__ = "1.0.0"
__author__ = "Trading System"

# All public exports
__all__ = [
    # Exceptions
    "BaseTradingException",
    "ErrorSeverity", 
    "ErrorCategory",
    "IBConnectionException",
    "IBConnectionTimeoutException", 
    "IBDisconnectedException",
    "IBAuthenticationException",
    "RateLimitException",
    "IBRateLimitException",
    "RequestThrottledException",
    "DataException",
    "DataValidationException",
    "DataNotFoundException",
    "DataCorruptionException",
    "SchemaValidationException",
    "MarketDataException",
    "MarketDataTimeoutException",
    "MarketDataUnavailableException",
    "InvalidSymbolException",
    "TimeoutException",
    "RequestTimeoutException",
    "SystemResourceException",
    "MemoryException",
    "DiskSpaceException",
    "CircuitBreakerException",
    "RepositoryException",
    "StorageException",
    "classify_ib_error",
    "is_recoverable_error",
    "get_error_severity",
    
    # Retry
    "RetryManager",
    "RetryConfig",
    "RetryStrategy",
    "JitterType",
    "RetryStatistics",
    "retry",
    "ib_api_retry",
    "market_data_retry",
    "data_operation_retry",
    
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "CircuitBreakerMetrics",
    "circuit_breaker",
    "get_circuit_breaker",
    
    # Decorators
    "ib_api_call",
    "market_data_request",
    "connection_operation", 
    "historical_data_request",
    "contract_details_request",
    "option_chain_request",
    "resilient_operation",
    "get_api_performance_stats",
    "reset_performance_stats",
    "log_circuit_breaker_status",
    
    # Logging
    "LogContext",
    "StructuredFormatter",
    "TradingLoggerAdapter",
    "log_context",
    "get_current_context",
    "set_current_context", 
    "clear_current_context",
    "get_trading_logger",
    "configure_trading_logging",
    "performance_log",
    "log_ib_api_call",
    "log_ib_api_response",
    "log_data_operation",
    "log_retry_attempt",
    "log_circuit_breaker_event"
]


# Quick setup function for convenience
def setup_reliability_system(
    log_level: str = "INFO",
    log_file: str = None,
    enable_performance_monitoring: bool = True,
    circuit_breaker_defaults: dict = None
):
    """
    Quick setup function for the reliability system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        enable_performance_monitoring: Enable API performance monitoring
        circuit_breaker_defaults: Default circuit breaker configuration
    """
    
    # Configure logging
    configure_trading_logging(
        level=log_level,
        format_type="json",
        log_file=log_file,
        console_logging=True
    )
    
    # Set up default circuit breaker configuration
    if circuit_breaker_defaults:
        # This would be used to set global defaults
        pass
    
    logger = get_trading_logger(__name__)
    logger.info("Trading reliability system initialized", extra={
        'log_level': log_level,
        'log_file': log_file,
        'performance_monitoring': enable_performance_monitoring
    })