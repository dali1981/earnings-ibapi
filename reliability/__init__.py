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

# Circuit breaker
from .circuit_breaker import (  # Core classes; Decorator and utilities
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerMetrics,
    CircuitBreakerState, circuit_breaker, get_circuit_breaker)
# Specialized decorators
from .decorators import (  # Main decorators; Monitoring utilities
    connection_operation, contract_details_request, get_api_performance_stats,
    historical_data_request, ib_api_call, log_circuit_breaker_status,
    market_data_request, option_chain_request, reset_performance_stats,
    resilient_operation)
# Core exceptions
from .exceptions import (  # Base exceptions; Connection exceptions; Rate limiting exceptions; Data exceptions; Market data exceptions; Timeout exceptions; System exceptions; Circuit breaker exceptions; Repository exceptions; Utility functions
    BaseTradingException, CircuitBreakerException, DataCorruptionException,
    DataException, DataNotFoundException, DataValidationException,
    DiskSpaceException, ErrorCategory, ErrorSeverity,
    IBAuthenticationException, IBConnectionException,
    IBConnectionTimeoutException, IBDisconnectedException,
    IBRateLimitException, InvalidSymbolException, MarketDataException,
    MarketDataTimeoutException, MarketDataUnavailableException,
    MemoryException, RateLimitException, RepositoryException,
    RequestThrottledException, RequestTimeoutException,
    SchemaValidationException, StorageException, SystemResourceException,
    TimeoutException, classify_ib_error, get_error_severity,
    is_recoverable_error)
# Structured logging
from .logging import (  # Core classes; Context management; Logger utilities; Specialized logging functions
    LogContext, StructuredFormatter, TradingLoggerAdapter,
    clear_current_context, configure_trading_logging, get_current_context,
    get_trading_logger, log_circuit_breaker_event, log_context,
    log_data_operation, log_ib_api_call, log_ib_api_response,
    log_retry_attempt, performance_log, set_current_context)
# Retry mechanisms
from .retry import RetryConfig  # Core classes; Decorators
from .retry import (JitterType, RetryManager, RetryStatistics, RetryStrategy,
                    data_operation_retry, ib_api_retry, market_data_retry,
                    retry)

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
    "log_circuit_breaker_event",
]


# Quick setup function for convenience
def setup_reliability_system(
    log_level: str = "INFO",
    log_file: str = None,
    enable_performance_monitoring: bool = True,
    circuit_breaker_defaults: dict = None,
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
        level=log_level, format_type="json", log_file=log_file, console_logging=True
    )

    # Set up default circuit breaker configuration
    if circuit_breaker_defaults:
        # This would be used to set global defaults
        pass

    logger = get_trading_logger(__name__)
    logger.info(
        "Trading reliability system initialized",
        extra={
            "log_level": log_level,
            "log_file": log_file,
            "performance_monitoring": enable_performance_monitoring,
        },
    )
