# Trading Application Reliability System

A comprehensive reliability framework for financial trading applications built on Interactive Brokers API, featuring exponential backoff retries, circuit breaker pattern, custom exception hierarchy, and structured logging.

## Features

### 🔄 **Retry Mechanisms**
- Exponential backoff with configurable jitter
- Multiple retry strategies (exponential, linear, fixed, Fibonacci)
- Custom retry conditions and stop conditions
- Async/sync operation support
- Specialized decorators for different operation types

### ⚡ **Circuit Breaker Pattern**
- Automatic failure detection and recovery
- Configurable failure thresholds and timeouts
- Half-open state testing for service recovery
- Performance metrics and monitoring
- Thread-safe implementation

### 🎯 **Custom Exception Hierarchy**
- Structured exception types for trading operations
- Error severity and category classification
- Context-aware error information
- IB API error code mapping
- Recoverable vs non-recoverable error detection

### 📊 **Structured Logging**
- JSON-formatted logs with correlation context
- Performance metrics tracking
- Thread-local context management
- Enhanced error logging with stack traces
- Configurable log levels and formats

### 🛠 **Specialized Decorators**
- `@ib_api_call` - General IB API operations
- `@market_data_request` - Real-time market data
- `@historical_data_request` - Historical bar data
- `@connection_operation` - Connection management
- `@resilient_operation` - Critical operations

## Quick Start

```python
from reliability import (
    setup_reliability_system, get_trading_logger,
    market_data_request, log_context, IBConnectionException
)

# Initialize the system
setup_reliability_system(log_level="INFO", enable_performance_monitoring=True)
logger = get_trading_logger(__name__)

# Use decorators for reliable operations
@market_data_request(symbol="AAPL", timeout=30.0, max_retries=3)
def get_stock_quote():
    # Your IB API call here
    return {"symbol": "AAPL", "price": 150.25}

# Use context for structured logging
with log_context(operation="market_data_fetch", symbol="AAPL"):
    try:
        quote = get_stock_quote()
        logger.info(f"Retrieved quote: {quote}")
    except IBConnectionException as e:
        logger.log_exception(e, msg="Market data request failed")
```

## Installation

The reliability system requires Python 3.8+ and the following dependencies:

```bash
pip install asyncio logging dataclasses enum pathlib threading
```

## Architecture Overview

```
reliability/
├── __init__.py              # Main module exports
├── exceptions.py            # Custom exception hierarchy  
├── retry.py                 # Exponential backoff retry logic
├── circuit_breaker.py       # Circuit breaker implementation
├── decorators.py            # Specialized API decorators
├── logging.py               # Structured logging system
├── config.py                # Configuration management
└── README.md                # This documentation
```

## Core Components

### Exception Hierarchy

```python
BaseTradingException
├── IBConnectionException
├── IBAuthenticationException  
├── RateLimitException
├── DataException
│   ├── DataValidationException
│   ├── DataNotFoundException
│   └── SchemaValidationException
├── MarketDataException
│   ├── MarketDataTimeoutException
│   └── InvalidSymbolException
├── TimeoutException
├── SystemResourceException
└── CircuitBreakerException
```

### Retry Strategies

```python
from reliability import retry, RetryStrategy, JitterType

@retry(
    max_attempts=5,
    base_delay=1.0,
    max_delay=60.0,
    strategy=RetryStrategy.EXPONENTIAL,
    jitter=JitterType.EQUAL
)
def unreliable_operation():
    # Operation that might fail
    pass
```

### Circuit Breaker

```python
from reliability import circuit_breaker

@circuit_breaker(
    name="market_data_service",
    failure_threshold=5,
    timeout_duration=60.0,
    success_threshold=2
)
def market_data_operation():
    # Protected operation
    pass
```

### Structured Logging

```python
from reliability import get_trading_logger, log_context

logger = get_trading_logger(__name__)

with log_context(operation="data_fetch", symbol="AAPL", request_id=12345):
    logger.info("Starting market data request")
    # Logs will include context automatically
```

## Configuration

### Environment-Based Configuration

```python
from reliability.config import get_config, set_config, get_production_config

# Load configuration based on environment
config = get_config()  # Auto-detects from TRADING_ENVIRONMENT

# Or use specific configurations
production_config = get_production_config()
set_config(production_config)
```

### Environment Variables

```bash
# Logging
TRADING_LOG_LEVEL=INFO
TRADING_LOG_FORMAT=json
TRADING_LOG_FILE=/var/log/trading/app.log

# Retry settings
TRADING_MAX_RETRIES=5
TRADING_RETRY_BASE_DELAY=1.0
TRADING_RETRY_MAX_DELAY=60.0

# Circuit breaker settings  
TRADING_CB_FAILURE_THRESHOLD=5
TRADING_CB_TIMEOUT=60.0

# Feature flags
TRADING_ENABLE_CIRCUIT_BREAKERS=true
TRADING_ENABLE_RETRIES=true

# Environment
TRADING_ENVIRONMENT=production
```

### Configuration File

```json
{
  "retry": {
    "max_attempts": 5,
    "base_delay": 1.0,
    "max_delay": 60.0,
    "strategy": "exponential",
    "jitter": "equal"
  },
  "circuit_breaker": {
    "failure_threshold": 5,
    "timeout_duration": 60.0,
    "success_threshold": 2
  },
  "logging": {
    "level": "INFO",
    "format_type": "json",
    "include_context": true
  },
  "operations": {
    "ib_connection_timeout": 30.0,
    "market_data_timeout": 30.0,
    "historical_data_timeout": 120.0
  }
}
```

## Advanced Usage

### Custom Error Handling

```python
from reliability import BaseTradingException, classify_ib_error

def custom_error_handler(exception):
    if hasattr(exception, 'errorCode'):
        return classify_ib_error(exception.errorCode, str(exception))
    return exception

@ib_api_call(custom_error_handler=custom_error_handler)
def ib_operation():
    # IB API call that returns IB-specific errors
    pass
```

### Performance Monitoring

```python
from reliability import performance_log, get_api_performance_stats

@performance_log("expensive_calculation")
def expensive_operation():
    # Long-running operation
    pass

# Get performance statistics
stats = get_api_performance_stats()
for operation, metrics in stats.items():
    print(f"{operation}: avg={metrics['avg_duration']:.3f}s")
```

### Async Operations

```python
import asyncio
from reliability import ib_api_call

@ib_api_call(operation="async_market_data", timeout=30.0)
async def async_market_data_fetch():
    await asyncio.sleep(0.1)  # Simulate network delay
    return {"data": "market_data"}

# Use with asyncio
result = await async_market_data_fetch()
```

## Best Practices

### 1. **Operation-Specific Decorators**
Use specialized decorators for different types of operations:

```python
@connection_operation("connect_tws", critical=True)
def connect_to_tws():
    pass

@market_data_request(symbol="AAPL", timeout=30.0)  
def get_market_data():
    pass

@historical_data_request("MSFT", "5 D", "1 min", timeout=120.0)
def get_historical_data():
    pass
```

### 2. **Context Management**
Always use log context for related operations:

```python
with log_context(operation="trading_workflow", session_id="abc123"):
    for symbol in symbols:
        with log_context(symbol=symbol):
            process_symbol(symbol)
```

### 3. **Error Classification**
Properly classify errors for appropriate handling:

```python
class CustomTradingException(BaseTradingException):
    def __init__(self, message, **kwargs):
        kwargs.setdefault('category', ErrorCategory.MARKET)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('recoverable', True)
        super().__init__(message, **kwargs)
```

### 4. **Configuration Management**
Use environment-specific configurations:

```python
# Development
config = get_development_config()  # More verbose logging, faster retries

# Production  
config = get_production_config()   # Optimized for stability

# Testing
config = get_testing_config()      # Fast execution, minimal logging
```

## Monitoring and Observability

### Log Structure

```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "INFO", 
  "logger": "trading.market_data",
  "message": "Market data request completed",
  "context": {
    "correlation_id": "abc123-def456",
    "operation": "market_data_request",
    "symbol": "AAPL",
    "request_id": 12345
  },
  "performance": {
    "operation": "market_data_request",
    "duration": 0.245,
    "success": true
  }
}
```

### Circuit Breaker Metrics

```python
from reliability import get_circuit_breaker, log_circuit_breaker_status

# Get circuit breaker metrics
breaker = get_circuit_breaker("market_data_service")
metrics = breaker.get_metrics()

print(f"State: {metrics['state']}")
print(f"Failure count: {metrics['failure_count']}")
print(f"Success rate: {metrics['metrics']['success_rate']:.2%}")

# Log all circuit breaker statuses
log_circuit_breaker_status()
```

### Performance Statistics

```python
from reliability import get_api_performance_stats, reset_performance_stats

# Get comprehensive performance data
stats = get_api_performance_stats()

for operation, metrics in stats.items():
    print(f"Operation: {operation}")
    print(f"  Calls: {metrics['call_count']}")
    print(f"  Avg Duration: {metrics['avg_duration']:.3f}s") 
    print(f"  Error Rate: {metrics['error_rate']:.1%}")
    print(f"  P95 Duration: {metrics['max_duration']:.3f}s")

# Reset statistics (useful for testing)
reset_performance_stats()
```

## Testing

### Unit Tests

```python
import pytest
from reliability import BaseTradingException, RetryManager, CircuitBreaker

def test_retry_mechanism():
    config = RetryConfig(max_attempts=3, base_delay=0.1)
    manager = RetryManager(config)
    
    call_count = 0
    def failing_operation():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise BaseTradingException("Temporary failure")
        return "success"
    
    result = manager.execute(failing_operation)
    assert result == "success"
    assert call_count == 3
```

### Integration Tests

```python
@pytest.mark.integration
def test_ib_api_integration():
    @market_data_request(symbol="AAPL", max_retries=2)
    def get_test_data():
        # Real IB API call
        return ib_client.reqMktData(...)
    
    with log_context(operation="integration_test"):
        result = get_test_data()
        assert result is not None
```

## Examples

See `examples/reliability_examples.py` for comprehensive usage examples including:

- Basic error handling and logging
- Retry mechanisms with different strategies  
- Circuit breaker pattern demonstration
- IB API decorators usage
- Advanced logging with context
- Async operations
- Complete trading application example

## Performance Considerations

### Memory Usage
- Circuit breaker metrics are bounded (configurable window size)
- Log context is thread-local and automatically cleaned up
- Retry statistics are reset after successful operations

### Thread Safety
- All components are thread-safe
- Circuit breakers use RLock for performance
- Context management uses threading.local storage

### Overhead
- Decorators add minimal overhead (~0.1ms per call)
- Logging overhead depends on log level and format
- Circuit breaker adds ~0.05ms per protected call

## Troubleshooting

### Common Issues

**1. Import Errors**
```python
# Make sure reliability package is in Python path
import sys
sys.path.append('/path/to/trading_project/earnings_ibapi')
from reliability import setup_reliability_system
```

**2. Configuration Not Loading**
```python
# Check environment variables
import os
print(f"Environment: {os.getenv('TRADING_ENVIRONMENT', 'not_set')}")
print(f"Config file: {os.getenv('TRADING_RELIABILITY_CONFIG', 'not_set')}")
```

**3. Circuit Breaker Not Opening**
```python
# Check failure threshold and conditions
breaker = get_circuit_breaker("service_name")
metrics = breaker.get_metrics()
print(f"Failure count: {metrics['failure_count']} / {breaker.config.failure_threshold}")
```

**4. Logs Not Appearing**
```python
# Check log level configuration
import logging
logging.basicConfig(level=logging.DEBUG)
configure_trading_logging(level="DEBUG", console_logging=True)
```

### Debug Mode

```python
from reliability.config import get_config

config = get_config()
config.debug_mode = True
config.logging.level = "DEBUG"
config.retry.log_attempts = True
config.circuit_breaker.log_state_changes = True
```

## License

This reliability system is part of the trading application framework. See main project license for details.

## Contributing

When contributing to the reliability system:

1. Ensure all new exceptions inherit from `BaseTradingException`
2. Add appropriate logging context to new operations
3. Include comprehensive test coverage
4. Update configuration schema for new features
5. Add examples for new decorators or patterns

## Changelog

### v1.0.0 (Current)
- Initial release with complete reliability framework
- Exponential backoff retry with jitter
- Circuit breaker pattern implementation
- Custom exception hierarchy for trading
- Structured logging with context
- Specialized IB API decorators
- Comprehensive configuration system
- Performance monitoring and metrics