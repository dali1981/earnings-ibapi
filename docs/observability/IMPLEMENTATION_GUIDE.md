# Trading Platform Observability Implementation Guide

This guide covers the comprehensive observability implementation using Sentry for error tracking, performance monitoring, and distributed tracing across the trading platform.

## Architecture Overview

The observability system is built around three key pillars:

1. **Error Tracking** - Automatic capture and categorization of exceptions
2. **Performance Monitoring** - Request/response timing and resource usage
3. **Distributed Tracing** - Cross-service request tracking and correlation

## Quick Start

### 1. Configuration

Copy `.env.example` to `.env` and configure your Sentry DSN:

```bash
APP_OBSERVABILITY__SENTRY_DSN=https://your-dsn@your-org.ingest.sentry.io/your-project-id
APP_OBSERVABILITY__SENTRY_ENVIRONMENT=dev
```

### 2. Automatic Initialization

Observability is automatically initialized when services start:

- **FastAPI Services**: Initialized in `@app.on_event("startup")`
- **Celery Workers**: Initialized via signal handlers
- **Background Jobs**: Manual initialization in job scripts

### 3. Zero-Config Monitoring

Most operations are automatically monitored:

- HTTP requests/responses
- Database operations
- Celery tasks
- IB API calls
- Repository operations

## Core Components

### 1. Error Tracking (`common/observability/core.py`)

Provides centralized error capture with trading-specific context:

```python
from common.observability import capture_trading_exception, capture_trading_event

# Automatic exception capture
try:
    result = risky_operation()
except Exception as e:
    capture_trading_exception(
        e,
        extra={"symbol": "AAPL", "operation": "bars_fetch"},
        tags={"data_source": "ib_api"}
    )
    raise

# Manual event logging
capture_trading_event(
    "Successfully processed option chain",
    level="info",
    extra={"symbol": "AAPL", "contracts_count": 150},
    tags={"operation": "option_chain_processing"}
)
```

### 2. Performance Metrics (`common/observability/metrics.py`)

Custom metrics collection integrated with Sentry:

```python
from common.observability import track_trading_operation

# Automatic operation tracking
with track_trading_operation("equity_bars_fetch", symbol="AAPL", data_source="ib_api"):
    bars = fetch_equity_bars("AAPL", start_date, end_date)
    # Metrics are automatically recorded
```

### 3. Distributed Tracing (`common/observability/tracing.py`)

Cross-service request correlation:

```python
from common.observability import trace_operation, trace_span

# Function-level tracing
@trace_operation("equity_bars_backfill", "Backfill historical equity bars")
def backfill_equity_bars(symbol: str, start_date: str, end_date: str):
    # Implementation automatically traced
    pass

# Manual span creation
with trace_span("database_query", tags={"symbol": "AAPL"}):
    records = repository.get_bars(symbol, start_date, end_date)
```

### 4. Context Management (`common/observability/context.py`)

Maintains correlation across async operations:

```python
from common.observability import trading_context, correlation_context

# Trading operation context
with trading_context("AAPL", "bars_fetch", correlation_id="req-123"):
    # All operations in this block are correlated
    process_symbol("AAPL")

# Cross-service correlation
with correlation_context("req-123"):
    # Maintains correlation across services
    await process_async_task()
```

## Integration Patterns

### 1. Repository Operations

Use `@monitor_repository_operation` for data layer monitoring:

```python
from common.observability import monitor_repository_operation

@monitor_repository_operation(
    "equity_bars_save",
    extract_symbol=lambda args, kwargs: kwargs.get('symbol'),
    extract_metadata=lambda args, kwargs: {
        'bar_size': kwargs.get('bar_size'),
        'records_count': len(args[1]) if len(args) > 1 else 0
    }
)
def save(self, df: pd.DataFrame, symbol: str, bar_size: str):
    # Repository implementation
    pass
```

### 2. API Endpoints

Use `@monitor_api_endpoint` for REST API monitoring:

```python
from common.observability import monitor_api_endpoint

@router.get("/equity/{symbol}/bars")
@monitor_api_endpoint(
    "equity_bars_get", 
    extract_symbol=lambda args, kwargs: args[0]
)
def get_bars(symbol: str, start: date, end: date):
    # Endpoint implementation
    pass
```

### 3. Celery Tasks

Use `@monitored_task` for background task monitoring:

```python
from common.observability import monitored_task

@celery_app.task
@monitored_task("equity_bars_backfill", tags={"data_source": "ib_api"})
def backfill_equity_bars(symbol: str, start_date: str, end_date: str):
    # Task implementation
    pass
```

### 4. IB API Calls

Use `@monitor_ib_api_call` for Interactive Brokers API monitoring:

```python
from common.observability import monitor_ib_api_call

@monitor_ib_api_call(
    "ib_historical_bars",
    extract_symbol=lambda args, kwargs: kwargs.get('contract', {}).get('symbol'),
    timeout_seconds=30
)
def req_historical_data(self, req_id: int, contract, end_date_time: str):
    # IB API implementation
    pass
```

## Dashboard Integration

The existing monitoring dashboard (`common/monitoring/dashboard.py`) includes observability metrics:

### New Endpoints

- `/api/observability/metrics` - Real-time trading metrics
- Updated `/api/config` - Observability configuration status

### Metrics Available

- **Operation Counts** - Success/failure rates by operation type
- **Performance Data** - Duration, throughput, resource usage
- **Error Analysis** - Error types, frequencies, affected symbols
- **Symbol Activity** - Most active symbols and operations

## Configuration Options

### Environment Variables

All configuration via environment variables with `APP_OBSERVABILITY__` prefix:

```bash
# Core Sentry Configuration
APP_OBSERVABILITY__SENTRY_DSN=https://...
APP_OBSERVABILITY__SENTRY_ENVIRONMENT=dev|test|prod
APP_OBSERVABILITY__SENTRY_RELEASE=1.0.0

# Performance Settings
APP_OBSERVABILITY__SENTRY_SAMPLE_RATE=1.0           # Error sampling (0.0-1.0)
APP_OBSERVABILITY__SENTRY_TRACES_SAMPLE_RATE=0.1    # Performance sampling
APP_OBSERVABILITY__SENTRY_PROFILES_SAMPLE_RATE=0.1  # Profiling sampling

# Feature Toggles
APP_OBSERVABILITY__ENABLE_ERROR_TRACKING=true
APP_OBSERVABILITY__ENABLE_PERFORMANCE_MONITORING=true
APP_OBSERVABILITY__ENABLE_DISTRIBUTED_TRACING=true
APP_OBSERVABILITY__ENABLE_CELERY_MONITORING=true
APP_OBSERVABILITY__ENABLE_CUSTOM_METRICS=true

# Alert Configuration
APP_OBSERVABILITY__ENABLE_ALERTS=true
APP_OBSERVABILITY__ALERT_EMAIL_RECIPIENTS=["admin@example.com"]
```

### Development vs Production

**Development Settings:**
- High sample rates (1.0) for comprehensive monitoring
- Debug mode enabled for detailed error information
- All features enabled

**Production Settings:**
- Lower sample rates (0.1-0.3) to reduce overhead
- Debug mode disabled
- Selective feature enablement based on needs

## Best Practices

### 1. Error Context

Always provide meaningful context with errors:

```python
# Good
capture_trading_exception(
    exception,
    extra={
        "symbol": symbol,
        "operation": "bars_fetch",
        "start_date": start_date,
        "end_date": end_date,
        "bar_size": bar_size
    },
    tags={
        "data_source": "ib_api",
        "operation_type": "historical_data"
    }
)

# Bad
capture_trading_exception(exception)
```

### 2. Performance Monitoring

Use appropriate sampling rates:

- **High-volume operations**: 0.01-0.1 (1-10%)
- **Medium-volume operations**: 0.1-0.3 (10-30%)
- **Low-volume operations**: 0.5-1.0 (50-100%)

### 3. Custom Tags

Use consistent tag naming:

```python
# Standard tags for trading operations
tags = {
    "symbol": symbol,                    # Trading symbol
    "operation": "bars_fetch",           # Operation type
    "data_source": "ib_api",            # Data source
    "bar_size": "1 day",                # Time frequency
    "environment": "prod"                # Environment
}
```

### 4. Correlation IDs

Always propagate correlation IDs across service boundaries:

```python
# In API handler
correlation_id = request.headers.get("X-Correlation-ID") or generate_id()

with correlation_context(correlation_id):
    # All operations are correlated
    result = await process_request(data)
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce sample rates
   - Limit breadcrumb count
   - Check for memory leaks in custom metrics

2. **Missing Traces**
   - Verify `SENTRY_DSN` is set
   - Check feature flags are enabled
   - Ensure proper context propagation

3. **Too Much Noise**
   - Implement before_send filters
   - Use appropriate log levels
   - Filter out expected exceptions

### Debug Mode

Enable debug logging to troubleshoot:

```bash
APP_OBSERVABILITY__SENTRY_DEBUG=true
APP_LOG_LEVEL=DEBUG
```

### Health Checks

Monitor observability health:

```python
from common.observability import get_sentry_hub

hub = get_sentry_hub()
if hub.client:
    print("Sentry client active")
else:
    print("Sentry client not initialized")
```

## Migration Guide

### Existing Code Integration

1. **Add decorators to existing functions:**
   ```python
   # Before
   def save_data(df, symbol):
       # implementation
   
   # After
   @monitor_repository_operation("data_save", extract_symbol=lambda args, kwargs: args[1])
   def save_data(df, symbol):
       # implementation
   ```

2. **Enhance error handling:**
   ```python
   # Before
   try:
       result = operation()
   except Exception as e:
       logger.error(f"Operation failed: {e}")
       raise
   
   # After
   try:
       result = operation()
   except Exception as e:
       capture_trading_exception(e, extra={"context": "important_context"})
       logger.error(f"Operation failed: {e}")
       raise
   ```

3. **Add manual instrumentation:**
   ```python
   # For critical operations without decorators
   with track_trading_operation("critical_operation", symbol="AAPL"):
       result = critical_operation()
   ```

## Support and Resources

- **Sentry Documentation**: https://docs.sentry.io/
- **Trading Platform Dashboard**: http://localhost:9090 (when running)
- **Observability Metrics**: `/api/observability/metrics`
- **Configuration Status**: `/api/config`

For issues or questions, check the logs at `data/logs/app.log` and verify configuration in `/api/config`.