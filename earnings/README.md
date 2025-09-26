# Earnings Calendar API - Refactored

## Overview

The earnings calendar API has been refactored to address the following issues:
1. **HTTP client creation in loops** - Fixed by properly managing client lifecycle
2. **Data schema inconsistencies** - Resolved with Pydantic models for validation
3. **Mixed sync/async calls** - Unified with proper async support and thread-safe sync wrappers

## Key Components

### 1. Pydantic Models (`models.py`)
- `EarningsEvent`: Core model with validation for earnings data
- `EarningsSource`: Enum for data sources (NASDAQ, FMP, etc.)
- `EarningsTimeOfDay`: Enum for announcement timing (BMO, AMC, DMT)
- `EarningsCalendarResponse`: Response wrapper with filtering/deduplication
- `NasdaqEarningsRow/Response`: NASDAQ-specific models

### 2. Async Fetcher (`async_calendar_fetcher.py`)
- Proper async/await implementation
- Connection pooling with context managers
- Rate limiting and retry logic
- Cache support for performance

### 3. Unified Fetcher (`unified_fetcher.py`)
- Both sync and async interfaces
- Thread-safe sync wrapper for async calls
- Singleton pattern for convenience
- Backward compatibility maintained

## Usage Examples

### Synchronous Usage
```python
from earnings import UnifiedEarningsCalendarFetcher

# Create fetcher
fetcher = UnifiedEarningsCalendarFetcher()

# Get upcoming earnings
response = fetcher.get_upcoming_earnings(
    symbols=["AAPL", "GOOGL", "MSFT"],
    days_ahead=7
)

for event in response.events:
    print(f"{event.symbol}: {event.earnings_date} ({event.time.value})")

# Clean up
fetcher.close()
```

### Asynchronous Usage
```python
import asyncio
from earnings import UnifiedEarningsCalendarFetcher

async def main():
    fetcher = UnifiedEarningsCalendarFetcher()
    
    response = await fetcher.async_get_upcoming_earnings(
        symbols=["AAPL", "GOOGL", "MSFT"],
        days_ahead=14
    )
    
    for event in response.events:
        print(f"{event.symbol}: {event.earnings_date}")
    
    fetcher.close()

asyncio.run(main())
```

### Convenience Functions
```python
from earnings import get_upcoming_earnings

# Simple function call - uses default singleton
events = get_upcoming_earnings(days_ahead=7)

for event in events:
    print(f"{event.symbol} on {event.earnings_date}")
```

### With Context Manager (Async)
```python
from earnings import AsyncEarningsCalendarFetcher

async with AsyncEarningsCalendarFetcher() as fetcher:
    response = await fetcher.get_upcoming_earnings(
        symbols=["TSLA", "AMZN"],
        days_ahead=30
    )
    # Client is automatically closed on exit
```

## Features

### Data Validation
- Automatic symbol normalization (lowercase → uppercase, trimming)
- Time parsing (various formats → standardized enums)
- Invalid data filtering
- Type safety with Pydantic

### Performance
- Connection pooling for efficient HTTP requests
- Caching with configurable TTL
- Rate limiting to respect API limits
- Parallel fetching for date ranges

### Error Handling
- Retry logic for transient failures
- Graceful degradation on API errors
- Validation errors logged but don't stop processing
- Thread-safe sync operations

## Migration from Legacy

The new implementation maintains backward compatibility:

```python
# Old way (still works)
from earnings import EarningsCalendarFetcher
fetcher = EarningsCalendarFetcher()
events = fetcher.get_upcoming_earnings()

# New way (recommended)
from earnings import UnifiedEarningsCalendarFetcher
fetcher = UnifiedEarningsCalendarFetcher()
response = fetcher.get_upcoming_earnings()
events = response.events
```

## Testing

Run the test suite:
```bash
python test_earnings_calendar.py
```

Tests cover:
- Pydantic model validation
- Sync and async interfaces
- Caching functionality
- Error handling
- Convenience functions

## Configuration

Environment variables:
- `RATE_LIMIT_NASDAQ`: Seconds between requests (default: 0.5)
- `EARNINGS_ALLOWED_SOURCES`: Comma-separated sources (default: "nasdaq")

## Architecture Benefits

1. **Separation of Concerns**: Models, fetching logic, and caching are separate
2. **Type Safety**: Pydantic ensures data integrity
3. **Testability**: Clear interfaces make testing easier
4. **Performance**: Proper async implementation with connection reuse
5. **Maintainability**: Clean code structure with proper error handling