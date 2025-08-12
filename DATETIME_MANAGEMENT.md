# Centralized Date/Time Management for Trading Application

## Overview

This document outlines the centralized date/time management system implemented to fix IB API formatting bugs and ensure consistency across the trading codebase.

## Problem Solved

**Original Issue**: IB API `reqHistoricalData` was receiving malformed `endDateTime` strings like `"20240105-00:00:00 US/Eastern"` (with hyphen) instead of the required format `"20240105 00:00:00 US/Eastern"` (with space).

**Error Message**: 
```
IB error 10314: End Date/Time: The date, time, or time-zone entered is invalid. 
The correct format is yyyymmdd hh:mm:ss xx/xxxx where yyyymmdd and xx/xxxx are optional.
```

## Solution: Centralized `ibx_time` Module

### Key Functions

#### 1. **`ib_end_datetime_instrument()`** - Primary Function
```python
from ibx_time import ib_end_datetime_instrument

# CORRECT: Always use without hyphen parameter (defaults to False)
end_str = ib_end_datetime_instrument(rt, contract, datetime_obj)

# WRONG: This was causing the bug
end_str = ib_end_datetime_instrument(rt, contract, datetime_obj, hyphen=True)  # ❌
```

#### 2. **`safe_ib_end_datetime_instrument()`** - With Validation
```python
from ibx_time import safe_ib_end_datetime_instrument

# Safe version with automatic validation
end_str = safe_ib_end_datetime_instrument(rt, contract, datetime_obj, validate=True)
```

#### 3. **`validate_ib_datetime_format()`** - Format Checker
```python
from ibx_time import validate_ib_datetime_format, debug_datetime_format

# Validate format before sending to IB API
is_valid = validate_ib_datetime_format("20240105 15:30:00 US/Eastern")  # True
is_valid = validate_ib_datetime_format("20240105-15:30:00 US/Eastern")  # False

# Debug format issues
debug_info = debug_datetime_format("20240105-15:30:00 US/Eastern")
print(debug_info['format_issues'])  # ['Uses hyphen separator - IB API requires space']
```

## Valid IB API DateTime Formats

✅ **VALID Formats:**
- `"20240105"` (date only)
- `"20240105 15:30:00"` (date and time with space)
- `"20240105 15:30:00 US/Eastern"` (date, time, timezone with spaces)
- `"20240105 15:30:00 UTC"` (UTC timezone)

❌ **INVALID Formats:**
- `"20240105-15:30:00"` (hyphen separator)
- `"2024-01-05"` (ISO date format)
- `"20240105-15:30:00 US/Eastern"` (hyphen with timezone)

## Files Fixed

### Core Components Fixed:
1. **`ibx_flows/source_ib.py`**
   - `get_equity_bars()`: Removed `hyphen=True`
   - `get_option_bars()`: Removed `hyphen=True`

2. **`streamer.py`**
   - `send_minute_bars()`: Removed `hyphen=True`
   - `send_requests()`: Removed `hyphen=True` (2 instances)
   - `request_historical_bars()`: Removed `hyphen=True`

### Enhanced `ibx_time` Module:
- Added validation functions
- Added safe wrapper functions
- Added debugging helpers
- Added timezone utilities
- Enhanced documentation

## Usage Guidelines

### 1. **For IB API Historical Data Requests**
```python
# Always use space separator for IB API
from ibx_time import ib_end_datetime_instrument

end_str = ib_end_datetime_instrument(rt, contract, end_datetime)
client.reqHistoricalData(..., endDateTime=end_str, ...)
```

### 2. **For Date/Time Validation**
```python
from ibx_time import validate_ib_datetime_format

def safe_historical_request(contract, end_datetime):
    end_str = ib_end_datetime_instrument(rt, contract, end_datetime)
    
    if not validate_ib_datetime_format(end_str):
        raise ValueError(f"Invalid datetime format for IB API: {end_str}")
    
    return client.reqHistoricalData(..., endDateTime=end_str, ...)
```

### 3. **For Debugging Format Issues**
```python
from ibx_time import debug_datetime_format

# When troubleshooting datetime format problems
datetime_str = "20240105-15:30:00 US/Eastern"
analysis = debug_datetime_format(datetime_str)

print(f"Valid for IB: {analysis['is_valid_for_ib']}")
print(f"Issues: {analysis['format_issues']}")
```

### 4. **For Timezone Handling**
```python
from ibx_time import ensure_eastern_timezone, format_date_for_ib

# Ensure US/Eastern timezone for US markets
eastern_dt = ensure_eastern_timezone(datetime.now(), symbol="AAPL")

# Format dates for IB API
ib_date_str = format_date_for_ib(datetime.now())  # "20240105"
```

## Migration Checklist

When updating code that interacts with IB API datetime formatting:

### ✅ Do:
- Use `ib_end_datetime_instrument(rt, contract, dt)` without hyphen parameter
- Use `safe_ib_end_datetime_instrument()` for critical operations  
- Validate datetime strings with `validate_ib_datetime_format()`
- Use centralized `ibx_time` functions for all datetime operations

### ❌ Don't:
- Use `hyphen=True` parameter for IB API calls
- Manually format datetime strings for IB API
- Use ISO date formats (`2024-01-05`) for IB API
- Mix different datetime formatting approaches

## Testing

### Unit Test Example:
```python
import pytest
from ibx_time import validate_ib_datetime_format, ib_end_datetime

def test_ib_datetime_validation():
    # Valid formats
    assert validate_ib_datetime_format("20240105")
    assert validate_ib_datetime_format("20240105 15:30:00")  
    assert validate_ib_datetime_format("20240105 15:30:00 US/Eastern")
    
    # Invalid formats  
    assert not validate_ib_datetime_format("20240105-15:30:00")
    assert not validate_ib_datetime_format("2024-01-05")
    
def test_ib_end_datetime():
    # Should never use hyphen for IB API
    dt = datetime(2024, 1, 5, 15, 30, 0)
    
    # Correct usage
    result = ib_end_datetime(dt, tz="US/Eastern")
    assert validate_ib_datetime_format(result)
    assert "20240105 15:30:00 US/Eastern" == result
```

## Future Prevention

### Code Review Checklist:
1. ✅ All IB API datetime calls use centralized `ibx_time` functions
2. ✅ No `hyphen=True` parameters in IB API related code
3. ✅ Datetime validation in place for critical operations
4. ✅ Consistent timezone handling across the application

### IDE/Editor Integration:
Consider adding linting rules to catch:
- `hyphen=True` in IB API contexts
- Manual datetime string formatting for IB API
- Missing validation for IB API datetime parameters

## Error Handling

### Robust Error Handling Pattern:
```python
from ibx_time import safe_ib_end_datetime_instrument
from reliability import ib_api_call, IBConnectionException

@ib_api_call(operation="historical_data", max_retries=3)
def get_historical_data(rt, contract, end_datetime, duration, bar_size):
    try:
        end_str = safe_ib_end_datetime_instrument(
            rt, contract, end_datetime, validate=True
        )
        return hist_service.bars(
            contract, 
            endDateTime=end_str,
            durationStr=duration, 
            barSizeSetting=bar_size
        )
    except ValueError as e:
        raise IBConnectionException(f"Invalid datetime format: {e}")
```

## Summary

The centralized date/time management system:

1. **Fixes the IB API formatting bug** by enforcing space separators
2. **Prevents future issues** with validation and safe wrappers
3. **Provides debugging tools** for troubleshooting datetime issues
4. **Standardizes timezone handling** across the trading application
5. **Integrates with the reliability system** for robust error handling

**Key Takeaway**: Always use the centralized `ibx_time` module functions for IB API datetime operations, never the `hyphen=True` parameter.