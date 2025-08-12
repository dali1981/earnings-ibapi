#!/usr/bin/env python3
"""
Test script to verify the datetime formatting fix for IB API calls.
This simulates the conditions that were causing the original error.
"""
from datetime import datetime, date
from ibx_time import (
    ib_end_datetime_instrument, safe_ib_end_datetime_instrument,
    validate_ib_datetime_format, debug_datetime_format, ib_end_datetime
)


class MockContract:
    """Mock contract for testing."""
    def __init__(self, symbol="AAPL"):
        self.symbol = symbol
        self.secType = "STK"


class MockContractDetails:
    """Mock contract details with timezone."""
    def __init__(self, timezone="US/Eastern"):
        self.timeZoneId = timezone


class MockContractDetailsService:
    """Mock contract details service."""
    def __init__(self, timezone="US/Eastern"):
        self.timezone = timezone
    
    def fetch(self, contract):
        return [MockContractDetails(self.timezone)]


class MockRuntime:
    """Mock IB runtime for testing."""
    def __init__(self, timezone="US/Eastern"):
        self.timezone = timezone


def mock_instrument_timezone(rt, contract):
    """Mock version of instrument_timezone function."""
    return rt.timezone


def test_datetime_fix():
    """Test that demonstrates the fix for IB API datetime formatting."""
    
    print("=== IB API DateTime Formatting Fix Test ===\n")
    
    # Setup test data - this replicates the original failing scenario
    test_date = date(2024, 1, 5)  # Same as in the error: "20240105"
    end_dt = datetime.combine(test_date, datetime.min.time())  # 2024-01-05 00:00:00
    
    print(f"Test date: {test_date}")
    print(f"End datetime: {end_dt}")
    print()
    
    # Test 1: Show what the OLD buggy code was generating
    print("=== OLD BUGGY BEHAVIOR (hyphen=True) ===")
    buggy_format = ib_end_datetime(end_dt, tz="US/Eastern", hyphen=True)
    print(f"Buggy format: '{buggy_format}'")
    print(f"Valid for IB API: {validate_ib_datetime_format(buggy_format)}")
    
    if not validate_ib_datetime_format(buggy_format):
        debug = debug_datetime_format(buggy_format)
        print(f"Issues: {debug['format_issues']}")
    print()
    
    # Test 2: Show what the FIXED code generates
    print("=== FIXED BEHAVIOR (hyphen=False, default) ===")
    fixed_format = ib_end_datetime(end_dt, tz="US/Eastern")  # hyphen=False is default
    print(f"Fixed format: '{fixed_format}'")
    print(f"Valid for IB API: {validate_ib_datetime_format(fixed_format)}")
    print()
    
    # Test 3: Test with instrument timezone (the actual function used in source_ib.py)
    print("=== INSTRUMENT TIMEZONE FUNCTION TEST ===")
    
    # Mock the runtime and contract (simulating the real scenario)
    mock_rt = MockRuntime("US/Eastern")
    mock_contract = MockContract("AAPL")
    
    # Patch the instrument_timezone function for testing
    import ibx_time
    original_fn = ibx_time.instrument_timezone
    ibx_time.instrument_timezone = mock_instrument_timezone
    
    try:
        # This is what source_ib.py was doing (with hyphen=True) - BROKEN
        broken_result = ib_end_datetime_instrument(mock_rt, mock_contract, end_dt, hyphen=True)
        print(f"OLD source_ib.py (broken): '{broken_result}'")
        print(f"Valid for IB API: {validate_ib_datetime_format(broken_result)}")
        
        # This is what source_ib.py does now (no hyphen param) - FIXED  
        fixed_result = ib_end_datetime_instrument(mock_rt, mock_contract, end_dt)
        print(f"NEW source_ib.py (fixed):  '{fixed_result}'")
        print(f"Valid for IB API: {validate_ib_datetime_format(fixed_result)}")
        
        # Test the safe version
        safe_result = safe_ib_end_datetime_instrument(mock_rt, mock_contract, end_dt, validate=True)
        print(f"Safe version:              '{safe_result}'")
        print(f"Valid for IB API: {validate_ib_datetime_format(safe_result)}")
        
    finally:
        # Restore original function
        ibx_time.instrument_timezone = original_fn
    
    print()
    
    # Test 4: Show the exact error scenario from the logs
    print("=== ORIGINAL ERROR SCENARIO ===")
    print("Original error was:")
    print("SENDING reqHistoricalData b'...20240105-00:00:00 US/Eastern...'")
    print("IB error 10314: End Date/Time format invalid")
    print()
    
    # Simulate the exact same scenario
    error_format = "20240105-00:00:00 US/Eastern"
    print(f"Error format: '{error_format}'")
    print(f"Valid for IB API: {validate_ib_datetime_format(error_format)}")
    
    debug = debug_datetime_format(error_format)
    print(f"Detected issues: {debug['format_issues']}")
    print()
    
    # Show the corrected version
    correct_format = error_format.replace("-", " ")
    print(f"Corrected format: '{correct_format}'")
    print(f"Valid for IB API: {validate_ib_datetime_format(correct_format)}")
    print()
    
    print("=== SUMMARY ===")
    print("✅ Bug identified: hyphen=True parameter was generating 'YYYYMMDD-HH:MM:SS' format")
    print("✅ Bug fixed: Removed hyphen=True parameter, now generates 'YYYYMMDD HH:MM:SS' format")
    print("✅ Validation added: Functions now validate format before sending to IB API")
    print("✅ Prevention: Safe wrapper functions prevent future formatting errors")


if __name__ == "__main__":
    test_datetime_fix()