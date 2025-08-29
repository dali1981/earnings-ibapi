#!/usr/bin/env python3
"""
Test script for the refactored earnings calendar API.
Tests both sync and async interfaces with Pydantic validation.
"""

from datetime import date, timedelta
from pathlib import Path

import asyncio
import sys


# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from common.utils.logging_setup import setup_logging
from earnings import (
    async_get_upcoming_earnings,
    EarningsEvent,
    EarningsSource,
    EarningsTimeOfDay,
    get_upcoming_earnings,
    UnifiedEarningsCalendarFetcher,
)


def test_pydantic_models():
    """Test Pydantic model validation."""
    print("\n" + "=" * 60)
    print("Testing Pydantic Models")
    print("=" * 60)

    # Test valid event
    try:
        event = EarningsEvent(
            symbol="AAPL",
            company_name="Apple Inc.",
            earnings_date=date.today() + timedelta(days=5),
            time=EarningsTimeOfDay.AMC,
            eps_estimate=1.25,
            source=EarningsSource.NASDAQ,
        )
        print(f"✅ Created valid event: {event.symbol} on {event.earnings_date}")
        print(f"   Priority score: {event.priority_score}")
        print(f"   Days until: {event.days_until_earnings}")
    except Exception as e:
        print(f"❌ Failed to create valid event: {e}")

    # Test symbol validation
    try:
        event = EarningsEvent(
            symbol="  aapl  ",  # Should be normalized to AAPL
            company_name="Apple Inc.",
            earnings_date=date.today(),
            source=EarningsSource.NASDAQ,
        )
        print(f"✅ Symbol normalization works: '  aapl  ' -> '{event.symbol}'")
    except Exception as e:
        print(f"❌ Symbol normalization failed: {e}")

    # Test time parsing
    test_times = ["bmo", "amc", "pre-market", "after", "unknown", None]
    for time_str in test_times:
        try:
            event = EarningsEvent(
                symbol="TEST",
                company_name="Test Co",
                earnings_date=date.today(),
                time=time_str,
                source=EarningsSource.NASDAQ,
            )
            print(f"✅ Time parsing: '{time_str}' -> {event.time.value}")
        except Exception as e:
            print(f"❌ Time parsing failed for '{time_str}': {e}")


def test_sync_interface():
    """Test synchronous fetcher interface."""
    print("\n" + "=" * 60)
    print("Testing Synchronous Interface")
    print("=" * 60)

    fetcher = UnifiedEarningsCalendarFetcher()

    # Test fetching upcoming earnings
    print("\n📅 Fetching upcoming earnings (7 days)...")
    test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"]

    try:
        response = fetcher.get_upcoming_earnings(
            symbols=test_symbols,
            days_ahead=7,
            use_cache=False,  # Force fresh fetch for testing
        )

        print(f"✅ Fetched {len(response.events)} earnings events")

        if response.events:
            print("\nFirst 5 events:")
            for i, event in enumerate(response.events[:5], 1):
                print(
                    f"{i}. {event.symbol}: {event.earnings_date} ({event.time.value})"
                )
                print(f"   Company: {event.company_name}")
                print(f"   Days until: {event.days_until_earnings}")
                if event.eps_estimate:
                    print(f"   EPS Estimate: ${event.eps_estimate:.2f}")
        else:
            print("No events found in the next 7 days")

    except Exception as e:
        print(f"❌ Sync fetch failed: {e}")
    finally:
        fetcher.close()


async def test_async_interface():
    """Test asynchronous fetcher interface."""
    print("\n" + "=" * 60)
    print("Testing Asynchronous Interface")
    print("=" * 60)

    fetcher = UnifiedEarningsCalendarFetcher()

    print("\n📅 Fetching upcoming earnings async (14 days)...")
    test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"]

    try:
        response = await fetcher.async_get_upcoming_earnings(
            symbols=test_symbols,
            days_ahead=14,
            use_cache=True,  # Use cache if available
        )

        print(f"✅ Async fetched {len(response.events)} earnings events")

        # Test filtering
        high_priority = [e for e in response.events if e.priority_score >= 80]
        print(f"   High priority events (score >= 80): {len(high_priority)}")

        # Group by date
        events_by_date = {}
        for event in response.events:
            if event.earnings_date not in events_by_date:
                events_by_date[event.earnings_date] = []
            events_by_date[event.earnings_date].append(event)

        print(f"   Events across {len(events_by_date)} different dates")

        # Show summary by date
        for earnings_date in sorted(events_by_date.keys())[:3]:
            events = events_by_date[earnings_date]
            symbols = [e.symbol for e in events]
            print(f"   {earnings_date}: {', '.join(symbols)}")

    except Exception as e:
        print(f"❌ Async fetch failed: {e}")
    finally:
        fetcher.close()


def test_convenience_functions():
    """Test convenience functions."""
    print("\n" + "=" * 60)
    print("Testing Convenience Functions")
    print("=" * 60)

    print("\n📅 Using get_upcoming_earnings()...")

    try:
        events = get_upcoming_earnings(days_ahead=3)
        print(f"✅ Found {len(events)} events in next 3 days")

        if events:
            # Show events by time of day
            bmo = [e for e in events if e.time == EarningsTimeOfDay.BMO]
            amc = [e for e in events if e.time == EarningsTimeOfDay.AMC]
            other = [
                e
                for e in events
                if e.time not in [EarningsTimeOfDay.BMO, EarningsTimeOfDay.AMC]
            ]

            print(f"   Before Market Open: {len(bmo)} events")
            print(f"   After Market Close: {len(amc)} events")
            print(f"   Other/Unknown: {len(other)} events")

    except Exception as e:
        print(f"❌ Convenience function failed: {e}")


def test_cache_functionality():
    """Test caching functionality."""
    print("\n" + "=" * 60)
    print("Testing Cache Functionality")
    print("=" * 60)

    fetcher = UnifiedEarningsCalendarFetcher()

    print("\n📅 First fetch (no cache)...")
    import time

    start = time.time()
    response1 = fetcher.get_upcoming_earnings(days_ahead=7, use_cache=False)
    elapsed1 = time.time() - start
    print(f"✅ First fetch: {len(response1.events)} events in {elapsed1:.2f}s")

    print("\n📅 Second fetch (should use cache)...")
    start = time.time()
    response2 = fetcher.get_upcoming_earnings(days_ahead=7, use_cache=True)
    elapsed2 = time.time() - start
    print(f"✅ Second fetch: {len(response2.events)} events in {elapsed2:.2f}s")

    if elapsed2 < elapsed1 * 0.5:  # Should be at least 2x faster
        print("✅ Cache is working (second fetch was faster)")
    else:
        print("⚠️  Cache might not be working properly")

    fetcher.close()


async def test_error_handling():
    """Test error handling and validation."""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)

    # Test with invalid symbol (should be filtered out)
    print("\n📅 Testing with invalid symbols...")
    fetcher = UnifiedEarningsCalendarFetcher()

    try:
        response = await fetcher.async_get_upcoming_earnings(
            symbols=["AAPL", "!!!INVALID", "MSFT"],  # Invalid symbol should be ignored
            days_ahead=7,
        )
        print(f"✅ Handled invalid symbols gracefully: {len(response.events)} events")

        symbols_found = {e.symbol for e in response.events}
        if "!!!INVALID" not in symbols_found:
            print("✅ Invalid symbol was properly filtered out")

    except Exception as e:
        print(f"❌ Failed to handle invalid symbols: {e}")
    finally:
        fetcher.close()


async def main():
    """Run all tests."""
    setup_logging()

    print("🚀 Starting Earnings Calendar API Tests")
    print("=" * 60)

    # Run sync tests
    test_pydantic_models()
    test_sync_interface()
    test_convenience_functions()
    test_cache_functionality()

    # Run async tests
    await test_async_interface()
    await test_error_handling()

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

    # Test async convenience function
    print("\n🎯 Quick test with async convenience function:")
    events = await async_get_upcoming_earnings(days_ahead=1)
    if events:
        print(f"Tomorrow's earnings: {', '.join([e.symbol for e in events[:5]])}")
    else:
        print("No earnings tomorrow")


if __name__ == "__main__":
    asyncio.run(main())
