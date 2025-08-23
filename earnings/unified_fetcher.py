#!/usr/bin/env python3
"""
Unified earnings calendar fetcher with both sync and async support.
Uses Pydantic models for validation and proper client management.
"""

import asyncio
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, List, Union
from contextlib import contextmanager

import httpx

from common.utils.logging_setup import get_logger
from earnings.models import (
    EarningsEvent,
    EarningsSource,
    EarningsCalendarResponse,
)
from earnings.async_calendar_fetcher import AsyncEarningsCalendarFetcher

logger = get_logger(__name__)


class UnifiedEarningsCalendarFetcher:
    """
    Unified fetcher that provides both sync and async interfaces.
    Properly manages HTTP clients to avoid creating them in loops.
    """
    
    def __init__(self, cache_dir: Path = Path("cache/earnings")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Async fetcher for async operations
        self._async_fetcher = AsyncEarningsCalendarFetcher(cache_dir)
        
        # Sync client (created on demand)
        self._sync_client: Optional[httpx.Client] = None
        
        # Headers for sync operations
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nasdaq.com/",
            "Origin": "https://www.nasdaq.com",
        }
    
    @contextmanager
    def _get_sync_client(self):
        """Context manager for sync HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                headers=self._headers,
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        try:
            yield self._sync_client
        finally:
            # Keep client alive for reuse
            pass
    
    def close(self):
        """Close the sync client if it exists."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
    
    # ========== Sync Methods ==========
    
    def get_earnings_range(
        self,
        symbols: Optional[List[str]] = None,
        days_back: int = 0,
        days_ahead: int = 30,
        use_cache: bool = True,
        cache_hours: int = 6,
    ) -> EarningsCalendarResponse:
        """
        Synchronous version of get_earnings_range.
        """
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, can't use asyncio.run()
            import concurrent.futures
            import threading
            
            result = None
            exception = None
            
            def run_in_thread():
                nonlocal result, exception
                try:
                    # Create a new event loop in this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(
                            self.async_get_earnings_range(
                                symbols=symbols,
                                days_back=days_back,
                                days_ahead=days_ahead,
                                use_cache=use_cache,
                                cache_hours=cache_hours,
                            )
                        )
                    finally:
                        new_loop.close()
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception:
                raise exception
            return result
            
        except RuntimeError:
            # No event loop running, we can use asyncio.run()
            return asyncio.run(self.async_get_earnings_range(
                symbols=symbols,
                days_back=days_back,
                days_ahead=days_ahead,
                use_cache=use_cache,
                cache_hours=cache_hours,
            ))
    
    def get_upcoming_earnings(
        self,
        symbols: Optional[List[str]] = None,
        days_ahead: int = 30,
        **kwargs
    ) -> EarningsCalendarResponse:
        """Synchronous version of get_upcoming_earnings."""
        return self.get_earnings_range(
            symbols=symbols,
            days_back=0,
            days_ahead=days_ahead,
            **kwargs
        )
    
    def get_historical_earnings(
        self,
        symbols: Optional[List[str]] = None,
        days_back: int = 30,
        **kwargs
    ) -> EarningsCalendarResponse:
        """Synchronous version of get_historical_earnings."""
        return self.get_earnings_range(
            symbols=symbols,
            days_back=days_back,
            days_ahead=0,
            **kwargs
        )
    
    # ========== Async Methods ==========
    
    async def async_get_earnings_range(
        self,
        symbols: Optional[List[str]] = None,
        days_back: int = 0,
        days_ahead: int = 30,
        use_cache: bool = True,
        cache_hours: int = 6,
    ) -> EarningsCalendarResponse:
        """
        Asynchronous version of get_earnings_range.
        """
        async with self._async_fetcher as fetcher:
            return await fetcher.get_earnings_range(
                symbols=symbols,
                days_back=days_back,
                days_ahead=days_ahead,
                use_cache=use_cache,
                cache_hours=cache_hours,
            )
    
    async def async_get_upcoming_earnings(
        self,
        symbols: Optional[List[str]] = None,
        days_ahead: int = 30,
        **kwargs
    ) -> EarningsCalendarResponse:
        """Asynchronous version of get_upcoming_earnings."""
        return await self.async_get_earnings_range(
            symbols=symbols,
            days_back=0,
            days_ahead=days_ahead,
            **kwargs
        )
    
    async def async_get_historical_earnings(
        self,
        symbols: Optional[List[str]] = None,
        days_back: int = 30,
        **kwargs
    ) -> EarningsCalendarResponse:
        """Asynchronous version of get_historical_earnings."""
        return await self.async_get_earnings_range(
            symbols=symbols,
            days_back=days_back,
            days_ahead=0,
            **kwargs
        )
    
    # ========== Compatibility Methods ==========
    
    def get_events_as_list(
        self,
        symbols: Optional[List[str]] = None,
        days_back: int = 0,
        days_ahead: int = 30,
    ) -> List[EarningsEvent]:
        """
        Get earnings events as a simple list (for backward compatibility).
        """
        response = self.get_earnings_range(
            symbols=symbols,
            days_back=days_back,
            days_ahead=days_ahead,
        )
        return response.events
    
    async def async_get_events_as_list(
        self,
        symbols: Optional[List[str]] = None,
        days_back: int = 0,
        days_ahead: int = 30,
    ) -> List[EarningsEvent]:
        """
        Async version to get earnings events as a simple list.
        """
        response = await self.async_get_earnings_range(
            symbols=symbols,
            days_back=days_back,
            days_ahead=days_ahead,
        )
        return response.events


# Singleton instance for convenience
_default_fetcher: Optional[UnifiedEarningsCalendarFetcher] = None


def get_default_fetcher() -> UnifiedEarningsCalendarFetcher:
    """Get or create the default fetcher instance."""
    global _default_fetcher
    if _default_fetcher is None:
        _default_fetcher = UnifiedEarningsCalendarFetcher()
    return _default_fetcher


# ========== Convenience Functions ==========

def get_upcoming_earnings(
    symbols: Optional[List[str]] = None,
    days_ahead: int = 30,
) -> List[EarningsEvent]:
    """
    Convenience function to get upcoming earnings.
    Returns a list of EarningsEvent objects.
    """
    fetcher = get_default_fetcher()
    response = fetcher.get_upcoming_earnings(symbols=symbols, days_ahead=days_ahead)
    return response.events


def get_historical_earnings(
    symbols: Optional[List[str]] = None,
    days_back: int = 30,
) -> List[EarningsEvent]:
    """
    Convenience function to get historical earnings.
    Returns a list of EarningsEvent objects.
    """
    fetcher = get_default_fetcher()
    response = fetcher.get_historical_earnings(symbols=symbols, days_back=days_back)
    return response.events


def get_earnings_range(
    symbols: Optional[List[str]] = None,
    days_back: int = 0,
    days_ahead: int = 30,
) -> List[EarningsEvent]:
    """
    Convenience function to get earnings in a date range.
    Returns a list of EarningsEvent objects.
    """
    fetcher = get_default_fetcher()
    response = fetcher.get_earnings_range(
        symbols=symbols,
        days_back=days_back,
        days_ahead=days_ahead
    )
    return response.events


async def async_get_upcoming_earnings(
    symbols: Optional[List[str]] = None,
    days_ahead: int = 30,
) -> List[EarningsEvent]:
    """
    Async convenience function to get upcoming earnings.
    """
    fetcher = get_default_fetcher()
    response = await fetcher.async_get_upcoming_earnings(
        symbols=symbols,
        days_ahead=days_ahead
    )
    return response.events


def main():
    """Test both sync and async interfaces."""
    from common.utils.logging_setup import setup_logging
    
    setup_logging()
    
    print("=" * 60)
    print("Testing Unified Earnings Calendar Fetcher")
    print("=" * 60)
    
    test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    
    # Test sync interface
    print("\n🔍 Testing SYNC interface...")
    fetcher = UnifiedEarningsCalendarFetcher()
    
    response = fetcher.get_upcoming_earnings(
        symbols=test_symbols,
        days_ahead=14
    )
    
    print(f"📊 Found {len(response.events)} earnings events")
    
    if response.events:
        # Show first 5 events
        for event in response.events[:5]:
            print(f"  • {event.symbol} on {event.earnings_date} ({event.time.value})")
    
    # Test convenience functions
    print("\n🔍 Testing convenience functions...")
    events = get_upcoming_earnings(symbols=test_symbols, days_ahead=7)
    print(f"📊 Convenience function returned {len(events)} events")
    
    # Clean up
    fetcher.close()
    
    # Test async interface
    async def test_async():
        print("\n🔍 Testing ASYNC interface...")
        fetcher = UnifiedEarningsCalendarFetcher()
        
        response = await fetcher.async_get_upcoming_earnings(
            symbols=test_symbols,
            days_ahead=14
        )
        
        print(f"📊 Found {len(response.events)} earnings events")
        
        if response.events:
            # Show first 5 events
            for event in response.events[:5]:
                print(f"  • {event.symbol} on {event.earnings_date} ({event.time.value})")
        
        # Test async convenience function
        events = await async_get_upcoming_earnings(symbols=test_symbols, days_ahead=7)
        print(f"📊 Async convenience function returned {len(events)} events")
        
        fetcher.close()
    
    # Run async test
    asyncio.run(test_async())
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()