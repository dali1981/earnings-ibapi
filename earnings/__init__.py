"""
Earnings Calendar Module

Provides earnings data fetching and priority-based scheduling capabilities
for the trading data system.
"""

# Legacy imports (keep for backward compatibility)
from .fetcher import EarningsCalendarFetcher as LegacyEarningsCalendarFetcher

# New Pydantic-based models
from .models import (
    EarningsEvent,
    EarningsSource,
    EarningsTimeOfDay,
    EarningsCalendarResponse,
    NasdaqEarningsResponse,
    NasdaqEarningsRow,
)

# New unified fetcher with proper async support
from .unified_fetcher import (
    UnifiedEarningsCalendarFetcher,
    get_default_fetcher,
    get_upcoming_earnings,
    get_historical_earnings,
    get_earnings_range,
    async_get_upcoming_earnings,
)

# Async fetcher for advanced usage
from .async_calendar_fetcher import AsyncEarningsCalendarFetcher

# Storage
from .services.storage import EarningsPersistence

# Default fetcher alias for backward compatibility
EarningsCalendarFetcher = UnifiedEarningsCalendarFetcher

__all__ = [
    # Models
    "EarningsEvent",
    "EarningsSource",
    "EarningsTimeOfDay",
    "EarningsCalendarResponse",
    "NasdaqEarningsResponse",
    "NasdaqEarningsRow",
    # Fetchers
    "UnifiedEarningsCalendarFetcher",
    "AsyncEarningsCalendarFetcher",
    "EarningsCalendarFetcher",  # Backward compatibility
    "LegacyEarningsCalendarFetcher",
    # Convenience functions
    "get_default_fetcher",
    "get_upcoming_earnings",
    "get_historical_earnings",
    "get_earnings_range",
    "async_get_upcoming_earnings",
    # Storage
    "EarningsPersistence",
]
