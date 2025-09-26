#!/usr/bin/env python3
"""
Multi-source async earnings calendar fetcher using source abstraction.
"""

from common.utils import get_logger
from datetime import date, timedelta
from earnings.models import EarningsEvent, EarningsSource
from earnings.sources import EarningsCalendarSource, SourceRegistry
from typing import Dict, List, Optional


logger = get_logger(__name__)


class AsyncEarningsCalendarFetcher:
    """Multi-source async fetcher for earnings calendars."""

    def __init__(self, sources: Optional[List[EarningsSource]] = None):
        """Initialize with specified sources or default to all supported.

        Uses SourceRegistry to construct concrete source implementations.
        """
        self._sources: Dict[EarningsSource, EarningsCalendarSource] = (
            SourceRegistry.get_instances(sources)
        )

    async def get_earnings_range(
        self,
        symbols: Optional[List[str]] = None,
        start_date: date = None,
        end_date: date = None,
    ) -> List[EarningsEvent]:
        """Fetch earnings for date range from all sources."""
        if start_date is None:
            start_date = date.today()
        if end_date is None:
            end_date = start_date + timedelta(days=7)

        all_events = []

        # Fetch from all configured sources
        import asyncio

        tasks = []
        for source_type, source in self._sources.items():
            task = source.fetch_earnings_range(start_date, end_date, symbols)
            tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    all_events.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Source fetch failed: {result}")

        return self._deduplicate_events(all_events)

    async def a_get_upcoming_earnings(
        self,
        symbols: Optional[List[str]] = None,
        days_ahead: int = 7,
        sources: Optional[List[EarningsSource]] = None,
    ) -> List[EarningsEvent]:
        """Get upcoming earnings for the next N days."""
        start_date = date.today()
        end_date = start_date + timedelta(days=days_ahead)
        if sources:
            # Build a temporary fetcher constrained to the requested sources
            temp = AsyncEarningsCalendarFetcher(sources=sources)
            return await temp.get_earnings_range(symbols, start_date, end_date)
        return await self.get_earnings_range(symbols, start_date, end_date)

    def _deduplicate_events(self, events: List[EarningsEvent]) -> List[EarningsEvent]:
        """Remove duplicate events, preferring higher quality data."""
        seen = {}
        unique = []

        for event in events:
            key = (event.symbol, event.earnings_date)

            if key not in seen:
                seen[key] = event
                unique.append(event)
            else:
                # Keep the one with more data
                existing = seen[key]

                # Prefer events with EPS estimates
                if event.eps_estimate is not None and existing.eps_estimate is None:
                    seen[key] = event
                    unique[unique.index(existing)] = event
                # Prefer events with revenue estimates
                elif (
                    event.revenue_estimate is not None
                    and existing.revenue_estimate is None
                    and event.eps_estimate == existing.eps_estimate
                ):
                    seen[key] = event
                    unique[unique.index(existing)] = event

        return unique
