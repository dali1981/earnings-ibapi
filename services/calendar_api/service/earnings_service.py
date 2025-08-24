from __future__ import annotations

from typing import List, Optional

from earnings.fetcher import EarningsCalendarFetcher, EarningsEvent, EarningsSource


class EarningsService:
    def __init__(self) -> None:
        self._fetcher = EarningsCalendarFetcher()

    async def get_upcoming(self, *, days_ahead: int, symbols: Optional[List[str]] = None) -> List[EarningsEvent]:
        return await self._fetcher.async_get_upcoming_earnings(symbols=symbols, days_ahead=days_ahead, sources=[EarningsSource.NASDAQ])

