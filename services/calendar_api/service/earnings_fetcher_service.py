from __future__ import annotations

from typing import List, Optional

from earnings.async_fetcher import AsyncEarningsCalendarFetcher
from earnings import EarningsEvent, EarningsSource  # Use Pydantic-based models/enums
from anyio import to_thread


class EarningsFetcherService:
    """Stateless service that fetches earnings from external sources.

    No persistence or DB logic; upstream of the calendar repository/API.
    """

    def __init__(self) -> None:
        self._fetcher = AsyncEarningsCalendarFetcher()

    async def get_upcoming(
        self,
        *,
        days_ahead: int,
        symbols: Optional[List[str]] = None,
        sources: Optional[List[EarningsSource]] = None,
    ) -> List[EarningsEvent]:
        # Use the async multi-source fetcher with correct enum types
        use_sources = sources or [EarningsSource.NASDAQ]
        return await self._fetcher.a_get_upcoming_earnings(
            symbols=symbols,
            days_ahead=days_ahead,
            sources=use_sources,
        )
