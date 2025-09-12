from __future__ import annotations

from anyio import to_thread
from common.reliability.logging import get_trading_logger
from earnings import EarningsEvent, EarningsSource  # Use Pydantic-based models/enums
from earnings.async_fetcher import AsyncEarningsCalendarFetcher
from typing import List, Optional


class EarningsFetcherService:
    """Stateless service that fetches earnings from external sources.

    No persistence or DB logic; upstream of the calendar repository/API.
    """

    def __init__(self) -> None:
        self._fetcher = AsyncEarningsCalendarFetcher()
        self._logger = get_trading_logger(__name__)

    async def get_upcoming(
        self,
        *,
        days_ahead: int,
        symbols: Optional[List[str]] = None,
        sources: Optional[List[EarningsSource]] = None,
    ) -> List[EarningsEvent]:
        """Fetch earnings with enhanced error handling and logging."""
        use_sources = sources or [EarningsSource.NASDAQ]
        source_names = [s.value if hasattr(s, "value") else str(s) for s in use_sources]

        self._logger.info(
            f"📊 Fetching earnings: {days_ahead} days ahead, "
            f"sources={source_names}, symbols={symbols or 'all'}"
        )

        try:
            events = await self._fetcher.a_get_upcoming_earnings(
                symbols=symbols,
                days_ahead=days_ahead,
                sources=use_sources,
            )

            self._logger.info(f"✅ Successfully fetched {len(events)} earnings events")

            # Log symbol-specific results when symbols were requested
            if symbols:
                found_symbols = list(set(e.symbol for e in events))
                missing_symbols = [s for s in symbols if s not in found_symbols]

                if found_symbols:
                    self._logger.info(f"📈 Found data for symbols: {found_symbols}")
                if missing_symbols:
                    self._logger.warning(
                        f"❌ No data found for symbols: {missing_symbols}"
                    )

            return events

        except Exception as e:
            self._logger.error(
                f"❌ Failed to fetch earnings from {source_names}: {str(e)}",
                exc_info=True,
            )
            # Re-raise the exception so the API can handle it appropriately
            raise
