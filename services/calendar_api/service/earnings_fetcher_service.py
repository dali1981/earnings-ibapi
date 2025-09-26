from __future__ import annotations

from anyio import to_thread
from common.reliability.logging import get_trading_logger
from datetime import date, timedelta
from earnings import EarningsEvent, EarningsSource  # Use Pydantic-based models/enums
from earnings.async_fetcher import AsyncEarningsCalendarFetcher
from typing import Dict, List, Optional, Union


class FetchMetadata:
    """Metadata about a fetch operation."""
    
    def __init__(
        self,
        sources_used: List[EarningsSource],
        symbols_requested: Optional[List[str]],
        symbols_found: List[str],
        missing_symbols: List[str],
        events_count: int,
        fetch_duration_seconds: float,
    ):
        self.sources_used = sources_used
        self.symbols_requested = symbols_requested
        self.symbols_found = symbols_found
        self.missing_symbols = missing_symbols
        self.events_count = events_count
        self.fetch_duration_seconds = fetch_duration_seconds


class EarningsFetcherService:
    """Enhanced stateless service that fetches earnings from external sources.

    Responsibilities:
    - External data acquisition from multiple sources
    - Parameter validation and normalization
    - Enhanced error handling and logging
    - Fetch metadata and performance tracking
    
    No persistence or DB logic; upstream of the calendar repository/API.
    """

    def __init__(self) -> None:
        self._fetcher = AsyncEarningsCalendarFetcher()
        self._logger = get_trading_logger(__name__)

    def validate_fetch_params(
        self,
        days_ahead: int,
        symbols: Optional[List[str]] = None,
        sources: Optional[List[EarningsSource]] = None,
    ) -> None:
        """Validate fetch parameters."""
        if days_ahead < 0 or days_ahead > 365:
            raise ValueError(f"days_ahead must be between 0 and 365, got {days_ahead}")
            
        if symbols:
            if len(symbols) > 100:
                raise ValueError(f"Too many symbols requested: {len(symbols)}. Maximum is 100.")
            invalid_symbols = [s for s in symbols if not s or not s.strip()]
            if invalid_symbols:
                raise ValueError(f"Invalid symbols found: {invalid_symbols}")
                
        if sources:
            # Check if Yahoo is requested without symbols
            if EarningsSource.YAHOO in sources and not symbols:
                raise ValueError("Yahoo Finance source requires symbols parameter")

    async def get_upcoming(
        self,
        *,
        days_ahead: int,
        symbols: Optional[List[str]] = None,
        sources: Optional[List[EarningsSource]] = None,
        include_metadata: bool = False,
    ) -> Union[List[EarningsEvent], tuple[List[EarningsEvent], FetchMetadata]]:
        """Fetch earnings with enhanced error handling, validation, and optional metadata.
        
        Args:
            days_ahead: Number of days ahead to fetch (0-365)
            symbols: Optional list of symbols to filter (max 100)
            sources: Data sources to use (defaults to NASDAQ)
            include_metadata: Whether to return fetch metadata
            
        Returns:
            List of earnings events, or tuple of (events, metadata) if include_metadata=True
            
        Raises:
            ValueError: For invalid parameters
            Exception: For fetch failures
        """
        import time
        
        # Validate parameters
        self.validate_fetch_params(days_ahead, symbols, sources)
        
        # Normalize and prepare
        use_sources = sources or [EarningsSource.NASDAQ]
        normalized_symbols = [s.strip().upper() for s in symbols] if symbols else None
        source_names = [s.value if hasattr(s, "value") else str(s) for s in use_sources]

        self._logger.info(
            f"📊 Fetching earnings: {days_ahead} days ahead, "
            f"sources={source_names}, symbols={normalized_symbols or 'all'}"
        )

        start_time = time.time()
        
        try:
            events = await self._fetcher.a_get_upcoming_earnings(
                symbols=normalized_symbols,
                days_ahead=days_ahead,
                sources=use_sources,
            )

            fetch_duration = time.time() - start_time
            self._logger.info(
                f"✅ Successfully fetched {len(events)} earnings events "
                f"in {fetch_duration:.2f}s"
            )

            # Analyze results
            found_symbols = list(set(e.symbol for e in events)) if events else []
            missing_symbols = []
            
            if normalized_symbols:
                missing_symbols = [s for s in normalized_symbols if s not in found_symbols]
                
                if found_symbols:
                    self._logger.info(f"📈 Found data for symbols: {found_symbols}")
                if missing_symbols:
                    self._logger.warning(
                        f"❌ No data found for symbols: {missing_symbols}"
                    )

            # Create metadata if requested
            if include_metadata:
                metadata = FetchMetadata(
                    sources_used=use_sources,
                    symbols_requested=normalized_symbols,
                    symbols_found=found_symbols,
                    missing_symbols=missing_symbols,
                    events_count=len(events),
                    fetch_duration_seconds=fetch_duration,
                )
                return events, metadata

            return events

        except Exception as e:
            fetch_duration = time.time() - start_time
            self._logger.error(
                f"❌ Failed to fetch earnings from {source_names} after {fetch_duration:.2f}s: {str(e)}",
                exc_info=True,
            )
            # Re-raise the exception so the API can handle it appropriately
            raise
            
    async def validate_source_connectivity(self, sources: Optional[List[EarningsSource]] = None) -> Dict[EarningsSource, bool]:
        """Test connectivity to external sources.
        
        Returns:
            Dictionary mapping sources to their availability status
        """
        test_sources = sources or [EarningsSource.NASDAQ, EarningsSource.YAHOO]
        results = {}
        
        for source in test_sources:
            try:
                # Test with minimal fetch (1 day, no symbols for non-Yahoo sources)
                test_symbols = ["AAPL"] if source == EarningsSource.YAHOO else None
                await self._fetcher.a_get_upcoming_earnings(
                    symbols=test_symbols,
                    days_ahead=1,
                    sources=[source],
                )
                results[source] = True
                self._logger.info(f"✅ Source {source.value} is available")
            except Exception as e:
                results[source] = False
                self._logger.warning(f"❌ Source {source.value} is unavailable: {e}")
                
        return results
