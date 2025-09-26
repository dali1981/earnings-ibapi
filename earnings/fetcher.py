#!/usr/bin/env python3
"""
Earnings Calendar Data Fetcher (NASDAQ-only)

This repository is configured to fetch earnings exclusively from the
NASDAQ API for reliability and consistency. Hooks for other sources
remain in code for future use, but the default execution path only
queries NASDAQ unless specific sources are explicitly provided.

Usage:
    fetcher = EarningsCalendarFetcher()
    earnings = fetcher.get_upcoming_earnings(days_ahead=14)
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncio
import httpx
import json
import os
import pandas as pd
import sys
import time


# Add path for centralized logging
sys.path.append(str(Path(__file__).parent.parent))
from common.utils.logging_setup import get_logger


logger = get_logger(__name__)


class EarningsSource(Enum):
    """Earnings data sources."""

    FMP = "financial_modeling_prep"
    FINNHUB = "finnhub"
    NASDAQ = "nasdaq"
    YAHOO = "yahoo"


@dataclass
class EarningsEvent:
    """Single earnings event."""

    symbol: str
    company_name: str
    earnings_date: date
    time: Optional[str] = (
        None  # "bmo", "amc", "dmt" (before market, after market, during market)
    )
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    source: str = "unknown"

    @property
    def days_until_earnings(self) -> int:
        """Days until earnings (negative if in past)."""
        return (self.earnings_date - date.today()).days

    @property
    def priority_score(self) -> float:
        """Priority score for scheduling (higher = more urgent)."""
        days_until = self.days_until_earnings

        # High priority for earnings in next 7 days
        if days_until <= 0:
            return 100.0  # Today or past
        elif days_until <= 3:
            return 90.0  # Next 3 days
        elif days_until <= 7:
            return 80.0  # Next week
        elif days_until <= 14:
            return 60.0  # Next 2 weeks
        elif days_until <= 30:
            return 40.0  # Next month
        else:
            return 20.0  # Future


class EarningsCalendarFetcher:
    def _allowed_sources(self) -> list:
        """Resolve allowed sources from env var EARNINGS_ALLOWED_SOURCES.

        Defaults to NASDAQ-only. Accepts comma-separated names, e.g.
        "nasdaq,fmp,finnhub". Unknown names are ignored.
        """
        raw = os.getenv("EARNINGS_ALLOWED_SOURCES", "nasdaq")
        allowed = set(s.strip().lower() for s in raw.split(",") if s.strip())
        mapping = {
            "nasdaq": EarningsSource.NASDAQ,
            "fmp": EarningsSource.FMP,
            "finnhub": EarningsSource.FINNHUB,
            "yahoo": EarningsSource.YAHOO,
        }
        return [mapping[k] for k in mapping.keys() if k in allowed]

    """Multi-source earnings calendar data fetcher."""

    def __init__(self, cache_dir: Path = Path("cache/earnings")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # API configurations (free tiers)
        self.api_configs = {
            EarningsSource.FMP: {
                "base_url": "https://financialmodelingprep.com/api/v3",
                "rate_limit": 250,  # per day free
                "api_key": None,  # Can be set via environment or config
            },
            EarningsSource.FINNHUB: {
                "base_url": "https://finnhub.io/api/v1",
                "rate_limit": 60,  # per minute free
                "api_key": None,
            },
        }

        self._default_headers = {"User-Agent": "Trading-Data-System/1.0"}
        # Async rate limiters initialized lazily to avoid importing config at import-time
        self._rate_limits = {
            "nasdaq": float(os.getenv("RATE_LIMIT_NASDAQ", "0.3")),
            "fmp": float(os.getenv("RATE_LIMIT_FMP", "0.25")),
            "finnhub": float(os.getenv("RATE_LIMIT_FINNHUB", "1.0")),
        }
        self._rl = {
            EarningsSource.NASDAQ: _AsyncRateLimiter(self._rate_limits["nasdaq"]),
            EarningsSource.FMP: _AsyncRateLimiter(self._rate_limits["fmp"]),
            EarningsSource.FINNHUB: _AsyncRateLimiter(self._rate_limits["finnhub"]),
        }

        self._async_client: Optional[httpx.AsyncClient] = None

    async def _async_get_json(
        self,
        url: str,
        params: Dict[str, Any],
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                headers=self._default_headers, timeout=timeout
            )
        h = {**self._default_headers, **(headers or {})}
        resp = await self._async_client.get(
            url, params=params, headers=h, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()

    def _get_json(
        self,
        url: str,
        params: Dict[str, Any],
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        return asyncio.run(
            self._async_get_json(url, params, timeout=timeout, headers=headers)
        )

    def get_earnings_range(
        self,
        symbols: Optional[List[str]] = None,
        days_back: int = 0,
        days_ahead: int = 30,
        sources: List[EarningsSource] = None,
    ) -> List[EarningsEvent]:
        """
        Get earnings events for a date range including past and future dates.

        Args:
            symbols: Optional list of symbols to filter
            days_back: Number of days back to look for historical earnings
            days_ahead: Number of days ahead to look for upcoming earnings
            sources: List of data sources to use

        Returns:
            List of earnings events from (today - days_back) to (today + days_ahead)
        """

        # Default to NASDAQ-only unless explicit sources are provided
        if sources is None:
            sources = [EarningsSource.NASDAQ]

        # Enforce allowed sources via env var
        allowed = set(self._allowed_sources())
        filtered = [s for s in sources if s in allowed]
        disallowed = [s for s in sources if s not in allowed]
        if disallowed:
            logger.warning(
                "Skipping disallowed sources: %s (allowed: %s)",
                [s.value for s in disallowed],
                [s.value for s in allowed],
            )
        sources = filtered or [EarningsSource.NASDAQ]

        all_earnings = []
        successful_sources = []

        # Calculate date range
        start_date = date.today() - timedelta(days=days_back)
        end_date = date.today() + timedelta(days=days_ahead)
        total_days = days_back + days_ahead + 1  # Include today

        logger.info(
            f"📅 Fetching earnings from {start_date} to {end_date} ({total_days} days)"
        )

        # Try each source until we get data
        for source in sources:
            try:
                if source == EarningsSource.FMP:
                    earnings = self._fetch_from_fmp_range(symbols, start_date, end_date)
                elif source == EarningsSource.FINNHUB:
                    earnings = self._fetch_from_finnhub_range(
                        symbols, start_date, end_date
                    )
                elif source == EarningsSource.NASDAQ:
                    earnings = self._fetch_from_nasdaq_range(
                        symbols, start_date, end_date
                    )
                elif source == EarningsSource.YAHOO:
                    earnings = self._fetch_from_yahoo(
                        symbols, days_ahead
                    )  # Yahoo only supports forward
                else:
                    continue

                if earnings:
                    all_earnings.extend(earnings)
                    successful_sources.append(source.value)
                    logger.info(f"✅ {source.value}: {len(earnings)} events")
                else:
                    logger.warning(f"⚠️ {source.value}: No data returned")

            except Exception as e:
                logger.error(f"❌ {source.value} failed: {e}")
                continue

        # Deduplicate and sort
        unique_earnings = self._deduplicate_earnings(all_earnings)
        unique_earnings.sort(key=lambda x: (x.earnings_date, x.symbol))

        logger.info(f"📊 Total unique earnings events: {len(unique_earnings)}")
        logger.info(f"🎯 Sources used: {successful_sources}")

        return unique_earnings

    def get_upcoming_earnings(
        self,
        symbols: Optional[List[str]] = None,
        days_ahead: int = 30,
        sources: List[EarningsSource] = None,
    ) -> List[EarningsEvent]:
        """Get upcoming earnings events with multi-source fallback (forward-looking only)."""

        return self.get_earnings_range(
            symbols=symbols,
            days_back=0,
            days_ahead=days_ahead,
            sources=sources or [EarningsSource.NASDAQ],
        )

    def get_historical_earnings(
        self,
        symbols: Optional[List[str]] = None,
        days_back: int = 30,
        sources: List[EarningsSource] = None,
    ) -> List[EarningsEvent]:
        """Get historical earnings events (backward-looking only)."""

        return self.get_earnings_range(
            symbols=symbols,
            days_back=days_back,
            days_ahead=0,
            sources=sources or [EarningsSource.NASDAQ],
        )

    def _get_upcoming_earnings_legacy(
        self,
        symbols: Optional[List[str]] = None,
        days_ahead: int = 30,
        sources: List[EarningsSource] = None,
    ) -> List[EarningsEvent]:
        """Legacy method - use get_earnings_range instead."""

        if sources is None:
            sources = [
                EarningsSource.FMP,
                EarningsSource.FINNHUB,
                EarningsSource.NASDAQ,
            ]

        all_earnings = []
        successful_sources = []

        for source in sources:
            try:
                logger.info(f"Fetching earnings from {source.value}")
                earnings = self._fetch_from_source(source, symbols, days_ahead)

                if earnings:
                    all_earnings.extend(earnings)
                    successful_sources.append(source)
                    logger.info(f"✅ {source.value}: {len(earnings)} events")
                else:
                    logger.warning(f"⚠️ {source.value}: No data returned")

            except Exception as e:
                logger.error(f"❌ {source.value} failed: {e}")
                continue

        if not all_earnings:
            logger.error("All earnings sources failed")
            return []

        # Deduplicate by symbol and date
        unique_earnings = self._deduplicate_earnings(all_earnings)

        # Sort by priority (most urgent first)
        unique_earnings.sort(key=lambda x: -x.priority_score)

        logger.info(f"📊 Total unique earnings events: {len(unique_earnings)}")
        logger.info(f"🎯 Sources used: {[s.value for s in successful_sources]}")

        return unique_earnings

    def _fetch_from_source(
        self, source: EarningsSource, symbols: Optional[List[str]], days_ahead: int
    ) -> List[EarningsEvent]:
        """Fetch from specific source."""

        if source == EarningsSource.FMP:
            return self._fetch_from_fmp(symbols, days_ahead)
        elif source == EarningsSource.FINNHUB:
            return self._fetch_from_finnhub(symbols, days_ahead)
        elif source == EarningsSource.NASDAQ:
            return self._fetch_from_nasdaq(symbols, days_ahead)
        elif source == EarningsSource.YAHOO:
            return self._fetch_from_yahoo(symbols, days_ahead)
        else:
            raise ValueError(f"Unknown source: {source}")

    def _fetch_from_fmp(
        self, symbols: Optional[List[str]], days_ahead: int
    ) -> List[EarningsEvent]:
        """Fetch from Financial Modeling Prep."""
        earnings = []

        # Date range
        start_date = date.today()
        end_date = start_date + timedelta(days=days_ahead)

        try:
            # FMP earnings calendar endpoint
            url = f"{self.api_configs[EarningsSource.FMP]['base_url']}/earning_calendar"
            params = {"from": start_date.isoformat(), "to": end_date.isoformat()}

            # Add API key if available
            api_key = self.api_configs[EarningsSource.FMP].get("api_key")
            if api_key:
                params["apikey"] = api_key

            data = self._get_json(url, params, timeout=30)

            for event in data:
                # Filter by symbols if specified
                symbol = event.get("symbol", "").upper()
                if symbols and symbol not in symbols:
                    continue

                # Parse date
                date_str = event.get("date")
                if not date_str:
                    continue

                earnings_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                earnings_event = EarningsEvent(
                    symbol=symbol,
                    company_name=event.get("name", symbol),
                    earnings_date=earnings_date,
                    time=event.get("time", "").lower(),
                    eps_estimate=self._safe_float(event.get("epsEstimated")),
                    eps_actual=self._safe_float(event.get("eps")),
                    revenue_estimate=self._safe_float(event.get("revenueEstimated")),
                    revenue_actual=self._safe_float(event.get("revenue")),
                    source=EarningsSource.FMP.value,
                )
                earnings.append(earnings_event)

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"FMP API request failed: {e}")
        except (ValueError, KeyError) as e:
            logger.error(f"FMP data parsing failed: {e}")

        return earnings

    def _fetch_from_fmp_range(
        self, symbols: Optional[List[str]], start_date: date, end_date: date
    ) -> List[EarningsEvent]:
        """Fetch from Financial Modeling Prep for a date range."""
        earnings = []

        try:
            # FMP earnings calendar endpoint
            url = f"{self.api_configs[EarningsSource.FMP]['base_url']}/earning_calendar"
            params = {
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "apikey": self.api_configs[EarningsSource.FMP]["api_key"],
            }

            data = self._get_json(url, params, timeout=30)

            for event in data:
                symbol = event.get("symbol", "").upper()

                # Filter by symbols if specified
                if symbols and symbol not in symbols:
                    continue

                earnings_date_str = event.get("date")
                if not earnings_date_str:
                    continue

                earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()

                earnings.append(
                    EarningsEvent(
                        symbol=symbol,
                        company_name=event.get(
                            "eps", ""
                        ),  # FMP may not have company name
                        earnings_date=earnings_date,
                        time=event.get("time", "unknown"),
                        eps_estimate=self._safe_float(event.get("epsEstimated")),
                        eps_actual=self._safe_float(event.get("eps")),
                        revenue_estimate=self._safe_float(
                            event.get("revenueEstimated")
                        ),
                        revenue_actual=self._safe_float(event.get("revenue")),
                        source="financial_modeling_prep",
                    )
                )

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"FMP range API request failed: {e}")
        except (ValueError, KeyError) as e:
            logger.error(f"FMP range data parsing failed: {e}")

        return earnings

    def _fetch_from_finnhub(
        self, symbols: Optional[List[str]], days_ahead: int
    ) -> List[EarningsEvent]:
        """Fetch from Finnhub."""
        earnings = []

        # Date range
        start_date = date.today()
        end_date = start_date + timedelta(days=days_ahead)

        try:
            # Finnhub earnings calendar endpoint
            url = f"{self.api_configs[EarningsSource.FINNHUB]['base_url']}/calendar/earnings"
            params = {"from": start_date.isoformat(), "to": end_date.isoformat()}

            # Add API key if available
            api_key = self.api_configs[EarningsSource.FINNHUB].get("api_key")
            if api_key:
                params["token"] = api_key

            data = self._get_json(url, params, timeout=30)
            earnings_list = data.get("earningsCalendar", [])

            for event in earnings_list:
                # Filter by symbols if specified
                symbol = event.get("symbol", "").upper()
                if symbols and symbol not in symbols:
                    continue

                # Parse date
                date_str = event.get("date")
                if not date_str:
                    continue

                earnings_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                earnings_event = EarningsEvent(
                    symbol=symbol,
                    company_name=event.get("name", symbol),
                    earnings_date=earnings_date,
                    time=event.get("hour", "").lower(),
                    eps_estimate=self._safe_float(event.get("epsEstimate")),
                    eps_actual=self._safe_float(event.get("epsActual")),
                    revenue_estimate=self._safe_float(event.get("revenueEstimate")),
                    revenue_actual=self._safe_float(event.get("revenueActual")),
                    source=EarningsSource.FINNHUB.value,
                )
                earnings.append(earnings_event)

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"Finnhub API request failed: {e}")
        except (ValueError, KeyError) as e:
            logger.error(f"Finnhub data parsing failed: {e}")

        return earnings

    def _fetch_from_finnhub_range(
        self, symbols: Optional[List[str]], start_date: date, end_date: date
    ) -> List[EarningsEvent]:
        """Fetch from Finnhub for a date range."""
        earnings = []

        try:
            # Finnhub earnings calendar endpoint
            url = f"{self.api_configs[EarningsSource.FINNHUB]['base_url']}/calendar/earnings"
            params = {
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "token": self.api_configs[EarningsSource.FINNHUB]["api_key"],
            }

            data = self._get_json(url, params, timeout=30)

            # Parse earnings events
            if "earningsCalendar" in data:
                for event in data["earningsCalendar"]:
                    symbol = event.get("symbol", "").upper()

                    # Filter by symbols if specified
                    if symbols and symbol not in symbols:
                        continue

                    earnings_date_str = event.get("date")
                    if not earnings_date_str:
                        continue

                    earnings_date = datetime.strptime(
                        earnings_date_str, "%Y-%m-%d"
                    ).date()

                    earnings_event = EarningsEvent(
                        symbol=symbol,
                        company_name=symbol,  # Finnhub may not provide company name
                        earnings_date=earnings_date,
                        time="unknown",  # Finnhub may not specify time
                        eps_estimate=self._safe_float(event.get("epsEstimate")),
                        eps_actual=self._safe_float(event.get("epsActual")),
                        revenue_estimate=self._safe_float(event.get("revenueEstimate")),
                        revenue_actual=self._safe_float(event.get("revenueActual")),
                        source="finnhub",
                    )

                    earnings.append(earnings_event)

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"Finnhub range API request failed: {e}")
        except (ValueError, KeyError) as e:
            logger.error(f"Finnhub range data parsing failed: {e}")

        return earnings

    def _fetch_from_nasdaq_range(
        self, symbols: Optional[List[str]], start_date: date, end_date: date
    ) -> List[EarningsEvent]:
        """Fetch from NASDAQ public API for a date range (supports historical data)."""
        earnings = []

        logger.info(f"📅 Fetching NASDAQ earnings from {start_date} to {end_date}")

        current_date = start_date
        while current_date <= end_date:
            # Skip weekends (NASDAQ typically has no earnings data)
            if current_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                current_date += timedelta(days=1)
                continue

            try:
                daily_earnings = self._fetch_nasdaq_single_day(current_date, symbols)
                if daily_earnings:
                    earnings.extend(daily_earnings)
                    logger.debug(
                        f"✅ NASDAQ {current_date}: {len(daily_earnings)} earnings"
                    )

                # Be respectful to NASDAQ servers - small delay
                import time

                time.sleep(1.0)

            except Exception as e:
                logger.error(f"NASDAQ single day fetch failed for {current_date}: {e}")

            current_date += timedelta(days=1)

        logger.info(f"📊 NASDAQ total earnings collected: {len(earnings)}")
        return earnings

    def _fetch_from_nasdaq(
        self, symbols: Optional[List[str]], days_ahead: int
    ) -> List[EarningsEvent]:
        """Fetch from NASDAQ public API with enhanced multi-day support (forward-looking only)."""
        start_date = date.today()
        end_date = start_date + timedelta(days=days_ahead)
        return self._fetch_from_nasdaq_range(symbols, start_date, end_date)

    def _fetch_nasdaq_single_day(
        self, target_date: date, symbols: Optional[List[str]]
    ) -> List[EarningsEvent]:
        """Fetch NASDAQ earnings for a single day with pagination support."""

        url = "https://api.nasdaq.com/api/calendar/earnings"
        base_params = {"date": target_date.strftime("%Y-%m-%d")}

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nasdaq.com/",
            "Origin": "https://www.nasdaq.com",
        }

        all_earnings = []
        page = 0
        max_pages = 10  # Safety limit to prevent infinite loops
        page_size = 100  # Try different page sizes

        logger.debug(f"Fetching NASDAQ earnings for {target_date} with pagination awareness")

        try:
            # First, try to get data without pagination parameters
            data = self._get_json(url, base_params, timeout=30, headers=headers)

            if "data" in data and data["data"] and "rows" in data["data"]:
                rows = data["data"]["rows"]
                initial_count = len(rows)
                logger.debug(f"Initial NASDAQ fetch returned {initial_count} rows for {target_date}")

                # Process first page of results
                page_earnings = self._process_nasdaq_rows(rows, target_date, symbols)
                all_earnings.extend(page_earnings)

                # Check for pagination indicators in response
                has_more_data = self._detect_nasdaq_pagination(data, initial_count)

                if has_more_data:
                    logger.info(f"Detected potential pagination for NASDAQ {target_date}, attempting to fetch additional pages")

                    # Try different pagination strategies
                    pagination_strategies = [
                        {"offset": initial_count, "limit": page_size},
                        {"page": 1, "pageSize": page_size},
                        {"start": initial_count, "rows": page_size},
                        {"skip": initial_count, "take": page_size}
                    ]

                    for strategy in pagination_strategies:
                        logger.debug(f"Trying pagination strategy: {strategy}")
                        additional_earnings = self._try_nasdaq_pagination_strategy(
                            url, base_params, headers, strategy, target_date, symbols, max_pages
                        )
                        if additional_earnings:
                            all_earnings.extend(additional_earnings)
                            logger.info(f"Successfully fetched {len(additional_earnings)} additional earnings using {strategy}")
                            break  # Use first successful strategy
                    else:
                        logger.warning(f"No pagination strategy worked for NASDAQ {target_date}")

            logger.info(f"Total NASDAQ earnings fetched for {target_date}: {len(all_earnings)}")
            return all_earnings

        except Exception as e:
            logger.error(f"NASDAQ single day fetch failed for {target_date}: {e}")
            return []

    def _detect_nasdaq_pagination(self, data: dict, row_count: int) -> bool:
        """Detect if NASDAQ response might have more pages available."""
        # Check for pagination metadata in response
        pagination_indicators = [
            "totalRecords", "totalRows", "total", "hasMore", "nextPage",
            "pagination", "pageInfo", "more", "hasNextPage"
        ]

        # Check top level and data level for pagination indicators
        for indicator in pagination_indicators:
            if indicator in data or (data.get("data") and indicator in data["data"]):
                logger.debug(f"Found pagination indicator: {indicator}")
                return True

        # Heuristic: if we got exactly 25, 50, 100, or 250 results (common page sizes)
        # there might be more pages
        common_page_sizes = [25, 50, 100, 250]
        if row_count in common_page_sizes:
            logger.debug(f"Row count ({row_count}) matches common page size, suggesting pagination")
            return True

        return False

    def _try_nasdaq_pagination_strategy(
        self, url: str, base_params: dict, headers: dict,
        strategy: dict, target_date: date, symbols: Optional[List[str]],
        max_pages: int
    ) -> List[EarningsEvent]:
        """Try a specific pagination strategy for NASDAQ API."""
        additional_earnings = []

        for page_num in range(1, max_pages + 1):
            try:
                # Build pagination parameters
                params = base_params.copy()
                if "offset" in strategy:
                    params["offset"] = strategy["offset"] + (page_num - 1) * strategy["limit"]
                    params["limit"] = strategy["limit"]
                elif "page" in strategy:
                    params["page"] = page_num
                    params["pageSize"] = strategy["pageSize"]
                elif "start" in strategy:
                    params["start"] = strategy["start"] + (page_num - 1) * strategy["rows"]
                    params["rows"] = strategy["rows"]
                elif "skip" in strategy:
                    params["skip"] = strategy["skip"] + (page_num - 1) * strategy["take"]
                    params["take"] = strategy["take"]

                logger.debug(f"Trying page {page_num} with params: {params}")
                data = self._get_json(url, params, timeout=30, headers=headers)

                if "data" in data and data["data"] and "rows" in data["data"]:
                    rows = data["data"]["rows"]
                    if not rows:  # No more data
                        break

                    page_earnings = self._process_nasdaq_rows(rows, target_date, symbols)
                    additional_earnings.extend(page_earnings)
                    logger.debug(f"Page {page_num} returned {len(page_earnings)} earnings")

                    # If we got fewer results than expected, we've reached the end
                    expected_size = strategy.get("limit") or strategy.get("pageSize") or strategy.get("rows") or strategy.get("take")
                    if len(rows) < expected_size:
                        break
                else:
                    break  # Invalid response, stop pagination

            except Exception as e:
                logger.debug(f"Pagination page {page_num} failed: {e}")
                break

        return additional_earnings

    def _process_nasdaq_rows(
        self, rows: list, target_date: date, symbols: Optional[List[str]]
    ) -> List[EarningsEvent]:
        """Process NASDAQ API rows into EarningsEvent objects."""
        earnings = []

        for row in rows:
            symbol = row.get("symbol", "").strip().upper()

            # Filter by symbols if specified
            if symbols and symbol not in symbols:
                continue

            # Skip invalid symbols
            if (
                not symbol
                or not symbol.replace(".", "").replace("-", "").isalnum()
            ):
                continue

            # Parse earnings time
            time_str = row.get("time", "").lower()
            if "pre-market" in time_str or "pre" in time_str:
                timing = "bmo"  # Before market open
            elif "after" in time_str or "post" in time_str:
                timing = "amc"  # After market close
            else:
                timing = "dmt"  # During market hours

            # Parse EPS forecast
            eps_forecast = self._safe_float(row.get("epsForecast"))

            # Parse market cap for filtering
            market_cap_str = row.get("marketCap", "")
            market_cap = self._parse_market_cap(market_cap_str)

            earnings_event = EarningsEvent(
                symbol=symbol,
                company_name=row.get("name", symbol),
                earnings_date=target_date,
                time=timing,
                eps_estimate=eps_forecast,
                source=EarningsSource.NASDAQ.value,
            )

            # Add market cap as custom attribute for filtering
            earnings_event.market_cap = market_cap

            earnings.append(earnings_event)

        return earnings

    def _parse_market_cap(self, market_cap_str: str) -> Optional[float]:
        """Parse market cap string to float value."""
        if not market_cap_str or market_cap_str in ["N/A", ""]:
            return None

        try:
            # Remove currency symbols and commas
            clean_str = market_cap_str.replace("$", "").replace(",", "").strip()

            if "B" in clean_str or "b" in clean_str:
                # Billion
                number = float(clean_str.replace("B", "").replace("b", ""))
                return number * 1e9
            elif "M" in clean_str or "m" in clean_str:
                # Million
                number = float(clean_str.replace("M", "").replace("m", ""))
                return number * 1e6
            elif "K" in clean_str or "k" in clean_str:
                # Thousand
                number = float(clean_str.replace("K", "").replace("k", ""))
                return number * 1e3
            else:
                # Raw number
                return float(clean_str)
        except (ValueError, AttributeError):
            return None

    def _fetch_from_yahoo(
        self, symbols: Optional[List[str]], days_ahead: int
    ) -> List[EarningsEvent]:
        """Emergency fallback: Yahoo Finance scraping."""
        earnings = []

        try:
            # Yahoo Finance earnings calendar URL
            url = "https://finance.yahoo.com/calendar/earnings"
            params = {"size": 100}

            headers = {"User-Agent": "Mozilla/5.0 (compatible; Trading-System/1.0)"}

            response = self.session.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            # Simple scraping approach (very basic)
            # In practice, you'd parse the HTML or look for JSON data in the page
            # This is just a placeholder for emergency fallback
            logger.warning(
                "Yahoo Finance scraping not fully implemented - emergency fallback"
            )

        except Exception as e:
            logger.error(f"Yahoo Finance scraping failed: {e}")

        return earnings

    def _deduplicate_earnings(
        self, earnings: List[EarningsEvent]
    ) -> List[EarningsEvent]:
        """Remove duplicate earnings events."""
        seen = {}
        unique = []

        for event in earnings:
            key = (event.symbol, event.earnings_date)

            if key not in seen:
                seen[key] = event
                unique.append(event)
            else:
                # Keep the one with more data
                existing = seen[key]
                if (
                    event.eps_estimate is not None and existing.eps_estimate is None
                ) or (
                    event.revenue_estimate is not None
                    and existing.revenue_estimate is None
                ):
                    seen[key] = event
                    unique[unique.index(existing)] = event

        return unique

    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert to float."""
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def save_to_cache(self, earnings: List[EarningsEvent], cache_key: str = None):
        """Save earnings data to cache."""
        if not cache_key:
            cache_key = f"earnings_{date.today().isoformat()}.json"

        cache_file = self.cache_dir / cache_key

        # Convert to serializable format
        data = {
            "timestamp": datetime.now().isoformat(),
            "earnings": [
                {
                    "symbol": e.symbol,
                    "company_name": e.company_name,
                    "earnings_date": e.earnings_date.isoformat(),
                    "time": e.time,
                    "eps_estimate": e.eps_estimate,
                    "eps_actual": e.eps_actual,
                    "revenue_estimate": e.revenue_estimate,
                    "revenue_actual": e.revenue_actual,
                    "source": e.source,
                    "priority_score": e.priority_score,
                }
                for e in earnings
            ],
        }

        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"💾 Cached {len(earnings)} earnings events to {cache_file}")

    def load_from_cache(
        self, cache_key: str = None, max_age_hours: int = 6
    ) -> Optional[List[EarningsEvent]]:
        """Load earnings data from cache if fresh enough."""
        if not cache_key:
            cache_key = f"earnings_{date.today().isoformat()}.json"

        cache_file = self.cache_dir / cache_key

        if not cache_file.exists():
            return None

        # Check if cache is too old
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if cache_age.total_seconds() > max_age_hours * 3600:
            logger.info(f"Cache too old ({cache_age}), refreshing")
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            earnings = []
            for item in data["earnings"]:
                earnings_event = EarningsEvent(
                    symbol=item["symbol"],
                    company_name=item["company_name"],
                    earnings_date=datetime.strptime(
                        item["earnings_date"], "%Y-%m-%d"
                    ).date(),
                    time=item.get("time"),
                    eps_estimate=item.get("eps_estimate"),
                    eps_actual=item.get("eps_actual"),
                    revenue_estimate=item.get("revenue_estimate"),
                    revenue_actual=item.get("revenue_actual"),
                    source=item["source"],
                )
                earnings.append(earnings_event)

            logger.info(f"📂 Loaded {len(earnings)} earnings events from cache")
            return earnings

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None


def main():
    """Test the earnings fetcher."""
    from common.utils.logging_setup import setup_logging

    setup_logging()

    fetcher = EarningsCalendarFetcher()

    print("🔍 Testing earnings calendar fetcher...")

    # Test with some popular symbols
    test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

    earnings = fetcher.get_upcoming_earnings(symbols=test_symbols, days_ahead=14)

    if earnings:
        print(f"\n📊 Found {len(earnings)} upcoming earnings events:")

        for event in earnings[:10]:  # Show first 10
            priority_emoji = "🔥" if event.priority_score >= 80 else "📅"
            print(
                f"{priority_emoji} {event.symbol}: {event.earnings_date} "
                f"({event.days_until_earnings} days) - Priority: {event.priority_score:.0f}"
            )

        # Save to cache for testing
        fetcher.save_to_cache(earnings)

    else:
        print("❌ No earnings data found")


if __name__ == "__main__":
    main()


# Simple async rate limiter enforcing a minimum interval between calls
class _AsyncRateLimiter:
    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval = max(0.0, float(min_interval_seconds))
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            wait = self.min_interval - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()
