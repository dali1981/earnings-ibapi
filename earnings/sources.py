#!/usr/bin/env python3
"""
Earnings calendar source abstractions and implementations.
"""

import asyncio
import random
from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import List, Optional
import httpx

from common.utils import get_logger
from earnings.models import EarningsEvent, EarningsSource
from typing import Dict, Type, Iterable

logger = get_logger(__name__)


class EarningsCalendarSource(ABC):
    """Abstract base class for earnings calendar data sources."""
    
    @abstractmethod
    async def fetch_earnings_range(
        self,
        start_date: date,
        end_date: date,
        symbols: Optional[List[str]] = None
    ) -> List[EarningsEvent]:
        """Fetch earnings events for a date range."""
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> EarningsSource:
        """Get the source identifier."""
        pass


class NasdaqEarningsSource(EarningsCalendarSource):
    """NASDAQ earnings calendar source implementation."""
    
    def __init__(self):
        self.base_url = "https://api.nasdaq.com/api/calendar/earnings"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.nasdaq.com/",
        }
    
    @property
    def source_name(self) -> EarningsSource:
        return EarningsSource.NASDAQ
    
    async def fetch_earnings_range(
        self,
        start_date: date,
        end_date: date,
        symbols: Optional[List[str]] = None
    ) -> List[EarningsEvent]:
        """Fetch earnings for date range from NASDAQ API."""
        all_events = []
        
        async with httpx.AsyncClient(
            headers=self.headers,
            timeout=30.0
        ) as client:
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:  # Skip weekends
                    events = await self._fetch_day(client, current, symbols)
                    all_events.extend(events)
                    await asyncio.sleep(0.5 + random.random() * 0.5)
                current += timedelta(days=1)
        
        return all_events
    
    async def _fetch_day(
        self,
        client: httpx.AsyncClient,
        day: date,
        symbols: Optional[List[str]]
    ) -> List[EarningsEvent]:
        """Fetch earnings for single day."""
        try:
            response = await client.get(
                self.base_url,
                params={"date": day.strftime("%Y-%m-%d")}
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_response(data, day, symbols)
        except Exception as e:
            logger.error(f"Failed to fetch NASDAQ data for {day}: {e}")
            return []
    
    def _parse_response(
        self,
        data: dict,
        day: date,
        symbols: Optional[List[str]]
    ) -> List[EarningsEvent]:
        """Parse NASDAQ response to EarningsEvent list."""
        try:
            from earnings.models import NasdaqEarningsRow
            
            rows = data.get("data", {}).get("rows", [])
            if not rows:
                return []
            
            symbol_set = {s.upper() for s in symbols} if symbols else None
            events = []
            
            for row_data in rows:
                try:
                    symbol = row_data.get("symbol", "").upper()
                    if symbol_set and symbol not in symbol_set:
                        continue
                    
                    row = NasdaqEarningsRow(**row_data)
                    event = row.to_earnings_event(day)
                    events.append(event)
                except Exception:
                    continue
            
            return events
            
        except Exception as e:
            logger.error(f"Parse failed for {day}: {e}")
            return []


class YahooEarningsSource(EarningsCalendarSource):
    """Yahoo Finance earnings calendar source implementation."""
    
    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com/v1/finance/search"
        self.calendar_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        # Base headers; per-request we add Referer/Origin
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
        }
    
    @property
    def source_name(self) -> EarningsSource:
        return EarningsSource.YAHOO
    
    async def fetch_earnings_range(
        self,
        start_date: date,
        end_date: date,
        symbols: Optional[List[str]] = None
    ) -> List[EarningsEvent]:
        """Fetch earnings for date range from Yahoo Finance.

        - If `symbols` provided, query quote summary per-symbol (narrow lookup).
        - Otherwise, use Yahoo's calendar endpoint day-by-day to fetch bulk results.
        """
        all_events: List[EarningsEvent] = []
        async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
            if symbols:
                for symbol in symbols:
                    try:
                        events = await self._fetch_symbol_earnings(client, symbol, start_date, end_date)
                        all_events.extend(events)
                        await asyncio.sleep(0.2 + random.random() * 0.3)
                    except Exception as e:
                        logger.error(f"Failed to fetch Yahoo data for {symbol}: {e}")
                        continue
            else:
                current = start_date
                while current <= end_date:
                    try:
                        # Weekends may still have listings, but keep calls reasonable
                        day_events = await self._fetch_day_bulk(client, current)
                        # Filter by date window already scoped by day
                        all_events.extend(day_events)
                        await asyncio.sleep(0.4 + random.random() * 0.4)
                    except Exception as e:
                        logger.error(f"Failed Yahoo calendar fetch for {current}: {e}")
                    current += timedelta(days=1)
        return all_events

    async def _fetch_symbol_earnings(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> List[EarningsEvent]:
        """Fetch earnings events for a specific symbol."""
        try:
            url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
            params = {
                "modules": "calendarEvents",
                "formatted": "true",
                "lang": "en-US",
                "region": "US",
                "corsDomain": "finance.yahoo.com",
            }
            # Add realistic Referer/Origin per request
            extra_headers = {
                "Referer": f"https://finance.yahoo.com/quote/{symbol}",
                "Origin": "https://finance.yahoo.com",
            }
            response = await client.get(url, params=params, headers=extra_headers)
            response.raise_for_status()
            data = response.json()
            return self._parse_yahoo_response(data, symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to fetch Yahoo earnings for {symbol}: {e}")
            return []

    async def _fetch_day_bulk(
        self,
        client: httpx.AsyncClient,
        day: date,
    ) -> List[EarningsEvent]:
        """Fetch bulk earnings for a given day using Yahoo's calendar API."""
        try:
            url = "https://query2.finance.yahoo.com/v7/finance/calendar/earnings"
            params = {"day": day.isoformat()}
            headers = {
                "Referer": f"https://finance.yahoo.com/calendar/earnings?day={day.isoformat()}",
                "Origin": "https://finance.yahoo.com",
            }
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            results = (
                data.get("finance", {})
                .get("result", [{}])[0]
                .get("calendarEvents", {})
                .get("earnings", {})
                .get("result", [])
            )
            events: List[EarningsEvent] = []
            for row in results:
                try:
                    sym = (row.get("symbol") or "").upper()
                    if not sym:
                        continue
                    company = row.get("companyshortname") or row.get("shortName") or sym
                    # Datetime can be epoch seconds in 'startdatetime' or date string in 'startdate'
                    edate: Optional[date] = None
                    ts = row.get("startdatetime") or row.get("startDate") or row.get("startdate")
                    if isinstance(ts, (int, float)):
                        from datetime import datetime
                        edate = datetime.fromtimestamp(int(ts)).date()
                    elif isinstance(ts, str) and len(ts) >= 10:
                        from datetime import datetime
                        try:
                            edate = datetime.strptime(ts[:10], "%Y-%m-%d").date()
                        except Exception:
                            edate = day
                    else:
                        edate = day
                    evt = EarningsEvent(
                        symbol=sym,
                        company_name=company,
                        earnings_date=edate,
                        source=EarningsSource.YAHOO,
                    )
                    events.append(evt)
                except Exception:
                    continue
            return events
        except Exception as e:
            logger.error(f"Parse failed for Yahoo calendar {day}: {e}")
            return []

    def _parse_yahoo_response(
        self,
        data: dict,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> List[EarningsEvent]:
        """Parse Yahoo Finance response to EarningsEvent list."""
        try:
            result = data.get("quoteSummary", {}).get("result", [])
            if not result:
                return []
            calendar_events = result[0].get("calendarEvents", {})
            earnings = calendar_events.get("earnings", {})
            events: List[EarningsEvent] = []
            earnings_date = earnings.get("earningsDate", [{}])[0]
            if earnings_date and "raw" in earnings_date:
                from datetime import datetime
                earnings_timestamp = earnings_date["raw"]
                earnings_dt = datetime.fromtimestamp(earnings_timestamp).date()
                if start_date <= earnings_dt <= end_date:
                    company_name = symbol
                    eps_estimate = None
                    revenue_estimate = None
                    eps_data = earnings.get("epsEstimate", {})
                    if eps_data and "raw" in eps_data:
                        eps_estimate = float(eps_data["raw"])
                    revenue_data = earnings.get("revenueEstimate", {})
                    if revenue_data and "raw" in revenue_data:
                        revenue_estimate = float(revenue_data["raw"])
                    event = EarningsEvent(
                        symbol=symbol.upper(),
                        company_name=company_name,
                        earnings_date=earnings_dt,
                        eps_estimate=eps_estimate,
                        revenue_estimate=revenue_estimate,
                        source=EarningsSource.YAHOO,
                    )
                    events.append(event)
            return events
        except Exception as e:
            logger.error(f"Parse failed for Yahoo {symbol}: {e}")
            return []


# -------- Source Registry --------
class SourceRegistry:
    """Factory/registry for EarningsCalendarSource implementations.

    Centralizes how concrete sources are constructed and selected so fetchers
    don't need to hard-code per-source wiring.
    """

    _factories: Dict[EarningsSource, Type[EarningsCalendarSource]] = {
        EarningsSource.NASDAQ: NasdaqEarningsSource,
        EarningsSource.YAHOO: YahooEarningsSource,
    }

    @classmethod
    def get_instances(
        cls, sources: Optional[Iterable[EarningsSource]] = None
    ) -> Dict[EarningsSource, EarningsCalendarSource]:
        """Build instances for requested sources (or all supported)."""
        selected = list(sources) if sources else list(cls._factories.keys())
        instances: Dict[EarningsSource, EarningsCalendarSource] = {}
        for s in selected:
            factory = cls._factories.get(s)
            if factory is not None:
                try:
                    instances[s] = factory()
                except Exception:
                    logger.exception(f"Failed to instantiate source: {s}")
        return instances

    @classmethod
    def supported_names(cls) -> Dict[str, EarningsSource]:
        """Lowercase name -> enum mapping for UX (e.g., from query strings)."""
        return {s.value: s for s in cls._factories.keys()}

    @classmethod
    def is_supported(cls, source: EarningsSource) -> bool:
        return source in cls._factories
