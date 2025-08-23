#!/usr/bin/env python3
"""
Simple async NASDAQ earnings fetcher.
"""

import asyncio
import random
from datetime import date, timedelta
from typing import Optional, List

import httpx

from common.utils import get_logger
from earnings import EarningsEvent, EarningsSource

logger = get_logger(__name__)


class AsyncEarningsCalendarFetcher:
    """Simple async fetcher for NASDAQ earnings."""
    
    def __init__(self):
        self.base_url = "https://api.nasdaq.com/api/calendar/earnings"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.nasdaq.com/",
        }

    async def get_earnings_range(
        self, 
        symbols: Optional[List[str]] = None,
        start_date: date = None, 
        end_date: date = None
    ) -> List[EarningsEvent]:
        """Fetch earnings for date range."""
        if start_date is None:
            start_date = date.today()
        if end_date is None:
            end_date = start_date + timedelta(days=7)
            
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
            logger.error(f"Failed to fetch {day}: {e}")
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
