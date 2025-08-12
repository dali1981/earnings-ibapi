#!/usr/bin/env python3
"""
Earnings Calendar Data Fetcher

Multi-source earnings data fetcher with fallback mechanisms for robust
earnings calendar data acquisition from free APIs.

Supported sources:
1. Financial Modeling Prep (FMP) - Primary
2. Finnhub - Secondary  
3. NASDAQ API - Fallback
4. Yahoo Finance scraping - Emergency fallback

Usage:
    fetcher = EarningsCalendarFetcher()
    earnings = fetcher.get_upcoming_earnings(days_ahead=14)
"""

import json
import logging
import requests
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


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
    time: Optional[str] = None  # "bmo", "amc", "dmt" (before market, after market, during market)
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
            return 90.0   # Next 3 days
        elif days_until <= 7:
            return 80.0   # Next week
        elif days_until <= 14:
            return 60.0   # Next 2 weeks
        elif days_until <= 30:
            return 40.0   # Next month
        else:
            return 20.0   # Future


class EarningsCalendarFetcher:
    """Multi-source earnings calendar data fetcher."""
    
    def __init__(self, cache_dir: Path = Path("cache/earnings")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API configurations (free tiers)
        self.api_configs = {
            EarningsSource.FMP: {
                "base_url": "https://financialmodelingprep.com/api/v3",
                "rate_limit": 250,  # per day free
                "api_key": None  # Can be set via environment or config
            },
            EarningsSource.FINNHUB: {
                "base_url": "https://finnhub.io/api/v1",
                "rate_limit": 60,   # per minute free
                "api_key": None
            }
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Trading-Data-System/1.0'
        })
        
    def get_upcoming_earnings(self, 
                            symbols: Optional[List[str]] = None,
                            days_ahead: int = 30,
                            sources: List[EarningsSource] = None) -> List[EarningsEvent]:
        """Get upcoming earnings events with multi-source fallback."""
        
        if sources is None:
            sources = [EarningsSource.FMP, EarningsSource.FINNHUB, EarningsSource.NASDAQ]
        
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
    
    def _fetch_from_source(self, 
                          source: EarningsSource, 
                          symbols: Optional[List[str]],
                          days_ahead: int) -> List[EarningsEvent]:
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
    
    def _fetch_from_fmp(self, symbols: Optional[List[str]], days_ahead: int) -> List[EarningsEvent]:
        """Fetch from Financial Modeling Prep."""
        earnings = []
        
        # Date range
        start_date = date.today()
        end_date = start_date + timedelta(days=days_ahead)
        
        try:
            # FMP earnings calendar endpoint
            url = f"{self.api_configs[EarningsSource.FMP]['base_url']}/earning_calendar"
            params = {
                'from': start_date.isoformat(),
                'to': end_date.isoformat()
            }
            
            # Add API key if available
            api_key = self.api_configs[EarningsSource.FMP].get('api_key')
            if api_key:
                params['apikey'] = api_key
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            for event in data:
                # Filter by symbols if specified
                symbol = event.get('symbol', '').upper()
                if symbols and symbol not in symbols:
                    continue
                
                # Parse date
                date_str = event.get('date')
                if not date_str:
                    continue
                
                earnings_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                
                earnings_event = EarningsEvent(
                    symbol=symbol,
                    company_name=event.get('name', symbol),
                    earnings_date=earnings_date,
                    time=event.get('time', '').lower(),
                    eps_estimate=self._safe_float(event.get('epsEstimated')),
                    eps_actual=self._safe_float(event.get('eps')),
                    revenue_estimate=self._safe_float(event.get('revenueEstimated')),
                    revenue_actual=self._safe_float(event.get('revenue')),
                    source=EarningsSource.FMP.value
                )
                earnings.append(earnings_event)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"FMP API request failed: {e}")
        except (ValueError, KeyError) as e:
            logger.error(f"FMP data parsing failed: {e}")
            
        return earnings
    
    def _fetch_from_finnhub(self, symbols: Optional[List[str]], days_ahead: int) -> List[EarningsEvent]:
        """Fetch from Finnhub."""
        earnings = []
        
        # Date range
        start_date = date.today()
        end_date = start_date + timedelta(days=days_ahead)
        
        try:
            # Finnhub earnings calendar endpoint
            url = f"{self.api_configs[EarningsSource.FINNHUB]['base_url']}/calendar/earnings"
            params = {
                'from': start_date.isoformat(),
                'to': end_date.isoformat()
            }
            
            # Add API key if available
            api_key = self.api_configs[EarningsSource.FINNHUB].get('api_key')
            if api_key:
                params['token'] = api_key
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            earnings_list = data.get('earningsCalendar', [])
            
            for event in earnings_list:
                # Filter by symbols if specified
                symbol = event.get('symbol', '').upper()
                if symbols and symbol not in symbols:
                    continue
                
                # Parse date
                date_str = event.get('date')
                if not date_str:
                    continue
                
                earnings_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                
                earnings_event = EarningsEvent(
                    symbol=symbol,
                    company_name=event.get('name', symbol),
                    earnings_date=earnings_date,
                    time=event.get('hour', '').lower(),
                    eps_estimate=self._safe_float(event.get('epsEstimate')),
                    eps_actual=self._safe_float(event.get('epsActual')),
                    revenue_estimate=self._safe_float(event.get('revenueEstimate')),
                    revenue_actual=self._safe_float(event.get('revenueActual')),
                    source=EarningsSource.FINNHUB.value
                )
                earnings.append(earnings_event)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Finnhub API request failed: {e}")
        except (ValueError, KeyError) as e:
            logger.error(f"Finnhub data parsing failed: {e}")
            
        return earnings
    
    def _fetch_from_nasdaq(self, symbols: Optional[List[str]], days_ahead: int) -> List[EarningsEvent]:
        """Fetch from NASDAQ public API with enhanced multi-day support."""
        earnings = []
        
        start_date = date.today()
        logger.info(f"📅 Fetching NASDAQ earnings for {days_ahead} days ahead")
        
        for i in range(days_ahead + 1):  # Include today
            check_date = start_date + timedelta(days=i)
            
            # Skip weekends (NASDAQ typically has no earnings data)
            if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                continue
            
            try:
                daily_earnings = self._fetch_nasdaq_single_day(check_date, symbols)
                if daily_earnings:
                    earnings.extend(daily_earnings)
                    logger.debug(f"✅ NASDAQ {check_date}: {len(daily_earnings)} earnings")
                
                # Be respectful to NASDAQ servers - small delay
                import time
                time.sleep(0.3)
                
            except Exception as e:
                logger.warning(f"⚠️ NASDAQ {check_date} failed: {e}")
                continue
        
        logger.info(f"📊 NASDAQ total earnings collected: {len(earnings)}")
        return earnings
    
    def _fetch_nasdaq_single_day(self, target_date: date, symbols: Optional[List[str]]) -> List[EarningsEvent]:
        """Fetch NASDAQ earnings for a single day."""
        
        url = "https://api.nasdaq.com/api/calendar/earnings"
        params = {
            'date': target_date.strftime('%Y-%m-%d')
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.nasdaq.com/',
            'Origin': 'https://www.nasdaq.com'
        }
        
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            daily_earnings = []
            
            # Parse NASDAQ response format
            if 'data' in data and data['data'] and 'rows' in data['data']:
                for row in data['data']['rows']:
                    symbol = row.get('symbol', '').strip().upper()
                    
                    # Filter by symbols if specified
                    if symbols and symbol not in symbols:
                        continue
                    
                    # Skip invalid symbols
                    if not symbol or not symbol.replace('.', '').replace('-', '').isalnum():
                        continue
                    
                    # Parse earnings time
                    time_str = row.get('time', '').lower()
                    if 'pre-market' in time_str or 'pre' in time_str:
                        timing = 'bmo'  # Before market open
                    elif 'after' in time_str or 'post' in time_str:
                        timing = 'amc'  # After market close
                    else:
                        timing = 'dmt'  # During market hours
                    
                    # Parse EPS forecast
                    eps_forecast = self._safe_float(row.get('epsForecast'))
                    
                    # Parse market cap for filtering
                    market_cap_str = row.get('marketCap', '')
                    market_cap = self._parse_market_cap(market_cap_str)
                    
                    earnings_event = EarningsEvent(
                        symbol=symbol,
                        company_name=row.get('name', symbol),
                        earnings_date=target_date,
                        time=timing,
                        eps_estimate=eps_forecast,
                        source=EarningsSource.NASDAQ.value
                    )
                    
                    # Add market cap as custom attribute for filtering
                    earnings_event.market_cap = market_cap
                    
                    daily_earnings.append(earnings_event)
            
            return daily_earnings
            
        except Exception as e:
            logger.error(f"NASDAQ single day fetch failed for {target_date}: {e}")
            return []
    
    def _parse_market_cap(self, market_cap_str: str) -> Optional[float]:
        """Parse market cap string to float value."""
        if not market_cap_str or market_cap_str in ['N/A', '']:
            return None
        
        try:
            # Remove currency symbols and commas
            clean_str = market_cap_str.replace('$', '').replace(',', '').strip()
            
            if 'B' in clean_str or 'b' in clean_str:
                # Billion
                number = float(clean_str.replace('B', '').replace('b', ''))
                return number * 1e9
            elif 'M' in clean_str or 'm' in clean_str:
                # Million  
                number = float(clean_str.replace('M', '').replace('m', ''))
                return number * 1e6
            elif 'K' in clean_str or 'k' in clean_str:
                # Thousand
                number = float(clean_str.replace('K', '').replace('k', ''))
                return number * 1e3
            else:
                # Raw number
                return float(clean_str)
        except (ValueError, AttributeError):
            return None
    
    def _fetch_from_yahoo(self, symbols: Optional[List[str]], days_ahead: int) -> List[EarningsEvent]:
        """Emergency fallback: Yahoo Finance scraping."""
        earnings = []
        
        try:
            # Yahoo Finance earnings calendar URL
            url = "https://finance.yahoo.com/calendar/earnings"
            params = {'size': 100}
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; Trading-System/1.0)'
            }
            
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Simple scraping approach (very basic)
            # In practice, you'd parse the HTML or look for JSON data in the page
            # This is just a placeholder for emergency fallback
            logger.warning("Yahoo Finance scraping not fully implemented - emergency fallback")
            
        except Exception as e:
            logger.error(f"Yahoo Finance scraping failed: {e}")
            
        return earnings
    
    def _deduplicate_earnings(self, earnings: List[EarningsEvent]) -> List[EarningsEvent]:
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
                if (event.eps_estimate is not None and existing.eps_estimate is None) or \
                   (event.revenue_estimate is not None and existing.revenue_estimate is None):
                    seen[key] = event
                    unique[unique.index(existing)] = event
        
        return unique
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert to float."""
        if value is None or value == '':
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
            'timestamp': datetime.now().isoformat(),
            'earnings': [
                {
                    'symbol': e.symbol,
                    'company_name': e.company_name,
                    'earnings_date': e.earnings_date.isoformat(),
                    'time': e.time,
                    'eps_estimate': e.eps_estimate,
                    'eps_actual': e.eps_actual,
                    'revenue_estimate': e.revenue_estimate,
                    'revenue_actual': e.revenue_actual,
                    'source': e.source,
                    'priority_score': e.priority_score
                }
                for e in earnings
            ]
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Cached {len(earnings)} earnings events to {cache_file}")
    
    def load_from_cache(self, cache_key: str = None, max_age_hours: int = 6) -> Optional[List[EarningsEvent]]:
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
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            earnings = []
            for item in data['earnings']:
                earnings_event = EarningsEvent(
                    symbol=item['symbol'],
                    company_name=item['company_name'],
                    earnings_date=datetime.strptime(item['earnings_date'], '%Y-%m-%d').date(),
                    time=item.get('time'),
                    eps_estimate=item.get('eps_estimate'),
                    eps_actual=item.get('eps_actual'),
                    revenue_estimate=item.get('revenue_estimate'),
                    revenue_actual=item.get('revenue_actual'),
                    source=item['source']
                )
                earnings.append(earnings_event)
            
            logger.info(f"📂 Loaded {len(earnings)} earnings events from cache")
            return earnings
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None


def main():
    """Test the earnings fetcher."""
    logging.basicConfig(level=logging.INFO)
    
    fetcher = EarningsCalendarFetcher()
    
    print("🔍 Testing earnings calendar fetcher...")
    
    # Test with some popular symbols
    test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    
    earnings = fetcher.get_upcoming_earnings(
        symbols=test_symbols,
        days_ahead=14
    )
    
    if earnings:
        print(f"\n📊 Found {len(earnings)} upcoming earnings events:")
        
        for event in earnings[:10]:  # Show first 10
            priority_emoji = "🔥" if event.priority_score >= 80 else "📅"
            print(f"{priority_emoji} {event.symbol}: {event.earnings_date} "
                  f"({event.days_until_earnings} days) - Priority: {event.priority_score:.0f}")
        
        # Save to cache for testing
        fetcher.save_to_cache(earnings)
        
    else:
        print("❌ No earnings data found")


if __name__ == "__main__":
    main()