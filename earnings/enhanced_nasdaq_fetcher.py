#!/usr/bin/env python3
"""
Enhanced NASDAQ Fetcher

Scrapes multiple days from NASDAQ to get broader market coverage
without requiring API keys.
"""

import requests
import logging
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from earnings.fetcher import EarningsEvent, EarningsSource

logger = logging.getLogger(__name__)


class EnhancedNASDAQFetcher:
    """Enhanced NASDAQ fetcher that scrapes multiple days."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; Trading-System/1.0)',
            'Accept': 'application/json'
        })
    
    def fetch_multiple_days(self, days_ahead: int = 21) -> List[EarningsEvent]:
        """Fetch earnings for multiple days from NASDAQ."""
        
        all_earnings = []
        start_date = date.today()
        
        logger.info(f"📅 Fetching NASDAQ earnings for {days_ahead} days")
        
        for i in range(days_ahead):
            check_date = start_date + timedelta(days=i)
            
            # Skip weekends (NASDAQ typically has no data)
            if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                continue
            
            try:
                daily_earnings = self._fetch_single_day(check_date)
                if daily_earnings:
                    all_earnings.extend(daily_earnings)
                    logger.info(f"✅ {check_date}: {len(daily_earnings)} earnings")
                else:
                    logger.debug(f"📅 {check_date}: No earnings")
                
                # Be nice to NASDAQ - small delay between requests
                import time
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"⚠️ {check_date} failed: {e}")
                continue
        
        logger.info(f"📊 Total NASDAQ earnings collected: {len(all_earnings)}")
        return all_earnings
    
    def _fetch_single_day(self, check_date: date) -> List[EarningsEvent]:
        """Fetch earnings for a single day."""
        
        url = "https://api.nasdaq.com/api/calendar/earnings"
        params = {
            'date': check_date.strftime('%Y-%m-%d')
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            earnings = []
            
            # Parse NASDAQ response format
            if 'data' in data and 'rows' in data['data']:
                for row in data['data']['rows']:
                    symbol = row.get('symbol', '').strip().upper()
                    
                    if symbol and symbol.replace('.', '').replace('-', '').isalnum():
                        earnings_event = EarningsEvent(
                            symbol=symbol,
                            company_name=row.get('companyName', symbol),
                            earnings_date=check_date,
                            time=row.get('time', '').lower(),
                            source=EarningsSource.NASDAQ.value
                        )
                        earnings.append(earnings_event)
            
            return earnings
            
        except Exception as e:
            logger.error(f"NASDAQ fetch failed for {check_date}: {e}")
            return []


def test_enhanced_fetcher():
    """Test the enhanced NASDAQ fetcher."""
    print("🔍 TESTING ENHANCED NASDAQ FETCHER")
    print("=" * 50)
    
    fetcher = EnhancedNASDAQFetcher()
    
    # Fetch next 10 trading days
    earnings = fetcher.fetch_multiple_days(days_ahead=10)
    
    if earnings:
        print(f"✅ Found {len(earnings)} earnings events")
        
        # Group by date
        by_date = {}
        for event in earnings:
            date_key = event.earnings_date
            if date_key not in by_date:
                by_date[date_key] = []
            by_date[date_key].append(event)
        
        print(f"\n📅 Earnings by date:")
        for earnings_date in sorted(by_date.keys()):
            events = by_date[earnings_date]
            weekday = earnings_date.strftime('%A')
            print(f"  {earnings_date} ({weekday}): {len(events)} companies")
            
            # Show sample symbols
            if events:
                sample_symbols = [e.symbol for e in events[:10]]
                print(f"    Sample: {', '.join(sample_symbols)}")
        
        # Show unique symbols
        unique_symbols = set(e.symbol for e in earnings)
        print(f"\n📊 Unique symbols: {len(unique_symbols)}")
        
        # Look for recognizable names
        well_known = []
        for event in earnings:
            if event.symbol in ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'META', 'AMZN', 'NFLX', 'NVDA']:
                well_known.append(event.symbol)
        
        if well_known:
            print(f"🎯 Well-known symbols found: {well_known}")
    
    else:
        print("❌ No earnings found")
        print("💡 This could mean:")
        print("   • No earnings scheduled for next few days")
        print("   • NASDAQ API structure changed")
        print("   • Network/access issues")


if __name__ == "__main__":
    from utils.logging_setup import setup_logging
    setup_logging()
    test_enhanced_fetcher()