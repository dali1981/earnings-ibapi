#!/usr/bin/env python3
"""
Debug Market Coverage - Check what APIs actually return
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from earnings.fetcher import EarningsCalendarFetcher

def main():
    print("🔍 CHECKING ACTUAL MARKET COVERAGE")
    print("=" * 60)
    
    fetcher = EarningsCalendarFetcher()
    
    # Test each source individually
    print("Testing earnings sources...")
    
    # Get all earnings (no symbol filter)
    all_earnings = fetcher.get_upcoming_earnings(
        symbols=None,  # No filter
        days_ahead=30
    )
    
    print(f"\n📊 RESULTS:")
    print(f"Total events found: {len(all_earnings)}")
    
    if all_earnings:
        # Analyze what we got
        symbols = set(e.symbol for e in all_earnings)
        print(f"Unique symbols: {len(symbols)}")
        print(f"Date range: {min(e.earnings_date for e in all_earnings)} to {max(e.earnings_date for e in all_earnings)}")
        
        # Show sample
        print(f"\nSample earnings events:")
        for i, event in enumerate(all_earnings[:10], 1):
            print(f"{i:2d}. {event.symbol:<8} - {event.earnings_date} - {event.company_name[:30]}")
        
        # Show sources
        sources = set(e.source for e in all_earnings)
        print(f"\nData sources used: {sources}")
        
        # Analyze by market cap proxy (symbol patterns)
        large_cap = [e for e in all_earnings if len(e.symbol) <= 4 and e.symbol.isalpha()]
        print(f"\nLarge cap candidates (simple filter): {len(large_cap)}")
        if large_cap:
            large_cap_symbols = [e.symbol for e in large_cap[:20]]
            print(f"Sample large caps: {large_cap_symbols}")
    
    else:
        print("❌ No earnings data - this means:")
        print("   • API keys not configured")
        print("   • API rate limits hit")  
        print("   • API endpoints changed")
        print("   • Network/firewall issues")
        
        print(f"\n💡 To get real market data:")
        print(f"   1. Get free API key: https://financialmodelingprep.com")
        print(f"   2. Set: export FMP_API_KEY='your_key'")
        print(f"   3. Or get Finnhub key: https://finnhub.io")
        print(f"   4. Set: export FINNHUB_API_KEY='your_key'")

if __name__ == "__main__":
    main()