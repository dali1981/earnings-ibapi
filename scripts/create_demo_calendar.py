#!/usr/bin/env python3
"""Create demo earnings calendar for portfolio symbols."""

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from jobs.run_daily_comprehensive import PortfolioManager

def create_demo_calendar():
    """Create demo earnings calendar."""
    
    # Create demo earnings with realistic dates
    today = date.today()

    demo_earnings = {
        'AAPL': {'company': 'Apple Inc.', 'date': today + timedelta(days=2), 'time': 'amc'},
        'GOOGL': {'company': 'Alphabet Inc.', 'date': today + timedelta(days=5), 'time': 'amc'},
        'MSFT': {'company': 'Microsoft Corp.', 'date': today + timedelta(days=8), 'time': 'amc'},
        'AMZN': {'company': 'Amazon.com Inc.', 'date': today + timedelta(days=12), 'time': 'amc'},
        'TSLA': {'company': 'Tesla Inc.', 'date': today + timedelta(days=15), 'time': 'amc'},
        'META': {'company': 'Meta Platforms Inc.', 'date': today + timedelta(days=18), 'time': 'amc'},
        'NVDA': {'company': 'NVIDIA Corp.', 'date': today + timedelta(days=3), 'time': 'amc'},
        'NFLX': {'company': 'Netflix Inc.', 'date': today + timedelta(days=22), 'time': 'amc'},
        'AMD': {'company': 'Advanced Micro Devices', 'date': today + timedelta(days=25), 'time': 'amc'},
        'CRM': {'company': 'Salesforce Inc.', 'date': today + timedelta(days=28), 'time': 'amc'},
    }

    # Load all symbols
    portfolio_manager = PortfolioManager()
    symbols = portfolio_manager.get_active_symbols()

    # Create demo earnings CSV
    demo_file = Path('exports/demo_earnings_2025-08-12.csv')
    demo_file.parent.mkdir(exist_ok=True)

    with open(demo_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow([
            'Symbol', 'Company_Name', 'Earnings_Date', 'Time', 'Days_Until', 'Priority', 'Status'
        ])
        
        for symbol in sorted(symbols):
            if symbol in demo_earnings:
                info = demo_earnings[symbol]
                days_until = (info['date'] - today).days
                
                # Calculate priority
                if days_until <= 1:
                    priority = '🔥 CRITICAL'
                elif days_until <= 7:
                    priority = '⚡ HIGH'
                elif days_until <= 14:
                    priority = '📊 MEDIUM'  
                else:
                    priority = '📅 LOW'
                
                writer.writerow([
                    symbol,
                    info['company'],
                    info['date'].isoformat(),
                    info['time'].upper(),
                    f'{days_until:+d}',
                    priority,
                    'HAS_EARNINGS'
                ])
            else:
                writer.writerow([
                    symbol,
                    '',
                    '',
                    '',
                    '',
                    '🔧 MAINTENANCE',
                    'NO_EARNINGS'
                ])

    print(f'✅ Demo earnings file created: {demo_file}')

    # Show earnings calendar
    print(f'\n📅 DEMO EARNINGS CALENDAR:')
    print(f'=' * 60)

    earnings_by_date = {}
    for symbol, info in demo_earnings.items():
        date_key = info['date']
        if date_key not in earnings_by_date:
            earnings_by_date[date_key] = []
        earnings_by_date[date_key].append((symbol, info))

    for earnings_date in sorted(earnings_by_date.keys()):
        weekday = earnings_date.strftime('%A')
        days_until = (earnings_date - today).days
        
        print(f'\n{earnings_date.strftime("%Y-%m-%d")} ({weekday}) - {days_until:+d} days')
        print('-' * 40)
        
        for symbol, info in earnings_by_date[earnings_date]:
            days_until = (info['date'] - today).days
            if days_until <= 1:
                emoji = '🔥'
            elif days_until <= 7:
                emoji = '⚡'
            elif days_until <= 14:
                emoji = '📊'
            else:
                emoji = '📅'
            
            time_str = info['time'].upper()
            company_name = info['company']
            print(f'  {emoji} {symbol:<6} {time_str:<3} - {company_name}')

    print(f'\n🔧 Symbols without earnings: {len(symbols) - len(demo_earnings)} symbols')
    no_earnings = [s for s in symbols if s not in demo_earnings]
    no_earnings_str = ', '.join(sorted(no_earnings))
    print(f'   {no_earnings_str}')
    
    return demo_file

if __name__ == "__main__":
    create_demo_calendar()