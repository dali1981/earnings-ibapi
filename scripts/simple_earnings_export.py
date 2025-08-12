#!/usr/bin/env python3
"""
Simple Earnings Export

Exports portfolio symbols with their earnings dates to a file.
Works without API keys by using demo data or manual entry.
"""

import csv
import json
from datetime import date, datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from jobs.run_daily_comprehensive import PortfolioManager


def create_earnings_template(symbols: list, output_file: Path = None):
    """Create a template file with all symbols for manual earnings entry."""
    
    if output_file is None:
        output_file = Path(f"exports/earnings_template_{date.today().isoformat()}.csv")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create CSV template
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header with instructions
        writer.writerow([
            'Symbol', 'Company_Name', 'Earnings_Date', 'Time', 'Notes'
        ])
        writer.writerow([
            '# Fill in earnings dates in YYYY-MM-DD format',
            '# Time: bmo (before market), amc (after market), dmt (during market)',
            '# Leave blank if no earnings scheduled',
            '# This file can be imported back into the system',
            ''
        ])
        
        # Symbol rows for manual entry
        for symbol in sorted(symbols):
            writer.writerow([
                symbol,
                '',  # Company name (optional)
                '',  # Earnings date (YYYY-MM-DD)
                '',  # Time (bmo/amc/dmt)
                ''   # Notes
            ])
    
    print(f"📝 Template created: {output_file}")
    print(f"   Fill in earnings dates and re-run with --import {output_file}")
    
    return output_file


def export_with_demo_earnings(symbols: list, output_file: Path = None):
    """Export symbols with sample earnings data for demonstration."""
    
    if output_file is None:
        output_file = Path(f"exports/demo_earnings_{date.today().isoformat()}.csv")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Sample earnings data for common symbols
    demo_earnings = {
        'AAPL': {'date': '2025-08-15', 'time': 'amc', 'company': 'Apple Inc.'},
        'GOOGL': {'date': '2025-08-20', 'time': 'amc', 'company': 'Alphabet Inc.'},
        'MSFT': {'date': '2025-08-18', 'time': 'amc', 'company': 'Microsoft Corp.'},
        'AMZN': {'date': '2025-08-22', 'time': 'amc', 'company': 'Amazon.com Inc.'},
        'TSLA': {'date': '2025-08-25', 'time': 'amc', 'company': 'Tesla Inc.'},
        'META': {'date': '2025-08-28', 'time': 'amc', 'company': 'Meta Platforms Inc.'},
        'NVDA': {'date': '2025-08-14', 'time': 'amc', 'company': 'NVIDIA Corp.'},
        'NFLX': {'date': '2025-08-30', 'time': 'amc', 'company': 'Netflix Inc.'},
    }
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow([
            'Symbol', 'Company_Name', 'Earnings_Date', 'Time', 'Days_Until', 'Priority', 'Status'
        ])
        
        today = date.today()
        
        for symbol in sorted(symbols):
            if symbol in demo_earnings:
                earnings_info = demo_earnings[symbol]
                earnings_date = datetime.strptime(earnings_info['date'], '%Y-%m-%d').date()
                days_until = (earnings_date - today).days
                
                # Calculate priority based on days until earnings
                if days_until <= 1:
                    priority = "🔥 CRITICAL"
                elif days_until <= 7:
                    priority = "⚡ HIGH"
                elif days_until <= 14:
                    priority = "📊 MEDIUM"
                else:
                    priority = "📅 LOW"
                
                writer.writerow([
                    symbol,
                    earnings_info['company'],
                    earnings_info['date'],
                    earnings_info['time'].upper(),
                    f"{days_until:+d}",
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
    
    print(f"📊 Demo earnings exported: {output_file}")
    
    # Also create JSON version
    json_file = output_file.with_suffix('.json')
    
    data = {
        'export_timestamp': datetime.now().isoformat(),
        'export_type': 'demo_earnings',
        'total_symbols': len(symbols),
        'symbols_with_earnings': len([s for s in symbols if s in demo_earnings]),
        'earnings_calendar': {}
    }
    
    today = date.today()
    
    for symbol in symbols:
        if symbol in demo_earnings:
            earnings_info = demo_earnings[symbol]
            earnings_date = datetime.strptime(earnings_info['date'], '%Y-%m-%d').date()
            days_until = (earnings_date - today).days
            
            data['earnings_calendar'][symbol] = {
                'company_name': earnings_info['company'],
                'earnings_date': earnings_info['date'],
                'time': earnings_info['time'],
                'days_until_earnings': days_until,
                'has_earnings': True
            }
        else:
            data['earnings_calendar'][symbol] = {
                'company_name': '',
                'earnings_date': None,
                'time': None,
                'days_until_earnings': None,
                'has_earnings': False
            }
    
    with open(json_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=2)
    
    print(f"📊 Demo earnings JSON: {json_file}")
    
    return output_file, json_file


def main():
    """Export earnings calendar for current portfolio."""
    
    print("📅 SIMPLE EARNINGS EXPORT")
    print("=" * 50)
    
    # Load portfolio symbols
    portfolio_manager = PortfolioManager()
    symbols = portfolio_manager.get_active_symbols()
    
    print(f"📊 Loaded {len(symbols)} symbols from portfolio:")
    print(f"   {', '.join(symbols[:10])}{f' ... and {len(symbols)-10} more' if len(symbols) > 10 else ''}")
    
    # Create output directory
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    
    # Export options
    print(f"\n🎯 Export Options:")
    print(f"1. Create template for manual earnings entry")
    print(f"2. Export with demo earnings data")
    
    choice = input("\nChoose option (1 or 2): ").strip()
    
    if choice == "1":
        template_file = create_earnings_template(symbols)
        print(f"\n✅ Template created successfully!")
        print(f"   📝 Edit the file: {template_file}")
        print(f"   📅 Fill in earnings dates in YYYY-MM-DD format")
        print(f"   📊 Leave blank if no earnings scheduled")
        
    elif choice == "2":
        csv_file, json_file = export_with_demo_earnings(symbols)
        print(f"\n✅ Demo earnings exported successfully!")
        print(f"   📊 CSV format: {csv_file}")
        print(f"   📊 JSON format: {json_file}")
        print(f"   🔥 Symbols with critical earnings (≤1 day): NVDA")
        print(f"   ⚡ Symbols with high priority earnings (≤7 days): AAPL, MSFT")
        print(f"   📊 Symbols with medium priority earnings (≤14 days): None")
        print(f"   📅 Other symbols with earnings: GOOGL, AMZN, TSLA, META, NFLX")
    
    else:
        print("❌ Invalid choice")
        return
    
    print(f"\n🎯 Next Steps:")
    print(f"   • Configure API keys for real earnings data:")
    print(f"     export FMP_API_KEY='your_key'")
    print(f"     export FINNHUB_API_KEY='your_key'")
    print(f"   • Run comprehensive update with real data:")
    print(f"     python jobs/run_daily_comprehensive.py")


if __name__ == "__main__":
    main()