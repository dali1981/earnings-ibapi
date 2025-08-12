#!/usr/bin/env python3
"""
Export Earnings Calendar to File

This script fetches earnings data for portfolio symbols and exports them
to various file formats with earnings dates.

Usage:
    python scripts/export_earnings_calendar.py --output earnings_calendar.csv
    python scripts/export_earnings_calendar.py --format json --days 30
"""

import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from earnings.fetcher import EarningsCalendarFetcher
from jobs.run_daily_comprehensive import PortfolioManager


def export_earnings_calendar(output_file: Path, 
                           portfolio_config: Path = None,
                           format_type: str = 'csv',
                           days_ahead: int = 30,
                           symbols: list = None) -> dict:
    """Export earnings calendar to file with multiple format options."""
    
    # Load portfolio symbols if not provided
    if symbols is None:
        portfolio_manager = PortfolioManager(portfolio_config)
        symbols = portfolio_manager.get_active_symbols()
    
    print(f"📊 Fetching earnings for {len(symbols)} symbols")
    print(f"📅 Looking {days_ahead} days ahead")
    
    # Initialize earnings fetcher
    earnings_fetcher = EarningsCalendarFetcher()
    
    # Fetch earnings events
    earnings_events = earnings_fetcher.get_upcoming_earnings(
        symbols=symbols,
        days_ahead=days_ahead
    )
    
    print(f"✅ Found {len(earnings_events)} earnings events")
    
    # Prepare output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Export based on format
    if format_type.lower() == 'csv':
        export_to_csv(earnings_events, output_file, symbols)
    elif format_type.lower() == 'json':
        export_to_json(earnings_events, output_file, symbols)
    elif format_type.lower() == 'txt':
        export_to_txt(earnings_events, output_file, symbols)
    elif format_type.lower() == 'calendar':
        export_to_calendar(earnings_events, output_file, symbols)
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    # Return summary
    earnings_symbols = {event.symbol for event in earnings_events}
    no_earnings_symbols = [s for s in symbols if s not in earnings_symbols]
    
    return {
        'total_symbols': len(symbols),
        'earnings_events': len(earnings_events),
        'symbols_with_earnings': len(earnings_symbols),
        'symbols_without_earnings': len(no_earnings_symbols),
        'output_file': str(output_file),
        'format': format_type
    }


def export_to_csv(earnings_events: list, output_file: Path, all_symbols: list):
    """Export to CSV format."""
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow([
            'Symbol', 'Company_Name', 'Earnings_Date', 'Time', 
            'Days_Until_Earnings', 'Priority_Score',
            'EPS_Estimate', 'EPS_Actual', 'Revenue_Estimate', 'Revenue_Actual',
            'Source'
        ])
        
        # Earnings events (sorted by date)
        sorted_events = sorted(earnings_events, key=lambda x: x.earnings_date)
        
        for event in sorted_events:
            writer.writerow([
                event.symbol,
                event.company_name,
                event.earnings_date.isoformat(),
                event.time or '',
                event.days_until_earnings,
                f"{event.priority_score:.1f}",
                event.eps_estimate or '',
                event.eps_actual or '',
                event.revenue_estimate or '',
                event.revenue_actual or '',
                event.source
            ])
        
        # Add symbols without earnings
        earnings_symbols = {event.symbol for event in earnings_events}
        no_earnings_symbols = [s for s in all_symbols if s not in earnings_symbols]
        
        for symbol in sorted(no_earnings_symbols):
            writer.writerow([
                symbol, '', '', '', '', '', '', '', '', '', 'no_earnings_found'
            ])
    
    print(f"💾 CSV exported to: {output_file}")


def export_to_json(earnings_events: list, output_file: Path, all_symbols: list):
    """Export to JSON format."""
    
    # Group events by symbol
    events_by_symbol = {}
    for event in earnings_events:
        events_by_symbol[event.symbol] = {
            'company_name': event.company_name,
            'earnings_date': event.earnings_date.isoformat(),
            'time': event.time,
            'days_until_earnings': event.days_until_earnings,
            'priority_score': event.priority_score,
            'eps_estimate': event.eps_estimate,
            'eps_actual': event.eps_actual,
            'revenue_estimate': event.revenue_estimate,
            'revenue_actual': event.revenue_actual,
            'source': event.source
        }
    
    # Add symbols without earnings
    earnings_symbols = {event.symbol for event in earnings_events}
    no_earnings_symbols = [s for s in all_symbols if s not in earnings_symbols]
    
    for symbol in no_earnings_symbols:
        events_by_symbol[symbol] = {
            'company_name': '',
            'earnings_date': None,
            'time': None,
            'days_until_earnings': None,
            'priority_score': 0,
            'eps_estimate': None,
            'eps_actual': None,
            'revenue_estimate': None,
            'revenue_actual': None,
            'source': 'no_earnings_found'
        }
    
    data = {
        'export_timestamp': datetime.now().isoformat(),
        'total_symbols': len(all_symbols),
        'earnings_events_found': len(earnings_events),
        'earnings_calendar': events_by_symbol
    }
    
    with open(output_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=2, default=str)
    
    print(f"💾 JSON exported to: {output_file}")


def export_to_txt(earnings_events: list, output_file: Path, all_symbols: list):
    """Export to human-readable text format."""
    
    with open(output_file, 'w', encoding='utf-8') as txtfile:
        txtfile.write("EARNINGS CALENDAR EXPORT\n")
        txtfile.write("=" * 50 + "\n")
        txtfile.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        txtfile.write(f"Total symbols: {len(all_symbols)}\n")
        txtfile.write(f"Earnings events found: {len(earnings_events)}\n\n")
        
        if earnings_events:
            txtfile.write("UPCOMING EARNINGS\n")
            txtfile.write("-" * 30 + "\n")
            
            # Sort by date and priority
            sorted_events = sorted(earnings_events, key=lambda x: (x.earnings_date, -x.priority_score))
            
            for event in sorted_events:
                priority_emoji = "🔥" if event.priority_score >= 80 else "📅"
                txtfile.write(f"{priority_emoji} {event.symbol:<6} - {event.earnings_date} "
                            f"({event.days_until_earnings:+3d} days)\n")
                txtfile.write(f"    Company: {event.company_name}\n")
                if event.time:
                    txtfile.write(f"    Time: {event.time.upper()}\n")
                if event.eps_estimate:
                    txtfile.write(f"    EPS Est: ${event.eps_estimate:.2f}\n")
                if event.revenue_estimate:
                    txtfile.write(f"    Revenue Est: ${event.revenue_estimate/1e9:.1f}B\n")
                txtfile.write(f"    Priority: {event.priority_score:.0f}\n")
                txtfile.write("\n")
        
        # Symbols without earnings
        earnings_symbols = {event.symbol for event in earnings_events}
        no_earnings_symbols = [s for s in all_symbols if s not in earnings_symbols]
        
        if no_earnings_symbols:
            txtfile.write("SYMBOLS WITHOUT EARNINGS DATA\n")
            txtfile.write("-" * 35 + "\n")
            
            for symbol in sorted(no_earnings_symbols):
                txtfile.write(f"🔧 {symbol} - No earnings data found\n")
    
    print(f"💾 Text report exported to: {output_file}")


def export_to_calendar(earnings_events: list, output_file: Path, all_symbols: list):
    """Export to calendar-style format grouped by date."""
    
    # Group events by date
    events_by_date = {}
    for event in earnings_events:
        date_key = event.earnings_date
        if date_key not in events_by_date:
            events_by_date[date_key] = []
        events_by_date[date_key].append(event)
    
    with open(output_file, 'w', encoding='utf-8') as calfile:
        calfile.write("EARNINGS CALENDAR\n")
        calfile.write("=" * 60 + "\n")
        calfile.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if events_by_date:
            # Sort dates
            sorted_dates = sorted(events_by_date.keys())
            
            for earnings_date in sorted_dates:
                events = events_by_date[earnings_date]
                weekday = earnings_date.strftime('%A')
                days_until = (earnings_date - date.today()).days
                
                calfile.write(f"{earnings_date.strftime('%Y-%m-%d')} ({weekday}) - {days_until:+d} days\n")
                calfile.write("-" * 50 + "\n")
                
                # Sort by priority within the day
                events.sort(key=lambda x: -x.priority_score)
                
                for event in events:
                    priority_emoji = "🔥" if event.priority_score >= 80 else "📅"
                    time_str = f" {event.time.upper()}" if event.time else ""
                    calfile.write(f"  {priority_emoji} {event.symbol:<8}{time_str:<4} - {event.company_name}\n")
                    
                    if event.eps_estimate or event.revenue_estimate:
                        estimates = []
                        if event.eps_estimate:
                            estimates.append(f"EPS: ${event.eps_estimate:.2f}")
                        if event.revenue_estimate:
                            estimates.append(f"Rev: ${event.revenue_estimate/1e9:.1f}B")
                        calfile.write(f"      Estimates: {', '.join(estimates)}\n")
                
                calfile.write("\n")
        
        # Summary of symbols without earnings
        earnings_symbols = {event.symbol for event in earnings_events}
        no_earnings_symbols = [s for s in all_symbols if s not in earnings_symbols]
        
        if no_earnings_symbols:
            calfile.write("SYMBOLS WITHOUT EARNINGS DATA\n")
            calfile.write("-" * 35 + "\n")
            calfile.write(f"Count: {len(no_earnings_symbols)}\n")
            calfile.write(f"Symbols: {', '.join(sorted(no_earnings_symbols))}\n")
    
    print(f"💾 Calendar exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Export earnings calendar for portfolio symbols",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export to CSV
    python scripts/export_earnings_calendar.py --output earnings.csv
    
    # Export to JSON with 60-day lookahead
    python scripts/export_earnings_calendar.py --format json --days 60 --output earnings.json
    
    # Export specific symbols to calendar format
    python scripts/export_earnings_calendar.py --symbols AAPL GOOGL MSFT --format calendar
    
    # Export with custom portfolio config
    python scripts/export_earnings_calendar.py --config config/my_portfolio.yaml --format txt
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path (default: earnings_YYYY-MM-DD.{format})"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=['csv', 'json', 'txt', 'calendar'],
        default='csv',
        help="Export format (default: csv)"
    )
    
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=30,
        help="Days to look ahead for earnings (default: 30)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Portfolio configuration file (default: config/portfolio.yaml)"
    )
    
    parser.add_argument(
        "--symbols", "-s",
        nargs="*",
        help="Specific symbols to export (overrides portfolio config)"
    )
    
    args = parser.parse_args()
    
    # Generate default output filename if not provided
    if not args.output:
        today = date.today().isoformat()
        args.output = Path(f"exports/earnings_{today}.{args.format}")
    
    try:
        # Export earnings calendar
        result = export_earnings_calendar(
            output_file=args.output,
            portfolio_config=args.config,
            format_type=args.format,
            days_ahead=args.days,
            symbols=args.symbols
        )
        
        # Print summary
        print(f"\n📊 EXPORT SUMMARY:")
        print(f"   Total symbols: {result['total_symbols']}")
        print(f"   Earnings events: {result['earnings_events']}")
        print(f"   Symbols with earnings: {result['symbols_with_earnings']}")
        print(f"   Symbols without earnings: {result['symbols_without_earnings']}")
        print(f"   Output file: {result['output_file']}")
        print(f"   Format: {result['format']}")
        
        if result['earnings_events'] == 0:
            print(f"\n⚠️  No earnings data found. This may be due to:")
            print(f"   • Missing API keys (set FMP_API_KEY, FINNHUB_API_KEY)")
            print(f"   • API rate limits exceeded")
            print(f"   • No earnings in the specified time period")
            print(f"   • Symbols not covered by earnings APIs")
    
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()