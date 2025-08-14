#!/usr/bin/env python3
"""
Earnings Discovery with Historical Data

Demonstrates the new historical earnings discovery capability.
Collects both past and future earnings for comprehensive analysis.

Usage:
    python scripts/discover_with_history.py
    python scripts/discover_with_history.py --days-back 14 --days-ahead 30
    python scripts/discover_with_history.py --historical-only --days-back 30
"""

from datetime import date, timedelta
from pathlib import Path

import argparse
import sys


# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_ROOT, LOGGING_CONFIG

import logging.config


# Set up centralized logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

from earnings.discovery import EarningsDiscoveryEngine


def main():
    """Discover earnings with historical data support."""

    parser = argparse.ArgumentParser(
        description="Discover earnings with historical data"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Number of days back to look for historical earnings",
    )
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=21,
        help="Number of days ahead to look for upcoming earnings",
    )
    parser.add_argument(
        "--historical-only",
        action="store_true",
        help="Only collect historical data (sets days-ahead to 0)",
    )
    parser.add_argument(
        "--future-only",
        action="store_true",
        help="Only collect future data (sets days-back to 0)",
    )

    args = parser.parse_args()

    # Handle exclusive options
    if args.historical_only:
        days_back = args.days_back
        days_ahead = 0
    elif args.future_only:
        days_back = 0
        days_ahead = args.days_ahead
    else:
        days_back = args.days_back
        days_ahead = args.days_ahead

    print("📊 EARNINGS DISCOVERY WITH HISTORICAL DATA")
    print("=" * 60)

    if days_back > 0 and days_ahead > 0:
        print(f"🔍 Collecting: {days_back} days back + {days_ahead} days ahead")
        start_date = date.today() - timedelta(days=days_back)
        end_date = date.today() + timedelta(days=days_ahead)
        print(f"📅 Date range: {start_date} to {end_date}")
    elif days_back > 0:
        print(f"📜 Historical only: {days_back} days back")
        start_date = date.today() - timedelta(days=days_back)
        print(f"📅 Date range: {start_date} to {date.today()}")
    else:
        print(f"🔮 Future only: {days_ahead} days ahead")
        end_date = date.today() + timedelta(days=days_ahead)
        print(f"📅 Date range: {date.today()} to {end_date}")

    # Initialize discovery engine
    engine = EarningsDiscoveryEngine()

    # Discover earnings
    discoveries = engine.discover_and_store_earnings(
        days_ahead=days_ahead, days_back=days_back
    )

    if discoveries:
        print(f"\n📊 Found {len(discoveries)} market discoveries")

        # Analyze time distribution
        historical_count = len([d for d in discoveries if d.days_until_earnings < 0])
        today_count = len([d for d in discoveries if d.days_until_earnings == 0])
        future_count = len([d for d in discoveries if d.days_until_earnings > 0])

        if historical_count > 0:
            print(f"   📜 Historical: {historical_count} earnings")
        if today_count > 0:
            print(f"   📅 Today: {today_count} earnings")
        if future_count > 0:
            print(f"   🔮 Upcoming: {future_count} earnings")

        # Show a sample from each category
        print(f"\n📊 Sample discoveries:")

        # Historical
        historical = [d for d in discoveries if d.days_until_earnings < 0]
        if historical:
            print(f"\n📜 HISTORICAL EARNINGS:")
            for i, d in enumerate(
                sorted(historical, key=lambda x: x.days_until_earnings)[:5], 1
            ):
                days_ago = abs(d.days_until_earnings)
                print(
                    f"   {i}. {d.symbol:<6} - {d.earnings_date} ({days_ago} days ago) - {d.company_name}"
                )

        # Future
        future = [d for d in discoveries if d.days_until_earnings > 0]
        if future:
            print(f"\n🔮 UPCOMING EARNINGS:")
            for i, d in enumerate(
                sorted(future, key=lambda x: x.days_until_earnings)[:5], 1
            ):
                print(
                    f"   {i}. {d.symbol:<6} - {d.earnings_date} (+{d.days_until_earnings} days) - {d.company_name}"
                )

        print(f"\n💾 Data exported to: data/exports/earnings_discoveries_*.csv")
        print(f"📊 For trading analysis: python trading/strategy_analyzer.py")

        # Show use cases for historical data
        if historical_count > 0:
            print(f"\n💡 Historical data use cases:")
            print(f"   • Post-earnings price movement analysis")
            print(f"   • Earnings surprise correlation studies")
            print(f"   • Historical volatility patterns")
            print(f"   • Strategy backtest validation")
            print(f"   • IV crush analysis after earnings")

    else:
        print("❌ No discoveries found")
        print("💡 This could be due to:")
        print("   • Missing earnings API keys")
        print("   • No earnings in the specified timeframe")
        print("   • Market filters too restrictive")


if __name__ == "__main__":
    main()
