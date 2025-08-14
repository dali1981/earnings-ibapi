#!/usr/bin/env python3
"""
Get Latest Earnings Discoveries

Simple utility script that other systems can use to get the latest
earnings discoveries. This demonstrates the clean separation between
discovery and trading analysis.

Usage:
    python scripts/get_latest_discoveries.py
    python scripts/get_latest_discoveries.py --format json
    python scripts/get_latest_discoveries.py --symbols AAPL,GOOGL,MSFT
"""

from pathlib import Path

import argparse
import json
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
    """Get latest earnings discoveries."""

    parser = argparse.ArgumentParser(description="Get latest earnings discoveries")
    parser.add_argument(
        "--format", choices=["csv", "json"], default="csv", help="Output format"
    )
    parser.add_argument(
        "--symbols", type=str, help="Comma-separated list of symbols to filter"
    )
    parser.add_argument(
        "--days", type=int, default=1, help="Maximum age of discoveries in days"
    )
    parser.add_argument(
        "--min-quality", type=float, default=0.0, help="Minimum quality score"
    )

    args = parser.parse_args()

    print("📊 LATEST EARNINGS DISCOVERIES")
    print("=" * 50)
    print("🔍 Pure discovery data - no trading analysis")

    # Get discovery engine
    engine = EarningsDiscoveryEngine()

    # Get recent discoveries
    discoveries_df = engine.get_recent_discoveries(max_age_days=args.days)

    if discoveries_df is None or discoveries_df.empty:
        print("❌ No recent discoveries found")
        print(f"💡 Run: python earnings/discovery.py")
        return

    # Filter by symbols if requested
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        discoveries_df = discoveries_df[discoveries_df["symbol"].isin(symbols)]
        print(f"🎯 Filtered for symbols: {symbols}")

    # Filter by quality score
    if args.min_quality > 0:
        discoveries_df = discoveries_df[
            discoveries_df["quality_score"] >= args.min_quality
        ]
        print(f"⭐ Minimum quality score: {args.min_quality}")

    if discoveries_df.empty:
        print("❌ No discoveries match your filters")
        return

    print(f"📊 Found {len(discoveries_df)} discoveries")

    if args.format == "json":
        # Convert to JSON
        discoveries_json = discoveries_df.to_json(orient="records", date_format="iso")
        print("\n" + discoveries_json)
    else:
        # Display as table
        print(
            f"\n{'Symbol':<6} {'Company':<30} {'Earnings':<12} {'Days':<5} {'Quality':<8} {'Market Cap':<12}"
        )
        print("-" * 80)

        for _, row in discoveries_df.iterrows():
            market_cap_str = (
                f"${row['market_cap']/1e9:.1f}B" if row["market_cap"] else "N/A"
            )
            print(
                f"{row['symbol']:<6} {row['company_name'][:29]:<30} "
                f"{row['earnings_date']:<12} {row['days_until']:>3}d "
                f"{row['quality_score']:>6.0f}% {market_cap_str:<12}"
            )

    print(f"\n💾 Full data available in: data/exports/earnings_discoveries_*.csv")
    print(f"📊 For trading analysis, use: python trading/strategy_analyzer.py")


if __name__ == "__main__":
    main()
