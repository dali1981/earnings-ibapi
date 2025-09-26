#!/usr/bin/env python3
"""
Simple Earnings Data Storage Script

Pure data storage script that fetches and stores earnings data without any analysis,
filtering, or strategy recommendations. Just raw data collection and persistence.

Usage:
    python scripts/store_earnings_data.py
    python scripts/store_earnings_data.py --days-ahead 30 --source nasdaq
    python scripts/store_earnings_data.py --symbols AAPL,GOOGL,MSFT
"""

from datetime import date
from pathlib import Path

import argparse
import sys


# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import configuration first
from config import DATA_ROOT, LOGGING_CONFIG

import logging.config


# Set up centralized logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

from earnings.fetcher import EarningsCalendarFetcher, EarningsSource
from earnings.services.storage import EarningsPersistence


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Simple earnings data storage (no analysis)"
    )

    parser.add_argument(
        "--days-ahead",
        type=int,
        default=30,
        help="Days ahead to fetch earnings data (default: 30)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (default: all symbols)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["nasdaq", "fmp", "finnhub"],
        help="Single data source to use (default: try all)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh even if recent data exists",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show storage summary, don't collect new data",
    )
    parser.add_argument(
        "--use-current-dir",
        action="store_true",
        help="Use current directory instead of configured paths",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Write to isolated demo path under data/exports/demo",
    )

    args = parser.parse_args()

    print("🗄️ SIMPLE EARNINGS DATA STORAGE")
    print("=" * 50)
    print("📝 This script stores RAW earnings data without analysis")
    print("📝 No filtering, no strategy recommendations, just pure data")
    print("")

    # Initialize storage service
    use_config = not args.use_current_dir
    base_root = None
    if args.demo:
        from common.utils.demo_mode import demo_base

        base_root = demo_base()
        print(f"🔎 Demo mode: writing under {base_root}")

    persistence = EarningsPersistence(
        use_config_paths=use_config, base_data_root=base_root
    )

    # Show current storage summary
    print("📊 CURRENT STORAGE STATUS:")
    print("-" * 30)
    summary = persistence.get_storage_summary()

    if "error" not in summary:
        print(f"Collections: {summary['total_collections']}")
        print(f"Total events: {summary['total_events']}")
        print(f"Storage size: {summary['storage_size_mb']} MB")
        print(
            f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}"
        )
    else:
        print(f"Error getting summary: {summary['error']}")

    # Exit if only summary requested
    if args.summary_only:
        print("\n✅ Summary complete")
        return

    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        print(f"\n🎯 Filtering for symbols: {symbols}")

    # Store earnings data
    print(f"\n📥 COLLECTING EARNINGS DATA:")
    print("-" * 30)

    # Script is responsible for external fetch; library only persists
    fetcher = EarningsCalendarFetcher()
    sources = None
    if args.source:
        source_map = {
            "nasdaq": EarningsSource.NASDAQ,
            "fmp": EarningsSource.FMP,
            "finnhub": EarningsSource.FINNHUB,
        }
        sources = [source_map[args.source]]
    raw = fetcher.get_upcoming_earnings(
        symbols=symbols, days_ahead=args.days_ahead, sources=sources
    )
    meta = {
        "collection_type": "script_storage",
        "days_ahead": args.days_ahead,
        "symbols_filter": symbols,
        "sources_requested": [
            s.value
            for s in (
                sources
                or [EarningsSource.NASDAQ, EarningsSource.FMP, EarningsSource.FINNHUB]
            )
        ],
        "total_symbols": len(set(e.symbol for e in raw)) if raw else 0,
    }
    # Compute event topic (demo-safe) and persist + notify
    from common.utils.demo_mode import dataset_topic as topic_for

    topic = topic_for("earnings", demo=bool(args.demo))
    result = persistence.persist_earnings_batch(
        raw, collection_date=date.today(), metadata=meta, dataset_topic=topic
    )

    # Display results
    print(f"\n📊 STORAGE RESULTS:")
    print("-" * 30)
    print(f"Status: {result['status']}")

    if result["status"] == "success":
        print(f"Events collected: {result['events_collected']}")
        # Derive summary from fetched batch
        unique_symbols = meta.get("total_symbols", 0)
        print(f"Unique symbols: {unique_symbols}")
        if raw:
            start = min(e.earnings_date for e in raw).isoformat()
            end = max(e.earnings_date for e in raw).isoformat()
            print(f"Date range: {start} to {end}")
            sources_used = sorted(set(e.source for e in raw))
            print(f"Sources used: {sources_used}")

    elif result["status"] == "skipped":
        print(f"Reason: {result['reason']}")
        print(f"Data age: {result['hours_old']:.1f} hours")

    elif result["status"] == "no_data":
        print(f"No data from sources: {result['sources_attempted']}")

    elif result["status"] == "error":
        print(f"Error: {result['error']}")

    print(f"\n✅ Storage operation complete")


if __name__ == "__main__":
    main()
