#!/usr/bin/env python3
"""
Demonstration of NASDAQ Data Persistence System

This script demonstrates the complete persistence workflow:
1. Data collection from NASDAQ API
2. Storage in partitioned Parquet files
3. Data retrieval and querying
4. System statistics and health monitoring
"""

from datetime import date, timedelta
from pathlib import Path

import logging
import sys


# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from common.repositories.earnings import EarningsCalendarRepository
from config import EARNINGS_PATH
from earnings.fetcher import EarningsCalendarFetcher, EarningsSource
from jobs.daily_earnings_collection import DailyEarningsCollector


def demonstrate_persistence(demo: bool = False):
    """Demonstrate the complete persistence system."""

    from common.utils.logging_setup import setup_logging

    setup_logging()

    print("🗄️ NASDAQ EARNINGS DATA PERSISTENCE DEMONSTRATION")
    print("=" * 70)

    # Initialize components
    if demo:
        from common.utils.demo_mode import demo_base

        base = demo_base()
        print(f"🔎 Demo mode: using base path {base}")
        repo = EarningsCalendarRepository(base)
    else:
        repo = EarningsCalendarRepository()

    print("\n1. 📊 CURRENT STORAGE STATISTICS")
    print("-" * 40)

    stats = repo.get_storage_stats()
    print(f"   Total collections: {stats['total_collections']}")
    print(f"   Total events: {stats['total_events']}")
    print(f"   Storage size: {stats['total_size_mb']} MB")
    print(f"   Unique symbols: {stats['unique_symbols']}")
    print(
        f"   Date range: {stats['date_range']['start']} to {stats['date_range']['end']}"
    )
    print(f"   Sources: {list(stats['sources'].keys())}")

    print("\n2. 🔍 LATEST COLLECTION DETAILS")
    print("-" * 40)

    latest_data = repo.get_latest_collection()
    if latest_data is not None:
        print(f"   Events in latest collection: {len(latest_data)}")
        print(f"   Collection date: {latest_data['collection_date'].iloc[0]}")
        print(f"   Sources: {latest_data['source'].unique().tolist()}")

        # Show sample earnings events
        print(f"\n   📅 Sample upcoming earnings:")
        upcoming = latest_data[latest_data["days_until_earnings"] >= 0].head(5)
        for _, row in upcoming.iterrows():
            days = row["days_until_earnings"]
            timing = row["time"] or "unknown"
            print(
                f"      {row['symbol']:<6} - {row['earnings_date']} (+{days} days) - {timing}"
            )

        # Show summary by days until earnings
        print(f"\n   📊 Earnings distribution by timing:")
        timing_dist = latest_data["days_until_earnings"].value_counts().sort_index()
        for days, count in timing_dist.head(5).items():
            if days >= 0:
                print(f"      In {days} days: {count} companies")

    else:
        print("   No data collections found")

        print("\n3. 🚀 COLLECTING FRESH DATA")
        print("-" * 40)

        # Collect some fresh data
        collector = DailyEarningsCollector()
        result = collector.run_daily_collection(days_ahead=7, force_refresh=True)

        print(f"   Collection status: {result.get('status', 'unknown')}")
        print(f"   Events collected: {result.get('total_events_collected', 0)}")
        print(f"   Data quality: {result.get('data_quality_score', 0):.1%}")
        print(f"   Processing time: {result.get('processing_time_seconds', 0):.1f}s")

        # Refresh stats
        stats = repo.get_storage_stats()
        print(f"   Updated total events: {stats['total_events']}")

    print("\n4. 🎯 DATA PERSISTENCE BENEFITS")
    print("-" * 40)
    print("   ✅ Persistent storage - Data survives restarts")
    print("   ✅ Partitioned by date and source - Efficient querying")
    print("   ✅ Parquet format - Compressed, columnar storage")
    print("   ✅ Metadata tracking - Data lineage and quality metrics")
    print("   ✅ Automatic cleanup - Retention policies prevent bloat")
    print("   ✅ Multi-source support - NASDAQ, FMP, Finnhub integration")
    print("   ✅ Circuit breakers - Resilient data collection")

    print("\n5. 📁 STORAGE STRUCTURE")
    print("-" * 40)

    # Show directory structure using configured path
    data_path = EARNINGS_PATH
    if data_path.exists():
        print(f"   Storage path: {data_path}")

        for collection_dir in sorted(data_path.iterdir()):
            if collection_dir.is_dir() and collection_dir.name.startswith(
                "collection_date="
            ):
                collection_date = collection_dir.name.replace("collection_date=", "")
                print(f"   📅 {collection_date}/")

                for source_dir in sorted(collection_dir.iterdir()):
                    if source_dir.is_dir() and source_dir.name.startswith("source="):
                        source_name = source_dir.name.replace("source=", "")
                        parquet_files = list(source_dir.glob("*.parquet"))
                        total_size = sum(f.stat().st_size for f in parquet_files)
                        print(
                            f"       └── 📊 {source_name}/ ({len(parquet_files)} files, {total_size/1024:.1f}KB)"
                        )

    print("\n6. 🔄 AUTOMATED COLLECTION WORKFLOW")
    print("-" * 40)
    print("   The daily collection job can be scheduled with:")
    print("   • Cron job: '0 6 * * * python jobs/daily_earnings_collection.py'")
    print("   • Systemd timer")
    print("   • Task scheduler")
    print("   • Docker container with restart policies")
    print("")
    print("   This ensures fresh earnings data every day automatically!")

    print("\n✅ PERSISTENCE SYSTEM FULLY OPERATIONAL")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Use isolated demo base")
    args = parser.parse_args()
    demonstrate_persistence(demo=args.demo)
