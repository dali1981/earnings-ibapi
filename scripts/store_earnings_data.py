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

import sys
import argparse
from datetime import date
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import configuration first
from config import LOGGING_CONFIG, DATA_ROOT
import logging.config

# Set up centralized logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

from earnings.fetcher import EarningsCalendarFetcher, EarningsSource
from repositories.earnings import EarningsRepository


class SimpleEarningsStorageService:
    """Simple service to fetch and store earnings data without analysis."""
    
    def __init__(self, use_config_paths: bool = True):
        """Initialize storage service with configurable paths."""
        
        if use_config_paths:
            # Use centralized config paths
            earnings_path = DATA_ROOT / "earnings"
            logger.info(f"📁 Using configured data path: {earnings_path}")
        else:
            # Fallback to current directory (old behavior)
            earnings_path = Path("data/earnings")
            logger.warning(f"📁 Using fallback data path: {earnings_path}")
        
        self.fetcher = EarningsCalendarFetcher()
        self.repository = EarningsRepository(earnings_path.parent)  # Pass base path
        
        logger.info("🗄️ Simple earnings storage service initialized")
    
    def store_earnings_data(self,
                          days_ahead: int = 30,
                          symbols: list = None,
                          sources: list = None,
                          force_refresh: bool = False) -> dict:
        """
        Fetch and store earnings data without any filtering or analysis.
        
        Args:
            days_ahead: Number of days ahead to fetch data
            symbols: Optional list of symbols to fetch (None = all symbols)
            sources: Optional list of sources (None = default sources)
            force_refresh: Force new collection even if recent data exists
            
        Returns:
            Dictionary with storage results
        """
        
        logger.info(f"📊 Fetching earnings data for next {days_ahead} days")
        
        # Check if we should skip due to recent data
        if not force_refresh:
            latest_data = self.repository.get_latest_collection()
            if latest_data is not None:
                collection_date = latest_data['collection_date'].iloc[0]
                if isinstance(collection_date, str):
                    from datetime import datetime
                    collection_date = datetime.strptime(collection_date, '%Y-%m-%d').date()
                
                hours_old = (date.today() - collection_date).total_seconds() / 3600
                if hours_old < 6:  # Less than 6 hours old
                    logger.info(f"⏭️ Skipping: Recent data exists ({hours_old:.1f} hours old)")
                    return {
                        "status": "skipped",
                        "reason": "Recent data available",
                        "hours_old": hours_old
                    }
        
        # Set default sources if none provided
        if sources is None:
            sources = [EarningsSource.NASDAQ, EarningsSource.FMP, EarningsSource.FINNHUB]
        elif isinstance(sources, str):
            # Convert string to enum
            source_map = {
                'nasdaq': EarningsSource.NASDAQ,
                'fmp': EarningsSource.FMP,
                'finnhub': EarningsSource.FINNHUB
            }
            sources = [source_map.get(sources.lower(), EarningsSource.NASDAQ)]
        
        # Fetch raw earnings data (no filtering)
        try:
            raw_earnings = self.fetcher.get_upcoming_earnings(
                symbols=symbols,
                days_ahead=days_ahead,
                sources=sources
            )
            
            if not raw_earnings:
                logger.warning("⚠️ No earnings data retrieved from any source")
                return {
                    "status": "no_data",
                    "events_collected": 0,
                    "sources_attempted": [s.value for s in sources]
                }
            
            # Store raw data directly (no filtering, no analysis)
            storage_result = self.repository.store_earnings_batch(
                earnings=raw_earnings,
                collection_date=date.today(),
                metadata={
                    "collection_type": "simple_storage",
                    "days_ahead": days_ahead,
                    "symbols_filter": symbols,
                    "sources_requested": [s.value for s in sources],
                    "total_symbols": len(set(e.symbol for e in raw_earnings))
                }
            )
            
            # Return simple results
            result = {
                "status": "success",
                "collection_date": date.today().isoformat(),
                "events_collected": len(raw_earnings),
                "unique_symbols": len(set(e.symbol for e in raw_earnings)),
                "sources_successful": list(set(e.source for e in raw_earnings)),
                "date_range": {
                    "start": min(e.earnings_date for e in raw_earnings).isoformat(),
                    "end": max(e.earnings_date for e in raw_earnings).isoformat()
                },
                "storage": storage_result
            }
            
            logger.info(f"✅ Successfully stored {len(raw_earnings)} earnings events")
            logger.info(f"   Unique symbols: {result['unique_symbols']}")
            logger.info(f"   Date range: {result['date_range']['start']} to {result['date_range']['end']}")
            logger.info(f"   Sources: {result['sources_successful']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Storage operation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg
            }
    
    def get_storage_summary(self) -> dict:
        """Get summary of currently stored data."""
        try:
            stats = self.repository.get_storage_stats()
            
            summary = {
                "total_collections": stats["total_collections"],
                "total_events": stats["total_events"],
                "storage_size_mb": stats["total_size_mb"],
                "unique_symbols": stats["unique_symbols"],
                "date_range": stats["date_range"],
                "sources": stats["sources"]
            }
            
            logger.info(f"📊 Storage Summary:")
            logger.info(f"   Collections: {summary['total_collections']}")
            logger.info(f"   Total events: {summary['total_events']}")
            logger.info(f"   Storage size: {summary['storage_size_mb']} MB")
            logger.info(f"   Unique symbols: {summary['unique_symbols']}")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get storage summary: {e}")
            return {"error": str(e)}


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Simple earnings data storage (no analysis)")
    
    parser.add_argument('--days-ahead', type=int, default=30,
                       help='Days ahead to fetch earnings data (default: 30)')
    parser.add_argument('--symbols', type=str,
                       help='Comma-separated list of symbols (default: all symbols)')
    parser.add_argument('--source', type=str, choices=['nasdaq', 'fmp', 'finnhub'],
                       help='Single data source to use (default: try all)')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh even if recent data exists')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only show storage summary, don\'t collect new data')
    parser.add_argument('--use-current-dir', action='store_true',
                       help='Use current directory instead of configured paths')
    
    args = parser.parse_args()
    
    print("🗄️ SIMPLE EARNINGS DATA STORAGE")
    print("=" * 50)
    print("📝 This script stores RAW earnings data without analysis")
    print("📝 No filtering, no strategy recommendations, just pure data")
    print("")
    
    # Initialize storage service
    use_config = not args.use_current_dir
    storage_service = SimpleEarningsStorageService(use_config_paths=use_config)
    
    # Show current storage summary
    print("📊 CURRENT STORAGE STATUS:")
    print("-" * 30)
    summary = storage_service.get_storage_summary()
    
    if "error" not in summary:
        print(f"Collections: {summary['total_collections']}")
        print(f"Total events: {summary['total_events']}")
        print(f"Storage size: {summary['storage_size_mb']} MB")
        print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    else:
        print(f"Error getting summary: {summary['error']}")
    
    # Exit if only summary requested
    if args.summary_only:
        print("\n✅ Summary complete")
        return
    
    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
        print(f"\n🎯 Filtering for symbols: {symbols}")
    
    # Store earnings data
    print(f"\n📥 COLLECTING EARNINGS DATA:")
    print("-" * 30)
    
    result = storage_service.store_earnings_data(
        days_ahead=args.days_ahead,
        symbols=symbols,
        sources=args.source,
        force_refresh=args.force_refresh
    )
    
    # Display results
    print(f"\n📊 STORAGE RESULTS:")
    print("-" * 30)
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"Events collected: {result['events_collected']}")
        print(f"Unique symbols: {result['unique_symbols']}")
        print(f"Date range: {result['date_range']['start']} to {result['date_range']['end']}")
        print(f"Sources used: {result['sources_successful']}")
        
    elif result['status'] == 'skipped':
        print(f"Reason: {result['reason']}")
        print(f"Data age: {result['hours_old']:.1f} hours")
        
    elif result['status'] == 'no_data':
        print(f"No data from sources: {result['sources_attempted']}")
        
    elif result['status'] == 'error':
        print(f"Error: {result['error']}")
    
    print(f"\n✅ Storage operation complete")


if __name__ == "__main__":
    main()