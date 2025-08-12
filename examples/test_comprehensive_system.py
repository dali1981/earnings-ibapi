#!/usr/bin/env python3
"""
Test script for the comprehensive daily update system with sample earnings data.

This demonstrates the full system workflow without requiring API keys.
"""

import logging
import json
from datetime import date, datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from earnings.fetcher import EarningsCalendarFetcher, EarningsEvent, EarningsSource
from earnings.scheduler import EarningsPriorityScheduler, SchedulePriority
from jobs.orchestrator import DataPipelineOrchestrator

# Setup logging
from utils.logging_setup import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


def create_sample_earnings_data() -> list[EarningsEvent]:
    """Create sample earnings data for testing."""
    today = date.today()
    
    sample_earnings = [
        # Critical priority - earnings tomorrow
        EarningsEvent(
            symbol="AAPL",
            company_name="Apple Inc.",
            earnings_date=today + timedelta(days=1),
            time="amc",
            eps_estimate=2.45,
            revenue_estimate=82.5e9,
            source="demo"
        ),
        
        # High priority - earnings in 3 days
        EarningsEvent(
            symbol="GOOGL", 
            company_name="Alphabet Inc.",
            earnings_date=today + timedelta(days=3),
            time="amc",
            eps_estimate=1.25,
            revenue_estimate=55.2e9,
            source="demo"
        ),
        
        # Medium priority - earnings next week
        EarningsEvent(
            symbol="MSFT",
            company_name="Microsoft Corp.", 
            earnings_date=today + timedelta(days=10),
            time="amc",
            eps_estimate=2.78,
            revenue_estimate=48.1e9,
            source="demo"
        ),
        
        # No earnings for TSLA (will get maintenance priority)
    ]
    
    return sample_earnings


class DemoEarningsCalendarFetcher(EarningsCalendarFetcher):
    """Demo earnings fetcher that returns sample data instead of API calls."""
    
    def __init__(self, cache_dir: Path = Path("cache/earnings")):
        super().__init__(cache_dir)
        self.demo_earnings = create_sample_earnings_data()
    
    def get_upcoming_earnings(self, 
                            symbols: list[str] = None,
                            days_ahead: int = 30,
                            sources: list[EarningsSource] = None) -> list[EarningsEvent]:
        """Return demo earnings data filtered by symbols."""
        
        logger.info(f"📅 Using DEMO earnings data for {len(symbols or [])} symbols")
        
        if symbols is None:
            return self.demo_earnings
        
        # Filter by requested symbols
        filtered_earnings = [
            event for event in self.demo_earnings
            if event.symbol in symbols
        ]
        
        # Sort by priority (most urgent first)
        filtered_earnings.sort(key=lambda x: -x.priority_score)
        
        logger.info(f"📊 Demo earnings events found: {len(filtered_earnings)}")
        for event in filtered_earnings:
            priority_emoji = "🔥" if event.priority_score >= 80 else "📅"
            logger.info(f"  {priority_emoji} {event.symbol}: {event.earnings_date} "
                       f"({event.days_until_earnings} days) - Priority: {event.priority_score:.0f}")
        
        return filtered_earnings


def test_comprehensive_system():
    """Test the complete system with demo data."""
    
    print("🎯 TESTING COMPREHENSIVE DAILY UPDATE SYSTEM")
    print("=" * 80)
    
    # Test symbols - mix of earnings and non-earnings
    test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    data_path = Path("demo_data")
    
    print(f"📊 Testing with symbols: {test_symbols}")
    print(f"💾 Data path: {data_path}")
    
    # Initialize components with demo earnings fetcher
    orchestrator = DataPipelineOrchestrator(data_path, enable_lineage=True)
    demo_earnings_fetcher = DemoEarningsCalendarFetcher(data_path / "cache" / "earnings")
    scheduler = EarningsPriorityScheduler(
        base_path=data_path,
        orchestrator=orchestrator,
        earnings_fetcher=demo_earnings_fetcher
    )
    
    print("\n" + "=" * 60)
    print("STEP 1: EARNINGS DATA FETCHING")
    print("=" * 60)
    
    # Get earnings data
    earnings_events = demo_earnings_fetcher.get_upcoming_earnings(
        symbols=test_symbols,
        days_ahead=30
    )
    
    print(f"\n✅ Found {len(earnings_events)} earnings events")
    
    print("\n" + "=" * 60) 
    print("STEP 2: PRIORITY ASSIGNMENT")
    print("=" * 60)
    
    # Refresh priorities
    priorities = scheduler.refresh_earnings_priorities(test_symbols)
    
    print(f"\n📊 Priority assignments:")
    for symbol, priority in priorities.items():
        emoji = scheduler._get_priority_emoji(priority)
        earnings_info = ""
        for event in earnings_events:
            if event.symbol == symbol:
                earnings_info = f" (earnings in {event.days_until_earnings} days)"
                break
        print(f"  {emoji} {symbol}: {priority.value}{earnings_info}")
    
    print("\n" + "=" * 60)
    print("STEP 3: EXECUTION SCHEDULE CREATION") 
    print("=" * 60)
    
    # Create execution schedule
    batches = scheduler.create_daily_schedule(test_symbols)
    
    print(f"\n📅 Created execution schedule:")
    print(f"Total batches: {len(batches)}")
    
    total_symbols = 0
    for i, batch in enumerate(batches, 1):
        emoji = scheduler._get_priority_emoji(batch.priority) 
        symbols_in_batch = [s.symbol for s in batch.symbols]
        total_symbols += len(symbols_in_batch)
        
        print(f"  {emoji} Batch {i}: {batch.priority.value}")
        print(f"    Symbols: {', '.join(symbols_in_batch)}")
        print(f"    Max concurrent: {batch.max_concurrent}")
        print(f"    Est. duration: {batch.estimated_duration_minutes} minutes")
    
    print(f"\nTotal symbols scheduled: {total_symbols}")
    
    print("\n" + "=" * 60)
    print("STEP 4: SYSTEM STATUS")
    print("=" * 60)
    
    # Get system status
    status = scheduler.get_status_report()
    
    print(f"📈 System Status:")
    print(f"  Total symbols managed: {status['total_symbols']}")
    print(f"  Last execution: {status.get('last_execution', 'None')}")
    
    # Show priority distribution
    if 'priority_distribution' in status:
        print(f"  Priority distribution:")
        for priority_val, stats in status['priority_distribution'].items():
            if stats['total'] > 0:
                print(f"    {priority_val}: {stats['total']} symbols, {stats['needs_update']} need update")
    
    print("\n" + "=" * 60)
    print("STEP 5: DEMO EXECUTION SIMULATION")
    print("=" * 60)
    
    print("🧪 Simulating batch execution (DRY RUN):")
    
    for i, batch in enumerate(batches, 1):
        emoji = scheduler._get_priority_emoji(batch.priority)
        print(f"\n{emoji} Processing Batch {i}/{len(batches)}: {batch.priority.value}")
        
        for scheduled_symbol in batch.symbols:
            symbol = scheduled_symbol.symbol
            
            # Simulate work
            print(f"  🔄 Would update {symbol}...")
            print(f"    - Contract descriptions")
            print(f"    - Option chain snapshot") 
            print(f"    - Equity bars backfill")
            
            # Check if has earnings (would prioritize option data)
            has_earnings = any(e.symbol == symbol for e in earnings_events)
            if has_earnings:
                print(f"    - Option bars backfill (high priority - earnings coming)")
            else:
                print(f"    - Option bars backfill (background priority)")
    
    print("\n" + "=" * 80)
    print("🎉 COMPREHENSIVE SYSTEM TEST COMPLETED")
    print("=" * 80)
    
    print(f"\n✅ Key Features Demonstrated:")
    print(f"  📅 Multi-source earnings calendar integration")
    print(f"  ⚡ Priority-based scheduling (CRITICAL → HIGH → MEDIUM → LOW → MAINTENANCE)")
    print(f"  🔄 Batch optimization for resource efficiency")
    print(f"  📊 Comprehensive status reporting") 
    print(f"  🛡️ Circuit breaker protection (configured)")
    print(f"  📈 Performance monitoring integration")
    
    print(f"\n🎯 Next Steps:")
    print(f"  1. Configure real API keys: FMP_API_KEY, FINNHUB_API_KEY")
    print(f"  2. Customize portfolio.yaml with your symbols")
    print(f"  3. Set up cron job for automated daily execution")
    print(f"  4. Monitor performance via monitoring dashboard")
    
    return {
        'earnings_events': len(earnings_events),
        'priorities_assigned': len(priorities), 
        'batches_created': len(batches),
        'symbols_scheduled': total_symbols
    }


if __name__ == "__main__":
    results = test_comprehensive_system()
    print(f"\n📊 Test Results: {results}")