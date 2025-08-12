#!/usr/bin/env python3
"""
Earnings-Driven Options Trading Pipeline

This is the main entry point for earnings-based options trading.
Replaces fixed portfolio management with dynamic opportunity discovery.

Workflow:
1. Discover ALL upcoming earnings across the market
2. Score opportunities for options strategies (calendar spreads, strangles, straddles)
3. Collect data for high-scoring candidates
4. Export results for strategy analysis

Usage:
    # Discover and collect data for earnings opportunities
    python jobs/run_earnings_pipeline.py
    
    # Custom parameters
    python jobs/run_earnings_pipeline.py --days 14 --min-score 60 --max-symbols 25
    
    # Export opportunities only (no data collection)
    python jobs/run_earnings_pipeline.py --discover-only
"""

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from earnings_trading.discovery import EarningsDiscoveryEngine
from earnings_trading.data_pipeline import EarningsDataPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'earnings_pipeline_{date.today().isoformat()}.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main earnings trading pipeline."""
    
    parser = argparse.ArgumentParser(
        description="Earnings-driven options trading data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard pipeline - discover and collect data
    python jobs/run_earnings_pipeline.py
    
    # Focus on near-term opportunities
    python jobs/run_earnings_pipeline.py --days 10 --min-score 70
    
    # High-volume processing
    python jobs/run_earnings_pipeline.py --max-symbols 100
    
    # Discovery only - no data collection
    python jobs/run_earnings_pipeline.py --discover-only
    
    # Export results to specific location
    python jobs/run_earnings_pipeline.py --output exports/my_opportunities.csv
        """
    )
    
    # Discovery parameters
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=21,
        help="Days ahead to look for earnings (default: 21)"
    )
    
    parser.add_argument(
        "--min-score",
        type=float,
        default=50.0,
        help="Minimum opportunity score (default: 50.0)"
    )
    
    # Data collection parameters
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=50,
        help="Maximum symbols to process (default: 50)"
    )
    
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover opportunities, don't collect data"
    )
    
    # Output parameters
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Custom output file for opportunities"
    )
    
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data"),
        help="Base data path (default: data)"
    )
    
    parser.add_argument(
        "--export-status",
        action="store_true",
        help="Export data collection status"
    )
    
    args = parser.parse_args()
    
    print("🎯 EARNINGS-DRIVEN OPTIONS TRADING PIPELINE")
    print("=" * 70)
    
    try:
        # Initialize pipeline components
        discovery_engine = EarningsDiscoveryEngine(args.data_path)
        
        if not args.discover_only:
            data_pipeline = EarningsDataPipeline(args.data_path, discovery_engine)
        
        print(f"📊 Pipeline Parameters:")
        print(f"   Lookhead period: {args.days} days")
        print(f"   Minimum score: {args.min_score}")
        print(f"   Max symbols: {args.max_symbols}")
        print(f"   Data collection: {'disabled' if args.discover_only else 'enabled'}")
        
        # Phase 1: Discover Earnings Opportunities
        print(f"\n{'='*60}")
        print("PHASE 1: EARNINGS OPPORTUNITY DISCOVERY")
        print(f"{'='*60}")
        
        opportunities = discovery_engine.discover_earnings_opportunities(
            days_ahead=args.days,
            min_score=args.min_score
        )
        
        if not opportunities:
            print("❌ No earnings opportunities found")
            print("💡 Possible reasons:")
            print("   • Missing API keys (FMP_API_KEY, FINNHUB_API_KEY)")
            print("   • No earnings in the specified timeframe")
            print("   • Minimum score too high")
            sys.exit(1)
        
        print(f"✅ Found {len(opportunities)} earnings opportunities")
        
        # Show top opportunities
        print(f"\n🎯 TOP OPPORTUNITIES:")
        for i, opp in enumerate(opportunities[:10], 1):
            strategy = opp.best_strategy.value if opp.best_strategy else "none"
            print(f"{i:2d}. {opp.symbol:<6} - {opp.earnings_date} "
                  f"({opp.days_until_earnings:+2d} days) - "
                  f"{opp.total_score:3.0f} pts - {strategy}")
        
        # Export opportunities
        output_file = args.output or args.data_path / "exports" / f"earnings_opportunities_{date.today().isoformat()}.csv"
        discovery_engine.export_opportunities(opportunities, output_file)
        
        print(f"\n💾 Opportunities exported to: {output_file}")
        
        # Phase 2: Data Collection (if enabled)
        if not args.discover_only:
            print(f"\n{'='*60}")
            print("PHASE 2: OPTIONS DATA COLLECTION")
            print(f"{'='*60}")
            
            # Limit opportunities for data collection
            collection_candidates = opportunities[:args.max_symbols]
            print(f"📊 Processing top {len(collection_candidates)} candidates for data collection")
            
            # Run data collection
            collection_results = data_pipeline.run_earnings_driven_collection(
                days_ahead=args.days,
                min_opportunity_score=args.min_score,
                max_symbols=args.max_symbols
            )
            
            # Show results
            print(f"\n📈 DATA COLLECTION RESULTS:")
            print(f"   Tasks created: {collection_results['tasks_created']}")
            print(f"   Tasks completed: {collection_results['tasks_completed']}")
            print(f"   Symbols processed: {collection_results['symbols_processed']}")
            
            # Export collection status
            if args.export_status or collection_results['tasks_created'] > 0:
                status_file = data_pipeline.export_collection_status()
                print(f"   Status exported: {status_file}")
            
            # Show ready for analysis
            ready_tasks = data_pipeline.get_ready_for_analysis()
            if ready_tasks:
                print(f"\n🎯 READY FOR STRATEGY ANALYSIS: {len(ready_tasks)} symbols")
                for task in ready_tasks[:5]:
                    strategies = ', '.join(s.value for s in task.strategies_needed)
                    print(f"   {task.symbol}: {task.completion_percentage:.0f}% complete, "
                          f"earnings in {task.days_until_earnings} days, "
                          f"strategies: {strategies}")
        
        # Phase 3: Summary and Next Steps
        print(f"\n{'='*60}")
        print("PHASE 3: SUMMARY AND NEXT STEPS")
        print(f"{'='*60}")
        
        # Categorize by strategy
        strategy_counts = {}
        for opp in opportunities:
            if opp.best_strategy:
                strategy_counts[opp.best_strategy.value] = strategy_counts.get(opp.best_strategy.value, 0) + 1
        
        print(f"🎯 Strategy Distribution:")
        for strategy, count in strategy_counts.items():
            print(f"   {strategy}: {count} opportunities")
        
        # Urgency analysis
        critical = [o for o in opportunities if o.days_until_earnings <= 3]
        high = [o for o in opportunities if 3 < o.days_until_earnings <= 7]
        medium = [o for o in opportunities if 7 < o.days_until_earnings <= 14]
        
        print(f"\n⏰ Timing Analysis:")
        print(f"   🔥 Critical (≤3 days): {len(critical)} opportunities")
        print(f"   ⚡ High (4-7 days): {len(high)} opportunities")
        print(f"   📊 Medium (8-14 days): {len(medium)} opportunities")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Review opportunities in: {output_file}")
        print(f"2. Analyze IV structure for top candidates")
        print(f"3. Set up options positions for selected strategies")
        print(f"4. Monitor positions through earnings events")
        
        if args.discover_only:
            print(f"5. Run data collection: python jobs/run_earnings_pipeline.py")
        else:
            print(f"5. Pipeline complete - ready for trading decisions")
        
        print(f"\n🎉 Earnings pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Pipeline failed with unexpected error")
        print(f"\n💥 Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()