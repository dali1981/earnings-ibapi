#!/usr/bin/env python3
"""
Test Earnings-Driven Options Pipeline with Demo Data

This demonstrates the correct workflow for earnings-driven options trading
without requiring API keys.
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from earnings.fetcher import EarningsCalendarFetcher, EarningsEvent
from earnings_trading.discovery import EarningsDiscoveryEngine, EarningsCandidate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoEarningsDiscoveryEngine(EarningsDiscoveryEngine):
    """Demo earnings discovery with realistic market data."""
    
    def __init__(self, data_path: Path = Path("data")):
        super().__init__(data_path)
        self.demo_earnings = self._create_realistic_earnings_data()
    
    def _create_realistic_earnings_data(self) -> list[EarningsEvent]:
        """Create realistic earnings data for testing."""
        today = date.today()
        
        # Realistic upcoming earnings for major optionable stocks
        earnings_data = [
            # Critical/High priority (next week)
            {'symbol': 'AAPL', 'company': 'Apple Inc.', 'days': 2, 'time': 'amc'},
            {'symbol': 'GOOGL', 'company': 'Alphabet Inc.', 'days': 4, 'time': 'amc'},
            {'symbol': 'MSFT', 'company': 'Microsoft Corp.', 'days': 6, 'time': 'amc'},
            {'symbol': 'NVDA', 'company': 'NVIDIA Corp.', 'days': 3, 'time': 'amc'},
            {'symbol': 'TSLA', 'company': 'Tesla Inc.', 'days': 7, 'time': 'amc'},
            
            # Medium priority (2 weeks out)
            {'symbol': 'META', 'company': 'Meta Platforms Inc.', 'days': 10, 'time': 'amc'},
            {'symbol': 'AMZN', 'company': 'Amazon.com Inc.', 'days': 12, 'time': 'amc'},
            {'symbol': 'NFLX', 'company': 'Netflix Inc.', 'days': 14, 'time': 'amc'},
            {'symbol': 'AMD', 'company': 'Advanced Micro Devices', 'days': 11, 'time': 'amc'},
            {'symbol': 'CRM', 'company': 'Salesforce Inc.', 'days': 13, 'time': 'amc'},
            
            # Lower priority (3+ weeks out)
            {'symbol': 'BA', 'company': 'Boeing Co.', 'days': 18, 'time': 'bmo'},
            {'symbol': 'JPM', 'company': 'JPMorgan Chase & Co.', 'days': 21, 'time': 'bmo'},
            {'symbol': 'JNJ', 'company': 'Johnson & Johnson', 'days': 25, 'time': 'bmo'},
            {'symbol': 'PG', 'company': 'Procter & Gamble Co.', 'days': 28, 'time': 'bmo'},
            {'symbol': 'KO', 'company': 'Coca-Cola Co.', 'days': 32, 'time': 'bmo'},
            
            # High IV candidates (good for strangles/straddles)
            {'symbol': 'GME', 'company': 'GameStop Corp.', 'days': 5, 'time': 'amc'},
            {'symbol': 'AMC', 'company': 'AMC Entertainment Holdings', 'days': 8, 'time': 'amc'},
            {'symbol': 'BBBY', 'company': 'Bed Bath & Beyond Inc.', 'days': 9, 'time': 'amc'},
            
            # Calendar spread candidates (stable, good for time decay)
            {'symbol': 'SPY', 'company': 'SPDR S&P 500 ETF', 'days': 15, 'time': 'n/a'},
            {'symbol': 'QQQ', 'company': 'Invesco QQQ Trust', 'days': 16, 'time': 'n/a'},
            {'symbol': 'IWM', 'company': 'iShares Russell 2000 ETF', 'days': 17, 'time': 'n/a'},
        ]
        
        events = []
        for item in earnings_data:
            earnings_date = today + timedelta(days=item['days'])
            
            event = EarningsEvent(
                symbol=item['symbol'],
                company_name=item['company'],
                earnings_date=earnings_date,
                time=item['time'],
                eps_estimate=2.50 if item['symbol'] not in ['SPY', 'QQQ', 'IWM'] else None,
                source="demo_market_data"
            )
            events.append(event)
        
        return events
    
    def discover_earnings_opportunities(self, 
                                      days_ahead: int = 21,
                                      min_score: float = 50.0) -> list[EarningsCandidate]:
        """Override to use demo data."""
        
        logger.info(f"🎯 Using demo earnings data for market discovery")
        
        # Filter by days ahead
        filtered_earnings = [
            event for event in self.demo_earnings
            if event.days_until_earnings <= days_ahead
        ]
        
        logger.info(f"📅 Found {len(filtered_earnings)} demo earnings events")
        
        # Convert to candidates and score
        candidates = []
        for event in filtered_earnings:
            candidate = self._create_candidate_from_earnings(event)
            if self._passes_basic_filters(candidate):
                self._score_options_strategies(candidate)
                
                # Enhance scoring with realistic market characteristics
                candidate = self._enhance_with_market_data(candidate)
                
                if candidate.total_score >= min_score:
                    candidates.append(candidate)
        
        # Sort by best opportunities
        candidates.sort(key=lambda x: -x.total_score)
        
        logger.info(f"🎯 {len(candidates)} high-quality candidates found")
        
        return candidates
    
    def _enhance_with_market_data(self, candidate: EarningsCandidate) -> EarningsCandidate:
        """Add realistic market characteristics for better scoring."""
        
        # Simulate market characteristics based on symbol type
        if candidate.symbol in ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']:
            # Large cap tech - excellent for all strategies
            candidate.calendar_score *= 1.2
            candidate.straddle_score *= 1.1
            candidate.strangle_score *= 1.1
            candidate.has_weekly_options = True
            candidate.has_liquid_options = True
            
        elif candidate.symbol in ['GME', 'AMC', 'BBBY']:
            # High volatility - great for volatility plays, poor for calendar
            candidate.calendar_score *= 0.6
            candidate.straddle_score *= 1.4
            candidate.strangle_score *= 1.3
            candidate.has_weekly_options = True
            candidate.has_liquid_options = True
            
        elif candidate.symbol in ['SPY', 'QQQ', 'IWM']:
            # ETFs - excellent for calendar spreads, moderate for volatility
            candidate.calendar_score *= 1.4
            candidate.straddle_score *= 0.8
            candidate.strangle_score *= 0.9
            candidate.has_weekly_options = True
            candidate.has_liquid_options = True
            
        else:
            # Large cap value - good for all strategies
            candidate.calendar_score *= 1.0
            candidate.straddle_score *= 1.0
            candidate.strangle_score *= 1.0
            candidate.has_weekly_options = True
            candidate.has_liquid_options = True
        
        # Set data availability
        candidate.option_chain_available = False  # Will be collected
        candidate.historical_iv_available = False  # Will be analyzed
        
        return candidate


def main():
    """Test the earnings-driven pipeline with demo data."""
    
    print("🎯 EARNINGS-DRIVEN OPTIONS PIPELINE DEMO")
    print("=" * 70)
    
    # Initialize demo discovery engine
    demo_engine = DemoEarningsDiscoveryEngine()
    
    print(f"📊 Pipeline Configuration:")
    print(f"   Market discovery: Demo mode (21+ realistic earnings)")
    print(f"   Options strategies: Calendar spreads, strangles, straddles")
    print(f"   Scoring: Based on timing, volatility, and liquidity")
    
    # Discover opportunities
    print(f"\n{'='*60}")
    print("EARNINGS OPPORTUNITY DISCOVERY")
    print(f"{'='*60}")
    
    opportunities = demo_engine.discover_earnings_opportunities(
        days_ahead=35,
        min_score=30.0  # Lower threshold for demo
    )
    
    if not opportunities:
        print("❌ No opportunities found")
        return
    
    print(f"✅ Found {len(opportunities)} earnings opportunities")
    
    # Show top opportunities
    print(f"\n🎯 TOP OPPORTUNITIES:")
    print(f"{'#':<3} {'Symbol':<6} {'Earnings':<12} {'Days':<5} {'Score':<5} {'Strategy':<15} {'Notes'}")
    print("-" * 70)
    
    for i, opp in enumerate(opportunities, 1):
        strategy = opp.best_strategy.value if opp.best_strategy else "none"
        
        # Add notes based on characteristics
        notes = []
        if opp.has_weekly_options:
            notes.append("weekly")
        if opp.symbol in ['AAPL', 'GOOGL', 'MSFT']:
            notes.append("high-liquid")
        elif opp.symbol in ['GME', 'AMC']:
            notes.append("high-vol")
        elif opp.symbol in ['SPY', 'QQQ']:
            notes.append("stable-etf")
        
        notes_str = ", ".join(notes) if notes else ""
        
        print(f"{i:<3} {opp.symbol:<6} {opp.earnings_date.strftime('%Y-%m-%d'):<12} "
              f"{opp.days_until_earnings:>3}d  {opp.total_score:>3.0f}  {strategy:<15} {notes_str}")
    
    # Strategy distribution analysis
    print(f"\n📊 STRATEGY DISTRIBUTION:")
    strategy_counts = {}
    for opp in opportunities:
        if opp.best_strategy:
            strategy_counts[opp.best_strategy.value] = strategy_counts.get(opp.best_strategy.value, 0) + 1
    
    for strategy, count in strategy_counts.items():
        print(f"   {strategy}: {count} opportunities")
    
    # Timing analysis
    print(f"\n⏰ TIMING ANALYSIS:")
    critical = [o for o in opportunities if o.days_until_earnings <= 3]
    high = [o for o in opportunities if 3 < o.days_until_earnings <= 7]
    medium = [o for o in opportunities if 7 < o.days_until_earnings <= 14]
    low = [o for o in opportunities if o.days_until_earnings > 14]
    
    print(f"   🔥 Critical (≤3 days): {len(critical)} opportunities")
    if critical:
        symbols = [o.symbol for o in critical]
        print(f"      Symbols: {', '.join(symbols)}")
    
    print(f"   ⚡ High (4-7 days): {len(high)} opportunities")
    if high:
        symbols = [o.symbol for o in high]
        print(f"      Symbols: {', '.join(symbols)}")
    
    print(f"   📊 Medium (8-14 days): {len(medium)} opportunities")
    if medium:
        symbols = [o.symbol for o in medium]
        print(f"      Symbols: {', '.join(symbols)}")
    
    print(f"   📅 Low (>14 days): {len(low)} opportunities")
    
    # Export opportunities
    export_file = demo_engine.export_opportunities(opportunities)
    print(f"\n💾 Opportunities exported to: {export_file}")
    
    # Show data collection priorities
    data_priorities = demo_engine.get_data_collection_list(opportunities)
    
    print(f"\n📋 DATA COLLECTION PRIORITIES:")
    total_symbols = 0
    for priority, symbols in data_priorities.items():
        if symbols:
            total_symbols += len(symbols)
            print(f"   {priority.upper()}: {len(symbols)} symbols")
            print(f"      {', '.join(symbols)}")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. 📊 Review exported opportunities: {export_file}")
    print(f"2. 🔄 Run data collection for {total_symbols} priority symbols")
    print(f"3. 📈 Analyze IV structure for top candidates")
    print(f"4. 💰 Execute options strategies based on analysis")
    print(f"5. ⏰ Monitor positions through earnings events")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"💡 To use with real data: set FMP_API_KEY and FINNHUB_API_KEY environment variables")


if __name__ == "__main__":
    main()