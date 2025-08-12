#!/usr/bin/env python3
"""
Earnings Discovery System

Discovers ALL upcoming earnings across the market and identifies candidates
for options trading strategies (calendar spreads, strangles, straddles) based
on IV characteristics and earnings timing.

This is the first step in the earnings options trading workflow:
1. Discover ALL earnings (this module)
2. Analyze IV structure for each candidate
3. Determine optimal options strategies
4. Execute data collection for selected symbols
"""

import logging
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Import our earnings fetcher
from earnings.fetcher import EarningsCalendarFetcher, EarningsEvent

logger = logging.getLogger(__name__)


class OptionsStrategy(Enum):
    """Potential options strategies for earnings plays."""
    CALENDAR_SPREAD = "calendar_spread"
    STRADDLE = "straddle" 
    STRANGLE = "strangle"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    COVERED_CALL = "covered_call"


@dataclass
class EarningsCandidate:
    """A stock with upcoming earnings suitable for options strategies."""
    symbol: str
    company_name: str
    earnings_date: date
    time: str  # bmo, amc, dmt
    days_until_earnings: int
    
    # Market characteristics  
    market_cap: Optional[float] = None
    avg_volume: Optional[int] = None
    price: Optional[float] = None
    eps_estimate: Optional[float] = None
    
    # Options characteristics
    has_weekly_options: bool = False
    has_liquid_options: bool = False
    iv_rank: Optional[float] = None
    iv_percentile: Optional[float] = None
    
    # Strategy suitability scores (0-100)
    calendar_score: float = 0.0
    straddle_score: float = 0.0
    strangle_score: float = 0.0
    
    # Data availability
    option_chain_available: bool = False
    historical_iv_available: bool = False
    
    @property
    def total_score(self) -> float:
        """Overall trading opportunity score."""
        return max(self.calendar_score, self.straddle_score, self.strangle_score)
    
    @property
    def best_strategy(self) -> Optional[OptionsStrategy]:
        """Best options strategy for this candidate."""
        scores = {
            OptionsStrategy.CALENDAR_SPREAD: self.calendar_score,
            OptionsStrategy.STRADDLE: self.straddle_score,
            OptionsStrategy.STRANGLE: self.strangle_score
        }
        
        best_strategy = max(scores, key=scores.get)
        if scores[best_strategy] > 50:  # Threshold for viable strategy
            return best_strategy
        return None


class EarningsDiscoveryEngine:
    """Discovers and analyzes ALL market earnings for options trading opportunities."""
    
    def __init__(self, data_path: Path = Path("data")):
        self.data_path = data_path
        self.earnings_fetcher = EarningsCalendarFetcher()
        
        # Filtering criteria for options trading
        self.min_market_cap = 1e9      # $1B minimum market cap
        self.min_avg_volume = 1e6      # 1M shares average volume
        self.min_price = 10.0          # $10 minimum price
        self.max_price = 1000.0        # $1000 maximum price
        
        # Options requirements
        self.require_weekly_options = True
        self.require_liquid_options = True
        self.min_iv_rank = 20          # Minimum IV rank for trades
        
        # Timing windows
        self.min_days_ahead = 1        # At least 1 day ahead
        self.max_days_ahead = 60       # Within 2 months (flexible)
        
        logger.info(f"📊 Initialized earnings discovery engine with filters:")
        logger.info(f"   Market cap: ${self.min_market_cap/1e9:.1f}B+")
        logger.info(f"   Volume: {self.min_avg_volume/1e6:.1f}M+ shares")
        logger.info(f"   Price range: ${self.min_price} - ${self.max_price}")
        logger.info(f"   Time window: {self.min_days_ahead}-{self.max_days_ahead} days")
    
    def discover_earnings_opportunities(self, 
                                      days_ahead: int = 21,
                                      min_score: float = 50.0) -> List[EarningsCandidate]:
        """
        Discover ALL earnings in the market and filter for options trading opportunities.
        
        This is the main entry point that replaces fixed portfolio management.
        """
        logger.info(f"🔍 Discovering ALL market earnings for next {days_ahead} days")
        
        # Step 1: Get ALL earnings events (no symbol filter)
        all_earnings = self.earnings_fetcher.get_upcoming_earnings(
            symbols=None,  # No filter - get everything
            days_ahead=days_ahead
        )
        
        logger.info(f"📅 Found {len(all_earnings)} total earnings events")
        
        if not all_earnings:
            logger.warning("❌ No earnings data available - check API keys")
            return []
        
        # Step 2: Convert to candidates and apply basic filters
        candidates = []
        for event in all_earnings:
            candidate = self._create_candidate_from_earnings(event)
            if self._passes_basic_filters(candidate):
                candidates.append(candidate)
        
        logger.info(f"📊 {len(candidates)} candidates passed basic filters")
        
        # Step 3: Score candidates for options strategies
        scored_candidates = []
        for candidate in candidates:
            self._score_options_strategies(candidate)
            if candidate.total_score >= min_score:
                scored_candidates.append(candidate)
        
        # Step 4: Sort by best opportunities
        scored_candidates.sort(key=lambda x: -x.total_score)
        
        logger.info(f"🎯 {len(scored_candidates)} high-quality candidates found")
        
        return scored_candidates
    
    def _create_candidate_from_earnings(self, event: EarningsEvent) -> EarningsCandidate:
        """Convert earnings event to trading candidate."""
        candidate = EarningsCandidate(
            symbol=event.symbol,
            company_name=event.company_name,
            earnings_date=event.earnings_date,
            time=event.time or 'unknown',
            days_until_earnings=event.days_until_earnings,
            eps_estimate=event.eps_estimate
        )
        
        # Pass through market cap if available (from NASDAQ data)
        if hasattr(event, 'market_cap') and event.market_cap:
            candidate.market_cap = event.market_cap
        
        return candidate
    
    def _passes_basic_filters(self, candidate: EarningsCandidate) -> bool:
        """Apply basic filtering criteria for options trading suitability."""
        
        # Timing filter
        if candidate.days_until_earnings < self.min_days_ahead:
            return False
        if candidate.days_until_earnings > self.max_days_ahead:
            return False
        
        # Market cap filter (if available from NASDAQ data)
        if hasattr(candidate, 'market_cap') and candidate.market_cap:
            if candidate.market_cap < self.min_market_cap:
                return False
        
        # Basic symbol validation for options trading
        symbol = candidate.symbol
        
        # Skip symbols that are unlikely to have liquid options
        if len(symbol) > 5:  # Very long symbols often don't have options
            return False
        
        # Skip symbols with special characters that indicate warrants, rights, etc.
        if any(char in symbol for char in ['.', '-', '/', '+', '~']):
            return False
        
        # Skip obvious penny stocks (very short, unusual symbols)
        if len(symbol) <= 1:
            return False
        
        # Basic price filter (if we had price data)
        # For now, use market cap as a proxy for reasonable price range
        if hasattr(candidate, 'market_cap') and candidate.market_cap:
            # Very small market cap likely means penny stock
            if candidate.market_cap < 100e6:  # $100M minimum
                return False
            # Very large market cap might have low volatility  
            if candidate.market_cap > 2000e9:  # $2T maximum (AAPL, MSFT level)
                # Still include mega caps, but note they might be less volatile
                pass
        
        return True
    
    def _score_options_strategies(self, candidate: EarningsCandidate):
        """Score candidate for different options strategies."""
        
        # Base score factors
        timing_score = self._calculate_timing_score(candidate)
        
        # Strategy-specific scoring (simplified without IV data for now)
        candidate.calendar_score = timing_score * self._calendar_multiplier(candidate)
        candidate.straddle_score = timing_score * self._straddle_multiplier(candidate)
        candidate.strangle_score = timing_score * self._strangle_multiplier(candidate)
    
    def _calculate_timing_score(self, candidate: EarningsCandidate) -> float:
        """Calculate base score based on earnings timing."""
        days = candidate.days_until_earnings
        
        # Optimal timing is 7-14 days for most strategies
        if 7 <= days <= 14:
            return 100.0
        elif 3 <= days <= 21:
            return 80.0
        elif 1 <= days <= 3:
            return 60.0  # Less time to set up
        else:
            return 40.0
    
    def _calendar_multiplier(self, candidate: EarningsCandidate) -> float:
        """Calendar spread suitability multiplier."""
        # Calendar spreads work best with time decay
        if candidate.days_until_earnings >= 14:
            return 1.0
        elif candidate.days_until_earnings >= 7:
            return 0.8
        else:
            return 0.5  # Too close for effective calendar
    
    def _straddle_multiplier(self, candidate: EarningsCandidate) -> float:
        """Straddle suitability multiplier."""
        # Straddles work best close to earnings for volatility expansion
        if candidate.days_until_earnings <= 5:
            return 1.0
        elif candidate.days_until_earnings <= 10:
            return 0.9
        else:
            return 0.7
    
    def _strangle_multiplier(self, candidate: EarningsCandidate) -> float:
        """Strangle suitability multiplier."""
        # Strangles similar to straddles but slightly more forgiving on timing
        if candidate.days_until_earnings <= 7:
            return 1.0
        elif candidate.days_until_earnings <= 14:
            return 0.8
        else:
            return 0.6
    
    def export_opportunities(self, 
                           candidates: List[EarningsCandidate],
                           output_file: Path = None) -> Path:
        """Export trading opportunities to file."""
        
        if output_file is None:
            output_file = self.data_path / "exports" / f"earnings_opportunities_{date.today().isoformat()}.csv"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame for easy export
        data = []
        for candidate in candidates:
            data.append({
                'Symbol': candidate.symbol,
                'Company': candidate.company_name,
                'Earnings_Date': candidate.earnings_date.isoformat(),
                'Time': candidate.time,
                'Days_Until': candidate.days_until_earnings,
                'Best_Strategy': candidate.best_strategy.value if candidate.best_strategy else 'none',
                'Total_Score': f"{candidate.total_score:.1f}",
                'Calendar_Score': f"{candidate.calendar_score:.1f}",
                'Straddle_Score': f"{candidate.straddle_score:.1f}",
                'Strangle_Score': f"{candidate.strangle_score:.1f}",
                'Priority': self._get_priority_emoji(candidate.total_score),
                'Action_Needed': self._get_action_needed(candidate)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        logger.info(f"💾 Exported {len(candidates)} opportunities to {output_file}")
        return output_file
    
    def _get_priority_emoji(self, score: float) -> str:
        """Get priority emoji based on score."""
        if score >= 90:
            return "🔥 EXCELLENT"
        elif score >= 75:
            return "⚡ VERY_GOOD"
        elif score >= 60:
            return "📊 GOOD"
        elif score >= 50:
            return "📅 CONSIDER"
        else:
            return "🔧 SKIP"
    
    def _get_action_needed(self, candidate: EarningsCandidate) -> str:
        """Determine next action needed for candidate."""
        actions = []
        
        if not candidate.option_chain_available:
            actions.append("GET_OPTION_CHAIN")
        
        if not candidate.historical_iv_available:
            actions.append("GET_IV_DATA")
        
        if candidate.best_strategy:
            actions.append(f"ANALYZE_{candidate.best_strategy.value.upper()}")
        
        return " | ".join(actions) if actions else "READY_TO_TRADE"
    
    def get_data_collection_list(self, candidates: List[EarningsCandidate]) -> Dict[str, List[str]]:
        """
        Generate prioritized list of symbols needing data collection.
        
        This replaces the fixed portfolio approach with dynamic symbol selection.
        """
        
        # Group by priority and data needs
        priority_groups = {
            'critical': [],      # Score 90+, earnings ≤3 days
            'high': [],         # Score 75+, earnings ≤7 days  
            'medium': [],       # Score 60+, earnings ≤14 days
            'low': []          # Score 50+, earnings >14 days
        }
        
        for candidate in candidates:
            score = candidate.total_score
            days = candidate.days_until_earnings
            
            if score >= 90 and days <= 3:
                priority_groups['critical'].append(candidate.symbol)
            elif score >= 75 and days <= 7:
                priority_groups['high'].append(candidate.symbol)
            elif score >= 60 and days <= 14:
                priority_groups['medium'].append(candidate.symbol)
            elif score >= 50:
                priority_groups['low'].append(candidate.symbol)
        
        logger.info("📋 Data collection priorities:")
        for priority, symbols in priority_groups.items():
            if symbols:
                logger.info(f"   {priority.upper()}: {len(symbols)} symbols")
        
        return priority_groups


def main():
    """Test the earnings discovery system."""
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 EARNINGS DISCOVERY SYSTEM TEST")
    print("=" * 60)
    
    engine = EarningsDiscoveryEngine()
    
    # Discover opportunities
    candidates = engine.discover_earnings_opportunities(
        days_ahead=21,
        min_score=40.0  # Lower threshold for testing
    )
    
    if candidates:
        print(f"\n🎯 Found {len(candidates)} trading opportunities:")
        
        for i, candidate in enumerate(candidates[:10], 1):  # Show top 10
            strategy_str = candidate.best_strategy.value if candidate.best_strategy else "none"
            print(f"{i:2d}. {candidate.symbol:<6} - {candidate.earnings_date} "
                  f"({candidate.days_until_earnings:+2d} days) - "
                  f"{candidate.total_score:.0f} pts - {strategy_str}")
        
        # Export opportunities
        export_file = engine.export_opportunities(candidates)
        print(f"\n💾 Opportunities exported to: {export_file}")
        
        # Show data collection needs
        data_priorities = engine.get_data_collection_list(candidates)
        
        print(f"\n📊 Next steps - Data collection needed:")
        for priority, symbols in data_priorities.items():
            if symbols:
                print(f"   {priority.upper()}: {symbols}")
                
    else:
        print("❌ No opportunities found")
        print("💡 This could be due to:")
        print("   • Missing earnings API keys")
        print("   • No earnings in the specified timeframe")
        print("   • Filters too restrictive")


if __name__ == "__main__":
    main()