#!/usr/bin/env python3
"""
Options Trading Strategy Analyzer

Analyzes pure earnings discoveries and determines optimal options trading
strategies. This module reads discoveries from the earnings discovery system
and adds trading-specific analysis.

Responsibilities:
1. Read pure earnings discoveries from discovery system
2. Analyze IV structure and options suitability
3. Score strategies (calendar spreads, strangles, straddles)
4. Generate trading recommendations
5. Export actionable trading opportunities

Usage:
    from trading.strategy_analyzer import OptionsStrategyAnalyzer
    
    analyzer = OptionsStrategyAnalyzer()
    opportunities = analyzer.analyze_discoveries()
"""

import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Import configuration first
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DISCOVERY_CONFIG, EXPORTS_PATH
from utils.logging_setup import get_logger

# Import earnings discovery module
from earnings.discovery import EarningsDiscoveryEngine, EarningsDiscovery

logger = get_logger(__name__)


class OptionsStrategy(Enum):
    """Potential options strategies for earnings plays."""
    CALENDAR_SPREAD = "calendar_spread"
    STRADDLE = "straddle" 
    STRANGLE = "strangle"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    COVERED_CALL = "covered_call"


@dataclass
class TradingOpportunity:
    """Trading opportunity based on earnings discovery + strategy analysis."""
    # Core discovery data
    symbol: str
    company_name: str
    earnings_date: date
    time: str
    days_until_earnings: int
    market_cap: Optional[float] = None
    quality_score: float = 0.0
    
    # Options characteristics (populated by analysis)
    has_weekly_options: bool = False
    has_liquid_options: bool = False
    iv_rank: Optional[float] = None
    iv_percentile: Optional[float] = None
    
    # Strategy suitability scores (0-100)
    calendar_score: float = 0.0
    straddle_score: float = 0.0
    strangle_score: float = 0.0
    
    # Analysis metadata
    analysis_date: date = None
    needs_data_collection: bool = True
    
    def __post_init__(self):
        if self.analysis_date is None:
            self.analysis_date = date.today()
    
    @property
    def total_score(self) -> float:
        """Overall trading opportunity score."""
        return max(self.calendar_score, self.straddle_score, self.strangle_score)
    
    @property
    def best_strategy(self) -> Optional[OptionsStrategy]:
        """Best options strategy for this opportunity."""
        scores = {
            OptionsStrategy.CALENDAR_SPREAD: self.calendar_score,
            OptionsStrategy.STRADDLE: self.straddle_score,
            OptionsStrategy.STRANGLE: self.strangle_score
        }
        
        best_strategy = max(scores, key=scores.get)
        if scores[best_strategy] > 50:  # Threshold for viable strategy
            return best_strategy
        return None


class OptionsStrategyAnalyzer:
    """Analyzes earnings discoveries and generates trading opportunities."""
    
    def __init__(self, config: dict = None):
        """
        Initialize options strategy analyzer.
        
        Args:
            config: Configuration dictionary (uses DISCOVERY_CONFIG if None)
        """
        
        self.config = config or DISCOVERY_CONFIG
        self.discovery_engine = EarningsDiscoveryEngine(config)
        
        # Strategy timing preferences from config
        self.strategy_optimal_start = self.config["strategy_optimal_start"]
        self.strategy_optimal_end = self.config["strategy_optimal_end"]
        
        # Scoring weights from config
        self.timing_weights = self.config["timing_weights"]
        self.strategy_multipliers = self.config["strategy_multipliers"]
        
        # Quality thresholds from config
        self.min_opportunity_score = self.config["min_opportunity_score"]
        self.excellent_threshold = self.config["excellent_threshold"]
        self.good_threshold = self.config["good_threshold"]
        
        logger.info(f"📊 Initialized options strategy analyzer:")
        logger.info(f"   Strategy optimal timing: {self.strategy_optimal_start}-{self.strategy_optimal_end} days")
        logger.info(f"   Min opportunity score: {self.min_opportunity_score}")
        logger.info(f"   Reads from: Pure earnings discovery system")
    
    def analyze_discoveries(self, max_discovery_age_days: int = 1) -> List[TradingOpportunity]:
        """
        Analyze recent earnings discoveries for trading opportunities.
        
        Args:
            max_discovery_age_days: Maximum age of discovery data to use
            
        Returns:
            List of trading opportunities with strategy analysis
        """
        
        logger.info(f"📊 ANALYZING EARNINGS DISCOVERIES FOR TRADING")
        logger.info(f"🔍 Looking for discoveries up to {max_discovery_age_days} days old")
        
        # Get recent earnings discoveries
        discoveries_df = self.discovery_engine.get_recent_discoveries(max_discovery_age_days)
        
        if discoveries_df is None or discoveries_df.empty:
            logger.warning("❌ No recent earnings discoveries found")
            logger.info("💡 Run earnings discovery first: python earnings/discovery.py")
            return []
        
        logger.info(f"📊 Analyzing {len(discoveries_df)} earnings discoveries")
        
        # Convert discoveries to trading opportunities
        opportunities = []
        for _, row in discoveries_df.iterrows():
            opportunity = self._create_opportunity_from_discovery(row)
            self._analyze_trading_strategies(opportunity)
            
            # Only keep opportunities that meet minimum score
            if opportunity.total_score >= self.min_opportunity_score:
                opportunities.append(opportunity)
        
        # Sort by best opportunities
        opportunities.sort(key=lambda x: -x.total_score)
        
        logger.info(f"🎯 Found {len(opportunities)} high-quality trading opportunities")
        
        # Export opportunities
        if opportunities:
            export_file = self._export_opportunities(opportunities)
            logger.info(f"💾 Exported trading opportunities to {export_file}")
        
        return opportunities
    
    def _create_opportunity_from_discovery(self, discovery_row: pd.Series) -> TradingOpportunity:
        """Convert earnings discovery to trading opportunity."""
        return TradingOpportunity(
            symbol=discovery_row['symbol'],
            company_name=discovery_row['company_name'],
            earnings_date=discovery_row['earnings_date'],
            time=discovery_row['time'],
            days_until_earnings=discovery_row['days_until'],
            market_cap=discovery_row.get('market_cap'),
            quality_score=discovery_row['quality_score']
        )
    
    def _analyze_trading_strategies(self, opportunity: TradingOpportunity):
        """Analyze and score trading strategies for the opportunity."""
        
        # Base score factors
        timing_score = self._calculate_timing_score(opportunity)
        
        # Strategy-specific scoring (simplified without IV data for now)
        opportunity.calendar_score = timing_score * self._calendar_multiplier(opportunity)
        opportunity.straddle_score = timing_score * self._straddle_multiplier(opportunity)
        opportunity.strangle_score = timing_score * self._strangle_multiplier(opportunity)
        
        # Mark if data collection is needed for deeper analysis
        opportunity.needs_data_collection = True  # Always true until we have options data
    
    def _calculate_timing_score(self, opportunity: TradingOpportunity) -> float:
        """Calculate base score based on earnings timing using configured weights."""
        days = opportunity.days_until_earnings
        
        # Use configured optimal timing windows
        if self.strategy_optimal_start <= days <= self.strategy_optimal_end:
            return self.timing_weights["optimal"]  # Default: 100.0
        elif 3 <= days <= 21:
            return self.timing_weights["good"]     # Default: 80.0
        elif 1 <= days <= 3:
            return self.timing_weights["fair"]     # Default: 60.0 - Less time to set up
        else:
            return self.timing_weights["poor"]     # Default: 40.0
    
    def _calendar_multiplier(self, opportunity: TradingOpportunity) -> float:
        """Calendar spread suitability multiplier using configured values."""
        days = opportunity.days_until_earnings
        multipliers = self.strategy_multipliers["calendar_spread"]
        
        if days >= 14:
            return multipliers["14_plus_days"]     # Default: 1.0
        elif days >= 7:
            return multipliers["7_to_14_days"]     # Default: 0.8
        else:
            return multipliers["under_7_days"]     # Default: 0.5 - Too close for effective calendar
    
    def _straddle_multiplier(self, opportunity: TradingOpportunity) -> float:
        """Straddle suitability multiplier using configured values."""
        days = opportunity.days_until_earnings
        multipliers = self.strategy_multipliers["straddle"]
        
        if days <= 5:
            return multipliers["under_5_days"]     # Default: 1.0
        elif days <= 10:
            return multipliers["5_to_10_days"]     # Default: 0.9
        else:
            return multipliers["over_10_days"]     # Default: 0.7
    
    def _strangle_multiplier(self, opportunity: TradingOpportunity) -> float:
        """Strangle suitability multiplier using configured values."""
        days = opportunity.days_until_earnings
        multipliers = self.strategy_multipliers["strangle"]
        
        if days <= 7:
            return multipliers["under_7_days"]     # Default: 1.0
        elif days <= 14:
            return multipliers["7_to_14_days"]     # Default: 0.8
        else:
            return multipliers["over_14_days"]     # Default: 0.6
    
    def _export_opportunities(self, opportunities: List[TradingOpportunity]) -> Path:
        """Export trading opportunities to file."""
        
        output_file = EXPORTS_PATH / f"trading_opportunities_{date.today().isoformat()}.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame for easy export
        data = []
        for opp in opportunities:
            data.append({
                'symbol': opp.symbol,
                'company_name': opp.company_name,
                'earnings_date': opp.earnings_date.isoformat(),
                'time': opp.time,
                'days_until': opp.days_until_earnings,
                'best_strategy': opp.best_strategy.value if opp.best_strategy else 'none',
                'total_score': f"{opp.total_score:.1f}",
                'calendar_score': f"{opp.calendar_score:.1f}",
                'straddle_score': f"{opp.straddle_score:.1f}",
                'strangle_score': f"{opp.strangle_score:.1f}",
                'priority': self._get_priority_emoji(opp.total_score),
                'needs_data_collection': opp.needs_data_collection,
                'market_cap': opp.market_cap,
                'analysis_date': opp.analysis_date.isoformat()
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        logger.info(f"💾 Exported {len(opportunities)} trading opportunities to {output_file}")
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
    
    def get_data_collection_priorities(self, opportunities: List[TradingOpportunity]) -> Dict[str, List[str]]:
        """
        Generate prioritized list of symbols needing data collection.
        """
        
        # Group by priority and data needs
        priority_groups = {
            'critical': [],      # Score 90+, earnings ≤3 days
            'high': [],         # Score 75+, earnings ≤7 days  
            'medium': [],       # Score 60+, earnings ≤14 days
            'low': []          # Score 50+, earnings >14 days
        }
        
        for opp in opportunities:
            if not opp.needs_data_collection:
                continue
                
            score = opp.total_score
            days = opp.days_until_earnings
            
            if score >= 90 and days <= 3:
                priority_groups['critical'].append(opp.symbol)
            elif score >= 75 and days <= 7:
                priority_groups['high'].append(opp.symbol)
            elif score >= 60 and days <= 14:
                priority_groups['medium'].append(opp.symbol)
            elif score >= 50:
                priority_groups['low'].append(opp.symbol)
        
        logger.info("📋 Data collection priorities:")
        for priority, symbols in priority_groups.items():
            if symbols:
                logger.info(f"   {priority.upper()}: {len(symbols)} symbols")
        
        return priority_groups


def main():
    """Test the options strategy analyzer."""
    from utils.logging_setup import setup_logging
    setup_logging()
    
    print("📊 OPTIONS STRATEGY ANALYZER TEST")
    print("=" * 60)
    print("🎯 Reads pure earnings discoveries and adds trading analysis")
    
    analyzer = OptionsStrategyAnalyzer()
    
    # Analyze recent discoveries
    opportunities = analyzer.analyze_discoveries(max_discovery_age_days=1)
    
    if opportunities:
        print(f"\n🎯 Found {len(opportunities)} trading opportunities:")
        
        for i, opp in enumerate(opportunities[:10], 1):  # Show top 10
            strategy_str = opp.best_strategy.value if opp.best_strategy else "none"
            print(f"{i:2d}. {opp.symbol:<6} - {opp.earnings_date} "
                  f"({opp.days_until_earnings:+2d} days) - "
                  f"{opp.total_score:.0f} pts - {strategy_str}")
        
        # Show data collection needs
        data_priorities = analyzer.get_data_collection_priorities(opportunities)
        
        print(f"\n📊 Next steps - Data collection needed:")
        for priority, symbols in data_priorities.items():
            if symbols:
                print(f"   {priority.upper()}: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
                
        print(f"\n💡 Workflow:")
        print(f"   1. ✅ Earnings discovered (pure discovery)")
        print(f"   2. ✅ Trading analysis applied")
        print(f"   3. 🔄 Collect options data for prioritized symbols")
        print(f"   4. 🔄 Execute trading strategies")
                
    else:
        print("❌ No trading opportunities found")
        print("💡 Make sure you have recent earnings discoveries:")
        print("   python earnings/discovery.py")


if __name__ == "__main__":
    main()