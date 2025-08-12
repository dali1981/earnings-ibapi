#!/usr/bin/env python3
"""
Pure Earnings Discovery System

Discovers ALL upcoming earnings across the market and stores them for other
systems to use. This module does NO trading analysis - just pure earnings
data discovery, filtering, and storage.

Responsibilities:
1. Discover ALL market earnings events
2. Apply basic market filters (market cap, volume, etc.)
3. Store discoveries for other systems to consume
4. Provide clean, standardized earnings data

Usage:
    from earnings.discovery import EarningsDiscoveryEngine
    
    engine = EarningsDiscoveryEngine()
    discoveries = engine.discover_and_store_earnings(days_ahead=30)
"""

import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict

# Import configuration first
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DISCOVERY_CONFIG, EARNINGS_CONFIG, EARNINGS_PATH, EXPORTS_PATH
from utils.logging_setup import get_logger

# Import our earnings fetcher and repository
from earnings.fetcher import EarningsCalendarFetcher, EarningsEvent
from repositories.earnings import EarningsRepository

logger = get_logger(__name__)


@dataclass
class EarningsDiscovery:
    """Pure earnings discovery data - no trading analysis."""
    symbol: str
    company_name: str
    earnings_date: date
    time: str  # bmo, amc, dmt
    days_until_earnings: int
    
    # Market characteristics (for basic filtering only)
    market_cap: Optional[float] = None
    avg_volume: Optional[int] = None
    price: Optional[float] = None
    eps_estimate: Optional[float] = None
    
    # Discovery metadata
    discovery_date: date = None
    source: str = "nasdaq"
    quality_score: float = 0.0
    
    def __post_init__(self):
        if self.discovery_date is None:
            self.discovery_date = date.today()


class EarningsDiscoveryEngine:
    """Pure earnings discovery engine - finds and stores ALL market earnings."""
    
    def __init__(self, config: dict = None):
        """
        Initialize pure earnings discovery engine.
        
        Args:
            config: Configuration dictionary (uses DISCOVERY_CONFIG if None)
        """
        
        self.config = config or DISCOVERY_CONFIG
        self.earnings_fetcher = EarningsCalendarFetcher()
        self.earnings_repository = EarningsRepository()
        
        # Load BASIC filtering criteria from configuration (no trading filters)
        self.min_market_cap = self.config["min_market_cap"]
        self.max_market_cap = self.config["max_market_cap"]
        self.min_avg_volume = self.config["min_avg_volume"]
        self.min_price = self.config["min_price"]
        self.max_price = self.config["max_price"]
        
        # Discovery timing windows (how far ahead to look)
        self.discovery_min_days = self.config["discovery_min_days"]
        self.discovery_max_days = self.config["discovery_max_days"]
        
        logger.info(f"📊 Initialized PURE earnings discovery engine:")
        logger.info(f"   Market cap filter: ${self.min_market_cap/1e9:.1f}B - ${self.max_market_cap/1e12:.1f}T")
        logger.info(f"   Volume filter: {self.min_avg_volume/1e6:.1f}M+ shares")
        logger.info(f"   Price filter: ${self.min_price} - ${self.max_price}")
        logger.info(f"   Discovery window: {self.discovery_min_days}-{self.discovery_max_days} days")
        logger.info(f"   NO TRADING ANALYSIS - Pure discovery only")
    
    def discover_and_store_earnings(self, 
                                  days_ahead: int = None) -> List[EarningsDiscovery]:
        """
        Discover ALL earnings in the market, filter, and store them.
        
        This is the main entry point for pure earnings discovery.
        
        Args:
            days_ahead: Number of days ahead to search (uses config default if None)
            
        Returns:
            List of discovered earnings (no trading analysis)
        """
        
        # Use configured defaults if not provided
        if days_ahead is None:
            days_ahead = EARNINGS_CONFIG["default_days_ahead"]
            
        logger.info(f"🔍 PURE EARNINGS DISCOVERY for next {days_ahead} days")
        logger.info(f"📊 NO trading analysis - just discovery and storage")
        
        # Step 1: Get ALL earnings events (no symbol filter)
        all_earnings = self.earnings_fetcher.get_upcoming_earnings(
            symbols=None,  # No filter - get everything
            days_ahead=days_ahead
        )
        
        logger.info(f"📅 Found {len(all_earnings)} total earnings events")
        
        if not all_earnings:
            logger.warning("❌ No earnings data available - check API keys")
            return []
        
        # Step 2: Store raw earnings data first
        storage_result = self.earnings_repository.store_earnings_batch(
            earnings=all_earnings,
            metadata={"discovery_purpose": "market_wide_discovery", "days_ahead": days_ahead}
        )
        logger.info(f"💾 Stored {storage_result['total_events_stored']} raw earnings events")
        
        # Step 3: Convert to discoveries and apply BASIC market filters only
        discoveries = []
        for event in all_earnings:
            discovery = self._create_discovery_from_earnings(event)
            if self._passes_market_filters(discovery):
                discoveries.append(discovery)
        
        logger.info(f"📊 {len(discoveries)} discoveries passed basic market filters")
        
        # Step 4: Export discoveries for other systems to use
        export_file = self._export_discoveries(discoveries)
        logger.info(f"💾 Exported {len(discoveries)} discoveries to {export_file}")
        
        return discoveries
    
    def _create_discovery_from_earnings(self, event: EarningsEvent) -> EarningsDiscovery:
        """Convert earnings event to pure discovery (no trading data)."""
        discovery = EarningsDiscovery(
            symbol=event.symbol,
            company_name=event.company_name,
            earnings_date=event.earnings_date,
            time=event.time or 'unknown',
            days_until_earnings=event.days_until_earnings,
            eps_estimate=event.eps_estimate,
            source=event.source
        )
        
        # Pass through market cap if available (from NASDAQ data)
        if hasattr(event, 'market_cap') and event.market_cap:
            discovery.market_cap = event.market_cap
            
        # Calculate basic quality score (no trading metrics)
        discovery.quality_score = self._calculate_quality_score(discovery)
        
        return discovery
    
    def _passes_market_filters(self, discovery: EarningsDiscovery) -> bool:
        """Apply BASIC market filtering criteria (NO trading filters)."""
        
        # Timing filter
        if discovery.days_until_earnings < self.discovery_min_days:
            return False
        if discovery.days_until_earnings > self.discovery_max_days:
            return False
        
        # Market cap filter (if available from NASDAQ data)
        if hasattr(discovery, 'market_cap') and discovery.market_cap:
            if discovery.market_cap < self.min_market_cap:
                return False
            if discovery.market_cap > self.max_market_cap:
                return False  # Exclude mega caps
        
        # Basic symbol validation
        symbol = discovery.symbol
        
        # Skip symbols that are problematic
        if len(symbol) > 5:  # Very long symbols
            return False
        
        # Skip symbols with special characters (warrants, rights, etc.)
        if any(char in symbol for char in ['.', '-', '/', '+', '~']):
            return False
        
        # Skip very short symbols
        if len(symbol) <= 1:
            return False
        
        # Additional market cap validation for penny stocks
        if hasattr(discovery, 'market_cap') and discovery.market_cap:
            if discovery.market_cap < 100e6:  # $100M absolute minimum
                return False
        
        return True
    
    def _calculate_quality_score(self, discovery: EarningsDiscovery) -> float:
        """Calculate basic data quality score (NO trading metrics)."""
        score = 0.0
        
        # Company name available
        if discovery.company_name and discovery.company_name.strip():
            score += 20.0
        
        # Earnings time specified
        if discovery.time and discovery.time != 'unknown':
            score += 20.0
        
        # EPS estimate available
        if discovery.eps_estimate is not None:
            score += 20.0
        
        # Market cap available
        if discovery.market_cap and discovery.market_cap > 0:
            score += 20.0
        
        # Recent discovery (not too far in future)
        if discovery.days_until_earnings <= 30:
            score += 20.0
        
        return min(score, 100.0)
    
    def _export_discoveries(self, discoveries: List[EarningsDiscovery]) -> Path:
        """Export pure discoveries to file for other systems to consume."""
        
        output_file = EXPORTS_PATH / f"earnings_discoveries_{date.today().isoformat()}.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame for easy export
        data = []
        for discovery in discoveries:
            data.append({
                'symbol': discovery.symbol,
                'company_name': discovery.company_name,
                'earnings_date': discovery.earnings_date.isoformat(),
                'time': discovery.time,
                'days_until': discovery.days_until_earnings,
                'market_cap': discovery.market_cap,
                'eps_estimate': discovery.eps_estimate,
                'quality_score': f"{discovery.quality_score:.1f}",
                'discovery_date': discovery.discovery_date.isoformat(),
                'source': discovery.source
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values(['days_until', 'market_cap'], ascending=[True, False])
        df.to_csv(output_file, index=False)
        
        logger.info(f"💾 Exported {len(discoveries)} pure discoveries to {output_file}")
        return output_file
    
    def get_recent_discoveries(self, max_age_days: int = 1) -> Optional[pd.DataFrame]:
        """Get recently discovered earnings from exports directory."""
        
        # Look for recent discovery files
        discovery_files = []
        for file_path in EXPORTS_PATH.glob("earnings_discoveries_*.csv"):
            try:
                # Extract date from filename
                date_str = file_path.stem.replace("earnings_discoveries_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                
                # Check if file is recent enough
                age = (date.today() - file_date).days
                if age <= max_age_days:
                    discovery_files.append((file_path, file_date))
                    
            except ValueError:
                continue
        
        if not discovery_files:
            logger.warning(f"No recent discovery files found (max age: {max_age_days} days)")
            return None
        
        # Get the most recent file
        latest_file, latest_date = max(discovery_files, key=lambda x: x[1])
        
        logger.info(f"📊 Loading discoveries from {latest_file} (date: {latest_date})")
        df = pd.read_csv(latest_file)
        df['earnings_date'] = pd.to_datetime(df['earnings_date']).dt.date
        df['discovery_date'] = pd.to_datetime(df['discovery_date']).dt.date
        
        return df
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored earnings data."""
        return self.earnings_repository.get_storage_stats()


def main():
    """Test the pure earnings discovery system."""
    from utils.logging_setup import setup_logging
    setup_logging()
    
    print("🔍 PURE EARNINGS DISCOVERY SYSTEM TEST")
    print("=" * 60)
    print("📊 NO TRADING ANALYSIS - Discovery only")
    
    engine = EarningsDiscoveryEngine()
    
    # Discover earnings
    discoveries = engine.discover_and_store_earnings(days_ahead=21)
    
    if discoveries:
        print(f"\n📊 Found {len(discoveries)} market discoveries:")
        
        # Show top 10 by quality score
        sorted_discoveries = sorted(discoveries, key=lambda x: -x.quality_score)
        
        for i, discovery in enumerate(sorted_discoveries[:10], 1):
            print(f"{i:2d}. {discovery.symbol:<6} - {discovery.earnings_date} "
                  f"({discovery.days_until_earnings:+2d} days) - "
                  f"{discovery.quality_score:.0f} pts - {discovery.company_name}")
        
        print(f"\n📈 Market Statistics:")
        market_caps = [d.market_cap for d in discoveries if d.market_cap]
        if market_caps:
            avg_market_cap = sum(market_caps) / len(market_caps)
            print(f"   Average market cap: ${avg_market_cap/1e9:.1f}B")
            print(f"   Companies with market cap data: {len(market_caps)}")
        
        print(f"\n💾 Data stored and exported for other systems to use")
        print(f"📊 Other systems can now analyze these {len(discoveries)} discoveries")
                
    else:
        print("❌ No discoveries found")
        print("💡 This could be due to:")
        print("   • Missing earnings API keys")
        print("   • No earnings in the specified timeframe")
        print("   • Market filters too restrictive")


if __name__ == "__main__":
    main()