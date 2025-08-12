#!/usr/bin/env python3
"""
DEPRECATED: Mixed Earnings Discovery + Trading Analysis

This module mixed earnings discovery with trading strategy analysis.
It has been separated into cleaner components:

1. Pure Earnings Discovery: earnings/discovery.py
   - Discovers ALL market earnings
   - Applies basic market filters only
   - Stores discoveries for other systems to use
   - NO trading analysis

2. Trading Strategy Analysis: trading/strategy_analyzer.py
   - Reads pure earnings discoveries
   - Applies trading-specific analysis
   - Scores options strategies
   - Generates trading opportunities

Usage Migration:

OLD (mixed approach):
    from earnings_trading.discovery import EarningsDiscoveryEngine
    engine = EarningsDiscoveryEngine()
    candidates = engine.discover_earnings_opportunities()

NEW (separated approach):
    # Step 1: Pure discovery
    from earnings.discovery import EarningsDiscoveryEngine
    discoverer = EarningsDiscoveryEngine()
    discoveries = discoverer.discover_and_store_earnings()
    
    # Step 2: Trading analysis (optional)
    from trading.strategy_analyzer import OptionsStrategyAnalyzer
    analyzer = OptionsStrategyAnalyzer()
    opportunities = analyzer.analyze_discoveries()

Benefits of separation:
- Pure earnings data can be used by ANY system
- Trading analysis is separate and optional
- Cleaner responsibilities and testing
- Other systems can build on pure discovery data
"""

import sys
from pathlib import Path

def main():
    """Show migration guidance."""
    print("⚠️  DEPRECATED MODULE")
    print("=" * 50)
    print("This module mixed earnings discovery with trading analysis.")
    print("It has been separated into cleaner components:")
    print()
    print("🔍 PURE EARNINGS DISCOVERY:")
    print("   python earnings/discovery.py")
    print("   - Discovers ALL market earnings")
    print("   - NO trading analysis")
    print("   - Stores data for other systems")
    print()
    print("📊 TRADING STRATEGY ANALYSIS:")  
    print("   python trading/strategy_analyzer.py")
    print("   - Reads pure earnings discoveries")
    print("   - Adds trading analysis")
    print("   - Generates opportunities")
    print()
    print("🛠️  UTILITY SCRIPTS:")
    print("   python scripts/get_latest_discoveries.py")
    print("   - Get latest discoveries in various formats")
    print("   - Filter by symbols, quality, etc.")
    print()
    print("💡 Benefits of separation:")
    print("   - Pure earnings data for ANY system")
    print("   - Trading analysis is optional")
    print("   - Cleaner responsibilities")
    print("   - Better testing and maintenance")


if __name__ == "__main__":
    main()