#!/usr/bin/env python3
"""
IB-Only Earnings Calendar Example

Pure Interactive Brokers API demonstration showing:
1. Earnings calendar from IB with BMO/AMC timing
2. Earnings date extraction from IB XML
3. XML parsing of IB calendar data

This example uses ONLY the IB API and does not call any external data sources.
Focus: Getting earnings timing (BMO/AMC) from IB calendar data.

Usage:
    python examples/ib_fundamental_data.py
    
Prerequisites:
    - IB Gateway or TWS running
    - ibapi package installed: pip install ibapi
    - Valid IB account with market data subscriptions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from datetime import date, timedelta
from typing import Dict, List, Optional, Any
import time

from utils.logging_setup import setup_logging, get_logger
from ib.fundamental_data import IBFundamentalDataFetcher, CompanyFundamentals, CalendarEvent, EarningsEstimate
from config import EXPORTS_PATH

logger = get_logger(__name__)


class IBEarningsCalendarAnalyzer:
    """Pure IB API earnings calendar analyzer - focuses on BMO/AMC timing extraction."""
    
    def __init__(self):
        """Initialize with IB fetcher only."""
        self.ib_fetcher = IBFundamentalDataFetcher()
        logger.info("📅 IB Earnings Calendar Analyzer initialized")
    
    def get_earnings_timing_analysis(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get earnings calendar with BMO/AMC timing analysis from IB.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            DataFrame with earnings timing data
        """
        logger.info(f"📅 Fetching IB earnings timing for {len(symbols)} symbols")
        
        if not self.ib_fetcher.connect():
            logger.error("Failed to connect to IB Gateway")
            return pd.DataFrame()
        
        try:
            calendar_events = self.ib_fetcher.get_earnings_calendar(symbols)
            logger.info(f"✅ Retrieved {len(calendar_events)} calendar events from IB")
            
            # Convert to DataFrame for analysis
            if not calendar_events:
                return pd.DataFrame()
            
            data = []
            for event in calendar_events:
                data.append({
                    "symbol": event.symbol,
                    "earnings_date": event.event_date,
                    "earnings_time": event.earnings_time,
                    "description": event.description,
                    "source": "interactive_brokers"
                })
            
            df = pd.DataFrame(data)
            return df
            
        finally:
            self.ib_fetcher.disconnect()
    
    def analyze_earnings_timing_patterns(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Analyze earnings timing patterns from IB calendar data.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with timing analysis results
        """
        logger.info(f"📊 Analyzing earnings timing patterns for {len(symbols)} symbols")
        
        # Get timing data
        timing_df = self.get_earnings_timing_analysis(symbols)
        
        if timing_df.empty:
            return {"error": "No timing data available"}
        
        # Analyze patterns
        results = {
            "total_earnings": len(timing_df),
            "timing_breakdown": {},
            "symbols_with_timing": 0,
            "symbols_missing_timing": 0
        }
        
        # Count timing patterns
        timing_counts = timing_df['earnings_time'].value_counts()
        for timing, count in timing_counts.items():
            results["timing_breakdown"][timing or "unknown"] = count
        
        # Count symbols with/without timing
        results["symbols_with_timing"] = len(timing_df[timing_df['earnings_time'].notna()])
        results["symbols_missing_timing"] = len(timing_df[timing_df['earnings_time'].isna()])
        
        return results
    
    def create_earnings_timing_report(self, symbols: List[str]) -> pd.DataFrame:
        """
        Create focused earnings timing report using only IB calendar data.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            DataFrame with earnings dates and BMO/AMC timing
        """
        logger.info(f"📅 Creating earnings timing report for {len(symbols)} symbols")
        
        return self.get_earnings_timing_analysis(symbols)
    
    def validate_earnings_timing(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Validate and analyze earnings timing data from IB.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with timing validation results
        """
        logger.info(f"🔍 Validating earnings timing for {len(symbols)} symbols")
        
        # Get timing data and patterns
        timing_df = self.get_earnings_timing_analysis(symbols)
        patterns = self.analyze_earnings_timing_patterns(symbols)
        
        validation_results = {
            "total_symbols_requested": len(symbols),
            "symbols_with_earnings_data": len(timing_df) if not timing_df.empty else 0,
            "data_completeness_pct": (len(timing_df) / len(symbols)) * 100 if symbols else 0,
            "timing_patterns": patterns,
            "detailed_timing": {}
        }
        
        # Add detailed timing info
        if not timing_df.empty:
            for _, row in timing_df.iterrows():
                symbol = row['symbol']
                validation_results["detailed_timing"][symbol] = {
                    "earnings_date": row.get('earnings_date'),
                    "timing": row.get('earnings_time', 'unknown'),
                    "has_bmo_amc": row.get('earnings_time') in ['BMO', 'AMC'],
                    "description": row.get('description', '')
                }
        
        logger.info(f"🔍 Timing validation completed: {validation_results['data_completeness_pct']:.1f}% data coverage")
        return validation_results


def main():
    """Demonstrate IB earnings calendar timing extraction."""
    setup_logging()
    
    print("📅 IB EARNINGS CALENDAR TIMING EXAMPLE")
    print("=" * 60)
    print("🎯 Focus: Extract BMO/AMC timing from IB calendar data")
    print("📋 No external data sources - IB API only")
    print()
    
    # Check if IB API is available
    try:
        from ibapi.client import EClient
        print("✅ IB API available")
    except ImportError:
        print("❌ IB API not available")
        print("💡 Install with: pip install ibapi")
        print("💡 This example requires the IB API to function")
        return
    
    # Test symbols - major companies with good IB data coverage
    test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    print(f"🧪 Testing earnings timing extraction for: {test_symbols}")
    print()
    
    analyzer = IBEarningsCalendarAnalyzer()
    
    try:
        # Test 1: Basic Earnings Calendar with Timing
        print("1️⃣ EARNINGS CALENDAR WITH BMO/AMC TIMING")
        print("-" * 50)
        
        timing_df = analyzer.get_earnings_timing_analysis(test_symbols)
        
        if not timing_df.empty:
            print(f"✅ Retrieved earnings timing for {len(timing_df)} companies:")
            print()
            
            for _, row in timing_df.iterrows():
                symbol = row['symbol']
                earnings_date = row.get('earnings_date', 'N/A')
                timing = row.get('earnings_time', 'TBD')
                
                print(f"   {symbol}: {earnings_date} ({timing})")
                
                # Highlight BMO/AMC timing
                if timing in ['BMO', 'AMC']:
                    print(f"     ✅ Timing confirmed: {timing}")
                elif timing == 'DMT':
                    print(f"     ⚠️  During market: {timing}")
                else:
                    print(f"     ❓ Timing unknown or missing")
                print()
        else:
            print("❌ No earnings timing data retrieved")
            print("💡 Check IB Gateway connection and data subscriptions")
        
        print("\n" + "=" * 60)
        
        # Test 2: Timing Analysis and Validation
        print("2️⃣ TIMING PATTERN ANALYSIS")
        print("-" * 40)
        
        validation_results = analyzer.validate_earnings_timing(test_symbols)
        
        print(f"📊 Analysis Results:")
        print(f"   Total symbols: {validation_results['total_symbols_requested']}")
        print(f"   With earnings data: {validation_results['symbols_with_earnings_data']}")
        print(f"   Data coverage: {validation_results['data_completeness_pct']:.1f}%")
        print()
        
        # Show timing breakdown
        if 'timing_patterns' in validation_results and 'timing_breakdown' in validation_results['timing_patterns']:
            timing_breakdown = validation_results['timing_patterns']['timing_breakdown']
            print("⏰ Timing Breakdown:")
            for timing_type, count in timing_breakdown.items():
                if timing_type in ['BMO', 'AMC']:
                    print(f"   ✅ {timing_type}: {count} companies")
                elif timing_type == 'DMT':
                    print(f"   ⚠️  {timing_type}: {count} companies")
                else:
                    print(f"   ❓ {timing_type}: {count} companies")
        
        # Export timing report
        timing_report = analyzer.create_earnings_timing_report(test_symbols)
        if not timing_report.empty:
            report_file = EXPORTS_PATH / f"ib_earnings_timing_{date.today().isoformat()}.csv"
            timing_report.to_csv(report_file, index=False)
            print(f"\n💾 Timing report exported to: {report_file}")
        
        print("\n" + "=" * 60)
        
        # Usage summary
        print("💡 IB EARNINGS TIMING EXTRACTION:")
        print("-" * 40)
        print("📅 Calendar Focus:")
        print("   • Extract BMO/AMC timing from IB earnings calendar")
        print("   • Parse IB XML responses for timing information")
        print("   • Validate earnings dates with timing data")
        print("   • No external API dependencies")
        print()
        print("🎯 Trading Applications:")
        print("   • Know exact earnings timing for options strategies")
        print("   • BMO = Before Market Open (safer for overnight positions)")
        print("   • AMC = After Market Close (react during market hours)")
        print("   • DMT = During Market Time (highest risk)")
        print()
        print("⚠️  Limitations:")
        print("   • IB fundamental data requires expensive subscriptions")
        print("   • Paper accounts have limited calendar data access")
        print("   • XML parsing needs refinement for better timing extraction")
        
    except Exception as e:
        logger.error(f"IB earnings timing example failed: {e}")
        print(f"❌ IB earnings timing example failed: {e}")
        print("💡 Ensure IB Gateway/TWS is running and properly configured")
    
    print(f"\n✅ IB Earnings Timing Example Complete!")


if __name__ == "__main__":
    main()