#!/usr/bin/env python3
"""
Earnings Data Sources Comparison Script

Compares earnings data from NASDAQ and Yahoo Finance for both upcoming 
and historical earnings announcements to evaluate data quality, coverage,
and consistency across sources.

Usage:
    python scripts/compare_earnings_sources.py --days-ahead 14 --compare-historical 7
    python scripts/compare_earnings_sources.py --symbols AAPL,GOOGL,MSFT --output comparison_report.csv
"""

import sys
import json
import logging
import argparse
import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from earnings.fetcher import EarningsCalendarFetcher, EarningsEvent, EarningsSource

logger = logging.getLogger(__name__)

@dataclass
class ComparisonMetrics:
    """Metrics for comparing earnings data sources."""
    source_name: str
    total_events: int
    unique_symbols: int
    events_with_eps: int
    events_with_timing: int
    average_market_cap: Optional[float] = None
    coverage_by_day: Dict[str, int] = None
    data_completeness: float = 0.0
    
    def __post_init__(self):
        if self.coverage_by_day is None:
            self.coverage_by_day = {}
        
        # Calculate data completeness score
        completeness_factors = []
        if self.total_events > 0:
            completeness_factors.append(self.events_with_eps / self.total_events)
            completeness_factors.append(self.events_with_timing / self.total_events)
        
        self.data_completeness = sum(completeness_factors) / len(completeness_factors) if completeness_factors else 0.0

@dataclass 
class SymbolComparison:
    """Comparison of a single symbol across sources."""
    symbol: str
    nasdaq_data: Optional[EarningsEvent] = None
    yahoo_data: Optional[EarningsEvent] = None
    date_match: bool = False
    eps_match: bool = False
    timing_match: bool = False
    data_quality_score: float = 0.0

class YahooEarningsFetcher:
    """Yahoo Finance earnings data fetcher for comparison."""
    
    def __init__(self):
        import requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def fetch_upcoming_earnings(self, days_ahead: int = 14) -> List[EarningsEvent]:
        """Fetch upcoming earnings from Yahoo Finance."""
        earnings = []
        
        try:
            # Yahoo Finance earnings calendar endpoint
            url = "https://finance.yahoo.com/calendar/earnings"
            
            # For now, return mock data since Yahoo scraping is complex
            # In a real implementation, you would parse the HTML or find an API endpoint
            logger.warning("Yahoo Finance fetcher using mock data - implement HTML parsing for production")
            
            # Generate some mock earnings data for comparison
            mock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA']
            start_date = date.today()
            
            for i, symbol in enumerate(mock_symbols):
                earnings_date = start_date + timedelta(days=i + 1)
                if earnings_date <= start_date + timedelta(days=days_ahead):
                    event = EarningsEvent(
                        symbol=symbol,
                        company_name=f"{symbol} Inc.",
                        earnings_date=earnings_date,
                        time='bmo' if i % 2 == 0 else 'amc',
                        eps_estimate=round(1.5 + (i * 0.3), 2),
                        source='yahoo_mock'
                    )
                    earnings.append(event)
                    
        except Exception as e:
            logger.error(f"Yahoo earnings fetch failed: {e}")
            
        return earnings
    
    def fetch_historical_earnings(self, days_back: int = 7) -> List[EarningsEvent]:
        """Fetch historical earnings from Yahoo Finance."""
        # Mock historical data for now
        earnings = []
        
        try:
            mock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            end_date = date.today()
            
            for i, symbol in enumerate(mock_symbols):
                earnings_date = end_date - timedelta(days=i + 1)
                if earnings_date >= end_date - timedelta(days=days_back):
                    event = EarningsEvent(
                        symbol=symbol,
                        company_name=f"{symbol} Inc.",
                        earnings_date=earnings_date,
                        time='bmo' if i % 2 == 0 else 'amc',
                        eps_estimate=round(1.2 + (i * 0.4), 2),
                        eps_actual=round(1.3 + (i * 0.4), 2),  # Mock actual results
                        source='yahoo_historical_mock'
                    )
                    earnings.append(event)
                    
        except Exception as e:
            logger.error(f"Yahoo historical earnings fetch failed: {e}")
            
        return earnings

class EarningsSourceComparator:
    """Compare earnings data from multiple sources."""
    
    def __init__(self, output_dir: Path = Path("data/comparisons")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.nasdaq_fetcher = EarningsCalendarFetcher()
        self.yahoo_fetcher = YahooEarningsFetcher()
    
    def compare_upcoming_earnings(self, 
                                 days_ahead: int = 14,
                                 symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare upcoming earnings data from NASDAQ and Yahoo."""
        
        logger.info(f"🔍 Comparing upcoming earnings data for next {days_ahead} days")
        
        # Fetch from NASDAQ
        logger.info("📊 Fetching from NASDAQ...")
        nasdaq_earnings = self.nasdaq_fetcher.get_upcoming_earnings(
            symbols=symbols,
            days_ahead=days_ahead,
            sources=[EarningsSource.NASDAQ]
        )
        
        # Fetch from Yahoo  
        logger.info("📊 Fetching from Yahoo...")
        yahoo_earnings = self.yahoo_fetcher.fetch_upcoming_earnings(days_ahead=days_ahead)
        
        # Filter by symbols if provided
        if symbols:
            nasdaq_earnings = [e for e in nasdaq_earnings if e.symbol in symbols]
            yahoo_earnings = [e for e in yahoo_earnings if e.symbol in symbols]
        
        # Analyze each source
        nasdaq_metrics = self._analyze_earnings_data(nasdaq_earnings, "NASDAQ")
        yahoo_metrics = self._analyze_earnings_data(yahoo_earnings, "Yahoo")
        
        # Compare overlapping symbols
        symbol_comparisons = self._compare_symbols(nasdaq_earnings, yahoo_earnings)
        
        return {
            'comparison_date': date.today().isoformat(),
            'comparison_type': 'upcoming_earnings',
            'days_ahead': days_ahead,
            'nasdaq_metrics': asdict(nasdaq_metrics),
            'yahoo_metrics': asdict(yahoo_metrics),
            'symbol_comparisons': [asdict(comp) for comp in symbol_comparisons],
            'summary': self._generate_summary(nasdaq_metrics, yahoo_metrics, symbol_comparisons)
        }
    
    def compare_historical_earnings(self,
                                   days_back: int = 7, 
                                   symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare historical earnings data from NASDAQ and Yahoo."""
        
        logger.info(f"🔍 Comparing historical earnings data for past {days_back} days")
        
        # For historical data, we'll use cached NASDAQ data and Yahoo historical
        logger.info("📊 Fetching historical from Yahoo...")
        yahoo_historical = self.yahoo_fetcher.fetch_historical_earnings(days_back=days_back)
        
        # For NASDAQ, we'd need to fetch historical data or use cached results
        # For now, use recent cached data as a proxy
        nasdaq_historical = []
        cache_file = Path("cache/earnings/earnings_historical.json")
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    for item in data.get('earnings', []):
                        event = EarningsEvent(
                            symbol=item['symbol'],
                            company_name=item['company_name'],
                            earnings_date=datetime.strptime(item['earnings_date'], '%Y-%m-%d').date(),
                            time=item.get('time'),
                            eps_estimate=item.get('eps_estimate'),
                            eps_actual=item.get('eps_actual'),
                            source=item['source']
                        )
                        nasdaq_historical.append(event)
            except Exception as e:
                logger.warning(f"Could not load historical NASDAQ data: {e}")
        
        # Filter by symbols if provided
        if symbols:
            nasdaq_historical = [e for e in nasdaq_historical if e.symbol in symbols]
            yahoo_historical = [e for e in yahoo_historical if e.symbol in symbols]
        
        # Analyze data
        nasdaq_metrics = self._analyze_earnings_data(nasdaq_historical, "NASDAQ_Historical")
        yahoo_metrics = self._analyze_earnings_data(yahoo_historical, "Yahoo_Historical")
        
        # Compare symbols
        symbol_comparisons = self._compare_symbols(nasdaq_historical, yahoo_historical)
        
        return {
            'comparison_date': date.today().isoformat(),
            'comparison_type': 'historical_earnings',
            'days_back': days_back,
            'nasdaq_metrics': asdict(nasdaq_metrics),
            'yahoo_metrics': asdict(yahoo_metrics),
            'symbol_comparisons': [asdict(comp) for comp in symbol_comparisons],
            'summary': self._generate_summary(nasdaq_metrics, yahoo_metrics, symbol_comparisons)
        }
    
    def _analyze_earnings_data(self, earnings: List[EarningsEvent], source_name: str) -> ComparisonMetrics:
        """Analyze earnings data and generate metrics."""
        
        if not earnings:
            return ComparisonMetrics(
                source_name=source_name,
                total_events=0,
                unique_symbols=0,
                events_with_eps=0,
                events_with_timing=0
            )
        
        # Basic counts
        total_events = len(earnings)
        unique_symbols = len(set(e.symbol for e in earnings))
        events_with_eps = len([e for e in earnings if e.eps_estimate is not None])
        events_with_timing = len([e for e in earnings if e.time and e.time != 'unknown'])
        
        # Coverage by day
        coverage_by_day = {}
        for event in earnings:
            day_str = event.earnings_date.isoformat()
            coverage_by_day[day_str] = coverage_by_day.get(day_str, 0) + 1
        
        # Average market cap (if available)
        market_caps = []
        for event in earnings:
            if hasattr(event, 'market_cap') and event.market_cap:
                market_caps.append(event.market_cap)
        
        avg_market_cap = sum(market_caps) / len(market_caps) if market_caps else None
        
        return ComparisonMetrics(
            source_name=source_name,
            total_events=total_events,
            unique_symbols=unique_symbols,
            events_with_eps=events_with_eps,
            events_with_timing=events_with_timing,
            average_market_cap=avg_market_cap,
            coverage_by_day=coverage_by_day
        )
    
    def _compare_symbols(self, 
                        nasdaq_earnings: List[EarningsEvent],
                        yahoo_earnings: List[EarningsEvent]) -> List[SymbolComparison]:
        """Compare earnings data for overlapping symbols."""
        
        # Create lookup dictionaries
        nasdaq_lookup = {e.symbol: e for e in nasdaq_earnings}
        yahoo_lookup = {e.symbol: e for e in yahoo_earnings}
        
        # Get all symbols from both sources
        all_symbols = set(nasdaq_lookup.keys()) | set(yahoo_lookup.keys())
        
        comparisons = []
        for symbol in all_symbols:
            nasdaq_event = nasdaq_lookup.get(symbol)
            yahoo_event = yahoo_lookup.get(symbol)
            
            # Compare dates
            date_match = False
            if nasdaq_event and yahoo_event:
                date_match = nasdaq_event.earnings_date == yahoo_event.earnings_date
            
            # Compare EPS estimates
            eps_match = False
            if (nasdaq_event and yahoo_event and 
                nasdaq_event.eps_estimate and yahoo_event.eps_estimate):
                # Allow 5% tolerance for EPS estimates
                eps_diff = abs(nasdaq_event.eps_estimate - yahoo_event.eps_estimate)
                tolerance = max(abs(nasdaq_event.eps_estimate) * 0.05, 0.01)
                eps_match = eps_diff <= tolerance
            
            # Compare timing
            timing_match = False
            if (nasdaq_event and yahoo_event and
                nasdaq_event.time and yahoo_event.time):
                timing_match = nasdaq_event.time.lower() == yahoo_event.time.lower()
            
            # Calculate quality score
            quality_factors = []
            if nasdaq_event: quality_factors.append(0.5)  # Has NASDAQ data
            if yahoo_event: quality_factors.append(0.5)   # Has Yahoo data
            if date_match: quality_factors.append(1.0)    # Dates match
            if eps_match: quality_factors.append(0.5)     # EPS estimates match
            if timing_match: quality_factors.append(0.5)  # Timing matches
            
            quality_score = sum(quality_factors) / 3.0  # Normalize to 0-1
            
            comparison = SymbolComparison(
                symbol=symbol,
                nasdaq_data=nasdaq_event,
                yahoo_data=yahoo_event,
                date_match=date_match,
                eps_match=eps_match,
                timing_match=timing_match,
                data_quality_score=quality_score
            )
            comparisons.append(comparison)
        
        # Sort by quality score (best first)
        comparisons.sort(key=lambda x: -x.data_quality_score)
        
        return comparisons
    
    def _generate_summary(self,
                         nasdaq_metrics: ComparisonMetrics,
                         yahoo_metrics: ComparisonMetrics,
                         symbol_comparisons: List[SymbolComparison]) -> Dict[str, Any]:
        """Generate comparison summary."""
        
        # Source coverage
        nasdaq_only = [c for c in symbol_comparisons if c.nasdaq_data and not c.yahoo_data]
        yahoo_only = [c for c in symbol_comparisons if c.yahoo_data and not c.nasdaq_data]
        both_sources = [c for c in symbol_comparisons if c.nasdaq_data and c.yahoo_data]
        
        # Agreement rates
        date_agreement_rate = 0.0
        eps_agreement_rate = 0.0
        timing_agreement_rate = 0.0
        
        if both_sources:
            date_agreement_rate = len([c for c in both_sources if c.date_match]) / len(both_sources)
            eps_agreement_rate = len([c for c in both_sources if c.eps_match]) / len(both_sources)
            timing_agreement_rate = len([c for c in both_sources if c.timing_match]) / len(both_sources)
        
        # Overall quality score
        avg_quality_score = sum(c.data_quality_score for c in symbol_comparisons) / len(symbol_comparisons) if symbol_comparisons else 0.0
        
        return {
            'total_symbols_compared': len(symbol_comparisons),
            'nasdaq_only_count': len(nasdaq_only),
            'yahoo_only_count': len(yahoo_only),
            'both_sources_count': len(both_sources),
            'date_agreement_rate': round(date_agreement_rate * 100, 1),
            'eps_agreement_rate': round(eps_agreement_rate * 100, 1),
            'timing_agreement_rate': round(timing_agreement_rate * 100, 1),
            'average_quality_score': round(avg_quality_score, 2),
            'nasdaq_completeness': round(nasdaq_metrics.data_completeness * 100, 1),
            'yahoo_completeness': round(yahoo_metrics.data_completeness * 100, 1),
            'recommendation': self._get_recommendation(nasdaq_metrics, yahoo_metrics, date_agreement_rate)
        }
    
    def _get_recommendation(self,
                           nasdaq_metrics: ComparisonMetrics,
                           yahoo_metrics: ComparisonMetrics,
                           agreement_rate: float) -> str:
        """Generate recommendation based on comparison results."""
        
        if nasdaq_metrics.total_events > yahoo_metrics.total_events * 2:
            return "NASDAQ provides significantly better coverage. Recommend using NASDAQ as primary source."
        elif yahoo_metrics.total_events > nasdaq_metrics.total_events * 2:
            return "Yahoo provides significantly better coverage. Consider Yahoo as primary source."
        elif agreement_rate > 0.8:
            return "High agreement between sources. Either can be used reliably."
        elif agreement_rate > 0.6:
            return "Moderate agreement between sources. Use NASDAQ as primary with Yahoo fallback."
        else:
            return "Low agreement between sources. Investigate data quality issues."
    
    def export_comparison_report(self, comparison_data: Dict[str, Any], output_file: Optional[Path] = None) -> Path:
        """Export comparison results to CSV and JSON."""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"earnings_comparison_{timestamp}.csv"
        
        # Create detailed CSV report
        report_data = []
        
        # Add summary row
        summary = comparison_data['summary']
        report_data.append({
            'Type': 'SUMMARY',
            'Symbol': 'ALL',
            'NASDAQ_Events': comparison_data['nasdaq_metrics']['total_events'],
            'Yahoo_Events': comparison_data['yahoo_metrics']['total_events'],
            'Both_Sources': summary['both_sources_count'],
            'Date_Agreement_%': summary['date_agreement_rate'],
            'EPS_Agreement_%': summary['eps_agreement_rate'],
            'Timing_Agreement_%': summary['timing_agreement_rate'],
            'Quality_Score': summary['average_quality_score'],
            'Recommendation': summary['recommendation']
        })
        
        # Add symbol-by-symbol comparison
        for comp in comparison_data['symbol_comparisons'][:20]:  # Top 20
            nasdaq_date = comp['nasdaq_data']['earnings_date'] if comp['nasdaq_data'] else 'N/A'
            yahoo_date = comp['yahoo_data']['earnings_date'] if comp['yahoo_data'] else 'N/A'
            
            nasdaq_eps = comp['nasdaq_data']['eps_estimate'] if comp['nasdaq_data'] else 'N/A'
            yahoo_eps = comp['yahoo_data']['eps_estimate'] if comp['yahoo_data'] else 'N/A'
            
            report_data.append({
                'Type': 'SYMBOL',
                'Symbol': comp['symbol'],
                'NASDAQ_Date': nasdaq_date,
                'Yahoo_Date': yahoo_date,
                'NASDAQ_EPS': nasdaq_eps,
                'Yahoo_EPS': yahoo_eps,
                'Date_Match': comp['date_match'],
                'EPS_Match': comp['eps_match'],
                'Timing_Match': comp['timing_match'],
                'Quality_Score': comp['data_quality_score']
            })
        
        # Write CSV
        df = pd.DataFrame(report_data)
        df.to_csv(output_file, index=False)
        
        # Write JSON (with date serialization handling)
        json_file = output_file.with_suffix('.json')
        with open(json_file, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        logger.info(f"💾 Comparison report exported to {output_file}")
        logger.info(f"💾 Detailed data exported to {json_file}")
        
        return output_file

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Compare earnings data from multiple sources")
    
    parser.add_argument('--days-ahead', type=int, default=14,
                       help='Days ahead to fetch upcoming earnings (default: 14)')
    parser.add_argument('--compare-historical', type=int, default=0,
                       help='Days back to compare historical earnings (default: 0, no historical)')
    parser.add_argument('--symbols', type=str,
                       help='Comma-separated list of symbols to compare (default: all)')
    parser.add_argument('--output', type=str,
                       help='Output filename for comparison report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup centralized logging
    from utils.logging_setup import setup_logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
        logger.info(f"🎯 Comparing specific symbols: {symbols}")
    
    # Initialize comparator
    comparator = EarningsSourceComparator()
    
    print("🔍 EARNINGS DATA SOURCES COMPARISON")
    print("=" * 60)
    
    # Compare upcoming earnings
    if args.days_ahead > 0:
        print(f"\n📊 UPCOMING EARNINGS COMPARISON ({args.days_ahead} days ahead)")
        print("-" * 60)
        
        comparison_data = comparator.compare_upcoming_earnings(
            days_ahead=args.days_ahead,
            symbols=symbols
        )
        
        # Display results
        summary = comparison_data['summary']
        print(f"📈 Total symbols compared: {summary['total_symbols_compared']}")
        print(f"📊 NASDAQ only: {summary['nasdaq_only_count']}")
        print(f"📊 Yahoo only: {summary['yahoo_only_count']}")  
        print(f"📊 Both sources: {summary['both_sources_count']}")
        print(f"📊 Date agreement: {summary['date_agreement_rate']}%")
        print(f"📊 EPS agreement: {summary['eps_agreement_rate']}%")
        print(f"📊 Quality score: {summary['average_quality_score']}/1.0")
        print(f"💡 {summary['recommendation']}")
        
        # Export report
        output_file = Path(args.output) if args.output else None
        report_file = comparator.export_comparison_report(comparison_data, output_file)
        print(f"\n💾 Report saved to: {report_file}")
    
    # Compare historical earnings
    if args.compare_historical > 0:
        print(f"\n📊 HISTORICAL EARNINGS COMPARISON ({args.compare_historical} days back)")
        print("-" * 60)
        
        historical_data = comparator.compare_historical_earnings(
            days_back=args.compare_historical,
            symbols=symbols
        )
        
        # Display results
        summary = historical_data['summary']
        print(f"📈 Total symbols compared: {summary['total_symbols_compared']}")
        print(f"📊 Date agreement: {summary['date_agreement_rate']}%")
        print(f"📊 Quality score: {summary['average_quality_score']}/1.0")
        print(f"💡 {summary['recommendation']}")
        
        # Export historical report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        historical_file = comparator.output_dir / f"earnings_historical_comparison_{timestamp}.csv"
        comparator.export_comparison_report(historical_data, historical_file)
        print(f"\n💾 Historical report saved to: {historical_file}")

if __name__ == "__main__":
    main()