#!/usr/bin/env python3
"""
Daily Earnings Data Collection Job

Automated job to collect, persist, and maintain earnings calendar data from multiple sources.
Designed to run daily (preferably early morning) to ensure fresh data for trading decisions.

Features:
- Multi-source earnings data collection (NASDAQ primary)
- Automatic data persistence with partitioning
- Data quality validation and reporting
- Historical data maintenance
- Error handling and recovery
- Performance monitoring and alerting

Usage:
    python jobs/daily_earnings_collection.py
    python jobs/daily_earnings_collection.py --days-ahead 30 --force-refresh
    python jobs/daily_earnings_collection.py --validate-only
"""

import sys
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import EARNINGS_PATH
from utils.logging_setup import get_logger
from earnings.fetcher import EarningsCalendarFetcher, EarningsSource
from repositories.earnings import EarningsRepository
from reliability.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from reliability.retry import retry
from lineage.decorators import track_lineage

logger = get_logger(__name__)


class DailyEarningsCollector:
    """Daily earnings data collection and persistence manager."""
    
    def __init__(self, 
                 data_path: Path = None,
                 max_retries: int = 3):
        self.fetcher = EarningsCalendarFetcher()
        # Use configured path if not provided
        if data_path is None:
            data_path = EARNINGS_PATH.parent
            logger.info(f"📁 Using configured earnings path: {EARNINGS_PATH}")
        self.repository = EarningsRepository(data_path)
        self.max_retries = max_retries
        
        # Circuit breakers for different data sources
        self.circuit_breakers = {
            EarningsSource.NASDAQ: CircuitBreaker(
                name="nasdaq",
                config=CircuitBreakerConfig(
                    failure_threshold=3,
                    timeout_duration=300  # 5 minutes
                )
            ),
            EarningsSource.FMP: CircuitBreaker(
                name="fmp",
                config=CircuitBreakerConfig(
                    failure_threshold=2,
                    timeout_duration=600  # 10 minutes  
                )
            ),
            EarningsSource.FINNHUB: CircuitBreaker(
                name="finnhub",
                config=CircuitBreakerConfig(
                    failure_threshold=2,
                    timeout_duration=600
                )
            )
        }
        
        # Collection configuration
        self.default_days_ahead = 60  # Collect 2 months ahead
        self.data_sources = [
            EarningsSource.NASDAQ,    # Primary (free, reliable)
            EarningsSource.FMP,       # Secondary (premium features)
            EarningsSource.FINNHUB    # Tertiary (good coverage)
        ]
        
        logger.info("🚀 Daily earnings collector initialized")
    
    def run_daily_collection(self,
                           days_ahead: int = None,
                           force_refresh: bool = False,
                           validate_only: bool = False) -> Dict[str, Any]:
        """
        Run the complete daily earnings collection process.
        
        Args:
            days_ahead: Number of days ahead to collect (default: 60)
            force_refresh: Force collection even if recent data exists
            validate_only: Only validate existing data, don't collect new
            
        Returns:
            Dictionary with collection results and metrics
        """
        start_time = datetime.now()
        
        if days_ahead is None:
            days_ahead = self.default_days_ahead
        
        collection_results = {
            "collection_date": date.today().isoformat(),
            "start_time": start_time.isoformat(),
            "days_ahead": days_ahead,
            "force_refresh": force_refresh,
            "validate_only": validate_only,
            "sources_attempted": [],
            "sources_successful": [],
            "total_events_collected": 0,
            "data_quality_score": 0.0,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Step 1: Check if collection needed
            if not validate_only and not force_refresh:
                if self._is_recent_collection_available():
                    logger.info("✅ Recent collection found, skipping. Use --force-refresh to override.")
                    collection_results["skipped"] = True
                    collection_results["reason"] = "Recent data available"
                    return collection_results
            
            # Step 2: Validate existing data
            validation_results = self._validate_existing_data()
            collection_results["validation"] = validation_results
            
            if validate_only:
                logger.info("📊 Validation complete")
                return collection_results
            
            # Step 3: Collect new earnings data
            logger.info(f"🔍 Starting earnings collection for next {days_ahead} days")
            
            all_earnings = []
            successful_sources = []
            
            for source in self.data_sources:
                collection_results["sources_attempted"].append(source.value)
                
                try:
                    # Check circuit breaker state
                    circuit_breaker = self.circuit_breakers.get(source)
                    if circuit_breaker and circuit_breaker.get_state().value == "open":
                        logger.warning(f"⚠️ Circuit breaker OPEN for {source.value}, skipping")
                        collection_results["warnings"].append(f"Circuit breaker open for {source.value}")
                        continue
                    
                    # Collect from this source using circuit breaker
                    if circuit_breaker:
                        source_earnings = circuit_breaker.call(
                            self._collect_from_source_direct, source, days_ahead
                        )
                    else:
                        source_earnings = self._collect_from_source_direct(source, days_ahead)
                    
                    if source_earnings:
                        all_earnings.extend(source_earnings)
                        successful_sources.append(source)
                        collection_results["sources_successful"].append(source.value)
                        logger.info(f"✅ {source.value}: {len(source_earnings)} events")
                    else:
                        logger.warning(f"⚠️ {source.value}: No data returned")
                        collection_results["warnings"].append(f"No data from {source.value}")
                
                except Exception as e:
                    error_msg = f"{source.value} collection failed: {str(e)}"
                    logger.error(f"❌ {error_msg}")
                    collection_results["errors"].append(error_msg)
                    continue
            
            # Step 4: Process and store collected data
            if all_earnings:
                # Deduplicate earnings events
                unique_earnings = self._deduplicate_earnings(all_earnings)
                collection_results["total_events_collected"] = len(unique_earnings)
                collection_results["duplicate_events_removed"] = len(all_earnings) - len(unique_earnings)
                
                # Calculate data quality score
                quality_score = self._calculate_data_quality(unique_earnings)
                collection_results["data_quality_score"] = quality_score
                
                # Store in repository
                storage_result = self.repository.store_earnings_batch(
                    earnings=unique_earnings,
                    collection_date=date.today(),
                    metadata={
                        "collection_job": "daily_earnings_collection",
                        "sources_used": [s.value for s in successful_sources],
                        "days_ahead": days_ahead,
                        "quality_score": quality_score
                    }
                )
                
                collection_results["storage"] = storage_result
                logger.info(f"💾 Stored {len(unique_earnings)} unique earnings events")
                
            else:
                logger.error("❌ No earnings data collected from any source")
                collection_results["errors"].append("No data collected from any source")
            
            # Step 5: Cleanup and maintenance
            self._perform_maintenance()
            
            # Step 6: Generate summary report
            processing_time = (datetime.now() - start_time).total_seconds()
            collection_results["processing_time_seconds"] = processing_time
            collection_results["status"] = "success" if all_earnings else "failed"
            
            # Log summary
            self._log_collection_summary(collection_results)
            
            return collection_results
            
        except Exception as e:
            error_msg = f"Daily collection job failed: {str(e)}"
            logger.error(f"💥 {error_msg}")
            collection_results["errors"].append(error_msg)
            collection_results["status"] = "error"
            return collection_results
    
    def _collect_from_source_direct(self, 
                                   source: EarningsSource, 
                                   days_ahead: int) -> List:
        """Collect earnings data directly from a specific source."""
        
        return self.fetcher.get_upcoming_earnings(
            symbols=None,
            days_ahead=days_ahead,
            sources=[source]
        )
    
    def _is_recent_collection_available(self, max_age_hours: int = 6) -> bool:
        """Check if a recent collection is already available."""
        
        try:
            latest_data = self.repository.get_latest_collection()
            if latest_data is None:
                return False
            
            # Check collection age
            latest_collection_date = latest_data['collection_date'].iloc[0]
            if isinstance(latest_collection_date, str):
                latest_collection_date = datetime.strptime(latest_collection_date, '%Y-%m-%d').date()
            
            age_hours = (date.today() - latest_collection_date).total_seconds() / 3600
            
            if age_hours < max_age_hours:
                logger.info(f"📊 Recent collection found ({age_hours:.1f} hours old)")
                return True
                
        except Exception as e:
            logger.warning(f"Could not check recent collection: {e}")
        
        return False
    
    def _validate_existing_data(self) -> Dict[str, Any]:
        """Validate existing earnings data quality and coverage."""
        
        validation_results = {
            "total_collections": 0,
            "date_coverage_days": 0,
            "data_quality_issues": [],
            "recommendations": []
        }
        
        try:
            # Get storage statistics
            stats = self.repository.get_storage_stats()
            validation_results["storage_stats"] = stats
            validation_results["total_collections"] = stats["total_collections"]
            
            # Check date coverage
            if stats["date_range"]["start"] and stats["date_range"]["end"]:
                start_date = datetime.strptime(stats["date_range"]["start"], '%Y-%m-%d').date()
                end_date = datetime.strptime(stats["date_range"]["end"], '%Y-%m-%d').date()
                validation_results["date_coverage_days"] = (end_date - start_date).days
            
            # Validate data quality
            latest_data = self.repository.get_latest_collection()
            if latest_data is not None:
                quality_issues = self._identify_quality_issues(latest_data)
                validation_results["data_quality_issues"] = quality_issues
                
                # Generate recommendations
                recommendations = self._generate_recommendations(stats, quality_issues)
                validation_results["recommendations"] = recommendations
            
            validation_results["status"] = "completed"
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results["status"] = "error"
            validation_results["error"] = str(e)
        
        return validation_results
    
    def _identify_quality_issues(self, data) -> List[str]:
        """Identify data quality issues in earnings data."""
        
        issues = []
        
        try:
            # Check for missing EPS estimates
            missing_eps = data['eps_estimate'].isna().sum()
            if missing_eps > len(data) * 0.3:  # More than 30% missing
                issues.append(f"High rate of missing EPS estimates: {missing_eps}/{len(data)} ({missing_eps/len(data)*100:.1f}%)")
            
            # Check for missing timing information
            missing_timing = (data['time'].isna() | (data['time'] == 'unknown')).sum()
            if missing_timing > len(data) * 0.2:  # More than 20% missing
                issues.append(f"High rate of missing timing info: {missing_timing}/{len(data)} ({missing_timing/len(data)*100:.1f}%)")
            
            # Check for old data
            today = date.today()
            old_earnings = data[data['earnings_date'] < today]
            if len(old_earnings) > len(data) * 0.5:  # More than 50% old
                issues.append(f"High rate of past earnings: {len(old_earnings)}/{len(data)} ({len(old_earnings)/len(data)*100:.1f}%)")
            
            # Check for data source concentration
            source_counts = data['source'].value_counts()
            if len(source_counts) == 1:
                issues.append(f"Single data source dependency: {source_counts.index[0]}")
            
        except Exception as e:
            issues.append(f"Quality analysis failed: {str(e)}")
        
        return issues
    
    def _generate_recommendations(self, stats, quality_issues) -> List[str]:
        """Generate recommendations based on data analysis."""
        
        recommendations = []
        
        # Storage recommendations
        if stats["total_events"] < 100:
            recommendations.append("Low data volume - consider increasing collection frequency")
        
        if stats["unique_symbols"] < 50:
            recommendations.append("Limited symbol coverage - verify data source connectivity")
        
        # Quality recommendations
        if "missing EPS estimates" in str(quality_issues):
            recommendations.append("Consider adding FMP or Finnhub as secondary sources for EPS data")
        
        if "missing timing info" in str(quality_issues):
            recommendations.append("Enable timing data collection from NASDAQ API")
        
        if "Single data source" in str(quality_issues):
            recommendations.append("Add backup data sources for redundancy")
        
        # Performance recommendations
        if stats["total_size_mb"] > 1000:  # 1GB
            recommendations.append("Consider implementing data archival for old collections")
        
        return recommendations
    
    def _deduplicate_earnings(self, earnings_list) -> List:
        """Remove duplicate earnings events across sources."""
        
        seen = {}
        unique_earnings = []
        
        for event in earnings_list:
            key = (event.symbol, event.earnings_date)
            
            if key not in seen:
                seen[key] = event
                unique_earnings.append(event)
            else:
                # Keep the event with better data quality
                existing = seen[key]
                if self._compare_event_quality(event, existing) > 0:
                    # Replace with better quality event
                    unique_earnings[unique_earnings.index(existing)] = event
                    seen[key] = event
        
        logger.info(f"📊 Deduplicated {len(earnings_list)} → {len(unique_earnings)} events")
        return unique_earnings
    
    def _compare_event_quality(self, event1, event2) -> int:
        """Compare quality of two earnings events. Returns 1 if event1 is better, -1 if event2 is better, 0 if equal."""
        
        score1 = 0
        score2 = 0
        
        # EPS estimate availability
        if event1.eps_estimate is not None:
            score1 += 2
        if event2.eps_estimate is not None:
            score2 += 2
        
        # Timing information
        if event1.time and event1.time != 'unknown':
            score1 += 1
        if event2.time and event2.time != 'unknown':
            score2 += 1
        
        # Market cap information (NASDAQ specific)
        if hasattr(event1, 'market_cap') and event1.market_cap:
            score1 += 1
        if hasattr(event2, 'market_cap') and event2.market_cap:
            score2 += 1
        
        # Source preference (NASDAQ preferred for coverage, FMP for detail)
        source_preference = {'nasdaq': 3, 'financial_modeling_prep': 2, 'finnhub': 1}
        score1 += source_preference.get(event1.source, 0)
        score2 += source_preference.get(event2.source, 0)
        
        if score1 > score2:
            return 1
        elif score2 > score1:
            return -1
        else:
            return 0
    
    def _calculate_data_quality(self, earnings_list) -> float:
        """Calculate overall data quality score (0.0 to 1.0)."""
        
        if not earnings_list:
            return 0.0
        
        total_score = 0.0
        max_possible_score = 0.0
        
        for event in earnings_list:
            event_score = 0.0
            event_max = 4.0  # Maximum possible score per event
            
            # EPS estimate availability (40% weight)
            if event.eps_estimate is not None:
                event_score += 1.6
            
            # Timing information (30% weight)
            if event.time and event.time != 'unknown':
                event_score += 1.2
            
            # Market cap information (20% weight)
            if hasattr(event, 'market_cap') and event.market_cap:
                event_score += 0.8
            
            # Recent/upcoming timing (10% weight)
            if abs(event.days_until_earnings) <= 30:
                event_score += 0.4
            
            total_score += event_score
            max_possible_score += event_max
        
        quality_score = total_score / max_possible_score if max_possible_score > 0 else 0.0
        return round(quality_score, 3)
    
    def _perform_maintenance(self):
        """Perform routine maintenance tasks."""
        
        try:
            # Repository handles its own cleanup via retention policy
            logger.info("🧹 Performing maintenance tasks")
            
            # Additional maintenance tasks could go here:
            # - Compress old data
            # - Generate performance reports
            # - Update data source health metrics
            
        except Exception as e:
            logger.error(f"Maintenance failed: {e}")
    
    def _log_collection_summary(self, results):
        """Log a summary of the collection results."""
        
        status_emoji = "✅" if results.get("status") == "success" else "❌"
        
        logger.info(f"{status_emoji} DAILY EARNINGS COLLECTION SUMMARY")
        logger.info(f"   Date: {results['collection_date']}")
        logger.info(f"   Events collected: {results['total_events_collected']}")
        logger.info(f"   Sources successful: {len(results['sources_successful'])}/{len(results['sources_attempted'])}")
        logger.info(f"   Data quality: {results['data_quality_score']:.1%}")
        logger.info(f"   Processing time: {results.get('processing_time_seconds', 0):.1f}s")
        
        if results.get("errors"):
            logger.warning(f"   Errors: {len(results['errors'])}")
        
        if results.get("warnings"):
            logger.info(f"   Warnings: {len(results['warnings'])}")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Daily earnings data collection job")
    
    parser.add_argument('--days-ahead', type=int, default=60,
                       help='Days ahead to collect earnings data (default: 60)')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force collection even if recent data exists')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing data, don\'t collect new')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path for earnings data storage (uses config default if not provided)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--output-report', type=str,
                       help='Output file for collection report (JSON)')
    
    args = parser.parse_args()
    
    # Setup centralized logging
    from utils.logging_setup import setup_logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    print("🚀 DAILY EARNINGS DATA COLLECTION")
    print("=" * 50)
    
    # Initialize collector
    data_path = Path(args.data_path) if args.data_path else None
    collector = DailyEarningsCollector(data_path=data_path)
    
    # Run collection
    results = collector.run_daily_collection(
        days_ahead=args.days_ahead,
        force_refresh=args.force_refresh,
        validate_only=args.validate_only
    )
    
    # Display results
    print(f"\n📊 COLLECTION RESULTS")
    print(f"Status: {results.get('status', 'unknown')}")
    print(f"Events collected: {results.get('total_events_collected', 0)}")
    print(f"Data quality: {results.get('data_quality_score', 0):.1%}")
    
    if results.get('errors'):
        print(f"\n❌ ERRORS ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"   • {error}")
    
    if results.get('warnings'):
        print(f"\n⚠️ WARNINGS ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"   • {warning}")
    
    # Save report if requested
    if args.output_report:
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Collection report saved to: {report_path}")
    
    # Exit with appropriate code
    exit_code = 0 if results.get('status') == 'success' else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()