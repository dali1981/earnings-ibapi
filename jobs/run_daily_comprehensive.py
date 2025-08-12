#!/usr/bin/env python3
"""
Comprehensive Daily Data Update System

Enterprise-grade daily update system that prioritizes symbols by upcoming earnings
and manages all data updates across the entire portfolio.

Features:
- Earnings-driven priority scheduling
- Multi-source symbol management
- Resource optimization
- Circuit breaker protection
- Comprehensive monitoring
- Recovery mechanisms

Usage:
    # Standard daily update
    python jobs/run_daily_comprehensive.py
    
    # Custom symbol portfolio
    python jobs/run_daily_comprehensive.py --config config/portfolio.yaml
    
    # Priority refresh only
    python jobs/run_daily_comprehensive.py --refresh-priorities-only
    
    # Monitoring mode
    python jobs/run_daily_comprehensive.py --monitor --status-report
"""

import argparse
import json
import logging
import sys
import yaml
from datetime import datetime, date, time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

# Core system components - relative imports for running from project root
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from jobs.orchestrator import DataPipelineOrchestrator
from earnings.scheduler import EarningsPriorityScheduler, SchedulePriority
from earnings.fetcher import EarningsCalendarFetcher

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'daily_comprehensive_{date.today().isoformat()}.log')
    ]
)
logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages symbol portfolios and configurations."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/portfolio.yaml")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load portfolio configuration."""
        if not self.config_path.exists():
            logger.info(f"No config found at {self.config_path}, using defaults")
            return self._create_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Loaded portfolio config from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default portfolio configuration."""
        return {
            'portfolios': {
                'main': {
                    'description': 'Main trading portfolio',
                    'symbols': [
                        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
                        'META', 'NVDA', 'NFLX', 'AMD', 'CRM'
                    ],
                    'enabled': True
                },
                'earnings_focus': {
                    'description': 'High-volume earnings plays',
                    'symbols': [
                        'SPY', 'QQQ', 'IWM', 'ARKK', 'XLF',
                        'XLE', 'XLK', 'XLV', 'GLD', 'TLT'
                    ],
                    'enabled': True
                }
            },
            'settings': {
                'max_daily_symbols': 100,
                'enable_earnings_priority': True,
                'circuit_breaker_enabled': True,
                'default_data_path': 'data',
                'earnings_cache_hours': 6,
                'api_rate_limit_buffer': 0.8
            },
            'schedule': {
                'market_hours_start': '09:30',
                'market_hours_end': '16:00',
                'pre_market_update_time': '08:00',
                'post_market_update_time': '17:00',
                'weekend_maintenance_time': '10:00'
            }
        }
    
    def save_config(self):
        """Save current configuration."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"💾 Saved portfolio config to {self.config_path}")
    
    def get_active_symbols(self) -> List[str]:
        """Get all active symbols from enabled portfolios."""
        all_symbols = set()
        
        for portfolio_name, portfolio in self.config.get('portfolios', {}).items():
            if portfolio.get('enabled', True):
                symbols = portfolio.get('symbols', [])
                all_symbols.update(symbols)
                logger.debug(f"Added {len(symbols)} symbols from {portfolio_name}")
        
        symbols_list = sorted(list(all_symbols))
        logger.info(f"📊 Active portfolio: {len(symbols_list)} unique symbols")
        
        return symbols_list
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get configuration setting."""
        return self.config.get('settings', {}).get(key, default)


class ComprehensiveUpdater:
    """Comprehensive daily update orchestrator."""
    
    def __init__(self, 
                 data_path: Path = Path("data"),
                 config_path: Path = None):
        
        self.data_path = data_path
        self.portfolio_manager = PortfolioManager(config_path)
        
        # Initialize core components
        self.orchestrator = DataPipelineOrchestrator(
            base_path=data_path,
            enable_lineage=True
        )
        
        self.earnings_fetcher = EarningsCalendarFetcher(
            cache_dir=data_path / "cache" / "earnings"
        )
        
        self.scheduler = EarningsPriorityScheduler(
            base_path=data_path,
            orchestrator=self.orchestrator,
            earnings_fetcher=self.earnings_fetcher
        )
        
        # Execution tracking
        self.execution_stats = {
            'start_time': None,
            'end_time': None,
            'symbols_processed': 0,
            'earnings_events_found': 0,
            'priority_distribution': {},
            'performance_metrics': {}
        }
        
        logger.info(f"🚀 Initialized comprehensive updater with data path: {data_path}")
    
    def run_daily_update(self, 
                        symbols: List[str] = None,
                        refresh_earnings: bool = True,
                        dry_run: bool = False) -> Dict[str, Any]:
        """Run comprehensive daily update."""
        
        self.execution_stats['start_time'] = datetime.now()
        logger.info("🌅 Starting comprehensive daily update")
        
        try:
            # Get symbols to process
            if symbols is None:
                symbols = self.portfolio_manager.get_active_symbols()
            
            max_symbols = self.portfolio_manager.get_setting('max_daily_symbols', 100)
            if len(symbols) > max_symbols:
                logger.warning(f"⚠️ Symbol count ({len(symbols)}) exceeds limit ({max_symbols}), truncating")
                symbols = symbols[:max_symbols]
            
            self.execution_stats['symbols_processed'] = len(symbols)
            
            # Refresh earnings priorities
            if refresh_earnings:
                logger.info("📅 Refreshing earnings calendar and priorities...")
                earnings_events = self.earnings_fetcher.get_upcoming_earnings(
                    symbols=symbols,
                    days_ahead=30
                )
                
                self.execution_stats['earnings_events_found'] = len(earnings_events)
                
                # Cache earnings data
                self.earnings_fetcher.save_to_cache(earnings_events)
                
                # Update scheduler priorities
                priorities = self.scheduler.refresh_earnings_priorities(symbols)
                self.execution_stats['priority_distribution'] = self._count_priorities(priorities)
            
            # Create and execute schedule
            logger.info("📋 Creating optimized execution schedule...")
            batches = self.scheduler.create_daily_schedule(symbols)
            
            if not batches:
                logger.info("✅ No updates needed - all data is current")
                return self._create_success_result("no_updates_needed")
            
            if dry_run:
                logger.info("🧪 DRY RUN MODE - No actual updates will be performed")
                self._log_dry_run_summary(batches)
                return self._create_success_result("dry_run_completed", batches)
            
            # Execute schedule
            logger.info(f"⚡ Executing {len(batches)} batches...")
            execution_results = self.scheduler.execute_daily_schedule(batches)
            
            # Analyze results
            self._analyze_execution_results(execution_results)
            
            return self._create_success_result("completed", execution_results)
            
        except Exception as e:
            logger.error(f"❌ Daily update failed: {e}")
            return self._create_error_result(str(e))
        
        finally:
            self.execution_stats['end_time'] = datetime.now()
            self._log_final_summary()
    
    def refresh_earnings_priorities(self, symbols: List[str] = None) -> Dict[str, SchedulePriority]:
        """Refresh earnings priorities only (no data updates)."""
        
        if symbols is None:
            symbols = self.portfolio_manager.get_active_symbols()
        
        logger.info(f"🔄 Refreshing earnings priorities for {len(symbols)} symbols")
        
        # Fetch latest earnings data
        earnings_events = self.earnings_fetcher.get_upcoming_earnings(
            symbols=symbols,
            days_ahead=30
        )
        
        # Update scheduler
        priorities = self.scheduler.refresh_earnings_priorities(symbols)
        
        # Log summary
        priority_counts = self._count_priorities(priorities)
        logger.info("📊 Updated priority distribution:")
        for priority, count in priority_counts.items():
            emoji = self.scheduler._get_priority_emoji(SchedulePriority(priority))
            logger.info(f"  {emoji} {priority}: {count} symbols")
        
        return priorities
    
    def get_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report."""
        
        portfolio_symbols = self.portfolio_manager.get_active_symbols()
        scheduler_status = self.scheduler.get_status_report()
        
        # System health checks
        health_checks = {
            'data_path_exists': self.data_path.exists(),
            'earnings_cache_fresh': self._check_earnings_cache_freshness(),
            'orchestrator_ready': hasattr(self.orchestrator, 'base_path'),
            'config_valid': len(portfolio_symbols) > 0
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio': {
                'total_symbols': len(portfolio_symbols),
                'sample_symbols': portfolio_symbols[:10],
                'config_path': str(self.portfolio_manager.config_path)
            },
            'scheduler': scheduler_status,
            'health_checks': health_checks,
            'last_execution': self.execution_stats if self.execution_stats['start_time'] else None
        }
    
    def _count_priorities(self, priorities: Dict[str, SchedulePriority]) -> Dict[str, int]:
        """Count symbols by priority level."""
        counts = {}
        for priority in priorities.values():
            priority_name = priority.value
            counts[priority_name] = counts.get(priority_name, 0) + 1
        return counts
    
    def _check_earnings_cache_freshness(self) -> bool:
        """Check if earnings cache is fresh enough."""
        cache_hours = self.portfolio_manager.get_setting('earnings_cache_hours', 6)
        cache_file = self.data_path / "cache" / "earnings" / f"earnings_{date.today().isoformat()}.json"
        
        if not cache_file.exists():
            return False
        
        cache_age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
        return cache_age_hours < cache_hours
    
    def _log_dry_run_summary(self, batches):
        """Log dry run summary."""
        logger.info("🧪 DRY RUN SUMMARY:")
        
        total_symbols = sum(batch.total_symbols for batch in batches)
        estimated_duration = sum(batch.estimated_duration_minutes for batch in batches)
        
        logger.info(f"  Would process: {total_symbols} symbols")
        logger.info(f"  Estimated time: {estimated_duration} minutes")
        logger.info(f"  Batch breakdown:")
        
        for i, batch in enumerate(batches, 1):
            emoji = self.scheduler._get_priority_emoji(batch.priority)
            logger.info(f"    {emoji} Batch {i}: {batch.priority.value} - {batch.total_symbols} symbols")
    
    def _analyze_execution_results(self, results: Dict[str, Any]):
        """Analyze and log execution results."""
        
        total_duration = results.get('total_duration', 0)
        success_rate = results['successful_symbols'] / max(results['total_symbols_processed'], 1) * 100
        
        self.execution_stats['performance_metrics'] = {
            'total_duration_seconds': total_duration,
            'symbols_per_minute': results['total_symbols_processed'] / max(total_duration / 60, 1),
            'success_rate_percent': success_rate,
            'batches_executed': len(results['batches'])
        }
    
    def _log_final_summary(self):
        """Log final execution summary."""
        if not self.execution_stats['start_time']:
            return
        
        duration = (self.execution_stats['end_time'] - self.execution_stats['start_time']).total_seconds() / 60
        
        logger.info("🎯 COMPREHENSIVE UPDATE SUMMARY:")
        logger.info(f"   Total duration: {duration:.1f} minutes")
        logger.info(f"   Symbols processed: {self.execution_stats['symbols_processed']}")
        logger.info(f"   Earnings events: {self.execution_stats['earnings_events_found']}")
        
        if self.execution_stats['priority_distribution']:
            logger.info("   Priority distribution:")
            for priority, count in self.execution_stats['priority_distribution'].items():
                logger.info(f"     {priority}: {count}")
    
    def _create_success_result(self, status: str, data: Any = None) -> Dict[str, Any]:
        """Create success result dictionary."""
        return {
            'success': True,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'execution_stats': self.execution_stats,
            'data': data
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result dictionary."""
        return {
            'success': False,
            'status': 'error',
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'execution_stats': self.execution_stats
        }


def main():
    """Main entry point for comprehensive daily updates."""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive daily data update with earnings prioritization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard daily update
    python jobs/run_daily_comprehensive.py
    
    # Use custom portfolio config
    python jobs/run_daily_comprehensive.py --config config/my_portfolio.yaml
    
    # Dry run to preview actions
    python jobs/run_daily_comprehensive.py --dry-run
    
    # Only refresh earnings priorities
    python jobs/run_daily_comprehensive.py --refresh-priorities-only
    
    # Status report and monitoring
    python jobs/run_daily_comprehensive.py --status-report
    
    # Custom symbol list
    python jobs/run_daily_comprehensive.py --symbols AAPL GOOGL MSFT
        """
    )
    
    # Configuration options
    parser.add_argument(
        "--config",
        type=Path,
        help="Portfolio configuration file (YAML format)"
    )
    
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data"),
        help="Base path for data storage (default: data)"
    )
    
    # Symbol specification
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Specific symbols to update (overrides portfolio config)"
    )
    
    # Operation modes
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without making changes"
    )
    
    parser.add_argument(
        "--refresh-priorities-only",
        action="store_true",
        help="Only refresh earnings priorities, no data updates"
    )
    
    parser.add_argument(
        "--status-report",
        action="store_true", 
        help="Generate status report and exit"
    )
    
    parser.add_argument(
        "--no-earnings-refresh",
        action="store_true",
        help="Skip earnings data refresh (use cached data)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        help="Export results to JSON file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize updater
    updater = ComprehensiveUpdater(
        data_path=args.data_path,
        config_path=args.config
    )
    
    try:
        # Handle different operation modes
        if args.status_report:
            print("📊 SYSTEM STATUS REPORT")
            print("=" * 60)
            
            status = updater.get_status_report()
            
            # Print formatted status
            print(f"Timestamp: {status['timestamp']}")
            print(f"\nPortfolio:")
            print(f"  Total symbols: {status['portfolio']['total_symbols']}")
            print(f"  Config path: {status['portfolio']['config_path']}")
            
            print(f"\nHealth Checks:")
            for check, result in status['health_checks'].items():
                emoji = "✅" if result else "❌"
                print(f"  {emoji} {check}")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(status, f, indent=2, default=str)
                print(f"\n💾 Status exported to: {args.output}")
            
            return
        
        elif args.refresh_priorities_only:
            print("🔄 REFRESHING EARNINGS PRIORITIES")
            print("=" * 60)
            
            priorities = updater.refresh_earnings_priorities(args.symbols)
            
            print(f"\n✅ Updated priorities for {len(priorities)} symbols")
            
            if args.output:
                priority_data = {symbol: priority.value for symbol, priority in priorities.items()}
                with open(args.output, 'w') as f:
                    json.dump(priority_data, f, indent=2)
                print(f"💾 Priorities exported to: {args.output}")
            
            return
        
        else:
            # Run comprehensive daily update
            print("🌅 COMPREHENSIVE DAILY UPDATE")
            print("=" * 60)
            
            result = updater.run_daily_update(
                symbols=args.symbols,
                refresh_earnings=not args.no_earnings_refresh,
                dry_run=args.dry_run
            )
            
            if result['success']:
                print(f"\n🎉 Update completed successfully: {result['status']}")
            else:
                print(f"\n❌ Update failed: {result['error']}")
                sys.exit(1)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"💾 Results exported to: {args.output}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Update interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        logger.exception("Unexpected error in main")
        sys.exit(1)


if __name__ == "__main__":
    main()