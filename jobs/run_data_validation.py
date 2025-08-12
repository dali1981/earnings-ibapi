#!/usr/bin/env python3
"""
Comprehensive data validation and health check.

This script validates data integrity, checks prerequisites, and generates
health reports for the trading data system.

Usage:
    python jobs/run_data_validation.py --symbols AAPL GOOGL
    python jobs/run_data_validation.py --config symbols.txt --full-report
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Any

from jobs.orchestrator import DataValidation, PrerequisiteError
from repositories import (
    EquityBarRepository,
    OptionBarRepository,
    OptionChainSnapshotRepository
)

# Lineage analysis if available
try:
    from lineage.query import LineageQueryEngine
    from lineage.visualizer import LineageVisualizer
    from lineage import get_global_tracker
    LINEAGE_AVAILABLE = True
except ImportError:
    LINEAGE_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataHealthChecker:
    """Comprehensive data health checker."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.validator = DataValidation(base_path)
        
        # Initialize repositories
        self.equity_repo = EquityBarRepository(base_path / "equity_bars")
        self.option_repo = OptionBarRepository(base_path / "option_bars") 
        self.chain_repo = OptionChainSnapshotRepository(base_path / "option_chains")
        
        # Initialize lineage if available
        if LINEAGE_AVAILABLE:
            self.tracker = get_global_tracker()
            if self.tracker:
                self.query_engine = LineageQueryEngine(self.tracker)
                self.visualizer = LineageVisualizer(self.tracker)
    
    def check_symbol_health(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive health check for a symbol."""
        health_report = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'checks': {}
        }
        
        # Check 1: Contract descriptions
        try:
            self.validator.validate_symbol_contracts_exist(symbol)
            health_report['checks']['contracts'] = {'status': 'ok', 'message': 'Available'}
        except PrerequisiteError as e:
            health_report['checks']['contracts'] = {'status': 'error', 'message': str(e)}
        
        # Check 2: Equity data
        try:
            self.validator.validate_equity_data_available(symbol, max_age_days=3)
            
            # Get equity data stats
            recent_date = date.today() - timedelta(days=7)
            present_dates = self.equity_repo.present_dates(symbol, "1 day", recent_date, date.today())
            
            health_report['checks']['equity_data'] = {
                'status': 'ok',
                'message': f'Current (last 7 days: {len(present_dates)} trading days)',
                'recent_coverage': len(present_dates)
            }
        except PrerequisiteError as e:
            health_report['checks']['equity_data'] = {'status': 'error', 'message': str(e)}
        except Exception as e:
            health_report['checks']['equity_data'] = {'status': 'warning', 'message': f'Check failed: {e}'}
        
        # Check 3: Option chain
        try:
            self.validator.validate_option_chain_current(symbol, max_age_hours=24)
            health_report['checks']['option_chain'] = {'status': 'ok', 'message': 'Current (<24h old)'}
        except PrerequisiteError as e:
            health_report['checks']['option_chain'] = {'status': 'warning', 'message': str(e)}
        
        # Check 4: Option prerequisites
        try:
            self.validator.validate_option_backfill_prerequisites(symbol)
            health_report['checks']['option_prerequisites'] = {'status': 'ok', 'message': 'All met'}
        except PrerequisiteError as e:
            health_report['checks']['option_prerequisites'] = {'status': 'error', 'message': str(e)}
        
        # Check 5: Data integrity
        try:
            self.validator.validate_data_integrity(symbol)
            health_report['checks']['data_integrity'] = {'status': 'ok', 'message': 'Consistent'}
        except Exception as e:
            health_report['checks']['data_integrity'] = {'status': 'error', 'message': str(e)}
        
        # Check 6: Option data availability (if prerequisites met)
        if health_report['checks']['option_prerequisites']['status'] == 'ok':
            try:
                option_dates = self.option_repo.get_available_dates(symbol)
                if option_dates:
                    health_report['checks']['option_data'] = {
                        'status': 'ok', 
                        'message': f'{len(option_dates)} days available',
                        'date_range': f"{min(option_dates)} to {max(option_dates)}"
                    }
                else:
                    health_report['checks']['option_data'] = {
                        'status': 'warning',
                        'message': 'Prerequisites met but no option data'
                    }
            except Exception as e:
                health_report['checks']['option_data'] = {
                    'status': 'warning',
                    'message': f'Check failed: {e}'
                }
        
        # Determine overall status
        check_statuses = [check['status'] for check in health_report['checks'].values()]
        if 'error' in check_statuses:
            health_report['overall_status'] = 'error'
        elif 'warning' in check_statuses:
            health_report['overall_status'] = 'warning'  
        else:
            health_report['overall_status'] = 'ok'
        
        return health_report
    
    def check_system_health(self) -> Dict[str, Any]:
        """System-wide health check."""
        system_report = {
            'timestamp': datetime.now().isoformat(),
            'repositories': {},
            'lineage': {},
            'storage': {}
        }
        
        # Repository status
        repos = {
            'equity_bars': self.equity_repo,
            'option_bars': self.option_repo,
            'option_chains': self.chain_repo
        }
        
        for name, repo in repos.items():
            try:
                stats = repo.get_stats()
                system_report['repositories'][name] = {
                    'status': 'ok',
                    'stats': stats
                }
            except Exception as e:
                system_report['repositories'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Lineage system status
        if LINEAGE_AVAILABLE and hasattr(self, 'query_engine'):
            try:
                lineage_summary = self.query_engine.build_lineage_summary()
                system_report['lineage'] = {
                    'status': 'ok',
                    'available': True,
                    'summary': lineage_summary.get('basic_stats', {})
                }
            except Exception as e:
                system_report['lineage'] = {
                    'status': 'error',
                    'available': True,
                    'error': str(e)
                }
        else:
            system_report['lineage'] = {
                'status': 'na',
                'available': False,
                'message': 'Lineage tracking not available'
            }
        
        # Storage health
        try:
            total_size = sum(
                sum(f.stat().st_size for f in Path(self.base_path).rglob("*.parquet"))
                for _ in [None]  # Single iteration
            )
            system_report['storage'] = {
                'status': 'ok',
                'total_size_mb': total_size / (1024 * 1024),
                'base_path': str(self.base_path.absolute())
            }
        except Exception as e:
            system_report['storage'] = {
                'status': 'warning',
                'error': str(e)
            }
        
        return system_report


def load_symbols_from_file(config_file: Path) -> List[str]:
    """Load symbols from configuration file."""
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    symbols = []
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                symbols.extend(line.split())
    
    return list(set(symbols))


def print_health_report(report: Dict[str, Any], verbose: bool = False):
    """Print formatted health report."""
    symbol = report['symbol']
    status = report['overall_status']
    
    status_emoji = {'ok': '✅', 'warning': '⚠️', 'error': '❌'}.get(status, '❓')
    print(f"{status_emoji} {symbol}: {status.upper()}")
    
    if verbose or status != 'ok':
        for check_name, check_result in report['checks'].items():
            check_status = check_result['status']
            check_emoji = {'ok': '✅', 'warning': '⚠️', 'error': '❌'}.get(check_status, '❓')
            print(f"  {check_emoji} {check_name}: {check_result['message']}")
    
    if verbose and status == 'ok':
        print(f"  📊 All {len(report['checks'])} checks passed")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive data validation and health check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick validation of specific symbols
    python jobs/run_data_validation.py --symbols AAPL GOOGL
    
    # Full report from config file
    python jobs/run_data_validation.py --config symbols.txt --full-report
    
    # System health check only
    python jobs/run_data_validation.py --system-only
    
    # Export results to JSON
    python jobs/run_data_validation.py --symbols AAPL --output health_report.json
        """
    )
    
    # Symbol specification
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Symbols to validate (e.g., AAPL GOOGL MSFT)"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="File containing symbols to validate"
    )
    
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data"),
        help="Base path for data storage (default: data)"
    )
    
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Show detailed reports for all symbols (not just failures)"
    )
    
    parser.add_argument(
        "--system-only",
        action="store_true",
        help="Only check system health, not individual symbols"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Export results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Initialize health checker
    health_checker = DataHealthChecker(args.data_path)
    
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'system_health': None,
        'symbol_reports': {},
        'summary': {}
    }
    
    # System health check
    if args.system_only or args.full_report:
        print(f"{'='*60}")
        print("SYSTEM HEALTH CHECK")
        print(f"{'='*60}")
        
        system_health = health_checker.check_system_health()
        validation_results['system_health'] = system_health
        
        # Print system status
        for category, status in system_health.items():
            if category == 'timestamp':
                continue
                
            if isinstance(status, dict) and 'status' in status:
                status_emoji = {'ok': '✅', 'warning': '⚠️', 'error': '❌', 'na': 'ℹ️'}.get(status['status'], '❓')
                print(f"{status_emoji} {category}: {status['status']}")
                
                if args.full_report and 'stats' in status:
                    for key, value in status['stats'].items():
                        if isinstance(value, (int, float)):
                            if key.endswith('_bytes'):
                                print(f"    {key}: {value:,} bytes ({value/(1024*1024):.1f} MB)")
                            else:
                                print(f"    {key}: {value:,}")
                        else:
                            print(f"    {key}: {value}")
    
    if args.system_only:
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            print(f"\nResults exported to: {args.output}")
        return
    
    # Get symbols to check
    if args.symbols:
        symbols = args.symbols
    elif args.config:
        symbols = load_symbols_from_file(args.config)
    else:
        print("❌ Must specify --symbols, --config, or --system-only")
        sys.exit(1)
    
    # Symbol validation
    print(f"\n{'='*60}")
    print(f"SYMBOL VALIDATION ({len(symbols)} symbols)")
    print(f"{'='*60}")
    
    healthy_symbols = []
    warning_symbols = []
    error_symbols = []
    
    for symbol in symbols:
        health_report = health_checker.check_symbol_health(symbol)
        validation_results['symbol_reports'][symbol] = health_report
        
        # Categorize by status
        status = health_report['overall_status']
        if status == 'ok':
            healthy_symbols.append(symbol)
        elif status == 'warning':
            warning_symbols.append(symbol)
        else:
            error_symbols.append(symbol)
        
        # Print report
        print_health_report(health_report, verbose=args.full_report)
    
    # Summary
    validation_results['summary'] = {
        'total_symbols': len(symbols),
        'healthy': len(healthy_symbols),
        'warnings': len(warning_symbols),
        'errors': len(error_symbols),
        'overall_health_rate': len(healthy_symbols) / len(symbols) * 100
    }
    
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    summary = validation_results['summary']
    print(f"Total symbols: {summary['total_symbols']}")
    print(f"✅ Healthy: {summary['healthy']}")
    print(f"⚠️ Warnings: {summary['warnings']}")
    print(f"❌ Errors: {summary['errors']}")
    print(f"📊 Health rate: {summary['overall_health_rate']:.1f}%")
    
    if error_symbols:
        print(f"\n❌ Symbols with errors: {', '.join(error_symbols)}")
    if warning_symbols:
        print(f"⚠️ Symbols with warnings: {', '.join(warning_symbols)}")
    
    # Export results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        print(f"\nResults exported to: {args.output}")
    
    # Exit code based on results
    if error_symbols:
        sys.exit(1)
    elif warning_symbols:
        sys.exit(2)  # Different exit code for warnings
    else:
        print("\n🎉 All symbols are healthy!")


if __name__ == "__main__":
    main()