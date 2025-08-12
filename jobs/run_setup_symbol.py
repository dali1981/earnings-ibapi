#!/usr/bin/env python3
"""
Setup new symbol with complete data dependencies.

This script sets up a new symbol with all required data dependencies
in the correct order using the orchestrator.

Usage:
    python jobs/run_setup_symbol.py AAPL --lookback-days 365 --data-path data
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from jobs.orchestrator import DataPipelineOrchestrator, PrerequisiteError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('setup_symbol.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Setup new symbol with complete data dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Setup AAPL with 1 year of history
    python jobs/run_setup_symbol.py AAPL --lookback-days 365
    
    # Setup multiple symbols with 6 months history  
    python jobs/run_setup_symbol.py AAPL GOOGL MSFT --lookback-days 180
    
    # Setup with custom data path
    python jobs/run_setup_symbol.py TSLA --data-path /custom/data/path
        """
    )
    
    parser.add_argument(
        "symbols",
        nargs="+",
        help="Symbol(s) to setup (e.g., AAPL GOOGL MSFT)"
    )
    
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Number of days of historical data to backfill (default: 365)"
    )
    
    parser.add_argument(
        "--data-path", 
        type=Path,
        default=Path("data"),
        help="Base path for data storage (default: data)"
    )
    
    parser.add_argument(
        "--enable-lineage",
        action="store_true",
        default=True,
        help="Enable data lineage tracking (default: True)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No actual data requests will be made")
        print(f"Would setup symbols: {args.symbols}")
        print(f"Lookback days: {args.lookback_days}")
        print(f"Data path: {args.data_path}")
        print(f"Lineage tracking: {args.enable_lineage}")
        return
    
    # Initialize orchestrator
    logger.info(f"Initializing orchestrator with data path: {args.data_path}")
    orchestrator = DataPipelineOrchestrator(
        base_path=args.data_path,
        enable_lineage=args.enable_lineage
    )
    
    # Process each symbol
    all_results = {}
    overall_success = True
    
    for symbol in args.symbols:
        logger.info(f"Starting setup for {symbol}")
        
        try:
            start_time = datetime.now()
            
            # Setup symbol with all dependencies
            results = orchestrator.setup_new_symbol(symbol, lookback_days=args.lookback_days)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Analyze results
            success_count = sum(1 for r in results if r.success)
            total_count = len(results)
            
            all_results[symbol] = {
                'results': results,
                'success_rate': success_count / total_count if total_count > 0 else 0,
                'execution_time': execution_time
            }
            
            if success_count == total_count:
                logger.info(f"✅ {symbol}: Successfully completed all {total_count} jobs in {execution_time:.1f}s")
            else:
                logger.warning(f"⚠️ {symbol}: {success_count}/{total_count} jobs successful in {execution_time:.1f}s")
                overall_success = False
                
                # Show failures
                failures = [r for r in results if not r.success]
                for failure in failures:
                    logger.error(f"   ❌ {failure.job_type.value}: {failure.error_message}")
            
        except PrerequisiteError as e:
            logger.error(f"❌ {symbol}: Prerequisites not met - {e}")
            all_results[symbol] = {'error': str(e), 'success_rate': 0}
            overall_success = False
            
        except Exception as e:
            logger.error(f"❌ {symbol}: Setup failed - {e}")
            all_results[symbol] = {'error': str(e), 'success_rate': 0}
            overall_success = False
    
    # Final summary
    print(f"\n{'='*60}")
    print("SETUP SUMMARY")
    print(f"{'='*60}")
    
    successful_symbols = []
    failed_symbols = []
    
    for symbol, result in all_results.items():
        success_rate = result.get('success_rate', 0)
        if success_rate == 1.0:
            successful_symbols.append(symbol)
            execution_time = result.get('execution_time', 0)
            print(f"✅ {symbol}: Complete ({execution_time:.1f}s)")
        else:
            failed_symbols.append(symbol)
            error = result.get('error', f"Partial success ({success_rate*100:.0f}%)")
            print(f"❌ {symbol}: {error}")
    
    print(f"\nSuccessful: {len(successful_symbols)} symbols")
    print(f"Failed: {len(failed_symbols)} symbols")
    
    if overall_success:
        print("\n🎉 All symbols setup successfully!")
        print(f"Data stored in: {args.data_path.absolute()}")
    else:
        print("\n⚠️ Some symbols failed - check logs for details")
        sys.exit(1)


if __name__ == "__main__":
    main()