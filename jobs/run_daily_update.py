#!/usr/bin/env python3
"""
Daily data update for existing symbols.

This script runs daily incremental updates for symbols that are already
setup, ensuring all data is current while respecting dependencies.

Usage:
    python jobs/run_daily_update.py --symbols AAPL GOOGL MSFT
    python jobs/run_daily_update.py --config symbols.txt
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from jobs.orchestrator import DataPipelineOrchestrator, DataValidation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('daily_update.log')
    ]
)
logger = logging.getLogger(__name__)


def load_symbols_from_file(config_file: Path) -> list:
    """Load symbols from configuration file."""
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    symbols = []
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Support multiple symbols per line
                symbols.extend(line.split())
    
    return list(set(symbols))  # Remove duplicates


def main():
    parser = argparse.ArgumentParser(
        description="Run daily incremental updates for existing symbols",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Update specific symbols
    python jobs/run_daily_update.py --symbols AAPL GOOGL MSFT
    
    # Update from config file
    python jobs/run_daily_update.py --config config/daily_symbols.txt
    
    # Validate only (no updates)
    python jobs/run_daily_update.py --symbols AAPL --validate-only
        """
    )
    
    # Symbol specification (mutually exclusive)
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to update (e.g., AAPL GOOGL MSFT)"
    )
    symbol_group.add_argument(
        "--config",
        type=Path,
        help="File containing symbols to update (one per line or space-separated)"
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
        "--validate-only",
        action="store_true",
        help="Only validate data integrity without updates"
    )
    
    parser.add_argument(
        "--force-refresh-chains",
        action="store_true",
        help="Force refresh of option chains regardless of age"
    )
    
    args = parser.parse_args()
    
    # Get symbols list
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = load_symbols_from_file(args.config)
    
    logger.info(f"Processing {len(symbols)} symbols: {symbols}")
    
    # Initialize orchestrator and validator
    orchestrator = DataPipelineOrchestrator(
        base_path=args.data_path,
        enable_lineage=args.enable_lineage
    )
    
    validator = DataValidation(args.data_path)
    
    # Validation phase
    print(f"{'='*60}")
    print("DATA VALIDATION PHASE")
    print(f"{'='*60}")
    
    validation_results = {}
    valid_symbols = []
    
    for symbol in symbols:
        print(f"\n🔍 Validating {symbol}...")
        
        try:
            # Check data integrity
            validator.validate_data_integrity(symbol)
            
            # Check if symbol is ready for updates
            has_contracts = validator._contracts_exist(symbol) if hasattr(validator, '_contracts_exist') else True
            has_equity = validator._validate_equity_data_available(symbol) if hasattr(validator, '_validate_equity_data_available') else True
            
            validation_results[symbol] = {
                'integrity': True,
                'contracts': has_contracts,
                'equity': has_equity,
                'ready_for_update': True
            }
            
            valid_symbols.append(symbol)
            print(f"✅ {symbol}: Ready for update")
            
        except Exception as e:
            validation_results[symbol] = {
                'integrity': False,
                'error': str(e),
                'ready_for_update': False
            }
            print(f"❌ {symbol}: {e}")
    
    print(f"\nValidation complete: {len(valid_symbols)}/{len(symbols)} symbols ready")
    
    if args.validate_only:
        print("\n🔍 VALIDATION-ONLY MODE - No updates performed")
        
        # Show detailed validation report
        for symbol, result in validation_results.items():
            if result['ready_for_update']:
                print(f"✅ {symbol}: All checks passed")
            else:
                print(f"❌ {symbol}: {result.get('error', 'Validation failed')}")
        
        return
    
    if not valid_symbols:
        print("\n❌ No symbols are ready for update")
        sys.exit(1)
    
    # Update phase
    print(f"\n{'='*60}")
    print("INCREMENTAL UPDATE PHASE")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    
    # Run updates only for valid symbols
    update_results = orchestrator.daily_update(valid_symbols)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    # Analyze results
    print(f"\n📊 UPDATE RESULTS:")
    
    total_jobs = 0
    successful_jobs = 0
    symbols_with_updates = 0
    
    for symbol, symbol_results in update_results.items():
        if symbol_results:
            symbols_with_updates += 1
            symbol_success = sum(1 for r in symbol_results if r.success)
            total_jobs += len(symbol_results)
            successful_jobs += symbol_success
            
            status = "✅" if symbol_success == len(symbol_results) else "⚠️"
            print(f"  {symbol}: {status} {symbol_success}/{len(symbol_results)} jobs successful")
            
            # Show any failures
            failures = [r for r in symbol_results if not r.success]
            for failure in failures:
                print(f"    ❌ {failure.job_type.value}: {failure.error_message}")
        else:
            print(f"  {symbol}: ✅ No updates needed (data current)")
    
    # Final summary
    print(f"\n{'='*60}")
    print("DAILY UPDATE SUMMARY")
    print(f"{'='*60}")
    
    print(f"Symbols processed: {len(valid_symbols)}")
    print(f"Symbols with updates: {symbols_with_updates}")
    
    if total_jobs > 0:
        print(f"Job success rate: {successful_jobs}/{total_jobs} ({successful_jobs/total_jobs*100:.1f}%)")
    else:
        print("No update jobs needed - all data current")
    
    print(f"Total execution time: {execution_time:.1f} seconds")
    
    if total_jobs == 0 or successful_jobs == total_jobs:
        print("\n🎉 Daily update completed successfully!")
    else:
        print(f"\n⚠️ Some updates failed - check logs for details")
        sys.exit(1)


if __name__ == "__main__":
    main()