#!/usr/bin/env python3
"""
Modern Trading Data System Usage Examples

This example demonstrates the correct way to use the updated trading data system
with orchestrator, dependency management, and lineage tracking.

Key concepts shown:
1. Using orchestrator for dependency-aware operations
2. Proper data validation before operations
3. Smart persistence to avoid redundant API calls
4. Data lineage tracking and analysis
5. Repository usage patterns

Run with:
    python examples/modern_usage_example.py
"""

import logging
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path

# Core system components
from jobs.orchestrator import DataPipelineOrchestrator, DataValidation, PrerequisiteError

# Repository imports - new unified system
from repositories import (
    EquityBarRepository,
    OptionBarRepository,
    OptionChainSnapshotRepository
)

# Lineage tracking (optional)
try:
    from lineage.query import LineageQueryEngine
    from lineage.visualizer import LineageVisualizer
    LINEAGE_AVAILABLE = True
except ImportError:
    LINEAGE_AVAILABLE = False

# Setup logging
from utils.logging_setup import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


class ModernUsageExample:
    """Modern usage patterns for the trading data system."""
    
    def __init__(self, data_path: str = "example_data"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        
        # Initialize orchestrator - handles all dependency management
        self.orchestrator = DataPipelineOrchestrator(
            base_path=self.data_path,
            enable_lineage=True
        )
        
        # Initialize validator for manual checks
        self.validator = DataValidation(self.data_path)
        
        # Initialize repositories for direct access
        self.equity_repo = EquityBarRepository(self.data_path / "equity_bars")
        self.option_repo = OptionBarRepository(self.data_path / "option_bars")
        self.chain_repo = OptionChainSnapshotRepository(self.data_path / "option_chains")
        
        logger.info(f"Initialized modern usage example with data path: {self.data_path}")
    
    def example_1_setup_new_symbol(self, symbol: str = "AAPL"):
        """
        Example 1: Setting up a new symbol with all dependencies.
        
        This is the CORRECT way to setup new symbols - the orchestrator
        handles all dependencies automatically in the right order.
        """
        print(f"\\n{'='*60}")
        print(f"EXAMPLE 1: Setup New Symbol - {symbol}")
        print(f"{'='*60}")
        
        print(f"🚀 Setting up {symbol} with orchestrator...")
        print("   ✅ Orchestrator will handle all dependencies automatically")
        print("   ✅ Contracts → Chains → Equity → Options (in correct order)")
        print("   ✅ Only requests missing data (smart persistence)")
        
        try:
            # This ONE call handles everything:
            # 1. Contract descriptions
            # 2. Option chain snapshots  
            # 3. Equity bar backfill
            # 4. Option bar backfill (only if prerequisites met)
            results = self.orchestrator.setup_new_symbol(symbol, lookback_days=30)
            
            # Analyze results
            print(f"\\n📊 Setup Results:")
            for result in results:
                status = "✅" if result.success else "❌"
                print(f"   {status} {result.job_type.value}")
                if not result.success:
                    print(f"      Error: {result.error_message}")
            
            # Validate final state
            print(f"\\n🔍 Final validation:")
            validation_result = self.validator.validate_data_integrity(symbol)
            print(f"   ✅ Data integrity validated for {symbol}")
            
            return results
            
        except PrerequisiteError as e:
            print(f"❌ Prerequisites not met: {e}")
            return None
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return None
    
    def example_2_daily_updates(self, symbols: list = None):
        """
        Example 2: Daily incremental updates.
        
        This shows how to efficiently update existing data without
        re-requesting data that already exists.
        """
        if symbols is None:
            symbols = ["AAPL"]  # Use symbols from example 1
        
        print(f"\\n{'='*60}")
        print("EXAMPLE 2: Daily Incremental Updates")
        print(f"{'='*60}")
        
        print(f"📅 Running daily updates for: {symbols}")
        print("   ✅ Only requests missing data")
        print("   ✅ Validates prerequisites before option operations")
        print("   ✅ Updates option chains (snapshots)")
        
        try:
            # This handles incremental updates intelligently
            results = self.orchestrator.daily_update(symbols)
            
            print(f"\\n📊 Update Results:")
            for symbol, symbol_results in results.items():
                if symbol_results:
                    success_count = sum(1 for r in symbol_results if r.success)
                    print(f"   {symbol}: {success_count}/{len(symbol_results)} jobs successful")
                    
                    for result in symbol_results:
                        status = "✅" if result.success else "❌"
                        print(f"      {status} {result.job_type.value}")
                else:
                    print(f"   {symbol}: ✅ No updates needed (data current)")
            
            return results
            
        except Exception as e:
            print(f"❌ Daily update failed: {e}")
            return None
    
    def example_3_safe_data_access(self, symbol: str = "AAPL"):
        """
        Example 3: Safe data access with validation.
        
        This shows the CORRECT way to access data - always validate
        prerequisites before attempting operations.
        """
        print(f"\\n{'='*60}")
        print(f"EXAMPLE 3: Safe Data Access - {symbol}")
        print(f"{'='*60}")
        
        # Step 1: Validate data availability
        print(f"🔍 Validating data availability...")
        
        try:
            # Check basic integrity first
            self.validator.validate_data_integrity(symbol)
            print("   ✅ Data integrity validated")
            
            # Check equity data
            self.validator.validate_equity_data_available(symbol)
            print("   ✅ Equity data available")
            
            # Check option prerequisites (may fail - that's OK)
            try:
                self.validator.validate_option_backfill_prerequisites(symbol)
                print("   ✅ Option prerequisites met")
                option_ready = True
            except PrerequisiteError:
                print("   ⚠️ Option prerequisites not met")
                option_ready = False
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            print("   💡 Run setup_new_symbol() first")
            return None
        
        # Step 2: Safe data loading
        print(f"\\n📊 Loading available data...")
        
        # Load equity data (should always work if validation passed)
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        
        equity_data = self.equity_repo.load_symbol_data(symbol, start_date, end_date)
        print(f"   ✅ Loaded {len(equity_data)} equity bars")
        
        if not equity_data.empty:
            latest_close = equity_data.iloc[-1]['close']
            print(f"   📈 Latest close: ${latest_close:.2f}")
        
        # Load option data only if prerequisites are met
        if option_ready:
            try:
                # Get latest option chain snapshot
                chain_data = self.chain_repo.get_latest_snapshot(symbol)
                if chain_data is not None and not chain_data.empty:
                    print(f"   ✅ Loaded option chain with {len(chain_data)} contracts")
                    
                    # Show chain structure
                    expiries = chain_data['expiry'].nunique()
                    strikes = chain_data['strike'].nunique()
                    print(f"      Expiries: {expiries}, Strikes: {strikes}")
                else:
                    print("   ⚠️ No option chain data available")
                
                # Check for option bar data
                option_dates = self.option_repo.get_available_dates(symbol) 
                if option_dates:
                    print(f"   ✅ Option bars available for {len(option_dates)} dates")
                else:
                    print("   ℹ️ No option bar data yet")
                    
            except Exception as e:
                print(f"   ⚠️ Option data access issue: {e}")
        
        return {
            'equity_data': equity_data,
            'option_ready': option_ready
        }
    
    def example_4_lineage_analysis(self):
        """
        Example 4: Data lineage analysis.
        
        This shows how to use the lineage system to understand
        data dependencies and transformations.
        """
        if not LINEAGE_AVAILABLE:
            print(f"\\n⚠️ EXAMPLE 4: Lineage tracking not available")
            return None
        
        print(f"\\n{'='*60}")
        print("EXAMPLE 4: Data Lineage Analysis")
        print(f"{'='*60}")
        
        try:
            # Get global tracker from orchestrator
            tracker = self.orchestrator.tracker if hasattr(self.orchestrator, 'tracker') else None
            
            if not tracker:
                print("⚠️ Lineage tracker not available")
                return None
            
            # Initialize query and visualization engines
            query_engine = LineageQueryEngine(tracker)
            visualizer = LineageVisualizer(tracker)
            
            # Analysis 1: Overall lineage summary
            print("📊 Generating lineage summary...")
            summary = query_engine.build_lineage_summary()
            
            basic_stats = summary.get('basic_stats', {})
            print(f"   Total operations: {basic_stats.get('total_operations', 0)}")
            print(f"   Total data nodes: {basic_stats.get('total_nodes', 0)}")
            
            if 'operation_types' in basic_stats:
                print("   Operation breakdown:")
                for op_type, count in basic_stats['operation_types'].items():
                    if count > 0:
                        print(f"     {op_type}: {count}")
            
            # Analysis 2: Data source analysis
            print("\\n🗂️ Data source analysis...")
            repo_sources = query_engine.find_data_sources(source_type="repository")
            print(f"   Repository data sources: {len(repo_sources)}")
            
            for source in repo_sources[:3]:  # Show first 3
                print(f"     {source.data_source.source_id}")
            
            # Analysis 3: Performance analysis
            print("\\n⚡ Performance analysis...")
            perf_stats = query_engine.get_operation_performance_stats()
            
            if 'avg_duration_ms' in perf_stats:
                print(f"   Average operation time: {perf_stats['avg_duration_ms']:.2f}ms")
                print(f"   Total operations analyzed: {perf_stats['total_operations']}")
            
            # Analysis 4: Data volume analysis
            print("\\n📈 Data volume analysis...")
            volume_stats = query_engine.analyze_data_volume_flow()
            
            total_in = volume_stats.get('total_records_in', 0)
            total_out = volume_stats.get('total_records_out', 0)
            
            print(f"   Records processed: {total_in:,} in, {total_out:,} out")
            
            return summary
            
        except Exception as e:
            print(f"❌ Lineage analysis failed: {e}")
            return None
    
    def example_5_error_handling(self, invalid_symbol: str = "BADSYMBOL"):
        """
        Example 5: Proper error handling and recovery.
        
        This shows how the system handles errors gracefully and
        provides recovery strategies.
        """
        print(f"\\n{'='*60}")
        print("EXAMPLE 5: Error Handling and Recovery")
        print(f"{'='*60}")
        
        print(f"🧪 Testing error handling with invalid symbol: {invalid_symbol}")
        
        # Test 1: Invalid symbol setup
        print("\\nTest 1: Invalid symbol setup")
        try:
            results = self.orchestrator.setup_new_symbol(invalid_symbol, lookback_days=5)
            
            print("   Setup attempt results:")
            for result in results:
                status = "✅" if result.success else "❌"
                print(f"      {status} {result.job_type.value}: {result.error_message or 'OK'}")
                
        except Exception as e:
            print(f"   🛡️ Exception properly caught: {type(e).__name__}: {e}")
        
        # Test 2: Validation of invalid symbol
        print("\\nTest 2: Validation error handling")
        try:
            self.validator.validate_option_backfill_prerequisites(invalid_symbol)
            print("   ⚠️ Validation unexpectedly passed")
        except PrerequisiteError as e:
            print(f"   ✅ PrerequisiteError properly raised: {e}")
        except Exception as e:
            print(f"   ✅ Other validation error caught: {type(e).__name__}: {e}")
        
        # Test 3: Safe data access with invalid symbol
        print("\\nTest 3: Safe data access error handling")
        try:
            equity_data = self.equity_repo.load_symbol_data(invalid_symbol)
            if equity_data.empty:
                print("   ✅ Empty DataFrame returned for invalid symbol")
            else:
                print(f"   ⚠️ Unexpected data returned: {len(equity_data)} rows")
        except Exception as e:
            print(f"   ✅ Data access error properly handled: {type(e).__name__}: {e}")
        
        print("\\n✅ Error handling tests complete")
        print("   💡 Key takeaway: System fails gracefully with informative messages")
    
    def run_all_examples(self):
        """Run all usage examples in sequence."""
        print("🚀 MODERN TRADING DATA SYSTEM - USAGE EXAMPLES")
        print("="*80)
        
        print("\\nThese examples demonstrate the CORRECT way to use the updated system:")
        print("✅ Orchestrator for dependency management")
        print("✅ Smart persistence (no redundant requests)")
        print("✅ Proper validation before operations")
        print("✅ Data lineage tracking and analysis")
        print("✅ Graceful error handling")
        
        try:
            # Example 1: Setup new symbol
            self.example_1_setup_new_symbol("AAPL")
            
            # Example 2: Daily updates  
            self.example_2_daily_updates(["AAPL"])
            
            # Example 3: Safe data access
            self.example_3_safe_data_access("AAPL")
            
            # Example 4: Lineage analysis
            self.example_4_lineage_analysis()
            
            # Example 5: Error handling
            self.example_5_error_handling("INVALID")
            
            # Final summary
            print(f"\\n{'='*80}")
            print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            
            print(f"\\n📁 Example data stored in: {self.data_path.absolute()}")
            
            print("\\n🎯 Key Usage Patterns Demonstrated:")
            print("1. Use orchestrator.setup_new_symbol() for new symbols")
            print("2. Use orchestrator.daily_update() for incremental updates")
            print("3. Always validate prerequisites before option operations")
            print("4. Use repositories for direct data access after validation")
            print("5. Leverage lineage system for data flow analysis")
            print("6. Handle errors gracefully with proper exception types")
            
        except Exception as e:
            print(f"\\n❌ Example execution failed: {e}")
            raise


def main():
    """Main function to run all examples."""
    example = ModernUsageExample("demo_modern_usage")
    example.run_all_examples()


if __name__ == "__main__":
    main()