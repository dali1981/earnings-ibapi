"""
Complete Trading Data Workflow Example

This example demonstrates the complete workflow for setting up and managing
trading data with proper dependency management, persistence strategy, and
data lineage tracking.

Key Concepts Demonstrated:
1. Data dependency management (contracts → chains → equity → options)
2. TWS API constraints (snapshot-only chains, throttling)  
3. Smart persistence (never re-request existing data)
4. Live vs historical data distinction
5. Comprehensive error handling and validation
6. Data lineage tracking and analysis
"""

import logging
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
import time

# Core repository imports
from repositories import (
    EquityBarRepository,
    OptionBarRepository,
    OptionChainSnapshotRepository
)

# Job orchestration
from jobs.orchestrator import DataPipelineOrchestrator, DataValidation, PrerequisiteError

# Lineage tracking
try:
    from lineage import LineageTracker, LineageMetadataRepository, set_global_tracker
    from lineage.query import LineageQueryEngine
    from lineage.visualizer import LineageVisualizer
    LINEAGE_AVAILABLE = True
except ImportError:
    print("⚠️ Lineage tracking not available")
    LINEAGE_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingDataWorkflow:
    """
    Complete trading data workflow manager.
    
    Demonstrates proper usage patterns and best practices for the trading data system.
    """
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Initialize orchestrator with lineage tracking
        self.orchestrator = DataPipelineOrchestrator(
            base_path=self.base_path,
            enable_lineage=LINEAGE_AVAILABLE
        )
        
        # Initialize validator
        self.validator = DataValidation(self.base_path)
        
        # Initialize repositories
        self.equity_repo = EquityBarRepository(self.base_path / "equity_bars")
        self.option_repo = OptionBarRepository(self.base_path / "option_bars")
        self.chain_repo = OptionChainSnapshotRepository(self.base_path / "option_chains")
        
        # Setup lineage analysis if available
        if LINEAGE_AVAILABLE:
            self.setup_lineage_analysis()
        
        logger.info(f"Initialized trading data workflow with base path: {self.base_path}")
    
    def setup_lineage_analysis(self):
        """Setup lineage tracking and analysis capabilities."""
        try:
            lineage_repo = LineageMetadataRepository(self.base_path / "lineage_metadata")
            self.tracker = LineageTracker(storage_backend=lineage_repo)
            set_global_tracker(self.tracker)
            
            self.query_engine = LineageQueryEngine(self.tracker)
            self.visualizer = LineageVisualizer(self.tracker)
            
            logger.info("✅ Lineage tracking enabled")
        except Exception as e:
            logger.warning(f"Failed to setup lineage tracking: {e}")
    
    def demonstrate_cold_start_workflow(self, symbol: str = "AAPL", lookback_days: int = 30):
        """
        Demonstrate cold start workflow for a new symbol.
        
        This shows the proper dependency order and validation process.
        """
        print(f"\\n{'='*60}")
        print(f"COLD START WORKFLOW DEMONSTRATION - {symbol}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Step 1: Setup new symbol (orchestrator handles dependencies)
            print(f"\\n🚀 Setting up new symbol: {symbol}")
            results = self.orchestrator.setup_new_symbol(symbol, lookback_days=lookback_days)
            
            # Step 2: Analyze results
            print(f"\\n📊 Setup Results:")
            success_count = 0
            for result in results:
                status = "✅ SUCCESS" if result.success else "❌ FAILED"
                print(f"  {result.job_type.value}: {status}")
                
                if result.success:
                    success_count += 1
                    if result.execution_time_seconds:
                        print(f"    Execution time: {result.execution_time_seconds:.2f}s")
                    if result.data_records:
                        print(f"    Records processed: {result.data_records:,}")
                else:
                    print(f"    Error: {result.error_message}")
            
            print(f"\\n📈 Overall Success Rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
            
            # Step 3: Validate final state
            print(f"\\n🔍 Validating final data state...")
            self.validate_symbol_data_state(symbol)
            
            execution_time = time.time() - start_time
            print(f"\\n⏱️ Total workflow time: {execution_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            print(f"❌ Cold start workflow failed: {e}")
            logger.error(f"Cold start failed for {symbol}: {e}")
            raise
    
    def demonstrate_incremental_updates(self, symbols: list = None):
        """
        Demonstrate daily incremental updates.
        
        Shows how to update existing data efficiently without re-requesting.
        """
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT"]
        
        print(f"\\n{'='*60}")
        print("INCREMENTAL UPDATE DEMONSTRATION")
        print(f"{'='*60}")
        
        print(f"\\n📅 Running daily updates for {len(symbols)} symbols...")
        
        start_time = time.time()
        results = self.orchestrator.daily_update(symbols)
        execution_time = time.time() - start_time
        
        print(f"\\n📊 Update Results:")
        total_jobs = 0
        successful_jobs = 0
        
        for symbol, symbol_results in results.items():
            if symbol_results:
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
        
        if total_jobs > 0:
            print(f"\\n📈 Overall Update Success: {successful_jobs}/{total_jobs} ({successful_jobs/total_jobs*100:.1f}%)")
        else:
            print("\\n✅ All symbols up to date - no updates needed")
        
        print(f"⏱️ Update time: {execution_time:.2f} seconds")
        
        return results
    
    def demonstrate_data_analysis(self, symbol: str = "AAPL"):
        """
        Demonstrate data analysis capabilities.
        
        Shows how to safely load and analyze data with proper validation.
        """
        print(f"\\n{'='*60}")
        print(f"DATA ANALYSIS DEMONSTRATION - {symbol}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Validate data availability
            print(f"\\n🔍 Validating data availability for {symbol}...")
            self.validator.validate_data_integrity(symbol)
            print("✅ Data integrity validated")
            
            # Step 2: Load equity data
            print(f"\\n📈 Loading equity data...")
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            
            equity_data = self.equity_repo.load_symbol_data(symbol, start_date, end_date)
            print(f"✅ Loaded {len(equity_data)} equity bars")
            
            if not equity_data.empty:
                print(f"   Date range: {equity_data['trade_date'].min()} to {equity_data['trade_date'].max()}")
                print(f"   Price range: ${equity_data['low'].min():.2f} - ${equity_data['high'].max():.2f}")
                print(f"   Avg volume: {equity_data['volume'].mean():,.0f}")
            
            # Step 3: Load option chain data (if available)
            print(f"\\n⚙️ Loading option chain data...")
            try:
                latest_chain = self.chain_repo.get_latest_snapshot(symbol)
                if latest_chain is not None and not latest_chain.empty:
                    print(f"✅ Loaded option chain with {len(latest_chain)} contracts")
                    
                    # Analyze chain structure
                    unique_expiries = latest_chain['expiry'].nunique()
                    unique_strikes = latest_chain['strike'].nunique()
                    call_count = len(latest_chain[latest_chain['right'] == 'C'])
                    put_count = len(latest_chain[latest_chain['right'] == 'P'])
                    
                    print(f"   Expiries: {unique_expiries}")
                    print(f"   Strikes: {unique_strikes}")
                    print(f"   Calls: {call_count}, Puts: {put_count}")
                else:
                    print("⚠️ No option chain data available")
                    
            except Exception as e:
                print(f"⚠️ Could not load option chain: {e}")
            
            # Step 4: Option bar analysis (if prerequisites met)
            print(f"\\n📊 Analyzing option data availability...")
            try:
                self.validator.validate_option_backfill_prerequisites(symbol)
                
                # Check what option data we have
                option_dates = self.option_repo.get_available_dates(symbol)
                if option_dates:
                    print(f"✅ Option bar data available for {len(option_dates)} dates")
                    print(f"   Date range: {min(option_dates)} to {max(option_dates)}")
                else:
                    print("ℹ️ No option bar data available")
                    
            except PrerequisiteError as e:
                print(f"⚠️ Option data prerequisites not met: {e}")
            
            return {
                'equity_records': len(equity_data),
                'option_chain_available': latest_chain is not None and not latest_chain.empty,
                'option_dates_available': len(option_dates) if 'option_dates' in locals() else 0
            }
            
        except Exception as e:
            print(f"❌ Data analysis failed: {e}")
            logger.error(f"Data analysis failed for {symbol}: {e}")
            raise
    
    def demonstrate_lineage_analysis(self):
        """
        Demonstrate data lineage analysis capabilities.
        
        Shows how to understand data dependencies and impact analysis.
        """
        if not LINEAGE_AVAILABLE:
            print("\\n⚠️ Lineage tracking not available - skipping demonstration")
            return
        
        print(f"\\n{'='*60}")
        print("DATA LINEAGE ANALYSIS DEMONSTRATION")
        print(f"{'='*60}")
        
        try:
            # Step 1: Generate overall lineage summary
            print("\\n🔍 Generating lineage summary...")
            summary = self.query_engine.build_lineage_summary()
            
            print("📊 Lineage Statistics:")
            stats = summary['basic_stats']
            print(f"   Total operations: {stats['total_operations']}")
            print(f"   Total data nodes: {stats['total_nodes']}")
            
            if 'operation_types' in stats:
                print("   Operation types:")
                for op_type, count in stats['operation_types'].items():
                    if count > 0:
                        print(f"     {op_type}: {count}")
            
            # Step 2: Find data sources
            print("\\n🗂️ Data Source Analysis:")
            sources = self.query_engine.find_data_sources(source_type="repository")
            print(f"   Repository sources: {len(sources)}")
            
            for source in sources[:5]:  # Show first 5
                print(f"     {source.data_source.source_id}")
            
            # Step 3: Data volume analysis
            print("\\n📈 Data Volume Analysis:")
            volume_stats = self.query_engine.analyze_data_volume_flow()
            print(f"   Total operations analyzed: {volume_stats['total_operations']}")
            print(f"   Total records in: {volume_stats.get('total_records_in', 0):,}")
            print(f"   Total records out: {volume_stats.get('total_records_out', 0):,}")
            
            # Step 4: Performance analysis
            print("\\n⚡ Performance Analysis:")
            perf_stats = self.query_engine.get_operation_performance_stats()
            if 'avg_duration_ms' in perf_stats:
                print(f"   Average operation time: {perf_stats['avg_duration_ms']:.2f}ms")
                print(f"   Fastest operation: {perf_stats.get('min_duration_ms', 0):.2f}ms")
                print(f"   Slowest operation: {perf_stats.get('max_duration_ms', 0):.2f}ms")
            
            # Step 5: Generate comprehensive report
            print("\\n📄 Generating comprehensive lineage report...")
            report = self.visualizer.generate_summary_report()
            
            # Show first few lines of report
            report_lines = report.split('\\n')[:10]
            for line in report_lines:
                print(f"   {line}")
            print("   ... (truncated)")
            
            return summary
            
        except Exception as e:
            print(f"❌ Lineage analysis failed: {e}")
            logger.error(f"Lineage analysis failed: {e}")
            raise
    
    def validate_symbol_data_state(self, symbol: str):
        """Comprehensive validation of symbol data state."""
        print(f"\\n🔍 Comprehensive validation for {symbol}:")
        
        validation_results = {
            'contracts': False,
            'equity_data': False,
            'option_chain': False,
            'option_prerequisites': False,
            'data_integrity': False
        }
        
        # Check contracts
        try:
            self.validator.validate_symbol_contracts_exist(symbol)
            validation_results['contracts'] = True
            print("   ✅ Contract descriptions: Available")
        except PrerequisiteError:
            print("   ❌ Contract descriptions: Missing")
        
        # Check equity data
        try:
            self.validator.validate_equity_data_available(symbol)
            validation_results['equity_data'] = True
            print("   ✅ Equity data: Current")
        except PrerequisiteError:
            print("   ❌ Equity data: Missing or stale")
        
        # Check option chain
        try:
            self.validator.validate_option_chain_current(symbol, max_age_hours=24)
            validation_results['option_chain'] = True
            print("   ✅ Option chain: Current")
        except PrerequisiteError:
            print("   ❌ Option chain: Missing or stale")
        
        # Check option prerequisites
        try:
            self.validator.validate_option_backfill_prerequisites(symbol)
            validation_results['option_prerequisites'] = True
            print("   ✅ Option prerequisites: Met")
        except PrerequisiteError:
            print("   ❌ Option prerequisites: Not met")
        
        # Check data integrity
        try:
            self.validator.validate_data_integrity(symbol)
            validation_results['data_integrity'] = True
            print("   ✅ Data integrity: Valid")
        except Exception:
            print("   ❌ Data integrity: Issues detected")
        
        # Summary
        passed_checks = sum(validation_results.values())
        total_checks = len(validation_results)
        print(f"\\n📊 Validation Summary: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks == total_checks:
            print("🎉 All validations passed - symbol ready for all operations")
        elif validation_results['option_prerequisites']:
            print("✅ Symbol ready for option trading")
        elif validation_results['equity_data']:
            print("⚠️ Symbol ready for equity trading only")
        else:
            print("❌ Symbol needs setup - run cold start workflow")
        
        return validation_results
    
    def demonstrate_error_handling(self, symbol: str = "INVALID_SYMBOL"):
        """
        Demonstrate proper error handling and recovery strategies.
        """
        print(f"\\n{'='*60}")
        print(f"ERROR HANDLING DEMONSTRATION - {symbol}")
        print(f"{'='*60}")
        
        print("\\n⚠️ Demonstrating error handling with invalid symbol...")
        
        try:
            # This should fail due to invalid symbol
            results = self.orchestrator.setup_new_symbol(symbol, lookback_days=5)
            
            print("\\n📊 Error Handling Results:")
            for result in results:
                if result.success:
                    print(f"   ✅ {result.job_type.value}: Unexpected success")
                else:
                    print(f"   ❌ {result.job_type.value}: {result.error_message}")
            
        except Exception as e:
            print(f"   🛡️ Exception properly caught: {e}")
        
        # Demonstrate validation failures
        print("\\n🔍 Demonstrating validation failures...")
        try:
            self.validator.validate_option_backfill_prerequisites(symbol)
            print("   ⚠️ Validation unexpectedly passed")
        except PrerequisiteError as e:
            print(f"   ✅ Validation properly failed: {e}")
        
        print("\\n✅ Error handling demonstration complete")


def main():
    """
    Main demonstration function.
    
    Runs through all major workflow scenarios to demonstrate the system capabilities.
    """
    print("🚀 TRADING DATA SYSTEM - COMPLETE WORKFLOW DEMONSTRATION")
    print("="*80)
    
    # Initialize workflow manager
    workflow = TradingDataWorkflow("demo_data")
    
    try:
        # 1. Cold Start Workflow
        print("\\n1️⃣ COLD START WORKFLOW")
        print("-" * 40)
        cold_start_results = workflow.demonstrate_cold_start_workflow("AAPL", lookback_days=10)
        
        # 2. Incremental Updates  
        print("\\n2️⃣ INCREMENTAL UPDATES")
        print("-" * 40)
        update_results = workflow.demonstrate_incremental_updates(["AAPL"])
        
        # 3. Data Analysis
        print("\\n3️⃣ DATA ANALYSIS")
        print("-" * 40)
        analysis_results = workflow.demonstrate_data_analysis("AAPL")
        
        # 4. Lineage Analysis
        print("\\n4️⃣ LINEAGE ANALYSIS")
        print("-" * 40)
        lineage_results = workflow.demonstrate_lineage_analysis()
        
        # 5. Error Handling
        print("\\n5️⃣ ERROR HANDLING")
        print("-" * 40)
        workflow.demonstrate_error_handling("BADTICKER")
        
        # Final Summary
        print("\\n" + "="*80)
        print("🎉 DEMONSTRATION COMPLETE")
        print("="*80)
        
        print("\\n📊 Summary of Demonstrated Capabilities:")
        print("✅ Data dependency management (contracts → chains → equity → options)")
        print("✅ TWS API constraint handling (snapshot-only chains)")
        print("✅ Smart persistence (no redundant requests)")
        print("✅ Incremental updates (only missing data)")
        print("✅ Comprehensive validation and error handling")
        print("✅ Data lineage tracking and analysis")
        print("✅ Performance monitoring and optimization")
        
        print("\\n🎯 Key Takeaways:")
        print("1. Always validate prerequisites before requesting option data")
        print("2. Use orchestrator to handle dependencies automatically")
        print("3. Check existing data before making API requests")
        print("4. Monitor data lineage for quality and consistency")
        print("5. Implement proper error handling and recovery")
        
        print(f"\\n📁 Demo data stored in: {workflow.base_path.absolute()}")
        
    except Exception as e:
        print(f"\\n❌ Demonstration failed: {e}")
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()