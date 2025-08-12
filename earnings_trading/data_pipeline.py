#!/usr/bin/env python3
"""
Earnings-Driven Options Data Pipeline

Collects options data for earnings candidates identified by the discovery system.
Replaces fixed portfolio management with dynamic, opportunity-driven data collection.

Workflow:
1. Discovery system identifies earnings opportunities
2. This pipeline collects required options data for analysis
3. Strategy analyzer determines optimal trades
4. Execution system implements trades
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass

# Import existing components
from jobs.orchestrator import DataPipelineOrchestrator, DataValidation, JobResult, JobType
from earnings_trading.discovery import EarningsDiscoveryEngine, EarningsCandidate, OptionsStrategy

logger = logging.getLogger(__name__)


@dataclass
class DataCollectionTask:
    """Task for collecting options data for an earnings candidate."""
    symbol: str
    priority: str  # critical, high, medium, low
    earnings_date: date
    days_until_earnings: int
    strategies_needed: List[OptionsStrategy]
    
    # Data requirements
    needs_option_chain: bool = True
    needs_historical_iv: bool = True
    needs_equity_data: bool = True
    needs_volume_analysis: bool = True
    
    # Collection status
    option_chain_collected: bool = False
    equity_data_collected: bool = False
    iv_data_collected: bool = False
    
    @property
    def completion_percentage(self) -> float:
        """Percentage of required data collected."""
        total_required = sum([
            self.needs_option_chain,
            self.needs_historical_iv, 
            self.needs_equity_data,
            self.needs_volume_analysis
        ])
        
        total_collected = sum([
            self.option_chain_collected and self.needs_option_chain,
            self.iv_data_collected and self.needs_historical_iv,
            self.equity_data_collected and self.needs_equity_data,
            # Volume analysis status would be tracked separately
        ])
        
        return (total_collected / max(total_required, 1)) * 100
    
    @property
    def urgency_score(self) -> float:
        """Urgency score for prioritizing data collection."""
        # Base score from time urgency
        if self.days_until_earnings <= 1:
            time_score = 100
        elif self.days_until_earnings <= 3:
            time_score = 90
        elif self.days_until_earnings <= 7:
            time_score = 70
        elif self.days_until_earnings <= 14:
            time_score = 50
        else:
            time_score = 30
        
        # Adjust based on priority
        priority_multipliers = {
            'critical': 1.0,
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }
        
        return time_score * priority_multipliers.get(self.priority, 0.5)


class EarningsDataPipeline:
    """
    Dynamic data collection pipeline for earnings-driven options trading.
    
    Replaces fixed portfolio management with opportunity-driven data collection.
    """
    
    def __init__(self, 
                 base_path: Path = Path("data"),
                 discovery_engine: EarningsDiscoveryEngine = None):
        
        self.base_path = base_path
        self.discovery_engine = discovery_engine or EarningsDiscoveryEngine(base_path)
        
        # Use existing orchestrator for actual data collection
        self.orchestrator = DataPipelineOrchestrator(base_path, enable_lineage=True)
        self.validator = DataValidation(base_path)
        
        # Track active collection tasks
        self.active_tasks: Dict[str, DataCollectionTask] = {}
        self.completed_tasks: List[DataCollectionTask] = []
        
        logger.info("📊 Initialized earnings data pipeline")
    
    def run_earnings_driven_collection(self, 
                                     days_ahead: int = 21,
                                     min_opportunity_score: float = 50.0,
                                     max_symbols: int = 50) -> Dict[str, Any]:
        """
        Main pipeline: Discover earnings opportunities and collect required data.
        
        This replaces the traditional portfolio-based approach.
        """
        
        logger.info("🔍 Starting earnings-driven data collection")
        
        # Step 1: Discover current market opportunities
        opportunities = self.discovery_engine.discover_earnings_opportunities(
            days_ahead=days_ahead,
            min_score=min_opportunity_score
        )
        
        if not opportunities:
            logger.warning("❌ No earnings opportunities found")
            return {
                'status': 'no_opportunities',
                'opportunities_found': 0,
                'tasks_created': 0
            }
        
        logger.info(f"🎯 Found {len(opportunities)} earnings opportunities")
        
        # Step 2: Limit to manageable number for data collection
        if len(opportunities) > max_symbols:
            logger.info(f"📊 Limiting to top {max_symbols} opportunities")
            opportunities = opportunities[:max_symbols]
        
        # Step 3: Create data collection tasks
        tasks = self._create_collection_tasks(opportunities)
        
        # Step 4: Execute data collection in priority order
        results = self._execute_data_collection(tasks)
        
        return {
            'status': 'completed',
            'opportunities_found': len(opportunities),
            'tasks_created': len(tasks),
            'tasks_completed': len([t for t in tasks if t.completion_percentage == 100]),
            'symbols_processed': len(set(t.symbol for t in tasks)),
            'execution_results': results
        }
    
    def _create_collection_tasks(self, opportunities: List[EarningsCandidate]) -> List[DataCollectionTask]:
        """Create data collection tasks from opportunities."""
        
        # Get priority groupings from discovery engine
        priority_groups = self.discovery_engine.get_data_collection_list(opportunities)
        
        tasks = []
        
        # Create tasks for each priority level
        for priority, symbols in priority_groups.items():
            for symbol in symbols:
                # Find the opportunity details
                opportunity = next((o for o in opportunities if o.symbol == symbol), None)
                if opportunity:
                    
                    # Determine required strategies
                    strategies_needed = []
                    if opportunity.calendar_score > 50:
                        strategies_needed.append(OptionsStrategy.CALENDAR_SPREAD)
                    if opportunity.straddle_score > 50:
                        strategies_needed.append(OptionsStrategy.STRADDLE)
                    if opportunity.strangle_score > 50:
                        strategies_needed.append(OptionsStrategy.STRANGLE)
                    
                    task = DataCollectionTask(
                        symbol=symbol,
                        priority=priority,
                        earnings_date=opportunity.earnings_date,
                        days_until_earnings=opportunity.days_until_earnings,
                        strategies_needed=strategies_needed
                    )
                    
                    tasks.append(task)
                    self.active_tasks[symbol] = task
        
        # Sort by urgency
        tasks.sort(key=lambda t: -t.urgency_score)
        
        logger.info(f"📋 Created {len(tasks)} data collection tasks")
        for priority, symbols in priority_groups.items():
            if symbols:
                logger.info(f"   {priority.upper()}: {len(symbols)} symbols")
        
        return tasks
    
    def _execute_data_collection(self, tasks: List[DataCollectionTask]) -> Dict[str, Any]:
        """Execute data collection for earnings candidates."""
        
        execution_results = {
            'start_time': datetime.now(),
            'symbols_processed': [],
            'successful_collections': 0,
            'failed_collections': 0,
            'job_results': []
        }
        
        logger.info(f"⚡ Executing data collection for {len(tasks)} candidates")
        
        for task in tasks:
            logger.info(f"🔄 Processing {task.symbol} (priority: {task.priority}, "
                       f"earnings in {task.days_until_earnings} days)")
            
            try:
                # Use existing orchestrator for data collection
                symbol_results = self.orchestrator.setup_new_symbol(
                    task.symbol, 
                    lookback_days=30  # Standard lookback for options analysis
                )
                
                # Update task status based on results
                self._update_task_status(task, symbol_results)
                
                execution_results['symbols_processed'].append(task.symbol)
                execution_results['job_results'].extend(symbol_results)
                
                if task.completion_percentage >= 80:  # Consider 80%+ as successful
                    execution_results['successful_collections'] += 1
                    logger.info(f"✅ {task.symbol} data collection: {task.completion_percentage:.0f}% complete")
                else:
                    execution_results['failed_collections'] += 1
                    logger.warning(f"⚠️ {task.symbol} data collection: {task.completion_percentage:.0f}% complete")
                
            except Exception as e:
                logger.error(f"❌ {task.symbol} collection failed: {e}")
                execution_results['failed_collections'] += 1
        
        execution_results['end_time'] = datetime.now()
        execution_results['total_duration'] = (
            execution_results['end_time'] - execution_results['start_time']
        ).total_seconds()
        
        self._log_execution_summary(execution_results)
        
        return execution_results
    
    def _update_task_status(self, task: DataCollectionTask, job_results: List[JobResult]):
        """Update task completion status based on job results."""
        
        for result in job_results:
            if result.success:
                if result.job_type == JobType.EQUITY_BACKFILL:
                    task.equity_data_collected = True
                elif result.job_type == JobType.OPTION_CHAIN_SNAPSHOT:
                    task.option_chain_collected = True
                # IV data collection would be a separate job type
    
    def _log_execution_summary(self, results: Dict[str, Any]):
        """Log execution summary."""
        
        duration_minutes = results['total_duration'] / 60
        success_rate = (results['successful_collections'] / 
                       max(len(results['symbols_processed']), 1) * 100)
        
        logger.info("📈 EARNINGS DATA COLLECTION SUMMARY")
        logger.info(f"   Duration: {duration_minutes:.1f} minutes")
        logger.info(f"   Symbols processed: {len(results['symbols_processed'])}")
        logger.info(f"   ✅ Successful: {results['successful_collections']}")
        logger.info(f"   ❌ Failed: {results['failed_collections']}")
        logger.info(f"   📊 Success rate: {success_rate:.1f}%")
    
    def get_ready_for_analysis(self) -> List[DataCollectionTask]:
        """Get tasks that are ready for options strategy analysis."""
        
        ready_tasks = []
        for task in self.active_tasks.values():
            if task.completion_percentage >= 80:  # Has sufficient data
                ready_tasks.append(task)
        
        # Sort by earnings proximity (most urgent first)
        ready_tasks.sort(key=lambda t: t.days_until_earnings)
        
        return ready_tasks
    
    def export_collection_status(self, output_file: Path = None) -> Path:
        """Export current data collection status."""
        
        if output_file is None:
            output_file = self.base_path / "exports" / f"collection_status_{date.today().isoformat()}.csv"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        import csv
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow([
                'Symbol', 'Priority', 'Earnings_Date', 'Days_Until', 
                'Completion_%', 'Urgency_Score', 'Strategies_Needed',
                'Option_Chain', 'Equity_Data', 'IV_Data', 'Status'
            ])
            
            # Active tasks
            for task in self.active_tasks.values():
                strategies = ', '.join(s.value for s in task.strategies_needed)
                status = 'READY' if task.completion_percentage >= 80 else 'IN_PROGRESS'
                
                writer.writerow([
                    task.symbol,
                    task.priority.upper(),
                    task.earnings_date.isoformat(),
                    task.days_until_earnings,
                    f"{task.completion_percentage:.0f}%",
                    f"{task.urgency_score:.0f}",
                    strategies,
                    '✅' if task.option_chain_collected else '⏳',
                    '✅' if task.equity_data_collected else '⏳',
                    '✅' if task.iv_data_collected else '⏳',
                    status
                ])
        
        logger.info(f"💾 Collection status exported to {output_file}")
        return output_file


def main():
    """Test the earnings-driven data pipeline."""
    logging.basicConfig(level=logging.INFO)
    
    print("🎯 EARNINGS-DRIVEN OPTIONS DATA PIPELINE")
    print("=" * 60)
    
    pipeline = EarningsDataPipeline()
    
    # Run earnings-driven collection
    results = pipeline.run_earnings_driven_collection(
        days_ahead=21,
        min_opportunity_score=40.0,  # Lower for testing
        max_symbols=10  # Limit for testing
    )
    
    print(f"\n📊 Pipeline Results:")
    print(f"   Opportunities found: {results['opportunities_found']}")
    print(f"   Data collection tasks: {results['tasks_created']}")
    print(f"   Tasks completed: {results['tasks_completed']}")
    print(f"   Symbols processed: {results['symbols_processed']}")
    
    # Export status
    if pipeline.active_tasks:
        status_file = pipeline.export_collection_status()
        print(f"\n💾 Status exported to: {status_file}")
        
        # Show ready for analysis
        ready_tasks = pipeline.get_ready_for_analysis()
        if ready_tasks:
            print(f"\n🎯 Ready for strategy analysis: {len(ready_tasks)} symbols")
            for task in ready_tasks[:5]:
                print(f"   {task.symbol}: {task.completion_percentage:.0f}% complete, "
                      f"earnings in {task.days_until_earnings} days")


if __name__ == "__main__":
    main()