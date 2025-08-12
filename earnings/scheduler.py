#!/usr/bin/env python3
"""
Earnings-Priority Scheduler

Priority-based task scheduler for daily data updates that prioritizes
symbols based on upcoming earnings events.

Key Features:
- Earnings-driven priority scheduling
- Resource allocation based on importance
- Time-window optimization
- Circuit breaker integration
- Recovery and retry logic
"""

import logging
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .fetcher import EarningsCalendarFetcher, EarningsEvent
from jobs.orchestrator import DataPipelineOrchestrator, JobType, JobResult

logger = logging.getLogger(__name__)


class SchedulePriority(Enum):
    """Schedule priority levels."""
    CRITICAL = "critical"      # Earnings today/tomorrow
    HIGH = "high"             # Earnings this week
    MEDIUM = "medium"         # Earnings next week
    LOW = "low"              # No upcoming earnings
    MAINTENANCE = "maintenance"  # Background updates


@dataclass
class ScheduledSymbol:
    """Symbol with scheduling metadata."""
    symbol: str
    priority: SchedulePriority
    earnings_event: Optional[EarningsEvent] = None
    last_updated: Optional[datetime] = None
    update_frequency_hours: int = 24
    retry_count: int = 0
    max_retries: int = 3
    circuit_breaker_until: Optional[datetime] = None
    
    @property
    def needs_update(self) -> bool:
        """Check if symbol needs update based on frequency."""
        if not self.last_updated:
            return True
        
        hours_since_update = (datetime.now() - self.last_updated).total_seconds() / 3600
        return hours_since_update >= self.update_frequency_hours
    
    @property
    def is_circuit_broken(self) -> bool:
        """Check if circuit breaker is active."""
        if not self.circuit_breaker_until:
            return False
        return datetime.now() < self.circuit_breaker_until
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker after successful operation."""
        self.circuit_breaker_until = None
        self.retry_count = 0
    
    def trigger_circuit_breaker(self, duration_minutes: int = 60):
        """Trigger circuit breaker after failures."""
        self.circuit_breaker_until = datetime.now() + timedelta(minutes=duration_minutes)
        logger.warning(f"🔴 Circuit breaker activated for {self.symbol} until {self.circuit_breaker_until}")


@dataclass
class ScheduleBatch:
    """Batch of symbols to process together."""
    priority: SchedulePriority
    symbols: List[ScheduledSymbol] = field(default_factory=list)
    max_concurrent: int = 5
    estimated_duration_minutes: int = 10
    
    @property
    def total_symbols(self) -> int:
        return len(self.symbols)


class EarningsPriorityScheduler:
    """Earnings-driven priority scheduler for daily updates."""
    
    def __init__(self, 
                 base_path: Path,
                 orchestrator: DataPipelineOrchestrator = None,
                 earnings_fetcher: EarningsCalendarFetcher = None):
        
        self.base_path = base_path
        self.orchestrator = orchestrator or DataPipelineOrchestrator(base_path, enable_lineage=True)
        self.earnings_fetcher = earnings_fetcher or EarningsCalendarFetcher()
        
        # Tracking
        self.scheduled_symbols: Dict[str, ScheduledSymbol] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.priority_configs = {
            SchedulePriority.CRITICAL: {
                'update_frequency_hours': 4,    # 4x per day
                'max_concurrent': 2,            # Conservative for critical
                'circuit_breaker_minutes': 30,  # Quick recovery
                'batch_size': 5
            },
            SchedulePriority.HIGH: {
                'update_frequency_hours': 8,    # 3x per day  
                'max_concurrent': 3,
                'circuit_breaker_minutes': 60,
                'batch_size': 10
            },
            SchedulePriority.MEDIUM: {
                'update_frequency_hours': 12,   # 2x per day
                'max_concurrent': 5,
                'circuit_breaker_minutes': 120,
                'batch_size': 15
            },
            SchedulePriority.LOW: {
                'update_frequency_hours': 24,   # Daily
                'max_concurrent': 8,
                'circuit_breaker_minutes': 240,
                'batch_size': 20
            },
            SchedulePriority.MAINTENANCE: {
                'update_frequency_hours': 72,   # Every 3 days
                'max_concurrent': 10,
                'circuit_breaker_minutes': 360,
                'batch_size': 50
            }
        }
    
    def refresh_earnings_priorities(self, symbols: List[str]) -> Dict[str, SchedulePriority]:
        """Refresh earnings data and calculate priorities."""
        logger.info(f"🔍 Refreshing earnings priorities for {len(symbols)} symbols")
        
        # Fetch upcoming earnings
        earnings_events = self.earnings_fetcher.get_upcoming_earnings(
            symbols=symbols,
            days_ahead=30  # Look ahead a month
        )
        
        # Create earnings lookup
        earnings_by_symbol = {event.symbol: event for event in earnings_events}
        
        # Calculate priorities
        symbol_priorities = {}
        
        for symbol in symbols:
            earnings_event = earnings_by_symbol.get(symbol)
            
            if earnings_event:
                days_until = earnings_event.days_until_earnings
                
                if days_until <= 1:
                    priority = SchedulePriority.CRITICAL
                elif days_until <= 7:
                    priority = SchedulePriority.HIGH
                elif days_until <= 14:
                    priority = SchedulePriority.MEDIUM
                else:
                    priority = SchedulePriority.LOW
            else:
                priority = SchedulePriority.MAINTENANCE  # No earnings info
            
            symbol_priorities[symbol] = priority
            
            # Update or create scheduled symbol
            if symbol in self.scheduled_symbols:
                scheduled = self.scheduled_symbols[symbol]
                scheduled.priority = priority
                scheduled.earnings_event = earnings_event
            else:
                config = self.priority_configs[priority]
                scheduled = ScheduledSymbol(
                    symbol=symbol,
                    priority=priority,
                    earnings_event=earnings_event,
                    update_frequency_hours=config['update_frequency_hours']
                )
                self.scheduled_symbols[symbol] = scheduled
        
        # Log priority distribution
        priority_counts = defaultdict(int)
        for priority in symbol_priorities.values():
            priority_counts[priority] += 1
        
        logger.info("📊 Priority distribution:")
        for priority, count in priority_counts.items():
            emoji = self._get_priority_emoji(priority)
            logger.info(f"  {emoji} {priority.value}: {count} symbols")
        
        return symbol_priorities
    
    def create_daily_schedule(self, symbols: List[str] = None) -> List[ScheduleBatch]:
        """Create optimized daily schedule based on priorities."""
        
        if symbols:
            # Refresh priorities for new symbols
            self.refresh_earnings_priorities(symbols)
        
        # Filter symbols that need updates and aren't circuit-broken
        symbols_to_update = [
            scheduled for scheduled in self.scheduled_symbols.values()
            if scheduled.needs_update and not scheduled.is_circuit_broken
        ]
        
        if not symbols_to_update:
            logger.info("✅ All symbols are up to date")
            return []
        
        logger.info(f"📅 Scheduling {len(symbols_to_update)} symbols for update")
        
        # Group by priority
        priority_groups = defaultdict(list)
        for scheduled in symbols_to_update:
            priority_groups[scheduled.priority].append(scheduled)
        
        # Create batches in priority order
        batches = []
        priority_order = [
            SchedulePriority.CRITICAL,
            SchedulePriority.HIGH,
            SchedulePriority.MEDIUM,
            SchedulePriority.LOW,
            SchedulePriority.MAINTENANCE
        ]
        
        for priority in priority_order:
            if priority not in priority_groups:
                continue
            
            symbols_for_priority = priority_groups[priority]
            config = self.priority_configs[priority]
            
            # Create batches for this priority level
            for i in range(0, len(symbols_for_priority), config['batch_size']):
                batch_symbols = symbols_for_priority[i:i + config['batch_size']]
                
                batch = ScheduleBatch(
                    priority=priority,
                    symbols=batch_symbols,
                    max_concurrent=config['max_concurrent'],
                    estimated_duration_minutes=len(batch_symbols) * 2  # Rough estimate
                )
                batches.append(batch)
        
        logger.info(f"📦 Created {len(batches)} batches for execution")
        return batches
    
    def execute_daily_schedule(self, batches: List[ScheduleBatch] = None) -> Dict[str, Any]:
        """Execute the daily schedule with monitoring."""
        
        if batches is None:
            batches = self.create_daily_schedule()
        
        if not batches:
            return {'status': 'no_updates_needed', 'batches_processed': 0}
        
        logger.info(f"🚀 Starting daily schedule execution: {len(batches)} batches")
        
        execution_start = datetime.now()
        results = {
            'execution_start': execution_start,
            'batches': [],
            'total_symbols_processed': 0,
            'successful_symbols': 0,
            'failed_symbols': 0,
            'circuit_breakers_triggered': 0
        }
        
        for batch_idx, batch in enumerate(batches, 1):
            logger.info(f"🔄 Processing batch {batch_idx}/{len(batches)}: "
                       f"{batch.priority.value} priority ({batch.total_symbols} symbols)")
            
            batch_result = self._execute_batch(batch)
            results['batches'].append(batch_result)
            
            # Update counters
            results['total_symbols_processed'] += batch_result['symbols_processed']
            results['successful_symbols'] += batch_result['successful_symbols']
            results['failed_symbols'] += batch_result['failed_symbols']
            results['circuit_breakers_triggered'] += batch_result['circuit_breakers_triggered']
            
            # Brief pause between batches
            if batch_idx < len(batches):
                time.sleep(5)
        
        results['execution_end'] = datetime.now()
        results['total_duration'] = (results['execution_end'] - execution_start).total_seconds()
        
        # Log summary
        self._log_execution_summary(results)
        
        # Store execution history
        self.execution_history.append(results)
        
        return results
    
    def _execute_batch(self, batch: ScheduleBatch) -> Dict[str, Any]:
        """Execute a single batch of symbols."""
        batch_start = datetime.now()
        
        batch_result = {
            'priority': batch.priority.value,
            'batch_start': batch_start,
            'symbols_processed': 0,
            'successful_symbols': 0,
            'failed_symbols': 0,
            'circuit_breakers_triggered': 0,
            'symbol_results': {}
        }
        
        for scheduled in batch.symbols:
            symbol = scheduled.symbol
            
            try:
                logger.info(f"🔄 Updating {symbol} (priority: {scheduled.priority.value})")
                
                # Execute daily update via orchestrator
                symbol_results = self.orchestrator.daily_update([symbol])
                update_success = self._analyze_symbol_results(symbol_results.get(symbol, []))
                
                if update_success:
                    scheduled.last_updated = datetime.now()
                    scheduled.reset_circuit_breaker()
                    batch_result['successful_symbols'] += 1
                    batch_result['symbol_results'][symbol] = 'success'
                    logger.info(f"✅ {symbol} updated successfully")
                else:
                    self._handle_symbol_failure(scheduled)
                    batch_result['failed_symbols'] += 1
                    batch_result['symbol_results'][symbol] = 'failed'
                    logger.warning(f"❌ {symbol} update failed")
                
                batch_result['symbols_processed'] += 1
                
            except Exception as e:
                logger.error(f"❌ {symbol} update error: {e}")
                self._handle_symbol_failure(scheduled)
                batch_result['failed_symbols'] += 1
                batch_result['symbols_processed'] += 1
                batch_result['symbol_results'][symbol] = f'error: {str(e)}'
        
        batch_result['batch_end'] = datetime.now()
        batch_result['batch_duration'] = (batch_result['batch_end'] - batch_start).total_seconds()
        
        return batch_result
    
    def _analyze_symbol_results(self, job_results: List[JobResult]) -> bool:
        """Analyze if symbol update was successful."""
        if not job_results:
            return False
        
        # Consider successful if at least equity data updated
        equity_jobs = [r for r in job_results if r.job_type == JobType.EQUITY_BACKFILL]
        if equity_jobs and any(job.success for job in equity_jobs):
            return True
        
        # Or if all jobs succeeded (even if no equity jobs)
        return all(job.success for job in job_results)
    
    def _handle_symbol_failure(self, scheduled: ScheduledSymbol):
        """Handle symbol update failure."""
        scheduled.retry_count += 1
        
        if scheduled.retry_count >= scheduled.max_retries:
            config = self.priority_configs[scheduled.priority]
            scheduled.trigger_circuit_breaker(config['circuit_breaker_minutes'])
    
    def _get_priority_emoji(self, priority: SchedulePriority) -> str:
        """Get emoji for priority level."""
        emojis = {
            SchedulePriority.CRITICAL: "🔥",
            SchedulePriority.HIGH: "⚡",
            SchedulePriority.MEDIUM: "📊", 
            SchedulePriority.LOW: "📅",
            SchedulePriority.MAINTENANCE: "🔧"
        }
        return emojis.get(priority, "📌")
    
    def _log_execution_summary(self, results: Dict[str, Any]):
        """Log execution summary."""
        duration_minutes = results['total_duration'] / 60
        success_rate = results['successful_symbols'] / max(results['total_symbols_processed'], 1) * 100
        
        logger.info("📈 DAILY SCHEDULE EXECUTION SUMMARY")
        logger.info(f"   Duration: {duration_minutes:.1f} minutes")
        logger.info(f"   Symbols processed: {results['total_symbols_processed']}")
        logger.info(f"   ✅ Successful: {results['successful_symbols']}")
        logger.info(f"   ❌ Failed: {results['failed_symbols']}")
        logger.info(f"   🔴 Circuit breakers: {results['circuit_breakers_triggered']}")
        logger.info(f"   📊 Success rate: {success_rate:.1f}%")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        now = datetime.now()
        
        # Count by priority and status
        priority_status = defaultdict(lambda: {'total': 0, 'needs_update': 0, 'circuit_broken': 0})
        
        for scheduled in self.scheduled_symbols.values():
            priority_status[scheduled.priority]['total'] += 1
            
            if scheduled.needs_update:
                priority_status[scheduled.priority]['needs_update'] += 1
            
            if scheduled.is_circuit_broken:
                priority_status[scheduled.priority]['circuit_broken'] += 1
        
        # Recent execution stats
        recent_executions = [
            ex for ex in self.execution_history 
            if (now - ex['execution_start']).days <= 7
        ]
        
        return {
            'timestamp': now.isoformat(),
            'total_symbols': len(self.scheduled_symbols),
            'priority_distribution': dict(priority_status),
            'recent_executions': len(recent_executions),
            'last_execution': self.execution_history[-1] if self.execution_history else None
        }


def main():
    """Test the earnings scheduler."""
    from utils.logging_setup import setup_logging

    setup_logging()
    
    # Test symbols with known earnings
    test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"]
    
    scheduler = EarningsPriorityScheduler(Path("data"))
    
    print("🗓️ Testing earnings-priority scheduler...")
    
    # Refresh earnings and create schedule
    priorities = scheduler.refresh_earnings_priorities(test_symbols)
    
    print(f"\n📊 Priority assignments:")
    for symbol, priority in priorities.items():
        emoji = scheduler._get_priority_emoji(priority)
        print(f"  {emoji} {symbol}: {priority.value}")
    
    # Create daily schedule
    batches = scheduler.create_daily_schedule()
    
    print(f"\n📅 Daily schedule ({len(batches)} batches):")
    for i, batch in enumerate(batches, 1):
        print(f"  Batch {i}: {batch.priority.value} - {batch.total_symbols} symbols")
    
    # Show status report
    status = scheduler.get_status_report()
    print(f"\n📈 Status report:")
    print(f"  Total symbols: {status['total_symbols']}")
    print(f"  Priority distribution: {status['priority_distribution']}")


if __name__ == "__main__":
    main()