"""
Custom metrics collection and tracking for trading operations.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

import sentry_sdk

from app_settings.settings import get_settings


logger = logging.getLogger(__name__)


@dataclass
class TradingOperationResult:
    """Result of a trading operation for metrics tracking."""
    operation: str
    symbol: Optional[str] = None
    success: bool = True
    duration_ms: float = 0.0
    records_processed: int = 0
    data_source: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradingMetrics:
    """
    Comprehensive metrics collection for trading operations.
    Integrates with Sentry for centralized monitoring.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._enabled = self.settings.observability.enable_custom_metrics
        self._sample_rate = self.settings.observability.metrics_sample_rate
        
        # Local metrics storage for dashboard
        self._operation_counts = defaultdict(int)
        self._operation_durations = defaultdict(list)
        self._error_counts = defaultdict(int)
        self._recent_operations = deque(maxlen=1000)
        
        logger.info(f"TradingMetrics initialized (enabled: {self._enabled}, sample_rate: {self._sample_rate})")
    
    def record_operation(self, result: TradingOperationResult) -> None:
        """Record metrics for a trading operation."""
        if not self._enabled:
            return
        
        try:
            # Record to local storage
            self._record_local_metrics(result)
            
            # Send to Sentry if enabled
            self._record_sentry_metrics(result)
            
        except Exception as e:
            logger.error(f"Failed to record metrics for {result.operation}: {e}")
    
    def _record_local_metrics(self, result: TradingOperationResult) -> None:
        """Record metrics locally for dashboard."""
        operation_key = f"{result.operation}"
        if result.symbol:
            operation_key += f":{result.symbol}"
        
        self._operation_counts[operation_key] += 1
        self._operation_durations[operation_key].append(result.duration_ms)
        
        if not result.success:
            error_key = f"{result.operation}:{result.error_type or 'unknown'}"
            self._error_counts[error_key] += 1
        
        # Store recent operation for real-time monitoring
        self._recent_operations.append({
            'timestamp': datetime.now().isoformat(),
            'operation': result.operation,
            'symbol': result.symbol,
            'success': result.success,
            'duration_ms': result.duration_ms,
            'records': result.records_processed,
            'data_source': result.data_source,
            'error_type': result.error_type,
        })
    
    def _record_sentry_metrics(self, result: TradingOperationResult) -> None:
        """Send metrics to Sentry."""
        if not sentry_sdk.Hub.current.client:
            return
        
        tags = {
            "operation": result.operation,
            "success": str(result.success),
        }
        
        if result.symbol:
            tags["symbol"] = result.symbol
        if result.data_source:
            tags["data_source"] = result.data_source
        if result.error_type:
            tags["error_type"] = result.error_type
        
        # Use Sentry's custom metrics if available, otherwise use events
        try:
            # Try new metrics API first
            if hasattr(sentry_sdk, 'metrics') and hasattr(sentry_sdk.metrics, 'incr'):
                # Operation counter
                sentry_sdk.metrics.incr(
                    key="trading.operations.total",
                    value=1,
                    tags=tags,
                    sample_rate=self._sample_rate
                )
                
                # Duration distribution
                sentry_sdk.metrics.distribution(
                    key="trading.operations.duration",
                    value=result.duration_ms,
                    unit="millisecond",
                    tags=tags,
                    sample_rate=self._sample_rate
                )
                
                # Records processed gauge
                if result.records_processed > 0:
                    sentry_sdk.metrics.gauge(
                        key="trading.operations.records_processed",
                        value=result.records_processed,
                        tags=tags,
                        sample_rate=self._sample_rate
                    )
                
                # Error counter
                if not result.success:
                    sentry_sdk.metrics.incr(
                        key="trading.operations.errors",
                        value=1,
                        tags=tags,
                        sample_rate=self._sample_rate
                    )
            else:
                # Fallback to events for older Sentry versions
                with sentry_sdk.configure_scope() as scope:
                    for key, value in tags.items():
                        scope.set_tag(key, value)
                    
                    scope.set_context("trading_metrics", {
                        "operation": result.operation,
                        "symbol": result.symbol,
                        "success": result.success,
                        "duration_ms": result.duration_ms,
                        "records_processed": result.records_processed,
                        "data_source": result.data_source
                    })
                    
                    # Send as breadcrumb
                    sentry_sdk.add_breadcrumb(
                        message=f"Trading operation: {result.operation}",
                        category="trading.metrics",
                        level="info",
                        data={
                            "operation": result.operation,
                            "symbol": result.symbol,
                            "success": result.success,
                            "duration_ms": result.duration_ms,
                            "records_processed": result.records_processed
                        }
                    )
        except Exception as e:
            # If metrics fail, log locally but don't break the application
            logger.error(f"Failed to send Sentry metrics: {e}")
    
    def get_operation_stats(self, operation: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Filter recent operations
        relevant_ops = [
            op for op in self._recent_operations
            if op['operation'] == operation and 
            datetime.fromisoformat(op['timestamp']) > cutoff
        ]
        
        if not relevant_ops:
            return {'operation': operation, 'total_count': 0}
        
        total_count = len(relevant_ops)
        success_count = sum(1 for op in relevant_ops if op['success'])
        error_count = total_count - success_count
        
        durations = [op['duration_ms'] for op in relevant_ops]
        avg_duration = sum(durations) / len(durations)
        
        records = [op['records'] for op in relevant_ops if op['records']]
        total_records = sum(records)
        
        return {
            'operation': operation,
            'total_count': total_count,
            'success_count': success_count,
            'error_count': error_count,
            'success_rate': success_count / total_count if total_count > 0 else 0,
            'avg_duration_ms': avg_duration,
            'total_records_processed': total_records,
            'unique_symbols': len(set(op['symbol'] for op in relevant_ops if op['symbol'])),
        }
    
    def get_symbol_stats(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistics for a specific symbol."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        relevant_ops = [
            op for op in self._recent_operations
            if op['symbol'] == symbol and 
            datetime.fromisoformat(op['timestamp']) > cutoff
        ]
        
        if not relevant_ops:
            return {'symbol': symbol, 'total_count': 0}
        
        operations = defaultdict(int)
        for op in relevant_ops:
            operations[op['operation']] += 1
        
        return {
            'symbol': symbol,
            'total_operations': len(relevant_ops),
            'operations_breakdown': dict(operations),
            'success_rate': sum(1 for op in relevant_ops if op['success']) / len(relevant_ops),
            'data_sources': list(set(op['data_source'] for op in relevant_ops if op['data_source'])),
        }
    
    def get_dashboard_data(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive metrics for dashboard display."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_ops = [
            op for op in self._recent_operations
            if datetime.fromisoformat(op['timestamp']) > cutoff
        ]
        
        # Overall statistics
        total_ops = len(recent_ops)
        successful_ops = sum(1 for op in recent_ops if op['success'])
        failed_ops = total_ops - successful_ops
        
        # Operation breakdown
        operation_counts = defaultdict(int)
        operation_success_rates = defaultdict(lambda: {'total': 0, 'success': 0})
        
        for op in recent_ops:
            operation_counts[op['operation']] += 1
            operation_success_rates[op['operation']]['total'] += 1
            if op['success']:
                operation_success_rates[op['operation']]['success'] += 1
        
        # Calculate success rates
        operation_stats = {}
        for operation, counts in operation_success_rates.items():
            operation_stats[operation] = {
                'total': counts['total'],
                'success_rate': counts['success'] / counts['total'] if counts['total'] > 0 else 0
            }
        
        # Symbol activity
        symbol_activity = defaultdict(int)
        for op in recent_ops:
            if op['symbol']:
                symbol_activity[op['symbol']] += 1
        
        return {
            'time_window_hours': hours,
            'total_operations': total_ops,
            'successful_operations': successful_ops,
            'failed_operations': failed_ops,
            'overall_success_rate': successful_ops / total_ops if total_ops > 0 else 0,
            'operation_counts': dict(operation_counts),
            'operation_stats': operation_stats,
            'top_symbols': dict(sorted(symbol_activity.items(), key=lambda x: x[1], reverse=True)[:20]),
            'recent_operations': list(self._recent_operations)[-50:],  # Last 50 operations
        }


# Global metrics instance
_metrics = None


def get_trading_metrics() -> TradingMetrics:
    """Get the global TradingMetrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = TradingMetrics()
    return _metrics


@contextmanager
def track_trading_operation(
    operation: str,
    symbol: Optional[str] = None,
    data_source: Optional[str] = None,
    **metadata
):
    """
    Context manager to track a trading operation with automatic metrics recording.
    
    Usage:
        with track_trading_operation("equity_bars_fetch", symbol="AAPL", data_source="ib_api"):
            # Your operation code here
            records = fetch_equity_bars(symbol)
            # Optionally set records count
            set_operation_records(len(records))
    """
    start_time = time.time()
    metrics = get_trading_metrics()
    result = TradingOperationResult(
        operation=operation,
        symbol=symbol,
        data_source=data_source,
        metadata=metadata
    )
    
    # Set Sentry context
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("trading.operation", operation)
        if symbol:
            scope.set_tag("trading.symbol", symbol)
        if data_source:
            scope.set_tag("trading.data_source", data_source)
        
        # Set operation context
        scope.set_context("trading_operation", {
            "operation": operation,
            "symbol": symbol,
            "data_source": data_source,
            **metadata
        })
    
    try:
        # Store context for nested operations
        _operation_context.operation_result = result
        
        yield result
        
        # Operation completed successfully
        result.success = True
        
    except Exception as e:
        # Operation failed
        result.success = False
        result.error_type = type(e).__name__
        
        # Capture exception with context
        sentry_sdk.capture_exception(e)
        
        # Re-raise the exception
        raise
    
    finally:
        # Calculate duration and record metrics
        result.duration_ms = (time.time() - start_time) * 1000
        metrics.record_operation(result)
        
        # Clear context
        _operation_context.operation_result = None


class _OperationContext:
    """Thread-local context for current operation."""
    def __init__(self):
        self.operation_result: Optional[TradingOperationResult] = None


_operation_context = _OperationContext()


def set_operation_records(count: int) -> None:
    """Set the number of records processed in the current operation."""
    if _operation_context.operation_result:
        _operation_context.operation_result.records_processed = count


def add_operation_metadata(key: str, value: Any) -> None:
    """Add metadata to the current operation."""
    if _operation_context.operation_result:
        _operation_context.operation_result.metadata[key] = value