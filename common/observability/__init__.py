"""
Observability module for comprehensive monitoring and error tracking.
"""

from .core import init_observability, get_sentry_hub, capture_trading_event, capture_trading_exception
from .metrics import TradingMetrics, track_trading_operation, get_trading_metrics
from .tracing import trace_operation, trace_span, set_trace_tags, add_breadcrumb
from .context import ObservabilityContext, trading_context, correlation_context
from .celery_integration import init_celery_observability, monitored_task
from .decorators import (
    monitor_repository_operation,
    monitor_api_endpoint,
    monitor_ib_api_call,
    monitor_data_processing,
    monitor_background_job,
)

__all__ = [
    "init_observability",
    "get_sentry_hub", 
    "capture_trading_event",
    "capture_trading_exception",
    "TradingMetrics",
    "track_trading_operation",
    "get_trading_metrics",
    "trace_operation",
    "trace_span",
    "set_trace_tags",
    "add_breadcrumb",
    "ObservabilityContext",
    "trading_context",
    "correlation_context",
    "init_celery_observability",
    "monitored_task",
    "monitor_repository_operation",
    "monitor_api_endpoint",
    "monitor_ib_api_call",
    "monitor_data_processing",
    "monitor_background_job",
]