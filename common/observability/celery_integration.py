"""
Celery integration for comprehensive task monitoring and error tracking.
"""

import functools
import logging
from typing import Any, Callable, Dict, Optional, TypeVar

import sentry_sdk
from celery import Celery
from celery.signals import (
    before_task_publish,
    task_failure,
    task_postrun,
    task_prerun,
    task_retry,
    task_success,
    worker_init,
    worker_shutdown,
)

from common.observability.context import ObservabilityContext
from common.observability.metrics import TradingOperationResult, get_trading_metrics
from app_settings.settings import get_settings


logger = logging.getLogger(__name__)


F = TypeVar('F', bound=Callable[..., Any])


def init_celery_observability(celery_app: Celery) -> None:
    """
    Initialize Celery observability with Sentry integration and custom metrics.
    
    Args:
        celery_app: The Celery application instance
    """
    settings = get_settings()
    obs_config = settings.observability
    
    if not obs_config.enable_celery_monitoring:
        logger.info("Celery monitoring disabled, skipping initialization")
        return
    
    # Connect to Celery signals for comprehensive monitoring
    _connect_celery_signals()
    
    logger.info("Celery observability initialized with Sentry integration")


def _connect_celery_signals():
    """Connect to Celery signals for monitoring task lifecycle."""
    
    @worker_init.connect
    def init_worker_observability(sender=None, **kwargs):
        """Initialize observability when worker starts."""
        from common.observability import init_observability
        init_observability(service_name="celery-worker", service_version="1.0.0")
        logger.info(f"Celery worker {sender} observability initialized")
    
    @worker_shutdown.connect
    def shutdown_worker_observability(sender=None, **kwargs):
        """Cleanup when worker shuts down."""
        logger.info(f"Celery worker {sender} observability shutdown")
    
    @before_task_publish.connect
    def track_task_published(sender=None, headers=None, body=None, properties=None, **kwargs):
        """Track when a task is published to the queue."""
        task_name = headers.get('task') if headers else 'unknown'
        correlation_id = headers.get('correlation_id') if headers else None
        
        # Add breadcrumb for task publishing
        sentry_sdk.add_breadcrumb(
            message=f"Task published: {task_name}",
            category="celery.publish",
            level="info",
            data={
                "task": task_name,
                "correlation_id": correlation_id,
                "routing_key": properties.get('routing_key') if properties else None,
            }
        )
    
    @task_prerun.connect
    def track_task_start(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
        """Track when a task starts execution."""
        task_name = sender.name if sender else 'unknown'
        
        # Extract correlation_id and trading context from task kwargs
        correlation_id = kwargs.get('correlation_id') if kwargs else None
        symbol = kwargs.get('symbol') if kwargs else None
        
        # Set up observability context for the task
        context = ObservabilityContext(
            correlation_id=correlation_id,
            symbol=symbol,
            operation=f"celery.task.{task_name}",
            data_source="celery"
        )
        
        # Store context in task for cleanup
        setattr(task, '_observability_context', context)
        
        with context:
            sentry_sdk.add_breadcrumb(
                message=f"Task started: {task_name}",
                category="celery.task",
                level="info",
                data={
                    "task_id": task_id,
                    "task": task_name,
                    "correlation_id": correlation_id,
                    "symbol": symbol,
                    "args_count": len(args) if args else 0,
                    "kwargs_keys": list(kwargs.keys()) if kwargs else [],
                }
            )
    
    @task_postrun.connect
    def track_task_end(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
        """Track when a task completes (success or failure)."""
        task_name = sender.name if sender else 'unknown'
        
        # Get context from task if available
        context = getattr(task, '_observability_context', None)
        
        if context:
            with context:
                sentry_sdk.add_breadcrumb(
                    message=f"Task completed: {task_name} (state: {state})",
                    category="celery.task",
                    level="info",
                    data={
                        "task_id": task_id,
                        "task": task_name,
                        "state": state,
                        "has_retval": retval is not None,
                    }
                )
            
            # Cleanup context
            delattr(task, '_observability_context')
    
    @task_success.connect
    def track_task_success(sender=None, result=None, **kwargs):
        """Track successful task completion."""
        task_name = sender.name if sender else 'unknown'
        
        # Record success metrics
        metrics = get_trading_metrics()
        operation_result = TradingOperationResult(
            operation=f"celery.task.{task_name}",
            success=True,
            records_processed=_extract_records_from_result(result),
        )
        metrics.record_operation(operation_result)
        
        logger.debug(f"Celery task succeeded: {task_name}")
    
    @task_failure.connect
    def track_task_failure(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
        """Track task failures."""
        task_name = sender.name if sender else 'unknown'
        
        # Capture exception with context
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("celery.task", task_name)
            scope.set_tag("celery.task_id", task_id)
            scope.set_context("celery_failure", {
                "task": task_name,
                "task_id": task_id,
                "exception_type": type(exception).__name__ if exception else None,
            })
        
        sentry_sdk.capture_exception(exception)
        
        # Record failure metrics
        metrics = get_trading_metrics()
        operation_result = TradingOperationResult(
            operation=f"celery.task.{task_name}",
            success=False,
            error_type=type(exception).__name__ if exception else 'unknown',
        )
        metrics.record_operation(operation_result)
        
        logger.error(f"Celery task failed: {task_name} - {exception}")
    
    @task_retry.connect
    def track_task_retry(sender=None, task_id=None, reason=None, einfo=None, **kwargs):
        """Track task retries."""
        task_name = sender.name if sender else 'unknown'
        
        sentry_sdk.add_breadcrumb(
            message=f"Task retry: {task_name}",
            category="celery.retry",
            level="warning",
            data={
                "task_id": task_id,
                "task": task_name,
                "reason": str(reason) if reason else None,
            }
        )
        
        logger.warning(f"Celery task retry: {task_name} - {reason}")


def _extract_records_from_result(result: Any) -> int:
    """Extract record count from task result for metrics."""
    if isinstance(result, dict):
        # Common patterns for record counts
        for key in ['records', 'count', 'processed', 'rows', 'items']:
            if key in result and isinstance(result[key], int):
                return result[key]
    
    return 0


def monitored_task(
    name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> Callable[[F], F]:
    """
    Decorator to add enhanced monitoring to Celery tasks.
    
    Args:
        name: Optional custom name for the operation
        tags: Optional tags to add to traces
    
    Usage:
        @celery_app.task
        @monitored_task(name="equity_bars_backfill", tags={"data_source": "ib_api"})
        def backfill_equity_bars(symbol: str, start_date: str, end_date: str):
            # Task implementation
            pass
    """
    def decorator(func: F) -> F:
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = name or f"celery.task.{func.__name__}"
            
            # Extract trading context from arguments
            symbol = kwargs.get('symbol')
            correlation_id = kwargs.get('correlation_id')
            data_source = kwargs.get('data_source', 'celery')
            
            with ObservabilityContext(
                correlation_id=correlation_id,
                symbol=symbol,
                operation=operation_name,
                data_source=data_source
            ):
                
                # Start Sentry transaction for the task
                with sentry_sdk.start_transaction(
                    op="celery.task",
                    name=operation_name,
                    description=f"Celery task: {func.__name__}"
                ) as transaction:
                    
                    # Add tags
                    transaction.set_tag("celery.task", func.__name__)
                    transaction.set_tag("celery.function", func.__name__)
                    
                    if symbol:
                        transaction.set_tag("trading.symbol", symbol)
                    if data_source:
                        transaction.set_tag("trading.data_source", data_source)
                    
                    # Add custom tags
                    if tags:
                        for key, value in tags.items():
                            transaction.set_tag(key, value)
                    
                    # Add safe task arguments as data
                    safe_args = _sanitize_task_args(args, kwargs)
                    transaction.set_data("celery.args", safe_args)
                    
                    try:
                        result = func(*args, **kwargs)
                        transaction.set_status("ok")
                        return result
                    
                    except Exception as e:
                        transaction.set_status("internal_error")
                        transaction.set_tag("error.type", type(e).__name__)
                        raise
        
        return wrapper
    return decorator


def _sanitize_task_args(args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Sanitize task arguments to remove sensitive data."""
    sensitive_keys = ['password', 'token', 'key', 'secret', 'dsn', 'api_key']
    
    sanitized = {
        'args_count': len(args),
        'kwargs_keys': list(kwargs.keys())
    }
    
    # Add safe arguments
    safe_kwargs = {}
    for key, value in kwargs.items():
        if not any(sensitive in key.lower() for sensitive in sensitive_keys):
            # Limit string length to avoid huge payloads
            if isinstance(value, str) and len(value) > 100:
                safe_kwargs[key] = value[:100] + "..."
            elif isinstance(value, (int, float, bool)):
                safe_kwargs[key] = value
            else:
                safe_kwargs[key] = str(type(value).__name__)
    
    if safe_kwargs:
        sanitized['safe_kwargs'] = safe_kwargs
    
    return sanitized