"""
Distributed tracing utilities for trading operations.
"""

import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, TypeVar, Union

import sentry_sdk
from sentry_sdk.tracing import Transaction, Span

from app_settings.settings import get_settings


logger = logging.getLogger(__name__)


F = TypeVar('F', bound=Callable[..., Any])


def trace_operation(
    operation_name: str,
    description: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> Callable[[F], F]:
    """
    Decorator to automatically trace a function as a Sentry transaction.
    
    Args:
        operation_name: Name of the operation for tracing
        description: Optional description of the operation
        tags: Optional tags to add to the span
    
    Usage:
        @trace_operation("equity_bars_fetch", "Fetch equity bars from IB API")
        def fetch_equity_bars(symbol: str, start_date: str, end_date: str):
            # Function implementation
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            settings = get_settings()
            
            # Skip tracing if disabled
            if not settings.observability.enable_distributed_tracing:
                return func(*args, **kwargs)
            
            with sentry_sdk.start_transaction(
                op="trading.operation",
                name=operation_name,
                description=description or f"Trading operation: {operation_name}"
            ) as transaction:
                
                # Add default tags
                transaction.set_tag("trading.operation", operation_name)
                transaction.set_tag("function", func.__name__)
                transaction.set_tag("module", func.__module__)
                
                # Add custom tags
                if tags:
                    for key, value in tags.items():
                        transaction.set_tag(key, value)
                
                # Extract common parameters as tags
                if args and hasattr(args[0], '__dict__') and 'symbol' in args[0].__dict__:
                    transaction.set_tag("trading.symbol", args[0].symbol)
                elif 'symbol' in kwargs:
                    transaction.set_tag("trading.symbol", kwargs['symbol'])
                
                try:
                    result = func(*args, **kwargs)
                    transaction.set_status("ok")
                    return result
                
                except Exception as e:
                    transaction.set_status("internal_error")
                    transaction.set_tag("error.type", type(e).__name__)
                    sentry_sdk.capture_exception(e)
                    raise
        
        return wrapper
    return decorator


@contextmanager
def trace_span(
    operation: str,
    description: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    data: Optional[Dict[str, Any]] = None
):
    """
    Context manager to create a traced span for a code block.
    
    Args:
        operation: Operation name for the span
        description: Optional description
        tags: Optional tags to add
        data: Optional data to attach
    
    Usage:
        with trace_span("database_query", "Fetch equity bars", tags={"symbol": "AAPL"}):
            records = repository.get_bars(symbol, start_date, end_date)
    """
    settings = get_settings()
    
    # Skip tracing if disabled
    if not settings.observability.enable_distributed_tracing:
        yield None
        return
    
    with sentry_sdk.start_span(
        op="trading.operation",
        description=description or operation
    ) as span:
        
        # Set operation name
        span.set_tag("operation", operation)
        
        # Add custom tags
        if tags:
            for key, value in tags.items():
                span.set_tag(key, value)
        
        # Add data
        if data:
            for key, value in data.items():
                span.set_data(key, value)
        
        try:
            yield span
            span.set_status("ok")
        
        except Exception as e:
            span.set_status("internal_error")
            span.set_tag("error.type", type(e).__name__)
            span.set_data("error.message", str(e))
            raise


def set_trace_tags(**tags: Union[str, int, float, bool]) -> None:
    """
    Set tags on the current transaction or span.
    
    Args:
        **tags: Tags to set as key-value pairs
    """
    current_span = sentry_sdk.Hub.current.scope.span
    if current_span:
        for key, value in tags.items():
            current_span.set_tag(key, str(value))


def set_trace_data(**data: Any) -> None:
    """
    Set data on the current transaction or span.
    
    Args:
        **data: Data to set as key-value pairs
    """
    current_span = sentry_sdk.Hub.current.scope.span
    if current_span:
        for key, value in data.items():
            current_span.set_data(key, value)


def add_breadcrumb(
    message: str,
    category: str = "trading",
    level: str = "info",
    data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add a breadcrumb to the current context for debugging.
    
    Args:
        message: Breadcrumb message
        category: Breadcrumb category
        level: Log level (debug, info, warning, error, fatal)
        data: Optional data to attach
    """
    sentry_sdk.add_breadcrumb(
        message=message,
        category=category,
        level=level,
        data=data or {}
    )


def trace_celery_task(
    task_name: str,
    tags: Optional[Dict[str, str]] = None
) -> Callable[[F], F]:
    """
    Decorator specifically for Celery tasks to ensure proper tracing.
    
    Args:
        task_name: Name of the Celery task
        tags: Optional tags to add
    
    Usage:
        @trace_celery_task("backfill_equity_bars")
        @celery_app.task
        def backfill_equity_bars(symbol: str, start_date: str, end_date: str):
            # Task implementation
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            settings = get_settings()
            
            # Skip tracing if disabled
            if not settings.observability.enable_celery_monitoring:
                return func(*args, **kwargs)
            
            # Extract task context from Celery
            task_id = getattr(func, 'request', {}).get('id')
            queue_name = getattr(func, 'request', {}).get('delivery_info', {}).get('routing_key')
            
            with sentry_sdk.start_transaction(
                op="celery.task",
                name=task_name,
                description=f"Celery task: {task_name}"
            ) as transaction:
                
                # Add Celery-specific tags
                transaction.set_tag("celery.task", task_name)
                transaction.set_tag("celery.function", func.__name__)
                if task_id:
                    transaction.set_tag("celery.task_id", task_id)
                if queue_name:
                    transaction.set_tag("celery.queue", queue_name)
                
                # Add custom tags
                if tags:
                    for key, value in tags.items():
                        transaction.set_tag(key, value)
                
                # Add task arguments as data (careful with sensitive data)
                if args or kwargs:
                    safe_args = _sanitize_task_args(args, kwargs)
                    transaction.set_data("celery.args", safe_args)
                
                try:
                    result = func(*args, **kwargs)
                    transaction.set_status("ok")
                    return result
                
                except Exception as e:
                    transaction.set_status("internal_error")
                    transaction.set_tag("error.type", type(e).__name__)
                    sentry_sdk.capture_exception(e)
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