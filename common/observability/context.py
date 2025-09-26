"""
Observability context management for maintaining trace context across async operations.
"""

import asyncio
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Optional

import sentry_sdk


logger = logging.getLogger(__name__)


# Context variables for async operations
_correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
_trading_symbol: ContextVar[Optional[str]] = ContextVar('trading_symbol', default=None)
_trading_operation: ContextVar[Optional[str]] = ContextVar('trading_operation', default=None)
_data_source: ContextVar[Optional[str]] = ContextVar('data_source', default=None)


class ObservabilityContext:
    """
    Context manager for maintaining observability context across operations.
    Handles correlation IDs, trading context, and Sentry scope management.
    """
    
    def __init__(
        self,
        correlation_id: Optional[str] = None,
        symbol: Optional[str] = None,
        operation: Optional[str] = None,
        data_source: Optional[str] = None,
        **extra_context
    ):
        self.correlation_id = correlation_id
        self.symbol = symbol
        self.operation = operation
        self.data_source = data_source
        self.extra_context = extra_context
        
        # Store previous context for restoration
        self._previous_correlation_id = None
        self._previous_symbol = None
        self._previous_operation = None
        self._previous_data_source = None
    
    def __enter__(self):
        # Store previous context
        self._previous_correlation_id = _correlation_id.get()
        self._previous_symbol = _trading_symbol.get()
        self._previous_operation = _trading_operation.get()
        self._previous_data_source = _data_source.get()
        
        # Set new context
        if self.correlation_id:
            _correlation_id.set(self.correlation_id)
        if self.symbol:
            _trading_symbol.set(self.symbol)
        if self.operation:
            _trading_operation.set(self.operation)
        if self.data_source:
            _data_source.set(self.data_source)
        
        # Configure Sentry scope
        sentry_sdk.configure_scope(self._configure_sentry_scope)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        _correlation_id.set(self._previous_correlation_id)
        _trading_symbol.set(self._previous_symbol)
        _trading_operation.set(self._previous_operation)
        _data_source.set(self._previous_data_source)
        
        # If there was an exception, capture it with context
        if exc_type and exc_val:
            self.capture_exception(exc_val)
    
    def _configure_sentry_scope(self, scope):
        """Configure Sentry scope with current context."""
        # Set correlation ID as user ID for tracking
        if self.correlation_id:
            scope.set_user({"id": self.correlation_id})
            scope.set_tag("correlation_id", self.correlation_id)
        
        # Set trading context
        trading_context = {}
        if self.symbol:
            scope.set_tag("trading.symbol", self.symbol)
            trading_context["symbol"] = self.symbol
        if self.operation:
            scope.set_tag("trading.operation", self.operation)
            trading_context["operation"] = self.operation
        if self.data_source:
            scope.set_tag("trading.data_source", self.data_source)
            trading_context["data_source"] = self.data_source
        
        # Add extra context
        if self.extra_context:
            trading_context.update(self.extra_context)
            for key, value in self.extra_context.items():
                if isinstance(value, str):
                    scope.set_tag(f"custom.{key}", value)
        
        if trading_context:
            scope.set_context("trading", trading_context)
    
    def capture_exception(self, exception: Exception) -> None:
        """Capture exception with current context."""
        with sentry_sdk.configure_scope() as scope:
            # Add current context as fingerprint for grouping
            fingerprint = ["trading-error", self.operation or "unknown"]
            if self.symbol:
                fingerprint.append(self.symbol)
            scope.fingerprint = fingerprint
            
            # Add context to exception
            scope.set_context("error_context", {
                "correlation_id": self.correlation_id,
                "symbol": self.symbol,
                "operation": self.operation,
                "data_source": self.data_source,
                **self.extra_context
            })
        
        sentry_sdk.capture_exception(exception)
    
    def capture_message(self, message: str, level: str = "info") -> None:
        """Capture message with current context."""
        with sentry_sdk.configure_scope() as scope:
            scope.set_context("message_context", {
                "correlation_id": self.correlation_id,
                "symbol": self.symbol,
                "operation": self.operation,
                "data_source": self.data_source,
                **self.extra_context
            })
        
        sentry_sdk.capture_message(message, level=level)
    
    @classmethod
    def current(cls) -> 'ObservabilityContext':
        """Get current context from context vars."""
        return cls(
            correlation_id=_correlation_id.get(),
            symbol=_trading_symbol.get(),
            operation=_trading_operation.get(),
            data_source=_data_source.get()
        )


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from context."""
    return _correlation_id.get()


def get_trading_symbol() -> Optional[str]:
    """Get current trading symbol from context."""
    return _trading_symbol.get()


def get_trading_operation() -> Optional[str]:
    """Get current trading operation from context."""
    return _trading_operation.get()


def get_data_source() -> Optional[str]:
    """Get current data source from context."""
    return _data_source.get()


@contextmanager
def trading_context(
    symbol: str,
    operation: str,
    correlation_id: Optional[str] = None,
    data_source: Optional[str] = None,
    **extra_context
):
    """
    Context manager for trading operations.
    
    Usage:
        with trading_context("AAPL", "bars_fetch", correlation_id="req-123", data_source="ib_api"):
            # Your trading operation code
            bars = fetch_bars(symbol)
    """
    with ObservabilityContext(
        correlation_id=correlation_id,
        symbol=symbol,
        operation=operation,
        data_source=data_source,
        **extra_context
    ):
        yield


@contextmanager
def correlation_context(correlation_id: str):
    """
    Context manager for correlation ID tracking.
    
    Usage:
        with correlation_context("req-123"):
            # All operations in this context will be correlated
            process_request()
    """
    with ObservabilityContext(correlation_id=correlation_id):
        yield


def propagate_context_to_task(task_func, *args, **kwargs):
    """
    Propagate current observability context to an asyncio task.
    
    Usage:
        # In async function
        current_context = ObservabilityContext.current()
        task = asyncio.create_task(
            propagate_context_to_task(some_async_function, arg1, arg2)
        )
    """
    current_context = ObservabilityContext.current()
    
    async def wrapped_task():
        with current_context:
            return await task_func(*args, **kwargs)
    
    return wrapped_task()


def copy_context_to_thread(target_func, *args, **kwargs):
    """
    Copy current observability context to a thread.
    
    Usage:
        import threading
        current_context = ObservabilityContext.current()
        thread = threading.Thread(
            target=copy_context_to_thread,
            args=(worker_function, arg1, arg2)
        )
        thread.start()
    """
    current_context = ObservabilityContext.current()
    
    def wrapped_target():
        with current_context:
            return target_func(*args, **kwargs)
    
    return wrapped_target()