"""
Structured logging integration for financial trading applications.
Provides enhanced logging with context, correlation IDs, and performance metrics.
"""
import json
import time
import logging
import threading
import uuid
from typing import Any, Dict, Optional, Union, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from functools import wraps
from datetime import datetime, timezone

from .exceptions import BaseTradingException, ErrorSeverity, ErrorCategory


# Thread-local storage for correlation context
_context_storage = threading.local()


@dataclass
class LogContext:
    """Context information for structured logging."""
    
    correlation_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    operation: Optional[str] = None
    symbol: Optional[str] = None
    request_id: Optional[int] = None
    client_id: Optional[int] = None
    
    # Performance tracking
    start_time: Optional[float] = None
    
    # Additional context
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        result = asdict(self)
        # Convert start_time to human-readable format
        if result['start_time']:
            result['start_time_iso'] = datetime.fromtimestamp(
                result['start_time'], timezone.utc
            ).isoformat()
        return {k: v for k, v in result.items() if v is not None}


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def __init__(self, include_context: bool = True, extra_fields: Optional[Dict[str, str]] = None):
        super().__init__()
        self.include_context = include_context
        self.extra_fields = extra_fields or {}
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        
        # Base log structure
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
        }
        
        # Add process info
        if hasattr(record, 'process'):
            log_entry['process'] = record.process
        
        # Add exception info
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add trading-specific exception details
        if hasattr(record, 'exception_details') and record.exception_details:
            log_entry['trading_exception'] = record.exception_details
        
        # Add correlation context
        if self.include_context:
            context = get_current_context()
            if context:
                log_entry['context'] = context.to_dict()
        
        # Add performance metrics
        if hasattr(record, 'performance'):
            log_entry['performance'] = record.performance
        
        # Add custom fields from record
        for attr in ['operation', 'symbol', 'request_id', 'client_id', 
                    'duration', 'retry_attempt', 'circuit_breaker', 
                    'error_code', 'recoverable']:
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)
        
        # Add extra fields from formatter configuration
        for field_name, record_attr in self.extra_fields.items():
            if hasattr(record, record_attr):
                log_entry[field_name] = getattr(record, record_attr)
        
        return json.dumps(log_entry, default=str)


class TradingLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter with trading-specific context."""
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """Process log message with context and extra fields."""
        
        # Get current context
        context = get_current_context()
        if context:
            # Add context fields to extra
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            kwargs['extra'].update(context.to_dict())
        
        # Add adapter's extra fields
        if self.extra:
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            kwargs['extra'].update(self.extra)
        
        return msg, kwargs
    
    def log_exception(self, exception: Exception, level: int = logging.ERROR, 
                     msg: Optional[str] = None, **kwargs):
        """Log exception with enhanced details."""
        
        if not msg:
            msg = f"Exception occurred: {exception}"
        
        # Prepare exception details
        exception_details = {
            'type': type(exception).__name__,
            'message': str(exception),
        }
        
        # Add trading exception details
        if isinstance(exception, BaseTradingException):
            exception_details.update(exception.to_dict())
        
        # Add to kwargs
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra']['exception_details'] = exception_details
        kwargs['exc_info'] = True
        
        self.log(level, msg, **kwargs)
    
    def log_performance(self, operation: str, duration: float, success: bool = True, 
                       **metrics):
        """Log performance metrics."""
        
        performance_data = {
            'operation': operation,
            'duration': duration,
            'success': success,
            **metrics
        }
        
        level = logging.INFO if success else logging.WARNING
        msg = f"Operation '{operation}' {'completed' if success else 'failed'} in {duration:.3f}s"
        
        self.log(level, msg, extra={'performance': performance_data})


# Context management functions

def get_current_context() -> Optional[LogContext]:
    """Get current logging context."""
    return getattr(_context_storage, 'context', None)


def set_current_context(context: LogContext):
    """Set current logging context."""
    _context_storage.context = context


def clear_current_context():
    """Clear current logging context."""
    if hasattr(_context_storage, 'context'):
        del _context_storage.context


@contextmanager
def log_context(
    operation: Optional[str] = None,
    symbol: Optional[str] = None,
    request_id: Optional[int] = None,
    client_id: Optional[int] = None,
    session_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    **metadata
):
    """Context manager for structured logging context."""
    
    # Generate correlation ID if not provided
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
    
    # Create context
    context = LogContext(
        correlation_id=correlation_id,
        session_id=session_id,
        operation=operation,
        symbol=symbol,
        request_id=request_id,
        client_id=client_id,
        start_time=time.time(),
        metadata=metadata
    )
    
    # Store previous context
    previous_context = get_current_context()
    
    try:
        set_current_context(context)
        yield context
    finally:
        # Restore previous context
        if previous_context:
            set_current_context(previous_context)
        else:
            clear_current_context()


def performance_log(operation: str = None):
    """Decorator to automatically log function performance."""
    
    def decorator(func: Callable) -> Callable:
        operation_name = operation or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_trading_logger(func.__module__)
            start_time = time.time()
            success = True
            exception = None
            
            try:
                with log_context(operation=operation_name):
                    result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                success = False
                exception = e
                raise
                
            finally:
                duration = time.time() - start_time
                
                # Log performance
                logger.log_performance(
                    operation_name, 
                    duration, 
                    success=success,
                    args_count=len(args),
                    kwargs_count=len(kwargs)
                )
                
                # Log exception if occurred
                if exception:
                    logger.log_exception(exception)
        
        return wrapper
    return decorator


def get_trading_logger(name: str, **extra_context) -> TradingLoggerAdapter:
    """Get a trading-specific logger adapter."""
    logger = logging.getLogger(name)
    return TradingLoggerAdapter(logger, extra_context)


def configure_trading_logging(
    level: Union[str, int] = logging.INFO,
    format_type: str = "json",  # "json" or "text"
    include_context: bool = True,
    log_file: Optional[str] = None,
    console_logging: bool = True
):
    """Configure structured logging for trading application."""
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    if format_type == "json":
        formatter = StructuredFormatter(include_context=include_context)
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Add console handler
    if console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    configure_library_loggers(level)


def configure_library_loggers(level: Union[str, int]):
    """Configure logging for specific libraries."""
    
    # Reduce noise from external libraries
    library_loggers = [
        'urllib3',
        'requests',
        'matplotlib',
        'pandas'
    ]
    
    for lib in library_loggers:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    # Set trading-specific loggers to debug if needed
    trading_loggers = [
        'reliability',
        'ibx_repos',
        'api',
        'streamer'
    ]
    
    for logger_name in trading_loggers:
        logging.getLogger(logger_name).setLevel(level)


# Enhanced logging functions

def log_ib_api_call(
    operation: str,
    request_id: int,
    symbol: Optional[str] = None,
    logger: Optional[TradingLoggerAdapter] = None
):
    """Log IB API call initiation."""
    if not logger:
        logger = get_trading_logger('api.ib')
    
    with log_context(operation=operation, symbol=symbol, request_id=request_id):
        logger.info(f"IB API call initiated: {operation}")


def log_ib_api_response(
    operation: str,
    request_id: int,
    success: bool = True,
    data_size: Optional[int] = None,
    error_code: Optional[str] = None,
    logger: Optional[TradingLoggerAdapter] = None
):
    """Log IB API call response."""
    if not logger:
        logger = get_trading_logger('api.ib')
    
    extra_data = {
        'request_id': request_id,
        'success': success
    }
    
    if data_size is not None:
        extra_data['data_size'] = data_size
    if error_code:
        extra_data['error_code'] = error_code
    
    level = logging.INFO if success else logging.ERROR
    status = "completed" if success else "failed"
    
    logger.log(level, f"IB API call {status}: {operation}", extra=extra_data)


def log_data_operation(
    operation: str,
    repository: str,
    record_count: Optional[int] = None,
    duration: Optional[float] = None,
    logger: Optional[TradingLoggerAdapter] = None
):
    """Log data repository operation."""
    if not logger:
        logger = get_trading_logger('data.repository')
    
    extra_data = {
        'operation': operation,
        'repository': repository
    }
    
    if record_count is not None:
        extra_data['record_count'] = record_count
    if duration is not None:
        extra_data['duration'] = duration
    
    logger.info(f"Data operation: {operation} on {repository}", extra=extra_data)


# Circuit breaker and retry logging integration

def log_retry_attempt(
    operation: str,
    attempt: int,
    max_attempts: int,
    delay: float,
    exception: Exception,
    logger: Optional[TradingLoggerAdapter] = None
):
    """Log retry attempt with context."""
    if not logger:
        logger = get_trading_logger('reliability.retry')
    
    extra_data = {
        'retry_attempt': attempt,
        'max_attempts': max_attempts,
        'delay': delay,
        'exception_type': type(exception).__name__
    }
    
    logger.warning(
        f"Retry attempt {attempt}/{max_attempts} for {operation} in {delay:.2f}s: {exception}",
        extra=extra_data
    )


def log_circuit_breaker_event(
    name: str,
    old_state: str,
    new_state: str,
    failure_count: int,
    logger: Optional[TradingLoggerAdapter] = None
):
    """Log circuit breaker state change."""
    if not logger:
        logger = get_trading_logger('reliability.circuit_breaker')
    
    extra_data = {
        'circuit_breaker': name,
        'old_state': old_state,
        'new_state': new_state,
        'failure_count': failure_count
    }
    
    level = logging.WARNING if new_state == 'open' else logging.INFO
    
    logger.log(
        level,
        f"Circuit breaker '{name}' state change: {old_state} -> {new_state}",
        extra=extra_data
    )