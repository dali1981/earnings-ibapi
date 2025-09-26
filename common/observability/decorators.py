"""
Observability decorators for easy integration with existing code.
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar

from .core import capture_trading_exception
from .metrics import track_trading_operation
from .tracing import trace_operation
from .context import trading_context


logger = logging.getLogger(__name__)


F = TypeVar('F', bound=Callable[..., Any])


def monitor_repository_operation(
    operation_name: str,
    extract_symbol: Optional[Callable] = None,
    extract_metadata: Optional[Callable] = None
) -> Callable[[F], F]:
    """
    Decorator to monitor repository operations with comprehensive observability.
    
    Args:
        operation_name: Name of the operation (e.g., 'equity_bars_save', 'option_chains_load')
        extract_symbol: Function to extract symbol from function arguments
        extract_metadata: Function to extract additional metadata from arguments
    
    Usage:
        @monitor_repository_operation(
            "equity_bars_save",
            extract_symbol=lambda args, kwargs: kwargs.get('symbol', args[1] if len(args) > 1 else None)
        )
        def save(self, bars_df: pd.DataFrame, symbol: str, bar_size: str):
            # Repository implementation
            pass
    """
    def decorator(func: F) -> F:
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract context information
            symbol = None
            if extract_symbol:
                try:
                    symbol = extract_symbol(args, kwargs)
                except Exception:
                    pass
            
            metadata = {}
            if extract_metadata:
                try:
                    metadata = extract_metadata(args, kwargs) or {}
                except Exception:
                    pass
            
            # Use comprehensive tracking
            with track_trading_operation(
                operation=operation_name,
                symbol=symbol,
                data_source="repository",
                **metadata
            ) as op_result:
                
                # Add tracing
                with trace_operation(operation_name, f"Repository operation: {operation_name}"):
                    
                    try:
                        result = func(*args, **kwargs)
                        
                        # Extract record count if possible
                        if hasattr(result, '__len__'):
                            op_result.records_processed = len(result)
                        elif isinstance(result, dict) and 'count' in result:
                            op_result.records_processed = result['count']
                        
                        return result
                    
                    except Exception as e:
                        # Log additional context for repository errors
                        logger.error(
                            f"Repository operation failed: {operation_name}",
                            extra={
                                'symbol': symbol,
                                'operation': operation_name,
                                'error': str(e),
                                'metadata': metadata
                            }
                        )
                        raise
        
        return wrapper
    return decorator


def monitor_api_endpoint(
    operation_name: Optional[str] = None,
    extract_symbol: Optional[Callable] = None
) -> Callable[[F], F]:
    """
    Decorator to monitor FastAPI endpoint operations.
    
    Args:
        operation_name: Optional custom operation name
        extract_symbol: Function to extract symbol from function arguments
    
    Usage:
        @router.get("/equity/{symbol}/bars")
        @monitor_api_endpoint("equity_bars_get", extract_symbol=lambda args, kwargs: args[0])
        def get_equity_bars(symbol: str, start: str, end: str):
            # Endpoint implementation
            pass
    """
    def decorator(func: F) -> F:
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"api.{func.__name__}"
            
            # Extract symbol
            symbol = None
            if extract_symbol:
                try:
                    symbol = extract_symbol(args, kwargs)
                except Exception:
                    pass
            
            # Monitor the API call
            with track_trading_operation(
                operation=op_name,
                symbol=symbol,
                data_source="api"
            ):
                
                with trace_operation(op_name, f"API endpoint: {func.__name__}"):
                    return func(*args, **kwargs)
        
        return wrapper
    return decorator


def monitor_ib_api_call(
    operation_name: str,
    extract_symbol: Optional[Callable] = None,
    timeout_seconds: int = 30
) -> Callable[[F], F]:
    """
    Decorator to monitor IB API calls with timeouts and error handling.
    
    Args:
        operation_name: Name of the IB API operation
        extract_symbol: Function to extract symbol from arguments
        timeout_seconds: Timeout for the operation
    
    Usage:
        @monitor_ib_api_call(
            "ib_historical_bars",
            extract_symbol=lambda args, kwargs: kwargs.get('contract', {}).get('symbol')
        )
        def req_historical_data(self, req_id: int, contract, end_date_time: str, duration: str, bar_size: str):
            # IB API call implementation
            pass
    """
    def decorator(func: F) -> F:
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract context
            symbol = None
            if extract_symbol:
                try:
                    symbol = extract_symbol(args, kwargs)
                except Exception:
                    pass
            
            with track_trading_operation(
                operation=operation_name,
                symbol=symbol,
                data_source="ib_api",
                timeout_seconds=timeout_seconds
            ):
                
                with trace_operation(operation_name, f"IB API call: {func.__name__}"):
                    start_time = time.time()
                    
                    try:
                        result = func(*args, **kwargs)
                        duration = time.time() - start_time
                        
                        # Log slow API calls
                        if duration > timeout_seconds * 0.8:
                            logger.warning(
                                f"Slow IB API call: {operation_name} took {duration:.2f}s",
                                extra={
                                    'symbol': symbol,
                                    'operation': operation_name,
                                    'duration': duration,
                                    'threshold': timeout_seconds
                                }
                            )
                        
                        return result
                    
                    except Exception as e:
                        duration = time.time() - start_time
                        
                        # Enhanced error context for IB API failures
                        error_context = {
                            'symbol': symbol,
                            'operation': operation_name,
                            'duration': duration,
                            'error': str(e),
                            'error_type': type(e).__name__
                        }
                        
                        logger.error(f"IB API call failed: {operation_name}", extra=error_context)
                        
                        # Capture with additional IB-specific context
                        capture_trading_exception(
                            e,
                            extra=error_context,
                            tags={
                                'api_provider': 'interactive_brokers',
                                'operation_type': 'api_call',
                                'symbol': symbol or 'unknown'
                            }
                        )
                        
                        raise
        
        return wrapper
    return decorator


def monitor_data_processing(
    operation_name: str,
    extract_symbol: Optional[Callable] = None,
    extract_records: Optional[Callable] = None
) -> Callable[[F], F]:
    """
    Decorator to monitor data processing operations (ETL, transformations, etc.).
    
    Args:
        operation_name: Name of the processing operation
        extract_symbol: Function to extract symbol from arguments
        extract_records: Function to extract record count from result
    
    Usage:
        @monitor_data_processing(
            "option_chain_processing",
            extract_symbol=lambda args, kwargs: kwargs.get('symbol'),
            extract_records=lambda result: len(result) if isinstance(result, list) else 0
        )
        def process_option_chain(symbol: str, chain_data: dict):
            # Processing implementation
            pass
    """
    def decorator(func: F) -> F:
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract context
            symbol = None
            if extract_symbol:
                try:
                    symbol = extract_symbol(args, kwargs)
                except Exception:
                    pass
            
            with track_trading_operation(
                operation=operation_name,
                symbol=symbol,
                data_source="processing"
            ) as op_result:
                
                with trace_operation(operation_name, f"Data processing: {func.__name__}"):
                    
                    result = func(*args, **kwargs)
                    
                    # Extract record count
                    if extract_records:
                        try:
                            record_count = extract_records(result)
                            op_result.records_processed = record_count
                        except Exception:
                            pass
                    
                    return result
        
        return wrapper
    return decorator


def monitor_background_job(
    job_name: str,
    extract_symbols: Optional[Callable] = None
) -> Callable[[F], F]:
    """
    Decorator to monitor background jobs and batch operations.
    
    Args:
        job_name: Name of the background job
        extract_symbols: Function to extract symbols list from arguments
    
    Usage:
        @monitor_background_job(
            "daily_update",
            extract_symbols=lambda args, kwargs: kwargs.get('symbols', [])
        )
        def daily_update_job(symbols: List[str], lookback_days: int = 5):
            # Job implementation
            pass
    """
    def decorator(func: F) -> F:
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract symbols list
            symbols = []
            if extract_symbols:
                try:
                    symbols = extract_symbols(args, kwargs) or []
                except Exception:
                    pass
            
            # Use the first symbol for tracking, or None if no symbols
            primary_symbol = symbols[0] if symbols else None
            
            with track_trading_operation(
                operation=job_name,
                symbol=primary_symbol,
                data_source="background_job",
                symbols_count=len(symbols),
                symbols=symbols[:10]  # Limit to first 10 symbols to avoid large payloads
            ):
                
                with trace_operation(job_name, f"Background job: {func.__name__}"):
                    return func(*args, **kwargs)
        
        return wrapper
    return decorator