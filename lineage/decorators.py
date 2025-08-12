"""
Decorators for automatic lineage tracking.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .core import DataSource, LineageTracker, OperationType

try:
    from reliability import get_trading_logger
except ImportError:
    import logging
    
    def get_trading_logger(name: str):
        return logging.getLogger(name)


# Global lineage tracker instance
_global_tracker: Optional[LineageTracker] = None


def set_global_tracker(tracker: LineageTracker):
    """Set the global lineage tracker instance."""
    global _global_tracker
    _global_tracker = tracker


def get_global_tracker() -> Optional[LineageTracker]:
    """Get the global lineage tracker instance."""
    return _global_tracker


def track_lineage(
    operation_type: OperationType,
    input_sources: Optional[List[str]] = None,
    output_sources: Optional[List[str]] = None,
    extract_inputs: Optional[Callable] = None,
    extract_outputs: Optional[Callable] = None,
    tracker: Optional[LineageTracker] = None,
    repository_type: Optional[str] = None
):
    """
    Decorator to automatically track data lineage for function calls.
    
    Args:
        operation_type: Type of operation being performed
        input_sources: List of parameter names that represent input data sources
        output_sources: List of return value attributes that represent output sources
        extract_inputs: Custom function to extract input data sources from function args
        extract_outputs: Custom function to extract output data sources from return value
        tracker: Specific lineage tracker to use (uses global if not provided)
        repository_type: Type of repository for context
    
    Example:
        @track_lineage(
            operation_type=OperationType.WRITE,
            input_sources=['df'],
            repository_type='equity_bars'
        )
        def save(self, df, symbol=None, **kwargs):
            # Function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get tracker
            lineage_tracker = tracker or _global_tracker
            if not lineage_tracker:
                # No tracker available, execute function normally
                return func(*args, **kwargs)
            
            logger = get_trading_logger(f"lineage.{func.__name__}")
            
            # Extract function context
            execution_context = _extract_execution_context(func, args, kwargs)
            if repository_type:
                execution_context['repository_type'] = repository_type
            
            # Start operation
            operation_id = lineage_tracker.start_operation(
                operation_type=operation_type,
                parameters=_extract_parameters(func, args, kwargs),
                execution_context=execution_context
            )
            
            try:
                # Extract and add input data sources
                inputs = _extract_input_sources(
                    func, args, kwargs, input_sources, extract_inputs, repository_type
                )
                for input_source in inputs:
                    lineage_tracker.add_input(operation_id, input_source)
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Extract and add output data sources
                outputs = _extract_output_sources(
                    func, args, kwargs, result, output_sources, extract_outputs, repository_type
                )
                for output_source in outputs:
                    lineage_tracker.add_output(operation_id, output_source)
                
                # Extract record counts if possible
                record_count_in = _extract_record_count(args, kwargs, 'df')
                record_count_out = _extract_record_count_from_result(result)
                
                # Complete operation successfully
                lineage_tracker.complete_operation(
                    operation_id,
                    record_count_in=record_count_in,
                    record_count_out=record_count_out
                )
                
                return result
                
            except Exception as e:
                # Complete operation with error
                lineage_tracker.complete_operation(
                    operation_id,
                    error_info=str(e)
                )
                raise
        
        return wrapper
    return decorator


def track_repository_operation(operation_type: OperationType):
    """
    Specialized decorator for repository operations that automatically
    extracts repository information and data sources.
    
    Args:
        operation_type: Type of repository operation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get tracker
            lineage_tracker = _global_tracker
            if not lineage_tracker:
                return func(self, *args, **kwargs)
            
            # Extract repository information
            repo_info = _extract_repository_info(self)
            
            # Start operation
            operation_id = lineage_tracker.start_operation(
                operation_type=operation_type,
                parameters=_extract_parameters(func, args, kwargs),
                execution_context={
                    'function': func.__name__,
                    'class': self.__class__.__name__,
                    'repository': repo_info
                }
            )
            
            try:
                # Add input sources based on operation type
                if operation_type in [OperationType.WRITE, OperationType.TRANSFORM]:
                    # For write operations, DataFrame is input
                    df = _find_dataframe_arg(args, kwargs)
                    if df is not None:
                        input_source = DataSource(
                            source_type="dataframe",
                            source_id=f"input_df_{id(df)}",
                            location="memory",
                            metadata={
                                'shape': getattr(df, 'shape', None),
                                'columns': list(getattr(df, 'columns', []))
                            }
                        )
                        lineage_tracker.add_input(operation_id, input_source)
                
                # Execute function
                result = func(self, *args, **kwargs)
                
                # Add output sources based on operation type  
                if operation_type == OperationType.READ:
                    # For read operations, result DataFrame is output
                    if hasattr(result, 'shape'):
                        output_source = DataSource(
                            source_type="repository",
                            source_id=f"{repo_info['dataset_name']}_{operation_id}",
                            location=str(repo_info['dataset_path']),
                            metadata={
                                'shape': result.shape,
                                'columns': list(result.columns) if hasattr(result, 'columns') else []
                            }
                        )
                        lineage_tracker.add_output(operation_id, output_source)
                
                elif operation_type == OperationType.WRITE:
                    # For write operations, repository dataset is output
                    df = _find_dataframe_arg(args, kwargs)
                    output_source = DataSource(
                        source_type="repository", 
                        source_id=repo_info['dataset_name'],
                        location=str(repo_info['dataset_path']),
                        metadata={
                            'record_count': len(df) if df is not None else None,
                            'operation_params': kwargs
                        }
                    )
                    lineage_tracker.add_output(operation_id, output_source)
                
                # Extract record counts
                record_count_in = None
                record_count_out = None
                
                if operation_type == OperationType.WRITE:
                    df = _find_dataframe_arg(args, kwargs)
                    record_count_in = len(df) if df is not None else None
                elif operation_type == OperationType.READ:
                    record_count_out = len(result) if hasattr(result, '__len__') else None
                
                # Complete operation
                lineage_tracker.complete_operation(
                    operation_id,
                    record_count_in=record_count_in,
                    record_count_out=record_count_out
                )
                
                return result
                
            except Exception as e:
                lineage_tracker.complete_operation(operation_id, error_info=str(e))
                raise
        
        return wrapper
    return decorator


def _extract_execution_context(func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Extract execution context from function call."""
    context = {
        'function': func.__name__,
        'module': func.__module__,
    }
    
    # Add class information if this is a method
    if args and hasattr(args[0], '__class__'):
        context['class'] = args[0].__class__.__name__
    
    return context


def _extract_parameters(func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Extract function parameters, excluding large objects like DataFrames."""
    params = {}
    
    # Get function signature
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    
    for name, value in bound_args.arguments.items():
        # Skip self parameter
        if name == 'self':
            continue
        
        # Skip large objects
        if hasattr(value, 'shape') and hasattr(value, 'columns'):
            # This looks like a DataFrame
            params[name] = f"<DataFrame shape={getattr(value, 'shape', 'unknown')}>"
        elif isinstance(value, (dict, list)) and len(str(value)) > 1000:
            params[name] = f"<{type(value).__name__} size={len(value)}>"
        else:
            # Include small parameters
            try:
                # Test if value is JSON serializable
                import json
                json.dumps(value, default=str)
                params[name] = value
            except (TypeError, ValueError):
                params[name] = str(value)
    
    return params


def _extract_input_sources(
    func: Callable, args: tuple, kwargs: dict, 
    input_sources: Optional[List[str]],
    extract_inputs: Optional[Callable],
    repository_type: Optional[str]
) -> List[DataSource]:
    """Extract input data sources from function arguments."""
    sources = []
    
    if extract_inputs:
        # Use custom extraction function
        try:
            custom_sources = extract_inputs(func, args, kwargs)
            if custom_sources:
                sources.extend(custom_sources)
        except Exception as e:
            logger = get_trading_logger("lineage.extract_inputs")
            logger.warning(f"Custom input extraction failed: {e}")
    
    elif input_sources:
        # Extract from specified parameter names
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        for param_name in input_sources:
            if param_name in bound_args.arguments:
                value = bound_args.arguments[param_name]
                
                # Create data source for this parameter
                source = DataSource(
                    source_type="parameter",
                    source_id=f"{param_name}_{id(value)}",
                    location="memory",
                    metadata={
                        'parameter_name': param_name,
                        'type': type(value).__name__,
                        'shape': getattr(value, 'shape', None),
                        'repository_type': repository_type
                    }
                )
                sources.append(source)
    
    return sources


def _extract_output_sources(
    func: Callable, args: tuple, kwargs: dict, result: Any,
    output_sources: Optional[List[str]],
    extract_outputs: Optional[Callable],
    repository_type: Optional[str]
) -> List[DataSource]:
    """Extract output data sources from function result."""
    sources = []
    
    if extract_outputs:
        # Use custom extraction function
        try:
            custom_sources = extract_outputs(func, args, kwargs, result)
            if custom_sources:
                sources.extend(custom_sources)
        except Exception as e:
            logger = get_trading_logger("lineage.extract_outputs")
            logger.warning(f"Custom output extraction failed: {e}")
    
    elif output_sources:
        # Extract from specified result attributes
        for attr_name in output_sources:
            try:
                if hasattr(result, attr_name):
                    value = getattr(result, attr_name)
                    source = DataSource(
                        source_type="result",
                        source_id=f"{attr_name}_{id(value)}",
                        location="memory",
                        metadata={
                            'attribute_name': attr_name,
                            'type': type(value).__name__,
                            'repository_type': repository_type
                        }
                    )
                    sources.append(source)
            except Exception:
                continue
    
    # If no specific sources defined but result looks like data, track it
    if not sources and hasattr(result, 'shape'):
        source = DataSource(
            source_type="result",
            source_id=f"result_{id(result)}",
            location="memory",
            metadata={
                'type': type(result).__name__,
                'shape': getattr(result, 'shape', None),
                'repository_type': repository_type
            }
        )
        sources.append(source)
    
    return sources


def _extract_repository_info(repository_instance) -> Dict[str, Any]:
    """Extract information from repository instance."""
    info = {
        'class': repository_instance.__class__.__name__
    }
    
    # Common repository attributes
    for attr in ['dataset_name', 'dataset_path', 'base_path']:
        if hasattr(repository_instance, attr):
            info[attr] = getattr(repository_instance, attr)
    
    return info


def _find_dataframe_arg(args: tuple, kwargs: dict):
    """Find DataFrame argument in function call."""
    # Check positional args
    for arg in args:
        if hasattr(arg, 'shape') and hasattr(arg, 'columns'):
            return arg
    
    # Check keyword args
    for value in kwargs.values():
        if hasattr(value, 'shape') and hasattr(value, 'columns'):
            return value
    
    # Check specific parameter names
    for param_name in ['df', 'dataframe', 'data']:
        if param_name in kwargs:
            return kwargs[param_name]
    
    return None


def _extract_record_count(args: tuple, kwargs: dict, param_name: str = 'df') -> Optional[int]:
    """Extract record count from DataFrame parameter."""
    # Try specific parameter name first
    if param_name in kwargs:
        df = kwargs[param_name]
        if hasattr(df, '__len__'):
            return len(df)
    
    # Find any DataFrame-like argument
    df = _find_dataframe_arg(args, kwargs)
    if df and hasattr(df, '__len__'):
        return len(df)
    
    return None


def _extract_record_count_from_result(result) -> Optional[int]:
    """Extract record count from function result."""
    if hasattr(result, '__len__'):
        return len(result)
    return None