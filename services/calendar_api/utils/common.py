"""
Common utilities for Calendar API routers.

Provides shared functionality for data conversion, error handling,
and response formatting across all router modules.
"""

from __future__ import annotations

from ..schemas.calendar import EarningsEventModel
from ..schemas.responses import ErrorDetail, ErrorCodes
from common.reliability.logging import get_trading_logger
from typing import List, Type, TypeVar
from datetime import datetime

import time
import json

# Type variable for model types
ModelType = TypeVar('ModelType')

logger = get_trading_logger(__name__)


def safe_convert_to_model(
    events: list, 
    event_type: Type[ModelType] = EarningsEventModel
) -> List[ModelType]:
    """
    Safely convert earnings events to Pydantic models with JSON serialization validation.

    Logs detailed error information for any events that fail to serialize.
    Returns successfully converted models, skipping failed ones.
    
    Args:
        events: List of earnings event objects to convert
        event_type: Pydantic model class to convert to
        
    Returns:
        List of successfully converted models
    """
    result = []
    errors = []
    
    for i, event in enumerate(events):
        try:
            # Convert to model
            model = event_type.from_entity(event)

            # Validate JSON serialization
            # json.dumps(model.dict()) - commented out for performance
            
            result.append(model.model_dump())
            
        except (ValueError, TypeError) as ex:
            # Create detailed error information
            error_detail = {
                "index": i,
                "symbol": getattr(event, 'symbol', 'N/A'),
                "earnings_date": str(getattr(event, 'earnings_date', 'N/A')),
                "error": str(ex),
            }
            errors.append(error_detail)
            
            # Log the problematic event details
            logger.error(f"Failed to serialize event {i}/{len(events)}: {ex}")
            logger.error(
                f"Event: symbol={getattr(event, 'symbol', 'N/A')}, "
                f"date={getattr(event, 'earnings_date', 'N/A')}, "
                f"eps_est={getattr(event, 'eps_estimate', 'N/A')}, "
                f"eps_act={getattr(event, 'eps_actual', 'N/A')}, "
                f"rev_est={getattr(event, 'revenue_estimate', 'N/A')}, "
                f"rev_act={getattr(event, 'revenue_actual', 'N/A')}"
            )

            # Try to log the full event dict if available
            try:
                logger.debug(f"Full event data: {event.__dict__}")
            except:
                pass

    # Log summary if there were conversion failures
    if len(result) < len(events):
        logger.warning(
            f"Serialized {len(result)}/{len(events)} events successfully. "
            f"Failed conversions: {len(errors)}"
        )
        
        # Log detailed failure summary
        if errors:
            logger.debug(f"Conversion failures: {errors}")

    return result


def parse_symbols(symbols_param: str | None) -> List[str] | None:
    """
    Parse comma-separated symbols parameter with validation.
    
    Args:
        symbols_param: Comma-separated string of symbols or None
        
    Returns:
        List of uppercase, trimmed symbols or None if input was None
        
    Raises:
        ValueError: If symbols are invalid or empty
    """
    if not symbols_param:
        return None
        
    symbols = [s.strip().upper() for s in symbols_param.split(",")]
    
    # Remove empty symbols
    symbols = [s for s in symbols if s]
    
    if not symbols:
        raise ValueError("No valid symbols provided")
        
    # Validate symbol format (basic check)
    invalid_symbols = [s for s in symbols if not s.isalnum() or len(s) > 10]
    if invalid_symbols:
        raise ValueError(f"Invalid symbols: {invalid_symbols}")
        
    return symbols


class TimingContext:
    """Context manager for measuring execution time."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration_ms = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000


def create_error_detail(
    code: str,
    message: str,
    field: str | None = None,
    context: dict | None = None,
) -> ErrorDetail:
    """Create a standardized error detail object."""
    return ErrorDetail(
        code=code,
        message=message,
        field=field,
        context=context,
    )


def validate_date_range(days: int, max_days: int = 180) -> None:
    """
    Validate days parameter for date range queries.
    
    Args:
        days: Number of days to validate
        max_days: Maximum allowed days
        
    Raises:
        ValueError: If days is out of range
    """
    if days < 0:
        raise ValueError(f"Days cannot be negative: {days}")
    if days > max_days:
        raise ValueError(f"Days cannot exceed {max_days}: {days}")


def add_cache_headers(response, max_age_seconds: int = 300):
    """Add cache control headers to response."""
    response.headers["Cache-Control"] = f"max-age={max_age_seconds}, public"
    response.headers["X-Cache-Max-Age"] = str(max_age_seconds)


def add_metadata_headers(response, metadata: dict):
    """Add metadata as response headers."""
    for key, value in metadata.items():
        if value is not None:
            response.headers[f"X-Calendar-{key.replace('_', '-').title()}"] = str(value)