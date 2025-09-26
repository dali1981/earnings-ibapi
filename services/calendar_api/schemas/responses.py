"""
Standardized API response patterns for Calendar API.

Provides consistent response formatting, error handling, and metadata
across all calendar endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from datetime import datetime
from enum import Enum

import uuid

# Generic type for response data
T = TypeVar('T')


class ResponseStatus(str, Enum):
    """Standard response status values."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"


class ResponseMetadata(BaseModel):
    """Standard metadata for all API responses."""
    
    status: ResponseStatus = ResponseStatus.SUCCESS
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_time_ms: Optional[float] = None
    api_version: str = "1.0"
    

class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    code: str = Field(description="Error code for programmatic handling")
    message: str = Field(description="Human-readable error message")
    field: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class APIResponse(BaseModel, Generic[T]):
    """
    Standard wrapper for all API responses.
    
    Provides consistent structure with data, metadata, and optional errors/warnings.
    """
    
    data: T = Field(description="Response payload")
    metadata: ResponseMetadata = Field(description="Response metadata")
    errors: Optional[List[ErrorDetail]] = Field(default=None, description="Error details if any")
    warnings: Optional[List[str]] = Field(default=None, description="Warning messages if any")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class PaginationMetadata(BaseModel):
    """Pagination information for list responses."""
    
    total_count: Optional[int] = None
    page_size: int = 100
    current_page: int = 1
    has_next_page: bool = False
    has_previous_page: bool = False


class ListResponse(APIResponse[List[T]]):
    """Standard response for list endpoints with pagination."""
    
    pagination: Optional[PaginationMetadata] = None


class HealthResponse(BaseModel):
    """Health check response format."""
    
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0"
    uptime_seconds: Optional[float] = None
    dependencies: Optional[Dict[str, str]] = None


class DataAvailabilityResponse(BaseModel):
    """Data availability analysis response."""
    
    total_events: int
    date_range_start: str
    date_range_end: str
    sources_available: List[str]
    symbols_available: int
    last_updated: Optional[datetime] = None


class StatsResponse(BaseModel):
    """Repository statistics response."""
    
    total_events: int
    total_symbols: int
    total_sources: int
    storage_size_mb: Optional[float] = None
    oldest_event_date: Optional[str] = None
    newest_event_date: Optional[str] = None
    partitions_count: Optional[int] = None


class FetchResultResponse(BaseModel):
    """Results from external data fetch operations."""
    
    events_fetched: int
    events_saved: int
    events_skipped: int = 0
    sources_used: List[str]
    symbols_requested: Optional[List[str]] = None
    symbols_found: List[str]
    missing_symbols: List[str] = []
    fetch_duration_seconds: float
    job_id: Optional[str] = None


def create_success_response(
    data: T,
    *,
    warnings: Optional[List[str]] = None,
    execution_time_ms: Optional[float] = None,
    request_id: Optional[str] = None,
) -> APIResponse[T]:
    """Create a successful API response with consistent metadata."""
    
    metadata = ResponseMetadata(
        status=ResponseStatus.SUCCESS,
        execution_time_ms=execution_time_ms,
    )
    
    if request_id:
        metadata.request_id = request_id
    
    return APIResponse[T](
        data=data,
        metadata=metadata,
        warnings=warnings,
    )


def create_error_response(
    error_code: str,
    error_message: str,
    *,
    field: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    execution_time_ms: Optional[float] = None,
    request_id: Optional[str] = None,
) -> APIResponse[None]:
    """Create an error API response with consistent structure."""
    
    metadata = ResponseMetadata(
        status=ResponseStatus.ERROR,
        execution_time_ms=execution_time_ms,
    )
    
    if request_id:
        metadata.request_id = request_id
    
    error = ErrorDetail(
        code=error_code,
        message=error_message,
        field=field,
        context=context,
    )
    
    return APIResponse[None](
        data=None,
        metadata=metadata,
        errors=[error],
    )


def create_partial_response(
    data: T,
    errors: List[ErrorDetail],
    *,
    warnings: Optional[List[str]] = None,
    execution_time_ms: Optional[float] = None,
    request_id: Optional[str] = None,
) -> APIResponse[T]:
    """Create a partial success response with some errors."""
    
    metadata = ResponseMetadata(
        status=ResponseStatus.PARTIAL,
        execution_time_ms=execution_time_ms,
    )
    
    if request_id:
        metadata.request_id = request_id
    
    return APIResponse[T](
        data=data,
        metadata=metadata,
        errors=errors,
        warnings=warnings,
    )


# Common error codes
class ErrorCodes:
    """Standard error codes for calendar API."""
    
    # Validation errors
    INVALID_PARAMETER = "INVALID_PARAMETER"
    MISSING_PARAMETER = "MISSING_PARAMETER"
    PARAMETER_OUT_OF_RANGE = "PARAMETER_OUT_OF_RANGE"
    
    # Data errors
    DATA_NOT_FOUND = "DATA_NOT_FOUND"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    
    # External service errors
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    EXTERNAL_SERVICE_TIMEOUT = "EXTERNAL_SERVICE_TIMEOUT"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # Storage errors
    STORAGE_ERROR = "STORAGE_ERROR"
    STORAGE_FULL = "STORAGE_FULL"
    
    # Generic errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"