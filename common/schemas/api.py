"""API response schemas."""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from .base import BaseAPIModel, RequestMetadata


class APIResponse(BaseAPIModel):
    """Standard API response wrapper."""
    
    success: bool = Field(True, description="Whether the request was successful")
    message: Optional[str] = Field(None, description="Human-readable message")
    data: Optional[Union[Dict[str, Any], List[Any], str, int, float]] = Field(
        None, description="Response data"
    )
    metadata: Optional[RequestMetadata] = Field(None, description="Request metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def success_response(
        cls,
        data: Any = None,
        message: str = "Request completed successfully",
        metadata: Optional[RequestMetadata] = None
    ) -> "APIResponse":
        """Create a success response."""
        return cls(
            success=True,
            message=message,
            data=data,
            metadata=metadata
        )
    
    @classmethod
    def error_response(
        cls,
        message: str,
        data: Any = None,
        metadata: Optional[RequestMetadata] = None
    ) -> "APIResponse":
        """Create an error response."""
        return cls(
            success=False,
            message=message,
            data=data,
            metadata=metadata
        )


class ErrorResponse(BaseAPIModel):
    """Detailed error response."""
    
    error_code: str = Field(..., description="Machine-readable error code")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    field_errors: Optional[Dict[str, List[str]]] = Field(
        None, description="Field-specific validation errors"
    )
    trace_id: Optional[str] = Field(None, description="Trace ID for debugging")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def validation_error(
        cls,
        message: str,
        field_errors: Dict[str, List[str]],
        trace_id: Optional[str] = None
    ) -> "ErrorResponse":
        """Create a validation error response."""
        return cls(
            error_code="VALIDATION_ERROR",
            error_message=message,
            field_errors=field_errors,
            trace_id=trace_id
        )
    
    @classmethod
    def not_found_error(
        cls,
        resource: str,
        identifier: str,
        trace_id: Optional[str] = None
    ) -> "ErrorResponse":
        """Create a not found error response."""
        return cls(
            error_code="NOT_FOUND",
            error_message=f"{resource} not found",
            details={"resource": resource, "identifier": identifier},
            trace_id=trace_id
        )
    
    @classmethod
    def internal_error(
        cls,
        message: str = "Internal server error occurred",
        trace_id: Optional[str] = None
    ) -> "ErrorResponse":
        """Create an internal server error response."""
        return cls(
            error_code="INTERNAL_ERROR",
            error_message=message,
            trace_id=trace_id
        )


class HealthResponse(BaseAPIModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status (healthy, degraded, unhealthy)")
    version: Optional[str] = Field(None, description="Service version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")
    dependencies: Optional[Dict[str, str]] = Field(
        None, description="Status of service dependencies"
    )
    metrics: Optional[Dict[str, Union[str, int, float]]] = Field(
        None, description="Service metrics"
    )
    
    @classmethod
    def healthy(
        cls,
        version: Optional[str] = None,
        uptime_seconds: Optional[float] = None,
        dependencies: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, Union[str, int, float]]] = None
    ) -> "HealthResponse":
        """Create a healthy status response."""
        return cls(
            status="healthy",
            version=version,
            uptime_seconds=uptime_seconds,
            dependencies=dependencies,
            metrics=metrics
        )
    
    @classmethod
    def unhealthy(
        cls,
        version: Optional[str] = None,
        dependencies: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, Union[str, int, float]]] = None
    ) -> "HealthResponse":
        """Create an unhealthy status response."""
        return cls(
            status="unhealthy",
            version=version,
            dependencies=dependencies,
            metrics=metrics
        )


class DataAvailabilityResponse(BaseAPIModel):
    """Response for data availability queries."""
    
    dataset: str = Field(..., description="Dataset name")
    symbol: Optional[str] = Field(None, description="Symbol if applicable")
    available_dates: List[str] = Field(..., description="Available dates in ISO format")
    date_range: Optional[Dict[str, str]] = Field(
        None, description="Min/max date range"
    )
    total_records: Optional[int] = Field(None, ge=0, description="Total record count")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    
    def model_post_init(self, __context) -> None:
        """Calculate date range if not provided."""
        if self.available_dates and not self.date_range:
            sorted_dates = sorted(self.available_dates)
            self.date_range = {
                "min_date": sorted_dates[0],
                "max_date": sorted_dates[-1]
            }


class JobStatusResponse(BaseAPIModel):
    """Async job status response."""
    
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(
        ..., 
        description="Job status (pending, running, completed, failed, cancelled)"
    )
    progress: Optional[float] = Field(
        None, ge=0, le=1, description="Job progress (0.0 to 1.0)"
    )
    message: Optional[str] = Field(None, description="Status message")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    
    result: Optional[Dict[str, Any]] = Field(None, description="Job result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    monitor_url: Optional[str] = Field(None, description="URL to monitor job progress")
    
    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in ("completed", "failed", "cancelled")
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None