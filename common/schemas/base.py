"""Base Pydantic models for consistent validation."""

from datetime import date, datetime
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List


class BaseAPIModel(BaseModel):
    """Base model for API requests/responses."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        use_enum_values=True,
        arbitrary_types_allowed=True
    )


class TimestampedModel(BaseAPIModel):
    """Base model with timestamp tracking."""
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()


class PaginatedResponse(BaseAPIModel):
    """Standard paginated response format."""
    
    items: List[Dict[str, Any]]
    total: int = Field(..., ge=0)
    page: int = Field(default=1, ge=1)
    size: int = Field(default=50, ge=1, le=1000)
    has_next: bool = Field(default=False)
    has_previous: bool = Field(default=False)
    
    @property
    def total_pages(self) -> int:
        """Calculate total number of pages."""
        return (self.total + self.size - 1) // self.size if self.total > 0 else 0


class DataRange(BaseAPIModel):
    """Represents a date range for data queries."""
    
    start_date: date
    end_date: date
    
    def model_post_init(self, __context) -> None:
        """Validate date range after initialization."""
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before or equal to end_date")
    
    @property
    def days(self) -> int:
        """Calculate number of days in the range."""
        return (self.end_date - self.start_date).days + 1


class RequestMetadata(BaseAPIModel):
    """Metadata for tracking requests."""
    
    request_id: str = Field(..., description="Unique request identifier")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")
    user_id: Optional[str] = Field(None, description="User making the request")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: Optional[str] = Field(None, description="Source of the request")
    
    def model_dump_minimal(self) -> Dict[str, Any]:
        """Return minimal representation for logging."""
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat()
        }