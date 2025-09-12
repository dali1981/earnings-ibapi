from __future__ import annotations

from datetime import date
from earnings.fetcher import EarningsEvent
from enum import Enum
from pydantic import BaseModel, validator
from typing import List, Optional

import math


class EarningsEventModel(BaseModel):
    symbol: str
    company_name: str
    earnings_date: date
    time: Optional[str] = None
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    source: str

    @validator(
        "eps_estimate", "eps_actual", "revenue_estimate", "revenue_actual", pre=True
    )
    def validate_float_fields(cls, v):
        """Convert NaN and Infinity to None for JSON compatibility."""
        if v is not None and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    @classmethod
    def from_entity(cls, e: EarningsEvent) -> "EarningsEventModel":
        # Handle enum types from Pydantic models gracefully
        time_val = (
            e.time.value
            if isinstance(getattr(e, "time", None), Enum)
            else getattr(e, "time", None)
        )
        source_val = (
            e.source.value
            if isinstance(getattr(e, "source", None), Enum)
            else getattr(e, "source", None)
        )
        return cls(
            symbol=e.symbol,
            company_name=e.company_name,
            earnings_date=e.earnings_date,
            time=time_val,
            eps_estimate=e.eps_estimate,
            eps_actual=e.eps_actual,
            revenue_estimate=e.revenue_estimate,
            revenue_actual=e.revenue_actual,
            source=source_val,
        )


class NewEarningsEvent(BaseModel):
    symbol: str
    company_name: str
    earnings_date: date
    time: Optional[str] = None
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    source: str


class EarningsQueryMetadata(BaseModel):
    """Metadata about an earnings query to help with UX."""

    total_events: int
    filtered_events: int
    date_range_start: date
    date_range_end: date
    symbols_requested: Optional[List[str]] = None
    symbols_found: List[str]
    latest_collection_date: Optional[str] = None
    suggestions: Optional[List[str]] = None


class EarningsResponseWithMetadata(BaseModel):
    """Enhanced response model with metadata for better UX."""

    events: List[EarningsEventModel]
    metadata: EarningsQueryMetadata
