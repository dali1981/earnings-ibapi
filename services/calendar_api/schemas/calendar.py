from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel

from earnings.fetcher import EarningsEvent
from enum import Enum


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

    @classmethod
    def from_entity(cls, e: EarningsEvent) -> "EarningsEventModel":
        # Handle enum types from Pydantic models gracefully
        time_val = e.time.value if isinstance(getattr(e, "time", None), Enum) else getattr(e, "time", None)
        source_val = e.source.value if isinstance(getattr(e, "source", None), Enum) else getattr(e, "source", None)
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
