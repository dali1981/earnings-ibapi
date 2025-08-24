from __future__ import annotations

from datetime import date
from typing import List

from pydantic import BaseModel


class EquityDateRange(BaseModel):
    dates: List[date]


class EquityBarsResponse(BaseModel):
    symbol: str
    bar_size: str
    start: date
    end: date
    rows: int
    sample: List[dict]

