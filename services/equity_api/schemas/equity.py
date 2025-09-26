from __future__ import annotations

from datetime import date
from pydantic import BaseModel
from typing import List


class EquityDateRange(BaseModel):
    dates: List[date]


class EquityBarsResponse(BaseModel):
    symbol: str
    bar_size: str
    start: date
    end: date
    rows: int
    sample: List[dict]
