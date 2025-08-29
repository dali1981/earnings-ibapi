from __future__ import annotations

from datetime import date
from pydantic import BaseModel


class OptionBarsSummary(BaseModel):
    underlying: str
    bar_size: str
    start: date
    end: date
    rows: int
