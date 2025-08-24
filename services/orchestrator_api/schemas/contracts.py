from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class SymbolBatch(BaseModel):
    symbols: List[str] = Field(..., description="List of ticker symbols")
    sec_types: Optional[List[str]] = Field(default=["STK"], description="Security types")
    currency: Optional[str] = Field(default="USD", description="Currency filter")
    exchanges: Optional[List[str]] = Field(
        default=["NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "IEX"],
        description="Primary exchanges",
    )


class ContractDetailsBatch(BaseModel):
    symbols: Optional[List[str]] = Field(default=None, description="Ticker symbols")
    conids: Optional[List[int]] = Field(default=None, description="IB contract IDs")
    timeout_seconds: Optional[float] = Field(default=20.0)


class OrchestratorResult(BaseModel):
    processed: int
    saved: int
    skipped: int = 0
    details: Optional[dict] = None

