from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from ops.schemas.option_contracts import SelectedContracts


class SelectionRequest(BaseModel):
    symbol: str = Field(..., description="Underlying symbol")
    correlation_id: Optional[str] = Field(
        default=None, description="Correlation ID for cross-service tracing"
    )
    snapshot_date: Optional[str] = Field(None, description="as_of date (ISO)")
    expiry_filter: Optional[str] = Field(None, description="expiry (ISO)")
    rights: Optional[List[str]] = Field(None, description="subset of ['C','P']")
    strikes_per_side: Optional[int] = Field(
        None, description="strike count around ATM per side"
    )
    max_contracts: int = Field(50, description="max total contracts")
    selection_mode: Optional[str] = Field(None, description="count | moneyness | delta")
    expiry_count: Optional[int] = Field(None, description="nearest N expiries")
    spot_price: Optional[float] = Field(None, description="manual spot override")
    moneyness_targets: Optional[List[float]] = Field(
        None, description="e.g. [0.9,1.0,1.1]"
    )
    delta_targets: Optional[List[float]] = Field(
        None, description="e.g. [0.25,0.5]"
    )
    iv_assumption: Optional[float] = Field(None, description="e.g. 0.25")
    use_underlying_iv: Optional[bool] = Field(
        None, description="use IV index if present"
    )


class EnrichRequest(BaseModel):
    contracts_result: SelectedContracts
    timeout_seconds: Optional[float] = 15.0
    correlation_id: Optional[str] = Field(
        default=None, description="Correlation ID for cross-service tracing"
    )


class SaveRequest(BaseModel):
    contracts_result: SelectedContracts
    correlation_id: Optional[str] = Field(
        default=None, description="Correlation ID for cross-service tracing"
    )


class OptionHistoricalRequest(BaseModel):
    symbol: str
    correlation_id: Optional[str] = Field(
        default=None, description="Correlation ID for cross-service tracing"
    )
    end_date_time: str
    duration_str: str
    bar_size_setting: str
    what_to_show: str = "TRADES"
    use_rth: int = 1
    snapshot_date: Optional[str] = None
    expiry_filter: Optional[str] = None
    max_contracts: int = 50
    rights: Optional[List[str]] = None
    strikes_per_side: Optional[int] = None
    selection_mode: Optional[str] = None
    expiry_count: Optional[int] = None
    spot_price: Optional[float] = None
    moneyness_targets: Optional[List[float]] = None
    delta_targets: Optional[List[float]] = None
    iv_assumption: Optional[float] = None
    use_underlying_iv: Optional[bool] = None
