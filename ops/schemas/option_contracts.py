from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class OptionSelection(BaseModel):
    symbol: str
    correlation_id: Optional[str] = None
    snapshot_date: Optional[str] = None
    expiry_filter: Optional[str] = None
    max_contracts: int = 50
    rights: Optional[List[str]] = None
    strikes_per_side: Optional[int] = None
    selection_mode: Optional[str] = None  # count | moneyness | delta
    expiry_count: Optional[int] = None
    spot_price: Optional[float] = None
    moneyness_targets: Optional[List[float]] = None
    delta_targets: Optional[List[float]] = None
    iv_assumption: Optional[float] = None
    use_underlying_iv: Optional[bool] = None
    root: Optional[str] = None


class OptionHistoricalArgs(BaseModel):
    symbol: str
    correlation_id: Optional[str] = None
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
    timeout_seconds: float = 20.0
    root: Optional[str] = None


class SelectedContracts(BaseModel):
    symbol: str
    contracts: List[Dict[str, Any]]
    contract_count: int
    snapshot_date: Optional[str] = None
    correlation_id: Optional[str] = None


class HistoricalResult(BaseModel):
    symbol: Optional[str] = None
    processed: int
    saved: int
    failed: int
    missing_symbols: Optional[List[str]] = None
    results: Optional[List[Dict[str, Any]]] = None
    correlation_id: Optional[str] = None


class SaveContractsResult(BaseModel):
    saved: int
    skipped: int
    missing_conid: int
    error: Optional[str] = None
