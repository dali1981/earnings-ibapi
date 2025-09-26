from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class SymbolBatch(BaseModel):
    symbols: List[str] = Field(..., description="List of ticker symbols")
    sec_types: Optional[List[str]] = Field(
        default=["STK"], description="Security types"
    )
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
    job_id: Optional[str] = None
    monitor_url: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: str  # PENDING, STARTED, SUCCESS, FAILURE, RETRY, REVOKED
    result: Optional[dict] = None
    error: Optional[str] = None
    progress: Optional[dict] = None
    children: Optional[List[str]] = None  # For group/chain jobs
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class JobList(BaseModel):
    jobs: List[JobStatus]
    total: int


class HistoricalDataRequest(BaseModel):
    """Request for historical data by contract specifications."""

    contracts: List[Dict[str, Any]] = Field(
        ...,
        description="List of IB contract specifications",
        examples=[
            [
                {
                    "symbol": "AAPL",
                    "secType": "STK",
                    "exchange": "SMART",
                    "currency": "USD",
                }
            ]
        ],
    )
    end_date_time: str = Field(
        ...,
        description="End date/time for historical data",
        examples=["20240823 23:59:59", ""],
    )
    duration_str: str = Field(
        ..., description="Duration string", examples=["1 D", "1 M", "1 Y", "2 W"]
    )
    bar_size_setting: str = Field(
        ...,
        description="Bar size setting",
        examples=["1 min", "5 mins", "1 hour", "1 day"],
    )
    what_to_show: str = Field(
        default="TRADES",
        description="Data type to retrieve",
        examples=["TRADES", "MIDPOINT", "BID", "ASK"],
    )
    use_rth: int = Field(
        default=1, description="1 for regular trading hours only, 0 for all hours"
    )
    repository_type: str = Field(
        default="equity",
        description="Repository type for saving data",
        examples=["equity", "option"],
    )
    timeout_seconds: float = Field(default=20.0)


class HistoricalDataBySymbolsRequest(BaseModel):
    """Request for historical data by ticker symbols (will resolve contracts first)."""

    symbols: List[str] = Field(..., description="List of ticker symbols")
    end_date_time: str = Field(
        ...,
        description="End date/time for historical data",
        examples=["20240823 23:59:59", ""],
    )
    duration_str: str = Field(
        ..., description="Duration string", examples=["1 D", "1 M", "1 Y", "2 W"]
    )
    bar_size_setting: str = Field(
        ...,
        description="Bar size setting",
        examples=["1 min", "5 mins", "1 hour", "1 day"],
    )
    what_to_show: str = Field(
        default="TRADES",
        description="Data type to retrieve",
        examples=["TRADES", "MIDPOINT", "BID", "ASK"],
    )
    use_rth: int = Field(
        default=1, description="1 for regular trading hours only, 0 for all hours"
    )
    repository_type: str = Field(
        default="equity",
        description="Repository type for saving data",
        examples=["equity", "option"],
    )
    timeout_seconds: float = Field(default=20.0)


class OptionChainRequest(BaseModel):
    """Request for option chain snapshot collection."""

    symbols: List[str] = Field(..., description="List of ticker symbols")
    correlation_id: Optional[str] = Field(
        default=None, description="Correlation ID for cross-service tracing"
    )
    snapshot_date: Optional[str] = Field(
        default=None,
        description="Snapshot date (ISO format), defaults to today",
        examples=["2024-08-25"],
    )


class OptionHistoricalDataRequest(BaseModel):
    """Request for historical option data collection."""

    symbols: List[str] = Field(..., description="List of ticker symbols")
    correlation_id: Optional[str] = Field(
        default=None, description="Correlation ID for cross-service tracing"
    )
    end_date_time: str = Field(
        ...,
        description="End date/time for historical data",
        examples=["20240823 23:59:59"],
    )
    duration_str: str = Field(
        ..., description="Duration string", examples=["1 D", "1 M", "1 Y", "2 W"]
    )
    bar_size_setting: str = Field(
        ...,
        description="Bar size setting",
        examples=["1 min", "5 mins", "1 hour", "1 day"],
    )
    what_to_show: str = Field(
        default="TRADES",
        description="Data type to retrieve",
        examples=["TRADES", "MIDPOINT", "BID", "ASK"],
    )
    use_rth: int = Field(
        default=1, description="1 for regular trading hours only, 0 for all hours"
    )
    snapshot_date: Optional[str] = Field(
        default=None,
        description="Option chain snapshot date (ISO format), defaults to latest",
        examples=["2024-08-25"],
    )
    expiry_filter: Optional[str] = Field(
        default=None,
        description="Filter to specific expiry date (ISO format)",
        examples=["2024-09-20"],
    )
    max_contracts: int = Field(
        default=50, description="Maximum number of option contracts per symbol"
    )
    timeout_seconds: float = Field(default=20.0)
