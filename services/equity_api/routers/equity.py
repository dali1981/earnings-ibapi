from __future__ import annotations

from ..schemas.equity import EquityBarsResponse, EquityDateRange
from common.repositories import EquityBarRepository
from common.observability import monitor_api_endpoint, capture_trading_event
from datetime import date
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path

import config
import pandas as pd


router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/equity/{symbol}/dates", response_model=EquityDateRange)
@monitor_api_endpoint("equity_dates_get", extract_symbol=lambda args, kwargs: args[0] if args else None)
def get_dates(symbol: str, bar_size: str = Query("1 day")):
    repo = EquityBarRepository(Path(config.DATA_ROOT) / "equity_bars")
    try:
        start = pd.Timestamp(date.today()).date().replace(year=date.today().year - 5)
        end = date.today()
        present = repo.present_dates(symbol, bar_size, start, end)
        
        # Log successful operation
        capture_trading_event(
            f"Successfully retrieved dates for {symbol}",
            level="info",
            extra={"symbol": symbol, "bar_size": bar_size, "dates_count": len(present)},
            tags={"operation": "equity_dates_get", "symbol": symbol}
        )
        
        return EquityDateRange(dates=sorted(present))
    except Exception as e:
        # Enhanced error context
        capture_trading_event(
            f"Failed to retrieve dates for {symbol}: {str(e)}",
            level="error",
            extra={"symbol": symbol, "bar_size": bar_size, "error": str(e)},
            tags={"operation": "equity_dates_get", "symbol": symbol, "error_type": type(e).__name__}
        )
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/equity/{symbol}/bars", response_model=EquityBarsResponse)
@monitor_api_endpoint("equity_bars_get", extract_symbol=lambda args, kwargs: args[0] if args else None)
def get_bars(symbol: str, start: date, end: date, bar_size: str = Query("1 day")):
    repo = EquityBarRepository(Path(config.DATA_ROOT) / "equity_bars")
    try:
        df = repo.load_range(
            symbol=symbol, bar_size=bar_size, start_date=start, end_date=end
        )
        sample = df.head(50).to_dict(orient="records") if not df.empty else []
        return EquityBarsResponse(
            symbol=symbol,
            bar_size=bar_size,
            start=start,
            end=end,
            rows=int(len(df)),
            sample=sample,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
