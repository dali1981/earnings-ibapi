from __future__ import annotations

from datetime import date
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

import config
from common.repositories import EquityBarRepository
import pandas as pd

from ..schemas.equity import EquityBarsResponse, EquityDateRange


router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/equity/{symbol}/dates", response_model=EquityDateRange)
def get_dates(symbol: str, bar_size: str = Query("1 day")):
    repo = EquityBarRepository(Path(config.DATA_ROOT) / "equity_bars")
    try:
        start = pd.Timestamp(date.today()).date().replace(year=date.today().year - 5)
        end = date.today()
        present = repo.present_dates(symbol, bar_size, start, end)
        return EquityDateRange(dates=sorted(present))
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/equity/{symbol}/bars", response_model=EquityBarsResponse)
def get_bars(symbol: str, start: date, end: date, bar_size: str = Query("1 day")):
    repo = EquityBarRepository(Path(config.DATA_ROOT) / "equity_bars")
    try:
        df = repo.load_range(symbol=symbol, bar_size=bar_size, start_date=start, end_date=end)
        sample = df.head(50).to_dict(orient="records") if not df.empty else []
        return EquityBarsResponse(symbol=symbol, bar_size=bar_size, start=start, end=end, rows=int(len(df)), sample=sample)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

