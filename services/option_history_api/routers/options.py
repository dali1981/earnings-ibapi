from __future__ import annotations

from ..schemas.options import OptionBarsSummary
from common.repositories import OptionBarRepository
from datetime import date
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path

import config


router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/options/{underlying}/contracts")
def list_contracts(underlying: str, limit: int = Query(20, ge=1, le=200)):
    repo = OptionBarRepository(Path(config.DATA_ROOT) / "option_bars")
    try:
        contracts = repo.list_known_contracts(underlying)
        out = []
        for c in contracts[:limit]:
            out.append(
                {
                    "underlying": c[0],
                    "expiry": (
                        c[1].isoformat() if hasattr(c[1], "isoformat") else str(c[1])
                    ),
                    "right": c[2],
                    "strike": float(c[3]),
                }
            )
        return out
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/options/{underlying}/bars", response_model=OptionBarsSummary)
def get_option_bars(
    underlying: str,
    expiry: date,
    right: str,
    strike: float,
    start: date,
    end: date,
    bar_size: str = Query("1 day"),
):
    repo = OptionBarRepository(Path(config.DATA_ROOT) / "option_bars")
    try:
        df = repo.load_contract_range(
            underlying=underlying,
            expiry=expiry,
            right=right,
            strike=strike,
            bar_size=bar_size,
            start_date=start,
            end_date=end,
        )
        return OptionBarsSummary(
            underlying=underlying,
            bar_size=bar_size,
            start=start,
            end=end,
            rows=int(len(df)),
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
