from __future__ import annotations

from common.repositories import OptionChainSnapshotRepository
from fastapi import APIRouter, HTTPException
from pathlib import Path

import config


router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/chains/{symbol}/latest")
def latest_snapshot_date(symbol: str):
    repo = OptionChainSnapshotRepository(Path(config.DATA_ROOT) / "option_chains")
    try:
        d = repo.get_latest_snapshot_date(symbol)
        if d is None:
            raise HTTPException(status_code=404, detail="No snapshots found")
        return {"symbol": symbol.upper(), "as_of": d.isoformat()}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
