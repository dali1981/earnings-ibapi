from __future__ import annotations

from common.repositories import OptionChainSnapshotRepository
from datetime import date, datetime
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List

import config


router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/chains/{symbol}/latest")
def latest_snapshot_date(symbol: str):
    # Pass base data root; repository appends dataset name internally
    repo = OptionChainSnapshotRepository(Path(config.DATA_ROOT))
    try:
        d = repo.get_latest_snapshot_date(symbol)
        if d is None:
            raise HTTPException(status_code=404, detail="No snapshots found")
        return {"symbol": symbol.upper(), "as_of": d.isoformat()}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


def _parse_as_of(as_of_str: str) -> date:
    try:
        return datetime.fromisoformat(as_of_str).date()
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid as_of date format; use YYYY-MM-DD"
        )


@router.get("/chains/{symbol}/as-ofs")
def list_as_ofs(symbol: str):
    repo = OptionChainSnapshotRepository(Path(config.DATA_ROOT))
    try:
        tbl = repo.load(underlying=symbol.upper(), columns=["as_of"])  # Arrow table
        if tbl is None or tbl.num_rows == 0:
            raise HTTPException(status_code=404, detail="No snapshots found")
        as_of_vals = [d.isoformat() for d in tbl.column("as_of").to_pylist() if d]
        as_of_vals = sorted(sorted(set(as_of_vals)))
        latest = as_of_vals[-1] if as_of_vals else None
        return {"symbol": symbol.upper(), "as_of_list": as_of_vals, "latest": latest}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chains/{symbol}/{as_of}/summary")
def snapshot_summary(symbol: str, as_of: str):
    repo = OptionChainSnapshotRepository(Path(config.DATA_ROOT))
    as_of_date = _parse_as_of(as_of)
    try:
        tbl = repo.load(underlying=symbol.upper(), as_of=as_of_date)
        if tbl is None or tbl.num_rows == 0:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        expirations_col = (
            tbl.column("expirations").to_pylist()
            if "expirations" in tbl.column_names
            else []
        )
        strikes_col = (
            tbl.column("strikes").to_pylist() if "strikes" in tbl.column_names else []
        )
        all_exp = sorted({e for lst in expirations_col for e in (lst or [])})
        all_str = sorted({float(s) for lst in strikes_col for s in (lst or [])})
        strike_range = [all_str[0], all_str[-1]] if all_str else [None, None]
        return {
            "symbol": symbol.upper(),
            "as_of": as_of_date.isoformat(),
            "param_sets": int(tbl.num_rows),
            "expiry_count": len(all_exp),
            "strike_range": strike_range,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chains/{symbol}/{as_of}/expiries")
def list_expiries(symbol: str, as_of: str):
    repo = OptionChainSnapshotRepository(Path(config.DATA_ROOT))
    as_of_date = _parse_as_of(as_of)
    try:
        tbl = repo.load(underlying=symbol.upper(), as_of=as_of_date)
        if tbl is None or tbl.num_rows == 0:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        expirations_col = (
            tbl.column("expirations").to_pylist()
            if "expirations" in tbl.column_names
            else []
        )
        all_exp = sorted({e for lst in expirations_col for e in (lst or [])})
        return {
            "symbol": symbol.upper(),
            "as_of": as_of_date.isoformat(),
            "expirations": all_exp,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chains/{symbol}/{as_of}/strikes")
def list_strikes(symbol: str, as_of: str):
    repo = OptionChainSnapshotRepository(Path(config.DATA_ROOT))
    as_of_date = _parse_as_of(as_of)
    try:
        tbl = repo.load(underlying=symbol.upper(), as_of=as_of_date)
        if tbl is None or tbl.num_rows == 0:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        strikes_col = (
            tbl.column("strikes").to_pylist() if "strikes" in tbl.column_names else []
        )
        all_str = sorted({float(s) for lst in strikes_col for s in (lst or [])})
        return {
            "symbol": symbol.upper(),
            "as_of": as_of_date.isoformat(),
            "strikes": all_str,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chains/{symbol}/{as_of}/params")
def get_params(symbol: str, as_of: str):
    repo = OptionChainSnapshotRepository(Path(config.DATA_ROOT))
    as_of_date = _parse_as_of(as_of)
    try:
        tbl = repo.load(underlying=symbol.upper(), as_of=as_of_date)
        if tbl is None or tbl.num_rows == 0:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        # Return parameter sets as JSON-serializable rows
        cols = {name: tbl.column(name).to_pylist() for name in tbl.column_names}
        rows: List[dict] = []
        for i in range(tbl.num_rows):
            rows.append({name: cols[name][i] for name in tbl.column_names})
        return {
            "symbol": symbol.upper(),
            "as_of": as_of_date.isoformat(),
            "params": rows,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
