from __future__ import annotations

from ..schemas.contracts import OrchestratorResult
from ..schemas.option_contracts import (
    SelectionRequest,
    EnrichRequest,
    SaveRequest,
    OptionHistoricalRequest,
)
from ops.schemas.option_contracts import OptionSelection, OptionHistoricalArgs, SelectedContracts
from fastapi import Response
from ibx_celery.app import app as celery_app
from fastapi import APIRouter, HTTPException, Query


router = APIRouter(prefix="/option-contracts", tags=["option-contracts"])


@router.post("/select", response_model=OrchestratorResult)
def select_option_contracts(payload: SelectionRequest) -> OrchestratorResult:
    """
    Selects option contracts for a symbol from the latest (or specified) chain snapshot.

    - Reads aggregated chain parameters (expirations/strikes) from option_chains repo
    - Applies your selection criteria (mode, expiry_count, rights, etc.)
    - Returns IB-style contracts (may not include conId unless previously enriched)
    """
    from ops.tasks.option_bars import get_option_contracts_for_bars_task

    sym = payload.symbol.strip().upper()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol must not be empty")

    # Pass JSON-serializable payload to Celery; task will coerce to model
    job = get_option_contracts_for_bars_task.s(selection=payload.model_dump()).apply_async()

    return OrchestratorResult(
        processed=1,
        saved=0,
        skipped=0,
        details={"task_id": job.id, "symbol": sym},
    )


@router.post("/fullflow", response_model=OrchestratorResult)
def option_fullflow(payload: OptionHistoricalRequest) -> OrchestratorResult:
    """
    One-click flow: select -> enrich -> save -> historical.

    - Uses repo-only selection
    - Enrich adds conId/localSymbol via IB
    - Save persists enriched option contracts
    - Historical fetches bars for selected contracts and saves to option_bars
    """
    from celery import chain
    from ops.tasks.option_bars import (
        get_option_contracts_for_bars_task,
        fetch_option_contract_details_task,
        save_selected_option_contracts_to_repo_task,
        fetch_option_historical_data_task,
    )

    selection = OptionSelection(**payload.model_dump())
    hist_kwargs = dict(
        end_date_time=payload.end_date_time,
        duration_str=payload.duration_str,
        bar_size_setting=payload.bar_size_setting,
        what_to_show=payload.what_to_show,
        use_rth=payload.use_rth,
        timeout_seconds=payload.timeout_seconds if hasattr(payload, 'timeout_seconds') else 20.0,
    )

    flow = (
        get_option_contracts_for_bars_task.s(selection=selection.model_dump())
        | fetch_option_contract_details_task.s()
        | save_selected_option_contracts_to_repo_task.s()
        | fetch_option_historical_data_task.s(**hist_kwargs)
    )
    res = flow.apply_async()
    return OrchestratorResult(processed=1, saved=0, skipped=0, details={"task_id": res.id, "symbol": payload.symbol})


@router.get("/option-bars")
def browse_option_bars(
    symbol: str = Query(..., description="Underlying symbol"),
    expiry: str | None = Query(None, description="Expiry date ISO (YYYY-MM-DD)"),
    right: str | None = Query(None, description="C or P"),
    strike: float | None = Query(None, description="Strike price"),
    bar_size: str | None = Query(None, description="Bar size (e.g., 1 day)"),
    start_date: str | None = Query(None, description="Start date ISO"),
    end_date: str | None = Query(None, description="End date ISO"),
    limit: int = Query(100, ge=1, le=2000),
):
    """Browse saved option bars filtered by symbol/expiry/right/strike/bar_size."""
    from datetime import date
    import pandas as pd
    from app_settings.settings import get_settings
    from common.repositories.option_bars import OptionBarRepository

    sym = symbol.strip().upper()
    settings = get_settings()
    repo = OptionBarRepository(settings.paths.data_root)

    kw = {"underlying": sym}
    if expiry:
        try:
            kw["expiry"] = pd.to_datetime(expiry).date()
        except Exception:
            pass
    if right:
        kw["right"] = right.strip().upper()
    if strike is not None:
        kw["strike"] = float(strike)
    if bar_size:
        kw["bar_size"] = bar_size

    df = repo.load(**kw)
    if df.empty:
        return {"rows": [], "total": 0}

    if start_date:
        try:
            sd = pd.to_datetime(start_date).date()
            df = df[df["trade_date"] >= sd]
        except Exception:
            pass
    if end_date:
        try:
            ed = pd.to_datetime(end_date).date()
            df = df[df["trade_date"] <= ed]
        except Exception:
            pass

    df = df.sort_values(["time"]).head(limit)
    # Ensure JSON serializable
    out = df.copy()
    for col in ["time"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.strftime("%Y-%m-%d")
    if "expiry" in out.columns:
        out["expiry"] = pd.to_datetime(out["expiry"]).dt.strftime("%Y-%m-%d")
    rows = out.to_dict(orient="records")
    return {"rows": rows, "total": int(len(df))}


@router.post("/select-sync", response_model=SelectedContracts)
def select_option_contracts_sync(payload: SelectionRequest) -> SelectedContracts:
    """Synchronously select and return option contracts from stored snapshots.

    Uses Celery to execute the same task but waits up to a short timeout
    (fast, repo-only). Returns the selected contracts directly to the client.
    """
    from celery.exceptions import TimeoutError

    selection = OptionSelection(**payload.model_dump())
    job = get_option_contracts_for_bars_task.s(selection=selection).apply_async()
    try:
        result = job.get(timeout=10.0)  # selection is repo-only; should be quick
    except TimeoutError:
        raise HTTPException(status_code=504, detail=f"Selection timed out (task_id={job.id})")
    # Ensure result is returned as SelectedContracts
    if isinstance(result, dict):
        return SelectedContracts(**result)
    return result


@router.get("/result/{task_id}")
def get_selection_result(task_id: str, response: Response):
    """Non-blocking check of a selection task result.

    - 200: returns SelectedContracts when task SUCCESS
    - 202: task not ready (PENDING/STARTED/RETRY) with {status}
    - 500: task FAILURE with {status,error}
    """
    async_res = celery_app.AsyncResult(task_id)
    state = async_res.state
    if state == "SUCCESS":
        payload = async_res.result
        if isinstance(payload, dict):
            return SelectedContracts(**payload)
        if isinstance(payload, SelectedContracts):
            return payload
        # Best effort fallback
        try:
            return SelectedContracts(**(payload or {}))
        except Exception:
            return {"status": "SUCCESS", "raw": payload}
    if state in ("PENDING", "STARTED", "RETRY"):
        response.status_code = 202
        return {"status": state, "task_id": task_id}
    # FAILURE or REVOKED -> return 200 with status to avoid frontend fetchJson throwing
    err = None
    tb = None
    try:
        err = str(async_res.result)
    except Exception:
        err = None
    try:
        tb = async_res.traceback
    except Exception:
        tb = None
    return {"status": state, "task_id": task_id, "error": err, "traceback": tb}


@router.post("/enrich", response_model=OrchestratorResult)
def enrich_option_contracts(payload: EnrichRequest) -> OrchestratorResult:
    """Optionally enrich selected contracts with conId/localSymbol via IB service."""
    from ops.tasks.option_bars import fetch_option_contract_details_task
    job = fetch_option_contract_details_task.s(
        contracts_result=payload.contracts_result.model_dump(),
        timeout_seconds=payload.timeout_seconds,
    ).apply_async()

    return OrchestratorResult(
        processed=1,
        saved=0,
        skipped=0,
        details={"task_id": job.id},
    )


@router.post("/save", response_model=OrchestratorResult)
def save_option_contracts(payload: SaveRequest) -> OrchestratorResult:
    """Persist option contracts that already have conId into the contracts repo."""
    from ops.tasks.option_bars import save_selected_option_contracts_to_repo_task
    job = save_selected_option_contracts_to_repo_task.s(
        contracts_result=payload.contracts_result.model_dump()
    ).apply_async()

    return OrchestratorResult(
        processed=1,
        saved=0,
        skipped=0,
        details={"task_id": job.id},
    )


@router.post("/historical", response_model=OrchestratorResult)
def option_historical(payload: OptionHistoricalRequest) -> OrchestratorResult:
    """
    Dispatch historical bars collection for selected option contracts.

    - Selection criteria (mode, expiries, rights) are applied first
    - Then we request historical bars for those contracts (repository_type=option)
    """
    from ops.tasks.option_bars import process_historical_option_data_task
    sym = payload.symbol.strip().upper()
    job = process_historical_option_data_task.s(request=payload.model_dump()).apply_async()

    return OrchestratorResult(
        processed=1,
        saved=0,
        skipped=0,
        details={"task_id": job.id, "symbol": sym},
    )
