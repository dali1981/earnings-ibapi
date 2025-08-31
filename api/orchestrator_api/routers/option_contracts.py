from __future__ import annotations

from ..schemas.contracts import OrchestratorResult
from ..schemas.option_contracts import (
    SelectionRequest,
    EnrichRequest,
    SaveRequest,
    OptionHistoricalRequest,
    FullFlowSmartRequest,
)
from ops.schemas.option_contracts import OptionSelection, OptionHistoricalArgs, SelectedContracts
from fastapi import Response
from ibx_celery.app import app as celery_app
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel


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


@router.get("/availability")
def get_availability(
    symbol: str = Query(..., description="Underlying symbol"),
    expiry: str | None = Query(None, description="Filter for a specific expiry (YYYY-MM-DD)"),
):
    """Report availability of option chain snapshots, option contracts, and option bars for a symbol.

    Returns counts and booleans to help the UI enable/disable actions.
    """
    import pandas as pd
    from app_settings.settings import get_settings
    from common.repositories.option_chains import OptionChainSnapshotRepository
    from common.repositories.option_contracts import OptionContractsRepository
    from common.repositories.option_bars import OptionBarRepository

    settings = get_settings()
    sym = symbol.strip().upper()

    # Chain snapshots
    chain_repo = OptionChainSnapshotRepository(settings.paths.data_root)
    try:
        latest = chain_repo.get_latest_snapshot_date(sym)
        has_chain = latest is not None
    except Exception:
        latest = None
        has_chain = False

    # Option contracts (enriched/saved)
    oc_repo = OptionContractsRepository(settings.paths.data_root)
    try:
        oc_tbl = oc_repo.load(symbol=sym)
        contracts_count = int(oc_tbl.num_rows)
    except Exception:
        contracts_count = 0

    # Option bars
    ob_repo = OptionBarRepository(settings.paths.data_root)
    try:
        if expiry:
            exp = pd.to_datetime(expiry).date()
            ob_tbl = ob_repo.load(underlying=sym, expiry=exp)
        else:
            ob_tbl = ob_repo.load(underlying=sym)
        bars_count = int(ob_tbl.num_rows)
    except Exception:
        bars_count = 0

    return {
        "symbol": sym,
        "has_chain_snapshot": has_chain,
        "latest_chain_as_of": latest.isoformat() if latest else None,
        "contracts_count": contracts_count,
        "bars_count": bars_count,
    }


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


@router.post("/fullflow-smart", response_model=OrchestratorResult)
def option_fullflow_smart(payload: FullFlowSmartRequest) -> OrchestratorResult:
    """
    Smart flow:
    - If option contracts already exist (and not forced), skip selection/enrich/save and run historical from repo.
    - Otherwise, run the full chain.
    Returns {task_id, plan} where plan describes which steps executed.
    """
    import pandas as pd
    from app_settings.settings import get_settings
    from common.repositories.option_contracts import OptionContractsRepository
    from ops.schemas.option_contracts import SelectedContracts
    from ops.tasks.option_bars import (
        get_option_contracts_for_bars_task,
        fetch_option_contract_details_task,
        save_selected_option_contracts_to_repo_task,
        fetch_option_historical_data_task,
    )

    settings = get_settings()
    base = settings.paths.data_root

    sym = payload.symbol.strip().upper()
    plan: list[str] = []

    # If not forced, check if we already have contracts saved
    have_contracts = False
    tbl = None
    try:
        oc_repo = OptionContractsRepository(base)
        if payload.expiry_filter:
            exp = pd.to_datetime(payload.expiry_filter).date()
            tbl = oc_repo.load(symbol=sym, expiry=exp)
        else:
            tbl = oc_repo.load(symbol=sym)
        have_contracts = tbl is not None and tbl.num_rows > 0
    except Exception:
        have_contracts = False

    # If contracts exist and user didn't force selection/enrich/save, run historical directly using repo contracts
    if have_contracts and not (payload.force_select or payload.force_enrich or payload.force_save):
        plan.append("historical_from_repo")
        # Build SelectedContracts from repo
        df = tbl.to_pandas()
        contracts = []
        for r in df.itertuples(index=False):
            try:
                expiry_raw = None
                if getattr(r, "expiry", None):
                    try:
                        expiry_raw = pd.to_datetime(r.expiry).strftime("%Y%m%d")
                    except Exception:
                        expiry_raw = None
                contracts.append({
                    "symbol": sym,
                    "secType": "OPT",
                    "exchange": getattr(r, "exchange", None) or "SMART",
                    "currency": getattr(r, "currency", None) or "USD",
                    "lastTradeDateOrContractMonth": expiry_raw,
                    "strike": float(getattr(r, "strike", None) or 0),
                    "right": str(getattr(r, "right", "")).upper(),
                    "multiplier": str(getattr(r, "multiplier", None) or 100),
                    "tradingClass": getattr(r, "trading_class", None) or sym,
                    "localSymbol": getattr(r, "local_symbol", None) or None,
                    "conId": int(getattr(r, "conid", 0) or 0) or None,
                })
            except Exception:
                continue
        sc = SelectedContracts(symbol=sym, contracts=contracts, contract_count=len(contracts))
        job = fetch_option_historical_data_task.s(
            contracts_result=sc.model_dump(),
            end_date_time=payload.end_date_time,
            duration_str=payload.duration_str,
            bar_size_setting=payload.bar_size_setting,
            what_to_show=payload.what_to_show,
            use_rth=payload.use_rth,
            timeout_seconds=getattr(payload, "timeout_seconds", 20.0),
        ).apply_async()
        return OrchestratorResult(
            processed=1,
            saved=0,
            skipped=0,
            details={"task_id": job.id, "symbol": sym, "plan": plan},
        )

    # Else fallback to full chain
    plan.append("full_chain")
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
    return OrchestratorResult(processed=1, saved=0, skipped=0, details={"task_id": res.id, "symbol": sym, "plan": plan})


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


@router.get("/expiries")
def list_expiries(
    symbol: str = Query(..., description="Underlying symbol"),
):
    """List known expiries for a symbol from option_contracts; fallback to latest chain snapshot."""
    import pandas as pd
    from app_settings.settings import get_settings
    from common.repositories.option_contracts import OptionContractsRepository
    from common.repositories.option_chains import OptionChainSnapshotRepository

    settings = get_settings()
    sym = symbol.strip().upper()

    expiries: list[str] = []

    # Try option_contracts first
    try:
        oc_repo = OptionContractsRepository(settings.paths.data_root)
        tbl = oc_repo.load(symbol=sym)
        if tbl and tbl.num_rows:
            df = tbl.select(["expiry"]).to_pandas()
            if not df.empty and "expiry" in df.columns:
                vals = pd.to_datetime(df["expiry"], errors="coerce").dt.date.dropna().unique().tolist()
                expiries = sorted({d.isoformat() for d in vals})
    except Exception:
        pass

    # Fallback to chain snapshot expirations
    if not expiries:
        try:
            ch_repo = OptionChainSnapshotRepository(settings.paths.data_root)
            latest = ch_repo.get_latest_snapshot_date(sym)
            if latest:
                tbl = ch_repo.load(filters=[("underlying", "==", sym), ("as_of", "==", latest)])
                if tbl and tbl.num_rows:
                    lsts = tbl.column("expirations").to_pylist()
                    all_exps = set()
                    for seq in lsts:
                        for v in (seq or []):
                            try:
                                all_exps.add(pd.to_datetime(str(v)).date().isoformat())
                            except Exception:
                                continue
                    expiries = sorted(all_exps)
        except Exception:
            pass

    return {"symbol": sym, "expiries": expiries}


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
