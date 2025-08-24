from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ..schemas.contracts import ContractDetailsBatch, OrchestratorResult, SymbolBatch
from ..service.ib_runtime import open_runtime


router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])
# Public (no prefix) mirror for convenience: /contracts/*
router_public = APIRouter(tags=["orchestrator"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/contracts/search", response_model=OrchestratorResult)
def search_contracts(payload: SymbolBatch) -> OrchestratorResult:
    """Search matching symbols via IB and snapshot contract descriptions.

    - Writes to `data/contract_descriptions` (vendor repo), append-only by day.
    - Filters by `sec_types`, `currency`, and `exchanges`.
    """
    from ibx_flows.symbol_search_ib import IBSymbolSearch
    from vendors.ibx_repos.contract_descriptions import ContractDescriptionsRepository
    from config import DATA_ROOT
    import pandas as pd

    symbols = sorted({s.strip().upper() for s in payload.symbols if s.strip()})
    if not symbols:
        raise HTTPException(status_code=400, detail="symbols must not be empty")

    repo = ContractDescriptionsRepository(DATA_ROOT / "contract_descriptions")
    rows: List[dict] = []
    with open_runtime() as rt:
        svc = IBSymbolSearch(rt)
        for sym in symbols:
            df = svc.search(sym)
            if df is None or df.empty:
                continue
            # Normalize and filter
            cols = {c.lower(): c for c in df.columns}
            def get(col):
                cl = col.lower(); return df[cols[cl]] if cl in cols else None
            out = pd.DataFrame({
                "symbol": get("symbol"),
                "conid": pd.to_numeric(get("conid"), errors="coerce"),
                "sec_type": (get("secType").astype(str) if get("secType") is not None else None),
                "currency": (get("currency").astype(str) if get("currency") is not None else None),
                "primary_exchange": (get("primaryExchange").astype(str) if get("primaryExchange") is not None else None),
                "derivative_sec_types": get("derivativeSecTypes"),
            })
            if not out.empty:
                if payload.sec_types:
                    out = out[out["sec_type"].isin([s.upper() for s in payload.sec_types])]
                if payload.currency:
                    out = out[out["currency"] == payload.currency]
                if payload.exchanges:
                    out = out[out["primary_exchange"].isin([e.upper() for e in payload.exchanges])]
                out = out.dropna(subset=["conid"]).drop_duplicates(subset=["conid"])
                if not out.empty:
                    rows.extend(out.to_dict(orient="records"))

    if not rows:
        return OrchestratorResult(processed=len(symbols), saved=0, skipped=0)

    repo.save(pd.DataFrame(rows))
    return OrchestratorResult(processed=len(symbols), saved=len(rows), skipped=0)


@router_public.post("/contracts/search", response_model=OrchestratorResult)
def search_contracts_public(
    payload: Optional[SymbolBatch] = None,
    symbols: Optional[str] = Query(None, description="Comma-separated symbols e.g. AAPL,MSFT"),
) -> OrchestratorResult:
    # Accept either JSON body or query-string fallback
    if payload is None:
        if not symbols:
            raise HTTPException(status_code=400, detail="Provide JSON body or ?symbols=")
        payload = SymbolBatch(symbols=[s.strip().upper() for s in symbols.split(",") if s.strip()])
    return search_contracts(payload)


@router.post("/contracts/details", response_model=OrchestratorResult)
def fetch_contract_details(payload: ContractDetailsBatch) -> OrchestratorResult:
    """Fetch contract details for provided `conids` or those discovered for `symbols`.

    - Reads from `contract_descriptions` snapshot to resolve `conids` if only `symbols` are given.
    - Writes detailed contracts to `data/contracts` repository (vendor repo).
    """
    from ibx_flows.contracts_backfill import _from_contract_details
    try:
        from ibx import ContractDetailsService  # type: ignore
        from ibx import make_stock  # type: ignore
    except Exception:  # pragma: no cover
        from ibx.services import ContractDetailsService  # type: ignore
        from ibx.contracts import make_stock  # type: ignore
    from vendors.ibx_repos.contracts import ContractRepository
    from vendors.ibx_repos.contract_descriptions import ContractDescriptionsRepository
    from config import DATA_ROOT
    import pandas as pd

    symbols = sorted({s.strip().upper() for s in (payload.symbols or []) if s.strip()})
    conids: List[int] = sorted({int(c) for c in (payload.conids or []) if c is not None})

    # Resolve conids via descriptions if symbols provided
    if symbols:
        desc_repo = ContractDescriptionsRepository(DATA_ROOT / "contract_descriptions")
        df = desc_repo.read(columns=["symbol", "conid"], symbol=symbols)
        if not df.empty:
            conids.extend([int(c) for c in df["conid"].dropna().tolist()])
    conids = sorted(set(conids))
    if not conids:
        raise HTTPException(status_code=400, detail="Provide symbols and/or conids to fetch details")

    repo = ContractRepository(DATA_ROOT / "contracts")
    existing = repo.present_conids()
    todo = [c for c in conids if c not in existing]

    saved = 0
    if not todo:
        return OrchestratorResult(processed=len(conids), saved=0, skipped=len(conids))

    with open_runtime() as rt:
        svc = ContractDetailsService(rt)
        rows: List[dict] = []
        for conid in todo:
            c = make_stock("?")
            c.conId = int(conid)
            try:
                cds = svc.fetch(c, timeout=payload.timeout_seconds or 20.0)
            except Exception:
                cds = None
            if not cds:
                continue
            row = _from_contract_details(cds[0])
            if row and row.get("conid"):
                rows.append(row)
        if rows:
            import pandas as pd
            repo.save(pd.DataFrame(rows))
            saved = len(rows)

    return OrchestratorResult(processed=len(conids), saved=saved, skipped=len(conids) - saved)


@router_public.post("/contracts/details", response_model=OrchestratorResult)
def fetch_contract_details_public(
    payload: Optional[ContractDetailsBatch] = None,
    symbols: Optional[str] = Query(None, description="Comma-separated symbols"),
    conids: Optional[str] = Query(None, description="Comma-separated conids"),
) -> OrchestratorResult:
    if payload is None:
        payload = ContractDetailsBatch()
        if symbols:
            payload.symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if conids:
            try:
                payload.conids = [int(x) for x in conids.split(",") if x.strip()]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid conids list")
        if not payload.symbols and not payload.conids:
            raise HTTPException(status_code=400, detail="Provide symbols and/or conids")
    return fetch_contract_details(payload)


@router.post("/contracts/setup", response_model=dict)
def setup_contracts(payload: SymbolBatch) -> dict:
    """High-level setup: search matching symbols then fetch details for discovered conids."""
    res_search = search_contracts(payload)
    # Try to fetch details for the just-processed symbols
    res_details = fetch_contract_details(ContractDetailsBatch(symbols=payload.symbols))
    return {
        "search": res_search.model_dump(),
        "details": res_details.model_dump(),
    }


@router_public.post("/contracts/setup", response_model=dict)
def setup_contracts_public(
    payload: Optional[SymbolBatch] = None,
    symbols: Optional[str] = Query(None, description="Comma-separated symbols"),
) -> dict:
    if payload is None:
        if not symbols:
            raise HTTPException(status_code=400, detail="Provide JSON body or ?symbols=")
        payload = SymbolBatch(symbols=[s.strip().upper() for s in symbols.split(",") if s.strip()])
    return setup_contracts(payload)
