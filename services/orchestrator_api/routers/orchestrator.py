from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..schemas.contracts import (
    ContractDetailsBatch, 
    OrchestratorResult, 
    SymbolBatch,
    HistoricalDataRequest,
    HistoricalDataBySymbolsRequest,
)
from app_settings.settings import get_settings


router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])
# Public (no prefix) mirror for convenience: /contracts/*
router_public = APIRouter(tags=["orchestrator"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/contracts/search", response_model=OrchestratorResult)
def search_contracts(payload: SymbolBatch) -> OrchestratorResult:
    """Asynchronously search matching symbols via IB API and save results.

    - Dispatches a Celery chain per symbol: fetch -> save (save is idempotent).
    - Rate limited to 10 requests/sec via Celery task settings.
    - Returns an orchestrator result with dispatch details.
    """
    from celery import chain, group
    from ibx_celery.app import app as celery_app
    from ops.tasks.search import fetch_matching_symbols_task, save_contract_descriptions_task

    symbols = sorted({s.strip().upper() for s in payload.symbols if s.strip()})
    if not symbols:
        raise HTTPException(status_code=400, detail="symbols must not be empty")

    chains = []
    for sym in symbols:
        sig_fetch = fetch_matching_symbols_task.s(
            pattern=sym,
            sec_types=payload.sec_types,
            currency=payload.currency,
            exchanges=payload.exchanges,
        )
        sig_save = save_contract_descriptions_task.s()
        chains.append(chain(sig_fetch, sig_save))

    job = group(chains)
    result = job.apply_async()
    return OrchestratorResult(
        processed=len(symbols),
        saved=0,
        skipped=0,
        details={"dispatched": len(chains), "group_id": result.id},
    )


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
    """Asynchronously fetch contract details for provided symbols.

    - Takes a list of ticker symbols
    - Checks if tickers are in contract repository
    - If not, sends contract details request to IB API
    - Writes results to contracts repository
    - Uses Celery chain: check DB -> fetch from IB -> write to DB
    """
    import logging
    from celery import chain
    from ops.tasks.contract_details import (
        check_existing_details_task,
        fetch_details_from_ib_task,
        save_details_to_repo_task,
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"fetch_contract_details called with payload: {payload}")

    # Extract and normalize symbols
    symbols = sorted({s.strip().upper() for s in (payload.symbols or []) if s.strip()})
    logger.info(f"Normalized symbols: {symbols}")
    
    # Handle conids if provided (convert to symbols first)
    if payload.conids and not symbols:
        # For backward compatibility, if only conids provided, we need to resolve them
        from vendors.ibx_repos.contract_descriptions import ContractDescriptionsRepository
        
        settings = get_settings()
        conids = sorted({int(c) for c in (payload.conids or []) if c is not None})
        if conids:
            desc_repo = ContractDescriptionsRepository(settings.paths.data_root / "contract_descriptions")
            df = desc_repo.read(columns=["symbol", "conid"])
            if not df.empty:
                df_filtered = df[df["conid"].isin(conids)]
                if not df_filtered.empty:
                    symbols = sorted({s for s in df_filtered["symbol"].dropna().unique()})
    
    if not symbols:
        raise HTTPException(status_code=400, detail="Provide symbols to fetch details")

    # Create Celery chain for processing
    # 1. Check which symbols already have details
    # 2. Fetch missing details from IB API
    # 3. Save fetched details to repository
    task_chain = chain(
        check_existing_details_task.s(symbols=symbols),
        fetch_details_from_ib_task.s(
            api_base=None,  # Will use settings default
            timeout_seconds=payload.timeout_seconds or 20,
        ),
        save_details_to_repo_task.s(),
    )
    
    # Execute the chain asynchronously
    result = task_chain.apply_async()
    
    return OrchestratorResult(
        processed=len(symbols),
        saved=0,  # Will be updated by async task
        skipped=0,  # Will be updated by async task
        details={
            "dispatched": len(symbols),
            "chain_id": result.id,
            "symbols": symbols,
        },
    )


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


@router.post("/historical/bars", response_model=OrchestratorResult)
def fetch_historical_bars(payload: HistoricalDataRequest) -> OrchestratorResult:
    """Asynchronously fetch historical bars for provided contracts.
    
    - Takes a list of IB contract specifications
    - Fetches historical bars from IB API (rate limited at 2/sec)
    - Writes results to equity_bars or option_bars repository
    - Uses Celery tasks: fetch bars -> save to repo
    """
    import logging
    from celery import chain, group
    from ops.tasks.historical_data import (
        fetch_historical_bars_task,
        save_historical_bars_task,
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"fetch_historical_bars called with {len(payload.contracts)} contracts")
    
    if not payload.contracts:
        raise HTTPException(status_code=400, detail="contracts must not be empty")
    
    # Create Celery chains for each contract
    chains = []
    for contract in payload.contracts:
        # Chain: fetch historical bars -> save to repository
        fetch_task = fetch_historical_bars_task.s(
            contract=contract,
            end_date_time=payload.end_date_time,
            duration_str=payload.duration_str,
            bar_size_setting=payload.bar_size_setting,
            what_to_show=payload.what_to_show,
            use_rth=payload.use_rth,
            timeout_seconds=payload.timeout_seconds,
        )
        save_task = save_historical_bars_task.s(
            repository_type=payload.repository_type,
        )
        chains.append(chain(fetch_task, save_task))
    
    # Execute all chains as a group
    job = group(chains)
    result = job.apply_async()
    
    return OrchestratorResult(
        processed=len(payload.contracts),
        saved=0,  # Will be updated by async tasks
        skipped=0,
        details={
            "dispatched": len(chains),
            "group_id": result.id,
            "contracts": [c.get("symbol", "N/A") for c in payload.contracts],
            "bar_size": payload.bar_size_setting,
            "duration": payload.duration_str,
        },
    )


@router.post("/historical/bars-by-symbols", response_model=OrchestratorResult)
def fetch_historical_bars_by_symbols(payload: HistoricalDataBySymbolsRequest) -> OrchestratorResult:
    """Asynchronously fetch historical bars for provided symbols.
    
    - Takes a list of ticker symbols
    - Resolves contracts from repositories
    - Fetches historical bars from IB API (rate limited at 2/sec)  
    - Writes results to equity_bars or option_bars repository
    - Uses Celery chain: resolve contracts -> fetch bars -> save to repo
    """
    import logging
    from ops.tasks.historical_data import resolve_contracts_for_symbols_task
    
    logger = logging.getLogger(__name__)
    
    symbols = sorted({s.strip().upper() for s in payload.symbols if s.strip()})
    if not symbols:
        raise HTTPException(status_code=400, detail="symbols must not be empty")
    
    logger.info(f"fetch_historical_bars_by_symbols called with symbols: {symbols}")
    
    # Create a custom chain that resolves contracts, then processes each one
    from ops.tasks.historical_data import (
        fetch_historical_bars_task,
        save_historical_bars_task,
        process_historical_data_for_resolved_contracts_task
    )
    
    # Chain: resolve contracts -> process each resolved contract
    result = process_historical_data_for_resolved_contracts_task.apply_async(kwargs={
        "symbols": symbols,
        "end_date_time": payload.end_date_time,
        "duration_str": payload.duration_str,
        "bar_size_setting": payload.bar_size_setting,
        "what_to_show": payload.what_to_show,
        "use_rth": payload.use_rth,
        "repository_type": payload.repository_type,
        "timeout_seconds": payload.timeout_seconds,
    })
    
    return OrchestratorResult(
        processed=len(symbols),
        saved=0,  # Will be updated by async tasks
        skipped=0,
        details={
            "dispatched": len(symbols),
            "chain_id": result.id,
            "symbols": symbols,
            "bar_size": payload.bar_size_setting,
            "duration": payload.duration_str,
        },
    )


@router_public.post("/historical/bars", response_model=OrchestratorResult)
def fetch_historical_bars_public(payload: HistoricalDataRequest) -> OrchestratorResult:
    """Public endpoint for historical bars by contracts."""
    return fetch_historical_bars(payload)


@router_public.post("/historical/bars-by-symbols", response_model=OrchestratorResult)
def fetch_historical_bars_by_symbols_public(payload: HistoricalDataBySymbolsRequest) -> OrchestratorResult:
    """Public endpoint for historical bars by symbols."""
    return fetch_historical_bars_by_symbols(payload)
