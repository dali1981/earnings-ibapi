from __future__ import annotations

from ..schemas.contracts import (
    ContractDetailsBatch,
    HistoricalDataBySymbolsRequest,
    HistoricalDataRequest,
    OptionChainRequest,
    OptionHistoricalDataRequest,
    OrchestratorResult,
    SymbolBatch,
)
from app_settings.settings import get_settings
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from celery import group
from ops.tasks.option_chains import (
    check_has_options_task,
    get_underlying_contract_task,
    fetch_option_params_task,
    build_chain_snapshot_task,
    save_chain_snapshot_task,
)
import uuid


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
    from ops.tasks.search import (
        fetch_matching_symbols_task,
        save_contract_descriptions_task,
    )

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
    symbols: Optional[str] = Query(
        None, description="Comma-separated symbols e.g. AAPL,MSFT"
    ),
) -> OrchestratorResult:
    # Accept either JSON body or query-string fallback
    if payload is None:
        if not symbols:
            raise HTTPException(
                status_code=400, detail="Provide JSON body or ?symbols="
            )
        payload = SymbolBatch(
            symbols=[s.strip().upper() for s in symbols.split(",") if s.strip()]
        )
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
    from celery import chain
    from ops.tasks.contract_details import (
        check_existing_details_task,
        fetch_details_from_ib_task,
        save_details_to_repo_task,
    )

    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"fetch_contract_details called with payload: {payload}")

    # Extract and normalize symbols
    symbols = sorted({s.strip().upper() for s in (payload.symbols or []) if s.strip()})
    logger.info(f"Normalized symbols: {symbols}")

    # Handle conids if provided (convert to symbols first)
    if payload.conids and not symbols:
        # For backward compatibility, if only conids provided, we need to resolve them
        from vendors.ibx_repos.contract_descriptions import (
            ContractDescriptionsRepository,
        )

        settings = get_settings()
        conids = sorted({int(c) for c in (payload.conids or []) if c is not None})
        if conids:
            desc_repo = ContractDescriptionsRepository(
                settings.paths.data_root / "contract_descriptions"
            )
            df = desc_repo.read(columns=["symbol", "conid"])
            if not df.empty:
                df_filtered = df[df["conid"].isin(conids)]
                if not df_filtered.empty:
                    symbols = sorted(
                        {s for s in df_filtered["symbol"].dropna().unique()}
                    )

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
            payload.symbols = [
                s.strip().upper() for s in symbols.split(",") if s.strip()
            ]
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
            raise HTTPException(
                status_code=400, detail="Provide JSON body or ?symbols="
            )
        payload = SymbolBatch(
            symbols=[s.strip().upper() for s in symbols.split(",") if s.strip()]
        )
    return setup_contracts(payload)


@router.post("/historical/bars", response_model=OrchestratorResult)
def fetch_historical_bars(payload: HistoricalDataRequest) -> OrchestratorResult:
    """Asynchronously fetch historical bars for provided contracts.

    - Takes a list of IB contract specifications
    - Fetches historical bars from IB API (rate limited at 2/sec)
    - Writes results to equity_bars or option_bars repository
    - Uses Celery tasks: fetch bars -> save to repo
    """
    from celery import chain, group
    from ops.tasks.historical_data import (
        fetch_historical_bars_task,
        save_historical_bars_task,
    )

    import logging

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
def fetch_historical_bars_by_symbols(
    payload: HistoricalDataBySymbolsRequest,
) -> OrchestratorResult:
    """Asynchronously fetch historical bars for provided symbols.

    - Takes a list of ticker symbols
    - For each symbol, creates a chain: resolve contract -> fetch bars -> save
    - All chains run in parallel as a group
    - Uses the new v2 tasks following the search.py pattern
    """
    from celery import chain, group
    from ops.tasks.historical_data import (
        fetch_bars_for_contract_v2_task,
        resolve_single_contract_v2_task,
        save_bars_v2_task,
    )

    import logging

    logger = logging.getLogger(__name__)

    symbols = sorted({s.strip().upper() for s in payload.symbols if s.strip()})
    if not symbols:
        raise HTTPException(status_code=400, detail="symbols must not be empty")

    logger.info(f"fetch_historical_bars_by_symbols called with symbols: {symbols}")

    # Create a chain for each symbol: resolve -> fetch -> save
    chains = []
    for sym in symbols:
        # Step 1: Resolve symbol to contract
        resolve_sig = resolve_single_contract_v2_task.s(symbol=sym)

        # Step 2: Fetch historical bars (contract comes from previous task)
        fetch_sig = fetch_bars_for_contract_v2_task.s(
            payload.end_date_time,
            payload.duration_str,
            payload.bar_size_setting,
            payload.what_to_show,
            payload.use_rth,
            payload.timeout_seconds,
            None,  # api_base
        )

        # Step 3: Save the fetched bars (fetch_result comes from previous task)
        save_sig = save_bars_v2_task.s(
            None,  # repository (will be created in task)
            payload.repository_type,
            None,  # root
        )

        # Chain them together
        chains.append(chain(resolve_sig, fetch_sig, save_sig))

    # Execute all chains as a group
    job = group(chains)
    result = job.apply_async()

    return OrchestratorResult(
        processed=len(symbols),
        saved=0,  # Will be updated by async tasks
        skipped=0,
        details={
            "dispatched": len(chains),
            "group_id": result.id,
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
def fetch_historical_bars_by_symbols_public(
    payload: HistoricalDataBySymbolsRequest,
) -> OrchestratorResult:
    """Public endpoint for historical bars by symbols."""
    return fetch_historical_bars_by_symbols(payload)


@router.post("/option-chains/snapshots", response_model=OrchestratorResult)
def fetch_option_chain_snapshots(payload: OptionChainRequest) -> OrchestratorResult:
    """Asynchronously fetch option chain snapshots for provided symbols.

    - Takes a list of ticker symbols
    - For each symbol, creates a task to collect option chain data
    - Checks if symbol has options, gets contracts, fetches parameters, saves to option_chains repository
    - Uses Celery tasks with rate limiting for IB API calls
    """
    import logging

    logger = logging.getLogger(__name__)

    symbols = sorted({s.strip().upper() for s in payload.symbols if s.strip()})
    if not symbols:
        raise HTTPException(status_code=400, detail="symbols must not be empty")

    logger.info(f"fetch_option_chain_snapshots called with symbols: {symbols}")

    # correlation id for tracing (generate if not provided)
    cid = payload.correlation_id or str(uuid.uuid4())

    # Create pipelines for each symbol using Celery canvas (inlined)
    pipelines = []
    for sym in symbols:
        pipeline = (
            check_has_options_task.si(symbol=sym, root=None, correlation_id=cid)
            | get_underlying_contract_task.si(symbol=sym, root=None, correlation_id=cid)
            | fetch_option_params_task.s(correlation_id=cid)
            | build_chain_snapshot_task.s(payload.snapshot_date)
            | save_chain_snapshot_task.s(root=None)
        )
        pipelines.append(pipeline)

    # Execute all pipelines as a group
    job = group(pipelines)
    result = job.apply_async()

    return OrchestratorResult(
        processed=len(symbols),
        saved=0,  # Will be updated by async tasks
        skipped=0,
        details={
            "dispatched": len(pipelines),
            "group_id": result.id,
            "symbols": symbols,
            "snapshot_date": payload.snapshot_date,
            "correlation_id": cid,
        },
    )


@router.post("/option-chains/historical", response_model=OrchestratorResult)
def fetch_option_historical_data(
    payload: OptionHistoricalDataRequest,
) -> OrchestratorResult:
    """Asynchronously fetch historical data for option contracts.

    - Takes a list of ticker symbols
    - For each symbol, gets option contracts from option_chains repository
    - Fetches historical bars for option contracts (rate limited)
    - Saves to option_bars repository
    """
    import logging

    logger = logging.getLogger(__name__)

    symbols = sorted({s.strip().upper() for s in payload.symbols if s.strip()})
    if not symbols:
        raise HTTPException(status_code=400, detail="symbols must not be empty")

    logger.info(f"fetch_option_historical_data called with symbols: {symbols}")

    # Create tasks for each symbol
    tasks = []
    for sym in symbols:
        req = OptionHistoricalArgs(
            symbol=sym,
            correlation_id=payload.correlation_id,
            end_date_time=payload.end_date_time,
            duration_str=payload.duration_str,
            bar_size_setting=payload.bar_size_setting,
            what_to_show=payload.what_to_show,
            use_rth=payload.use_rth,
            snapshot_date=payload.snapshot_date,
            expiry_filter=payload.expiry_filter,
            max_contracts=payload.max_contracts,
            timeout_seconds=payload.timeout_seconds,
        )
        task = process_historical_option_data_task.s(request=req.model_dump())
        tasks.append(task)

    # Execute all tasks as a group
    job = group(tasks)
    result = job.apply_async()

    return OrchestratorResult(
        processed=len(symbols),
        saved=0,  # Will be updated by async tasks
        skipped=0,
        details={
            "dispatched": len(tasks),
            "group_id": result.id,
            "symbols": symbols,
            "bar_size": payload.bar_size_setting,
            "duration": payload.duration_str,
            "max_contracts": payload.max_contracts,
        },
    )


@router_public.post("/option-chains/snapshots", response_model=OrchestratorResult)
def fetch_option_chain_snapshots_public(
    payload: Optional[OptionChainRequest] = None,
    symbols: Optional[str] = Query(None, description="Comma-separated symbols"),
    snapshot_date: Optional[str] = Query(
        None, description="Snapshot date (ISO format)"
    ),
) -> OrchestratorResult:
    """Public endpoint for option chain snapshots."""
    if payload is None:
        if not symbols:
            raise HTTPException(
                status_code=400, detail="Provide JSON body or ?symbols="
            )
        payload = OptionChainRequest(
            symbols=[s.strip().upper() for s in symbols.split(",") if s.strip()],
            snapshot_date=snapshot_date,
        )
    return fetch_option_chain_snapshots(payload)


@router_public.post("/option-chains/historical", response_model=OrchestratorResult)
def fetch_option_historical_data_public(
    payload: OptionHistoricalDataRequest,
) -> OrchestratorResult:
    """Public endpoint for option historical data."""
    return fetch_option_historical_data(payload)
