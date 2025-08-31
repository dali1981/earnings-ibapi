from __future__ import annotations

from app_settings.settings import get_settings
from celery import shared_task
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import logging
from common.utils.logging_extra import (
    set_correlation_id,
    CorrelationIdAdapter,
)

# Re-export option bars tasks from the new module to preserve import paths
from ops.tasks.option_bars import (
    get_option_contracts_for_bars_task,
    fetch_option_historical_data_task,
    process_historical_option_data_task,
)


@shared_task(
    # bind=True,
    name="option_chains.check_has_options",
    queue="contracts",
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def check_has_options_task(*, symbol: str, root: str | None = None, correlation_id: str | None = None) -> dict:
    """Check if a symbol has option contracts available using repo-only data."""
    import logging

    set_correlation_id(correlation_id)
    logger = CorrelationIdAdapter(logging.getLogger(__name__), {"cid": correlation_id})
    settings = get_settings()
    base = Path(root) if root else settings.paths.data_root

    # Use contract descriptions to check derivative types include OPT
    from common.repositories.contract_descriptions import (
        ContractDescriptionsRepository,
    )

    repo = ContractDescriptionsRepository(base)
    filters = [
        ("symbol", "==", symbol.upper()),
        ("derivative_sec_types", "contains", "OPT"),
    ]
    tbl = repo.load(
        columns=["conid", "symbol", "sec_type", "currency", "derivative_sec_types"],
        filters=filters,
    )

    if tbl is None or tbl.num_rows == 0:
        from celery.exceptions import Ignore
        logger.info(f"No option contracts found for {symbol}")
        raise Ignore(f"no_options:{symbol}")

    logger.info(f"Found {int(tbl.num_rows)} option-capable contracts for {symbol}")

    # Build a small sample without converting full table to pandas
    def _table_to_records(t: pa.Table, limit: int = 3) -> List[Dict[str, Any]]:
        t = t.slice(0, limit)
        cols = t.column_names
        arrays = [t.column(i) for i in range(len(cols))]
        out: List[Dict[str, Any]] = []
        for i in range(t.num_rows):
            row = {name: arr[i].as_py() for name, arr in zip(cols, arrays)}
            out.append(row)
        return out

    return {
        "symbol": symbol,
        "has_options": True,
        "option_count": int(tbl.num_rows),
        "sample_contracts": _table_to_records(tbl, 3),
    }


@shared_task(
    bind=True,
    name="option_chains.get_underlying_contract",
    queue="contracts",
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def get_underlying_contract_task(self, *, symbol: str, root: str | None = None, correlation_id: str | None = None) -> dict:
    """Get the underlying stock contract for a symbol from repository only."""
    set_correlation_id(correlation_id)
    logger = CorrelationIdAdapter(logging.getLogger(__name__), {"cid": correlation_id})
    settings = get_settings()
    base_path = Path(root) if root else settings.paths.data_root
    from common.repositories.contracts import ContractRepository
    repo = ContractRepository(base_path)
    filter_ = ("symbol", "in", symbol.upper())
    tbl = repo.load(columns=["conid", "symbol", "sec_type", "currency"], filters=[filter_])
    if tbl is None or tbl.num_rows == 0:
        logger.info(f"No contracts found for {symbol}")
        return {"error": f"No contracts found for {symbol}", "symbol": symbol, "correlation_id": correlation_id}
    cols = {name: tbl.column(name) for name in tbl.column_names}
    i = 0
    try:
        conid = int(cols["conid"][i].as_py())
    except Exception:
        return {"error": f"Invalid contract id for {symbol}", "symbol": symbol, "correlation_id": correlation_id}
    sym = str(cols["symbol"][i].as_py() or "").upper()
    sec_type = cols["sec_type"][i].as_py() or "STK"
    currency = cols["currency"][i].as_py() or "USD"
    logger.info(f"Resolved underlying for {symbol} conId={conid}")
    return {"conId": conid, "symbol": sym, "secType": sec_type, "currency": currency, "exchange": "SMART", "correlation_id": correlation_id}


@shared_task(
    bind=True,
    name="option_chains.fetch_option_params",
    queue="option_chains",
    rate_limit="2/s",
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def fetch_option_params_task(
    self, contract_result: dict, api_base: Optional[str] = None, correlation_id: str | None = None
) -> dict:
    """Fetch option chain parameters (secDefOptParams) from IB API."""
    import logging

    set_correlation_id(correlation_id)
    logger = CorrelationIdAdapter(logging.getLogger(__name__), {"cid": correlation_id})
    if "error" in contract_result:
        from celery.exceptions import Ignore
        logger.info(
            f"⚠️ Skipping option params fetch due to contract error: {contract_result.get('error')}"
        )
        raise Ignore(contract_result.get("error"))

    settings = get_settings()
    base_url = api_base or str(settings.api.services.ib_api.url)
    symbol = contract_result.get("symbol", "N/A")
    conid = contract_result.get("conId")

    logger.info(f"📊 OPTION PARAMS: Fetching option parameters for {symbol} (conid: {conid})")

    payload = {
        "symbol": symbol,
        "conid": conid,
        "sec_type": "STK",
        "exchange": "",
        "timeout": 20.0,
    }

    try:
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            resp = client.post("/ib/secdef/option-params", json=payload)
            resp.raise_for_status()
            params = resp.json()
            logger.info(f"📊 OPTION PARAMS: Got {len(params)} parameter sets for {symbol}")
            if not params:
                from celery.exceptions import Ignore
                raise Ignore("no_params")
            return {
                "symbol": symbol,
                "contract": contract_result,
                "option_params": params,
                "param_count": len(params),
                "correlation_id": correlation_id,
            }
    except Exception as e:
        logger.error(
            f"📊 OPTION PARAMS: Error fetching option parameters for {symbol}: {e}"
        )
        from celery.exceptions import Ignore
        raise Ignore(str(e))


@shared_task(
    name="option_chains.build_chain_snapshot",
    queue="option_chains",
)
def build_chain_snapshot_task(
    params_result: dict, snapshot_date: str | None = None
) -> dict:
    """Build aggregated option chain snapshot rows (lists of expirations/strikes)."""
    import logging

    set_correlation_id(params_result.get("correlation_id"))
    logger = CorrelationIdAdapter(logging.getLogger(__name__), {"cid": params_result.get("correlation_id")})

    if "error" in params_result:
        logger.info(
            f"⚠️ Skipping chain building due to params error: {params_result.get('error')}"
        )
        return params_result

    symbol = params_result.get("symbol", "N/A")
    option_params = params_result.get("option_params", [])
    if not option_params:
        logger.warning(f"📊 CHAIN BUILD: No option parameters available for {symbol}")
        return {"symbol": symbol, "chain_rows": [], "rows_built": 0}

    # Parse snapshot date
    snap_date = (
        datetime.fromisoformat(snapshot_date).date() if snapshot_date else date.today()
    )

    logger.info(f"📊 CHAIN BUILD: Building option chain for {symbol} as of {snap_date}")

    rows: List[Dict[str, Any]] = []
    for p in option_params:
        exchange = (p.get("exchange") if isinstance(p, dict) else getattr(p, "exchange", None))
        trading_class = (p.get("tradingClass") if isinstance(p, dict) else getattr(p, "tradingClass", None))
        multiplier = (p.get("multiplier") if isinstance(p, dict) else getattr(p, "multiplier", None))
        expirations = (p.get("expirations", []) if isinstance(p, dict) else getattr(p, "expirations", []))
        strikes = (p.get("strikes", []) if isinstance(p, dict) else getattr(p, "strikes", []))

        exps = []
        for e in expirations or []:
            try:
                exps.append(pd.to_datetime(str(e)).date().isoformat())
            except Exception:
                continue
        s_list = []
        for v in strikes or []:
            try:
                s_list.append(float(v))
            except Exception:
                continue

        rows.append({
            "underlying": symbol,
            "underlying_conid": params_result.get("contract", {}).get("conId"),
            "exchange": exchange or "SMART",
            "trading_class": trading_class or symbol,
            "multiplier": str(multiplier) if multiplier not in (None, "") else None,
            "expirations": sorted(set(exps)),
            "strikes": sorted(set(s_list)),
            "as_of": snap_date,
        })

    logger.info(f"📊 CHAIN BUILD: Built {len(rows)} parameter sets for {symbol}")
    return {
        "symbol": symbol,
        "as_of": snap_date.isoformat(),
        "chain_rows": rows,
        "rows_built": len(rows),
        "correlation_id": params_result.get("correlation_id"),
    }


@shared_task(name="option_chains.ensure_has_options", queue="contracts")
def ensure_has_options_task(check_result: dict, symbol: str, root: str | None = None) -> dict:
    """Gate the workflow: raise Ignore to stop if no options available."""
    from celery.exceptions import Ignore
    if not check_result or not check_result.get("has_options"):
        raise Ignore(f"No options available for {symbol}")
    return {"symbol": symbol, "root": root}

@shared_task(
    name="option_chains.save_chain_snapshot",
    queue="option_chains",
)
def save_chain_snapshot_task(chain_result: dict, root: str | None = None) -> dict:
    """Save option chain snapshot to repository."""
    import logging

    set_correlation_id(chain_result.get("correlation_id"))
    logger = CorrelationIdAdapter(logging.getLogger(__name__), {"cid": chain_result.get("correlation_id")})

    if "error" in chain_result:
        return {
            "saved": 0,
            "error": chain_result.get("error"),
            "symbol": chain_result.get("symbol"),
        }

    settings = get_settings()
    base = Path(root) if root else settings.paths.data_root
    symbol = chain_result.get("symbol", "N/A")
    chain_rows = chain_result.get("chain_rows", [])
    as_of_str = chain_result.get("as_of")

    if not chain_rows:
        logger.warning(f"💾 No chain rows to save for {symbol}")
        return {"saved": 0, "symbol": symbol, "as_of": as_of_str, "correlation_id": chain_result.get("correlation_id")}

    try:
        df = pd.DataFrame(chain_rows)
        if df.empty:
            return {"saved": 0, "symbol": symbol}
        as_of = datetime.fromisoformat(as_of_str).date()

        from common.repositories.option_chains import OptionChainSnapshotRepository

        repo = OptionChainSnapshotRepository(base)
        repo.save(df, underlying=symbol, as_of=as_of)

        
        # Summaries across all parameter sets
        all_exp = sorted(set(e for exps in df["expirations"] for e in (exps or [])))
        all_str = sorted(set(float(v) for arr in df["strikes"] for v in (arr or [])))
        return {
            "saved": len(df),
            "symbol": symbol,
            "as_of": as_of_str,
            "expiry_count": len(all_exp),
            "strike_range": [all_str[0] if all_str else None, all_str[-1] if all_str else None],
            "correlation_id": chain_result.get("correlation_id"),
        }
    except Exception as e:
        logger.error(f"💾 Error saving option chain for {symbol}: {e}")
        return {"saved": 0, "symbol": symbol, "error": str(e), "correlation_id": chain_result.get("correlation_id")}


@shared_task(name="option_chains.ensure_params_ok", queue="option_chains")
def ensure_params_ok_task(params_result: dict, correlation_id: str | None = None) -> dict:
    """Gate the chain after fetch_option_params: stop if error or empty params.

    Raises Ignore to prevent build/save from running on failure.
    """
    from celery.exceptions import Ignore
    set_correlation_id(correlation_id or params_result.get("correlation_id"))
    logger = CorrelationIdAdapter(logging.getLogger(__name__), {"cid": get_correlation_id() if 'get_correlation_id' in globals() else correlation_id})
    if params_result.get("error") or not params_result.get("option_params"):
        logger.info("No option params available; stopping chain")
        raise Ignore(params_result.get("error") or "no_params")
    return params_result


@shared_task(
    # bind=True,
    name="option_chains.process_for_symbol",
    queue="option_chains",
)
def process_option_chain_for_symbol_task(
    # self,
    symbol: str,
    snapshot_date: str | None = None,
    root: str | None = None,
) -> dict:
    """Dispatch a Celery chain for option chain processing (no inline task.run)."""
    from celery import chain

    # Build pipeline: check -> ensure -> resolve -> secdef -> build -> save
    pipeline = (
        check_has_options_task.si(symbol=symbol, root=root)
        | ensure_has_options_task.s(symbol=symbol, root=root)
        | get_underlying_contract_task.si(symbol=symbol, root=root)
        | fetch_option_params_task.s()
        | build_chain_snapshot_task.s(snapshot_date)
        | save_chain_snapshot_task.s(root=root)
    )
    result = pipeline.apply_async()
    return {"symbol": symbol, "dispatched": True, "chain_id": result.id}


@shared_task(
    bind=True,
    name="option_chains.process_for_symbols",
    queue="option_chains",
)
def process_option_chains_for_symbols_task(
    self, symbols: List[str], snapshot_date: str | None = None, root: str | None = None
) -> dict:
    """Process option chains for multiple symbols."""
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"🔄 STARTING: Option chain processing for {len(symbols)} symbols: {symbols}"
    )

    processed_count = 0
    saved_count = 0
    failed_count = 0
    results: List[dict] = []

    for i, sym in enumerate(symbols):
        logger.info(f"📍 {i + 1}/{len(symbols)}: Processing option chains for {sym}")
        try:
            result = process_option_chain_for_symbol_task(
                symbol=sym, snapshot_date=snapshot_date, root=root
            )
            if result.get("processed", False):
                processed_count += 1
                saved_count += result.get("saved", 0)
                if result.get("error"):
                    failed_count += 1
            results.append(result)
        except Exception as e:
            logger.error(f"   ❌ Exception processing {sym}: {e}")
            failed_count += 1
            results.append(
                {"symbol": sym, "processed": False, "saved": 0, "error": str(e)}
            )

    return {
        "processed": processed_count,
        "saved": saved_count,
        "failed": failed_count,
        "total_symbols": len(symbols),
        "results": results,
    }
