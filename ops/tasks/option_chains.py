from __future__ import annotations

from app_settings.settings import get_settings
from celery import shared_task
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc


@shared_task(
    # bind=True,
    name="option_chains.check_has_options",
    queue="contracts",
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def check_has_options_task(
    # self,
    *,
    symbol: str,
    root: str | None = None,
) -> dict:
    """
    Check if a symbol has option contracts available.
    Uses the contracts repository to find option contracts.
    """
    import logging

    logger = logging.getLogger(__name__)

    settings = get_settings()
    base = Path(root) if root else settings.paths.data_root

    # Use common.repositories with Arrow table reads
    from common.repositories.contracts import ContractRepository

    repo = ContractRepository(base / "contracts")

    tbl = repo.read(
        columns=["conid", "symbol", "sec_type", "currency"],
        symbol=[symbol.upper()],
        sec_type=["OPT"],
    )

    if tbl is None or tbl.num_rows == 0:
        logger.info(f"No option contracts found for {symbol}")
        return {"symbol": symbol, "has_options": False, "option_count": 0}

    logger.info(f"Found {int(tbl.num_rows)} option contracts for {symbol}")

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
def get_underlying_contract_task(
    self,
    *,
    symbol: str,
    root: str | None = None,
) -> dict:
    """
    Get the underlying stock contract for a symbol.
    Reuses logic from historical data tasks.
    """
    from ops.tasks.historical_data import resolve_single_contract_v2_task

    # Reuse the existing contract resolution logic
    return resolve_single_contract_v2_task(self, symbol=symbol, root=root)


@shared_task(
    bind=True,
    name="option_chains.fetch_option_params",
    queue="option_chains",
    rate_limit="2/s",  # Rate limit IB API calls
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def fetch_option_params_task(
    self,
    contract_result: dict,
    api_base: Optional[str] = None,
) -> dict:
    """
    Fetch option chain parameters (secDefOptParams) from IB API.

    Args:
        contract_result: Result from get_underlying_contract_task
        api_base: IB API base URL override

    Returns:
        Dict with option parameters or error
    """
    import logging

    logger = logging.getLogger(__name__)

    # Check for error from previous task
    if "error" in contract_result:
        logger.info(
            f"⚠️ Skipping option params fetch due to contract error: {contract_result.get('error')}"
        )
        return contract_result

    settings = get_settings()
    base_url = api_base or str(settings.api.services.ib_api.url)

    symbol = contract_result.get("symbol", "N/A")
    conid = contract_result.get("conId")

    logger.info(
        f"📊 OPTION PARAMS: Fetching option parameters for {symbol} (conid: {conid})"
    )

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

            logger.info(
                f"📊 OPTION PARAMS: Got {len(params)} parameter sets for {symbol}"
            )

            return {
                "symbol": symbol,
                "contract": contract_result,
                "option_params": params,
                "param_count": len(params),
            }

    except Exception as e:
        logger.error(
            f"📊 OPTION PARAMS: Error fetching option parameters for {symbol}: {e}"
        )
        return {
            "symbol": symbol,
            "contract": contract_result,
            "option_params": [],
            "error": str(e),
        }


@shared_task(
    name="option_chains.build_chain_snapshot",
    queue="option_chains",
)
def build_chain_snapshot_task(
    params_result: dict,
    snapshot_date: str | None = None,
) -> dict:
    """
    Build option chain snapshot from option parameters.

    Args:
        params_result: Result from fetch_option_params_task
        snapshot_date: Date for snapshot (ISO format), defaults to today

    Returns:
        Dict with chain data or error
    """
    import logging

    logger = logging.getLogger(__name__)

    # Check for error from previous task
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
    if snapshot_date:
        snap_date = datetime.fromisoformat(snapshot_date).date()
    else:
        snap_date = date.today()

    logger.info(f"📊 CHAIN BUILD: Building option chain for {symbol} as of {snap_date}")

    rows = []

    for p in option_params:
        # Extract parameters (handle both dict and object formats)
        exchange = (
            p.get("exchange") if isinstance(p, dict) else getattr(p, "exchange", None)
        )
        trading_class = (
            p.get("tradingClass")
            if isinstance(p, dict)
            else getattr(p, "tradingClass", None)
        )
        multiplier = (
            p.get("multiplier")
            if isinstance(p, dict)
            else getattr(p, "multiplier", None)
        )
        expirations = (
            p.get("expirations", [])
            if isinstance(p, dict)
            else getattr(p, "expirations", [])
        )
        strikes = (
            p.get("strikes", []) if isinstance(p, dict) else getattr(p, "strikes", [])
        )

        logger.info(
            f"   Processing param set: {len(expirations)} expirations, {len(strikes)} strikes"
        )

        for exp in expirations:
            try:
                # Parse expiration date
                expiry_date = pd.to_datetime(str(exp), errors="coerce").date()
                if pd.isna(expiry_date):
                    continue

                # Skip expired options
                if expiry_date <= snap_date:
                    continue

            except Exception:
                continue

            # Build option contracts for each strike and right
            for right in ("C", "P"):  # Call and Put
                for strike in strikes:
                    try:
                        strike_price = float(strike)
                    except (ValueError, TypeError):
                        continue

                    rows.append(
                        {
                            "underlying": symbol,
                            "snapshot_date": snap_date,
                            "expiry": expiry_date,
                            "strike": strike_price,
                            "right": right,
                            "bid": None,
                            "ask": None,
                            "last": None,
                            "volume": None,
                            "open_interest": None,
                            "implied_volatility": None,
                            "delta": None,
                            "gamma": None,
                            "theta": None,
                            "vega": None,
                            "contract_id": None,
                            "local_symbol": None,
                            "trading_class": trading_class,
                            "exchange": exchange,
                            "currency": "USD",
                            "multiplier": (
                                int(multiplier)
                                if multiplier not in (None, "")
                                else None
                            ),
                        }
                    )

    logger.info(f"📊 CHAIN BUILD: Built {len(rows)} option chain rows for {symbol}")

    return {
        "symbol": symbol,
        "snapshot_date": snap_date.isoformat(),
        "chain_rows": rows,
        "rows_built": len(rows),
    }


@shared_task(
    name="option_chains.save_chain_snapshot",
    queue="option_chains",
)
def save_chain_snapshot_task(
    chain_result: dict,
    root: str | None = None,
) -> dict:
    """
    Save option chain snapshot to repository.

    Args:
        chain_result: Result from build_chain_snapshot_task
        root: Data root path override

    Returns:
        Dict with save results
    """
    import logging

    logger = logging.getLogger(__name__)

    # Check for error from previous task
    if "error" in chain_result:
        logger.info(
            f"⚠️ Skipping chain save due to build error: {chain_result.get('error')}"
        )
        return {
            "saved": 0,
            "error": chain_result.get("error"),
            "symbol": chain_result.get("symbol"),
        }

    settings = get_settings()
    base = Path(root) if root else settings.paths.data_root

    symbol = chain_result.get("symbol", "N/A")
    chain_rows = chain_result.get("chain_rows", [])
    snapshot_date_str = chain_result.get("snapshot_date")

    if not chain_rows:
        logger.warning(f"💾 No chain rows to save for {symbol}")
        return {"saved": 0, "symbol": symbol, "snapshot_date": snapshot_date_str}

    try:
        # Convert to DataFrame
        df = pd.DataFrame(chain_rows)

        if df.empty:
            logger.warning(f"💾 Empty DataFrame after conversion for {symbol}")
            return {"saved": 0, "symbol": symbol}

        # Parse snapshot date
        snapshot_date = datetime.fromisoformat(snapshot_date_str).date()

        # Create repository and save
        from common.repositories.option_chains import OptionChainSnapshotRepository

        repo = OptionChainSnapshotRepository(base)
        repo.save_chain_snapshot(df, underlying=symbol, snapshot_date=snapshot_date)

        logger.info(
            f"💾 Saved {len(df)} option chain rows for {symbol} as of {snapshot_date}"
        )

        return {
            "saved": len(df),
            "symbol": symbol,
            "snapshot_date": snapshot_date_str,
            "expiry_count": len(df["expiry"].unique()),
            "strike_range": [float(df["strike"].min()), float(df["strike"].max())],
        }

    except Exception as e:
        logger.error(f"💾 Error saving option chain for {symbol}: {e}")
        return {"saved": 0, "symbol": symbol, "error": str(e)}


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
    """
    Complete option chain processing workflow for a single symbol.

    This task:
    1. Checks if symbol has options
    2. Gets underlying contract
    3. Fetches option parameters
    4. Builds option chain snapshot
    5. Saves to repository

    Args:
        symbol: Stock symbol
        snapshot_date: Date for snapshot (ISO format), defaults to today
        root: Data root path override

    Returns:
        Dict with processing results
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info(f"🔄 STARTING: Option chain processing for {symbol}")

    # Step 1: Check if symbol has options
    logger.info(f"📍 STEP 1: Checking if {symbol} has options...")
    has_options = check_has_options_task(symbol=symbol, root=root)

    if not has_options.get("has_options", False):
        logger.info(f"⚠️ {symbol} has no option contracts available")
        return {
            "symbol": symbol,
            "processed": False,
            "saved": 0,
            "reason": "No options available",
        }

    logger.info(
        f"✅ {symbol} has {has_options.get('option_count', 0)} option contracts"
    )

    # Step 2: Get underlying contract
    logger.info(f"📍 STEP 2: Getting underlying contract for {symbol}...")
    contract_result = get_underlying_contract_task(self, symbol=symbol, root=root)

    if "error" in contract_result:
        logger.error(
            f"❌ Failed to get contract for {symbol}: {contract_result['error']}"
        )
        return {
            "symbol": symbol,
            "processed": False,
            "saved": 0,
            "error": contract_result["error"],
        }

    logger.info(f"✅ Got contract for {symbol} (conId: {contract_result.get('conId')})")

    # Step 3: Fetch option parameters
    logger.info(f"📍 STEP 3: Fetching option parameters for {symbol}...")
    params_result = fetch_option_params_task(self, contract_result=contract_result)

    if "error" in params_result:
        logger.error(
            f"❌ Failed to get option parameters for {symbol}: {params_result['error']}"
        )
        return {
            "symbol": symbol,
            "processed": False,
            "saved": 0,
            "error": params_result["error"],
        }

    param_count = params_result.get("param_count", 0)
    logger.info(f"✅ Got {param_count} option parameter sets for {symbol}")

    # Step 4: Build option chain snapshot
    logger.info(f"📍 STEP 4: Building option chain snapshot for {symbol}...")
    chain_result = build_chain_snapshot_task(
        params_result=params_result, snapshot_date=snapshot_date
    )

    rows_built = chain_result.get("rows_built", 0)
    logger.info(f"✅ Built {rows_built} option chain rows for {symbol}")

    # Step 5: Save to repository
    logger.info(f"📍 STEP 5: Saving option chain for {symbol}...")
    save_result = save_chain_snapshot_task(chain_result=chain_result, root=root)

    saved_count = save_result.get("saved", 0)
    if save_result.get("error"):
        logger.error(
            f"❌ Failed to save option chain for {symbol}: {save_result['error']}"
        )
        return {
            "symbol": symbol,
            "processed": True,
            "saved": 0,
            "error": save_result["error"],
        }

    logger.info(f"✅ Saved {saved_count} option chain rows for {symbol}")

    return {
        "symbol": symbol,
        "processed": True,
        "saved": saved_count,
        "snapshot_date": chain_result.get("snapshot_date"),
        "expiry_count": save_result.get("expiry_count", 0),
        "strike_range": save_result.get("strike_range", []),
    }


@shared_task(
    bind=True,
    name="option_chains.get_option_contracts_for_bars",
    queue="option_chains",
)
def get_option_contracts_for_bars_task(
    self,
    symbol: str,
    snapshot_date: str | None = None,
    expiry_filter: str | None = None,
    max_contracts: int = 50,
    root: str | None = None,
) -> dict:
    """
    Get option contracts for historical bars collection.
    Loads from option_chains repository and formats for historical data tasks.

    Args:
        symbol: Stock symbol
        snapshot_date: Snapshot date to load (ISO format), defaults to latest
        expiry_filter: Optional expiry date filter (ISO format)
        max_contracts: Maximum number of contracts to return
        root: Data root path override

    Returns:
        Dict with option contracts formatted for historical data tasks
    """
    import logging

    logger = logging.getLogger(__name__)

    settings = get_settings()
    base = Path(root) if root else settings.paths.data_root

    logger.info(f"📊 CONTRACT LOOKUP: Getting option contracts for {symbol}")

    try:
        from common.repositories.option_chains import OptionChainSnapshotRepository

        repo = OptionChainSnapshotRepository(base)

        # Parse dates
        if snapshot_date:
            snap_date = datetime.fromisoformat(snapshot_date).date()
            tbl = repo.load(underlying=symbol, snapshot_date=snap_date)
        else:
            # Load all data for symbol and get latest snapshot
            tbl = repo.load(underlying=symbol)
            if tbl.num_rows > 0:
                latest_scalar = pc.max(tbl.column("snapshot_date"))
                latest_date = latest_scalar.as_py()
                mask = pc.equal(tbl.column("snapshot_date"), pa.scalar(latest_date, type=tbl.schema.field("snapshot_date").type))
                idx = pc.indices_nonzero(mask)
                tbl = tbl.take(idx)
                logger.info(f"📊 Using latest snapshot date: {latest_date}")

        if tbl.num_rows == 0:
            logger.warning(f"📊 No option chain data found for {symbol}")
            return {"symbol": symbol, "contracts": [], "contract_count": 0}

        # Apply expiry filter if specified
        if expiry_filter:
            expiry_date = datetime.fromisoformat(expiry_filter).date()
            mask = pc.equal(tbl.column("expiry"), pa.scalar(expiry_date, type=tbl.schema.field("expiry").type))
            tbl = tbl.take(pc.indices_nonzero(mask))
            logger.info(f"📊 Filtered to expiry {expiry_date}: {int(tbl.num_rows)} contracts")

        # Limit number of contracts
        if tbl.num_rows > max_contracts:
            # Prefer ATM options - sort by distance from median strike
            med = pc.median(tbl.column("strike")).as_py()
            dist = pc.abs(pc.subtract(tbl.column("strike"), pa.scalar(med)))
            # Append a temp column for sorting
            tbl = tbl.append_column("_dist", dist)
            order = pc.sort_indices(tbl, sort_keys=[("_dist", "ascending")])
            tbl = tbl.take(order).slice(0, max_contracts)
            # Drop temp column
            keep_cols = [n for n in tbl.column_names if n != "_dist"]
            tbl = tbl.select(keep_cols)
            logger.info(f"📊 Limited to {max_contracts} contracts closest to strike {med}")

        # Format contracts for IB API (similar to stock contracts)
        contracts: List[Dict[str, Any]] = []
        cols = {name: tbl.column(name) for name in tbl.column_names}
        n = tbl.num_rows
        for i in range(n):
            expiry_val = cols["expiry"][i].as_py()
            expiry_str = pd.Timestamp(expiry_val).strftime("%Y%m%d")
            right_val = cols["right"][i].as_py()
            strike_val = float(cols["strike"][i].as_py())
            exchange_val = cols.get("exchange").to_pylist()[i] if "exchange" in cols else None
            currency_val = cols.get("currency").to_pylist()[i] if "currency" in cols else None
            multiplier_val = cols.get("multiplier").to_pylist()[i] if "multiplier" in cols else None
            trading_class_val = cols.get("trading_class").to_pylist()[i] if "trading_class" in cols else None
            contract_id_val = cols.get("contract_id").to_pylist()[i] if "contract_id" in cols else None

            contract: Dict[str, Any] = {
                "symbol": symbol,
                "secType": "OPT",
                "exchange": exchange_val or "SMART",
                "currency": currency_val or "USD",
                "lastTradeDateOrContractMonth": expiry_str,
                "strike": strike_val,
                "right": right_val,
                "multiplier": str(multiplier_val or 100),
                "tradingClass": trading_class_val or symbol,
            }
            if contract_id_val is not None:
                try:
                    contract["conId"] = int(contract_id_val)
                except Exception:
                    pass
            contracts.append(contract)

        logger.info(
            f"📊 CONTRACT LOOKUP: Found {len(contracts)} option contracts for {symbol}"
        )

        # Derive snapshot_date from the first row if available
        snap_date_val = tbl.column("snapshot_date")[0].as_py() if tbl.num_rows else None

        return {
            "symbol": symbol,
            "contracts": contracts,
            "contract_count": len(contracts),
            "snapshot_date": snap_date_val.isoformat() if snap_date_val else None,
        }

    except Exception as e:
        logger.error(
            f"📊 CONTRACT LOOKUP: Error getting option contracts for {symbol}: {e}"
        )
        return {"symbol": symbol, "contracts": [], "contract_count": 0, "error": str(e)}


@shared_task(
    bind=True,
    name="option_chains.fetch_option_historical_data",
    queue="option_chains",
)
def fetch_option_historical_data_task(
    self,
    contracts_result: dict,
    end_date_time: str,
    duration_str: str,
    bar_size_setting: str,
    what_to_show: str = "TRADES",
    use_rth: int = 1,
    timeout_seconds: float = 20.0,
    root: str | None = None,
) -> dict:
    """
    Fetch historical data for option contracts and save to option_bars repository.
    Reuses existing historical data tasks with option repository type.

    Args:
        contracts_result: Result from get_option_contracts_for_bars_task
        end_date_time: End date/time for historical data
        duration_str: Duration string
        bar_size_setting: Bar size setting
        what_to_show: Data type to retrieve
        use_rth: Regular trading hours only flag
        timeout_seconds: Request timeout
        root: Data root path override

    Returns:
        Dict with processing results
    """
    import logging

    logger = logging.getLogger(__name__)

    # Check for error from previous task
    if "error" in contracts_result:
        logger.info(
            f"⚠️ Skipping option historical data due to contracts error: {contracts_result.get('error')}"
        )
        return contracts_result

    symbol = contracts_result.get("symbol", "N/A")
    contracts = contracts_result.get("contracts", [])

    if not contracts:
        logger.warning(
            f"📊 No option contracts available for historical data: {symbol}"
        )
        return {
            "symbol": symbol,
            "processed": 0,
            "saved": 0,
            "failed": 0,
            "results": [],
        }

    logger.info(
        f"📊 OPTION BARS: Fetching historical data for {len(contracts)} option contracts of {symbol}"
    )

    # Reuse existing historical data processing, but with repository_type="option"
    from ops.tasks.historical_data import fetch_historical_data_from_ib_task

    # Format as resolve_result for compatibility
    resolve_result = {"resolved_contracts": contracts, "missing_symbols": []}

    # Call existing historical data task with option repository type
    result = fetch_historical_data_from_ib_task(
        self,
        resolve_result=resolve_result,
        end_date_time=end_date_time,
        duration_str=duration_str,
        bar_size_setting=bar_size_setting,
        what_to_show=what_to_show,
        use_rth=use_rth,
        repository_type="option",  # Key difference: use option repository
        timeout_seconds=timeout_seconds,
        root=root,
    )

    logger.info(
        f"📊 OPTION BARS: Historical data processing complete for {symbol}: {result}"
    )

    return result


@shared_task(
    bind=True,
    name="option_chains.process_historical_option_data",
    queue="option_chains",
)
def process_historical_option_data_task(
    self,
    symbol: str,
    end_date_time: str,
    duration_str: str,
    bar_size_setting: str,
    what_to_show: str = "TRADES",
    use_rth: int = 1,
    snapshot_date: str | None = None,
    expiry_filter: str | None = None,
    max_contracts: int = 50,
    timeout_seconds: float = 20.0,
    root: str | None = None,
) -> dict:
    """
    Complete historical option data workflow for a symbol.

    This task:
    1. Gets option contracts from option_chains repository
    2. Fetches historical bars for option contracts
    3. Saves to option_bars repository

    Args:
        symbol: Stock symbol
        end_date_time: End date/time for historical data
        duration_str: Duration string
        bar_size_setting: Bar size setting
        what_to_show: Data type to retrieve
        use_rth: Regular trading hours only flag
        snapshot_date: Snapshot date for option contracts (ISO format)
        expiry_filter: Optional expiry date filter (ISO format)
        max_contracts: Maximum number of contracts to process
        timeout_seconds: Request timeout
        root: Data root path override

    Returns:
        Dict with processing results
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info(f"🔄 STARTING: Historical option data processing for {symbol}")

    # Step 1: Get option contracts
    logger.info(f"📍 STEP 1: Getting option contracts for {symbol}...")
    contracts_result = get_option_contracts_for_bars_task(
        self,
        symbol=symbol,
        snapshot_date=snapshot_date,
        expiry_filter=expiry_filter,
        max_contracts=max_contracts,
        root=root,
    )

    if "error" in contracts_result:
        logger.error(
            f"❌ Failed to get option contracts for {symbol}: {contracts_result['error']}"
        )
        return {
            "symbol": symbol,
            "processed": 0,
            "saved": 0,
            "error": contracts_result["error"],
        }

    contract_count = contracts_result.get("contract_count", 0)
    logger.info(f"✅ Got {contract_count} option contracts for {symbol}")

    if contract_count == 0:
        logger.info(f"⚠️ No option contracts available for {symbol}")
        return {
            "symbol": symbol,
            "processed": 0,
            "saved": 0,
            "reason": "No option contracts available",
        }

    # Step 2: Fetch historical data
    logger.info(
        f"📍 STEP 2: Fetching historical data for {contract_count} option contracts..."
    )
    result = fetch_option_historical_data_task(
        self,
        contracts_result=contracts_result,
        end_date_time=end_date_time,
        duration_str=duration_str,
        bar_size_setting=bar_size_setting,
        what_to_show=what_to_show,
        use_rth=use_rth,
        timeout_seconds=timeout_seconds,
        root=root,
    )

    processed_count = result.get("processed", 0)
    saved_count = result.get("saved", 0)
    failed_count = result.get("failed", 0)

    logger.info(
        f"✅ Historical option data processing complete for {symbol}: {processed_count} processed, {saved_count} bars saved, {failed_count} failed"
    )

    return result


@shared_task(
    bind=True,
    name="option_chains.process_for_symbols",
    queue="option_chains",
)
def process_option_chains_for_symbols_task(
    self,
    symbols: List[str],
    snapshot_date: str | None = None,
    root: str | None = None,
) -> dict:
    """
    Process option chains for multiple symbols.

    Args:
        symbols: List of stock symbols
        snapshot_date: Date for snapshot (ISO format), defaults to today
        root: Data root path override

    Returns:
        Dict with overall processing results
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info(
        f"🔄 STARTING: Option chain processing for {len(symbols)} symbols: {symbols}"
    )

    processed_count = 0
    saved_count = 0
    failed_count = 0
    results = []

    for i, symbol in enumerate(symbols):
        logger.info(f"📍 {i+1}/{len(symbols)}: Processing option chains for {symbol}")

        try:
            result = process_option_chain_for_symbol_task(
                self, symbol=symbol, snapshot_date=snapshot_date, root=root
            )

            if result.get("processed", False):
                processed_count += 1
                saved_count += result.get("saved", 0)

                if result.get("error"):
                    failed_count += 1
                    logger.error(f"   ❌ Error processing {symbol}: {result['error']}")
                else:
                    logger.info(
                        f"   ✅ Processed {symbol}: {result.get('saved', 0)} rows saved"
                    )
            else:
                logger.info(
                    f"   ⚠️ Skipped {symbol}: {result.get('reason', 'Unknown reason')}"
                )

            results.append(result)

        except Exception as e:
            logger.error(f"   ❌ Exception processing {symbol}: {e}")
            failed_count += 1
            results.append(
                {"symbol": symbol, "processed": False, "saved": 0, "error": str(e)}
            )

    logger.info(
        f"Option chain processing complete: {processed_count} processed, {saved_count} rows saved, {failed_count} failed"
    )

    return {
        "processed": processed_count,
        "saved": saved_count,
        "failed": failed_count,
        "total_symbols": len(symbols),
        "results": results,
    }
