from __future__ import annotations

from app_settings.settings import get_settings
from celery import shared_task
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx
import pandas as pd


@shared_task(
    bind=True,
    name="historical.fetch_bars_from_ib",
    queue="historical",
    rate_limit="2/s",  # IB API rate limit for historical data
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def fetch_historical_bars_task(
    self,
    contract: Dict[str, Any],
    end_date_time: str,
    duration_str: str,
    bar_size_setting: str,
    what_to_show: str = "TRADES",
    use_rth: int = 1,
    format_date: int = 1,
    keep_up_to_date: bool = False,
    timeout_seconds: int = 20,
    api_base: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch historical bars from IB API for a contract.

    Args:
        contract: Contract specification (symbol, secType, exchange, currency, etc.)
        end_date_time: End date/time for historical data (e.g., "20240823 23:59:59")
        duration_str: Duration string (e.g., "1 D", "1 M", "1 Y")
        bar_size_setting: Bar size (e.g., "1 min", "5 mins", "1 hour", "1 day")
        what_to_show: Data type ("TRADES", "MIDPOINT", "BID", "ASK")
        use_rth: 1 for regular trading hours only, 0 for all hours
        format_date: Date format (1 for yyyyMMdd HH:mm:ss, 2 for seconds since epoch)
        keep_up_to_date: Whether to keep data updated
        timeout_seconds: Request timeout
        api_base: IB API base URL override

    Returns:
        Dict with 'bars' list and metadata
    """
    import logging

    logger = logging.getLogger(__name__)

    settings = get_settings()
    base_url = api_base or str(settings.api.services.ib_api.url)

    logger.info(
        f"📡 FETCH: Fetching historical bars for contract: {contract.get('symbol', 'N/A')} (conid: {contract.get('conId')})"
    )
    logger.info(f"📡 URL: {base_url}/ib/historical/bars")

    payload = {
        "contract": contract,
        "endDateTime": end_date_time,
        "durationStr": duration_str,
        "barSizeSetting": bar_size_setting,
        "whatToShow": what_to_show,
        "useRTH": use_rth,
        "formatDate": format_date,
        "keepUpToDate": keep_up_to_date,
        "timeout": timeout_seconds,
    }

    logger.info(f"📡 PAYLOAD: {payload}")

    try:
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            logger.info(f"📡 Making POST request to {base_url}/ib/historical/bars...")
            resp = client.post("/ib/historical/bars", json=payload)
            logger.info(f"📡 Response status: {resp.status_code}")

            resp.raise_for_status()
            bars = resp.json()

            logger.info(
                f"📡 SUCCESS: Fetched {len(bars)} bars for {contract.get('symbol', 'N/A')}"
            )
            if bars:
                logger.info(f"📡 Sample bar: {bars[0] if bars else 'None'}")

            return {
                "bars": bars,
                "contract": contract,
                "parameters": {
                    "end_date_time": end_date_time,
                    "duration_str": duration_str,
                    "bar_size_setting": bar_size_setting,
                    "what_to_show": what_to_show,
                },
            }
    except Exception as e:
        logger.error(
            f"📡 ERROR: Error fetching historical bars for {contract.get('symbol', 'N/A')}: {e}"
        )
        return {
            "bars": [],
            "contract": contract,
            "error": str(e),
            "parameters": {
                "end_date_time": end_date_time,
                "duration_str": duration_str,
                "bar_size_setting": bar_size_setting,
                "what_to_show": what_to_show,
            },
        }


@shared_task(
    bind=True,
    name="historical.save_bars_to_repo",
    queue="historical",
)
def save_historical_bars_task(
    self,
    fetch_result: Dict[str, Any],
    repository_type: str = "equity",  # "equity" or "option"
    root: Optional[str] = None,
) -> Dict[str, int]:
    """Save fetched historical bars to the appropriate repository."""
    import logging

    logger = logging.getLogger(__name__)

    settings = get_settings()
    base = Path(root) if root else settings.paths.data_root

    bars = fetch_result.get("bars", [])
    contract = fetch_result.get("contract", {})
    parameters = fetch_result.get("parameters", {})
    symbol = contract.get("symbol", "N/A")

    if not bars:
        logger.warning(f"💾 No bars to save for {symbol}")
        return {"saved_count": 0, "error": fetch_result.get("error")}

    try:
        df = _normalize_historical_bars(bars, contract, parameters)

        if df.empty:
            logger.warning(f"💾 No valid bars after normalization for {symbol}")
            return {"saved_count": 0}

        # Choose repository
        if repository_type == "equity":
            from common.repositories.equity_bars import EquityBarRepository

            repo = EquityBarRepository(base)
        elif repository_type == "option":
            from common.repositories.option_bars import OptionBarRepository

            repo = OptionBarRepository(base)
        else:
            raise ValueError(f"Unknown repository type: {repository_type}")

        repo.save(df)
        logger.info(f"💾 Saved {len(df)} bars for {symbol}")

        return {
            "saved_count": len(df),
            "symbol": symbol,
            "bar_size": parameters.get("bar_size_setting"),
            "date_range": {
                "start": df["time"].min().isoformat() if not df.empty else None,
                "end": df["time"].max().isoformat() if not df.empty else None,
            },
        }

    except Exception as e:
        logger.error(f"💾 Error saving bars for {symbol}: {e}")
        return {"saved_count": 0, "error": str(e)}


@shared_task(
    bind=True,
    name="historical.resolve_contract_for_symbol",
    queue="contracts",
)
def resolve_contract_for_symbol_task(
    self,
    *,
    symbol: str,
    sec_types: Optional[Sequence[str]] = None,
    currency: Optional[str] = None,
    exchanges: Optional[Sequence[str]] = None,
    api_base: Optional[str] = None,
) -> Dict[str, List]:
    """
    Resolve contract *only* from the contracts repository (ignore contract_descriptions).

    - Pushdown filtering by symbol to avoid full scans.
    - Returns all matching contracts per requested symbol.

    Returns:
        {
          "resolved_contracts": List[dict],   # IB-like dicts
          "missing_symbols":   List[str],     # requested symbols not present in contracts
        }
    """
    import logging

    logger = logging.getLogger(__name__)

    # Normalize input
    if not symbol:
        return {"resolved_contracts": [], "missing_symbols": []}

    sym = symbol.strip().upper()

    # Settings & repo
    from app_settings.settings import get_settings

    settings = get_settings()
    base = settings.paths.data_root

    try:
        from vendors.ibx_repos.contracts import ContractRepository
    except ImportError:
        from ibx_repos.contracts import ContractRepository  # type: ignore

    contracts_repo = ContractRepository(base / "contracts")

    # Columns we need to build IB contract dicts
    wanted_cols = ["conid", "symbol", "sec_type", "currency"]

    # ---- Pushdown filter on symbol (no full load) ----
    try:
        contracts_df = contracts_repo.read(columns=wanted_cols, symbol=[sym])
    except Exception:
        # Fallback if .read() with filters isn't supported—last resort
        contracts_df = contracts_repo.load()
        if not contracts_df.empty:
            for c in wanted_cols:
                if c not in contracts_df.columns:
                    contracts_df[c] = ""
            contracts_df = contracts_df[wanted_cols]
            contracts_df["symbol"] = contracts_df["symbol"].astype(str).str.upper()
            contracts_df = contracts_df[contracts_df["symbol"] == sym]

    if contracts_df.empty:
        logger.info(f"No contracts found for symbol {sym}")
        return {"resolved_contracts": [], "missing_symbols": [sym]}

    # Normalize
    contracts_df["symbol"] = contracts_df["symbol"].astype(str).str.upper()
    contracts_df["conid"] = (
        pd.to_numeric(contracts_df["conid"], errors="coerce").dropna().astype("int64")
    )
    for c in ("sec_type", "currency", "primary_exchange"):
        if c not in contracts_df.columns:
            contracts_df[c] = ""
        else:
            contracts_df[c] = contracts_df[c].fillna("").astype(str)

    # Build output (include all matches)
    resolved_contracts = [
        {
            "conId": int(r.conid),
            "symbol": r.symbol,
            "secType": r.sec_type,
            "currency": r.currency,
            "exchange": "SMART",  # default routing
        }
        for r in contracts_df.itertuples(index=False)
    ]

    found_symbols = set(contracts_df["symbol"].unique())
    missing_symbols = [sym] if sym not in found_symbols else []

    return {
        "resolved_contracts": resolved_contracts,
        "missing_symbols": missing_symbols,
    }


@shared_task(
    bind=True,
    name="historical.fetch_historical_data_from_ib",
    queue="historical",
)
def fetch_historical_data_from_ib_task(
    self,
    resolve_result: Dict[str, List],
    end_date_time: str,
    duration_str: str,
    bar_size_setting: str,
    what_to_show: str = "TRADES",
    use_rth: int = 1,
    repository_type: str = "equity",
    timeout_seconds: float = 20.0,
    root: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch historical data from IB API for resolved contracts.


    Args:
        resolve_result: use resolve_results["resolved_contracts"] list
        end_date_time: End date/time for historical data
        duration_str: Duration string
        bar_size_setting: Bar size setting
        what_to_show: Data type to retrieve
        use_rth: Regular trading hours only flag
        repository_type: Repository type for saving
        timeout_seconds: Request timeout
        root: Base path override

    Returns:
        Dict with processing results
    """

    import logging

    logger = logging.getLogger(__name__)

    contracts = resolve_result.get("resolved_contracts", [])
    missing_symbols = resolve_result.get("missing_symbols", [])

    logger.info(
        f"🔄 STARTING: Processing historical data for symbols: {[c.get('symbol') for c in contracts]}"
    )
    logger.info(
        f"📋 PARAMS: end_date={end_date_time}, duration={duration_str}, bar_size={bar_size_setting}, repository={repository_type}"
    )

    if resolve_result.get("error"):
        logger.error(f"❌ ERROR in contract resolution: {resolve_result['error']}")
        return {
            "processed": 0,
            "saved": 0,
            "failed": 100,
            "error": resolve_result["error"],
        }

    if not contracts:
        logger.warning(
            f"⚠️  NO CONTRACTS resolved for symbols, missing: {missing_symbols}"
        )
        return {
            "processed": 0,
            "saved": 0,
            "failed": 0,
            "missing_symbols": missing_symbols,
        }

    processed_count = 0
    saved_count = 0
    failed_count = 0
    results = []

    for i, contract in enumerate(contracts):
        contract_symbol = contract.get("symbol", "N/A")
        logger.info(f"📍 {i + 1}: Processing contract {contract_symbol}")

        try:
            # Fetch historical bars
            logger.info(f"   🔄 Fetching historical bars for {contract_symbol}...")
            fetch_result = fetch_historical_bars_task(
                self,
                contract=contract,
                end_date_time=end_date_time,
                duration_str=duration_str,
                bar_size_setting=bar_size_setting,
                what_to_show=what_to_show,
                use_rth=use_rth,
                timeout_seconds=timeout_seconds,
            )

            logger.info(
                f"   📊 Fetch result for {contract_symbol}: {len(fetch_result.get('bars', []))} bars, error: {fetch_result.get('error', 'None')}"
            )

            if fetch_result.get("error"):
                logger.error(
                    f"   ❌ Failed to fetch bars for {contract_symbol}: {fetch_result['error']}"
                )
                failed_count += 1
                results.append(
                    {
                        "symbol": contract_symbol,
                        "saved_count": 0,
                        "fetch_error": fetch_result["error"],
                    }
                )
                continue

            # Save bars to repository
            logger.info(
                f"   💾 Saving {len(fetch_result.get('bars', []))} bars for {contract_symbol} to {repository_type} repository..."
            )
            save_result = save_historical_bars_task(
                self,
                fetch_result=fetch_result,
                repository_type=repository_type,
                root=root,
            )

            logger.info(f"   💾 Save result for {contract_symbol}: {save_result}")

            processed_count += 1
            if save_result.get("saved_count", 0) > 0:
                saved_count += save_result["saved_count"]
                logger.info(
                    f"   ✅ Saved {save_result['saved_count']} bars for {contract_symbol}"
                )
            else:
                logger.warning(
                    f"   ⚠️  No bars saved for {contract_symbol}: {save_result}"
                )

            results.append(
                {
                    "symbol": contract_symbol,
                    "saved_count": save_result.get("saved_count", 0),
                    "error": save_result.get("error"),
                }
            )

        except Exception as e:
            logger.error(f"   ❌ Exception processing contract {contract_symbol}: {e}")
            failed_count += 1
            results.append(
                {"symbol": contract_symbol, "saved_count": 0, "error": str(e)}
            )

    logger.info(
        f"Historical data processing complete: {processed_count} processed, {saved_count} bars saved, {failed_count} failed"
    )

    return {
        "processed": processed_count,
        "saved": saved_count,
        "failed": failed_count,
        "missing_symbols": missing_symbols,
        "results": results,
    }


def _normalize_historical_bars(
    bars: List[Dict[str, Any]], contract: Dict[str, Any], parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Normalize historical bars data into repository format."""
    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)

    # Ensure required columns exist
    required_cols = ["time", "open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            if col == "time":
                df[col] = pd.NaT
            elif col == "volume":
                df[col] = 0
            else:
                df[col] = 0.0

    # Add contract and parameter information
    df["symbol"] = contract.get("symbol", "")
    df["bar_size"] = parameters.get("bar_size_setting", "")
    df["data_type"] = parameters.get("what_to_show", "TRADES").lower()

    # Convert time to datetime if it's a string
    if not df.empty and not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    # Add trade_date for partitioning (date part of time)
    if not df.empty:
        df["trade_date"] = df["time"].dt.date

    # Optional columns that may be present
    optional_cols = ["wap", "bar_count"]
    for col in optional_cols:
        if col not in df.columns:
            df[col] = None

    return df


@shared_task(
    bind=True,
    name="historical.process_for_resolved_contracts",
    queue="historical",
)
def process_historical_data_for_resolved_contracts_task(
    self,
    symbols: List[str],
    end_date_time: str,
    duration_str: str,
    bar_size_setting: str,
    what_to_show: str = "TRADES",
    use_rth: int = 1,
    repository_type: str = "equity",
    timeout_seconds: float = 20.0,
    root: Optional[str] = None,
) -> Dict[str, Any]:
    """Process historical data for resolved contracts.

    This task:
    1. Resolves contracts for symbols
    2. For each resolved contract, fetches historical bars
    3. Saves bars to repository

    Args:
        symbols: List of ticker symbols
        end_date_time: End date/time for historical data
        duration_str: Duration string
        bar_size_setting: Bar size setting
        what_to_show: Data type to retrieve
        use_rth: Regular trading hours only flag
        repository_type: Repository type for saving
        timeout_seconds: Request timeout
        root: Base path override

    Returns:
        Dict with processing results
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info(f"🔄 STARTING: Processing historical data for symbols: {symbols}")
    logger.info(
        f"📋 PARAMS: end_date={end_date_time}, duration={duration_str}, bar_size={bar_size_setting}, repository={repository_type}"
    )

    # Step 1: Resolve contracts for symbols
    logger.info(f"📍 STEP 1: Resolving contracts for symbols: {symbols}")
    # For now, resolve each symbol individually since we don't have a batch resolver
    resolved_contracts = []
    missing_symbols = []

    for sym in symbols:
        result = resolve_contract_for_symbol_task(self, symbol=sym)
        if result.get("resolved_contracts"):
            resolved_contracts.extend(result["resolved_contracts"])
        if result.get("missing_symbols"):
            missing_symbols.extend(result["missing_symbols"])

    resolve_result = {
        "resolved_contracts": resolved_contracts,
        "missing_symbols": missing_symbols,
    }
    logger.info(f"✅ RESOLVED: {resolve_result}")

    resolved_contracts = resolve_result.get("resolved_contracts", [])
    missing_symbols = resolve_result.get("missing_symbols", [])

    if resolve_result.get("error"):
        logger.error(f"❌ ERROR in contract resolution: {resolve_result['error']}")
        return {
            "processed": 0,
            "saved": 0,
            "failed": len(symbols),
            "error": resolve_result["error"],
        }

    if not resolved_contracts:
        logger.warning(
            f"⚠️  NO CONTRACTS resolved for symbols: {symbols}, missing: {missing_symbols}"
        )
        return {
            "processed": len(symbols),
            "saved": 0,
            "failed": 0,
            "missing_symbols": missing_symbols,
        }

    logger.info(
        f"📍 STEP 2: Resolved {len(resolved_contracts)} contracts, processing historical data..."
    )
    for i, contract in enumerate(resolved_contracts):
        logger.info(
            f"   Contract {i+1}: {contract.get('symbol')} (conid: {contract.get('conId')})"
        )

    # Step 2: Process each resolved contract
    processed_count = 0
    saved_count = 0
    failed_count = 0
    results = []

    for i, contract in enumerate(resolved_contracts):
        contract_symbol = contract.get("symbol", "N/A")
        logger.info(f"📍 STEP 3.{i+1}: Processing contract {contract_symbol}")

        try:
            # Fetch historical bars
            logger.info(f"   🔄 Fetching historical bars for {contract_symbol}...")
            fetch_result = fetch_historical_bars_task(
                self,
                contract=contract,
                end_date_time=end_date_time,
                duration_str=duration_str,
                bar_size_setting=bar_size_setting,
                what_to_show=what_to_show,
                use_rth=use_rth,
                timeout_seconds=timeout_seconds,
            )

            logger.info(
                f"   📊 Fetch result for {contract_symbol}: {len(fetch_result.get('bars', []))} bars, error: {fetch_result.get('error', 'None')}"
            )

            if fetch_result.get("error"):
                logger.error(
                    f"   ❌ Failed to fetch bars for {contract_symbol}: {fetch_result['error']}"
                )
                failed_count += 1
                results.append(
                    {
                        "symbol": contract_symbol,
                        "saved_count": 0,
                        "fetch_error": fetch_result["error"],
                    }
                )
                continue

            # Save bars to repository
            logger.info(
                f"   💾 Saving {len(fetch_result.get('bars', []))} bars for {contract_symbol} to {repository_type} repository..."
            )
            save_result = save_historical_bars_task(
                self,
                fetch_result=fetch_result,
                repository_type=repository_type,
                root=root,
            )

            logger.info(f"   💾 Save result for {contract_symbol}: {save_result}")

            processed_count += 1
            if save_result.get("saved_count", 0) > 0:
                saved_count += save_result["saved_count"]
                logger.info(
                    f"   ✅ Saved {save_result['saved_count']} bars for {contract_symbol}"
                )
            else:
                logger.warning(
                    f"   ⚠️  No bars saved for {contract_symbol}: {save_result}"
                )

            results.append(
                {
                    "symbol": contract_symbol,
                    "saved_count": save_result.get("saved_count", 0),
                    "error": save_result.get("error"),
                }
            )

        except Exception as e:
            logger.error(f"   ❌ Exception processing contract {contract_symbol}: {e}")
            failed_count += 1
            results.append(
                {"symbol": contract_symbol, "saved_count": 0, "error": str(e)}
            )

    logger.info(
        f"Historical data processing complete: {processed_count} processed, {saved_count} bars saved, {failed_count} failed"
    )

    return {
        "processed": processed_count,
        "saved": saved_count,
        "failed": failed_count,
        "missing_symbols": missing_symbols,
        "results": results,
    }


# === New v2 tasks following search.py pattern ===
@shared_task(
    bind=True,
    name="historical.resolve_single_contract_v2",
    queue="contracts",
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def resolve_single_contract_v2_task(
    self,
    *,
    symbol: str,
    root: str | None = None,
) -> dict:
    """
    Resolve a single symbol to its best contract.
    Returns a single contract dict suitable for fetch_historical_bars_task.
    """
    from app_settings.settings import get_settings

    settings = get_settings()
    base = Path(root) if root else settings.paths.data_root

    from common.repositories.contracts import ContractRepository
    import pyarrow as pa
    import pyarrow.compute as pc

    repo = ContractRepository(base / "contracts")
    tbl = repo.read(
        columns=["conid", "symbol", "sec_type", "currency"], symbol=[symbol.upper()]
    )

    if tbl is None or tbl.num_rows == 0:
        return {"error": f"No contracts found for {symbol}", "symbol": symbol}

    # Take the first contract (they should all be for the same symbol)
    cols = {name: tbl.column(name) for name in tbl.column_names}
    i = 0
    conid = int(cols["conid"][i].as_py())
    sym = str(cols["symbol"][i].as_py() or "").upper()
    sec_type = cols["sec_type"][i].as_py() or "STK"
    currency = cols["currency"][i].as_py() or "USD"

    return {
        "conId": conid,
        "symbol": sym,
        "secType": sec_type,
        "currency": currency,
        "exchange": "SMART",
    }


@shared_task(
    # bind=True,
    name="historical.fetch_bars_for_contract_v2",
    queue="historical",
    rate_limit="2/s",
    max_retries=3,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def fetch_bars_for_contract_v2_task(
    contract: dict,
    end_date_time: str,
    duration_str: str,
    bar_size_setting: str,
    what_to_show: str = "TRADES",
    use_rth: int = 1,
    timeout_seconds: float = 20.0,
    api_base: str | None = None,
) -> dict:
    """
    Fetch historical bars for a single contract.
    Input comes from resolve_single_contract_v2_task.
    Output goes to save_bars_v2_task.
    """
    import httpx
    import logging

    logger = logging.getLogger(__name__)

    # Check for error from previous task
    if "error" in contract:
        logger.info(
            f"⚠️ Skipping fetch for contract with error: {contract.get('error')}"
        )
        return contract

    settings = get_settings()
    base_url = api_base or str(settings.api.services.ib_api.url)

    logger.info(
        f"📡 FETCH v2: Fetching historical bars for {contract.get('symbol', 'N/A')} (conid: {contract.get('conId')})"
    )

    payload = {
        "contract": contract,
        "endDateTime": end_date_time,
        "durationStr": duration_str,
        "barSizeSetting": bar_size_setting,
        "whatToShow": what_to_show,
        "useRTH": use_rth,
        "formatDate": 1,
        "keepUpToDate": False,
        "timeout": int(timeout_seconds),
    }

    try:
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            resp = client.post("/ib/historical/bars", json=payload)
            resp.raise_for_status()
            bars = resp.json()

            logger.info(
                f"📡 FETCH v2: Got {len(bars)} bars for {contract.get('symbol', 'N/A')}"
            )

            return {
                "bars": bars,
                "contract": contract,
                "parameters": {
                    "end_date_time": end_date_time,
                    "duration_str": duration_str,
                    "bar_size_setting": bar_size_setting,
                    "what_to_show": what_to_show,
                },
            }
    except Exception as e:
        logger.error(
            f"📡 FETCH v2: Error fetching bars for {contract.get('symbol', 'N/A')}: {e}"
        )
        return {
            "bars": [],
            "contract": contract,
            "error": str(e),
            "parameters": {
                "end_date_time": end_date_time,
                "duration_str": duration_str,
                "bar_size_setting": bar_size_setting,
                "what_to_show": what_to_show,
            },
        }


@shared_task(
    name="historical.save_bars_v2",
    queue="historical",
)
def save_bars_v2_task(
    fetch_result: dict,
    repository=None,
    repository_type: str = "equity",
    root: str | None = None,
) -> dict:
    """
    Save historical bars from fetch_bars_for_contract_v2_task.
    Returns summary of what was saved.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Check for error from previous task
    if "error" in fetch_result:
        return {
            "saved": 0,
            "skipped": 1,
            "error": fetch_result.get("error"),
            "symbol": fetch_result.get("symbol"),
        }

    settings = get_settings()
    base = Path(root) if root else settings.paths.data_root

    bars = fetch_result.get("bars", [])
    contract = fetch_result.get("contract", {})
    parameters = fetch_result.get("parameters", {})
    symbol = contract.get("symbol", "N/A")

    if not bars:
        logger.warning(f"💾 No bars to save for {symbol}")
        return {"saved_count": 0, "error": fetch_result.get("error")}

    try:
        df = _normalize_historical_bars(bars, contract, parameters)

        if df.empty:
            logger.warning(f"💾 No valid bars after normalization for {symbol}")
            return {"saved_count": 0}

        # Use provided repository or create one
        if repository is None:
            try:
                from common.factories.repository_factory import create_repository
            except ImportError:
                from factories.repository_factory import create_repository
            repository = create_repository(repository_type, base)

        repository.save(df)
        logger.info(f"💾 Saved {len(df)} bars for {symbol}")

        return {
            "saved_count": len(df),
            "symbol": symbol,
            "bar_size": parameters.get("bar_size_setting"),
        }

    except Exception as e:
        logger.error(f"💾 Error saving bars for {symbol}: {e}")
        return {"saved_count": 0, "error": str(e)}
