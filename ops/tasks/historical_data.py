from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, date

from celery import shared_task
import pandas as pd
import httpx

from app_settings.settings import get_settings


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
    
    logger.info(f"📡 FETCH: Fetching historical bars for contract: {contract.get('symbol', 'N/A')} (conid: {contract.get('conId')})")
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
            
            logger.info(f"📡 SUCCESS: Fetched {len(bars)} bars for {contract.get('symbol', 'N/A')}")
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
        logger.error(f"📡 ERROR: Error fetching historical bars for {contract.get('symbol', 'N/A')}: {e}")
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
    """Save fetched historical bars to the appropriate repository.
    
    Args:
        fetch_result: Result from fetch_historical_bars_task
        repository_type: Type of repository ("equity" or "option")
        root: Base path override
        
    Returns:
        Dict with 'saved_count' and metadata
    """
    import logging
    logger = logging.getLogger(__name__)
    
    settings = get_settings()
    base = Path(root) if root else settings.paths.data_root
    
    bars = fetch_result.get("bars", [])
    contract = fetch_result.get("contract", {})
    parameters = fetch_result.get("parameters", {})
    
    logger.info(f"💾 SAVE: Starting save for {contract.get('symbol', 'N/A')}, {len(bars)} bars, repository_type={repository_type}")
    logger.info(f"💾 Base path: {base}")
    
    if not bars:
        logger.warning(f"💾 WARNING: No bars to save for {contract.get('symbol', 'N/A')}")
        return {"saved_count": 0, "error": fetch_result.get("error")}
    
    try:
        # Convert bars to DataFrame
        logger.info(f"💾 Normalizing {len(bars)} bars...")
        logger.info(f"💾 Sample raw bar: {bars[0] if bars else 'None'}")
        
        df = _normalize_historical_bars(bars, contract, parameters)
        
        logger.info(f"💾 After normalization: {len(df)} rows, columns: {list(df.columns) if not df.empty else 'empty'}")
        if not df.empty:
            logger.info(f"💾 Sample normalized row: {df.iloc[0].to_dict()}")
        
        if df.empty:
            logger.warning(f"💾 WARNING: No valid bars after normalization for {contract.get('symbol', 'N/A')}")
            return {"saved_count": 0}
        
        # Choose appropriate repository
        if repository_type == "equity":
            from common.repositories.equity_bars import EquityBarRepository
            repo = EquityBarRepository(base)
            logger.info(f"💾 Using EquityBarRepository at {base}")
        elif repository_type == "option":
            from common.repositories.option_bars import OptionBarRepository
            repo = OptionBarRepository(base)
            logger.info(f"💾 Using OptionBarRepository at {base}")
        else:
            raise ValueError(f"Unknown repository type: {repository_type}")
        
        # Save to repository
        logger.info(f"💾 Calling repo.save() with DataFrame shape: {df.shape}")
        repo.save(df)
        logger.info(f"💾 SUCCESS: Saved {len(df)} bars for {contract.get('symbol', 'N/A')}")
        
        return {
            "saved_count": len(df),
            "symbol": contract.get("symbol"),
            "bar_size": parameters.get("bar_size_setting"),
            "date_range": {
                "start": df["time"].min().isoformat() if not df.empty else None,
                "end": df["time"].max().isoformat() if not df.empty else None,
            },
        }
        
    except Exception as e:
        logger.error(f"💾 ERROR: Error saving historical bars for {contract.get('symbol', 'N/A')}: {e}")
        import traceback
        logger.error(f"💾 TRACEBACK: {traceback.format_exc()}")
        return {"saved_count": 0, "error": str(e)}


@shared_task(
    bind=True,
    name="historical.resolve_contracts_for_symbols",
    queue="contracts",
)
def resolve_contracts_for_symbols_task(
    self,
    symbols: List[str],
    root: Optional[str] = None,
) -> Dict[str, List]:
    """Resolve contract details for symbols from contract_descriptions and contracts repositories.
    
    Args:
        symbols: List of ticker symbols
        root: Base path override
        
    Returns:
        Dict with 'resolved_contracts', 'missing_symbols'
    """
    import logging
    logger = logging.getLogger(__name__)
    
    settings = get_settings()
    base = Path(root) if root else settings.paths.data_root
    
    # Normalize symbols
    symbols = sorted({s.strip().upper() for s in symbols if s.strip()})
    if not symbols:
        return {"resolved_contracts": [], "missing_symbols": []}
    
    logger.info(f"Resolving contracts for symbols: {symbols}")
    
    resolved_contracts = []
    missing_symbols = []
    
    try:
        # Import repositories with fallback
        try:
            from vendors.ibx_repos.contract_descriptions import ContractDescriptionsRepository
            from vendors.ibx_repos.contracts import ContractRepository
        except Exception:
            from ibx_repos.contract_descriptions import ContractDescriptionsRepository
            from ibx_repos.contracts import ContractRepository
        
        # Get contract descriptions
        desc_repo = ContractDescriptionsRepository(base / "contract_descriptions")
        contracts_repo = ContractRepository(base / "contracts")
        
        # Read contract descriptions for symbols
        desc_df = desc_repo.read(columns=["conid", "symbol"])
        
        # Filter by symbols if any exist
        if not desc_df.empty and symbols:
            desc_df = desc_df[desc_df["symbol"].isin(symbols)]
        
        if not desc_df.empty:
            # Get unique conids from descriptions
            conids = desc_df["conid"].dropna().astype(int).unique().tolist()
            
            # Read full contract details for these conids
            contracts_df = contracts_repo.load()
            
            if not contracts_df.empty:
                # Filter contracts for our conids
                matching_contracts = contracts_df[contracts_df["conid"].isin(conids)]
                
                # Convert to contract format for IB API
                for _, row in matching_contracts.iterrows():
                    contract = {
                        "conId": int(row["conid"]),
                        "symbol": row["symbol"],
                        "secType": row["sec_type"],
                        "currency": row["currency"],
                        "exchange": "SMART",  # Default to SMART routing
                        "primaryExchange": row.get("primary_exchange", ""),
                    }
                    resolved_contracts.append(contract)
                    
                    # Remove from missing if found
                    if row["symbol"] in symbols:
                        symbols = [s for s in symbols if s != row["symbol"]]
        
        # Any remaining symbols are missing
        missing_symbols = symbols
        
        logger.info(f"Resolved {len(resolved_contracts)} contracts, {len(missing_symbols)} missing")
        
        return {
            "resolved_contracts": resolved_contracts,
            "missing_symbols": missing_symbols,
        }
        
    except Exception as e:
        logger.error(f"Error resolving contracts: {e}")
        return {
            "resolved_contracts": [],
            "missing_symbols": symbols,
            "error": str(e),
        }


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
    logger.info(f"📋 PARAMS: end_date={end_date_time}, duration={duration_str}, bar_size={bar_size_setting}, repository={repository_type}")
    
    # Step 1: Resolve contracts for symbols
    logger.info(f"📍 STEP 1: Resolving contracts for symbols: {symbols}")
    resolve_result = resolve_contracts_for_symbols_task(symbols, root)
    logger.info(f"✅ RESOLVED: {resolve_result}")
    
    resolved_contracts = resolve_result.get("resolved_contracts", [])
    missing_symbols = resolve_result.get("missing_symbols", [])
    
    if resolve_result.get("error"):
        logger.error(f"❌ ERROR in contract resolution: {resolve_result['error']}")
        return {
            "processed": 0,
            "saved": 0,
            "failed": len(symbols),
            "error": resolve_result["error"]
        }
    
    if not resolved_contracts:
        logger.warning(f"⚠️  NO CONTRACTS resolved for symbols: {symbols}, missing: {missing_symbols}")
        return {
            "processed": len(symbols),
            "saved": 0,
            "failed": 0,
            "missing_symbols": missing_symbols
        }
    
    logger.info(f"📍 STEP 2: Resolved {len(resolved_contracts)} contracts, processing historical data...")
    for i, contract in enumerate(resolved_contracts):
        logger.info(f"   Contract {i+1}: {contract.get('symbol')} (conid: {contract.get('conId')})")
    
    # Step 2: Process each resolved contract
    processed_count = 0
    saved_count = 0
    failed_count = 0
    results = []
    
    for i, contract in enumerate(resolved_contracts):
        contract_symbol = contract.get('symbol', 'N/A')
        logger.info(f"📍 STEP 3.{i+1}: Processing contract {contract_symbol}")
        
        try:
            # Fetch historical bars
            logger.info(f"   🔄 Fetching historical bars for {contract_symbol}...")
            fetch_result = fetch_historical_bars_task(
                contract=contract,
                end_date_time=end_date_time,
                duration_str=duration_str,
                bar_size_setting=bar_size_setting,
                what_to_show=what_to_show,
                use_rth=use_rth,
                timeout_seconds=timeout_seconds,
            )
            
            logger.info(f"   📊 Fetch result for {contract_symbol}: {len(fetch_result.get('bars', []))} bars, error: {fetch_result.get('error', 'None')}")
            
            if fetch_result.get("error"):
                logger.error(f"   ❌ Failed to fetch bars for {contract_symbol}: {fetch_result['error']}")
                failed_count += 1
                results.append({
                    "symbol": contract_symbol,
                    "saved_count": 0,
                    "fetch_error": fetch_result['error']
                })
                continue
                
            # Save bars to repository
            logger.info(f"   💾 Saving {len(fetch_result.get('bars', []))} bars for {contract_symbol} to {repository_type} repository...")
            save_result = save_historical_bars_task(
                fetch_result=fetch_result,
                repository_type=repository_type,
                root=root,
            )
            
            logger.info(f"   💾 Save result for {contract_symbol}: {save_result}")
            
            processed_count += 1
            if save_result.get("saved_count", 0) > 0:
                saved_count += save_result["saved_count"]
                logger.info(f"   ✅ Saved {save_result['saved_count']} bars for {contract_symbol}")
            else:
                logger.warning(f"   ⚠️  No bars saved for {contract_symbol}: {save_result}")
            
            results.append({
                "symbol": contract_symbol,
                "saved_count": save_result.get("saved_count", 0),
                "error": save_result.get("error")
            })
            
        except Exception as e:
            logger.error(f"   ❌ Exception processing contract {contract_symbol}: {e}")
            failed_count += 1
            results.append({
                "symbol": contract_symbol,
                "saved_count": 0,
                "error": str(e)
            })
    
    logger.info(f"Historical data processing complete: {processed_count} processed, {saved_count} bars saved, {failed_count} failed")
    
    return {
        "processed": processed_count,
        "saved": saved_count,
        "failed": failed_count,
        "missing_symbols": missing_symbols,
        "results": results
    }


def _normalize_historical_bars(
    bars: List[Dict[str, Any]], 
    contract: Dict[str, Any], 
    parameters: Dict[str, Any]
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