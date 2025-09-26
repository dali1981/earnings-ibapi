from __future__ import annotations

from app_settings.settings import get_settings
from celery import shared_task
from collections import defaultdict
from common.repositories.contract_descriptions import ContractDescriptionsRepository
from common.repositories.contracts import ContractRepository
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import pandas as pd


@shared_task(
    bind=True,
    name="contracts.check_existing_details",
    queue="contracts",
)
def check_existing_details_task(
    self,
    symbols: List[str],
    root: Optional[str] = None,
) -> Dict[str, List]:
    """Check which symbols already have contract details in the repository.

    Returns:
        Dict with 'existing_conids' and 'missing_symbols' lists
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"check_existing_details_task called with symbols: {symbols}")

    settings = get_settings()
    base = Path(root) if root else settings.paths.data_root

    # Normalize symbols
    symbols = sorted({s.strip().upper() for s in symbols if s.strip()})
    if not symbols:
        return {"existing_conids": [], "missing_symbols": []}

    # Get conids for symbols from contract descriptions
    from celery.contrib import rdb

    import pyarrow as pa
    import pyarrow.compute as pc

    rdb.set_trace()

    desc_repo = ContractDescriptionsRepository(base)
    filter_ = ("symbol", "in", symbols)
    tbl = desc_repo.load(columns=["symbol", "conid"], filters=[filter_])

    symbol_to_conids: Dict[str, List[int]] = defaultdict(list)
    if tbl is not None and tbl.num_rows > 0:
        sym_col = tbl.column("symbol")
        conid_col = tbl.column("conid")
        for i in range(tbl.num_rows):
            sym = (sym_col[i].as_py() or "").upper()
            conid_val = conid_col[i].as_py()
            if conid_val is None:
                continue
            symbol_to_conids.setdefault(sym, []).append(int(conid_val))

    # Check which conids already have details
    contracts_repo = ContractRepository(base)
    existing = contracts_repo.present_conids()

    existing_conids = []
    missing_symbols = []

    for sym in symbols:
        if sym in symbol_to_conids:
            sym_conids = symbol_to_conids[sym]
            # Check if any conid for this symbol already has details
            has_details = any(c in existing for c in sym_conids)
            if has_details:
                existing_conids.extend([c for c in sym_conids if c in existing])
            else:
                missing_symbols.append(sym)
        else:
            # Symbol not found in descriptions at all
            missing_symbols.append(sym)

    return {
        "existing_conids": sorted(set(existing_conids)),
        "missing_symbols": sorted(set(missing_symbols)),
        "symbol_to_conids": symbol_to_conids,
    }


@shared_task(
    bind=True,
    rate_limit="10/m",
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
    name="contracts.fetch_details_from_ib",
    queue="contracts",
)
def fetch_details_from_ib_task(
    self,
    check_result: Dict[str, List],
    api_base: Optional[str] = None,
    timeout_seconds: int = 20,
) -> Dict[str, List]:
    """Fetch contract details from IB API for missing symbols.

    Returns:
        Dict with 'fetched_details' list of contract detail records
    """
    settings = get_settings()
    # Use IB API service base URL from settings
    ib_api_base_url = api_base or str(settings.api.ib_api_base)

    missing_symbols = check_result.get("missing_symbols", [])
    symbol_to_conids = check_result.get("symbol_to_conids", {})

    if not missing_symbols:
        return {"fetched_details": []}

    fetched_details = []

    with httpx.Client(base_url=ib_api_base_url, timeout=30.0) as client:
        for symbol in missing_symbols:
            # Get conids for this symbol
            conids = symbol_to_conids.get(symbol, [])

            # if not conids:
            #     # Need to search for the symbol first
            #     try:
            #         resp = client.post(
            #             "/ib/matching-symbols",
            #             json={"pattern": symbol, "timeout": 10.0},
            #         )
            #         resp.raise_for_status()
            #         matches = resp.json()
            #
            #         # Extract conids from matches
            #         for match in matches:
            #             if match.get("symbol") == symbol:
            #                 conid = match.get("conId")
            #                 if conid:
            #                     conids.append(int(conid))
            #     except Exception as e:
            #         print(f"Error searching for symbol {symbol}: {e}")
            #         continue

            # Fetch details for each conid
            # for conid in conids[:1]:  # Take first conid for the symbol
            for conid in conids:  # Take first conid for the symbol
                try:
                    resp = client.post(
                        "/ib/contract-details",
                        json={
                            "contract": {"conId": int(conid)},
                            "timeout": timeout_seconds,
                        },
                    )
                    resp.raise_for_status()
                    details = resp.json()

                    if details and isinstance(details, list):
                        for detail in details:
                            row = _from_contract_details_json(detail)
                            if row.get("conid"):
                                row["symbol"] = symbol  # Ensure symbol is set
                                fetched_details.append(row)
                                break  # One detail per symbol is enough
                except Exception as e:
                    print(f"Error fetching details for conid {conid}: {e}")
                    continue

    from celery.contrib import rdb

    rdb.set_trace()

    return {"fetched_details": fetched_details}


@shared_task(
    bind=True,
    name="contracts.save_details_to_repo",
    queue="contracts",
)
def save_details_to_repo_task(
    self,
    fetch_result: Dict[str, List],
    root: Optional[str] = None,
) -> Dict[str, int]:
    """Save fetched contract details to the repository.

    Returns:
        Dict with 'saved_count' indicating number of records saved
    """
    try:
        from ibx_repos.contracts import ContractRepository
    except Exception:
        from vendors.ibx_repos.contracts import ContractRepository

    from celery.contrib import rdb

    rdb.set_trace()

    settings = get_settings()
    base = Path(root) if root else settings.paths.data_root

    fetched_details = fetch_result.get("fetched_details", [])

    if not fetched_details:
        return {"saved_count": 0}

    # Save to repository
    contracts_repo = ContractRepository(base / "contracts")
    df = pd.DataFrame(fetched_details)

    # Ensure no duplicates by conid
    if not df.empty:
        df = df.drop_duplicates(subset=["conid"], keep="first")
        contracts_repo.save(df)
        saved_count = len(df)
    else:
        saved_count = 0

    return {"saved_count": saved_count}


def _from_contract_details_json(cd: dict) -> dict:
    """Convert contract details JSON to repository row format."""
    c = cd.get("contract", {}) if isinstance(cd, dict) else {}

    def g(obj, key, default=None):
        if not isinstance(obj, dict):
            return default
        return obj.get(key, default)

    row = {
        "conid": int(g(c, "conId", 0) or 0),
        "symbol": g(c, "symbol"),
        "sec_type": g(c, "secType"),
        "currency": g(c, "currency"),
        "primary_exchange": g(c, "primaryExchange"),
        "exchange": g(c, "exchange"),
        "local_symbol": g(c, "localSymbol"),
        "trading_class": g(c, "tradingClass"),
        "long_name": cd.get("longName"),
        "category": cd.get("category"),
        "sub_category": cd.get("subcategory", cd.get("subCategory")),
        "industry": cd.get("industry"),
        "time_zone": cd.get("timeZoneId"),
    }

    row["is_us_listing"] = (row.get("currency") == "USD") and (
        row.get("primary_exchange") in {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "IEX"}
    )

    return row
