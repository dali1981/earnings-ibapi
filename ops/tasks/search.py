from __future__ import annotations

from app_settings.settings import get_settings
from celery import shared_task
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx
import pandas as pd


try:
    from vendors.ibx_repos.contract_descriptions import ContractDescriptionsRepository
except Exception:  # pragma: no cover
    ContractDescriptionsRepository = None  # type: ignore


def _normalize_matches(matches: List[dict]) -> pd.DataFrame:
    if not matches:
        return pd.DataFrame(
            columns=[
                "symbol",
                "conid",
                "sec_type",
                "currency",
                "primary_exchange",
                "derivative_sec_types",
            ]
        )
    rows: List[Dict[str, Any]] = []
    for m in matches:
        if not m:
            continue
        # Extract contract details from nested structure
        contract = m.get("contract", {})
        if contract:
            row = {
                "symbol": contract.get("symbol"),
                "conid": contract.get("conId"),
                "sec_type": contract.get("secType"),
                "currency": contract.get("currency"),
                "primary_exchange": contract.get("primaryExchange"),
                "derivative_sec_types": m.get("derivativeSecTypes", []),
                "description": contract.get("description", ""),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # No need to rename columns since we set them correctly above
    # Ensure all required columns exist
    for col in [
        "symbol",
        "conid",
        "sec_type",
        "currency",
        "primary_exchange",
        "derivative_sec_types",
    ]:
        if col not in df.columns:
            df[col] = None
    df["conid"] = pd.to_numeric(df["conid"], errors="coerce")
    return df


@shared_task(
    name="search.fetch_matching_symbols",
    rate_limit="10/s",
    bind=True,
    max_retries=3,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def fetch_matching_symbols_task(
    self,
    *,
    pattern: str,
    sec_types: Optional[Sequence[str]] = None,
    currency: Optional[str] = None,
    exchanges: Optional[Sequence[str]] = None,
    api_base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    s = get_settings()
    base = api_base or str(s.api.contracts_service_base)
    payload = {"pattern": pattern, "timeout": 10.0}
    with httpx.Client(base_url=base, timeout=30.0) as client:
        resp = client.post("/ib/matching-symbols", json=payload)
        resp.raise_for_status()
        matches = resp.json()
    df = _normalize_matches(matches)
    if df.empty:
        return []
    if sec_types:
        df = df[df["sec_type"].astype(str).isin([s.upper() for s in sec_types])]
    if currency:
        df = df[df["currency"].astype(str) == currency]
    if exchanges:
        df = df[df["primary_exchange"].astype(str).isin([e.upper() for e in exchanges])]
    df = df.dropna(subset=["conid"]).drop_duplicates(subset=["conid"]).copy()
    return df.to_dict(orient="records")


@shared_task(name="search.save_contract_descriptions")
def save_contract_descriptions_task(
    rows: List[Dict[str, Any]], base_path: Optional[str] = None
) -> int:
    if not rows:
        return 0
    s = get_settings()
    base = (
        Path(base_path) if base_path else (s.paths.data_root / "contract_descriptions")
    )
    repo = ContractDescriptionsRepository(base)
    # Create DataFrame directly from already-processed rows (don't normalize again)
    df = pd.DataFrame(rows)
    # Ensure derivative_sec_types column exists and is list-like
    if "derivative_sec_types" not in df.columns:
        df["derivative_sec_types"] = [[] for _ in range(len(df))]
    repo.save(df)
    return len(df)
