from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from celery import shared_task

from ibx_config.settings import get_settings
from ibx_flows.contracts_backfill import ContractsConfig, backfill_us_equity_contracts
try:
    from ibx_repos.contracts import ContractRepository  # type: ignore
except Exception:  # pragma: no cover - fallback for vendored layout
    from vendors.ibx_repos.contracts import ContractRepository  # type: ignore

try:
    from ibx import IBRuntime  # type: ignore
except Exception:  # pragma: no cover - fallback for vendored layout
    from ibx.runtime import IBRuntime  # type: ignore

from ops.ib_service import get_runtime
import os
import httpx
import pandas as pd


def _expand_patterns_if_needed(patterns: Optional[Sequence[str]]) -> Optional[List[str]]:
    # If None, let ContractsConfig default expansion handle it inside task
    if patterns is None:
        return None
    # Normalize to upper-case strings
    return [str(p).upper() for p in patterns]


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
    name="contracts.backfill_fetch",
)
def backfill_contracts_task(
    self,
    *,
    root: Optional[str] = None,
    exchanges: Optional[Sequence[str]] = None,
    currency: str = "USD",
    patterns: Optional[Sequence[str]] = None,
    use_service: bool = False,
    use_api: bool = False,
    api_base: Optional[str] = None,
    ib_host: Optional[str] = None,
    ib_port: Optional[int] = None,
    client_id: Optional[int] = None,
) -> int:
    """Backfill US equity contract descriptions.

    Returns the number of patterns processed (0 if using default expansion).
    """
    settings = get_settings()
    base = Path(root) if root else settings.repo.contracts_path
    base.mkdir(parents=True, exist_ok=True)
    repo = ContractRepository(base)

    cfg = ContractsConfig(
        exchanges=[str(e).upper() for e in (exchanges or ["NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "IEX"])],
        currency=currency,
        patterns=_expand_patterns_if_needed(patterns),
    ).normalized()

    if use_api:
        base_url = api_base or os.environ.get("IB_API_BASE", str(settings.api.contracts_service_base))
        _backfill_via_api(base, cfg, base_url)
    elif use_service:
        rt = get_runtime()
        backfill_us_equity_contracts(rt, repo, cfg)
    else:
        host = ib_host or settings.ib.host
        port = int(ib_port or settings.ib.port)
        cid = int(client_id or settings.ib.client_id_contracts)
        with IBRuntime(host=host, port=port, client_id=cid) as rt:
            backfill_us_equity_contracts(rt, repo, cfg)

    return len(cfg.patterns) if cfg.patterns is not None else 0


def _backfill_via_api(base: Path, cfg: ContractsConfig, base_url: str) -> None:
    repo = ContractRepository(base)
    existing = repo.present_conids()
    to_save: List[dict] = []

    # Helper: filter matches and pick unique new conids
    def _filter_matches(matches: List[dict]) -> pd.DataFrame:
        if not matches:
            return pd.DataFrame()
        # Normalize keys to lower
        rows = []
        for m in matches:
            d = {str(k).lower(): v for k, v in (m or {}).items()}
            rows.append(d)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.rename(columns={
            "sectype": "sec_type",
            "primaryexchange": "primary_exchange",
        })
        for col in ["symbol", "conid", "sec_type", "currency", "primary_exchange"]:
            if col not in df.columns:
                df[col] = None
        df["sec_type"] = df["sec_type"].astype(str)
        df["currency"] = df["currency"].astype(str)
        df["primary_exchange"] = df["primary_exchange"].astype(str)
        df = df[df["sec_type"].isin(cfg.sec_types)]
        df = df[df["currency"] == cfg.currency]
        df = df[df["primary_exchange"].isin(cfg.exchanges)]
        df["conid"] = pd.to_numeric(df["conid"], errors="coerce")
        df = df.dropna(subset=["conid"]).drop_duplicates(subset=["conid"]).copy()
        if not df.empty:
            df = df[~df["conid"].astype(int).isin(existing)]
        return df

    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        for pat in (cfg.patterns or []):
            try:
                resp = client.post("/ib/matching-symbols", json={"pattern": pat, "timeout": 10.0})
                resp.raise_for_status()
                matches = resp.json()
            except Exception:
                continue
            df = _filter_matches(matches)
            if df.empty:
                continue
            for conid in df["conid"].astype(int).tolist():
                try:
                    resp = client.post("/ib/contract-details", json={"contract": {"conId": int(conid)}, "timeout": 20.0})
                    resp.raise_for_status()
                    details = resp.json()
                except Exception:
                    continue
                if not details:
                    continue
                row = _from_contract_details_json(details[0])
                if row.get("conid") and row["conid"] not in existing:
                    to_save.append(row)
                    existing.add(row["conid"])
    if to_save:
        repo.save(pd.DataFrame(to_save))


def _from_contract_details_json(cd: dict) -> dict:
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
