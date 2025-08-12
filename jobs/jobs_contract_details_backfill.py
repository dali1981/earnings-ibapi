
from __future__ import annotations
import argparse
import logging
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, List

import pandas as pd

try:
    from ibx import IBRuntime, ContractDetailsService
except Exception:
    from ibx.runtime import IBRuntime  # type: ignore
    from ibx.services import ContractDetailsService  # type: ignore

def make_contract_by_conid(conid: int):
    try:
        from ibapi.contract import Contract  # type: ignore
        c = Contract()
        c.conId = int(conid)
        c.exchange = "SMART"
        return c
    except Exception:
        try:
            from ibx.contracts import make_by_conid  # type: ignore
            return make_by_conid(conid)
        except Exception as e:
            raise RuntimeError("Cannot build Contract by conid; please provide a factory") from e

from contract_descriptions_repo import ContractDescriptionsRepo
from contract_details_repo import ContractDetailsRepo

log = logging.getLogger("jobs.contract_details_backfill")

@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 8
    desc_path: str = "data/contract_descriptions"
    out_path: str = "data/contract_details"
    timeout: float = 20.0
    as_of: dt.date = dt.date.today()
    queries: List[str] | None = None
    sec_types: List[str] | None = None

def map_detail_row(cd: Any, as_of: dt.date) -> Dict[str, Any]:
    c = cd.contract
    return {
        "conid": int(getattr(c, "conId", 0) or 0),
        "symbol": getattr(c, "symbol", "") or "",
        "local_symbol": getattr(c, "localSymbol", "") or "",
        "sec_type": getattr(c, "secType", "") or "",
        "currency": getattr(c, "currency", "") or "",
        "primary_exchange": getattr(c, "primaryExchange", "") or "",
        "exchange": getattr(c, "exchange", "") or "",
        "trading_class": getattr(c, "tradingClass", "") or "",
        "description": getattr(cd, "longName", "") or "",
        "long_name": getattr(cd, "longName", "") or "",
        "multiplier": getattr(c, "multiplier", "") or "",
        "min_tick": getattr(cd, "minTick", None),
        "price_magnifier": getattr(cd, "priceMagnifier", None),
        "md_size_multiplier": getattr(cd, "mdSizeMultiplier", None),
        "time_zone": getattr(cd, "timeZoneId", "") or "",
        "valid_exchanges": getattr(cd, "validExchanges", "").split(",") if getattr(cd, "validExchanges", "") else [],
        "order_types": getattr(cd, "orderTypes", "").split(",") if getattr(cd, "orderTypes", "") else [],
        "as_of_date": as_of,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--client-id", type=int, default=8)
    ap.add_argument("--desc-path", default="data/contract_descriptions")
    ap.add_argument("--out", default="data/contract_details")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--queries", default="")
    ap.add_argument("--sec-types", default="")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    cfg = Config(
        host=args.host, port=args.port, client_id=args.client_id,
        desc_path=args.desc_path, out_path=args.out, timeout=args.timeout,
        queries=[s for s in args.queries.split(",") if s.strip()] or None,
        sec_types=[s.upper() for s in args.sec_types.split(",") if s.strip()] or None,
    )

    desc_repo = ContractDescriptionsRepo(cfg.desc_path)
    det_repo = ContractDetailsRepo(cfg.out_path)

    filt = {}
    if cfg.queries:  filt["query"] = cfg.queries
    if cfg.sec_types: filt["sec_type"] = cfg.sec_types
    desc_df = desc_repo.read(columns=["conid","query","sec_type"], **filt)
    desc_df = desc_df.dropna(subset=["conid"]).drop_duplicates(subset=["conid"])
    target_conids = [int(x) for x in desc_df["conid"].tolist()]

    existing = det_repo.existing_conids()
    todo = [c for c in target_conids if c not in existing]
    log.info("Conids to fetch: %d (existing=%d)", len(todo), len(existing))

    if not todo:
        return

    with IBRuntime(cfg.host, cfg.port, cfg.client_id) as rt:
        svc = ContractDetailsService(rt)
        for conid in todo:
            try:
                cd_list = svc.fetch(make_contract_by_conid(conid), timeout=cfg.timeout)
                rows = [map_detail_row(cd, cfg.as_of) for cd in (cd_list or [])]
                if rows:
                    det_repo.save_rows(rows)
                    log.info("Saved details for conid=%s", conid)
                else:
                    log.info("No details for conid=%s", conid)
            except Exception as e:
                log.warning("Details fetch failed for conid=%s: %s", conid, e)

if __name__ == "__main__":
    main()
