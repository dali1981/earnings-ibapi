from __future__ import annotations

import string
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Set, Dict, Any, Sequence
import pandas as pd
try:
    from ibx import ContractDetailsService  # type: ignore
except Exception:
    from ibx.services import ContractDetailsService  # type: ignore
try:
    from ibx import IBRuntime  # type: ignore
except Exception:
    from ibx.runtime import IBRuntime  # type: ignore
try:
    from ibx import make_stock  # type: ignore
except Exception:
    from ibx.contracts import make_stock  # type: ignore

from ibx_repos.contract_descriptions import ContractDescriptionsRepository
from ibx_flows.symbol_search_ib import IBSymbolSearch


@dataclass
class ContractsConfig:
    exchanges: List[str] = None
    currency: str = "USD"
    sec_types: List[str] = None
    patterns: List[str] = None
    def __post_init__(self):
        if self.exchanges is None: self.exchanges = ["NYSE","NASDAQ","AMEX","ARCA","BATS","IEX"]
        if self.sec_types is None: self.sec_types = ["STK"]
        if self.patterns is None:
            alphabet = string.ascii_uppercase  # 'A'..'Z'
            self.patterns = [''.join(p) for p in product(alphabet, repeat=3)]

    def normalized(self) -> "ContractsConfig":
        # ensure upper-case filters
        self.exchanges  = [x.upper() for x in self.exchanges ]
        # self.currencies = [x.upper() for x in self.currencies]
        self.sec_types  = [x.upper() for x in self.sec_types ]
        return self

def _from_contract_details(cd) -> dict:
    c = cd.contract
    row = {"conid": int(getattr(c,"conId",0) or 0), "symbol": getattr(c,"symbol",None), "sec_type": getattr(c,"secType",None), "currency": getattr(c,"currency",None), "primary_exchange": getattr(c,"primaryExchange",None), "exchange": getattr(c,"exchange",None), "local_symbol": getattr(c,"localSymbol",None), "trading_class": getattr(c,"tradingClass",None), "long_name": getattr(cd,"longName",None), "category": getattr(cd,"category",None), "sub_category": getattr(cd,"subcategory",None) if hasattr(cd,"subcategory") else getattr(cd,"subCategory",None), "industry": getattr(cd,"industry",None), "time_zone": getattr(cd,"timeZoneId",None)}
    row["is_us_listing"] = (row.get("currency") == "USD") and (row.get("primary_exchange") in {"NYSE","NASDAQ","AMEX","ARCA","BATS","IEX"}); return row


def backfill_us_equity_contracts(rt: IBRuntime, repo: ContractRepository, cfg: ContractsConfig) -> None:
    existing: Set[int] = repo.present_conids();
    to_save_rows: list[dict] = [];
    search = IBSymbolSearch(rt)

    for pat in cfg.patterns:
        matches = search.search(pat)
        if matches is None or matches.empty: continue
        cols = {c.lower(): c for c in matches.columns}
        def get(col): col_l = col.lower(); return matches[cols[col_l]] if col_l in cols else pd.Series([None]*len(matches))
        df = pd.DataFrame({"symbol": get("symbol"), "conid": pd.to_numeric(get("conid"), errors="coerce"), "sec_type": get("secType").astype(str), "currency": get("currency").astype(str), "primary_exchange": get("primaryExchange").astype(str)})
        df = df[df["sec_type"].isin(cfg.sec_types)]; df = df[df["currency"] == cfg.currency]; df = df[df["primary_exchange"].isin(cfg.exchanges)]
        df = df.dropna(subset=["conid"]).drop_duplicates(subset=["conid"]); df = df[~df["conid"].astype(int).isin(existing)]

        if df.empty:
            continue

        cds = ContractDetailsService(rt)
        for conid in df["conid"].astype(int).tolist():
            c = make_stock("?"); c.conId = int(conid)
            try: details = cds.fetch(c, timeout=20.0)
            except Exception: continue
            if not details: continue
            row = _from_contract_details(details[0])
            if row.get("conid") and row["conid"] not in existing:
                to_save_rows.append(row); existing.add(row["conid"])
    if to_save_rows: repo.save(pd.DataFrame(to_save_rows))


# ---------------------------------------------------------------------
# Backfill Job
# ---------------------------------------------------------------------

class ContractDescriptionsSnapshotJob:
    def __init__(self,
                 cfg: ContractsConfig,
                 rt: IBRuntime,
                 repo: ContractDescriptionsRepository):
        self.cfg = cfg.normalized()
        self.rt = rt
        self.repo = repo



    def run(self) -> None:
        sec_types: Sequence[str] = self.cfg.sec_types
        exchanges: Sequence[str] = self.cfg.exchanges

        search = IBSymbolSearch(self.rt)

        for pattern in self.cfg.patterns:
            matches = search.search(pattern)
            if matches is None or matches.empty:
                continue

            cols = {c.lower(): c for c in matches.columns}

            def get(col):
                col_l = col.lower(); return matches[cols[col_l]] if col_l in cols else pd.Series([None] * len(matches))

            df = pd.DataFrame({"symbol": get("symbol"),
                               "conid": pd.to_numeric(get("conid"), errors="coerce"),
                               "sec_type": get("secType").astype(str),
                               "currency": get("currency").astype(str),
                               "primary_exchange": get("primaryExchange").astype(str),
                               "derivative_sec_types": get("derivativeSecTypes"),
                               })

            df = df[df["sec_type"].isin(sec_types)]
            df = df[df["currency"] == self.cfg.currency]
            df = df[df["primary_exchange"].isin(exchanges)]
            df = df.dropna(subset=["conid"]).drop_duplicates(subset=["conid"])

            if df.empty:
                continue

            self.repo.save(df)

