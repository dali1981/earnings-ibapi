
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Dict, Any, Callable

import pandas as pd

# --- ibx runtime/services (kept as-is to match your project layout) ---
try:
    from ibx import IBRuntime, ContractDetailsService
except Exception:
    from ibx.runtime import IBRuntime  # type: ignore
    from ibx.services import ContractDetailsService  # type: ignore

try:
    from ibx import make_stock
except Exception:
    from ibx.contracts import make_stock  # type: ignore

log = logging.getLogger("contracts_backfill")


# ---------------------------------------------------------------------
# Data model & config
# ---------------------------------------------------------------------

@dataclass
class BackfillConfig:
    # IB connection
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1

    # Filters
    exchanges: Sequence[str] = field(default_factory=lambda: ["NYSE", "NASDAQ", "ARCA", "BATS", "ISLAND", "SMART"])
    currencies: Sequence[str] = field(default_factory=lambda: ["USD"])
    sec_types: Sequence[str] = field(default_factory=lambda: ["STK"])

    # Input universe
    symbols: Sequence[str] = field(default_factory=list)
    symbols_file: Optional[str] = None  # CSV or newline file with a 'symbol' column or one symbol per line
    limit: Optional[int] = None  # stop after N symbols (for testing)

    # Execution
    chunk_size: int = 50
    timeout_sec: float = 20.0

    # Storage
    out_path: str = "data/contracts"  # hive-partitioned parquet dataset

    def normalized(self) -> "BackfillConfig":
        # ensure upper-case filters
        self.exchanges = tuple(x.upper() for x in self.exchanges)
        self.currencies = tuple(x.upper() for x in self.currencies)
        self.sec_types = tuple(x.upper() for x in self.sec_types)
        return self


# ---------------------------------------------------------------------
# Repository: parquet (hive partitions)
#   base/primary_exchange=NYSE/symbol=AAPL/part-*.parquet
# ---------------------------------------------------------------------

class ContractsRepository:
    REQUIRED_COLS = [
        "conid", "symbol", "local_symbol", "sec_type", "currency",
        "primary_exchange", "exchange", "trading_class", "description",
        "min_tick", "price_magnifier", "md_size_multiplier"
    ]

    def __init__(self, base_path: str):
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)

    def _dataset(self):
        import pyarrow.dataset as ds
        return ds.dataset(self.base, format="parquet", partitioning="hive")

    def existing_conids(self) -> Set[int]:
        try:
            import pyarrow.dataset as ds, pyarrow.compute as pc
            dset = self._dataset()
            # only scan conid column to keep it fast
            tbl = dset.to_table(columns=["conid"])
            if tbl.num_rows == 0:
                return set()
            arr = tbl.column("conid").to_pylist()
            return {int(x) for x in arr if x is not None}
        except Exception:
            # dataset might not exist yet
            return set()

    def save(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        # ensure required columns exist
        for c in self.REQUIRED_COLS:
            if c not in df.columns:
                df[c] = None

        # partition columns
        part_cols = ["primary_exchange", "symbol"]
        table = pa.Table.from_pandas(df[self.REQUIRED_COLS + part_cols], preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(self.base),
            partition_cols=part_cols,
            existing_data_behavior="overwrite_or_ignore",
            compression="snappy",
            use_dictionary=True,
        )


# ---------------------------------------------------------------------
# Mappers
# ---------------------------------------------------------------------

def _row_from_contract_details(cd: Any) -> Dict[str, Any]:
    """
    Map ibapi.contract.ContractDetails -> row dict
    This keeps only persistent and stable attributes for your repo.
    """
    c = cd.contract
    return {
        "conid": int(getattr(c, "conId", 0) or 0),
        "symbol": (getattr(c, "symbol", None) or "")[:64],
        "local_symbol": (getattr(c, "localSymbol", None) or "")[:64],
        "sec_type": (getattr(c, "secType", None) or "").upper(),
        "currency": (getattr(c, "currency", None) or "").upper(),
        "primary_exchange": (getattr(c, "primaryExchange", None) or "").upper(),
        "exchange": (getattr(c, "exchange", None) or "").upper(),
        "trading_class": (getattr(c, "tradingClass", None) or "")[:64],
        "description": (getattr(cd, "longName", None) or "")[:256],
        "min_tick": getattr(cd, "minTick", None),
        "price_magnifier": getattr(cd, "priceMagnifier", None),
        "md_size_multiplier": getattr(cd, "mdSizeMultiplier", None),
    }


# ---------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------

class ContractDetailsFetcher:
    def __init__(self, rt: IBRuntime, timeout_sec: float = 20.0):
        self.rt = rt
        self.cds = ContractDetailsService(rt)
        self.timeout = timeout_sec

    def fetch_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        try:
            details = self.cds.fetch(make_stock(symbol), timeout=self.timeout)
        except Exception as e:
            log.warning("fetch_symbol(%s) failed: %s", symbol, e)
            return []
        rows = []
        for d in details or []:
            try:
                rows.append(_row_from_contract_details(d))
            except Exception as e:
                log.debug("row map failed for %s: %s", symbol, e)
        return rows


# ---------------------------------------------------------------------
# Backfill Job
# ---------------------------------------------------------------------

class ContractsBackfillJob:
    def __init__(self, cfg: BackfillConfig, repo: ContractsRepository):
        self.cfg = cfg.normalized()
        self.repo = repo

    @staticmethod
    def _load_symbols(cfg: BackfillConfig) -> List[str]:
        symbols: List[str] = []
        if cfg.symbols:
            symbols.extend(cfg.symbols)
        if cfg.symbols_file:
            p = Path(cfg.symbols_file)
            if not p.exists():
                raise FileNotFoundError(f"symbols_file not found: {p}")
            if p.suffix.lower() in {'.csv', '.tsv'}:
                df = pd.read_csv(p)
                col = "symbol" if "symbol" in df.columns else df.columns[0]
                symbols.extend(df[col].astype(str).str.strip().tolist())
            else:
                # one per line
                symbols.extend(x.strip() for x in p.read_text().splitlines() if x.strip())

        # unique, uppercase
        uniq = []
        seen = set()
        for s in symbols:
            u = s.upper()
            if u not in seen:
                uniq.append(u)
                seen.add(u)
        if cfg.limit is not None:
            uniq = uniq[: int(cfg.limit)]
        return uniq

    def _filter_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return []
        df = pd.DataFrame(rows)
        df = df[df["sec_type"].isin(self.cfg.sec_types)]
        df = df[df["currency"].isin(self.cfg.currencies)]
        if "primary_exchange" in df.columns and len(self.cfg.exchanges) > 0:
            df = df[df["primary_exchange"].isin(self.cfg.exchanges)]
        # Drop missing conids and duplicates by conid
        df = df.dropna(subset=["conid"])
        if "conid" in df.columns:
            df["conid"] = df["conid"].astype(int)
            df = df.drop_duplicates(subset=["conid"])
        return df.to_dict("records")

    def run(self) -> None:
        symbols = self._load_symbols(self.cfg)
        if not symbols:
            log.warning("No symbols to backfill. Provide --symbols or --symbols-file.")
            return

        existing = self.repo.existing_conids()
        log.info("Existing contracts in repo: %d", len(existing))

        to_save: List[Dict[str, Any]] = []
        total = 0

        with IBRuntime(self.cfg.host, self.cfg.port, self.cfg.client_id) as rt:
            fetcher = ContractDetailsFetcher(rt, timeout_sec=self.cfg.timeout_sec)

            for i, sym in enumerate(symbols, start=1):
                rows = self._filter_rows(fetcher.fetch_symbol(sym))
                rows = [r for r in rows if r.get("conid") not in existing]
                if not rows:
                    if i % 100 == 0:
                        log.info("Progress %d/%d", i, len(symbols))
                    continue

                to_save.extend(rows)
                for r in rows:
                    existing.add(int(r["conid"]))
                total += len(rows)

                if len(to_save) >= self.cfg.chunk_size:
                    self._flush(to_save)

            # final flush
            if to_save:
                self._flush(to_save)

        log.info("Backfill done. Added %d new contracts.", total)

    def _flush(self, buffer: List[Dict[str, Any]]) -> None:
        df = pd.DataFrame(buffer)
        if df.empty:
            return
        self.repo.save(df)
        buffer.clear()
        log.info("Saved %d rows.", len(df))


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _parse_args() -> BackfillConfig:
    p = argparse.ArgumentParser(description="Backfill/refresh IB contract details into a Parquet dataset.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--client-id", type=int, default=1)

    p.add_argument("--exchanges", default="NYSE,NASDAQ,ARCA,BATS,ISLAND,SMART")
    p.add_argument("--currencies", default="USD")
    p.add_argument("--sec-types", default="STK")

    p.add_argument("--symbols", default="", help="Comma-separated list of symbols")
    p.add_argument("--symbols-file", default=None, help="CSV with a 'symbol' column, or newline-delimited file")
    p.add_argument("--limit", type=int, default=None)

    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--timeout", type=float, default=20.0)

    p.add_argument("--out", default="data/contracts")

    args = p.parse_args()

    cfg = BackfillConfig(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        exchanges=[x.strip().upper() for x in args.exchanges.split(",") if x.strip()],
        currencies=[x.strip().upper() for x in args.currencies.split(",") if x.strip()],
        sec_types=[x.strip().upper() for x in args.sec_types.split(",") if x.strip()],
        symbols=[x.strip().upper() for x in args.symbols.split(",") if x.strip()],
        symbols_file=args.symbols_file,
        limit=args.limit,
        chunk_size=args.chunk_size,
        timeout_sec=args.timeout,
        out_path=args.out,
    )
    return cfg

def _setup_logging():
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def main() -> None:
    _setup_logging()
    cfg = _parse_args()
    repo = ContractsRepository(cfg.out_path)
    job = ContractsBackfillJob(cfg, repo)
    log.info("Starting contracts backfill with config: %s", json.dumps(asdict(cfg), indent=2, default=str))
    job.run()

if __name__ == "__main__":
    main()
