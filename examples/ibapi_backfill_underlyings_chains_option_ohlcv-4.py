# Project layout (refined)
# ├─ ibx/                     # runtime/types/utilities (requests layer)
# │  └─ types.py              # shared typed records
# ├─ ibx_repos/               # persistence layer (Parquet)
# │  ├─ equity_bars.py        # EquityBarRepository
# │  ├─ option_chains.py      # OptionChainSnapshotRepository
# │  ├─ option_bars.py        # OptionBarRepository
# │  └─ _util.py              # parquet utils
# ├─ ibx_flows/               # orchestration (pure functions)
# │  ├─ source_interface.py   # MarketDataSource protocol
# │  ├─ source_ib.py          # IB adapter (wire to your services)
# │  ├─ select_strikes.py     # choose expiries/strikes from chain
# │  └─ backfill.py           # backfill_* flows that glue source->repo
# └─ jobs/                    # thin CLIs
#    ├─ backfill_equity_bars.py
#    ├─ backfill_option_chains.py
#    ├─ backfill_option_bars.py
#    └─ backfill_chain_then_option_bars.py

# =============================
# ibx/types.py
# =============================
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Literal

Right = Literal["C", "P"]

@dataclass(frozen=True)
class OptionKey:
    underlying: str
    expiry: date
    right: Right
    strike: float

@dataclass(frozen=True)
class ContractMeta:
    conid: int
    local_symbol: str
    trading_class: str
    exchange: str
    currency: str
    multiplier: int
    key: OptionKey

# =============================
# ibx_repos/_util.py
# =============================
from __future__ import annotations
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_table(df: pd.DataFrame, schema: pa.schema | None) -> pa.Table:
    if schema is None:
        return pa.Table.from_pandas(df, preserve_index=False)
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False)


def write_dataset(df: pd.DataFrame, root_path: Path, schema: pa.schema | None, partition_cols: list[str]) -> None:
    _ensure_dir(root_path)
    table = _to_table(df, schema)
    pq.write_to_dataset(
        table,
        root_path=str(root_path),
        partition_cols=partition_cols,
        use_dictionary=True,
        compression="snappy",
    )


def _dataset_hive(root_path: Path) -> ds.Dataset:
    return ds.dataset(str(root_path), format="parquet", partitioning="hive")


def _build_filter(**kwargs):
    """
    Build a pyarrow.dataset filter expression from keyword equality tests.
    Accepts scalars or iterables (treated as isin). For dates, accepts str or date.
    """
    expr = None
    for key, value in kwargs.items():
        if value is None:
            continue
        field = ds.field(key)
        if isinstance(value, (list, tuple, set)):
            term = field.isin(list(value))
        else:
            if key in {"asof","date"}:
                try:
                    value = pd.to_datetime(value).date()
                except Exception:
                    pass
            term = field == value
        expr = term if expr is None else (expr & term)
    return expr


def read_dataset(root_path: Path, columns: list[str] | None = None, **filters) -> pd.DataFrame:
    ds_ = _dataset_hive(root_path)
    expr = _build_filter(**filters)
    tbl = ds_.to_table(filter=expr, columns=columns)
    return tbl.to_pandas() if tbl.num_rows else pd.DataFrame(columns=columns or [])

# =============================
from __future__ import annotations
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_table(df: pd.DataFrame, schema: pa.schema | None) -> pa.Table:
    if schema is None:
        return pa.Table.from_pandas(df, preserve_index=False)
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False)


def write_dataset(df: pd.DataFrame, root_path: Path, schema: pa.schema | None, partition_cols: list[str]) -> None:
    _ensure_dir(root_path)
    table = _to_table(df, schema)
    pq.write_to_dataset(
        table,
        root_path=str(root_path),
        partition_cols=partition_cols,
        use_dictionary=True,
        compression="snappy",
    )

# =============================
# ibx_repos/equity_bars.py
# =============================
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Set
import pandas as pd
import pyarrow as pa

from ._util import write_dataset, read_dataset

BarSize = Literal["1 min", "5 min", "15 min", "1 hour", "1 day"]

@dataclass
class EquityBarRepository:
    base_path: Path

    @property
    def schema(self) -> pa.schema:
        return pa.schema([
            pa.field("symbol", pa.string()),
            pa.field("bar_size", pa.string()),
            pa.field("time", pa.timestamp("ns")),  # bar end time (UTC)
            pa.field("date", pa.date32()),
            pa.field("open", pa.float64()),
            pa.field("high", pa.float64()),
            pa.field("low", pa.float64()),
            pa.field("close", pa.float64()),
            pa.field("volume", pa.int64()),
            pa.field("trade_count", pa.int64()).with_nullable(True),
        ])

    def save(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("EquityBarRepository.save expects a pandas DataFrame")
        if df.empty:
            return
        df = df.copy()
        req = {"symbol","bar_size","time","open","high","low","close","volume"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"EquityBarRepository.save missing columns: {sorted(missing)}")
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df["date"] = df["time"].dt.date
        # de-dupe before write
        df = df.drop_duplicates(subset=["symbol","bar_size","time"])  
        is_daily = (df["bar_size"].astype(str) == "1 day").all()
        if is_daily:
            df["year"] = pd.to_datetime(df["time"]).dt.year
            df["month"] = pd.to_datetime(df["time"]).dt.month
            partition_cols = ["bar_size","symbol","year","month"]
        else:
            partition_cols = ["bar_size","symbol","date"]
        write_dataset(df, self.base_path, self.schema, partition_cols)

    def load(self, symbol: str, bar_size: str, start: Optional[str | pd.Timestamp] = None, end: Optional[str | pd.Timestamp] = None) -> pd.DataFrame:
        filt = {"symbol": symbol, "bar_size": bar_size}
        df = read_dataset(self.base_path, **filt)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"], utc=True)
        if start is not None:
            df = df[df["time"] >= pd.to_datetime(start, utc=True)]
        if end is not None:
            df = df[df["time"] <= pd.to_datetime(end, utc=True)]
        return df

    def present_dates(self, symbol: str, bar_size: str, start: pd.Timestamp, end: pd.Timestamp) -> Set[pd.Timestamp]:
        """Return set of trade dates already stored for the window."""
        df = read_dataset(self.base_path, columns=["date"], symbol=symbol, bar_size=bar_size)
        if df.empty:
            return set()
        s = pd.to_datetime(df["date"])  # naive date
        mask = (s >= pd.to_datetime(start).normalize()) & (s <= pd.to_datetime(end).normalize())
        return set(pd.to_datetime(s[mask]).dt.date)

# =============================
# ibx_repos/option_chains.py
# =============================
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd
import pyarrow as pa

from ._util import write_dataset, read_dataset

@dataclass
class OptionChainSnapshotRepository:
    base_path: Path

    @property
    def schema(self) -> pa.schema:
        return pa.schema([
            pa.field("asof", pa.date32()),
            pa.field("underlying", pa.string()),
            pa.field("expiry", pa.date32()),
            pa.field("right", pa.string()),
            pa.field("strike", pa.float64()),
            pa.field("conid", pa.int64()).with_nullable(True),
            pa.field("local_symbol", pa.string()).with_nullable(True),
            pa.field("trading_class", pa.string()).with_nullable(True),
            pa.field("exchange", pa.string()).with_nullable(True),
            pa.field("currency", pa.string()).with_nullable(True),
            pa.field("multiplier", pa.int32()).with_nullable(True),
        ])

    def save(self, df: pd.DataFrame) -> None:
        req = {"asof","underlying","expiry","right","strike"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"OptionChainSnapshotRepository.save missing columns: {sorted(missing)}")
        df = df.copy()
        df["asof"] = pd.to_datetime(df["asof"]).dt.date
        write_dataset(df, self.base_path, self.schema, partition_cols=["underlying","asof"])

    def load(self, underlying: str, asof_start: Optional[str | pd.Timestamp] = None, asof_end: Optional[str | pd.Timestamp] = None) -> pd.DataFrame:
        df = read_dataset(self.base_path, underlying=underlying)
        if df.empty:
            return df
        df["asof"] = pd.to_datetime(df["asof"]).dt.date
        if asof_start is not None:
            df = df[df["asof"] >= pd.to_datetime(asof_start).date()]
        if asof_end is not None:
            df = df[df["asof"] <= pd.to_datetime(asof_end).date()]
        return df

# =============================
# ibx_repos/option_bars.py
# =============================
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Set
import pandas as pd
import pyarrow as pa

from ._util import write_dataset, read_dataset

OptBarSize = Literal["1 day", "1 hour", "15 min", "5 min", "1 min"]

@dataclass
class OptionBarRepository:
    base_path: Path

    @property
    def schema(self) -> pa.schema:
        return pa.schema([
            pa.field("underlying", pa.string()),
            pa.field("expiry", pa.date32()),
            pa.field("right", pa.string()),
            pa.field("strike", pa.float64()),
            pa.field("conid", pa.int64()).with_nullable(True),
            pa.field("bar_size", pa.string()),
            pa.field("time", pa.timestamp("ns")),
            pa.field("date", pa.date32()),
            pa.field("open", pa.float64()),
            pa.field("high", pa.float64()),
            pa.field("low", pa.float64()),
            pa.field("close", pa.float64()),
            pa.field("volume", pa.int64()),
            pa.field("wap", pa.float64()).with_nullable(True),
            pa.field("trades", pa.int64()).with_nullable(True),
        ])

    def save(self, df: pd.DataFrame) -> None:
        req = {"underlying","expiry","right","strike","bar_size","time","open","high","low","close","volume"}
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"OptionBarRepository.save missing columns: {sorted(missing)}")
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df["date"] = df["time"].dt.date
        # de-dupe before write
        df = df.drop_duplicates(subset=["underlying","expiry","right","strike","bar_size","time"])  
        is_daily = (df["bar_size"].astype(str) == "1 day").all()
        if is_daily:
            df["year"] = pd.to_datetime(df["time"]).dt.year
            df["month"] = pd.to_datetime(df["time"]).dt.month
            partition_cols = ["underlying","expiry","right","strike","bar_size","year","month"]
        else:
            partition_cols = ["underlying","expiry","right","strike","bar_size","date"]
        write_dataset(df, self.base_path, self.schema, partition_cols)

    def load(self,
             underlying: str,
             bar_size: str,
             expiry: Optional[str | pd.Timestamp] = None,
             right: Optional[str] = None,
             strike: Optional[float] = None,
             start: Optional[str | pd.Timestamp] = None,
             end: Optional[str | pd.Timestamp] = None) -> pd.DataFrame:
        filters = {"underlying": underlying, "bar_size": bar_size}
        if expiry is not None:
            filters["expiry"] = pd.to_datetime(expiry).date()
        if right is not None:
            filters["right"] = right
        if strike is not None:
            filters["strike"] = float(strike)
        df = read_dataset(self.base_path, **filters)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"], utc=True)
        if start is not None:
            df = df[df["time"] >= pd.to_datetime(start, utc=True)]
        if end is not None:
            df = df[df["time"] <= pd.to_datetime(end, utc=True)]
        return df

    def present_dates_for_contract(self,
                                   underlying: str,
                                   expiry: pd.Timestamp | str,
                                   right: str,
                                   strike: float,
                                   bar_size: str,
                                   start: pd.Timestamp,
                                   end: pd.Timestamp) -> Set[pd.Timestamp]:
        filters = {
            "underlying": underlying,
            "expiry": pd.to_datetime(expiry).date(),
            "right": right,
            "strike": float(strike),
            "bar_size": bar_size,
        }
        df = read_dataset(self.base_path, columns=["date"], **filters)
        if df.empty:
            return set()
        s = pd.to_datetime(df["date"])  # naive date
        mask = (s >= pd.to_datetime(start).normalize()) & (s <= pd.to_datetime(end).normalize())
        return set(pd.to_datetime(s[mask]).dt.date)

# =============================
# ibx_flows/source_interface.py
# =============================
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Protocol
import pandas as pd

class MarketDataSource(Protocol):
    def get_equity_bars(self, symbol: str, start: date, end: date, bar_size: str = "1 day") -> pd.DataFrame: ...
    def get_option_chain(self, underlying: str, asof: date) -> pd.DataFrame: ...
    def get_option_bars(self, contracts: pd.DataFrame, start: date, end: date, bar_size: str = "1 day") -> pd.DataFrame: ...

# =============================
# ibx_flows/source_ib.py (adapter – wired to your services)
# =============================
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Optional
import atexit
import time
from collections import deque
import pandas as pd

# Prefer the packaged ibx runtime/services; fall back to local modules if needed
try:
    from ibx import IBRuntime  # type: ignore
    from ibx import ContractDetailsService, SecDefService, HistoricalService  # type: ignore
except Exception:
    from runtime import IBRuntime  # type: ignore
    from services import ContractDetailsService, SecDefService, HistoricalService  # type: ignore

try:
    # prefer project-level helpers if exposed
    from ibx import make_stock, make_option  # type: ignore
except Exception:
    from contracts import make_stock, make_option  # type: ignore

# Optional: access to registry/future if your wrapper uses them for symbolSamples
try:
    from ibx import IBFuture  # type: ignore
except Exception:
    try:
        from futures import IBFuture  # type: ignore
    except Exception:
        IBFuture = None  # will fall back to ContractDetails-only enumeration if needed


# -------- helpers --------

def _duration_for(start: date, end: date, bar_size: str) -> str:
    days = max(1, (end - start).days + 1)
    if bar_size == "1 day":
        return f"{days} D" if days <= 365 else f"{days//30} M"
    return f"{days} D"


def _parse_ib_datetime(s: object) -> pd.Timestamp:
    if isinstance(s, (int, float)):
        try:
            return pd.to_datetime(int(s), unit="s", utc=True)
        except Exception:
            pass
    try:
        return pd.to_datetime(s, utc=True)
    except Exception:
        return pd.NaT


def _canon_bars(df: pd.DataFrame, bar_size: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["time","open","high","low","close","volume","wap","bar_count"])  # type: ignore
    colmap = {c.lower(): c for c in df.columns}
    rename = {}
    if "date" in colmap:
        rename[colmap["date"]] = "time"
    if "barcount" in colmap:
        rename[colmap["barcount"]] = "bar_count"
    if "wap" in colmap and colmap["wap"] != "wap":
        rename[colmap["wap"]] = "wap"
    out = df.rename(columns=rename, errors="ignore").copy()
    if "time" in out.columns:
        out["time"] = out["time"].map(_parse_ib_datetime)
    for k in ["open","high","low","close","volume","wap"]:
        if k in out.columns:
            out[k] = pd.to_numeric(out[k], errors="coerce")
    return out


class _Pacer:
    def __init__(self, max_per_min: int = 40):
        self.max = max_per_min
        self.q: deque[float] = deque()
    def wait(self):
        now = time.monotonic()
        self.q.append(now)
        while self.q and now - self.q[0] > 60.0:
            self.q.popleft()
        if len(self.q) > self.max:
            to_sleep = 60.0 - (now - self.q[0])
            time.sleep(max(0.01, to_sleep))


# -------- Adapter --------

class IBSource:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 101
    max_hist_per_min: int = 40

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 101, max_hist_per_min: int = 40):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.max_hist_per_min = max_hist_per_min
        self.rt = IBRuntime(self.host, self.port, self.client_id)
        self.rt.start()
        atexit.register(self.rt.stop)
        self.hist = HistoricalService(self.rt)
        self.cds = ContractDetailsService(self.rt)
        self.secdef = SecDefService(self.rt)
        self._pacer = _Pacer(self.max_hist_per_min)

    # ---- Equity ----
    def get_equity_bars(self, symbol: str, start: date, end: date, bar_size: str = "1 day") -> pd.DataFrame:
        stk = make_stock(symbol)
        duration = _duration_for(start, end, bar_size)
        end_dt = datetime.combine(end, datetime.min.time())
        end_str = end_dt.strftime("%Y%m%d %H:%M:%S")
        self._pacer.wait()
        rows = self.hist.bars(
            stk,
            endDateTime=end_str,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=0,
            timeout=90.0,
        )
        df = pd.DataFrame(rows)
        df = _canon_bars(df, bar_size)
        if df.empty:
            return df
        df.insert(0, "symbol", symbol)
        df.insert(1, "bar_size", bar_size)
        return df[["symbol","bar_size","time","open","high","low","close","volume"] + (["bar_count"] if "bar_count" in df.columns else [])]

    # ---- Chain snapshot ----
    def get_option_chain(self, underlying: str, asof: date) -> pd.DataFrame:
        cds = self.cds.fetch(make_stock(underlying), timeout=20.0)
        params = self.secdef.option_params(underlying, cds[0].contract.conId, sec_type="STK", exchange="SMART", timeout=30.0)
        rows = []
        for p in params:
            exch = getattr(p, "exchange", None) or (p.get("exchange") if isinstance(p, dict) else None)
            tclass = getattr(p, "tradingClass", None) or (p.get("tradingClass") if isinstance(p, dict) else None)
            mult = getattr(p, "multiplier", None) or (p.get("multiplier") if isinstance(p, dict) else None)
            expirations = getattr(p, "expirations", None) or (p.get("expirations") if isinstance(p, dict) else [])
            strikes = getattr(p, "strikes", None) or (p.get("strikes") if isinstance(p, dict) else [])
            for exp in expirations:
                try:
                    expd = pd.to_datetime(str(exp), errors="coerce").date()
                except Exception:
                    continue
                for right in ("C","P"):
                    for k in strikes:
                        try:
                            kf = float(k)
                        except Exception:
                            continue
                        rows.append({
                            "asof": asof,
                            "underlying": underlying,
                            "expiry": expd,
                            "right": right,
                            "strike": kf,
                            "conid": None,
                            "local_symbol": None,
                            "trading_class": tclass,
                            "exchange": exch,
                            "currency": "USD",
                            "multiplier": int(mult) if mult not in (None, "") else None,
                        })
        return pd.DataFrame(rows)

    # ---- Option bars ----
    def get_option_bars(self, contracts: pd.DataFrame, start: date, end: date, bar_size: str = "1 day") -> pd.DataFrame:
        if contracts is None or contracts.empty:
            return pd.DataFrame()
        frames = []
        duration = _duration_for(start, end, bar_size)
        end_dt = datetime.combine(end, datetime.min.time())
        end_str = end_dt.strftime("%Y%m%d %H:%M:%S")
        for _, row in contracts.iterrows():
            exp = row["expiry"]
            if not isinstance(exp, (date, datetime)):
                exp = pd.to_datetime(exp).date()
            exp_str = pd.Timestamp(exp).strftime("%Y%m%d")
            opt = make_option(row["underlying"], exp_str, float(row["strike"]), row["right"])  # conid left to resolver
            self._pacer.wait()
            rows = self.hist.bars(
                opt,
                endDateTime=end_str,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=0,
                timeout=90.0,
            )
            df = _canon_bars(pd.DataFrame(rows), bar_size)
            if df.empty:
                continue
            df.insert(0, "underlying", row["underlying"])
            df.insert(1, "expiry", pd.to_datetime(exp).date())
            df.insert(2, "right", row["right"])
            df.insert(3, "strike", float(row["strike"]))
            df.insert(4, "conid", row.get("conid", None))
            df.insert(5, "bar_size", bar_size)
            frames.append(df)
        if frames:
            out = pd.concat(frames, ignore_index=True)
            for col in ["wap","trades"]:
                if col not in out.columns:
                    out[col] = pd.NA
            return out
        return pd.DataFrame()

    # ---- Symbol search (reqMatchingSymbols) ----
    def search_symbols(self, pattern: str, timeout: float = 30.0) -> pd.DataFrame:
        """Return DataFrame of matching symbol samples for the given pattern.
        Requires wrapper to forward symbolSamples -> registry.resolve(reqId, data).
        """
        if IBFuture is None:
            # Without a Future/registry bridge, we cannot call reqMatchingSymbols here.
            raise RuntimeError("IBFuture/registry not available for reqMatchingSymbols; add wrapper support or provide external symbol list.")
        req_id = self.rt.next_request_id()
        fut = self.rt.registry.register(req_id, IBFuture(expect_many=False, timeout=timeout))
        self.rt.client.reqMatchingSymbols(req_id, pattern)
        data = fut.result()
        # Normalize into a DataFrame
        rows = []
        try:
            # contractDescriptions: list of ContractDescription
            for d in data:
                c = getattr(d, "contract", None)
                if c is None and isinstance(d, dict):
                    c = d.get("contract")
                if c is None:
                    continue
                rows.append({
                    "symbol": getattr(c, "symbol", None),
                    "conid": getattr(c, "conId", None),
                    "secType": getattr(c, "secType", None),
                    "currency": getattr(c, "currency", None),
                    "primaryExchange": getattrine(end, datetime.min.time())
        end_str = end_dt.strftime("%Y%m%d %H:%M:%S")
        for _, row in contracts.iterrows():
            exp = row["expiry"]
            if not isinstance(exp, (date, datetime)):
                exp = pd.to_datetime(exp).date()
            exp_str = pd.Timestamp(exp).strftime("%Y%m%d")
            opt = make_option(row["underlying"], exp_str, float(row["strike"]), row["right"])  # conid left to resolver
            self._pacer.wait()
            rows = self.hist.bars(
                opt,
                endDateTime=end_str,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=0,
                timeout=90.0,
            )
            df = _canon_bars(pd.DataFrame(rows), bar_size)
            if df.empty:
                continue
            df.insert(0, "underlying", row["underlying"])
            df.insert(1, "expiry", pd.to_datetime(exp).date())
            df.insert(2, "right", row["right"])
            df.insert(3, "strike", float(row["strike"]))
            df.insert(4, "conid", row.get("conid", None))
            df.insert(5, "bar_size", bar_size)
            frames.append(df)
        if frames:
            out = pd.concat(frames, ignore_index=True)
            for col in ["wap","trades"]:
                if col not in out.columns:
                    out[col] = pd.NA
            return out
        return pd.DataFrame()

    # ---- Symbol search (reqMatchingSymbols) ----
    def search_symbols(self, pattern: str, timeout: float = 30.0) -> pd.DataFrame:
        """Return DataFrame of matching symbol samples for the given pattern.
        Requires wrapper to forward symbolSamples -> registry.resolve(reqId, data).
        """
        if IBFuture is None:
            # Without a Future/registry bridge, we cannot call reqMatchingSymbols heremport annotations
from datetime import date
from typing import List, Tuple
import pandas as pd


def missing_windows(present_dates: set, start: date, end: date) -> List[Tuple[date, date]]:
    """Return list of contiguous [start,end] business-day windows that are missing."""
    desired = pd.bdate_range(start, end).date
    missing = [d for d in desired if d not in present_dates]
    if not missing:
        return []
    windows: List[Tuple[date, date]] = []
    s = missing[0]
    prev = s
    for d in missing[1:]:
        if (pd.Timestamp(d) - pd.Timestamp(prev)).days == 1:
            prev = d
        else:
            windows.append((s, prev))
            s = d
            prev = d
    windows.append((s, prev))
    return windows

# =============================
# ibx_flows/backfill.py
# =============================
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Optional, List, Tuple
import pandas as pd

from ibx_repos.equity_bars import EquityBarRepository
from ibx_repos.option_chains import OptionChainSnapshotRepository
from ibx_repos.option_bars import OptionBarRepository
from .source_interface import MarketDataSource
from .select_strikes import StrikeSelectionConfig, select_contracts
from .windows import missing_windows

@dataclass
class BackfillConfig:
    underlying: str
    start: date
    end: date
    bar_size: str = "1 day"
    expiries_max: int = 8
    strikes_per_side: int = 6
    selection_mode: str = "k_around_atm"  # or "moneyness_bands"



def backfill_equity_bars(src: MarketDataSource, repo: EquityBarRepository, cfg: BackfillConfig) -> None:
    # Find existing coverage
    present = repo.present_dates(cfg.underlying, cfg.bar_size, pd.to_datetime(cfg.start), pd.to_datetime(cfg.end))
    windows = missing_windows(present, cfg.start, cfg.end)
    if not windows:
        return
    frames = []
    for ws, we in windows:
        df = src.get_equity_bars(cfg.underlying, ws, we, cfg.bar_size)
        if not df.empty:
            frames.append(df)
    if frames:
        repo.save(pd.concat(frames, ignore_index=True))


def backfill_option_chain_daily(src: MarketDataSource, repo: OptionChainSnapshotRepository, cfg: BackfillConfig) -> None:
    asofs = pd.bdate_range(cfg.start, cfg.end).date
    frames = []
    for d in asofs:
        try:
            snap = src.get_option_chain(cfg.underlying, d)
        except Exception:
            continue
        frames.append(snap)
    if frames:
        repo.save(pd.concat(frames, ignore_index=True))


def backfill_option_bars(
    src: MarketDataSource,
    repo: OptionBarRepository,
    chain_repo: OptionChainSnapshotRepository | None,
    eq_repo: EquityBarRepository | None,
    cfg: BackfillConfig,
) -> None:
    # Build universe using chain at END date
    chain = src.get_option_chain(cfg.underlying, cfg.end)

    # Spot from equity bars at END
    eq = src.get_equity_bars(cfg.underlying, cfg.end, cfg.end, cfg.bar_size)
    if eq.empty:
        raise RuntimeError("No equity bars for spot selection on end date")
    spot = float(eq.iloc[-1]["close"])  # last close

    filtered = select_contracts(
        chain_snapshot=chain,
        spot_price=spot,
        cfg=StrikeSelectionConfig(expiries_max=cfg.expiries_max, strikes_per_side=cfg.strikes_per_side),
        mode=cfg.selection_mode,
    )

    frames = []
    # For each contract, compute missing windows and fetch only gaps
    for _, row in filtered.iterrows():
        present = repo.present_dates_for_contract(
            underlying=row["underlying"],
            expiry=row["expiry"],
            right=row["right"],
            strike=float(row["strike"]),
            bar_size=cfg.bar_size,
            start=pd.to_datetime(cfg.start),
            end=pd.to_datetime(cfg.end),
        )
        windows = missing_windows(present, cfg.start, cfg.end)
        if not windows:
            continue
        # Build a tiny DF for this contract for get_option_bars
        contract_df = pd.DataFrame([{k: row[k] for k in ["underlying","expiry","right","strike","conid"] if k in row}])
        for ws, we in windows:
            bars = src.get_option_bars(contract_df, ws, we, cfg.bar_size)
            if not bars.empty:
                frames.append(bars)

    if frames:
        repo.save(pd.concat(frames, ignore_index=True))

# =============================
# jobs/check_coverage.py
# =============================
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from ibx_repos.equity_bars import EquityBarRepository
from ibx_repos.option_bars import OptionBarRepository
from ibx_repos.option_chains import OptionChainSnapshotRepository
from ibx_flows.windows import missing_windows

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dry-run coverage check (no IB calls)")
    sub = p.add_subparsers(dest="mode", required=True)

    pe = sub.add_parser("equity")
    pe.add_argument("symbol")
    pe.add_argument("start")
    pe.add_argument("end")
    pe.add_argument("--bar", default="1 day")
    pe.add_argument("--root", type=Path, default=Path("data/equity_bars"))

    po = sub.add_parser("options")
    po.add_argument("symbol")
    po.add_argument("start")
    po.add_argument("end")
    po.add_argument("--bar", default="1 day")
    po.add_argument("--root", type=Path, default=Path("data/option_bars"))
    po.add_argument("--from-chain-root", type=Path, default=None, help="if set, use chain as-of end date to enumerate contracts")
    po.add_argument("--expiry")
    po.add_argument("--right", choices=["C","P"])
    po.add_argument("--strike", type=float)

    args = p.parse_args()

    if args.mode == "equity":
        repo = EquityBarRepository(args.root)
        present = repo.present_dates(args.symbol, args.bar, pd.to_datetime(args.start), pd.to_datetime(args.end))
        gaps = missing_windows(present, pd.to_datetime(args.start).date(), pd.to_datetime(args.end).date())
        print({"symbol": args.symbol, "bar_size": args.bar, "gaps": gaps, "present_days": len(present)})

    else:
        repo = OptionBarRepository(args.root)
        start = pd.to_datetime(args.start).date()
        end = pd.to_datetime(args.end).date()

        contracts = []
        if args.from_chain_root:
            # enumerate contracts from chain snapshot as-of end date (no IB calls)
            chain_repo = OptionChainSnapshotRepository(args.from_chain_root)
            chain = chain_repo.load(args.symbol, asof_start=args.end, asof_end=args.end)
            if args.expiry: chain = chain[chain["expiry"] == pd.to_datetime(args.expiry).date()]
            if args.right:  chain = chain["right"].where(chain["right"] == args.right).dropna()
            if args.strike is not None:
                chain = chain[abs(chain["strike"].astype(float) - float(args.strike)) < 1e-9]
            if hasattr(chain, 'to_dict'):
                contracts = chain[["underlying","expiry","right","strike"]].drop_duplicates().to_dict("records")
        else:
            # enumerate from bars that already exist
            df = repo.load(args.symbol, args.bar)
            if not df.empty:
                if args.expiry: df = df[df["expiry"] == pd.to_datetime(args.expiry).date()]
                if args.right:  df = df[df["right"] == args.right]
                if args.strike is not None: df = df[abs(df["strike"].astype(float) - float(args.strike)) < 1e-9]
                contracts = df[["underlying","expiry","right","strike"]].drop_duplicates().to_dict("records")

        out = []
        for c in contracts:
            present = repo.present_dates_for_contract(
                underlying=c["underlying"],
                expiry=c["expiry"],
                right=c["right"],
                strike=float(c["strike"]),
                bar_size=args.bar,
                start=pd.to_datetime(args.start),
                end=pd.to_datetime(args.end),
            )
            gaps = missing_windows(present, start, end)
            out.append({**c, "gaps": gaps, "present_days": len(present)})
        print(out)

# =============================
# ibx_repos/contracts.py
# =============================
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set
import pandas as pd
import pyarrow as pa

from ._util import write_dataset, read_dataset

@dataclass
class ContractRepository:
    base_path: Path

    @property
    def schema(self) -> pa.schema:
        return pa.schema([
            pa.field("conid", pa.int64()),
            pa.field("symbol", pa.string()),
            pa.field("sec_type", pa.string()),
            pa.field("currency", pa.string()),
            pa.field("primary_exchange", pa.string()).with_nullable(True),
            pa.field("exchange", pa.string()).with_nullable(True),
            pa.field("local_symbol", pa.string()).with_nullable(True),
            pa.field("trading_class", pa.string()).with_nullable(True),
            pa.field("long_name", pa.string()).with_nullable(True),
            pa.field("category", pa.string()).with_nullable(True),
            pa.field("sub_category", pa.string()).with_nullable(True),
            pa.field("industry", pa.string()).with_nullable(True),
            pa.field("time_zone", pa.string()).with_nullable(True),
            pa.field("is_us_listing", pa.bool_()).with_nullable(True),
        ])

    def _with_partitions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # symbol prefix for sharding
        prefix = df["symbol"].astype(str).str[0].str.upper().fillna("")
        prefix = prefix.where(prefix.str.match(r"[A-Z]"), "#")
        df["sym_prefix"] = prefix
        return df

    def save(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        df = df.copy()
        req = {"conid","symbol","sec_type","currency"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"ContractRepository.save missing columns: {sorted(missing)}")
        # de-dupe by conid
        df = df.drop_duplicates(subset=["conid"])  
        df = self._with_partitions(df)
        write_dataset(df, self.base_path, self.schema, partition_cols=["sec_type","is_us_listing","primary_exchange","sym_prefix"])

    def load(self, sec_type: Optional[str] = None, primary_exchange: Optional[str] = None) -> pd.DataFrame:
        filters = {}
        if sec_type: filters["sec_type"] = sec_type
        if primary_exchange: filters["primary_exchange"] = primary_exchange
        df = read_dataset(self.base_path, **filters)
        return df

    def present_conids(self) -> Set[int]:
        df = read_dataset(self.base_path, columns=["conid"])
        return set(df["conid"].astype(int)) if not df.empty else set()

# =============================
# ibx_flows/contracts_backfill.py
# =============================
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Set
import time
import pandas as pd

try:
    from ibx import ContractDetailsService  # type: ignore
except Exception:
    from services import ContractDetailsService  # type: ignore

try:
    from ibx import IBRuntime  # type: ignore
except Exception:
    from runtime import IBRuntime  # type: ignore

try:
    from ibx import make_stock  # type: ignore
except Exception:
    from contracts import make_stock  # type: ignore

from ibx_repos.contracts import ContractRepository
from ibx_flows.symbol_search_ib import IBSymbolSearch


@dataclass
class ContractsConfig:
    exchanges: List[str] = None
    currency: str = "USD"
    sec_types: List[str] = None
    patterns: List[str] = None  # symbol search patterns

    def __post_init__(self):
        if self.exchanges is None:
            self.exchanges = ["NYSE","NASDAQ","AMEX","ARCA","BATS","IEX"]
        if self.sec_types is None:
            self.sec_types = ["STK"]
        if self.patterns is None:
            self.patterns = [chr(c) for c in range(ord('A'), ord('Z')+1)] + [str(d) for d in range(10)]


def _from_contract_details(cd) -> dict:
    c = cd.contract
    row = {
        "conid": int(getattr(c, "conId", 0) or 0),
        "symbol": getattr(c, "symbol", None),
        "sec_type": getattr(c, "secType", None),
        "currency": getattr(c, "currency", None),
        "primary_exchange": getattr(c, "primaryExchange", None),
        "exchange": getattr(c, "exchange", None),
        "local_symbol": getattr(c, "localSymbol", None),
        "trading_class": getattr(c, "tradingClass", None),
        "long_name": getattr(cd, "longName", None),
        "category": getattr(cd, "category", None),
        "sub_category": getattr(cd, "subcategory", None) if hasattr(cd, "subcategory") else getattr(cd, "subCategory", None),
        "industry": getattr(cd, "industry", None),
        "time_zone": getattr(cd, "timeZoneId", None),
    }
    row["is_us_listing"] = (row.get("currency") == "USD") and (row.get("primary_exchange") in {"NYSE","NASDAQ","AMEX","ARCA","BATS","IEX"})
    return row


def backfill_us_equity_contracts(rt: IBRuntime, repo: ContractRepository, cfg: ContractsConfig) -> None:
    existing: Set[int] = repo.present_conids()
    to_save_rows: List[dict] = []

    search = IBSymbolSearch(rt)

    for pat in cfg.patterns:
        matches = search.search(pat)
        if matches is None or matches.empty:
            continue
        cols = {c.lower(): c for c in matches.columns}
        def get(col):
            col_l = col.lower()
            return matches[cols[col_l]] if col_l in cols else pd.Series([None]*len(matches))

        df = pd.DataFrame({
            "symbol": get("symbol"),
            "conid": pd.to_numeric(get("conid"), errors="coerce"),
            "sec_type": get("secType").astype(str),
            "currency": get("currency").astype(str),
            "primary_exchange": get("primaryExchange").astype(str),
        })
        df = df[df["sec_type"].isin(cfg.sec_types)]
        df = df[df["currency"] == cfg.currency]
        df = df[df["primary_exchange"].isin(cfg.exchanges)]
        df = df.dropna(subset=["conid"]).drop_duplicates(subset=["conid"])
        df = df[~df["conid"].astype(int).isin(existing)]
        if df.empty:
            continue

        cds = ContractDetailsService(rt)
        for conid in df["conid"].astype(int).tolist():
            c = make_stock("?")
            c.conId = int(conid)
            try:
                details = cds.fetch(c, timeout=20.0)
            except Exception:
                continue
            if not details:
                continue
            row = _from_contract_details(details[0])
            if row.get("conid") and row["conid"] not in existing:
                to_save_rows.append(row)
                existing.add(row["conid"])  # avoid re-fetch

    if to_save_rows:
        repo.save(pd.DataFrame(to_save_rows))
(src: IBSource, repo: ContractRepository, cfg: ContractsConfig) -> None:
    existing: Set[int] = repo.present_conids()
    to_save_rows: List[dict] = []


# =============================
# ibx_flows/symbol_search_ib.py
# =============================
from __future__ import annotations
import pandas as pd

try:
    from ibx import IBFuture  # type: ignore
except Exception:
    from futures import IBFuture  # type: ignore

class IBSymbolSearch:
    """Thin helper to call reqMatchingSymbols and return a DataFrame.
    Requires IBWrapperBridge.symbolSamples to resolve registry futures.
    """
    def __init__(self, rt):
        self.rt = rt

    def search(self, pattern: str, timeout: float = 30.0) -> pd.DataFrame:
        rid = self.rt.sequencer.next()
        fut = self.rt.registry.register(rid, IBFuture(expect_many=False, timeout=timeout))
        self.rt.client.reqMatchingSymbols(rid, pattern)
        data = fut.result()
        rows = []
        for d in data or []:
            c = getattr(d, "contract", None)
            if c is None and isinstance(d, dict):
                c = d.get("contract")
            if c is None:
                continue
            rows.append({
                "symbol": getattr(c, "symbol", None),
                "conid": getattr(c, "conId", None),
                "secType": getattr(c, "secType", None),
                "currency": getattr(c, "currency", None),
                "primaryExchange": getattr(c, "primaryExchange", None),
            })
        return pd.DataFrame(rows)

# =============================
# jobs/backfill_contracts_us_rt.py
# =============================
from __future__ import annotations
import argparse
from pathlib import Path

from ibx_repos.contracts import ContractRepository
from ibx_flows.contracts_backfill import ContractsConfig, backfill_us_equity_contracts

try:
    from ibx import IBRuntime  # type: ignore
except Exception:
    from runtime import IBRuntime  # type: ignore

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/contracts"))
    p.add_argument("--exchanges", nargs="*", default=["NYSE","NASDAQ","AMEX","ARCA","BATS","IEX"]) 
    p.add_argument("--currency", default="USD")
    p.add_argument("--patterns", nargs="*", default=None)
    args = p.parse_args()

    cfg = ContractsConfig(exchanges=args.exchanges, currency=args.currency, patterns=args.patterns)
    repo = ContractRepository(args.root)

    with IBRuntime() as rt:
        backfill_us_equity_contracts(rt, repo, cfg)
