from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from collections import deque
import atexit, time
import pandas as pd

try:
    from ibx import IBRuntime, ContractDetailsService, SecDefService, HistoricalService  # type: ignore
    from ibx import make_stock, make_option  # type: ignore
except Exception:
    from ibx.runtime import IBRuntime  # type: ignore
    from ibx.services import ContractDetailsService, SecDefService, HistoricalService  # type: ignore
    from ibx.contracts import make_stock, make_option  # type: ignore
from config import IB_HOST, IB_PORT, IB_CLIENT_IDS
from ibx_time import ib_end_datetime_instrument, parse_ib_datetime_series

def _duration_for(start: date, end: date, bar_size: str) -> str:
    days = max(1, (end - start).days + 1)
    if bar_size == "1 day":
        return f"{days} D" if days <= 365 else f"{days//30} M"
    return f"{days} D"

def _canon_bars(df: pd.DataFrame, bar_size: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["time","open","high","low","close","volume","wap","bar_count"])
    colmap = {c.lower(): c for c in df.columns}
    rename = {}
    if "date" in colmap: rename[colmap["date"]] = "time"
    if "barcount" in colmap: rename[colmap["barcount"]] = "bar_count"
    if "wap" in colmap and colmap["wap"] != "wap": rename[colmap["wap"]] = "wap"
    out = df.rename(columns=rename, errors="ignore").copy()
    if "time" in out.columns: out["time"] = parse_ib_datetime_series(out["time"])
    for k in ["open","high","low","close","volume","wap"]:
        if k in out.columns: out[k] = pd.to_numeric(out[k], errors="coerce")
    return out

class _Pacer:
    def __init__(self, max_per_min: int = 40):
        self.max = max_per_min; self.q = deque()
    def wait(self):
        now = time.monotonic(); self.q.append(now)
        while self.q and now - self.q[0] > 60.0: self.q.popleft()
        if len(self.q) > self.max:
            time.sleep(max(0.01, 60.0 - (now - self.q[0])))

@dataclass
class IBSource:
    host: str = IB_HOST
    port: int = IB_PORT
    client_id: int = IB_CLIENT_IDS["ib_source"]
    max_hist_per_min: int = 40

    def __post_init__(self):
        self.rt = IBRuntime(self.host, self.port, self.client_id); self.rt.start(); atexit.register(self.rt.stop)
        self.hist = HistoricalService(self.rt); self.cds = ContractDetailsService(self.rt)
        try: self.secdef = SecDefService(self.rt)
        except Exception: self.secdef = None
        self._pacer = _Pacer(self.max_hist_per_min)

    def get_equity_bars(self, symbol: str, start: date, end: date, bar_size: str="1 day") -> pd.DataFrame:
        stk = make_stock(symbol)
        duration = _duration_for(start, end, bar_size)
        end_dt = datetime.combine(end, datetime.min.time())
        # FIXED: Remove hyphen=True - IB API requires space separator for endDateTime
        end_str = ib_end_datetime_instrument(self.rt, stk, end_dt)
        self._pacer.wait()
        rows = self.hist.bars(stk, endDateTime=end_str, durationStr=duration, barSizeSetting=bar_size, whatToShow="TRADES", useRTH=0, timeout=90.0)
        df = _canon_bars(pd.DataFrame(rows), bar_size)
        if df.empty: return df
        df.insert(0, "symbol", symbol); df.insert(1, "bar_size", bar_size)
        return df[["symbol","bar_size","time","open","high","low","close","volume"] + (["bar_count"] if "bar_count" in df.columns else [])]

    def get_option_chain(self, underlying: str, asof: date) -> pd.DataFrame:
        cds = self.cds.fetch(make_stock(underlying), timeout=20.0)
        rows = []; params = []
        if self.secdef:
            try: params = self.secdef.option_params(underlying, cds[0].contract.conId, sec_type="STK", exchange="SMART", timeout=30.0)
            except Exception: params = []
        for p in params:
            exch = getattr(p,"exchange",None) or (p.get("exchange") if isinstance(p,dict) else None)
            tclass = getattr(p,"tradingClass",None) or (p.get("tradingClass") if isinstance(p,dict) else None)
            mult = getattr(p,"multiplier",None) or (p.get("multiplier") if isinstance(p,dict) else None)
            expirations = getattr(p,"expirations",None) or (p.get("expirations") if isinstance(p,dict) else [])
            strikes = getattr(p,"strikes",None) or (p.get("strikes") if isinstance(p,dict) else [])
            for exp in expirations:
                try: expd = pd.to_datetime(str(exp), errors="coerce").date()
                except Exception: continue
                for right in ("C","P"):
                    for k in strikes:
                        try: kf = float(k)
                        except Exception: continue
                        rows.append({"asof": asof, "underlying": underlying, "expiry": expd, "right": right, "strike": kf, "conid": None, "local_symbol": None, "trading_class": tclass, "exchange": exch, "currency": "USD", "multiplier": int(mult) if mult not in (None,"") else None})
        return pd.DataFrame(rows)

    def get_option_bars(self, contracts: pd.DataFrame, start: date, end: date, bar_size: str="1 day") -> pd.DataFrame:
        if contracts is None or contracts.empty: return pd.DataFrame()
        frames = []; duration = _duration_for(start, end, bar_size)
        end_dt = datetime.combine(end, datetime.min.time())
        for _, row in contracts.iterrows():
            expd = pd.to_datetime(row["expiry"]).date(); exp_str = pd.Timestamp(expd).strftime("%Y%m%d")
            opt = make_option(row["underlying"], exp_str, float(row["strike"]), row["right"])
            # FIXED: Remove hyphen=True - IB API requires space separator for endDateTime
            end_str = ib_end_datetime_instrument(self.rt, opt, end_dt)
            self._pacer.wait()
            rows = self.hist.bars(opt, endDateTime=end_str, durationStr=duration, barSizeSetting=bar_size, whatToShow="TRADES", useRTH=0, timeout=90.0)
            df = _canon_bars(pd.DataFrame(rows), bar_size)
            if df.empty: continue
            df.insert(0,"underlying",row["underlying"]); df.insert(1,"expiry",expd); df.insert(2,"right",row["right"]); df.insert(3,"strike",float(row["strike"])); df.insert(4,"conid", row.get("conid", None) if hasattr(row,"get") else None); df.insert(5,"bar_size",bar_size)
            frames.append(df)
        if frames:
            out = pd.concat(frames, ignore_index=True)
            for col in ["wap","trades"]:
                if col not in out.columns: out[col] = pd.NA
            return out
        return pd.DataFrame()
