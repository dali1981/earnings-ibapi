# ibx_flows/greeks_historical.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Literal, Dict
import pandas as pd
from ibx_pricing.black_scholes import implied_vol_from_price, delta, gamma, vega, theta, rho, Right
from ibx_time.timebox import ensure_utc

@dataclass(frozen=True)
class OptionMeta:
    underlying: str
    expiry: str     # YYYYMMDD
    right: Right
    strike: float

@dataclass(frozen=True)
class AsOfInputs:
    ts: datetime           # tz-aware UTC
    S: float               # underlying spot at ts
    opt_price: float       # option mid/close at ts
    r: float               # cont comp rate
    q: float               # cont dividend yield

def year_fraction(ts: datetime, expiry_yyyymmdd: str, daycount: Literal[\"ACT/365\",\"ACT/252\"]=\"ACT/365\") -> float:
    ts = ensure_utc(ts)
    expiry = datetime.strptime(expiry_yyyymmdd, \"%Y%m%d\").replace(tzinfo=timezone.utc)
    days = (expiry - ts).total_seconds() / (24*3600)
    if days <= 0: 
        return 1e-9
    if daycount == \"ACT/365\":
        return days / 365.0
    elif daycount == \"ACT/252\":
        return days / 252.0
    else:
        raise ValueError(\"Unsupported daycount\")

def compute_row(meta: OptionMeta, x: AsOfInputs, daycount: Literal[\"ACT/365\",\"ACT/252\"]=\"ACT/365\") -> Dict:
    T = year_fraction(x.ts, meta.expiry, daycount)
    iv = implied_vol_from_price(x.opt_price, x.S, meta.strike, x.r, x.q, T, meta.right)
    if iv is None:
        greeks = dict(delta=None,gamma=None,vega=None,theta=None,rho=None)
    else:
        greeks = dict(
            delta=float(delta(x.S, meta.strike, x.r, x.q, iv, T, meta.right)),
            gamma=float(gamma(x.S, meta.strike, x.r, x.q, iv, T)),
            vega =float(vega (x.S, meta.strike, x.r, x.q, iv, T)),
            theta=float(theta(x.S, meta.strike, x.r, x.q, iv, T, meta.right)),
            rho  =float(rho  (x.S, meta.strike, x.r, x.q, iv, T, meta.right)),
        )
    return dict(
        ts=x.ts, underlying=meta.underlying, expiry=meta.expiry, right=meta.right, strike=meta.strike,
        S=x.S, opt_price=x.opt_price, r=x.r, q=x.q, T=T, iv=iv, **greeks
    )

def compute_table(meta: OptionMeta, inputs: Iterable[AsOfInputs], daycount: Literal[\"ACT/365\",\"ACT/252\"]=\"ACT/365\") -> pd.DataFrame:
    rows = [compute_row(meta, x, daycount) for x in inputs]
    return pd.DataFrame(rows)