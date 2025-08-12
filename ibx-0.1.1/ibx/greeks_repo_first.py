"""
ibx.greeks_repo_first — Repo-first greeks/IV utilities (patched to remove datetime parse warnings)

Changes in this patch
---------------------
- Prefer robust IB parser (`parse_ib_datetime_series`) before `pd.to_datetime`.
- When falling back to pandas parsing, silence the "Could not infer format" warning locally.
- Apply this both in `_load_repo_bar` and `_fetch_and_persist_bars`.

Drop-in replacement for your existing ibx/greeks_repo_first.py.
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ibx import HistoricalService, SubscriptionService, make_option
from ibx.time import ib_end_datetime_instrument, parse_ib_datetime_series

# ---------------- Black–Scholes helpers ----------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_price(S: float, K: float, r: float, q: float, sigma: float, T: float, right: str) -> float:
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if right == "C" else (K - S))
    fwd = S * math.exp((r - q) * T)
    d1 = (math.log(fwd / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if right == "C":
        return math.exp(-r * T) * (fwd * _norm_cdf(d1) - K * _norm_cdf(d2))
    else:
        return math.exp(-r * T) * (K * _norm_cdf(-d2) - fwd * _norm_cdf(-d1))


def _vega(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.0
    fwd = S * math.exp((r - q) * T)
    d1 = (math.log(fwd / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return math.exp(-r * T) * fwd * _norm_pdf(d1) * math.sqrt(T)


def _bs_delta(S: float, K: float, r: float, q: float, sigma: float, T: float, right: str) -> float:
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.0 if right == "C" else -1.0
    fwd = S * math.exp((r - q) * T)
    d1 = (math.log(fwd / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    if right == "C":
        return math.exp(-q * T) * _norm_cdf(d1)
    else:
        return math.exp(-q * T) * (_norm_cdf(d1) - 1.0)


def _implied_vol_from_price(price: float, S: float, K: float, r: float, q: float, T: float, right: str,
                            *, tol: float = 1e-6, max_iter: int = 60) -> Optional[float]:
    if price is None or not (price == price) or price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    intrinsic = max(0.0, (S - K) if right == "C" else (K - S))
    p = min(max(price, intrinsic), S)
    lo, hi = 1e-6, 5.0
    f_lo = _bs_price(S, K, 0.0, 0.0, lo, T, right) - p
    f_hi = _bs_price(S, K, 0.0, 0.0, hi, T, right) - p
    tries = 0
    while f_lo * f_hi > 0 and tries < 10:
        hi *= 1.5
        f_hi = _bs_price(S, K, 0.0, 0.0, hi, T, right) - p
        tries += 1
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = _bs_price(S, K, 0.0, 0.0, mid, T, right) - p
        if abs(f_mid) < tol:
            return max(mid, 1e-6)
        v = _vega(S, K, 0.0, 0.0, mid, T)
        if v > 1e-12:
            newton = mid - f_mid / v
            if lo < newton < hi:
                mid = newton
                f_mid = _bs_price(S, K, 0.0, 0.0, mid, T, right) - p
                if abs(f_mid) < tol:
                    return max(mid, 1e-6)
        if f_lo * f_mid <= 0:
            hi = mid; f_hi = f_mid
        else:
            lo = mid; f_lo = f_mid
    return max(mid, 1e-6)

# ---------------- Repo-first helpers ----------------

@dataclass
class AsOfBar:
    price: Optional[float]
    ts: Optional[pd.Timestamp]


def _load_repo_bar(opt_repo, symbol: str, expiry: str, strike: float, right: str,
                   as_of: Optional[str], *, bar_size: str = "1 min") -> AsOfBar:
    """Load the last bar <= as_of from the options repo for this contract.
    Returns (price, ts) or (None, None) if not found.
    """
    try:
        df = opt_repo.load(underlying=symbol, expiry=expiry, right=right, strike=float(strike))
    except TypeError:
        df = opt_repo.load()
        if not df.empty:
            for col, val in (("underlying", symbol), ("expiry", expiry), ("right", right), ("strike", float(strike))):
                if col in df.columns:
                    df = df[df[col] == val]
    if df.empty:
        return AsOfBar(price=None, ts=None)

    tcol = None
    for c in ("time","date","datetime","bar_time"):
        if c in df.columns:
            tcol = c; break
    if tcol is None:
        return AsOfBar(price=None, ts=None)

    # Prefer IB-aware parser first; fallback to pandas without warnings
    s = df[tcol].astype(str)
    ts = parse_ib_datetime_series(s)
    if ts.notna().sum() == 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ts = pd.to_datetime(s, errors="coerce")
    df = df.copy()
    df["_ts"] = ts
    df = df.dropna(subset=["_ts"]).sort_values("_ts")

    if as_of:
        cutoff = pd.to_datetime(as_of)
        df = df[df["_ts"] <= cutoff]
        if df.empty:
            return AsOfBar(price=None, ts=None)

    for col in ("close","wap","open"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                return AsOfBar(price=float(s.iloc[-1]), ts=df["_ts"].iloc[-1])

    return AsOfBar(price=None, ts=df["_ts"].iloc[-1])


def _fetch_and_persist_bars(hist: HistoricalService, opt_repo, symbol: str, expiry: str, strike: float, right: str,
                             as_of: Optional[str], *, duration: str = "1 D", bar_size: str = "1 min",
                             what_to_show: str = "TRADES", timeout: float = 120.0) -> AsOfBar:
    contract = make_option(symbol, expiry, strike, right)
    end = ib_end_datetime_instrument(hist.rt, contract, as_of)
    rows = hist.bars(
        contract,
        endDateTime=end,
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=0,
        timeout=timeout,
    )
    if not rows:
        return AsOfBar(price=None, ts=None)
    df = pd.DataFrame(rows)

    # Persist immediately
    try:
        opt_repo.save(
            df,
            underlying=symbol,
            expiry=expiry,
            right=right,
            strike=float(strike),
            bar_size=bar_size,
            what_to_show=what_to_show,
        )
    except Exception:
        pass

    # Extract last timestamp/price (IB-aware parsing first)
    t = None
    for c in ("time","date","datetime","bar_time"):
        if c in df.columns:
            s = df[c].astype(str)
            ts = parse_ib_datetime_series(s)
            if ts.notna().any():
                t = ts.dropna(); break
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                t = pd.to_datetime(s, errors="coerce").dropna()
            if t.size:
                break

    price = None
    for col in ("close","wap","open"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.size:
                price = float(s.iloc[-1]); break

    return AsOfBar(price=price, ts=t.iloc[-1] if t is not None and t.size else None)

# ---------------- Public APIs ----------------

def option_greeks_repo_first(
    hist: HistoricalService,
    opt_repo,
    subs: SubscriptionService,
    symbol: str,
    expiry: str,
    strike: float,
    right: str,
    as_of: Optional[str],
    S_asof: float,
    *,
    bar_size: str = "1 min",
    what_to_show: str = "TRADES",
    timeout: float = 120.0,
) -> Tuple[Optional[float], Optional[float]]:
    # 1) Repo first
    q = _load_repo_bar(opt_repo, symbol, expiry, strike, right, as_of, bar_size=bar_size)
    if q.price is not None:
        T = _ttm_years(expiry, as_of)
        iv = _implied_vol_from_price(q.price, S_asof, float(strike), 0.0, 0.0, T, right)
        d = None if iv is None else _bs_delta(S_asof, float(strike), 0.0, 0.0, iv, T, right)
        return iv, d

    # 2) Fetch minimal bars from IB and persist
    q = _fetch_and_persist_bars(hist, opt_repo, symbol, expiry, strike, right, as_of,
                                duration="1 D", bar_size=bar_size, what_to_show=what_to_show, timeout=timeout)
    if q.price is not None:
        T = _ttm_years(expiry, as_of)
        iv = _implied_vol_from_price(q.price, S_asof, float(strike), 0.0, 0.0, T, right)
        d = None if iv is None else _bs_delta(S_asof, float(strike), 0.0, 0.0, iv, T, right)
        return iv, d

    # 3) Last resort (only when as_of is None): brief streaming greeks
    if as_of is None:
        try:
            h = subs.market_data(make_option(symbol, expiry, strike, right), genericTicks="106", snapshot=False)
            deadline = time.time() + 4.0
            iv = delta = None
            try:
                for evt in h:
                    if evt.get("type") == "tickOption":
                        if evt.get("impliedVol") is not None: iv = float(evt["impliedVol"])  # decimal
                        if evt.get("delta") is not None:      delta = float(evt["delta"])   # decimal
                        if iv is not None and delta is not None: break
                    if time.time() > deadline: break
            finally:
                h.cancel()
            return iv, delta
        except Exception:
            pass

    return None, None


def compute_deltas_asof_repo_first(
    hist: HistoricalService,
    opt_repo,
    symbol: str,
    expiry: str,
    strikes: Iterable[float],
    right: str,
    as_of: str,
    S_asof: float,
    *,
    bar_size: str = "1 min",
    what_to_show: str = "TRADES",
    timeout: float = 120.0,
) -> pd.DataFrame:
    out = []
    T = _ttm_years(expiry, as_of)
    for k in sorted({float(s) for s in strikes}):
        iv, d = option_greeks_repo_first(
            hist, opt_repo, None, symbol, expiry, k, right, as_of, S_asof,
            bar_size=bar_size, what_to_show=what_to_show, timeout=timeout,
        )
        q = _load_repo_bar(opt_repo, symbol, expiry, k, right, as_of, bar_size=bar_size)
        out.append({
            "strike": float(k),
            "price": q.price,
            "iv": iv,
            "delta": d,
            "ts": q.ts,
        })
    return pd.DataFrame(out)

# ---------------- time to maturity ----------------

def _ttm_years(expiry_yyyymmdd: str, as_of: Optional[str]) -> float:
    asof_dt = pd.to_datetime(as_of) if as_of else pd.Timestamp.now()
    exp_dt = pd.to_datetime(expiry_yyyymmdd, format="%Y%m%d")
    return max((exp_dt - asof_dt).total_seconds(), 0.0) / (365.0 * 24.0 * 3600.0)
