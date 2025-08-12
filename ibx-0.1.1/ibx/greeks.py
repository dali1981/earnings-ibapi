"""
ibx.greeks — model greeks + IV solver and historical (as-of) delta selection.

Use cases
---------
1) **Now** (near realtime): ask IB for greeks with a streaming market data sub
   and cancel immediately once `impliedVol`/`delta` arrive.
2) **Past as-of**: compute IV and delta from option price bars at the as-of time.

Key APIs
--------
- bs_price(S, K, r, q, sigma, T, right) -> price
- bs_delta(S, K, r, q, sigma, T, right) -> delta
- implied_vol_from_price(price, S, K, r, q, T, right) -> sigma
- delta_from_price(price, S, K, r, q, T, right) -> (sigma, delta)
- compute_deltas_asof(hist, symbol, expiry, strikes, right, as_of, S_asof,
                      *, price_field="close", timeout=60.0) -> DataFrame

Notes
-----
- We default r=q=0.0 for simplicity. You can pass richer curves later.
- For American-style equity options, BS is an approximation; good enough
  for strike selection (25Δ/75Δ) and term visualization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from ibx import HistoricalService, make_option
from ibx.time import ib_end_datetime_instrument

# ---------------- Black–Scholes helpers ----------------

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def bs_price(S: float, K: float, r: float, q: float, sigma: float, T: float, right: str) -> float:
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if right == "C" else (K - S))
    fwd = S * math.exp((r - q) * T)
    d1 = (math.log(fwd / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if right == "C":
        return math.exp(-r * T) * (fwd * _norm_cdf(d1) - K * _norm_cdf(d2))
    else:
        return math.exp(-r * T) * (K * _norm_cdf(-d2) - fwd * _norm_cdf(-d1))


def bs_delta(S: float, K: float, r: float, q: float, sigma: float, T: float, right: str) -> float:
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.0 if right == "C" else -1.0  # degenerate; won't be used for selection
    fwd = S * math.exp((r - q) * T)
    d1 = (math.log(fwd / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    if right == "C":
        return math.exp(-q * T) * _norm_cdf(d1)
    else:
        return math.exp(-q * T) * (_norm_cdf(d1) - 1.0)


def _vega(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.0
    fwd = S * math.exp((r - q) * T)
    d1 = (math.log(fwd / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return math.exp(-r * T) * fwd * _norm_pdf(d1) * math.sqrt(T)


def implied_vol_from_price(price: float, S: float, K: float, r: float, q: float, T: float, right: str,
                            *, tol: float = 1e-6, max_iter: int = 60) -> Optional[float]:
    """Bracketing + Newton fallback. Returns None if not solvable.
    Bounds: [1e-6, 5.0]. If price is outside no-arb bounds, clamp and continue.
    """
    if price is None or not (price == price) or price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None

    # no-arbitrage bounds (discounted intrinsic to S upper bound)
    intrinsic = max(0.0, (S - K) if right == "C" else (K - S))
    upper = S  # loose
    p = min(max(price, intrinsic), upper)

    lo, hi = 1e-6, 5.0
    f_lo = bs_price(S, K, r, q, lo, T, right) - p
    f_hi = bs_price(S, K, r, q, hi, T, right) - p
    # Ensure bracket: expand hi if needed (limited times)
    tries = 0
    while f_lo * f_hi > 0 and tries < 10:
        hi *= 1.5
        f_hi = bs_price(S, K, r, q, hi, T, right) - p
        tries += 1
    if f_lo * f_hi > 0:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_price(S, K, r, q, mid, T, right) - p
        if abs(f_mid) < tol:
            return max(mid, 1e-6)
        # Newton step around mid if vega not tiny
        v = _vega(S, K, r, q, mid, T)
        if v > 1e-12:
            step = f_mid / v
            newton = mid - step
            if lo < newton < hi:
                # accept Newton if it stays in bracket
                mid = newton
                f_mid = bs_price(S, K, r, q, mid, T, right) - p
                if abs(f_mid) < tol:
                    return max(mid, 1e-6)
        # Bisection update
        if f_lo * f_mid <= 0:
            hi = mid; f_hi = f_mid
        else:
            lo = mid; f_lo = f_mid
    return max(mid, 1e-6)


def delta_from_price(price: float, S: float, K: float, r: float, q: float, T: float, right: str) -> Tuple[Optional[float], Optional[float]]:
    sigma = implied_vol_from_price(price, S, K, r, q, T, right)
    if sigma is None:
        return None, None
    return sigma, bs_delta(S, K, r, q, sigma, T, right)

# --------------- Historical as-of pricing helpers ---------------

@dataclass
class AsOfQuote:
    price: Optional[float]
    ts: Optional[pd.Timestamp]


def _option_price_asof(hist: HistoricalService, symbol: str, expiry: str, strike: float, right: str,
                        as_of: str, *, price_field: str = "close", timeout: float = 60.0) -> AsOfQuote:
    contract = make_option(symbol, expiry, strike, right)
    end = ib_end_datetime_instrument(hist.rt, contract, as_of, hyphen=True)
    rows = hist.bars(
        contract,
        endDateTime=end,
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=0,
        timeout=timeout,
    )
    if not rows:
        return AsOfQuote(price=None, ts=None)
    df = pd.DataFrame(rows)
    ts = pd.to_datetime(df.get("time", df.get("date")), errors="coerce")
    df["_ts"] = ts
    df = df.sort_values("_ts").dropna(subset=["_ts"])  # in case
    px = pd.to_numeric(df.get(price_field, df.get("wap", df.get("close"))), errors="coerce").dropna()
    if px.empty:
        return AsOfQuote(price=None, ts=df["_ts"].iloc[-1] if not df.empty else None)
    return AsOfQuote(price=float(px.iloc[-1]), ts=df["_ts"].iloc[-1])


def compute_deltas_asof(
    hist: HistoricalService,
    symbol: str,
    expiry: str,
    strikes: Iterable[float],
    right: str,
    as_of: str,
    S_asof: float,
    *,
    r: float = 0.0,
    q: float = 0.0,
    timeout: float = 60.0,
    price_field: str = "close",
) -> pd.DataFrame:
    """Return a DataFrame with columns: strike, price, iv, delta, ts.
    We fetch a 1D/1min window up to `as_of` for each strike and compute IV & delta.
    """
    out = []
    asof_dt = pd.to_datetime(as_of)
    exp_dt = pd.to_datetime(expiry, format="%Y%m%d")
    T = max((exp_dt - asof_dt).total_seconds(), 0) / (365.0 * 24 * 3600)

    for k in sorted({float(s) for s in strikes}):
        qte = _option_price_asof(hist, symbol, expiry, k, right, as_of, price_field=price_field, timeout=timeout)
        if qte.price is None:
            out.append({"strike": k, "price": None, "iv": None, "delta": None, "ts": qte.ts})
            continue
        iv, d = delta_from_price(qte.price, S_asof, k, r, q, T, right)
        out.append({"strike": k, "price": qte.price, "iv": iv, "delta": d, "ts": qte.ts})

    return pd.DataFrame(out)
