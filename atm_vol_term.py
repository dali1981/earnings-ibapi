#!/usr/bin/env python3
"""
atm_vol_term.py — Compute ATM implied-vol term structure across expiries

What it does
------------
- Loads the latest option-chain snapshot for a symbol from your repo
- Finds a reference underlying price at an AS-OF time (instrument TZ aware)
- For each expiry, picks the ATM strike (nearest to spot at AS-OF)
- Queries IB snapshot greeks (genericTicks=106) to get implied vol/delta
- Returns a tidy DataFrame of the ATM vol term structure (and prints it)

Usage
-----
python atm_vol_term.py \
  --symbol AAPL --as-of "20250808 15:50:00" \
  --port 4002 --client-id 42 \
  --chains-base data/option_chains --limit 10

You can also import and call `compute_atm_vol_term(...)` from your own code.
"""
from __future__ import annotations

import argparse
import time
from datetime import date
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from ibx import IBRuntime, SubscriptionService, HistoricalService, make_stock, make_option
from ibx_repos import ChainRepository
from ibx_time import ib_end_datetime_instrument, parse_ib_datetime_series

# -------------------- small utils --------------------

def _to_list(x):
    if x is None:
        return []
    try:
        if pd.isna(x):  # pandas NA
            return []
    except Exception:
        pass
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    try:
        return list(x)
    except Exception:
        return [x]


def _nearest_strike(strikes: Iterable[float], S: float) -> float:
    arr = sorted({float(s) for s in strikes})
    return min(arr, key=lambda k: abs(k - float(S)))

# -------------------- spot at AS-OF --------------------

def _stock_snapshot_last(subs: SubscriptionService, symbol: str, timeout_sec: float = 3.5) -> Optional[float]:
    h = subs.market_data(make_stock(symbol), genericTicks="", snapshot=True)
    deadline = time.time() + timeout_sec
    last = bid = ask = close = None
    try:
        for evt in h:
            last = evt.get("last", last)
            bid = evt.get("bid", bid)
            ask = evt.get("ask", ask)
            close = evt.get("close", close)
            if time.time() > deadline:
                break
    finally:
        h.cancel()
    if last is not None:
        return float(last)
    if bid is not None and ask is not None:
        return (float(bid) + float(ask)) / 2.0
    if close is not None:
        return float(close)
    return None


def _ref_price_asof(hist: HistoricalService, subs: SubscriptionService, symbol: str, as_of: Optional[str]) -> float:
    end = ""
    if as_of:
        end = ib_end_datetime_instrument(hist.rt, make_stock(symbol), as_of, hyphen=True)
    rows = hist.bars(
        make_stock(symbol),
        endDateTime=end,
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=0,
        timeout=45.0,
    )
    S = None
    if rows:
        df = pd.DataFrame(rows)
        for col in ("close", "wap", "open"):
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if not s.empty:
                    S = float(s.iloc[-1])
                    break
    if S is None:
        S = _stock_snapshot_last(subs, symbol)
    if S is None or not (S == S) or S <= 0:
        raise RuntimeError("Could not determine a valid AS-OF reference price for the underlying")
    return S

# -------------------- chain snapshot --------------------

def _load_latest_chain(chain_repo: ChainRepository, symbol: str) -> pd.DataFrame:
    df = chain_repo.load(underlying=symbol)
    if df.empty:
        raise RuntimeError(f"No chain snapshot found in repo for {symbol}.")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])  # ensure ts
    latest = df["snapshot_date"].max()
    return df[df["snapshot_date"] == latest].copy()


def _next_expiries(chain_df: pd.DataFrame, limit: Optional[int]) -> List[str]:
    exps = []
    for xs in chain_df["expirations"]:
        exps.extend(_to_list(xs))
    today = pd.Timestamp(date.today())
    vals = sorted({pd.to_datetime(x, format="%Y%m%d", errors="coerce") for x in exps if x})
    vals = [x for x in vals if pd.notna(x) and x >= today]
    if limit:
        vals = vals[:limit]
    return [x.strftime("%Y%m%d") for x in vals]

# -------------------- greeks snapshot --------------------

def _option_snapshot_greeks(
    subs: SubscriptionService,
    symbol: str,
    expiry: str,
    strike: float,
    right: str,
    timeout_sec: float = 3.5,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (impliedVol, delta) for a single option using snapshot greeks."""
    h = subs.market_data(make_option(symbol, expiry, strike, right), genericTicks="106", snapshot=True)
    deadline = time.time() + timeout_sec
    iv = delta = None
    try:
        for evt in h:
            if evt.get("type") == "tickOption":
                if "impliedVol" in evt and evt["impliedVol"] is not None:
                    iv = float(evt["impliedVol"])  # e.g., 0.25 means 25%
                if "delta" in evt and evt["delta"] is not None:
                    delta = float(evt["delta"])   # call ~[0,1], put ~[-1,0]
                if iv is not None and delta is not None:
                    break
            if time.time() > deadline:
                break
    finally:
        h.cancel()
    return iv, delta

# -------------------- main computation --------------------

def compute_atm_vol_term(
    rt: IBRuntime,
    chain_repo: ChainRepository,
    hist: HistoricalService,
    subs: SubscriptionService,
    symbol: str,
    as_of: Optional[str],
    *,
    limit: Optional[int] = 10,
) -> pd.DataFrame:
    chain = _load_latest_chain(chain_repo, symbol)
    expiries = _next_expiries(chain, limit)

    # union of strikes across all rows
    strike_union = sorted({float(s) for xs in chain["strikes"] for s in _to_list(xs)})

    S_asof = _ref_price_asof(hist, subs, symbol, as_of)

    rows = []
    asof_dt = pd.to_datetime(as_of) if as_of else pd.Timestamp.now()

    for expiry in expiries:
        k_atm = _nearest_strike(strike_union, S_asof)
        iv_c, d_c = _option_snapshot_greeks(subs, symbol, expiry, k_atm, "C")
        iv_p, d_p = _option_snapshot_greeks(subs, symbol, expiry, k_atm, "P")

        # combine: prefer average when both present, else whichever exists
        iv_atm = None
        if iv_c is not None and iv_p is not None:
            iv_atm = 0.5 * (iv_c + iv_p)
        elif iv_c is not None:
            iv_atm = iv_c
        elif iv_p is not None:
            iv_atm = iv_p

        exp_dt = pd.to_datetime(expiry, format="%Y%m%d")
        ttm_years = max((exp_dt - asof_dt).total_seconds(), 0) / (365.0 * 24 * 3600)

        rows.append({
            "expiry": expiry,
            "ttm_years": ttm_years,
            "k_atm": float(k_atm),
            "spot_asof": float(S_asof),
            "iv_call": iv_c,
            "iv_put": iv_p,
            "iv_atm": iv_atm,
            "delta_call": d_c,
            "delta_put": d_p,
        })

    df = pd.DataFrame(rows).sort_values("expiry").reset_index(drop=True)
    return df

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--as-of", default=None, help='AS-OF time "YYYYMMDD HH:MM:SS" (instrument TZ applied).')
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4002)
    ap.add_argument("--client-id", type=int, default=77)
    ap.add_argument("--chains-base", default="data/option_chains")
    ap.add_argument("--limit", type=int, default=10, help="Max expiries to include")
    ap.add_argument("--rows", type=int, default=50)
    args = ap.parse_args()

    chain_repo = ChainRepository(args.chains_base)

    with IBRuntime(host=args.host, port=args.port, client_id=args.client_id) as rt:
        subs = SubscriptionService(rt)
        hist = HistoricalService(rt)
        df = compute_atm_vol_term(rt, chain_repo, hist, subs, args.symbol, args.as_of, limit=args.limit)

    if df.empty:
        print("(empty)")
        return

    # pretty print
    show = df.head(args.rows).copy()
    for c in ("iv_call", "iv_put", "iv_atm"):
        if c in show:
            show[c] = (show[c] * 100.0).round(2)  # percent
    print(show.to_string(index=False))

if __name__ == "__main__":
    main()


# python atm_vol_term.py \
#   --symbol AAPL \
#   --as-of "20250808 15:50:00" \
#   --port 4002 --client-id 42 \
#   --chains-base data/option_chains \
#   --limit 12