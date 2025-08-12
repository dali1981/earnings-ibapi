#!/usr/bin/env python3
"""
client_chain_backfill_v2_patched.py

- Centralises time handling via ibx_time.py
- Picks ATM and ~±25/±75 BS-delta at an AS-OF timestamp (instrument TZ-aware)
- Robust spot selection: try historical bars; if NaN/empty, fall back to market data snapshot
- Backfills 1-min option bars (chunked, overlap-safe) and reads back from repo

Run:
  python client_chain_backfill_v2_patched.py \
    --symbol AAPL --as-of "20250808 15:50:00" \
    --port 4002 --client-id 42 \
    --chains-base data/option_chains \
    --options-base data/option_bars --rows 12
"""
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, date
from typing import Dict, List, Tuple, Iterable, Optional

import pandas as pd

from ibx import IBRuntime, SubscriptionService, HistoricalService, make_stock, make_option
from ibx_repos import ChainRepository, OptionBarRepository, OptionMeta
from ibx_time import ib_end_datetime_instrument, parse_ib_datetime_series

# -------------------- helpers: snapshot last --------------------

def _stock_snapshot_last(subs: SubscriptionService, symbol: str, timeout_sec: float = 3.5) -> Optional[float]:
    """Get a quick last (or mid) via one-shot snapshot. Returns None if unavailable."""
    h = subs.market_data(make_stock(symbol), genericTicks="", snapshot=True)
    deadline = time.time() + timeout_sec
    last = bid = ask = close = None
    try:
        for evt in h:
            # Common fields in our wrapper dicts
            last = evt.get("last", last)
            bid = evt.get("bid", bid)
            ask = evt.get("ask", ask)
            close = evt.get("close", close)
            # Some events are typed
            if evt.get("type") == "tickPrice":
                tick = str(evt.get("tick", "")).upper()
                price = evt.get("price") or evt.get("value")
                if tick == "LAST" and price is not None:
                    last = price
                elif tick == "BID" and price is not None:
                    bid = price
                elif tick == "ASK" and price is not None:
                    ask = price
                elif tick == "CLOSE" and price is not None:
                    close = price
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

# -------------------- date/time helpers --------------------

def _ref_price_asof(hist: HistoricalService, subs: SubscriptionService, symbol: str, as_of: Optional[str]) -> float:
    """Reference price at AS-OF (instrument TZ-aware). Falls back to snapshot mid/last if needed."""
    end = ""
    if as_of:
        end = ib_end_datetime_instrument(hist.rt, make_stock(symbol), as_of)
    rows = hist.bars(make_stock(symbol), endDateTime=end, durationStr="1 D",
                     barSizeSetting="1 min", whatToShow="TRADES", useRTH=0, timeout=45.0)
    S = None
    if rows:
        df = pd.DataFrame(rows)
        # pick last non-null among close, wap, open
        for col in ("close", "wap", "open"):
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if not s.empty:
                    S = float(s.iloc[-1])
                    break
    if S is None:
        S = _stock_snapshot_last(subs, symbol)
    if S is None or not (S == S) or S <= 0:  # NaN or invalid
        raise RuntimeError("Could not determine a valid AS-OF reference price for the underlying")
    return S

# -------------------- chain utilities --------------------

def _parse_chain_latest(chain_repo: ChainRepository, symbol: str) -> pd.DataFrame:
    df = chain_repo.load(underlying=symbol)
    if df.empty:
        raise RuntimeError(f"No chain snapshot found in repo for {symbol}. Run snapshot_chain(symbol) first.")
    latest = pd.to_datetime(df["snapshot_date"]).max()
    return df[pd.to_datetime(df["snapshot_date"]) == latest].copy()


def _next_two_expiries(chain_df: pd.DataFrame) -> List[str]:
    exps: List[str] = []
    for xs in chain_df["expirations"]:
        if xs is None:
            continue
        try:
            exps.extend(list(xs))
        except Exception:
            try:
                exps.extend(list(xs.to_pylist()))
            except Exception:
                pass
    today = pd.Timestamp(date.today())
    vals = sorted({pd.to_datetime(x, format="%Y%m%d", errors="coerce") for x in exps if x})
    vals = [x for x in vals if pd.notna(x) and x >= today]
    if len(vals) < 2:
        raise RuntimeError("Chain does not have at least two future expiries")
    return [vals[0].strftime("%Y%m%d"), vals[1].strftime("%Y%m%d")]


def _nearest_strikes(strikes: Iterable[float], S: float, limit: int = 30) -> List[float]:
    if S is None:
        raise ValueError("Reference price S is None")
    arr = sorted({float(s) for s in strikes})
    arr = sorted(arr, key=lambda k: abs(k - float(S)))
    return arr[:limit]

# -------------------- greeks & selection --------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_delta(S: float, K: float, sigma: Optional[float], T_years: float, side: str) -> Optional[float]:
    if sigma is None or sigma <= 0 or T_years <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T_years) / (sigma * math.sqrt(T_years))
    return _norm_cdf(d1) if side == "C" else (_norm_cdf(d1) - 1.0)


def _option_snapshot_greeks(subs: SubscriptionService, symbol: str, expiry: str, strike: float, right: str,
                            timeout_sec: float = 3.5):
    contract = make_option(symbol, expiry, strike, right)
    handle = subs.market_data(contract, genericTicks="106", snapshot=True)
    deadline = time.time() + timeout_sec
    delta = iv = None
    try:
        for evt in handle:
            if evt.get("type") == "tickOption":
                delta = evt.get("delta", delta)
                iv = evt.get("impliedVol", iv)
                if delta is not None and iv is not None:
                    break
            if time.time() > deadline:
                break
    finally:
        handle.cancel()
    try:
        return (float(delta) if delta is not None else None,
                float(iv) if iv is not None else None)
    except Exception:
        return (None, None)


def _select_targets_by_delta_at_time(subs: SubscriptionService, strikes: List[float], symbol: str,
                                     expiry: str, S_asof: float, as_of: str) -> Dict[Tuple[str, float], Tuple[Optional[float], Optional[float]]]:
    exp_dt = pd.to_datetime(expiry, format="%Y%m%d")
    asof_dt = pd.to_datetime(as_of) if as_of else pd.Timestamp.now()
    T = max((exp_dt - asof_dt).total_seconds(), 0) / (365.0 * 24 * 3600)

    atm_strike = min(strikes, key=lambda k: abs(k - S_asof))
    out: Dict[Tuple[str, float], Tuple[Optional[float], Optional[float]]] = {}

    for side in ("C", "P"):
        _, iv_now = _option_snapshot_greeks(subs, symbol, expiry, atm_strike, side)
        iv_for_bs = iv_now or 0.25
        bs = {k: _bs_delta(S_asof, float(k), iv_for_bs, T, side) for k in strikes}
        bs = {k: d for k, d in bs.items() if d is not None}
        out[(side, atm_strike)] = (bs.get(atm_strike), iv_now)
        for tgt in (0.25, 0.75):
            if not bs:
                continue
            goal = tgt if side == "C" else -tgt
            best_k = min(bs.keys(), key=lambda k: abs(bs[k] - goal))
            out[(side, best_k)] = (bs[best_k], iv_now)

    return out

# -------------------- backfill --------------------

def _backfill_all_minutes(hist: HistoricalService, opt_repo: OptionBarRepository, meta: OptionMeta,
                          chunk: str = "1 W", max_loops: int = 52) -> int:
    total = 0
    end = ""
    last_earliest: Optional[pd.Timestamp] = None

    contract = make_option(meta.underlying, meta.expiry, meta.strike, meta.right)
    # Convert naive -> instrument TZ + hyphen
    end_ib = "" if not end else ib_end_datetime_instrument(hist.rt, contract, end, hyphen=True)
    print("DEBUG end ->", end, "| end_ib ->", end_ib)  # optional sanity check

    for _ in range(max_loops):
        rows = hist.bars(
            contract,
            endDateTime=end_ib,
            durationStr=chunk,
            barSizeSetting=meta.bar_size,
            whatToShow=meta.what_to_show or "TRADES",
            useRTH=0,
            timeout=60.0
        )
        df = pd.DataFrame(rows)
        if df.empty:
            break
        opt_repo.save(df, meta)
        total += len(df)
        ts = parse_ib_datetime_series(df["date"] if "date" in df.columns else df.get("time"))
        earliest = ts.min()
        if pd.isna(earliest) or (last_earliest is not None and earliest >= last_earliest):
            break
        last_earliest = earliest
        end = (earliest - pd.Timedelta(seconds=1)).strftime("%Y%m%d %H:%M:%S")
        time.sleep(0.25)

    return total

# -------------------- build flow --------------------

def build_dataframe_for_symbol(symbol: str,
                               chain_repo: ChainRepository,
                               opt_repo: OptionBarRepository,
                               hist: HistoricalService,
                               subs: SubscriptionService,
                               as_of: Optional[str]) -> pd.DataFrame:
    chain = _parse_chain_latest(chain_repo, symbol)
    expiries = _next_two_expiries(chain)
    S_asof = _ref_price_asof(hist, subs, symbol, as_of)

    union_strikes = sorted({float(s) for xs in chain["strikes"] for s in (list(xs) if xs is not None else [])})
    candidates = _nearest_strikes(union_strikes, S_asof, limit=40)

    chosen: List[Tuple[str, str, float, str, float]] = []
    for expiry in expiries:
        sel = _select_targets_by_delta_at_time(subs, candidates, symbol, expiry, S_asof, as_of or "")
        atm_k = min(candidates, key=lambda k: abs(k - S_asof))
        for (side, k), (d_est, iv_used) in sel.items():
            if abs(k - atm_k) < 1e-9:
                note = "ATM"
            elif d_est is not None and abs(abs(d_est) - 0.25) < 0.15:
                note = "25Δ"
            elif d_est is not None and abs(abs(d_est) - 0.75) < 0.20:
                note = "75Δ"
            else:
                note = "Δ?"
            chosen.append((expiry, side, float(k), note, float(d_est) if d_est is not None else float("nan")))

    total_rows = 0
    for expiry, side, k, note, d in chosen:
        meta = OptionMeta(underlying=symbol, expiry=expiry, strike=k, right=side, bar_size="1 min", what_to_show="TRADES")
        got = _backfill_all_minutes(hist, opt_repo, meta, chunk="1 W", max_loops=52)
        d_txt = "" if pd.isna(d) else f"{d:.2f}"
        print(f"Backfilled {got:5d} rows for {symbol} {expiry} {side} {k} ({note}, Δ≈{d_txt})")
        total_rows += got
    print(f"Total rows saved: {total_rows}")

    frames = []
    for expiry, side, k, note, d in chosen:
        df = opt_repo.load(underlying=symbol, expiry=expiry, right=side, strike=k)
        if df.empty:
            continue
        df["time"] = pd.to_datetime(df.get("time", pd.to_datetime(df["date"], errors="coerce")))
        df = df.sort_values("time").drop_duplicates(subset=["time"]).copy()
        df["expiry"] = expiry; df["right"] = side; df["strike"] = k; df["note"] = note
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out.sort_values(["expiry", "right", "strike", "time"], inplace=True)
    out.set_index(["expiry", "right", "strike", "time"], inplace=True)
    return out[["open","high","low","close","volume","wap","bar_count"]].sort_index()

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--as-of", default=None, help='Reference timestamp "YYYYMMDD HH:MM:SS" (TWS local time).')
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4002)
    ap.add_argument("--client-id", type=int, default=42)
    ap.add_argument("--chains-base", default="data/option_chains")
    ap.add_argument("--options-base", default="data/option_bars")
    ap.add_argument("--rows", type=int, default=10, help="Rows to print at the end.")
    args = ap.parse_args()

    chain_repo = ChainRepository(args.chains_base)
    opt_repo   = OptionBarRepository(args.options_base)

    with IBRuntime(host=args.host, port=args.port, client_id=args.client_id) as rt:
        subs = SubscriptionService(rt)
        hist = HistoricalService(rt)
        df = build_dataframe_for_symbol(args.symbol, chain_repo, opt_repo, hist, subs, args.as_of)

    if df.empty:
        print("Combined DF is empty.")
        return

    print(f"Combined DF shape: {df.shape}")
    print(df.head(args.rows).to_string())

if __name__ == "__main__":
    main()
