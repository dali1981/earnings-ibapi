# client_chain_backfill.py
from __future__ import annotations

import time
from datetime import datetime, date
from typing import Dict, List, Tuple, Iterable, Optional

import pandas as pd

from ibx import IBRuntime, SubscriptionService, HistoricalService, make_stock, make_option
from ibx_repos import (
    OptionChainSnapshotRepository,
    OptionBarRepository, OptionMeta,
    EquityBarRepository,    # used only to get underlying ref if you prefer via history
)

# -------- helpers --------

def _parse_chain_latest(chain_repo: OptionChainSnapshotRepository, symbol: str) -> pd.DataFrame:
    """Load the most recent chain snapshot for symbol."""
    df = chain_repo.load(underlying=symbol)
    if df.empty:
        raise RuntimeError(f"No chain snapshot found in repo for {symbol}. Run snapshot_chain(symbol) first.")
    # take latest snapshot_date
    latest = df["snapshot_date"].max()
    snap = df[df["snapshot_date"] == latest].copy()
    return snap

def _next_two_expiries(chain_df: pd.DataFrame) -> List[str]:
    """Get the next two expiries from union of expirations across rows."""
    exps: List[str] = []
    for xs in chain_df["expirations"]:
        exps.extend(list(xs))
    # normalize, unique, future only
    today = pd.Timestamp(date.today())
    vals = sorted({pd.to_datetime(x, format="%Y%m%d", errors="coerce") for x in exps if x})
    vals = [x for x in vals if pd.notna(x) and x >= today]
    if len(vals) < 2:
        raise RuntimeError("Chain does not have at least two future expiries")
    return [vals[0].strftime("%Y%m%d"), vals[1].strftime("%Y%m%d")]

def _underlying_ref_price(hist: HistoricalService, symbol: str) -> float:
    """Get a recent last price from 1-min bars (fallback to close of last bar)."""
    rows = hist.bars(make_stock(symbol), endDateTime="", durationStr="1 D",
                     barSizeSetting="1 min", whatToShow="TRADES", useRTH=0, timeout=30.0)
    if not rows:
        raise RuntimeError("No underlying history returned for ref price")
    return float(pd.DataFrame(rows).iloc[-1]["close"])

def _nearest_strikes(strikes: Iterable[float], S: float, limit: int = 30) -> List[float]:
    """Take strikes closest to S (limit count)."""
    arr = sorted({float(s) for s in strikes})
    arr = sorted(arr, key=lambda k: abs(k - S))
    return arr[:limit]

def _option_delta_snapshot(subs: SubscriptionService, symbol: str, expiry: str, strike: float, right: str,
                           timeout_sec: float = 3.5) -> Optional[float]:
    """
    One-shot request for option greeks via market data snapshot.
    Requires appropriate IB market data permissions. Returns delta or None.
    """
    contract = make_option(symbol, expiry, strike, right)
    handle = subs.market_data(contract, genericTicks="106", snapshot=True)
    deadline = time.time() + timeout_sec
    delta = None
    for evt in handle:
        if evt.get("type") == "tickOption":
            d = evt.get("delta", None)
            if d is not None:
                delta = float(d)
                break
        if time.time() > deadline:
            break
    handle.cancel()
    return delta

def _select_targets_by_delta(subs: SubscriptionService, strikes: List[float], symbol: str, expiry: str,
                             S: float, sides=("C","P"), targets=(0.25, 0.75)) -> Dict[Tuple[str, float], float]:
    """
    Returns mapping {(side, selected_strike) -> delta} for each requested side/target,
    plus includes ATM selection (target 'ATM' ~ nearest strike to S) for each side.
    """
    # ATM per side
    selections: Dict[Tuple[str, float], float] = {}
    atm_strike = min(strikes, key=lambda k: abs(k - S))
    for side in sides:
        # for ATM we don't need delta; we can still fetch for reporting
        d_atm = _option_delta_snapshot(subs, symbol, expiry, atm_strike, side) or (0.0 if side == "C" else -0.0)
        selections[(side, atm_strike)] = d_atm

    # Pre-compute deltas for a short list of candidate strikes near S
    # (you can widen this window if needed)
    candidates = strikes[:]
    # Gather deltas
    deltas: Dict[Tuple[str, float], float] = {}
    for side in sides:
        for k in candidates:
            d = _option_delta_snapshot(subs, symbol, expiry, k, side)
            if d is not None:
                deltas[(side, k)] = d

    # For each target, pick nearest by absolute error
    for side in sides:
        for tgt in targets:
            goal = tgt if side == "C" else -tgt
            # filter only computed deltas for this side
            pool = {k: v for (s, k), v in deltas.items() if s == side}
            if not pool:
                continue
            best_k = min(pool.keys(), key=lambda k: abs(pool[k] - goal))
            selections[(side, best_k)] = pool[best_k]

    return selections

def _backfill_all_minutes(hist: HistoricalService, opt_repo: OptionBarRepository, meta: OptionMeta,
                          chunk: str = "1 W", max_loops: int = 52) -> int:
    """
    Pull 1-min bars backwards in chunks until no more. Returns total rows saved.
    """
    total = 0
    end = ""  # now
    last_earliest: Optional[pd.Timestamp] = None

    for _ in range(max_loops):
        rows = hist.bars(
            make_option(meta.underlying, meta.expiry, meta.strike, meta.right),
            endDateTime=end,
            durationStr=chunk,
            barSizeSetting=meta.bar_size,
            whatToShow=meta.what_to_show or "TRADES",
            useRTH=0,
            timeout=60.0
        )
        df = pd.DataFrame(rows)
        if df.empty:
            break
        # save chunk
        opt_repo.save(df, meta)
        total += len(df)

        # compute next end (go further back)
        # 'date' is string like 'YYYYMMDD  HH:MM:SS' for intraday with formatDate=1
        # df["ts"] = pd.to_datetime(df["date"], errors="coerce")
        df["ts"] = pd.to_datetime(df["date"], format="%Y%m%d %H:%M:%S", errors="coerce")
        # and if some are daily:
        mask = df["ts"].isna()
        df.loc[mask, "ts"] = pd.to_datetime(df.loc[mask, "date"], format="%Y%m%d", errors="coerce")

        earliest = df["ts"].min()
        if pd.isna(earliest) or (last_earliest is not None and earliest >= last_earliest):
            break
        last_earliest = earliest
        # IB expects 'YYYYMMDD HH:MM:SS' local; passing iso-like string works in practice:
        end = earliest.strftime("%Y%m%d %H:%M:%S")

        # Be nice to rate limits
        time.sleep(0.25)

    return total

# -------- main flow --------

def build_dataframe_for_symbol(symbol: str,
                               chain_repo: ChainRepository,
                               opt_repo: OptionBarRepository,
                               hist: HistoricalService,
                               subs: SubscriptionService) -> pd.DataFrame:
    """
    - load chain snapshot
    - choose next 2 expiries
    - select ATM and ~25/75 delta C & P per expiry
    - backfill all minute data for those options into repo
    - load back from repo and return a combined DataFrame
    """
    chain = _parse_chain_latest(chain_repo, symbol)
    expiries = _next_two_expiries(chain)
    S = _underlying_ref_price(hist, symbol)

    # Union strikes across exchanges for simplicity
    union_strikes = sorted({float(s) for xs in chain["strikes"] for s in xs})
    candidates = _nearest_strikes(union_strikes, S, limit=40)

    chosen: List[Tuple[str, str, float, str, float]] = []  # (expiry, side, strike, note, delta)
    for expiry in expiries:
        sel = _select_targets_by_delta(subs, candidates, symbol, expiry, S, sides=("C","P"), targets=(0.25, 0.75))
        # ensure ATM for both sides present even if delta snapshots failed
        atm_k = min(candidates, key=lambda k: abs(k - S))
        for side in ("C","P"):
            if (side, atm_k) not in sel:
                sel[(side, atm_k)] = 0.0 if side == "C" else -0.0
        for (side, k), d in sel.items():
            note = "ATM" if abs(k - atm_k) < 1e-9 else (f"{'25' if abs(d) < 0.5 else '75'}Δ")
            chosen.append((expiry, side, float(k), note, float(d)))

    # Backfill into repo
    total_rows = 0
    for expiry, side, k, note, d in chosen:
        meta = OptionMeta(underlying=symbol, expiry=expiry, strike=k, right=side, bar_size="1 min", what_to_show="TRADES")
        got = _backfill_all_minutes(hist, opt_repo, meta, chunk="1 W", max_loops=52)
        print(f"Backfilled {got:5d} rows for {symbol} {expiry} {side} {k} ({note}, Δ≈{d:.2f})")
        total_rows += got

    print(f"Total rows saved: {total_rows}")

    # Load back from repo and assemble a tidy DataFrame
    frames = []
    for expiry, side, k, note, d in chosen:
        df = opt_repo.load(underlying=symbol, expiry=expiry, right=side, strike=k)
        if df.empty:
            continue
        df["expiry"] = expiry
        df["right"] = side
        df["strike"] = k
        df["note"] = note
        # normalize a proper timestamp
        df["time"] = pd.to_datetime(df.get("time", pd.to_datetime(df["date"], errors="coerce")))
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out.sort_values(["expiry", "right", "strike", "time"], inplace=True)
    # Set a helpful multiindex
    out.set_index(["expiry", "right", "strike", "time"], inplace=True)
    return out[["open","high","low","close","volume","wap","bar_count"]].sort_index()

# -------- example usage --------

if __name__ == "__main__":
    SYMBOL = "CMCL"
    # Repos (adjust paths)
    chain_repo = OptionChainSnapshotRepository("data/option_chains")
    opt_repo   = OptionBarRepository("data/option_bars")

    with IBRuntime(port=4002, client_id=42) as rt:
        subs = SubscriptionService(rt)
        hist = HistoricalService(rt)

        df = build_dataframe_for_symbol(SYMBOL, chain_repo, opt_repo, hist, subs)

    print("Combined DF shape:", df.shape)
    print(df.tail(10))
