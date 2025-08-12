#!/usr/bin/env python3
"""
client_chain_backfill_v3_repo_first.py — Backfill minute bars for ATM and target‑delta options (repo‑first)

- Loads latest option chain for a symbol (partitioned by underlying)
- Gets AS‑OF spot (instrument TZ aware)
- For the next N expiries, selects:
    * ATM (nearest strike)
    * ~25Δ and ~75Δ for Calls and Puts, **computed at AS‑OF** from repo price bars
      (fetches + persists minimal bars if missing).
- Backfills all available 1‑minute bars for each selected contract (chunked, robust)
- Reloads from repo and prints a combined preview

Run
---
python client_chain_backfill_v3_repo_first.py \
  --symbol AAPL \
  --as-of "20250808 15:50:00" \
  --port 4002 --client-id 42 \
  --chains-base data/option_chains \
  --options-base data/option_bars \
  --rows 12
"""
from __future__ import annotations

# --- project-root import shim (so this runs from examples/ without installing) ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------------------------------------------

import argparse
import time
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from ibx import IBRuntime, SubscriptionService, HistoricalService, make_stock, make_option
from ibx.time import ib_end_datetime_instrument, parse_ib_datetime_series
from ibx.greeks_repo_first import compute_deltas_asof_repo_first
from ibx_repos import ChainRepository, OptionsRepository

# ========================== small utils ==========================

def _to_list(x):
    if x is None:
        return []
    try:
        if pd.isna(x):
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


# ==================== spot (reference) at AS-OF ====================

def _stock_snapshot_last(subs: SubscriptionService, symbol: str, timeout_sec: float = 4.0) -> Optional[float]:
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
        end = ib_end_datetime_instrument(hist.rt, make_stock(symbol), as_of, hyphen=False)
    rows = hist.bars(
        make_stock(symbol),
        endDateTime=end,
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=0,
        timeout=90.0,
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


# ========================= chain access =========================

def _load_latest_chain(chain_repo: ChainRepository, symbol: str) -> pd.DataFrame:
    try:
        df = chain_repo.load(underlying=symbol)
    except TypeError:
        df = chain_repo.load()
    if df.empty:
        raise RuntimeError(f"No chain snapshot found (base={getattr(chain_repo, 'base_path', '???')}).")
    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
        latest = df["snapshot_date"].max()
        df = df[df["snapshot_date"] == latest].copy()
    for col in ("underlying", "trading_class", "symbol"):
        if col in df.columns:
            sub = df[df[col] == symbol]
            if not sub.empty:
                df = sub
                break
    return df


def _next_expiries(chain_df: pd.DataFrame, limit: int = 2) -> List[str]:
    exps = []
    for xs in chain_df["expirations"]:
        exps.extend(_to_list(xs))
    today = pd.Timestamp(date.today())
    vals = sorted({pd.to_datetime(x, format="%Y%m%d", errors="coerce") for x in exps if x})
    vals = [x for x in vals if pd.notna(x) and x >= today]
    return [x.strftime("%Y%m%d") for x in vals[:limit]]


# =================== delta‑based strike selection ===================

def _pick_by_delta(df: pd.DataFrame, target: float, side: str) -> Optional[float]:
    want = target if side == 'C' else -target
    df = df.dropna(subset=['delta']).copy()
    if df.empty:
        return None
    df['err'] = (df['delta'] - want).abs()
    return float(df.loc[df['err'].idxmin(), 'strike'])


from dataclasses import dataclass
@dataclass
class Selection:
    expiry: str
    k_atm: float
    k_c25: Optional[float]
    k_c75: Optional[float]
    k_p25: Optional[float]
    k_p75: Optional[float]


def _select_contracts_for_expiry(
    hist: HistoricalService,
    opt_repo: OptionsRepository,
    symbol: str,
    expiry: str,
    union_strikes: Sequence[float],
    S_asof: float,
    as_of: Optional[str],
    *,
    probe: int = 40,
) -> Selection:
    sorted_strikes = sorted({float(s) for s in union_strikes})
    by_dist = sorted(sorted_strikes, key=lambda k: abs(k - S_asof))
    cand = by_dist[:probe]

    dc = compute_deltas_asof_repo_first(hist, opt_repo, symbol, expiry, cand, 'C', as_of or "", S_asof, timeout=120.0)
    dp = compute_deltas_asof_repo_first(hist, opt_repo, symbol, expiry, cand, 'P', as_of or "", S_asof, timeout=120.0)

    k_atm = _nearest_strike(cand, S_asof)
    k_c25 = _pick_by_delta(dc, 0.25, 'C')
    k_c75 = _pick_by_delta(dc, 0.75, 'C')
    k_p25 = _pick_by_delta(dp, 0.25, 'P')
    k_p75 = _pick_by_delta(dp, 0.75, 'P')

    return Selection(expiry, k_atm, k_c25, k_c75, k_p25, k_p75)


# ========================= backfill engine =========================

@dataclass
class BackfillMeta:
    symbol: str
    expiry: str
    right: str
    strike: float
    bar_size: str = "1 min"
    what_to_show: str = "TRADES"
    use_rth: int = 0


def _backfill_all_minutes(hist: HistoricalService, opt_repo: OptionsRepository, meta: BackfillMeta,
                          *, chunk: str = "1 W", max_loops: int = 52) -> int:
    total = 0
    end = ""
    last_earliest: Optional[pd.Timestamp] = None
    contract = make_option(meta.symbol, meta.expiry, meta.strike, meta.right)

    for loop in range(max_loops):
        end_ib = ib_end_datetime_instrument(hist.rt, contract, end or None, hyphen=False) if end != "" else ""
        rows = hist.bars(
            contract,
            endDateTime=end_ib,
            durationStr=chunk,
            barSizeSetting=meta.bar_size,
            whatToShow=meta.what_to_show,
            useRTH=meta.use_rth,
            timeout=180.0,
        )
        if not rows:
            break
        df = pd.DataFrame(rows)

        candidates = []
        for col in ("time", "date", "datetime", "bar_time"):
            if col in df.columns:
                ts = pd.to_datetime(df[col], errors="coerce")
                if ts.notna().any():
                    candidates.append(ts)
                    continue
                try:
                    ts2 = parse_ib_datetime_series(df[col].astype(str))
                    if ts2.notna().any():
                        candidates.append(ts2)
                except Exception:
                    pass
        if candidates:
            ts_all = pd.concat(candidates, axis=1).min(axis=1)
            earliest = ts_all.min()
            latest = ts_all.max()
        else:
            earliest = pd.NaT
            latest = pd.NaT

        print(f"DEBUG chunk rows={len(df)} tmin={earliest if pd.notna(earliest) else 'NaT'} "
              f"tmax={latest if pd.notna(latest) else 'NaT'} prev_end='{end}' end_ib='{end_ib}'")

        for c in ("open","high","low","close","wap"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        if "bar_count" in df.columns:
            df["bar_count"] = pd.to_numeric(df["bar_count"], errors="coerce")

        opt_repo.save(
            df,
            underlying=meta.symbol,
            expiry=meta.expiry,
            right=meta.right,
            strike=float(meta.strike),
            bar_size=meta.bar_size,
            what_to_show=meta.what_to_show,
        )
        total += len(df)

        if pd.isna(earliest):
            break
        if last_earliest is not None and earliest >= last_earliest:
            step_seconds = 61 if (meta.bar_size or "").startswith("1 min") else 1
            end = (last_earliest - pd.Timedelta(seconds=step_seconds)).strftime("%Y%m%d %H:%M:%S")
            continue
        last_earliest = earliest
        end = (earliest - pd.Timedelta(seconds=1)).strftime("%Y%m%d %H:%M:%S")

    return total


# ========================= loading view =========================

def _load_option_df(opt_repo: OptionsRepository, symbol: str, expiry: str, right: str, strike: float) -> pd.DataFrame:
    df = opt_repo.load(underlying=symbol, expiry=expiry, right=right, strike=float(strike))
    if df.empty:
        return df
    tcol = None
    for c in ("time","date","datetime","bar_time"):
        if c in df.columns:
            tcol = c; break
    if tcol is None:
        return df
    ts = pd.to_datetime(df[tcol], errors="coerce")
    df = df.drop(columns=[tcol]).assign(time=ts).dropna(subset=["time"]).set_index("time").sort_index()
    return df


# ============================== CLI ==============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--as-of", default=None, help='AS-OF time "YYYYMMDD HH:MM:SS" (instrument TZ applied).')
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4002)
    ap.add_argument("--client-id", type=int, default=77)
    ap.add_argument("--chains-base", default="data/option_chains")
    ap.add_argument("--options-base", default="data/option_bars")
    ap.add_argument("--expiries", type=int, default=2)
    ap.add_argument("--probe", type=int, default=40, help="# strikes around ATM to evaluate deltas")
    ap.add_argument("--rows", type=int, default=12)
    args = ap.parse_args()

    chain_repo = ChainRepository(args.chains_base)
    opt_repo = OptionsRepository(args.options_base)

    with IBRuntime(host=args.host, port=args.port, client_id=args.client_id) as rt:
        subs = SubscriptionService(rt)
        hist = HistoricalService(rt)

        chain = _load_latest_chain(chain_repo, args.symbol)
        expiries = _next_expiries(chain, args.expiries)
        strikes_union = sorted({float(s) for xs in chain["strikes"] for s in _to_list(xs)})

        S_asof = _ref_price_asof(hist, subs, args.symbol, args.as_of)

        selections = []
        for expiry in expiries:
            sel = _select_contracts_for_expiry(hist, opt_repo, args.symbol, expiry, strikes_union, S_asof, args.as_of, probe=args.probe)
            selections.append(sel)

        total_saved = 0
        for sel in selections:
            picks = [
                ("C", sel.k_atm, "ATM"),
                ("P", sel.k_atm, "ATM"),
                ("C", sel.k_c25, "25Δ"),
                ("C", sel.k_c75, "75Δ"),
                ("P", sel.k_p25, "25Δ"),
                ("P", sel.k_p75, "75Δ"),
            ]
            for side, k, tag in picks:
                if k is None:
                    continue
                meta = BackfillMeta(args.symbol, sel.expiry, side, float(k))
                got = _backfill_all_minutes(hist, opt_repo, meta, chunk="1 W", max_loops=52)
                total_saved += got
                print(f"Backfilled {got:5d} rows for {args.symbol} {sel.expiry} {side} {k} ({tag})")

    print(f"Total rows saved: {total_saved}")

    frames = []
    for sel in selections:
        for side, k in (("C", sel.k_atm),("P", sel.k_atm),("C", sel.k_c25),("C", sel.k_c75),("P", sel.k_p25),("P", sel.k_p75)):
            if k is None:
                continue
            df = _load_option_df(opt_repo, args.symbol, sel.expiry, side, float(k))
            if df.empty:
                continue
            df["expiry"] = sel.expiry
            df["right"] = side
            df["strike"] = float(k)
            frames.append(df)
    if frames:
        combo = pd.concat(frames).reset_index().set_index(["expiry","right","strike","time"]).sort_index()
        print(f"Combined DF shape: {combo.shape}")
        print(combo.head(args.rows))
    else:
        print("No data reloaded from repo.")


if __name__ == "__main__":
    main()
