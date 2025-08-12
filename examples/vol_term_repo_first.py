#!/usr/bin/env python3
"""
vol_term_repo_first.py — ATM implied‑vol term structure (+skew) with realized‑vol overlays

Repo‑first design:
- Uses **option price bars from your options repo** to compute IV & delta at AS‑OF.
- If a needed bar is missing, fetches minimal bars from IB **once**, **persists**
  to the repo, then computes. Streaming greeks are only a last resort when AS‑OF is None.

Run example:
  python vol_term_repo_first.py \
    --symbol AAPL --as-of "20250808 15:50:00" \
    --port 4002 --client-id 42 \
    --chains-base data/option_chains \
    --options-base data/option_bars \
    --limit 12 --lookbacks 10,20,60 \
    --plot --save aapl_term.png
"""
from __future__ import annotations

# --- project-root import shim (so this runs from examples/ without installing) ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------------------------------------------

import argparse
import math
import time
from datetime import date
from typing import Iterable, List, Optional, Tuple, Dict

import pandas as pd

from ibx import IBRuntime, SubscriptionService, HistoricalService, make_stock, make_option
from ibx_repos import ChainRepository, OptionBarRepository
from ibx.time import ib_end_datetime_instrument
from ibx.greeks_repo_first import option_greeks_repo_first

# -------------------- small utils --------------------

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
        return list(x)  # pyarrow list scalar etc.
    except Exception:
        return [x]


def _nearest_strike(strikes: Iterable[float], S: float) -> float:
    arr = sorted({float(s) for s in strikes})
    return min(arr, key=lambda k: abs(k - float(S)))

# -------------------- spot at AS-OF --------------------

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

# -------------------- chain snapshot --------------------

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

# -------------------- main computation --------------------

def compute_atm_vol_term_repo_first(
    rt: IBRuntime,
    chain_repo: ChainRepository,
    opt_repo: OptionsRepository,
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
        iv_c, d_c = option_greeks_repo_first(hist, opt_repo, subs, symbol, expiry, k_atm, "C", as_of, S_asof)
        iv_p, d_p = option_greeks_repo_first(hist, opt_repo, subs, symbol, expiry, k_atm, "P", as_of, S_asof)

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
            "expiry_dt": exp_dt,
            "ttm_years": ttm_years,
            "k_atm": float(k_atm),
            "spot_asof": float(S_asof),
            "iv_call": iv_c,
            "iv_put": iv_p,
            "iv_atm": iv_atm,
            "skew": (None if (iv_c is None or iv_p is None) else (iv_c - iv_p)),
            "delta_call": d_c,
            "delta_put": d_p,
        })

    df = pd.DataFrame(rows).sort_values("expiry").reset_index(drop=True)
    return df

# -------------------- realized (historical) volatility --------------------

def compute_realized_vol(
    hist: HistoricalService,
    symbol: str,
    as_of: Optional[str],
    lookbacks: List[int],
) -> Dict[int, float]:
    if not lookbacks:
        return {}
    max_lb = max(lookbacks)
    duration = f"{max_lb + 40} D"
    end = ""
    if as_of:
        end = ib_end_datetime_instrument(hist.rt, make_stock(symbol), as_of, hyphen=False)
    rows = hist.bars(
        make_stock(symbol),
        endDateTime=end,
        durationStr=duration,
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=0,
        timeout=90.0,
    )
    if not rows:
        return {lb: float("nan") for lb in lookbacks}
    df = pd.DataFrame(rows)
    closes = pd.to_numeric(df.get("close"), errors="coerce").dropna()
    if closes.size < 2:
        return {lb: float("nan") for lb in lookbacks}
    rets = (closes.astype(float).pct_change().apply(lambda x: math.log1p(x))).dropna()
    out: Dict[int, float] = {}
    for lb in lookbacks:
        if rets.size >= lb:
            vol = float(rets.tail(lb).std(ddof=1)) * (252.0 ** 0.5)
            out[lb] = vol
        else:
            out[lb] = float("nan")
    return out

# -------------------- plotting --------------------

def plot_term(df: pd.DataFrame, lookbacks: List[int], title: str | None = None, save: str | None = None):
    import matplotlib.pyplot as plt

    x = df["ttm_years"].astype(float)
    y = df["iv_atm"].astype(float) * 100.0

    fig1, ax1 = plt.subplots()
    ax1.plot(x, y, marker="o", label="ATM IV")
    if df["iv_call"].notna().any():
        ax1.plot(x, df["iv_call"].astype(float) * 100.0, marker="x", linestyle="--", label="Call IV")
    if df["iv_put"].notna().any():
        ax1.plot(x, df["iv_put"].astype(float) * 100.0, marker="x", linestyle=":", label="Put IV")

    xmax = float(x.max()) if len(x) else 1.0
    for lb in lookbacks:
        col = f"hv_{lb}"
        if col in df.columns and pd.notna(df[col]).any():
            hvp = float(df[col].dropna().iloc[0]) * 100.0
            ax1.hlines(hvp, xmin=0.0, xmax=xmax * 1.05, linestyles="dashed", label=f"HV{lb}")

    ax1.set_xlabel("TTM (years)")
    ax1.set_ylabel("Vol (%)")
    ax1.set_title(title or "ATM Term Structure with HV Overlays")
    ax1.legend()

    if df["skew"].notna().any():
        fig2, ax2 = plt.subplots()
        ax2.plot(x, df["skew"].astype(float) * 100.0, marker="o", label="ATM Skew (C−P)")
        ax2.axhline(0, linestyle="dotted")
        ax2.set_xlabel("TTM (years)")
        ax2.set_ylabel("Skew (pp)")
        ax2.set_title("ATM Skew vs TTM")
        ax2.legend()

    if save:
        base = save.rsplit(".", 1)[0]
        fig1.savefig(f"{base}_term.png", bbox_inches="tight", dpi=160)
        if df["skew"].notna().any():
            fig2.savefig(f"{base}_skew.png", bbox_inches="tight", dpi=160)
    else:
        plt.show()

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--as-of", default=None, help='AS-OF time "YYYYMMDD HH:MM:SS" (instrument TZ applied).')
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4002)
    ap.add_argument("--client-id", type=int, default=77)
    ap.add_argument("--chains-base", default="data/option_chains")
    ap.add_argument("--options-base", default="data/option_bars")
    ap.add_argument("--limit", type=int, default=10, help="Max expiries to include")
    ap.add_argument("--lookbacks", default="10,20,60", help="CSV of day windows for realized vol")
    ap.add_argument("--rows", type=int, default=50)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--save", default=None, help="Path prefix to save figures (e.g., term.png)")
    args = ap.parse_args()

    lookbacks = [int(x) for x in str(args.lookbacks).split(",") if str(x).strip()]

    chain_repo = ChainRepository(args.chains_base)
    opt_repo = OptionBarRepository(args.options_base)

    with IBRuntime(host=args.host, port=args.port, client_id=args.client_id) as rt:
        subs = SubscriptionService(rt)
        hist = HistoricalService(rt)
        df = compute_atm_vol_term_repo_first(rt, chain_repo, opt_repo, hist, subs, args.symbol, args.as_of, limit=args.limit)
        hv = compute_realized_vol(hist, args.symbol, args.as_of, lookbacks)

    if df.empty:
        print("(empty)")
        return

    for lb, val in hv.items():
        df[f"hv_{lb}"] = val

    show = df.head(args.rows).copy()
    for c in ("iv_call", "iv_put", "iv_atm"):
        if c in show:
            show[c] = (show[c] * 100.0).round(2)
    for lb in lookbacks:
        c = f"hv_{lb}"
        if c in show:
            show[c] = (show[c] * 100.0).round(2)
    if "skew" in show:
        show["skew"] = (show["skew"] * 100.0).round(2)

    print(show.to_string(index=False))

    if args.plot:
        title = f"{args.symbol} ATM Term ({args.as_of or 'now'})"
        plot_term(df, lookbacks, title=title, save=args.save)

if __name__ == "__main__":
    main()
