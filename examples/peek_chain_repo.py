#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from ibx_repos import ChainRepository

def _to_list(x):
    """Return a Python list for any list-like; empty list for None/NA."""
    if x is None:
        return []
    # Treat pandas NA
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass
    # Common containers
    if isinstance(x, (list, tuple)):
        return list(x)
    # NumPy array
    if isinstance(x, np.ndarray):
        return x.tolist()
    # Fallback: try list() (works for many Arrow list scalars)
    try:
        return list(x)
    except Exception:
        return [x]

def _safe_len(x) -> int:
    try:
        return len(_to_list(x))
    except Exception:
        return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/option_chains")
    ap.add_argument("--underlying", required=True)
    ap.add_argument("--rows", type=int, default=5)
    args = ap.parse_args()

    repo = ChainRepository(args.base)
    df = repo.load(underlying=args.underlying)

    print(f"Chain repo: {args.base} (underlying={args.underlying})")
    if df.empty:
        print("(empty)")
        return

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    latest = df["snapshot_date"].max()
    snap = df[df["snapshot_date"] == latest].copy()

    # Flatten expirations robustly
    exp_lists = snap["expirations"].apply(_to_list)
    expiries = sorted({e for e in exp_lists.explode().dropna().tolist()})

    print(
        f"latest snapshot: {latest.date()}, "
        f"trading_classes={snap['trading_class'].nunique(dropna=True)}, "
        f"exchanges={snap['exchange'].nunique(dropna=True)}, "
        f"expiries={len(expiries)}"
    )

    # Small table preview
    rows = []
    for row in snap.head(args.rows).itertuples(index=False):
        rows.append({
            "exchange": row.exchange,
            "trading_class": row.trading_class,
            "expirations": _safe_len(getattr(row, "expirations", None)),
            "strikes": _safe_len(getattr(row, "strikes", None)),
        })

    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
