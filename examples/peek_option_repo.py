#!/usr/bin/env python3
import argparse
import pandas as pd
from ibx_repos import OptionBarRepository

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/option_bars")
    ap.add_argument("--underlying", required=True)
    ap.add_argument("--expiry", required=True, help="YYYYMMDD")
    ap.add_argument("--right", choices=["C","P"], required=True)
    ap.add_argument("--strike", type=float, default=None)
    ap.add_argument("--trade-date", dest="trade_date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--rows", type=int, default=10)
    args = ap.parse_args()

    repo = OptionBarRepository(args.base)
    df = repo.load(
        underlying=args.underlying,
        expiry=args.expiry,
        right=args.right,
        strike=args.strike,
        trade_date=args.trade_date,
    )

    print(f"Option repo: {args.base}")
    print(f"{args.underlying} {args.expiry} {args.right} {args.strike if args.strike is not None else ''}")
    print(f"shape={df.shape}, columns={list(df.columns)}")
    if not df.empty:
        # normalize/clean for display
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.sort_values("time").drop_duplicates(subset=["time"])
        cols = [c for c in ["time","open","high","low","close","volume","wap","bar_count"] if c in df.columns]
        print(df[cols].head(args.rows).to_string(index=False))
    else:
        print("(empty)")

if __name__ == "__main__":
    main()
