#!/usr/bin/env python3
import argparse
import pandas as pd
from ibx_repos import EquityBarRepository

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/equity_bars")
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--trade-date", dest="trade_date", default=None,
                    help="YYYY-MM-DD (only for intraday partitions)")
    ap.add_argument("--rows", type=int, default=10)
    args = ap.parse_args()

    repo = EquityBarRepository(args.base)
    df = repo.load(symbol=args.symbol, trade_date=args.trade_date)

    print(f"Equity repo: {args.base}")
    print(f"shape={df.shape}, columns={list(df.columns)}")
    if not df.empty:
        # normalize timestamp if present
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.sort_values("time")
        print(df.head(args.rows).to_string(index=False))
    else:
        print("(empty)")

if __name__ == "__main__":
    main()
