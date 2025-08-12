#!/usr/bin/env python3
"""
Backfill job: compute *calculated* greeks from a simple as-of CSV/Parquet.
Input must contain columns: ts (UTC ISO), S, opt_price, r, q.
Option identity is passed by flags (underlying/expiry/right/strike).
"""
from __future__ import annotations
import argparse
import pandas as pd
from datetime import datetime, timezone

from ibx_flows.greeks_historical import OptionMeta, AsOfInputs, compute_table
from ibx_repos.greeks import CalculatedGreeksRepo

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', required=True)
    p.add_argument('--expiry', required=True, help='YYYYMMDD')
    p.add_argument('--right', required=True, choices=['C','P'])
    p.add_argument('--strike', required=True, type=float)
    p.add_argument('--input', required=True, help='CSV or Parquet with ts,S,opt_price[,r,q]')
    p.add_argument('--rate', type=float, default=None, help='fallback continuous r')
    p.add_argument('--yield', dest='div_yield', type=float, default=None, help='fallback continuous q')
    p.add_argument('--save-root', default='data/greeks')
    return p.parse_args()

def load_frame(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def main():
    args = parse_args()
    meta = OptionMeta(args.symbol, args.expiry, args.right, float(args.strike))

    df = load_frame(args.input).copy()
    for col in ['ts','S','opt_price']:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    if 'r' not in df.columns:
        if args.rate is None:
            raise SystemExit("Column 'r' absent and no --rate provided.")
        df['r'] = float(args.rate)
    if 'q' not in df.columns:
        df['q'] = float(args.div_yield or 0.0)

    inputs = [
        AsOfInputs(ts=row.ts.to_pydatetime(), S=float(row.S), opt_price=float(row.opt_price),
                   r=float(row.r), q=float(row.q))
        for row in df[['ts','S','opt_price','r','q']].itertuples(index=False)
    ]
    out = compute_table(meta, inputs)

    CalculatedGreeksRepo(args.save_root).save(out)
    print(f"Saved {len(out)} rows to {args.save_root} (partitioned).")

if __name__ == '__main__':
    main()