#!/usr/bin/env python3
"""
Legacy CLI refactor: compute *calculated* greeks by fetching historical bars
from IB via your `ibx` runtime/services. Writes to CalculatedGreeksRepo.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from typing import List
import pandas as pd

from ibx.runtime import IBRuntime
from ibx.services import HistoricalService
from ibx.contracts import make_stock, make_option

from ibx_time.timebox import to_ib_end_datetime, ensure_utc
from ibx_flows.greeks_historical import OptionMeta, AsOfInputs, compute_table
from ibx_repos.greeks import CalculatedGreeksRepo

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', required=True)
    p.add_argument('--expiry', required=True, help='YYYYMMDD')
    p.add_argument('--right', required=True, choices=['C','P'])
    p.add_argument('--strike', required=True, type=float)
    p.add_argument('--duration', default='1 M')
    p.add_argument('--bar-size', default='1 day', choices=['1 day','1 hour','15 mins','5 mins','1 min'])
    p.add_argument('--rate', type=float, default=0.03, help='continuous r')
    p.add_argument('--yield', dest='div_yield', type=float, default=0.0, help='continuous q')
    p.add_argument('--end-utc', default='', help='YYYYMMDD HH:MM:SS (UTC); empty for now')
    p.add_argument('--save-root', default='data/greeks')
    return p.parse_args()

def main():
    args = parse_args()
    meta = OptionMeta(args.symbol, args.expiry, args.right, float(args.strike))

    with IBRuntime() as rt:
        hist = HistoricalService(rt)
        # Underlying bars
        u_rows = hist.fetch(
            make_stock(args.symbol),
            endDateTime=args.end_utc or "",
            durationStr=args.duration,
            barSizeSetting=args.bar_size,
            whatToShow="TRADES",
            useRTH=1,
            formatDate=2
        )
        # Option bars (MIDPOINT preferred for pricing)
        o_rows = hist.fetch(
            make_option(args.symbol, args.expiry, float(args.strike), args.right),
            endDateTime=args.end_utc or "",
            durationStr=args.duration,
            barSizeSetting=args.bar_size,
            whatToShow="MIDPOINT",
            useRTH=1,
            formatDate=2
        )

    # Convert rows -> DataFrames; expect 'date','open','high','low','close','volume'
    df_u = pd.DataFrame(u_rows).rename(columns={'date':'ts','close':'S'})[['ts','S']]
    df_o = pd.DataFrame(o_rows).rename(columns={'date':'ts','close':'opt_price'})[['ts','opt_price']]
    # Ensure tz-aware UTC
    df_u['ts'] = pd.to_datetime(df_u['ts'], utc=True)
    df_o['ts'] = pd.to_datetime(df_o['ts'], utc=True)

    df = df_u.merge(df_o, on='ts', how='inner')
    df['r'] = float(args.rate)
    df['q'] = float(args.div_yield)

    inputs = [
        AsOfInputs(ts=row.ts.to_pydatetime(), S=float(row.S), opt_price=float(row.opt_price),
                   r=float(row.r), q=float(row.q))
        for row in df.itertuples(index=False)
    ]
    out = compute_table(meta, inputs)

    CalculatedGreeksRepo(args.save_root).save(out)
    print(f"Saved {len(out)} rows to {args.save_root} (partitioned).")

if __name__ == '__main__':
    main()
