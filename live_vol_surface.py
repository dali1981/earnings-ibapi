#!/usr/bin/env python3
"""
Standalone script to fetch implied volatilities (snapshot) for options and historical price data for the underlying from Interactive Brokers.

Usage:
    python vol_surface_client.py \
        --symbol AAPL \
        --host 127.0.0.1 --port 7497 --client-id 17 \
        --expirations 20250117,20250221 --strikes 100,110,120
"""
import argparse
import datetime as dt
from typing import List

from ibapi.contract import Contract

from request_sequencer import RequestSequencer
from ib_client import IBClient
from sink import Sink
from utils import RateLimiter, make_option, make_stock
from ibx_time import ib_end_datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch option implied vols (snapshot) and underlying historical prices from IB"
    )
    parser.add_argument("--symbol", required=True, help="Underlying symbol, e.g. AAPL")
    parser.add_argument("--host", default="127.0.0.1", help="IB gateway hostname")
    parser.add_argument("--port", type=int, default=7497, help="IB gateway port")
    parser.add_argument("--client-id", type=int, default=17, help="IB client ID")
    parser.add_argument(
        "--expirations", required=True,
        help="Comma-separated list of expiration dates (YYYYMMDD)"
    )
    parser.add_argument(
        "--strikes", required=True,
        help="Comma-separated list of strike prices"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    expirations = [e.strip() for e in args.expirations.split(',') if e.strip()]
    strikes: List[float] = [float(s) for s in args.strikes.split(',') if s.strip()]

    # Initialize requests, sink (prints to console), limiter, and IB client
    sequencer = RequestSequencer()
    sink = Sink(out_root=None, cfg=None)  # customize Sink to print or analyze results
    limiter = RateLimiter(1)
    client = IBClient(sequencer, sink, limiter)

    # Connect to IB
    client.connect(args.host, args.port, clientId=args.client_id)

    def on_ready(_first_id: int):
        # 1) Fetch underlying historical prices for last month (for realized vol)
        stock_contract: Contract = make_stock(args.symbol)
        end_ts = ib_end_datetime(dt.datetime.now(dt.timezone.utc))
        rid_hist = sequencer.next("hist_underlying")
        client.reqHistoricalData(
            rid_hist,
            stock_contract,
            endDateTime=end_ts,
            durationStr="1 M",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=0,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

        # 2) Fetch implied volatility snapshot for each option
        for expiry in expirations:
            for strike in strikes:
                opt_contract: Contract = make_option(
                    symbol=args.symbol,
                    expiry=expiry,
                    strike=strike,
                    currency="USD"
                )
                rid_iv = sequencer.next(f"iv_{expiry}_{strike}")
                # snapshot=True to get one-time implied vol (tick 106)
                client.reqMktData(
                    rid_iv,
                    opt_contract,
                    marketDataType="106",
                    snapshot=True,
                    regulatorySnapshot=False,
                    mktDataOptions=[]
                )

        # Disconnect after all requests complete
        sequencer.register_on_all_done(client.disconnect)

    # Register the ready callback and start the IB loop
    sequencer.when_ready(on_ready)
    client.run()

if __name__ == '__main__':
    main()
