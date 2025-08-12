
"""
Standalone script to fetch implied and historical volatilities for an option vol surface from Interactive Brokers.

Usage:
    python vol_surface_client.py \
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
from streamer import Config
from utils import RateLimiter, make_option
from ibx_time import ib_end_datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch vol surface data from Interactive Brokers"
    )
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

    cfg = Config(symbol='IBM',  # Example symbol, adjust as needed
                 date=dt.datetime.strptime("2024-07-10", "%Y-%m-%d").date(),
                 client_id=args.client_id,
                 port=4002,  # TWS paper trading port
                 days_before=5, days_after=5)
    # Initialize requests, sink (prints to console), limiter, and IB client
    sequencer = RequestSequencer()
    sink = Sink(out_dir=cfg.out_dir)  # adjust Sink implementation to log/print
    limiter = RateLimiter(1)
    client = IBClient(sequencer, sink, limiter)

    # Connect to IB
    client.connect(args.host, args.port, clientId=args.client_id)

    # Define the on-ready callback to send all vol requests
    def on_ready(_first_id: int):
        # Loop through expirations and strikes
        for expiry in expirations:
            for strike in strikes:
                # Build option contract
                opt_contract: Contract = make_option(
                    symbol=cfg.symbol,
                    expiry=expiry,
                    strike=strike,
                    currency="USD",
                    right="C",  # Call options, change to "P" for puts
                )
                # Implied volatility (tick 106)
                # rid_iv = requests.next(f"iv_{expiry}_{strike}")
                # client.reqMktData(rid_iv, opt_contract, "106", False, False, [])
                # Historical volatility (1 month daily bars)
                end_ts = ib_end_datetime(dt.datetime.now(dt.timezone.utc))
                rid_hv = sequencer.next(f"hv_{expiry}_{strike}")
                client.reqHistoricalData(
                    rid_hv,
                    opt_contract,
                    endDateTime=end_ts,
                    durationStr="1 M",
                    barSizeSetting="1 day",
                    whatToShow="OPTION_IMPLIED_VOLATILITY",
                    useRTH=0,
                    formatDate=1,
                    keepUpToDate=False,
                    chartOptions=[]
                )
        # Disconnect after all requests complete
        sequencer.register_on_all_done(client.disconnect)

    # Register on-ready and start the client loop
    sequencer.when_ready(on_ready)
    client.run()

if __name__ == '__main__':
    main()
