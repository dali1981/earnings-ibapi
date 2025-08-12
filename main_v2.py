"""
 earnings_ibapi_streamer_v2.py  ▸  _lifecycle‑fix_
 ------------------------------------------------
 Removes the busy‑wait loop that kept the script alive and sometimes left the
 socket open when you hit *Stop* in your IDE.  The main thread now simply
 **joins the IB reader thread**; when `IBClient` calls `disconnect()`, the reader
 loop exits → `join()` returns → script ends → port is free for the next run.

 Key adjustments
 ===============
 • **No while‑sleep loop.** `EarningsStreamer.run()` blocks on
   `self.sink.thread.join()` instead.
 • **`IBClient.disconnect()` no longer calls `join()`** (that could deadlock if
   the disconnect originates from the reader thread itself).
 • Everything else (rate‑limiter, parquet sink, random clientId) is unchanged.
"""

from __future__ import annotations

import datetime as dt
import logging
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pyarrow as pa
import pyarrow.parquet as pq
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.utils import iswrapper

# ————————————————————————————————————————————————————————————————
# Logging (set level to DEBUG to see wire trace)
# ————————————————————————————————————————————————————————————————
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ib_logger = logging.getLogger("ibapi.utils")
ib_logger.setLevel(logging.INFO)

# ————————————————————————————————————————————————————————————————
# Contract helpers (official ibapi uses a single Contract class)
# ————————————————————————————————————————————————————————————————

def make_stock(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = exchange
    c.currency = currency
    return c


def make_option(
    symbol: str,
    expiry: str,
    strike: float,
    right: str,
    *,
    exchange: str = "SMART",
    trading_class: str | None = None,
    multiplier: str | int | None = "100",
    currency: str = "USD",
) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "OPT"
    c.exchange = exchange
    c.currency = currency
    c.lastTradeDateOrContractMonth = expiry
    c.strike = strike
    c.right = right
    if trading_class:
        c.tradingClass = trading_class
    if multiplier:
        c.multiplier = str(multiplier)
    return c

# ————————————————————————————————————————————————————————————————
# Simple token‑bucket pacing
# ————————————————————————————————————————————————————————————————

class RateLimiter:
    def __init__(self, max_tokens: int = 60, refill_seconds: int = 600):
        self.max_tokens = max_tokens
        self.refill_seconds = refill_seconds
        self.tokens = max_tokens
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last
                if elapsed > 0:
                    restored = int(elapsed * self.max_tokens / self.refill_seconds)
                    if restored:
                        self.tokens = min(self.max_tokens, self.tokens + restored)
                        self.last = now
                if self.tokens > 0:
                    self.tokens -= 1
                    return
            time.sleep(0.2)

# ————————————————————————————————————————————————————————————————
# Parquet append helper
# ————————————————————————————————————————————————————————————————

def append_parquet(path: Path, records: List[Dict[str, Any]]):
    if not records:
        return
    table = pa.Table.from_pylist(records)
    if path.exists():
        table = pa.concat_tables([pq.read_table(path), table])
    pq.write_table(table, path)

# ————————————————————————————————————————————————————————————————
# Streaming EWrapper/EClient implementation
# ————————————————————————————————————————————————————————————————

class IBSink(EWrapper, EClient):
    """Streams data to disk; disconnects when all requests are done."""

    def __init__(self, out_dir: Path, limiter: RateLimiter):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        self.out_dir = out_dir
        self.limiter = limiter
        self.req_id = 1
        self.lock = threading.Lock()
        self.pending: Dict[int, str] = {}
        self.buffers: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    # — helpers —
    def _next(self, tag: str) -> int:
        with self.lock:
            rid = self.req_id
            self.req_id += 1
            self.pending[rid] = tag
            return rid

    def _done(self, rid: int):
        tag = self.pending.pop(rid, None)
        if tag:
            logging.info("✔ %s complete (id=%d)", tag, rid)
        if not self.pending:
            logging.info("All requests done • disconnecting …")
            self.disconnect()

    def disconnect(self):  # just close the socket; main thread will join()
        if self.isConnected():
            super().disconnect()

    # — bar callbacks —
    @iswrapper
    def historicalData(self, reqId: int, bar):  # type: ignore
        self.buffers[reqId].append(
            {
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "wap": bar.average,
            }
        )

    @iswrapper
    def historicalDataEnd(self, reqId: int, start: str, end: str):  # type: ignore
        append_parquet(self.out_dir / f"bars_{reqId}.parquet", self.buffers.pop(reqId, []))
        self._done(reqId)

    # — snapshot callbacks —
    @iswrapper
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib):  # type: ignore
        if tickType in (1, 2, 4, 6):
            self.buffers[reqId].append({"tickType": tickType, "price": price})

    @iswrapper
    def tickOptionComputation(self, reqId: int, tickType: int, *_):  # type: ignore
        iv, delta, gamma, vega, theta = _[:5]
        self.buffers[reqId].append(
            {
                "tickType": tickType,
                "iv": iv,
                "delta": delta,
                "gamma": gamma,
                "vega": vega,
                "theta": theta,
            }
        )

    @iswrapper
    def tickSnapshotEnd(self, reqId: int):  # type: ignore
        append_parquet(self.out_dir / f"snapshot_{reqId}.parquet", self.buffers.pop(reqId, []))
        self._done(reqId)

# ————————————————————————————————————————————————————————————————
# High‑level orchestrator
# ————————————————————————————————————————————————————————————————

@dataclass
class EarningsStreamer:
    symbol: str
    earnings_date: dt.date
    host: str = "127.0.0.1"
    port: int = 4002 # 7497
    client_id: int = field(default_factory=lambda: random.randint(10000, 99999))
    exchange: str = "SMART"
    currency: str = "USD"
    strikes_width: int = 3
    minute_window: Tuple[int, int] = (-2, 2)
    days_before: int = 30# 365
    days_after: int = 5
    out_root: Path = Path("./earnings_data")
    limiter: RateLimiter = field(default_factory=RateLimiter, init=False)
    sink: IBSink = field(init=False)

    def run(self):
        out_dir = self.out_root / f"{self.symbol}_{self.earnings_date:%Y%m%d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.sink = IBSink(out_dir, self.limiter)
        self.sink.connect(self.host, self.port, clientId=self.client_id)
        while not self.sink.isConnected():
            time.sleep(0.05)
        logging.info("Connected (clientId=%d) – submitting requests …", self.client_id)
        self._submit_all()
        # Wait until reader thread finishes (disconnect triggers exit)
        self.sink.thread.join()
        logging.info("Done – socket closed, exiting.")

    # — submission helpers —
    def _stk(self):
        return make_stock(self.symbol, self.exchange, self.currency)

    def _submit_all(self):
        stk = self._stk()
        end = (self.earnings_date + dt.timedelta(days=self.days_after)).strftime("%Y%m%d %H:%M:%S")
        rid = self

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("earnings_ibapi_streamer_v2")
    parser.add_argument("--symbol",   required=True, help="Underlying ticker, e.g. IBM")
    parser.add_argument("--earnings", required=True, help="Earnings date YYYY-MM-DD")
    parser.add_argument("--port",     type=int, default=7497, help="IB API port (TWS 7497 / Gateway 4002)")
    parser.add_argument("--client-id", type=int, default=None, help="Override random clientId")

    args = parser.parse_args()

    streamer = EarningsStreamer(
        symbol=args.symbol.upper(),
        earnings_date=dt.datetime.strptime(args.earnings, "%Y-%m-%d").date(),
        port=args.port,
        client_id=args.client_id or random.randint(10000, 99999),
    )
    streamer.run()
