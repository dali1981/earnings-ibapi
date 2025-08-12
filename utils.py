# ───────────────────────────────────────────────────────────────────────────────
# Helpers – construct official IB *Contract* objects
# ───────────────────────────────────────────────────────────────────────────────
import threading
import time
from pathlib import Path
from typing import List, Dict, Any
import pyarrow as pa
import pyarrow.parquet as pq

from ibapi.contract import Contract


def make_stock(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = exchange
    c.currency = currency
    return c


def make_option(
    symbol: str,
    expiry: str,  # YYYYMMDD
    strike: float,
    right: str,  # "C" / "P"
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

# ───────────────────────────────────────────────────────────────────────────────
# Rate limiter – token bucket
# ───────────────────────────────────────────────────────────────────────────────

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

# ───────────────────────────────────────────────────────────────────────────────
# Arrow sink helpers – incremental parquet writing
# ───────────────────────────────────────────────────────────────────────────────

def append_parquet(path: Path, records: List[Dict[str, Any]]):
    if not records:
        return
    table = pa.Table.from_pylist(records)
    if path.exists():
        existing = pq.read_table(path)
        table = pa.concat_tables([existing, table])
    pq.write_table(table, path)
