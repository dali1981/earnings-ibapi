"""
Utilities package for trading project.

Provides common utilities like logging setup, configuration helpers, etc.
Also exposes light-weight IO helpers expected by callers (e.g., append_parquet).
"""

from .logging_setup import (
    disable_noisy_loggers,
    get_logger,
    setup_logging,
    setup_script_logging,
)
from pathlib import Path

# Minimal IO helper re-exposed here to avoid import confusion between the
# top-level `utils.py` module and this `utils` package.
from typing import Any, Dict, List


try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional in some environments
    pa = None
    pq = None


def append_parquet(path: Path, records: List[Dict[str, Any]]):
    """Append records to a parquet file (create if missing).

    This mirrors the implementation in the root-level `utils.py` to provide a
    stable import path: `from utils import append_parquet`.
    """
    if not records:
        return
    if pa is None or pq is None:
        raise RuntimeError("pyarrow not available for append_parquet")
    table = pa.Table.from_pylist(records)
    if path.exists():
        existing = pq.read_table(path)
        table = pa.concat_tables([existing, table])
    pq.write_table(table, path)


class RateLimiter:
    """Simple token bucket rate limiter (re-exported for compatibility)."""

    def __init__(self, max_tokens: int = 60, refill_seconds: int = 600):
        import threading as _th
        import time as _t

        self.max_tokens = max_tokens
        self.refill_seconds = refill_seconds
        self.tokens = max_tokens
        self.last = _t.monotonic()
        self.lock = _th.Lock()

    def acquire(self):
        import time as _t

        while True:
            with self.lock:
                now = _t.monotonic()
                elapsed = now - self.last
                if elapsed > 0:
                    restored = int(elapsed * self.max_tokens / self.refill_seconds)
                    if restored:
                        self.tokens = min(self.max_tokens, self.tokens + restored)
                        self.last = now
                if self.tokens > 0:
                    self.tokens -= 1
                    return
            _t.sleep(0.2)


# IB contract helpers (compat exports)
def make_stock(symbol: str, exchange: str = "SMART", currency: str = "USD"):
    from ibapi.contract import Contract

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
    exchange: str = "SMART",
    trading_class: str | None = None,
    multiplier: str | int | None = "100",
    currency: str = "USD",
):
    from ibapi.contract import Contract

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


__all__ = [
    "setup_logging",
    "get_logger",
    "setup_script_logging",
    "disable_noisy_loggers",
    "append_parquet",
    "RateLimiter",
    "make_stock",
    "make_option",
]
