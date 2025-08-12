"""
 earnings_ibapi_streamer.py
 -------------------------
 Event‑driven / non‑blocking variant that **streams** data from the official IB API as it arrives and
 writes it straight to disk – no ugly per‑request waits in the main flow.  The main thread fires off all
 requests up‑front, then just idles while the background *EWrapper* callbacks persist each fragment.

 Key design shifts vs the blocking sample
 ========================================
 1. **Fire‑and‑forget requests.** We register everything (daily/minute bars, chain snapshots, greeks) in one go.
 2. **Loader callbacks → parquet sink.** Every callback appends to an in‑memory buffer; when an *end* event
    fires we flush that buffer to a partitioned parquet file.
 3. **Atomic flush with PyArrow.** Keeps file I/O cheap and avoids pandas’ global‑interpreter‑lock issues.
 4. **Graceful shutdown.** We track outstanding tasks; once *all* end events fire we close the API and exit.

 Tested with TWS 10.24 / *ibapi* 10.24, Python 3.11.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from config import IB_HOST, IB_PORT, IB_CLIENT_ID, DATA_ROOT

# Logging is now configured globally in config.py
log = logging.getLogger(__name__)

from streamer import EarningsStreamer, Config

# ───────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli = argparse.ArgumentParser("earnings_ibapi_streamer")
    cli.add_argument("--symbol", required=True, help="Stock symbol to stream earnings for")
    cli.add_argument("--earnings", required=True, help="YYYY-MM-DD")
    ns = cli.parse_args()

    log.info(f"Starting streamer for {ns.symbol} with earnings date {ns.earnings}")

    # Configuration is now sourced from the central config file
    cfg = Config(
        symbol=ns.symbol.upper(),
        date=dt.datetime.strptime(ns.earnings, "%Y-%m-%d").date(),
        host=IB_HOST,
        port=IB_PORT,
        client_id=IB_CLIENT_ID,
        out_dir=DATA_ROOT,
        days_before=5,
        days_after=5
    )

    streamer = EarningsStreamer(cfg)
    streamer.run()
