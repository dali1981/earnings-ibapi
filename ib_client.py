# ————————————————————————————————————————————————————————————————
# Streaming EWrapper/EClient implementation
# ————————————————————————————————————————————————————————————————
import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

from ibapi.client import EClient
from ibapi.utils import iswrapper
from ibapi.wrapper import EWrapper

from request_sequencer import RequestSequencer
from response_sequencer import ResponseSequencer
from sink import Sink
from utils import append_parquet, RateLimiter

log = logging.getLogger(__name__)


class IBClient(EWrapper, EClient):
    """Streams data to disk; disconnects when all requests are done."""

    TIMEOUT = 120  # seconds per request

    def __init__(self,
                 requests: RequestSequencer,
                 responses: ResponseSequencer,
                 sink: Sink, limiter: RateLimiter):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)

        self.requests = requests
        self.responses = responses
        self.sink = sink
        self.limiter = limiter

        # self.thread = threading.Thread(target=self.run, daemon=True)

    # # — helpers —
    # def _next(self, tag: str) -> int:
    #     with self.lock:
    #         rid = self.req_id
    #         self.req_id += 1
    #         self.pending[rid] = tag
    #         return rid
    #
    # def _done(self, rid: int):
    #     tag = self.pending.pop(rid, None)
    #     if tag:
    #         logging.info("✔ %s complete (id=%d)", tag, rid)
    #     if not self.pending:
    #         logging.info("All requests done • disconnecting …")
    #         self.disconnect()

    def disconnect(self):  # just close the socket; main thread will join()
        if self.isConnected():
            super().disconnect()

    @iswrapper
    def nextValidId(self, orderId: int):
        self.requests.seed(orderId)
        log.info("nextValidId received: %d", orderId)

    # — error callback —
    @iswrapper
    def error(self, reqId: int, errorCode: int, errorString: str, *_):
        # Ignore common non‑fatal codes (e.g., 2104 market data farm) when reqId == -1
        # if reqId and reqId in self.pending:
        log.error("IB error %d on req %d: %s", errorCode, reqId, errorString)
        # self.failure = True
        # append_parquet(self.out_dir / f"error_{reqId}.parquet", self.buffers.pop(reqId, []))
        if reqId > 0:
            self.requests.done(reqId)

    @iswrapper
    def contractDetails(self, reqId: int, contractDetails):
        """Handles contract details request."""
        log.info("Contract details for reqId %d: %s", reqId, contractDetails)
        # self.sink.buffers[reqId] = {
        #     "contract": {
        #         "conId": contractDetails.contract.conId,
        #         "symbol": contractDetails.contract.symbol,
        #         "secType": contractDetails.contract.secType,
        #         "currency": contractDetails.contract.currency,
        #         "exchange": contractDetails.contract.exchange,
        #     },
        #     "details": {
        #         "longName": contractDetails.longName,
        #         "marketName": contractDetails.marketName,
        #         "minTick": contractDetails.minTick,
        #         "priceMagnifier": contractDetails.priceMagnifier,
        #     }
        # }

    @iswrapper
    def contractDetailsEnd(self, reqId: int):  # type: ignore
        """Handles end of contract details request."""
        log.info("Contract details end for reqId %d", reqId)
        self.requests.done(reqId)

    # — bar callbacks —
    @iswrapper
    def historicalData(self, reqId: int, bar):  # type: ignore
        # self.sink.historical_data(reqId, bar)
        self.responses.record(reqId, {
            "datetime": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "wap": bar.wap,
            "bar_count": bar.barCount,
        })

    @iswrapper
    def historicalDataEnd(self, reqId: int, start: str, end: str):  # type: ignore
        # self.sink.historical_data_end(reqId, start, end)
        self.responses.complete(reqId)
        self.requests.done(reqId)

    # — snapshot callbacks —
    @iswrapper
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib):  # type: ignore
        if tickType in (1, 2, 4, 6):
            # self.buffers[reqId].append({"tickType": tickType, "price": price})
            log.info(f"req id: {reqId} tickPrice received: {tickType}, price: {price}")
    @iswrapper
    def tickOptionComputation(self, reqId: int, tickType: int, *_):  # type: ignore
        self.sink.tick_option_computation(reqId, tickType)

    @iswrapper
    def tickSnapshotEnd(self, reqId: int):  # type: ignore
        self.sink.tick_snapshot_end(reqId)
        self.requests.done(reqId)

    @iswrapper
    def securityDefinitionOptionParameter(self, reqId: int, exchange: str, underlyingConId: int, tradingClass: str,
                                 multiplier: str, expirations, strikes):
        """Handles option secdef params request."""
        log.info("Option secdef params for reqId %d: exchange=%s, underlyingConId=%d, tradingClass=%s, "
                 "multiplier=%s, expirations=%s, strikes=%s", reqId, exchange, underlyingConId,
                 tradingClass, multiplier, expirations, strikes)
        self.sink.add_value(reqId,  {
            "exchange": exchange,
            "underlying_conid": underlyingConId,
            "trading_class": tradingClass,
            "multiplier": multiplier,
            "expirations": expirations,
            "strikes": strikes
        } )

    @iswrapper
    def securityDefinitionOptionParameterEnd(self, reqId: int):
        """Handles end of option secdef params request."""
        log.info("Option secdef params end for reqId %d", reqId)
        self.sink.option_chain(reqId)
        self.requests.done(reqId)