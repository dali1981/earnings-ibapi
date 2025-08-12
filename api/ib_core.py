# ib_core.py
import logging, threading
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

from api.subscriptions import SubscriptionRegistry

log = logging.getLogger("ib.core")

# ---------- ReqId sequencing ----------
class RequestIdSequencer:
    def __init__(self):
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._next: Optional[int] = None

    def set_base(self, order_id: int):
        with self._lock:
            self._next = order_id
        self._ready.set()

    def wait_ready(self, timeout: float = 10.0):
        if not self._ready.wait(timeout):
            raise RuntimeError("IB not ready (nextValidId not received)")

    def next(self, n: int = 1) -> int:
        with self._lock:
            if self._next is None:
                raise RuntimeError("Sequencer not initialized yet")
            rid = self._next
            self._next += n
            return rid

# ---------- Minimal future + registry ----------
class IBFuture:
    def __init__(self, expect_many: bool, timeout: float):
        self.expect_many = expect_many
        self.timeout = timeout
        self._ev = threading.Event()
        self._lock = threading.Lock()
        self._items: List[Any] = []
        self._error: Optional[Tuple[int, str]] = None

    def add(self, item: Any):
        with self._lock:
            self._items.append(item)

    def finish(self):
        self._ev.set()

    def set_error(self, code: int, msg: str):
        with self._lock:
            self._error = (code, msg)
        self._ev.set()

    def result(self) -> Any:
        if not self._ev.wait(self.timeout):
            raise TimeoutError(f"IB future timed out after {self.timeout}s")
        if self._error:
            code, msg = self._error
            raise RuntimeError(f"IB error {code}: {msg}")
        return self._items if self.expect_many else (self._items[0] if self._items else None)

class FutureRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._by_id: Dict[int, IBFuture] = {}

    def register(self, req_id: int, fut: IBFuture):
        with self._lock:
            self._by_id[req_id] = fut
        return fut

    def add_item(self, req_id: int, item: Any):
        with self._lock:
            fut = self._by_id.get(req_id)
        if fut:
            fut.add(item)

    def finish(self, req_id: int):
        with self._lock:
            fut = self._by_id.pop(req_id, None)
        if fut:
            fut.finish()

    def set_error(self, req_id: int, code: int, msg: str):
        with self._lock:
            fut = self._by_id.pop(req_id, None)
        if fut:
            fut.set_error(code, msg)
        else:
            log.warning("IB error reqId=%s code=%s: %s", req_id, code, msg)

# ---------- Wrapper: callbacks only ----------
class IBWrapperBridge(EWrapper):
    def __init__(self, sequencer: RequestIdSequencer, registry: FutureRegistry):
        super().__init__()
        self._seq = sequencer
        self._reg = registry
        self._subs = SubscriptionRegistry()

    # lifecycle / readiness
    def nextValidId(self, orderId: int):
        self._seq.set_base(orderId)

    # errors
    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        if reqId in (-1, 0):
            log.warning("IB error (no reqId) code=%s msg=%s", errorCode, errorString)
            return
        self._reg.set_error(reqId, errorCode, errorString)
        self._subs.set_error(reqId, errorCode, errorString)

    # ---- Contract Details ----
    def contractDetails(self, reqId, contractDetails):
        self._reg.add_item(reqId, contractDetails)

    def contractDetailsEnd(self, reqId):
        self._reg.finish(reqId)

    # ---- SecDef Option Params ----
    def securityDefinitionOptionParameter(self, reqId, exchange, underlyingConId, tradingClass, multiplier,
                                          expirations, strikes):
        self._reg.add_item(reqId, {
            "exchange": exchange,
            "underlyingConId": underlyingConId,
            "tradingClass": tradingClass,
            "multiplier": multiplier,
            "expirations": list(expirations),
            "strikes": list(strikes),
        })

    def securityDefinitionOptionParameterEnd(self, reqId):
        self._reg.finish(reqId)

    # ---- Historical Data ----
    def historicalData(self, reqId, bar):
        self._reg.add_item(reqId, {
            "date": bar.date, "open": bar.open, "high": bar.high,
            "low": bar.low, "close": bar.close, "volume": bar.volume,
            "wap": getattr(bar, "wap", None), "barCount": getattr(bar, "barCount", None)
        })

    def historicalDataEnd(self, reqId, start, end):
        self._reg.finish(reqId)

        # ---- STREAMING MARKET DATA ----
        def marketDataType(self, reqId, marketDataType):
            self._subs.dispatch(reqId, {"type": "marketDataType", "mode": marketDataType})

        def tickPrice(self, reqId, tickType, price, attrib):
            self._subs.dispatch(reqId, {
                "type": "tickPrice",
                "tick": TickTypeEnum.to_str(tickType),
                "price": price,
                "attrib": {
                    "canAutoExecute": getattr(attrib, "canAutoExecute", None),
                    "pastLimit": getattr(attrib, "pastLimit", None),
                    "preOpen": getattr(attrib, "preOpen", None),
                }
            })

        def tickSize(self, reqId, tickType, size):
            self._subs.dispatch(reqId, {"type": "tickSize", "tick": TickTypeEnum.to_str(tickType), "size": size})

        def tickString(self, reqId, tickType, value):
            self._subs.dispatch(reqId, {"type": "tickString", "tick": TickTypeEnum.to_str(tickType), "value": value})

        def tickGeneric(self, reqId, tickType, value):
            self._subs.dispatch(reqId, {"type": "tickGeneric", "tick": TickTypeEnum.to_str(tickType), "value": value})

        def tickOptionComputation(self, reqId, tickType, impliedVol, delta, optPrice, pvDividend, gamma, vega, theta,
                                  undPrice):
            self._subs.dispatch(reqId, {
                "type": "tickOption", "tick": TickTypeEnum.to_str(tickType),
                "impliedVol": impliedVol, "delta": delta, "optPrice": optPrice,
                "pvDividend": pvDividend, "gamma": gamma, "vega": vega, "theta": theta,
                "underPrice": undPrice
            })

# ---------- Runtime: connection + run thread ----------
class IBRuntime:
    def __init__(self, host="127.0.0.1", port=7497, client_id=1):
        self.host, self.port, self.client_id = host, port, client_id
        self.sequencer = RequestIdSequencer()
        self.registry = FutureRegistry()
        self.subs = SubscriptionRegistry()
        self.wrapper = IBWrapperBridge(self.sequencer, self.registry,
                                         self.subs)
        self.client = EClient(self.wrapper)

        self._run_thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()

    def start(self, ready_timeout: float = 10.0):
        if self.client.isConnected():
            return
        self.client.connect(self.host, self.port, self.client_id)
        self._run_thread = threading.Thread(target=self.client.run, name="ib-reader", daemon=False)
        self._run_thread.start()
        self.sequencer.wait_ready(ready_timeout)

    def stop(self, join_timeout: float = 5.0):
        self._stopped.set()
        try:
            if self.client.isConnected():
                self.client.disconnect()
        finally:
            t = self._run_thread
            if t and t.is_alive():
                t.join(timeout=join_timeout)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

# ---------- Services (composition) ----------
class ContractDetailsService:
    def __init__(self, rt: IBRuntime):
        self.rt = rt

    def fetch(self, contract: Contract, timeout: float = 10.0) -> List[Any]:
        rid = self.rt.sequencer.next()
        fut = self.rt.registry.register(rid, IBFuture(expect_many=True, timeout=timeout))
        self.rt.client.reqContractDetails(rid, contract)
        return fut.result()

class SecDefService:
    def __init__(self, rt: IBRuntime):
        self.rt = rt

    def option_params(self, symbol: str, conid: int, sec_type="STK", exchange: str = "",
                      timeout: float = 10.0) -> List[Dict[str, Any]]:
        rid = self.rt.sequencer.next()
        fut = self.rt.registry.register(rid, IBFuture(expect_many=True, timeout=timeout))
        self.rt.client.reqSecDefOptParams(rid, symbol, exchange, sec_type, conid)
        return fut.result()

class HistoricalService:
    def __init__(self, rt: IBRuntime):
        self.rt = rt

    def bars(self, contract: Contract, endDateTime: str, durationStr: str,
             barSizeSetting: str, whatToShow: str, useRTH: int,
             formatDate: int = 1, keepUpToDate: bool = False, chartOptions=None,
             timeout: float = 20.0) -> List[Dict[str, Any]]:
        rid = self.rt.sequencer.next()
        fut = self.rt.registry.register(rid, IBFuture(expect_many=True, timeout=timeout))
        self.rt.client.reqHistoricalData(
            rid, contract, endDateTime, durationStr, barSizeSetting,
            whatToShow, useRTH, formatDate, keepUpToDate, chartOptions or []
        )
        return fut.result()

# ---------- Small helper ----------
def make_stock(symbol: str, exch="SMART", curr="USD") -> Contract:
    c = Contract(); c.symbol=symbol; c.secType="STK"; c.exchange=exch; c.currency=curr
    return c
