# --- NEW: subscriptions ---
import queue
import threading
from typing import Callable, Optional

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.ticktype import TickTypeEnum

from api.ib_core import IBRuntime, RequestIdSequencer, FutureRegistry


class Subscription:
    """Open-ended event stream for a reqId. Thread-safe delivery + optional callback."""
    def __init__(self, on_event: Optional[Callable[[dict], None]] = None, qsize: int = 1000):
        self._q = queue.Queue(maxsize=qsize)
        self._on_event = on_event
        self._closed = threading.Event()

    def push(self, event: dict):
        # fire user callback (non-blocking, protected)
        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                log.exception("Subscription on_event handler raised")
        # enqueue (drop oldest if full)
        try:
            self._q.put_nowait(event)
        except queue.Full:
            try:
                _ = self._q.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            finally:
                try:
                    self._q.put_nowait(event)
                except queue.Full:
                    pass  # extremely rare if consumer is fully stuck

    def close(self):
        self._closed.set()
        # sentinel to unblock consumers
        try:
            self._q.put_nowait({"type": "__closed__"})
        except queue.Full:
            pass

    def closed(self) -> bool:
        return self._closed.is_set()

    def get(self, timeout: Optional[float] = None) -> Optional[dict]:
        """Blocking get; returns None if closed."""
        if self._closed.is_set() and self._q.empty():
            return None
        item = self._q.get(timeout=timeout)
        if item.get("type") == "__closed__":
            return None
        return item

    def __iter__(self):
        while True:
            item = self.get()
            if item is None:
                break
            yield item

class SubscriptionRegistry:
    """reqId -> Subscription routing for streaming endpoints."""
    def __init__(self):
        self._lock = threading.Lock()
        self._by_id: Dict[int, Subscription] = {}

    def register(self, req_id: int, sub: Subscription):
        with self._lock:
            self._by_id[req_id] = sub
        return sub

    def dispatch(self, req_id: int, event: dict):
        with self._lock:
            sub = self._by_id.get(req_id)
        if sub:
            sub.push(event)

    def finish(self, req_id: int):
        with self._lock:
            sub = self._by_id.pop(req_id, None)
        if sub:
            sub.close()

    def set_error(self, req_id: int, code: int, msg: str):
        with self._lock:
            sub = self._by_id.pop(req_id, None)
        if sub:
            sub.push({"type": "error", "code": code, "message": msg})
            sub.close()

class SubscriptionHandle:
    """User handle to consume/cancel a subscription."""
    def __init__(self, rt: "IBRuntime", req_id: int, sub: Subscription):
        self._rt = rt
        self._req_id = req_id
        self._sub = sub

    @property
    def req_id(self) -> int:
        return self._req_id

    def cancel(self):
        try:
            self._rt.client.cancelMktData(self._req_id)
        finally:
            self._rt.subs.finish(self._req_id)

    def get(self, timeout: Optional[float] = None) -> Optional[dict]:
        return self._sub.get(timeout=timeout)

    def __iter__(self):
        return iter(self._sub)

# --- MODIFY: IBRuntime to create a SubscriptionRegistry and pass it to wrapper ---
# class IBRuntime:  (add/modify lines shown)
#   def __init__(...):
#       ...
#       self.sequencer = RequestIdSequencer()
#       self.registry = FutureRegistry()
#       self.subs = SubscriptionRegistry()                  # NEW
#       self.wrapper = IBWrapperBridge(self.sequencer, self.registry, self.subs)  # changed ctor
#       self.client = EClient(self.wrapper)

# --- MODIFY: IBWrapperBridge to accept subscriptions and route streaming ticks ---
class IBWrapperBridge(EWrapper):
    def __init__(self, sequencer: RequestIdSequencer, registry: FutureRegistry, subs: SubscriptionRegistry):
        super().__init__()
        self._seq = sequencer
        self._reg = registry
        self._subs = subs

    # ... existing methods (nextValidId, error → future, contractDetails*, historicalData*)
    # tweak error() to also route to subscriptions:
    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        if reqId in (-1, 0):
            log.warning("IB error (no reqId) code=%s msg=%s", errorCode, errorString)
            return
        # first try futures; if none, try subscriptions
        if not self._try_future_error(reqId, errorCode, errorString):
            self._subs.set_error(reqId, errorCode, errorString)

    def _try_future_error(self, reqId: int, code: int, msg: str) -> bool:
        # helper: attempt to set error on a waiting future
        with self._reg._lock:
            fut = self._reg._by_id.pop(reqId, None)
        if fut:
            fut.set_error(code, msg)
            return True
        return False

    # ---- STREAMING EVENTS → subscriptions ----
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
        self._subs.dispatch(reqId, {
            "type": "tickSize",
            "tick": TickTypeEnum.to_str(tickType),
            "size": size
        })

    def tickString(self, reqId, tickType, value):
        self._subs.dispatch(reqId, {
            "type": "tickString",
            "tick": TickTypeEnum.to_str(tickType),
            "value": value
        })

    def tickGeneric(self, reqId, tickType, value):
        self._subs.dispatch(reqId, {
            "type": "tickGeneric",
            "tick": TickTypeEnum.to_str(tickType),
            "value": value
        })

    def tickOptionComputation(self, reqId, tickType, impliedVol, delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
        self._subs.dispatch(reqId, {
            "type": "tickOption",
            "tick": TickTypeEnum.to_str(tickType),
            "impliedVol": impliedVol, "delta": delta, "optPrice": optPrice,
            "pvDividend": pvDividend, "gamma": gamma, "vega": vega, "theta": theta,
            "underPrice": undPrice
        })

    # Tick-by-tick (if you enable it)
    def tickByTickAllLast(self, reqId, tickType, time, price, size, tickAttribLast, exchange, specialConditions):
        self._subs.dispatch(reqId, {
            "type": "tbtLast", "subtype": "all" if tickType == 1 else "last",
            "time": time, "price": price, "size": size,
            "attrib": {
                "pastLimit": getattr(tickAttribLast, "pastLimit", None),
                "unreported": getattr(tickAttribLast, "unreported", None),
            },
            "exchange": exchange, "special": specialConditions
        })

    def tickByTickBidAsk(self, reqId, time, bidPrice, askPrice, bidSize, askSize, tickAttribBidAsk):
        self._subs.dispatch(reqId, {
            "type": "tbtBidAsk", "time": time,
            "bidPrice": bidPrice, "askPrice": askPrice,
            "bidSize": bidSize, "askSize": askSize,
            "attrib": {
                "bidPastLow": getattr(tickAttribBidAsk, "bidPastLow", None),
                "askPastHigh": getattr(tickAttribBidAsk, "askPastHigh", None),
            }
        })

    def tickByTickMidPoint(self, reqId, time, midPoint):
        self._subs.dispatch(reqId, {"type": "tbtMid", "time": time, "mid": midPoint})

# --- NEW: Subscription service for reqMktData ---
class SubscriptionService:
    def __init__(self, rt: IBRuntime):
        self.rt = rt

    def market_data(self, contract: Contract, genericTicks: str = "",
                    snapshot: bool = False, regulatorySnapshot: bool = False,
                    mktDataOptions=None,
                    on_event: Optional[Callable[[dict], None]] = None,
                    qsize: int = 1000) -> SubscriptionHandle:
        rid = self.rt.sequencer.next()
        sub = Subscription(on_event=on_event, qsize=qsize)
        self.rt.subs.register(rid, sub)
        self.rt.client.reqMktData(
            rid, contract, genericTicks, snapshot, regulatorySnapshot, mktDataOptions or []
        )
        return SubscriptionHandle(self.rt, rid, sub)
