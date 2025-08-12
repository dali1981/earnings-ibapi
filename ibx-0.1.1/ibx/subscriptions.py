
import threading, queue
from typing import Any, Dict, Optional, Callable

class Subscription:
    """Open-ended event stream for a reqId."""
    def __init__(self, on_event: Optional[Callable[[dict], None]] = None, qsize: int = 1000):
        self._q = queue.Queue(maxsize=qsize)
        self._on_event = on_event
        self._closed = threading.Event()

    def push(self, event: Dict[str, Any]):
        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass
        try:
            self._q.put_nowait(event)
        except queue.Full:
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            finally:
                try:
                    self._q.put_nowait(event)
                except queue.Full:
                    pass

    def close(self):
        self._closed.set()
        try:
            self._q.put_nowait({"type": "__closed__"})
        except queue.Full:
            pass

    def closed(self) -> bool:
        return self._closed.is_set()

    def get(self, timeout: Optional[float] = None) -> Optional[dict]:
        if self._closed.is_set() and self._q.empty():
            return None
        item = self._q.get(timeout=timeout)
        if isinstance(item, dict) and item.get("type") == "__closed__":
            return None
        return item

    def __iter__(self):
        while True:
            item = self.get()
            if item is None:
                break
            yield item

class SubscriptionRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._by_id: Dict[int, Subscription] = {}

    def register(self, req_id: int, sub: Subscription) -> Subscription:
        with self._lock:
            self._by_id[req_id] = sub
        return sub

    def dispatch(self, req_id: int, event: Dict[str, Any]):
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
