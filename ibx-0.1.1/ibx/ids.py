
import threading
from typing import Optional

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
                raise RuntimeError("Sequencer not initialized yet (no nextValidId)")
            rid = self._next
            self._next += n
            return rid
