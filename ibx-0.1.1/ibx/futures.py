
import threading
from typing import Any, Dict, List, Optional, Tuple

class IBFuture:
    """Minimal thread-safe future for IB ...End style requests."""
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
        if self.expect_many:
            return self._items
        return self._items[0] if self._items else None

class FutureRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._by_id: Dict[int, IBFuture] = {}

    def register(self, req_id: int, fut: IBFuture) -> IBFuture:
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

    def set_error(self, req_id: int, code: int, msg: str) -> bool:
        """Return True if a waiting future existed and was signaled."""
        with self._lock:
            fut = self._by_id.pop(req_id, None)
        if fut:
            fut.set_error(code, msg)
            return True
        return False
