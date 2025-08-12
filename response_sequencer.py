import threading
from datetime import datetime, date
from typing import Callable, Dict, List, Optional, Any
import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract


class ResponseSequencer:
    """
    Buffers incoming IB API responses keyed by reqId, matching them to request callbacks.
    Signals completion and emits a pandas DataFrame via the registered callback.
    """
    def __init__(self):
        self._events: Dict[int, threading.Event] = {}
        self._buffer: Dict[int, List[dict]] = {}
        self._callbacks: Dict[int, Callable[[pd.DataFrame], None]] = {}
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def add(
        self,
        req_id: int,
        callback: Callable[[pd.DataFrame], None],
        metadata: Dict[str, Any]
    ) -> None:
        """
        Register a new request: allocate event, buffer, callback, and store request metadata.
        """
        with self._lock:
            self._events[req_id] = threading.Event()
            self._buffer[req_id] = []
            self._callbacks[req_id] = callback
            self._metadata[req_id] = metadata

    def record(self, req_id: int, record: dict) -> None:
        """Append a single response record for the given reqId."""
        with self._lock:
            if req_id in self._buffer:
                self._buffer[req_id].append(record)

    def complete(self, req_id: int) -> None:
        """Mark request complete, enrich and invoke callback, and signal waiters."""
        with self._lock:
            event = self._events.pop(req_id, None)
            records = self._buffer.pop(req_id, [])
            cb = self._callbacks.pop(req_id, None)
            meta = self._metadata.pop(req_id, {})
        if cb:
            df = pd.DataFrame(records)
            # attach metadata as DataFrame attributes
            for k, v in meta.items():
                df[k] = v
            cb(df)
        if event:
            event.set()

    def wait_for(self, req_id: int, timeout: Optional[float] = None) -> bool:
        """Block until the given reqId is finalized or timeout expires."""
        event = self._events.get(req_id)
        return event.wait(timeout) if event else False