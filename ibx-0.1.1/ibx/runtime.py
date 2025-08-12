
import threading
from typing import Optional
from ibapi.client import EClient
from .ids import RequestIdSequencer
from .futures import FutureRegistry
from .subscriptions import SubscriptionRegistry
from .wrapper import IBWrapperBridge

class IBRuntime:
    """Connection + network thread. Composition root for wrapper/registries."""
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.host, self.port, self.client_id = host, port, client_id
        self.sequencer = RequestIdSequencer()
        self.registry = FutureRegistry()
        self.subs = SubscriptionRegistry()
        self.wrapper = IBWrapperBridge(self.sequencer, self.registry, self.subs)
        self.client = EClient(self.wrapper)
        self._run_thread: Optional[threading.Thread] = None

    def start(self, ready_timeout: float = 10.0):
        if self.client.isConnected():
            return
        self.client.connect(self.host, self.port, self.client_id)
        self._run_thread = threading.Thread(target=self.client.run, name="ib-reader", daemon=False)
        self._run_thread.start()
        self.sequencer.wait_ready(ready_timeout)

    def stop(self, join_timeout: float = 5.0):
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
