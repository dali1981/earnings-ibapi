from __future__ import annotations

import atexit
import time
from typing import Optional

from app_settings.settings import get_settings


from ibx import IBRuntime

_rt_singleton: Optional[IBRuntime] = None
_last_attempt_ts: float = 0.0
_attempt_backoff_sec: float = 2.0


def get_runtime() -> IBRuntime:
    """Return a per-process IBRuntime.

    Avoids raw port probing; instead, attempts an official IB connect with
    a short ready timeout and a small backoff between attempts so we don't
    block or spam when the gateway isn't up.
    """
    global _rt_singleton, _last_attempt_ts
    s = get_settings()
    if _rt_singleton is None:
        _rt_singleton = IBRuntime(host=s.ib.host, port=s.ib.port, client_id=s.ib.client_id_contracts)

    try:
        if not _rt_singleton.client.isConnected():
            now = time.monotonic()
            if now - _last_attempt_ts >= _attempt_backoff_sec:
                _last_attempt_ts = now
                # Use a short ready timeout to avoid hanging when IB GW is down
                _rt_singleton.start(ready_timeout=2.0)
    except Exception:
        # Leave unstarted; callers should handle not-connected cases
        pass
    return _rt_singleton


def _shutdown() -> None:
    global _rt_singleton
    try:
        if _rt_singleton is not None:
            _rt_singleton.stop()
    finally:
        _rt_singleton = None


atexit.register(_shutdown)
