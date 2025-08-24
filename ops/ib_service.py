from __future__ import annotations

import atexit
from typing import Optional

try:
    from ibx import IBRuntime  # type: ignore
except Exception:  # pragma: no cover - fallback for vendored layout
    from ibx.runtime import IBRuntime  # type: ignore

from ibx_config.settings import get_settings

_rt_singleton: Optional[IBRuntime] = None


def get_runtime() -> IBRuntime:
    """Return a long-lived IBRuntime instance per worker process.

    Creates and starts the runtime on first use, reusing it for subsequent tasks.
    """
    global _rt_singleton
    if _rt_singleton is None:
        s = get_settings()
        host = s.ib.host
        port = int(s.ib.port)
        client_id = int(s.ib.client_id_contracts)
        _rt_singleton = IBRuntime(host=host, port=port, client_id=client_id)
        _rt_singleton.start()
    return _rt_singleton


def _shutdown() -> None:
    global _rt_singleton
    try:
        if _rt_singleton is not None:
            _rt_singleton.stop()
    finally:
        _rt_singleton = None


atexit.register(_shutdown)
