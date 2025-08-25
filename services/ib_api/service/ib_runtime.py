from __future__ import annotations

import atexit
from typing import Optional

from app_settings.settings import get_settings


from ibx import IBRuntime

_rt_singleton: Optional[IBRuntime] = None


def get_runtime() -> IBRuntime:
    global _rt_singleton
    if _rt_singleton is None:
        s = get_settings()
        _rt_singleton = IBRuntime(host=s.ib.host, port=s.ib.port, client_id=s.ib.client_id_contracts)
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
