from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from config import IB_HOST, IB_PORT, IB_CLIENT_IDS


@contextmanager
def open_runtime(host: str | None = None, port: int | None = None, client_id: int | None = None) -> Iterator["IBRuntime"]:
    try:
        # Prefer ibx import
        from ibx import IBRuntime  # type: ignore
    except Exception:  # pragma: no cover - fallback for local envs
        from ibx.runtime import IBRuntime  # type: ignore

    rt = IBRuntime(host or IB_HOST, port or IB_PORT, client_id or IB_CLIENT_IDS.get("default", 77))
    try:
        with rt:
            yield rt
    finally:
        try:
            rt.close()
        except Exception:
            pass

