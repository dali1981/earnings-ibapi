import logging
import threading
import time
from typing import Callable, List


class RequestSequencer:
    """Owns the *nextValidId* and fires queued callbacks once ready."""

    def __init__(self):
        self._next_id: int | None = None
        self._on_ready: List[Callable[[int], None]] = []
        self._pending: dict[int, str] = {}
        self._on_all_done: List[Callable[[], None]] = []
        self._lock = threading.Lock()

    def seed(self, start: int):
        """
                Initialize the requests with the first valid ID (nextValidId event).
                Fires and clears any callbacks queued before seeding.
                """
        with self._lock:
            self._next_id = start
            logging.info("Sequencer seeded: next_id=%d", start)
            callbacks = self._on_ready[:]
            self._on_ready.clear()

        for fn in callbacks:
            try:
                fn(start)
            except Exception:
                logging.exception("Error in when_ready callback")

    def next(self, tag: str | None = None) -> int:
        """
                Allocate and return the next request ID. Optionally tag it for tracking.
                Raises if requests is not yet seeded.
                """
        with self._lock:
            if self._next_id is None:
                raise RuntimeError("nextValidId not received yet. Cannot allocate request ID.")
            rid = self._next_id
            self._next_id += 1
            if tag:
                self._pending[rid] = tag
                logging.debug("Allocated %-20s → %d", tag, rid)
            return rid

    def done(self, rid: int, on_all_done: Callable[[], None] | None = None):
        """
        Mark a request as complete; remove it from pending.
        If no pending requests remain, invoke registered on_all_done callbacks.
        """
        callbacks: List[Callable[[], None]] = []
        with self._lock:
            tag = self._pending.pop(rid, None)
            if tag:
                logging.info("✔ %s complete (id=%d)", tag, rid)
            else:
                logging.warning("Done called for unknown request id %d", rid)
            # If no more pending, prepare to fire on_all_done
            if not self._pending and self._on_all_done:
                callbacks = self._on_all_done[:]
                # Optionally clear to ensure single invocation
                self._on_all_done.clear()

        for fn in callbacks:
            try:
                fn()
            except Exception:
                logging.exception("Error in on_all_done callback")

    def when_ready(self, fn: Callable[[int], None]):
        """
                Queue a callback to be invoked once the requests is seeded,
                or invoke immediately if already seeded.

                - If not seeded: store fn in the queue.
                - If seeded: call fn immediately with the current next ID.

                Callbacks queued before seeding are fired exactly once in seed().
                """
        with self._lock:
            if self._next_id is None:
                self._on_ready.append(fn)
                return
            next_id = self._next_id

        try:
            fn(next_id)
        except Exception:
            logging.exception("Error in when_ready callback")

    def register_on_all_done(self, fn: Callable[[], None]):
        """
        Register a callback to be invoked once all pending requests are completed.
        If there are no pending requests at registration time, invoke immediately.
        """
        # with self._lock:
        #     if not self._pending:
        #         # No pending; invoke immediately outside lock
        #         call_now = True
        #     else:
        #         self._on_all_done.append(fn)
        #         call_now = False
        #
        # if call_now:
        #     try:
        #         fn()
        #     except Exception:
        #         logging.exception("Error in on_all_done immediate callback")

        self._on_all_done.append(fn)