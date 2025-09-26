from __future__ import annotations

from .checkpoint import load_checkpoint, save_checkpoint
from pathlib import Path
from typing import Optional

import config
import time


class ProcessedIndex:
    def __init__(
        self, service: str, topic: str, state_root: Optional[Path] = None
    ) -> None:
        self.service = service
        self.topic = topic
        self.state_root = (
            Path(state_root) if state_root else (config.CACHE_PATH / "state")
        )
        self.path = self.state_root / service / f"{topic}_processed.json"
        self._cache = None

    def _load(self) -> dict:
        if self._cache is None:
            self._cache = load_checkpoint(self.path) or {"keys": {}}
        return self._cache

    def is_processed(self, key: str) -> bool:
        data = self._load()
        return key in data.get("keys", {})

    def mark_processed(self, key: str) -> None:
        data = self._load()
        keys = data.setdefault("keys", {})
        keys[key] = time.time()
        save_checkpoint(self.path, data)
