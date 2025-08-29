from __future__ import annotations

from .checkpoint import load_checkpoint, save_checkpoint
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import config
import json


@dataclass
class ConsumerCheckpoint:
    filename: str
    offset: int
    last_ts: float

    @classmethod
    def from_dict(cls, d: Dict) -> "ConsumerCheckpoint":
        return cls(
            filename=d.get("filename", ""),
            offset=int(d.get("offset", 0)),
            last_ts=float(d.get("last_ts", 0.0)),
        )

    def to_dict(self) -> Dict:
        return {
            "filename": self.filename,
            "offset": self.offset,
            "last_ts": self.last_ts,
        }


class JSONLConsumer:
    def __init__(
        self,
        topic: str,
        *,
        cache_root: Optional[Path] = None,
        state_root: Optional[Path] = None,
    ):
        self.topic = topic
        self.cache_root = Path(cache_root) if cache_root else (config.CACHE_PATH)
        self.events_path = Path(self.cache_root) / "events" / f"{topic}.jsonl"
        self.state_root = (
            Path(state_root) if state_root else (config.CACHE_PATH / "state")
        )

    def _cp_path(self, service: str) -> Path:
        return self.state_root / service / f"{self.topic}.json"

    def _load_cp(self, service: str) -> ConsumerCheckpoint:
        d = load_checkpoint(self._cp_path(service))
        return ConsumerCheckpoint.from_dict(d)

    def _save_cp(self, service: str, cp: ConsumerCheckpoint) -> None:
        save_checkpoint(self._cp_path(service), cp.to_dict())

    def read(self, service: str, limit: Optional[int] = None) -> List[Dict]:
        cp = self._load_cp(service)
        filename = cp.filename or str(self.events_path)
        path = Path(filename)
        if not path.exists():
            path = self.events_path
        events: List[Dict] = []
        last_ts = cp.last_ts or 0.0
        offset = 0

        if path.exists():
            with path.open("rb") as f:
                if str(path) == cp.filename and cp.offset:
                    try:
                        f.seek(cp.offset)
                    except Exception:
                        pass
                while True:
                    line = f.readline()
                    if not line:
                        break
                    try:
                        item = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue
                    events.append(item)
                    last_ts = max(last_ts, float(item.get("ts", 0.0)))
                    if limit and len(events) >= limit:
                        break
                offset = f.tell()

        new_cp = ConsumerCheckpoint(filename=str(path), offset=offset, last_ts=last_ts)
        self._save_cp(service, new_cp)
        return events
