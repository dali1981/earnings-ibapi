from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_checkpoint(path: Path, data: Dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(data, ensure_ascii=False, separators=(",", ":")))

