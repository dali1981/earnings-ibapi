from __future__ import annotations

from pathlib import Path

import config
import inspect
import os


def guard_production_write(base_path: Path | str) -> None:
    """Prevent accidental writes to production DATA_ROOT from scripts/.

    - If target base matches config.DATA_ROOT and caller originates from a file
      inside the scripts/ folder, block unless ALLOW_PROD_WRITES=1 is set.
    - Jobs and library code are unaffected.
    """
    try:
        base = Path(base_path).resolve()
        data_root = Path(config.DATA_ROOT).resolve()
    except Exception:
        return

    if base != data_root:
        return

    # Allow explicit override
    if os.environ.get("ALLOW_PROD_WRITES") == "1":
        return

    # Inspect stack to find caller files
    for frameinfo in inspect.stack()[1:]:
        try:
            fpath = Path(frameinfo.filename)
        except Exception:
            continue
        # Block if call originated from scripts/
        if "scripts" in fpath.parts:
            raise PermissionError(
                "Writes to production data are blocked from scripts/. "
                "Use --demo or set ALLOW_PROD_WRITES=1 to override."
            )
