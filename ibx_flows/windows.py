from __future__ import annotations
from datetime import date
from typing import List, Tuple
import pandas as pd
def missing_windows(present_dates: set, start: date, end: date) -> List[Tuple[date, date]]:
    desired = pd.bdate_range(start, end).date
    missing = [d for d in desired if d not in present_dates]
    if not missing: return []
    windows = []; s = missing[0]; prev = s
    for d in missing[1:]:
        if (pd.Timestamp(d) - pd.Timestamp(prev)).days == 1:
            prev = d
        else:
            windows.append((s, prev)); s = d; prev = d
    windows.append((s, prev)); return windows
