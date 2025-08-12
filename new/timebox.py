# ibx_time/timebox.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Literal

IB_HIST_FMT = "%Y%m%d %H:%M:%S"
IB_DATE_FMT = "%Y%m%d"

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("Naive datetime received; pass tz-aware UTC.")
    return dt.astimezone(timezone.utc)

def to_ib_end_datetime(dt_utc: datetime) -> str:
    dt_utc = ensure_utc(dt_utc)
    return dt_utc.strftime(IB_HIST_FMT)

BarUnit = Literal["1 min","5 mins","15 mins","1 hour","1 day"]

def floor_time(dt_utc: datetime, bar: BarUnit) -> datetime:
    dt_utc = ensure_utc(dt_utc)
    if bar == "1 day":
        return dt_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    mins = {"1 min":1, "5 mins":5, "15 mins":15, "1 hour":60}[bar]
    minutes = (dt_utc.minute // mins) * mins
    return dt_utc.replace(minute=minutes, second=0, microsecond=0)