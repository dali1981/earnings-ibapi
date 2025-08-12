"""
ibx_time.py — Centralised date/time helpers for IB.

Goals
-----
- One place to parse IB bar timestamps (daily + intraday with optional TZ).
- One place to build `endDateTime` strings with an explicit time zone
  (instrument TZ or UTC) for `reqHistoricalData` / `reqMktData`.

Usage
-----
from ibx_time import (
    parse_ib_datetime_series,
    instrument_timezone,
    ib_end_datetime,
    ib_end_datetime_instrument,
)

end = ib_end_datetime_instrument(rt, make_stock("AAPL"), "20250808 15:50:00")
# -> "20250808 15:50:00 US/Eastern" (or as returned by ContractDetails)

rows = hist.bars(contract, endDateTime=end, ...)

Notes
-----
- Accepts naive strings/datetimes and appends a TZ (instrument or UTC).
- Supports both space and hyphen separators (IB accepts either).
- Parser is robust to strings like "YYYYMMDD", "YYYYMMDD HH:MM:SS",
  and "YYYYMMDD HH:MM:SS US/Eastern".
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union
import re

import pandas as pd

# ----------------------------- Parsing -----------------------------

_TZ_TOKEN_RE = re.compile(r"\b(UTC|[A-Za-z]+/[A-Za-z_+\-0-9]+)\b")


def parse_ib_datetime_series(s: pd.Series) -> pd.Series:
    """Parse IB 'date' strings: daily/intraday with optional trailing TZ.

    Returns a pandas Series of Timestamps (UTC-naive; caller can localize).
    """
    if s is None:
        return pd.Series(pd.NaT, index=[])
    s = s.astype(str)

    def _canon(x: str) -> str:
        # e.g., '20250807 04:00:00 US/Eastern' -> '20250807 04:00:00'
        m = _TZ_TOKEN_RE.search(x)
        if m:
            x = x[: m.start()].strip()
        parts = x.split()
        if len(parts) >= 2 and len(parts[0]) == 8 and parts[0].isdigit():
            if len(parts[1]) == 8 and parts[1].count(":") == 2:
                return parts[0] + " " + parts[1]
            return parts[0]
        return x

    canon = s.map(_canon)
    ts = pd.to_datetime(canon, format="%Y%m%d %H:%M:%S", errors="coerce")
    mask = ts.isna()
    if mask.any():
        ts2 = pd.to_datetime(canon[mask], format="%Y%m%d", errors="coerce")
        ts[mask] = ts2
    if ts.isna().any():
        ts = pd.to_datetime(canon, errors="coerce")
    return ts


# ----------------------------- Formatting -----------------------------

@dataclass
class EndOptions:
    tz: Optional[str] = None  # e.g., "US/Eastern" or "UTC"; if None, leave as-is when already present
    hyphen: bool = False      # use 'YYYYMMDD-HH:MM:SS' instead of space


def _has_tz(s: str) -> bool:
    return bool(_TZ_TOKEN_RE.search(s))


def _fmt_datetime_str(as_of: str, hyphen: bool) -> str:
    # Allow 'YYYYMMDD HH:MM:SS' or already 'YYYYMMDD-HH:MM:SS'
    if hyphen:
        return as_of.replace(" ", "-")
    return as_of


def ib_end_datetime(
    as_of: Optional[Union[str, datetime]], *, tz: Optional[str] = "UTC", hyphen: bool = False
) -> str:
    """Return an IB-ready endDateTime string.

    - If as_of is None -> "" (NOW).
    - If as_of is a datetime: format as 'YYYYMMDD HH:MM:SS <TZ>' using tz (or existing tzinfo if aware and tz is None/'')
    - If as_of is a string: if it already contains a TZ token, keep it; else append tz.
    """
    if as_of is None:
        return ""

    if isinstance(as_of, datetime):
        # If aware and tz is None/empty, keep tzinfo name if available; otherwise use provided tz or UTC
        ts = as_of
        stamp = ts.strftime("%Y%m%d %H:%M:%S")
        tzid = tz or ts.tzname() or "UTC"
        return _fmt_datetime_str(stamp, hyphen) + (f" {tzid}" if tzid else "")

    # string path
    s = str(as_of).strip()
    if _has_tz(s):
        return _fmt_datetime_str(s, hyphen)
    tzid = tz or "UTC"
    return _fmt_datetime_str(s, hyphen) + (f" {tzid}" if tzid else "")


# ---------------------- Instrument timezone ----------------------

def instrument_timezone(rt, contract) -> str:
    """Lookup instrument time zone via ContractDetails. Fallback 'US/Eastern'."""
    try:
        from ibx import ContractDetailsService
    except Exception:
        # Local import fallback name if user placed this file elsewhere
        from ibx.services import ContractDetailsService  # type: ignore

    cds = ContractDetailsService(rt).fetch(contract)
    tz = getattr(cds[0], "timeZoneId", None)
    return tz or "US/Eastern"


def ib_end_datetime_instrument(rt, contract, as_of: Optional[Union[str, datetime]], *, hyphen: bool = False) -> str:
    """Return 'endDateTime' using the instrument's time zone.

    - If `as_of` already contains a TZ, pass through.
    - Otherwise, append the instrument TZ (via ContractDetails).
    """
    if as_of is None:
        return ""
    if isinstance(as_of, str) and _has_tz(as_of):
        return _fmt_datetime_str(as_of, hyphen)

    tzid = instrument_timezone(rt, contract)
    if isinstance(as_of, datetime):
        stamp = as_of.strftime("%Y%m%d %H:%M:%S")
    else:
        stamp = str(as_of).strip()
    return _fmt_datetime_str(stamp, hyphen) + f" {tzid}"
