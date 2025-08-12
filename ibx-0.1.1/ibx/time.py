"""
ibx.time — Centralised date/time helpers for Interactive Brokers (IB)

Provides:
- parse_ib_datetime_series(series): robustly parses IB bar timestamps (daily or intraday, with optional TZ tokens)
- ib_end_datetime(as_of, tz="UTC", hyphen=True): formats an IB-compliant endDateTime string
- instrument_timezone(rt, contract): returns the instrument's exchange time zone (e.g., "US/Eastern")
- ib_end_datetime_instrument(rt, contract, as_of, hyphen=True): endDateTime using the instrument's TZ

Notes
-----
- IB now prefers explicit time zones and the hyphen format (YYYYMMDD-HH:MM:SS).
- If you pass an empty string to endDateTime, IB interprets it as "now" (which is allowed).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union
import re

import pandas as pd

__all__ = [
    "parse_ib_datetime_series",
    "ib_end_datetime",
    "instrument_timezone",
    "ib_end_datetime_instrument",
]

# ----------------------------- Parsing -----------------------------

_TZ_TOKEN_RE = re.compile(r"\b(UTC|[A-Za-z]+/[A-Za-z_+\-0-9]+)\b")


def parse_ib_datetime_series(s: pd.Series) -> pd.Series:
    """Parse IB 'date' strings: daily/intraday with optional trailing TZ.

    Returns a pandas Series of pandas.Timestamp (naive).
    Accepts forms like:
      - 'YYYYMMDD'
      - 'YYYYMMDD HH:MM:SS'
      - 'YYYYMMDD HH:MM:SS US/Eastern'
    """
    if s is None:
        return pd.Series(pd.NaT, index=[])
    s = s.astype(str)

    def _canon(x: str) -> str:
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
    tz: Optional[str] = None  # e.g., "US/Eastern" or "UTC"; if None, keep existing tz in string
    hyphen: bool = True       # use 'YYYYMMDD-HH:MM:SS' by default (IB-preferred)


def _has_tz(s: str) -> bool:
    return bool(_TZ_TOKEN_RE.search(s))


def _fmt_datetime_str(as_of: str, hyphen: bool) -> str:
    return as_of.replace(" ", "-") if hyphen else as_of


def ib_end_datetime(
    as_of: Optional[Union[str, datetime]], *, tz: Optional[str] = "UTC", hyphen: bool = True
) -> str:
    """Return an IB-ready endDateTime string.

    - None -> "" (NOW)
    - datetime -> 'YYYYMMDD HH:MM:SS <TZ>' (TZ from arg or dt.tzname())
    - string -> append TZ if missing; otherwise pass-through (optionally hyphenize)
    """
    if as_of is None:
        return ""

    if isinstance(as_of, datetime):
        stamp = as_of.strftime("%Y%m%d %H:%M:%S")
        tzid = tz or as_of.tzname() or "UTC"
        return _fmt_datetime_str(stamp, hyphen) + (f" {tzid}" if tzid else "")

    s = str(as_of).strip()
    if _has_tz(s):
        return _fmt_datetime_str(s, hyphen)
    tzid = tz or "UTC"
    return _fmt_datetime_str(s, hyphen) + (f" {tzid}" if tzid else "")

# ---------------------- Instrument timezone ----------------------

def instrument_timezone(rt, contract) -> str:
    """Lookup instrument time zone via ContractDetails. Fallback 'US/Eastern'."""
    try:
        # Preferred public import
        from ibx import ContractDetailsService  # type: ignore
    except Exception:
        # Internal layout fallback
        from ibx.services import ContractDetailsService  # type: ignore

    cds = ContractDetailsService(rt).fetch(contract)
    tz = getattr(cds[0], "timeZoneId", None)
    return tz or "US/Eastern"


def ib_end_datetime_instrument(
    rt,
    contract,
    as_of: Optional[Union[str, datetime]],
    *,
    hyphen: bool = True,
) -> str:
    """Return 'endDateTime' using the instrument's time zone.

    - If `as_of` already contains a TZ, pass through (hyphenized optionally).
    - Otherwise, append the instrument TZ (via ContractDetails).
    - If `as_of` is None, return "" (NOW).
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
