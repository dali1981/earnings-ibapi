"""
ibx_time.py — Centralized date/time helpers for IB API.

Goals
-----
- One place to parse IB bar timestamps (daily + intraday with optional TZ).
- One place to build `endDateTime` strings with an explicit time zone
  (instrument TZ or UTC) for `reqHistoricalData` / `reqMktData`.
- Enforce timezone consistency across all trading operations.
- Provide validation and error handling for date/time formats.

Usage
-----
from ibx_time import (
    parse_ib_datetime_series,
    instrument_timezone,
    ib_end_datetime,
    ib_end_datetime_instrument,
    validate_ib_datetime_format,
    safe_ib_end_datetime,
)

end = ib_end_datetime_instrument(rt, make_stock("AAPL"), "20250808 15:50:00")
# -> "20250808 15:50:00 US/Eastern" (or as returned by ContractDetails)

rows = hist.bars(contract, endDateTime=end, ...)

Notes
-----
- Accepts naive strings/datetimes and appends a TZ (instrument or UTC).
- IB API requires SPACE separator (not hyphen) for endDateTime parameter.
- Parser is robust to strings like "YYYYMMDD", "YYYYMMDD HH:MM:SS",
  and "YYYYMMDD HH:MM:SS US/Eastern".
- Includes validation to prevent common formatting errors.
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
    # IMPORTANT: IB API reqHistoricalData requires SPACE separator for endDateTime
    # The hyphen format is used internally for UTC timestamps only
    if hyphen:
        # This format is for internal use only - IB API will reject this format
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
    
    IMPORTANT: For IB API reqHistoricalData, always use hyphen=False (default)
    as IB API requires space separator between date and time.
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


# ---------------------- Enhanced Validation & Safety ----------------------

def validate_ib_datetime_format(datetime_str: str) -> bool:
    """Validate that a datetime string conforms to IB API expectations.
    
    Valid formats:
    - "20240105" (date only)
    - "20240105 15:30:00" (date and time with space)
    - "20240105 15:30:00 US/Eastern" (date, time, timezone with spaces)
    
    Invalid formats:
    - "20240105-15:30:00" (hyphen separator - IB API will reject)
    - "2024-01-05" (ISO date format)
    """
    if not datetime_str or not isinstance(datetime_str, str):
        return False
    
    datetime_str = datetime_str.strip()
    
    # Check for invalid hyphen format that IB API rejects
    if re.match(r'\d{8}-\d{2}:\d{2}:\d{2}', datetime_str):
        return False
    
    # Valid patterns
    patterns = [
        r'^\d{8}$',  # YYYYMMDD
        r'^\d{8} \d{2}:\d{2}:\d{2}$',  # YYYYMMDD HH:MM:SS
        r'^\d{8} \d{2}:\d{2}:\d{2} [A-Za-z]+/[A-Za-z_+\-0-9]+$',  # YYYYMMDD HH:MM:SS TZ
        r'^\d{8} \d{2}:\d{2}:\d{2} UTC$',  # YYYYMMDD HH:MM:SS UTC
    ]
    
    return any(re.match(pattern, datetime_str) for pattern in patterns)


def safe_ib_end_datetime(
    as_of: Optional[Union[str, datetime]], 
    tz: str = "UTC",
    validate: bool = True
) -> str:
    """Safe wrapper for ib_end_datetime with validation.
    
    Returns a validated IB API compatible datetime string.
    Raises ValueError if validation fails and validate=True.
    """
    result = ib_end_datetime(as_of, tz=tz, hyphen=False)  # Force hyphen=False for safety
    
    if validate and result and not validate_ib_datetime_format(result):
        raise ValueError(
            f"Generated datetime string '{result}' is not valid for IB API. "
            f"Expected format: 'YYYYMMDD HH:MM:SS TZ' (with spaces, not hyphens)"
        )
    
    return result


def safe_ib_end_datetime_instrument(
    rt, 
    contract, 
    as_of: Optional[Union[str, datetime]], 
    validate: bool = True
) -> str:
    """Safe wrapper for ib_end_datetime_instrument with validation.
    
    Returns a validated IB API compatible datetime string using instrument timezone.
    Raises ValueError if validation fails and validate=True.
    """
    result = ib_end_datetime_instrument(rt, contract, as_of, hyphen=False)  # Force hyphen=False
    
    if validate and result and not validate_ib_datetime_format(result):
        raise ValueError(
            f"Generated datetime string '{result}' is not valid for IB API. "
            f"Expected format: 'YYYYMMDD HH:MM:SS TZ' (with spaces, not hyphens)"
        )
    
    return result


def format_date_for_ib(dt: Union[datetime, str]) -> str:
    """Format a date/datetime for IB API usage with proper timezone handling.
    
    Args:
        dt: datetime object or ISO date string
        
    Returns:
        IB API compatible date string (YYYYMMDD format)
    """
    if isinstance(dt, str):
        if re.match(r'^\d{4}-\d{2}-\d{2}', dt):  # ISO format
            dt = datetime.fromisoformat(dt.split('T')[0])  # Handle both date and datetime ISO strings
        else:
            dt = pd.to_datetime(dt).to_pydatetime()
    
    return dt.strftime("%Y%m%d")


def ensure_eastern_timezone(dt: datetime, symbol: str = "") -> datetime:
    """Ensure datetime is in US/Eastern timezone for US market operations.
    
    Most US equity and option trading happens in Eastern time.
    This function helps standardize timezone handling.
    """
    try:
        import pytz
        eastern = pytz.timezone('US/Eastern')
        
        if dt.tzinfo is None:
            # Assume naive datetime is already in Eastern
            return eastern.localize(dt)
        else:
            # Convert to Eastern
            return dt.astimezone(eastern)
    except ImportError:
        # Fallback if pytz not available
        import warnings
        warnings.warn("pytz not available, returning datetime as-is. Install pytz for proper timezone handling.")
        return dt


# ---------------------- Logging and Debugging ----------------------

def debug_datetime_format(datetime_str: str) -> dict:
    """Debug helper to analyze datetime string format issues.
    
    Returns analysis of the datetime string for troubleshooting.
    """
    analysis = {
        'input': datetime_str,
        'is_valid_for_ib': validate_ib_datetime_format(datetime_str),
        'has_timezone': _has_tz(datetime_str),
        'format_issues': []
    }
    
    if not datetime_str:
        analysis['format_issues'].append('Empty or None datetime string')
        return analysis
    
    # Check for common issues
    if '-' in datetime_str and re.search(r'\d{8}-\d{2}:\d{2}:\d{2}', datetime_str):
        analysis['format_issues'].append('Uses hyphen separator - IB API requires space')
    
    if re.match(r'\d{4}-\d{2}-\d{2}', datetime_str):
        analysis['format_issues'].append('Uses ISO date format - IB API requires YYYYMMDD')
    
    if len(datetime_str) > 30:
        analysis['format_issues'].append('String too long - possible corruption')
    
    # Extract components
    parts = datetime_str.split()
    if parts:
        analysis['date_part'] = parts[0] if parts else ''
        analysis['time_part'] = parts[1] if len(parts) > 1 else ''
        analysis['tz_part'] = ' '.join(parts[2:]) if len(parts) > 2 else ''
    
    return analysis
