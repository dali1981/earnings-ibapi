"""EndDateTime class for IB API datetime formatting."""

from __future__ import annotations

from datetime import datetime
from typing import Union

import pytz


class EndDateTime:
    """Encapsulates IB API datetime formatting requirements.

    Handles timezone-aware datetime formatting for IB API calls to prevent
    the recurring "IB error 2174: Warning: You submitted request with date-time
    attributes without explicit time zone" error.
    """

    def __init__(self, dt: Union[str, datetime], timezone: str = "US/Eastern"):
        """Initialize EndDateTime.

        Args:
            dt: DateTime as string (YYYYMMDD HH:MM:SS) or datetime object
            timezone: Timezone string (default: US/Eastern for US markets)
        """
        self.timezone = timezone
        self._dt = self._parse_datetime(dt)

    def _parse_datetime(self, dt: Union[str, datetime]) -> datetime:
        """Parse input datetime to timezone-aware datetime object."""
        if isinstance(dt, str):
            # Parse common IB format: "20240825 23:59:59"
            if len(dt) == 17 and " " in dt:
                dt_obj = datetime.strptime(dt, "%Y%m%d %H:%M:%S")
            elif len(dt) == 8:  # Just date: "20240825"
                dt_obj = datetime.strptime(dt + " 23:59:59", "%Y%m%d %H:%M:%S")
            else:
                raise ValueError(f"Unsupported datetime format: {dt}")
        elif isinstance(dt, datetime):
            dt_obj = dt
        else:
            raise ValueError(f"Unsupported datetime type: {type(dt)}")

        # Ensure timezone awareness
        if dt_obj.tzinfo is None:
            tz = pytz.timezone(self.timezone)
            dt_obj = tz.localize(dt_obj)

        return dt_obj

    def to_ib_format(self) -> str:
        """Format datetime for IB API calls.

        Returns:
            String in format "yyyymmdd hh:mm:ss US/Eastern" as required by IB API
        """
        return self._dt.strftime("%Y%m%d %H:%M:%S") + f" {self.timezone}"

    def __str__(self) -> str:
        """String representation for IB API calls."""
        return self.to_ib_format()

    def __repr__(self) -> str:
        """Developer representation."""
        return f"EndDateTime('{self.to_ib_format()}')"

    @classmethod
    def now(cls, timezone: str = "US/Eastern") -> EndDateTime:
        """Create EndDateTime for current time in specified timezone."""
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        return cls(now, timezone)

    @classmethod
    def market_close(cls, date_str: str, timezone: str = "US/Eastern") -> EndDateTime:
        """Create EndDateTime for market close (16:00) on given date.

        Args:
            date_str: Date in YYYYMMDD format (e.g., "20240825")
            timezone: Market timezone
        """
        dt_str = f"{date_str} 16:00:00"
        return cls(dt_str, timezone)
