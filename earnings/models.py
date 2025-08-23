#!/usr/bin/env python3
"""
Pydantic models for earnings calendar data.
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class EarningsSource(str, Enum):
    """Earnings data sources."""
    FMP = "financial_modeling_prep"
    FINNHUB = "finnhub"
    NASDAQ = "nasdaq"
    YAHOO = "yahoo"


class EarningsTimeOfDay(str, Enum):
    """When earnings are announced relative to market hours."""
    BMO = "bmo"  # Before Market Open
    AMC = "amc"  # After Market Close
    DMT = "dmt"  # During Market Trading
    UNKNOWN = "unknown"


class EarningsEvent(BaseModel):
    """Single earnings event with Pydantic validation."""
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    symbol: str = Field(..., min_length=1, max_length=20, description="Stock ticker symbol")
    company_name: str = Field(..., min_length=1, description="Company name")
    earnings_date: date = Field(..., description="Date of earnings announcement")
    time: Optional[EarningsTimeOfDay] = Field(
        default=EarningsTimeOfDay.UNKNOWN,
        description="Time of day for earnings announcement"
    )
    eps_estimate: Optional[float] = Field(None, description="Estimated earnings per share")
    eps_actual: Optional[float] = Field(None, description="Actual earnings per share")
    revenue_estimate: Optional[float] = Field(None, description="Estimated revenue")
    revenue_actual: Optional[float] = Field(None, description="Actual revenue")
    source: EarningsSource = Field(default=EarningsSource.NASDAQ, description="Data source")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    
    # Additional NASDAQ fields
    last_year_eps: Optional[float] = Field(None, description="Last year's EPS")
    fiscal_quarter_ending: Optional[str] = Field(None, description="Fiscal quarter ending (e.g., 'Jun/2025')")
    num_estimates: Optional[int] = Field(None, description="Number of analyst estimates")
    last_year_report_date: Optional[date] = Field(None, description="Last year's report date")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        v = v.upper().strip()
        if not v.replace('.', '').replace('-', '').isalnum():
            raise ValueError(f"Invalid symbol: {v}")
        return v
    
    @field_validator('time', mode='before')
    @classmethod
    def parse_time(cls, v: Any) -> Optional[EarningsTimeOfDay]:
        """Parse various time representations."""
        if v is None:
            return EarningsTimeOfDay.UNKNOWN
        if isinstance(v, EarningsTimeOfDay):
            return v
        
        v_str = str(v).lower().strip()
        if v_str in ('bmo', 'pre', 'pre-market', 'before'):
            return EarningsTimeOfDay.BMO
        elif v_str in ('amc', 'post', 'after', 'after-market'):
            return EarningsTimeOfDay.AMC
        elif v_str in ('dmt', 'during', 'regular'):
            return EarningsTimeOfDay.DMT
        else:
            return EarningsTimeOfDay.UNKNOWN
    
    @property
    def days_until_earnings(self) -> int:
        """Days until earnings (negative if in past)."""
        return (self.earnings_date - date.today()).days
    
    @property
    def priority_score(self) -> float:
        """Priority score for scheduling (higher = more urgent)."""
        days_until = self.days_until_earnings
        
        if days_until <= 0:
            return 100.0  # Today or past
        elif days_until <= 3:
            return 90.0  # Next 3 days
        elif days_until <= 7:
            return 80.0  # Next week
        elif days_until <= 14:
            return 60.0  # Next 2 weeks
        elif days_until <= 30:
            return 40.0  # Next month
        else:
            return 20.0  # Future
    
    def dict_for_cache(self) -> Dict[str, Any]:
        """Convert to dictionary for caching."""
        return {
            "symbol": self.symbol,
            "company_name": self.company_name,
            "earnings_date": self.earnings_date.isoformat(),
            "time": self.time.value,
            "eps_estimate": self.eps_estimate,
            "eps_actual": self.eps_actual,
            "revenue_estimate": self.revenue_estimate,
            "revenue_actual": self.revenue_actual,
            "source": self.source.value,
            "market_cap": self.market_cap,
            "last_year_eps": self.last_year_eps,
            "fiscal_quarter_ending": self.fiscal_quarter_ending,
            "num_estimates": self.num_estimates,
            "last_year_report_date": self.last_year_report_date.isoformat() if self.last_year_report_date else None,
            "priority_score": self.priority_score,
        }
    
    @classmethod
    def from_cache_dict(cls, data: Dict[str, Any]) -> "EarningsEvent":
        """Create from cached dictionary."""
        # Convert date strings back to date objects
        if isinstance(data.get("earnings_date"), str):
            data["earnings_date"] = datetime.strptime(
                data["earnings_date"], "%Y-%m-%d"
            ).date()
        
        if data.get("last_year_report_date") and isinstance(data["last_year_report_date"], str):
            data["last_year_report_date"] = datetime.strptime(
                data["last_year_report_date"], "%Y-%m-%d"
            ).date()
        
        # Remove priority_score as it's computed
        data.pop("priority_score", None)
        
        return cls(**data)


class NasdaqEarningsRow(BaseModel):
    """Raw NASDAQ API response row with all fields."""
    
    model_config = ConfigDict(extra='allow')
    
    # Required fields
    symbol: str
    name: Optional[str] = None
    
    # Time field
    time: Optional[str] = None  # 'time-pre-market', 'time-after-hours', 'time-not-supplied'
    
    # Financial data
    epsForecast: Optional[str] = None  # Can be '$1.69' or '($0.12)' for negative
    lastYearEPS: Optional[str] = None  # Last year's EPS
    marketCap: Optional[str] = None  # e.g., '$174,559,728,968'
    
    # Fiscal period
    fiscalQuarterEnding: Optional[str] = None  # e.g., 'Jun/2025'
    
    # Estimates metadata
    noOfEsts: Optional[str] = None  # Number of estimates or 'N/A'
    
    # Historical data
    lastYearRptDt: Optional[str] = None  # e.g., '8/26/2024' or 'N/A'
    
    def parse_eps_value(self, eps_str: Optional[str]) -> Optional[float]:
        """Parse EPS values like '$1.69' or '($0.12)' for negative."""
        if not eps_str or eps_str in ['N/A', '', None]:
            return None
        
        try:
            # Remove $ and commas
            clean = eps_str.replace('$', '').replace(',', '').strip()
            
            # Handle parentheses for negative values
            if clean.startswith('(') and clean.endswith(')'):
                clean = '-' + clean[1:-1]
            
            return float(clean)
        except (ValueError, TypeError):
            return None
    
    def parse_fiscal_quarter(self) -> Optional[tuple[str, int]]:
        """Parse fiscal quarter like 'Jun/2025' to (month, year)."""
        if not self.fiscalQuarterEnding or self.fiscalQuarterEnding == 'N/A':
            return None
        
        try:
            parts = self.fiscalQuarterEnding.split('/')
            if len(parts) == 2:
                return (parts[0], int(parts[1]))
        except (ValueError, IndexError):
            pass
        return None
    
    def parse_estimate_count(self) -> Optional[int]:
        """Parse number of estimates."""
        if not self.noOfEsts or self.noOfEsts == 'N/A':
            return None
        
        try:
            return int(self.noOfEsts)
        except (ValueError, TypeError):
            return None
    
    def parse_last_year_date(self) -> Optional[date]:
        """Parse last year's report date like '8/26/2024'."""
        if not self.lastYearRptDt or self.lastYearRptDt == 'N/A':
            return None
        
        try:
            from datetime import datetime
            # Handle M/D/YYYY format
            return datetime.strptime(self.lastYearRptDt, '%m/%d/%Y').date()
        except (ValueError, TypeError):
            return None
    
    def to_earnings_event(self, earnings_date: date) -> EarningsEvent:
        """Convert to EarningsEvent with all parsed data."""
        # Parse time
        time_str = (self.time or "").lower()
        if "pre" in time_str:
            timing = EarningsTimeOfDay.BMO
        elif "after" in time_str:
            timing = EarningsTimeOfDay.AMC
        else:
            timing = EarningsTimeOfDay.DMT
        
        # Create event with all fields
        return EarningsEvent(
            symbol=self.symbol.upper(),
            company_name=self.name or self.symbol,
            earnings_date=earnings_date,
            time=timing,
            eps_estimate=self.parse_eps_value(self.epsForecast),
            source=EarningsSource.NASDAQ,
            market_cap=parse_market_cap(self.marketCap) if self.marketCap else None,
            last_year_eps=self.parse_eps_value(self.lastYearEPS),
            fiscal_quarter_ending=self.fiscalQuarterEnding if self.fiscalQuarterEnding != 'N/A' else None,
            num_estimates=self.parse_estimate_count(),
            last_year_report_date=self.parse_last_year_date(),
        )


class NasdaqEarningsResponse(BaseModel):
    """NASDAQ API response structure."""
    
    model_config = ConfigDict(extra='allow')
    
    data: Optional[Dict[str, Any]] = None
    
    def get_rows(self) -> List[NasdaqEarningsRow]:
        """Extract rows from response."""
        if not self.data or "rows" not in self.data:
            return []
        
        rows = []
        for row_data in self.data["rows"]:
            try:
                rows.append(NasdaqEarningsRow(**row_data))
            except Exception:
                continue
        return rows


def parse_market_cap(market_cap_str: str) -> Optional[float]:
    """Parse market cap string to float value."""
    if not market_cap_str or market_cap_str in ["N/A", ""]:
        return None
    
    try:
        # Remove currency symbols and commas
        clean_str = market_cap_str.replace("$", "").replace(",", "").strip()
        
        if "B" in clean_str or "b" in clean_str:
            # Billion
            number = float(clean_str.replace("B", "").replace("b", ""))
            return number * 1e9
        elif "M" in clean_str or "m" in clean_str:
            # Million
            number = float(clean_str.replace("M", "").replace("m", ""))
            return number * 1e6
        elif "K" in clean_str or "k" in clean_str:
            # Thousand
            number = float(clean_str.replace("K", "").replace("k", ""))
            return number * 1e3
        else:
            # Raw number
            return float(clean_str)
    except (ValueError, AttributeError):
        return None


class EarningsCalendarResponse(BaseModel):
    """Response containing multiple earnings events."""
    
    events: List[EarningsEvent]
    source: EarningsSource
    fetch_date: datetime = Field(default_factory=datetime.now)
    
    def filter_by_symbols(self, symbols: List[str]) -> "EarningsCalendarResponse":
        """Filter events by symbols."""
        symbol_set = {s.upper() for s in symbols}
        filtered_events = [e for e in self.events if e.symbol in symbol_set]
        return EarningsCalendarResponse(
            events=filtered_events,
            source=self.source,
            fetch_date=self.fetch_date
        )
    
    def deduplicate(self) -> "EarningsCalendarResponse":
        """Remove duplicate events."""
        seen = {}
        unique = []
        
        for event in self.events:
            key = (event.symbol, event.earnings_date)
            
            if key not in seen:
                seen[key] = event
                unique.append(event)
            else:
                # Keep the one with more data
                existing = seen[key]
                if (event.eps_estimate is not None and existing.eps_estimate is None) or \
                   (event.revenue_estimate is not None and existing.revenue_estimate is None):
                    seen[key] = event
                    unique[unique.index(existing)] = event
        
        return EarningsCalendarResponse(
            events=unique,
            source=self.source,
            fetch_date=self.fetch_date
        )