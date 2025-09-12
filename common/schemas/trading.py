"""Trading domain Pydantic models."""

from decimal import Decimal
from datetime import date
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from .base import BaseAPIModel
from ..constants import SecType, OptionRight, Exchange


class ContractBase(BaseAPIModel):
    """Base contract information."""
    
    symbol: str = Field(..., min_length=1, max_length=10, description="Trading symbol")
    sec_type: SecType = Field(..., description="Security type")
    exchange: str = Field(default="SMART", description="Exchange")
    currency: str = Field(default="USD", max_length=3, description="Currency")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        return v.upper().strip()
    
    @field_validator('currency')
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate and normalize currency."""
        return v.upper().strip()


class EquityContract(ContractBase):
    """Equity contract specification."""
    
    sec_type: SecType = Field(default=SecType.STOCK, frozen=True)
    primary_exchange: Optional[str] = Field(None, description="Primary exchange")
    
    def to_ib_contract_dict(self) -> dict:
        """Convert to IB API contract dictionary."""
        contract_dict = {
            "symbol": self.symbol,
            "secType": self.sec_type.value,
            "exchange": self.exchange,
            "currency": self.currency
        }
        if self.primary_exchange:
            contract_dict["primaryExchange"] = self.primary_exchange
        return contract_dict


class OptionContract(ContractBase):
    """Option contract specification."""
    
    sec_type: SecType = Field(default=SecType.OPTION, frozen=True)
    expiry: date = Field(..., description="Expiration date")
    strike: Decimal = Field(..., gt=0, description="Strike price")
    right: OptionRight = Field(..., description="Call or Put")
    multiplier: Optional[str] = Field("100", description="Contract multiplier")
    
    @field_validator('strike')
    @classmethod
    def validate_strike(cls, v: Decimal) -> Decimal:
        """Validate strike price."""
        if v <= 0:
            raise ValueError("Strike price must be positive")
        # Round to 2 decimal places
        return round(v, 2)
    
    def to_ib_contract_dict(self) -> dict:
        """Convert to IB API contract dictionary."""
        return {
            "symbol": self.symbol,
            "secType": self.sec_type.value,
            "exchange": self.exchange,
            "currency": self.currency,
            "lastTradeDateOrContractMonth": self.expiry.strftime("%Y%m%d"),
            "strike": float(self.strike),
            "right": self.right.value,
            "multiplier": self.multiplier or "100"
        }
    
    @property
    def option_symbol(self) -> str:
        """Generate standardized option symbol."""
        expiry_str = self.expiry.strftime("%y%m%d")
        right_str = self.right.value
        strike_str = f"{float(self.strike):08.3f}".replace(".", "")
        return f"{self.symbol}{expiry_str}{right_str}{strike_str}"


class BarData(BaseAPIModel):
    """Market bar/candlestick data."""
    
    timestamp: date = Field(..., description="Bar timestamp")
    open_price: Decimal = Field(..., gt=0, description="Opening price")
    high_price: Decimal = Field(..., gt=0, description="High price")
    low_price: Decimal = Field(..., gt=0, description="Low price")
    close_price: Decimal = Field(..., gt=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Volume")
    bar_count: Optional[int] = Field(None, ge=0, description="Number of trades")
    wap: Optional[Decimal] = Field(None, gt=0, description="Weighted average price")
    
    def model_post_init(self, __context) -> None:
        """Validate OHLC relationships."""
        if not (self.low_price <= self.open_price <= self.high_price):
            raise ValueError("Open price must be between low and high")
        if not (self.low_price <= self.close_price <= self.high_price):
            raise ValueError("Close price must be between low and high")


class EarningsAnnouncement(BaseAPIModel):
    """Earnings announcement data."""
    
    symbol: str = Field(..., min_length=1, max_length=10)
    announcement_date: date = Field(..., description="Earnings announcement date")
    period_ending: Optional[date] = Field(None, description="Period ending date")
    time_of_day: Optional[Literal["BMO", "AMC", "DMT"]] = Field(
        None, description="Before market open, After market close, During market time"
    )
    eps_estimate: Optional[Decimal] = Field(None, description="EPS estimate")
    eps_actual: Optional[Decimal] = Field(None, description="Actual EPS reported")
    revenue_estimate: Optional[Decimal] = Field(None, description="Revenue estimate")
    revenue_actual: Optional[Decimal] = Field(None, description="Actual revenue")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        return v.upper().strip()


class OptionChainSnapshot(BaseAPIModel):
    """Option chain snapshot data."""
    
    underlying_symbol: str = Field(..., min_length=1, max_length=10)
    snapshot_date: date = Field(..., description="Date of snapshot")
    underlying_price: Decimal = Field(..., gt=0, description="Underlying asset price")
    expiry_date: date = Field(..., description="Option expiry date")
    strike: Decimal = Field(..., gt=0, description="Strike price")
    option_type: OptionRight = Field(..., description="Call or Put")
    
    # Option Greeks and pricing
    bid_price: Optional[Decimal] = Field(None, ge=0)
    ask_price: Optional[Decimal] = Field(None, ge=0)
    last_price: Optional[Decimal] = Field(None, ge=0)
    volume: Optional[int] = Field(None, ge=0)
    open_interest: Optional[int] = Field(None, ge=0)
    
    # Greeks
    delta: Optional[Decimal] = Field(None)
    gamma: Optional[Decimal] = Field(None)
    theta: Optional[Decimal] = Field(None)
    vega: Optional[Decimal] = Field(None)
    implied_volatility: Optional[Decimal] = Field(None, ge=0, le=10)  # 0 to 1000%
    
    @field_validator('underlying_symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        return v.upper().strip()
    
    @property
    def days_to_expiry(self) -> int:
        """Calculate days to expiry from snapshot date."""
        return (self.expiry_date - self.snapshot_date).days
    
    @property
    def moneyness(self) -> Decimal:
        """Calculate moneyness (underlying/strike for calls, strike/underlying for puts)."""
        if self.option_type == OptionRight.CALL:
            return self.underlying_price / self.strike
        else:
            return self.strike / self.underlying_price