"""Trading domain Pydantic models."""

from decimal import Decimal
from datetime import date, datetime
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal, List, Union, Tuple, Dict, Any
from enum import Enum
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
            "secType": self.sec_type,
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
            "secType": self.sec_type,
            "exchange": self.exchange,
            "currency": self.currency,
            "lastTradeDateOrContractMonth": self.expiry.strftime("%Y%m%d"),
            "strike": float(self.strike),
            "right": self.right,
            "multiplier": self.multiplier or "100"
        }
    
    @property
    def option_symbol(self) -> str:
        """Generate standardized option symbol."""
        expiry_str = self.expiry.strftime("%y%m%d")
        right_str = self.right
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


# ============================================================================
# Trading Engine Schemas
# ============================================================================

class OrderAction(str, Enum):
    """Order action enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"
    BRACKET = "BRACKET"


class TimeInForce(str, Enum):
    """Time in force enumeration."""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING_SUBMIT = "PendingSubmit"
    SUBMITTED = "Submitted"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    PENDING_CANCEL = "PendingCancel"
    REJECTED = "Rejected"


class SignalType(str, Enum):
    """Trade signal type enumeration."""
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    MODIFY = "MODIFY"


class Order(BaseAPIModel):
    """Trading order specification."""
    
    order_id: Optional[int] = Field(None, description="Broker-assigned order ID")
    client_order_id: str = Field(..., description="Client-generated unique ID")
    contract: Union[EquityContract, OptionContract] = Field(..., description="Trading contract")
    action: OrderAction = Field(..., description="Buy or Sell")
    quantity: int = Field(..., gt=0, description="Order quantity")
    order_type: OrderType = Field(..., description="Order type")
    limit_price: Optional[Decimal] = Field(None, gt=0, description="Limit price")
    stop_price: Optional[Decimal] = Field(None, gt=0, description="Stop price")
    time_in_force: TimeInForce = Field(default=TimeInForce.DAY, description="Time in force")
    account: Optional[str] = Field(None, description="Trading account")
    parent_order_id: Optional[int] = Field(None, description="Parent order for bracket orders")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    @model_validator(mode='after')
    def validate_order_prices(self):
        """Validate order type and required prices."""
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit price required for limit orders")
        
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop price required for stop orders")
        
        if self.order_type == OrderType.STOP_LIMIT:
            if self.limit_price is None:
                raise ValueError("Limit price required for stop-limit orders")
            if self.stop_price is None:
                raise ValueError("Stop price required for stop-limit orders")
        
        return self
    
    def to_ib_order_dict(self) -> Dict[str, Any]:
        """Convert to IB API order dictionary format."""
        ib_order = {
            "action": self.action,
            "totalQuantity": self.quantity,
            "orderType": self.order_type,
            "tif": self.time_in_force
        }
        
        if self.limit_price is not None:
            ib_order["lmtPrice"] = float(self.limit_price)
        
        if self.stop_price is not None:
            ib_order["auxPrice"] = float(self.stop_price)
        
        if self.account:
            ib_order["account"] = self.account
        
        return ib_order


class ExecutionReport(BaseAPIModel):
    """Order execution report."""
    
    execution_id: str = Field(..., description="Unique execution ID")
    order_id: int = Field(..., description="Order ID")
    contract: Union[EquityContract, OptionContract] = Field(..., description="Executed contract")
    action: OrderAction = Field(..., description="Buy or Sell action")
    quantity: int = Field(..., gt=0, description="Executed quantity")
    price: Decimal = Field(..., gt=0, description="Execution price")
    commission: Optional[Decimal] = Field(None, ge=0, description="Commission paid")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Execution timestamp")


class Position(BaseAPIModel):
    """Trading position representation."""
    
    contract: Union[EquityContract, OptionContract] = Field(..., description="Position contract")
    position_size: int = Field(..., description="Position size (positive=long, negative=short)")
    avg_cost: Decimal = Field(..., description="Average cost basis")
    market_price: Decimal = Field(..., gt=0, description="Current market price")
    market_value: Decimal = Field(..., description="Current market value")
    unrealized_pnl: Decimal = Field(..., description="Unrealized P&L")
    realized_pnl: Decimal = Field(default=Decimal("0"), description="Realized P&L")
    account: Optional[str] = Field(None, description="Trading account")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.position_size > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.position_size < 0
    
    def update_market_price(self, new_price: Decimal) -> None:
        """Update market price and recalculate values."""
        self.market_price = new_price
        self.market_value = new_price * abs(self.position_size)
        
        if self.is_long:
            self.unrealized_pnl = (new_price - self.avg_cost) * self.position_size
        else:
            self.unrealized_pnl = (self.avg_cost - new_price) * abs(self.position_size)
        
        self.last_updated = datetime.utcnow()


class PortfolioSummary(BaseAPIModel):
    """Portfolio summary information."""
    
    total_value: Decimal = Field(..., description="Total portfolio value")
    buying_power: Decimal = Field(..., description="Available buying power")
    net_liquidation: Decimal = Field(..., description="Net liquidation value")
    unrealized_pnl: Decimal = Field(..., description="Total unrealized P&L")
    realized_pnl: Decimal = Field(..., description="Total realized P&L")
    positions: List[Position] = Field(default_factory=list, description="Current positions")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")


class RiskAssessment(BaseAPIModel):
    """Risk assessment for trades and positions."""
    
    max_loss: Decimal = Field(..., description="Maximum potential loss")
    win_probability: Optional[float] = Field(None, ge=0, le=1, description="Probability of profit")
    risk_reward_ratio: Optional[float] = Field(None, gt=0, description="Risk/reward ratio")
    portfolio_heat: Optional[float] = Field(None, ge=0, le=1, description="Portfolio heat percentage")
    sector_concentration: Optional[Dict[str, float]] = Field(None, description="Sector concentration")


class PositionTarget(BaseAPIModel):
    """Target position specification for trade signals."""
    
    contract: Union[EquityContract, OptionContract] = Field(..., description="Target contract")
    target_quantity: int = Field(..., description="Target quantity")
    entry_price_range: Optional[Tuple[Decimal, Decimal]] = Field(None, description="Entry price range")
    stop_loss: Optional[Decimal] = Field(None, description="Stop loss price")
    profit_target: Optional[Decimal] = Field(None, description="Profit target price")


class TradeSignal(BaseAPIModel):
    """Trading signal with position targets and risk assessment."""
    
    signal_id: str = Field(..., description="Unique signal identifier")
    strategy_name: str = Field(..., description="Strategy that generated signal")
    symbol: str = Field(..., description="Primary symbol")
    signal_type: SignalType = Field(..., description="Signal type")
    confidence: float = Field(..., ge=0, le=1, description="Signal confidence (0-1)")
    target_positions: List[PositionTarget] = Field(..., description="Target positions")
    risk_metrics: RiskAssessment = Field(..., description="Risk assessment")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation time")
    expires_at: Optional[datetime] = Field(None, description="Signal expiration time")