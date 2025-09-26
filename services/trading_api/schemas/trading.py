"""
Trading API request/response schemas.
"""

from decimal import Decimal
from datetime import datetime, date
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Tuple
from enum import Enum

from common.schemas.trading import (
    EquityContract, OptionContract, OrderAction, OrderType, 
    TimeInForce, OrderStatus, SignalType
)


# ============================================================================
# Request Schemas
# ============================================================================

class CreateOrderRequest(BaseModel):
    """Request schema for creating orders."""
    
    contract: Union[EquityContract, OptionContract] = Field(..., description="Trading contract")
    action: OrderAction = Field(..., description="Buy or Sell")
    quantity: int = Field(..., gt=0, description="Order quantity")
    order_type: OrderType = Field(..., description="Order type")
    limit_price: Optional[Decimal] = Field(None, gt=0, description="Limit price")
    stop_price: Optional[Decimal] = Field(None, gt=0, description="Stop price")
    time_in_force: TimeInForce = Field(default=TimeInForce.DAY, description="Time in force")
    account: Optional[str] = Field(None, description="Trading account")


class TradeRiskAssessmentRequest(BaseModel):
    """Request schema for trade risk assessment."""
    
    positions: List[Dict[str, Any]] = Field(..., description="Proposed positions")
    portfolio_value: float = Field(..., gt=0, description="Current portfolio value")
    strategy_name: Optional[str] = Field(None, description="Strategy name")


class StrategyParametersRequest(BaseModel):
    """Request schema for updating strategy parameters."""
    
    parameters: Dict[str, Any] = Field(..., description="Strategy parameters")


# ============================================================================
# Response Schemas
# ============================================================================

class CreateOrderResponse(BaseModel):
    """Response schema for order creation."""
    
    order_id: int = Field(..., description="Broker-assigned order ID")
    client_order_id: str = Field(..., description="Client order ID")
    status: OrderStatus = Field(..., description="Order status")
    created_at: datetime = Field(..., description="Creation timestamp")
    message: Optional[str] = Field(None, description="Status message")


class OrderStatusResponse(BaseModel):
    """Response schema for order status."""
    
    order_id: int = Field(..., description="Order ID")
    status: OrderStatus = Field(..., description="Current status")
    filled_quantity: int = Field(default=0, description="Filled quantity")
    remaining_quantity: int = Field(default=0, description="Remaining quantity")
    avg_fill_price: Optional[Decimal] = Field(None, description="Average fill price")
    last_updated: datetime = Field(..., description="Last update time")
    commission: Optional[Decimal] = Field(None, description="Commission paid")


class PositionResponse(BaseModel):
    """Response schema for single position."""
    
    contract: Union[EquityContract, OptionContract] = Field(..., description="Position contract")
    position_size: int = Field(..., description="Position size")
    avg_cost: Decimal = Field(..., description="Average cost basis")
    market_price: Decimal = Field(..., description="Current market price")
    market_value: Decimal = Field(..., description="Current market value")
    unrealized_pnl: Decimal = Field(..., description="Unrealized P&L")
    realized_pnl: Decimal = Field(default=Decimal("0"), description="Realized P&L")
    last_updated: datetime = Field(..., description="Last update time")


class PositionsResponse(BaseModel):
    """Response schema for multiple positions."""
    
    positions: List[PositionResponse] = Field(..., description="List of positions")
    total_positions: int = Field(..., description="Total position count")
    last_updated: datetime = Field(..., description="Last update time")


class PortfolioResponse(BaseModel):
    """Response schema for portfolio summary."""
    
    total_value: Decimal = Field(..., description="Total portfolio value")
    buying_power: Decimal = Field(..., description="Available buying power")
    net_liquidation: Decimal = Field(..., description="Net liquidation value")
    unrealized_pnl: Decimal = Field(..., description="Total unrealized P&L")
    realized_pnl: Decimal = Field(..., description="Total realized P&L")
    positions_count: int = Field(..., description="Number of positions")
    last_updated: datetime = Field(..., description="Last update time")


class StrategyStatusResponse(BaseModel):
    """Response schema for strategy status."""
    
    enabled: bool = Field(..., description="Engine enabled status")
    total_strategies: int = Field(..., description="Total registered strategies")
    enabled_strategies: List[str] = Field(..., description="Enabled strategy names")
    disabled_strategies: List[str] = Field(..., description="Disabled strategy names")
    total_signals_generated: int = Field(..., description="Total signals generated")
    last_signal_time: Optional[datetime] = Field(None, description="Last signal time")


class StrategyActionResponse(BaseModel):
    """Response schema for strategy enable/disable actions."""
    
    strategy_name: str = Field(..., description="Strategy name")
    enabled: bool = Field(..., description="New enabled status")
    action_time: datetime = Field(default_factory=datetime.utcnow, description="Action timestamp")
    message: str = Field(..., description="Action message")


class SignalGenerationResponse(BaseModel):
    """Response schema for signal generation."""
    
    results: List[Dict[str, Any]] = Field(..., description="Signal generation results")
    total_signals: int = Field(..., description="Total signals generated")
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")


class SignalResponse(BaseModel):
    """Response schema for individual trading signal."""
    
    signal_id: str = Field(..., description="Signal ID")
    strategy_name: str = Field(..., description="Strategy name")
    symbol: str = Field(..., description="Symbol")
    signal_type: SignalType = Field(..., description="Signal type")
    confidence: float = Field(..., ge=0, le=1, description="Signal confidence")
    generated_at: datetime = Field(..., description="Generation time")
    expires_at: Optional[datetime] = Field(None, description="Expiration time")


class RiskAssessmentResponse(BaseModel):
    """Response schema for risk assessment."""
    
    approved: bool = Field(..., description="Risk approval status")
    max_loss: Decimal = Field(..., description="Maximum potential loss")
    win_probability: Optional[float] = Field(None, description="Win probability")
    risk_reward_ratio: Optional[float] = Field(None, description="Risk/reward ratio")
    portfolio_heat: Optional[float] = Field(None, description="Portfolio heat percentage")
    risk_message: Optional[str] = Field(None, description="Risk assessment message")
    assessment_time: datetime = Field(default_factory=datetime.utcnow, description="Assessment time")


class RiskLimitsResponse(BaseModel):
    """Response schema for risk limits."""
    
    max_position_size: int = Field(..., description="Maximum position size")
    max_portfolio_heat: float = Field(..., description="Maximum portfolio heat")
    max_daily_loss: Decimal = Field(..., description="Maximum daily loss")
    max_sector_concentration: float = Field(..., description="Maximum sector concentration")
    current_portfolio_heat: Optional[float] = Field(None, description="Current portfolio heat")
    current_daily_pnl: Optional[Decimal] = Field(None, description="Current daily P&L")


# ============================================================================
# Error Response Schemas
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response schema."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")


class ValidationErrorResponse(BaseModel):
    """Validation error response schema."""
    
    error: str = Field(default="validation_error", description="Error type")
    message: str = Field(..., description="Validation error message")
    field_errors: List[Dict[str, Any]] = Field(..., description="Field-specific errors")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


# ============================================================================
# Status Response Schemas
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(default="1.0.0", description="Service version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check time")
    dependencies: Optional[Dict[str, str]] = Field(None, description="Dependency status")


class ServiceStatusResponse(BaseModel):
    """Service status response schema."""
    
    service_name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    connected_services: List[str] = Field(..., description="Connected services")
    active_orders: int = Field(..., description="Active orders count")
    active_positions: int = Field(..., description="Active positions count")
    enabled_strategies: int = Field(..., description="Enabled strategies count")
    last_signal_time: Optional[datetime] = Field(None, description="Last signal time")