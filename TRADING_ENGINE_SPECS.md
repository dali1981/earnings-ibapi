# Trading Engine Specifications

## Overview

This document specifies the design and architecture for a comprehensive trading engine that integrates with the existing earnings-focused data platform. The trading engine will enable automated execution of options strategies based on earnings announcements, volatility patterns, and market data analysis.

## Architecture Integration

### Current System Analysis

The existing system follows a **local-first, event-driven architecture**:
- **Data Layer**: Parquet repositories under `data/`
- **Event System**: File-based JSONL event bus at `data/cache/events/`
- **Services**: FastAPI microservices for data ingestion and analysis
- **IB API Integration**: Existing connection pool and core API wrapper in `common/api/`

### Trading Engine Positioning

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │────│  Trading Engine  │────│   IB Execution  │
│   (Parquet)     │    │   (New Layer)    │    │   Layer         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Event Bus      │────│  Strategy Engine │────│  Order Manager  │
│  (JSONL)        │    │  (New)           │    │  (New)          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Core Components

### 1. Trading Engine Service (`services/trading_api/`)

#### 1.1 Order Manager (`common/trading/order_manager.py`)

**Responsibilities:**
- Order placement, modification, cancellation
- Order state management and tracking
- Risk management and position limits
- Integration with IB TWS API

**Key Features:**
```python
class OrderManager:
    def place_order(self, contract: Contract, order: Order) -> OrderExecution
    def modify_order(self, order_id: int, modifications: OrderModification) -> bool
    def cancel_order(self, order_id: int) -> bool
    def get_order_status(self, order_id: int) -> OrderStatus
    def get_open_orders(self) -> List[OrderStatus]
    def get_execution_history(self, filter: ExecutionFilter) -> List[Execution]
```

**Order Types Supported:**
- Market Orders
- Limit Orders
- Stop Orders
- Stop-Limit Orders
- Bracket Orders (Parent + Profit Target + Stop Loss)
- One-Cancels-Other (OCO) Orders
- Good-Till-Cancelled (GTC)
- Immediate-or-Cancel (IOC)
- IB Algo Orders (TWAP, VWAP, etc.)

#### 1.2 Position Manager (`common/trading/position_manager.py`)

**Responsibilities:**
- Real-time position tracking
- P&L calculation
- Portfolio metrics
- Risk exposure monitoring

**Key Features:**
```python
class PositionManager:
    def get_positions(self) -> List[Position]
    def get_position(self, contract: Contract) -> Optional[Position]
    def calculate_portfolio_pnl(self) -> PortfolioPnL
    def get_risk_metrics(self) -> RiskMetrics
    def check_position_limits(self, proposed_trade: Trade) -> bool
```

#### 1.3 Strategy Engine (`common/trading/strategy_engine.py`)

**Responsibilities:**
- Strategy definition and execution logic
- Signal generation from market data
- Strategy backtesting capabilities
- Performance attribution

**Supported Strategies:**
- **Calendar Spreads**: Different expiration dates, same strike
- **Straddles**: Same strike, same expiration (both call and put)
- **Strangles**: Different strikes, same expiration
- **Iron Condors**: Four-leg spread strategy
- **Covered Calls**: Long stock + short call
- **Protective Puts**: Long stock + long put
- **Custom Earnings Plays**: Based on volatility analysis

### 2. Risk Management (`common/trading/risk_manager.py`)

#### 2.1 Pre-Trade Risk Checks
- **Position Limits**: Maximum position size per symbol
- **Sector Concentration**: Limits on sector exposure
- **Portfolio Heat**: Overall risk exposure limits
- **Liquidity Requirements**: Minimum volume and open interest
- **Greeks Limits**: Delta, gamma, theta, vega exposure limits

#### 2.2 Real-Time Risk Monitoring
- **Portfolio Delta**: Net directional exposure
- **Volatility Risk**: Vega exposure monitoring
- **Time Decay**: Theta exposure tracking
- **Stop Loss Triggers**: Automatic exit conditions
- **Margin Requirements**: Real-time margin calculation

```python
class RiskManager:
    def pre_trade_check(self, trade: ProposedTrade) -> RiskCheckResult
    def monitor_positions(self) -> List[RiskAlert]
    def calculate_var(self, confidence_level: float = 0.95) -> float
    def get_stress_test_scenarios(self) -> List[StressTestResult]
    def enforce_stop_losses(self) -> List[OrderExecution]
```

### 3. Data Integration Layer

#### 3.1 Market Data Feed (`common/trading/market_data.py`)

**Integration with Existing Repositories:**
- Leverage existing `EquityBarRepository` for historical prices
- Extend `OptionChainRepository` for current option data
- Real-time market data streaming via IB API

**Features:**
- Real-time quotes (bid/ask/last)
- Greeks calculation (delta, gamma, theta, vega)
- Implied volatility surfaces
- Historical volatility analysis
- Volume and open interest tracking

#### 3.2 Earnings Integration

**Event-Driven Strategy Triggers:**
```python
class EarningsTradeGenerator:
    def on_earnings_announcement(self, event: EarningsEvent) -> List[TradeSignal]
    def analyze_iv_rank(self, symbol: str) -> IVAnalysis
    def generate_calendar_opportunities(self, earnings_date: date) -> List[CalendarSpread]
    def screen_straddle_candidates(self, days_to_earnings: int) -> List[StraddleOpportunity]
```

## Technical Specifications

### 4. API Endpoints (`services/trading_api/`)

#### 4.1 Order Management Endpoints

```
POST   /trading/orders                    # Place new order
GET    /trading/orders/{order_id}         # Get order status
PUT    /trading/orders/{order_id}         # Modify order
DELETE /trading/orders/{order_id}         # Cancel order
GET    /trading/orders                    # List orders (with filters)
GET    /trading/executions                # Get execution history
```

#### 4.2 Position Management Endpoints

```
GET    /trading/positions                 # Get all positions
GET    /trading/positions/{symbol}        # Get position for symbol
GET    /trading/portfolio/pnl             # Portfolio P&L
GET    /trading/portfolio/risk            # Risk metrics
GET    /trading/portfolio/performance     # Performance analytics
```

#### 4.3 Strategy Endpoints

```
GET    /trading/strategies                # List available strategies
POST   /trading/strategies/{strategy}/backtest  # Backtest strategy
GET    /trading/strategies/{strategy}/signals   # Get current signals
POST   /trading/strategies/{strategy}/execute   # Execute strategy
GET    /trading/strategies/performance    # Strategy performance
```

### 5. Data Models (`common/schemas/trading.py`)

#### 5.1 Order Models

```python
class Order(BaseModel):
    order_id: Optional[int] = None
    contract: Union[EquityContract, OptionContract]
    action: Literal["BUY", "SELL"]
    quantity: int
    order_type: OrderType
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    account: Optional[str] = None
    
class OrderStatus(BaseModel):
    order_id: int
    status: OrderStatusType
    filled_quantity: int = 0
    remaining_quantity: int = 0
    avg_fill_price: Optional[Decimal] = None
    commission: Optional[Decimal] = None
    timestamp: datetime
```

#### 5.2 Position Models

```python
class Position(BaseModel):
    contract: Union[EquityContract, OptionContract]
    position_size: int  # Positive for long, negative for short
    avg_cost: Decimal
    market_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal("0")
    
class PortfolioSummary(BaseModel):
    total_value: Decimal
    buying_power: Decimal
    net_liquidation: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    positions: List[Position]
```

#### 5.3 Strategy Models

```python
class TradeSignal(BaseModel):
    signal_id: str
    strategy_name: str
    symbol: str
    signal_type: SignalType  # ENTRY, EXIT, MODIFY
    confidence: float  # 0.0 to 1.0
    target_positions: List[PositionTarget]
    risk_metrics: RiskAssessment
    generated_at: datetime
    expires_at: Optional[datetime] = None

class PositionTarget(BaseModel):
    contract: Union[EquityContract, OptionContract] 
    target_quantity: int
    entry_price_range: Optional[Tuple[Decimal, Decimal]] = None
    stop_loss: Optional[Decimal] = None
    profit_target: Optional[Decimal] = None
```

### 6. Database Schema Extensions

#### 6.1 Trading Tables

```sql
-- Orders table
CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    parent_order_id BIGINT NULL,
    strategy_id VARCHAR(50) NULL,
    account VARCHAR(20) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    sec_type VARCHAR(10) NOT NULL,
    action VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    filled_qty INTEGER DEFAULT 0,
    avg_fill_price DECIMAL(10,4) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Positions table  
CREATE TABLE positions (
    position_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    account VARCHAR(20) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    sec_type VARCHAR(10) NOT NULL,
    position_size INTEGER NOT NULL,
    avg_cost DECIMAL(10,4) NOT NULL,
    realized_pnl DECIMAL(15,2) DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_position (account, symbol, sec_type)
);

-- Strategy Performance table
CREATE TABLE strategy_performance (
    performance_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    strategy_name VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    entry_price DECIMAL(10,4) NOT NULL,
    exit_price DECIMAL(10,4) NULL,
    pnl DECIMAL(15,2) NULL,
    return_pct DECIMAL(8,4) NULL,
    trade_status VARCHAR(20) DEFAULT 'OPEN',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 7. Event Bus Integration

#### 7.1 Trading Events

```python
# New event types to add to existing event bus
TRADING_EVENTS = [
    "order_placed",      # Order submitted to broker
    "order_filled",      # Order execution completed
    "order_cancelled",   # Order cancellation
    "position_opened",   # New position established
    "position_closed",   # Position fully closed
    "strategy_signal",   # Strategy generated signal
    "risk_alert",        # Risk threshold exceeded
    "pnl_update",        # P&L recalculation
]
```

#### 7.2 Event Handlers

```python
class TradingEventHandler:
    def on_earnings_event(self, event: EarningsEvent):
        """Generate trading signals based on earnings announcements."""
        signals = self.strategy_engine.analyze_earnings_opportunity(event)
        for signal in signals:
            self.event_bus.publish("strategy_signal", signal)
    
    def on_option_chains_event(self, event: OptionChainsEvent):
        """Update option strategies based on new chain data."""
        updated_strategies = self.strategy_engine.refresh_option_strategies(event)
        for strategy in updated_strategies:
            self.event_bus.publish("strategy_updated", strategy)
```

## Security and Risk Controls

### 8.1 Authentication and Authorization
- API key-based authentication for trading endpoints
- Role-based access control (RBAC)
- Audit logging for all trading operations
- Rate limiting on order submission

### 8.2 Risk Controls
- Maximum position size limits per symbol
- Maximum portfolio value at risk (VaR)
- Daily loss limits with automatic shutdown
- Pre-market risk checks before order placement
- Real-time margin monitoring

### 8.3 Circuit Breakers
- Automatic trading halt on excessive losses
- Volatility-based trading suspension
- Connection loss failsafe procedures
- Manual emergency stop functionality

## Implementation Phases

### Phase 1: Core Infrastructure (4-6 weeks)
1. Order Manager implementation
2. Position Manager development
3. Basic risk controls
4. Database schema setup
5. Core API endpoints

### Phase 2: Strategy Framework (6-8 weeks)  
1. Strategy engine architecture
2. Basic strategies (straddles, strangles)
3. Backtesting framework
4. Signal generation system
5. Event bus integration

### Phase 3: Advanced Features (8-10 weeks)
1. Complex multi-leg strategies
2. Real-time risk monitoring
3. Performance analytics
4. Advanced order types
5. Portfolio optimization

### Phase 4: Production Readiness (4-6 weeks)
1. Comprehensive testing
2. Security hardening  
3. Monitoring and alerting
4. Documentation and training
5. Production deployment

## Configuration

### 8.1 Environment Variables
```bash
# Trading Engine Configuration
TRADING_ENABLED=true
TRADING_MODE=paper|live
MAX_DAILY_LOSS=10000.00
MAX_POSITION_SIZE=1000
RISK_CHECK_ENABLED=true

# IB API Configuration  
IB_TRADING_HOST=127.0.0.1
IB_TRADING_PORT=7497  # Paper trading
IB_TRADING_CLIENT_ID=2
```

### 8.2 Risk Configuration (`config/risk_limits.yaml`)
```yaml
position_limits:
  max_position_value: 50000
  max_sector_concentration: 0.3
  max_single_position: 0.1

risk_metrics:
  max_portfolio_delta: 1000
  max_portfolio_gamma: 500
  max_portfolio_vega: 2000
  var_confidence_level: 0.95
  
stop_loss:
  default_stop_pct: 0.15
  trailing_stop_enabled: true
  max_loss_per_trade: 1000
```

## Monitoring and Observability

### 9.1 Metrics
- Orders per second
- Fill rates and slippage
- Strategy performance
- Risk exposure levels
- P&L attribution

### 9.2 Alerts
- Failed order executions
- Risk limit breaches  
- Unusual P&L movements
- Connection issues
- Strategy performance degradation

### 9.3 Dashboards
- Real-time portfolio view
- Strategy performance tracking
- Risk exposure monitoring
- Order execution analysis
- Market data health checks

## Testing Strategy

### 10.1 Unit Testing
- Order management logic
- Position calculations
- Risk management rules
- Strategy algorithms
- Data model validation

### 10.2 Integration Testing
- IB API integration
- Database operations
- Event bus messaging
- API endpoint testing
- Cross-service communication

### 10.3 Paper Trading
- Full system testing with paper account
- Strategy validation
- Risk system verification
- Performance monitoring
- User acceptance testing

This specification provides a comprehensive foundation for implementing a production-ready trading engine that integrates seamlessly with the existing earnings-focused data platform while maintaining high standards for risk management, security, and operational reliability.