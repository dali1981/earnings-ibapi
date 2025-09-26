"""
Unit tests for trading domain schemas and validation.
Following TDD approach - these tests will drive the implementation.
"""

import pytest
from decimal import Decimal
from datetime import datetime, date
from typing import Optional
from pydantic import ValidationError

# These imports will fail initially (Red phase)
from common.schemas.trading import (
    Order, OrderAction, OrderType, TimeInForce, OrderStatus,
    ExecutionReport, Position, PortfolioSummary,
    EquityContract, OptionContract, OptionRight,
    TradeSignal, SignalType, PositionTarget, RiskAssessment
)


@pytest.mark.unit
@pytest.mark.trading
class TestOrderAction:
    """Test OrderAction enum."""
    
    def test_order_action_values(self):
        """Test OrderAction enum has correct values."""
        assert OrderAction.BUY == "BUY"
        assert OrderAction.SELL == "SELL"


@pytest.mark.unit
@pytest.mark.trading
class TestOrderType:
    """Test OrderType enum."""
    
    def test_order_type_values(self):
        """Test OrderType enum has correct values."""
        assert OrderType.MARKET == "MKT"
        assert OrderType.LIMIT == "LMT" 
        assert OrderType.STOP == "STP"
        assert OrderType.STOP_LIMIT == "STP LMT"
        assert OrderType.BRACKET == "BRACKET"


@pytest.mark.unit
@pytest.mark.trading
class TestOrder:
    """Test Order schema validation and behavior."""
    
    def test_order_creation_with_valid_data(self):
        """Test creating order with all required fields."""
        contract = EquityContract(symbol="AAPL")
        order = Order(
            client_order_id="test-123",
            contract=contract,
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        assert order.client_order_id == "test-123"
        assert order.contract.symbol == "AAPL"
        assert order.action == OrderAction.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.time_in_force == TimeInForce.DAY  # Default
        assert order.order_id is None  # Not set initially
        assert isinstance(order.created_at, datetime)
    
    def test_order_validation_fails_with_zero_quantity(self):
        """Test order validation fails with zero quantity."""
        contract = EquityContract(symbol="AAPL")
        
        with pytest.raises(ValidationError) as exc_info:
            Order(
                client_order_id="test-123",
                contract=contract,
                action=OrderAction.BUY,
                quantity=0,  # Invalid
                order_type=OrderType.MARKET
            )
        
        assert "quantity" in str(exc_info.value)
    
    def test_order_validation_fails_with_negative_quantity(self):
        """Test order validation fails with negative quantity."""
        contract = EquityContract(symbol="AAPL")
        
        with pytest.raises(ValidationError) as exc_info:
            Order(
                client_order_id="test-123",
                contract=contract,
                action=OrderAction.BUY,
                quantity=-10,  # Invalid
                order_type=OrderType.MARKET
            )
        
        assert "quantity" in str(exc_info.value)
    
    def test_limit_order_requires_limit_price(self):
        """Test limit order validation requires limit price."""
        contract = EquityContract(symbol="AAPL")
        
        with pytest.raises(ValidationError) as exc_info:
            Order(
                client_order_id="test-123",
                contract=contract,
                action=OrderAction.BUY,
                quantity=100,
                order_type=OrderType.LIMIT,
                # limit_price missing
            )
        
        assert "Limit price required" in str(exc_info.value)
    
    def test_stop_order_requires_stop_price(self):
        """Test stop order validation requires stop price."""
        contract = EquityContract(symbol="AAPL")
        
        with pytest.raises(ValidationError) as exc_info:
            Order(
                client_order_id="test-123",
                contract=contract,
                action=OrderAction.BUY,
                quantity=100,
                order_type=OrderType.STOP,
                # stop_price missing
            )
        
        assert "Stop price required" in str(exc_info.value)
    
    def test_stop_limit_order_requires_both_prices(self):
        """Test stop-limit order requires both stop and limit prices."""
        contract = EquityContract(symbol="AAPL")
        
        with pytest.raises(ValidationError) as exc_info:
            Order(
                client_order_id="test-123",
                contract=contract,
                action=OrderAction.BUY,
                quantity=100,
                order_type=OrderType.STOP_LIMIT,
                limit_price=Decimal("150.00"),
                # stop_price missing
            )
        
        assert "Stop price required" in str(exc_info.value)
    
    def test_market_order_accepts_no_prices(self):
        """Test market order doesn't require limit or stop prices."""
        contract = EquityContract(symbol="AAPL")
        
        order = Order(
            client_order_id="test-123",
            contract=contract,
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET
            # No prices required
        )
        
        assert order.limit_price is None
        assert order.stop_price is None
    
    def test_order_to_ib_order_dict_conversion(self):
        """Test conversion to IB API order dictionary format."""
        contract = EquityContract(symbol="AAPL")
        order = Order(
            client_order_id="test-123",
            contract=contract,
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150.00"),
            time_in_force=TimeInForce.GTC
        )
        
        ib_order_dict = order.to_ib_order_dict()
        
        assert ib_order_dict["action"] == "BUY"
        assert ib_order_dict["totalQuantity"] == 100
        assert ib_order_dict["orderType"] == "LMT"
        assert ib_order_dict["lmtPrice"] == 150.00
        assert ib_order_dict["tif"] == "GTC"
        assert "auxPrice" not in ib_order_dict  # No stop price


@pytest.mark.unit
@pytest.mark.trading
class TestExecutionReport:
    """Test ExecutionReport schema."""
    
    def test_execution_report_creation(self):
        """Test creating execution report with valid data."""
        contract = EquityContract(symbol="AAPL")
        execution = ExecutionReport(
            execution_id="exec-123",
            order_id=12345,
            contract=contract,
            action=OrderAction.BUY,
            quantity=100,
            price=Decimal("150.50"),
            commission=Decimal("1.00")
        )
        
        assert execution.execution_id == "exec-123"
        assert execution.order_id == 12345
        assert execution.quantity == 100
        assert execution.price == Decimal("150.50")
        assert execution.commission == Decimal("1.00")
        assert isinstance(execution.timestamp, datetime)
    
    def test_execution_report_validation_positive_price(self):
        """Test execution report requires positive price."""
        contract = EquityContract(symbol="AAPL")
        
        with pytest.raises(ValidationError) as exc_info:
            ExecutionReport(
                execution_id="exec-123",
                order_id=12345,
                contract=contract,
                action=OrderAction.BUY,
                quantity=100,
                price=Decimal("-10.00"),  # Invalid negative price
                commission=Decimal("1.00")
            )
        
        assert "price" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.trading  
class TestPosition:
    """Test Position schema."""
    
    def test_position_creation_long(self):
        """Test creating long position."""
        contract = EquityContract(symbol="AAPL")
        position = Position(
            contract=contract,
            position_size=100,
            avg_cost=Decimal("150.00"),
            market_price=Decimal("155.00"),
            market_value=Decimal("15500.00"),
            unrealized_pnl=Decimal("500.00")
        )
        
        assert position.position_size == 100
        assert position.is_long is True
        assert position.is_short is False
        assert position.avg_cost == Decimal("150.00")
        assert position.unrealized_pnl == Decimal("500.00")
    
    def test_position_creation_short(self):
        """Test creating short position."""
        contract = EquityContract(symbol="AAPL")
        position = Position(
            contract=contract,
            position_size=-100,
            avg_cost=Decimal("150.00"),
            market_price=Decimal("145.00"),
            market_value=Decimal("-14500.00"),
            unrealized_pnl=Decimal("500.00")  # Profit from short
        )
        
        assert position.position_size == -100
        assert position.is_long is False
        assert position.is_short is True
        assert position.unrealized_pnl == Decimal("500.00")
    
    def test_position_update_market_price(self):
        """Test updating position market price."""
        contract = EquityContract(symbol="AAPL")
        position = Position(
            contract=contract,
            position_size=100,
            avg_cost=Decimal("150.00"),
            market_price=Decimal("155.00"),
            market_value=Decimal("15500.00"),
            unrealized_pnl=Decimal("500.00")
        )
        
        position.update_market_price(Decimal("160.00"))
        
        assert position.market_price == Decimal("160.00")
        assert position.market_value == Decimal("16000.00")  # 100 * 160
        assert position.unrealized_pnl == Decimal("1000.00")  # (160 - 150) * 100


@pytest.mark.unit
@pytest.mark.trading
class TestOptionContract:
    """Test OptionContract schema."""
    
    def test_option_contract_creation(self):
        """Test creating option contract."""
        option = OptionContract(
            symbol="AAPL",
            expiry=date(2024, 1, 19),
            strike=Decimal("150.00"),
            right=OptionRight.CALL,
            multiplier="100"
        )
        
        assert option.symbol == "AAPL"
        assert option.expiry == date(2024, 1, 19)
        assert option.strike == Decimal("150.00")
        assert option.right == OptionRight.CALL
        assert option.multiplier == "100"
    
    def test_option_symbol_generation(self):
        """Test standardized option symbol generation."""
        option = OptionContract(
            symbol="AAPL",
            expiry=date(2024, 1, 19),
            strike=Decimal("150.00"),
            right=OptionRight.CALL
        )
        
        option_symbol = option.option_symbol
        
        # Format: AAPL240119C00150000
        assert option_symbol.startswith("AAPL")
        assert "240119" in option_symbol  # Date format
        assert "C" in option_symbol  # Call
        assert "150000" in option_symbol  # Strike format
    
    def test_option_to_ib_contract_dict(self):
        """Test conversion to IB API contract dictionary."""
        option = OptionContract(
            symbol="AAPL",
            expiry=date(2024, 1, 19),
            strike=Decimal("150.00"),
            right=OptionRight.PUT,
            multiplier="100"
        )
        
        ib_dict = option.to_ib_contract_dict()
        
        assert ib_dict["symbol"] == "AAPL"
        assert ib_dict["secType"] == "OPT"
        assert ib_dict["lastTradeDateOrContractMonth"] == "20240119"
        assert ib_dict["strike"] == 150.0
        assert ib_dict["right"] == "P"
        assert ib_dict["multiplier"] == "100"


@pytest.mark.unit
@pytest.mark.trading
class TestTradeSignal:
    """Test TradeSignal schema."""
    
    def test_trade_signal_creation(self):
        """Test creating trade signal."""
        target_position = PositionTarget(
            contract=EquityContract(symbol="AAPL"),
            target_quantity=100,
            entry_price_range=(Decimal("149.00"), Decimal("151.00")),
            stop_loss=Decimal("140.00"),
            profit_target=Decimal("160.00")
        )
        
        risk_assessment = RiskAssessment(
            max_loss=Decimal("1000.00"),
            win_probability=0.65,
            risk_reward_ratio=2.5
        )
        
        signal = TradeSignal(
            signal_id="signal-123",
            strategy_name="earnings_straddle",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            confidence=0.8,
            target_positions=[target_position],
            risk_metrics=risk_assessment
        )
        
        assert signal.signal_id == "signal-123"
        assert signal.strategy_name == "earnings_straddle"
        assert signal.symbol == "AAPL"
        assert signal.confidence == 0.8
        assert len(signal.target_positions) == 1
        assert isinstance(signal.generated_at, datetime)
    
    def test_signal_validation_confidence_range(self):
        """Test signal confidence must be between 0 and 1."""
        with pytest.raises(ValidationError) as exc_info:
            TradeSignal(
                signal_id="signal-123",
                strategy_name="test",
                symbol="AAPL",
                signal_type=SignalType.ENTRY,
                confidence=1.5,  # Invalid > 1.0
                target_positions=[],
                risk_metrics=RiskAssessment(max_loss=Decimal("100"))
            )
        
        assert "confidence" in str(exc_info.value)