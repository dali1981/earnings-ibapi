"""
Unit tests for IB API integration.
Following TDD approach - these tests will drive the implementation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from decimal import Decimal
from datetime import datetime
import asyncio

# These imports will fail initially (Red phase)
from common.trading.ib_integration import (
    IBClient, IBConnectionError, IBOrderError, IBContract, IBOrder
)
from common.schemas.trading import (
    Order, OrderAction, OrderType, OrderStatus, TimeInForce, 
    EquityContract, OptionContract
)
from common.constants import SecType, OptionRight


@pytest.fixture
def mock_ib_wrapper():
    """Mock IB API wrapper."""
    mock_wrapper = Mock()
    mock_wrapper.connect = AsyncMock(return_value=True)
    mock_wrapper.disconnect = AsyncMock()
    mock_wrapper.placeOrder = Mock()
    mock_wrapper.cancelOrder = Mock()
    mock_wrapper.reqAllOpenOrders = Mock()
    mock_wrapper.isConnected = Mock(return_value=True)
    return mock_wrapper


@pytest.fixture
def ib_client(mock_ib_wrapper):
    """IBClient instance for testing."""
    return IBClient(ib_wrapper=mock_ib_wrapper, client_id=1)


@pytest.fixture
def sample_equity_contract():
    """Sample equity contract for testing."""
    return EquityContract(
        symbol="AAPL",
        exchange="SMART",
        currency="USD"
    )


@pytest.fixture
def sample_option_contract():
    """Sample option contract for testing."""
    from datetime import date
    return OptionContract(
        symbol="AAPL",
        strike=150.0,
        right=OptionRight.CALL,
        expiry=date(2024, 11, 15),
        exchange="SMART",
        currency="USD"
    )


@pytest.fixture
def sample_order(sample_equity_contract):
    """Sample order for testing."""
    return Order(
        client_order_id="test-order-123",
        contract=sample_equity_contract,
        action=OrderAction.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
        created_at=datetime(2024, 1, 15, 10, 30, 0)
    )


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.order_management
class TestIBClient:
    """Test IBClient functionality."""
    
    def test_ib_client_initialization(self, ib_client):
        """Test IBClient initializes correctly."""
        assert ib_client.client_id == 1
        assert ib_client.ib_wrapper is not None
        assert hasattr(ib_client, 'next_order_id')
        assert ib_client.next_order_id >= 1
        assert ib_client.is_connected() is True
    
    async def test_connect_success(self, ib_client, mock_ib_wrapper):
        """Test successful connection to IB."""
        mock_ib_wrapper.connect.return_value = True
        
        result = await ib_client.connect("127.0.0.1", 7497)
        
        assert result is True
        mock_ib_wrapper.connect.assert_called_once_with("127.0.0.1", 7497, 1)
    
    async def test_connect_failure(self, ib_client, mock_ib_wrapper):
        """Test connection failure to IB."""
        mock_ib_wrapper.connect.side_effect = Exception("Connection failed")
        
        with pytest.raises(IBConnectionError) as exc_info:
            await ib_client.connect("127.0.0.1", 7497)
        
        assert "connection failed" in str(exc_info.value).lower()
    
    async def test_disconnect(self, ib_client, mock_ib_wrapper):
        """Test disconnection from IB."""
        await ib_client.disconnect()
        
        mock_ib_wrapper.disconnect.assert_called_once()
    
    async def test_place_order_market_order(self, ib_client, sample_order, mock_ib_wrapper):
        """Test placing a market order."""
        expected_order_id = 1001
        ib_client.next_order_id = expected_order_id
        
        order_id = await ib_client.place_order(sample_order)
        
        assert order_id == expected_order_id
        assert ib_client.next_order_id == expected_order_id + 1
        
        # Verify IB wrapper was called
        mock_ib_wrapper.placeOrder.assert_called_once()
        call_args = mock_ib_wrapper.placeOrder.call_args[0]
        
        # Verify order ID
        assert call_args[0] == expected_order_id
        
        # Verify IB contract
        ib_contract = call_args[1]
        assert ib_contract.symbol == "AAPL"
        assert ib_contract.secType == "STK"
        assert ib_contract.exchange == "SMART"
        assert ib_contract.currency == "USD"
        
        # Verify IB order
        ib_order = call_args[2]
        assert ib_order.action == "BUY"
        assert ib_order.totalQuantity == 100
        assert ib_order.orderType == "MKT"
    
    async def test_place_order_limit_order(self, ib_client, sample_equity_contract, mock_ib_wrapper):
        """Test placing a limit order."""
        limit_order = Order(
            client_order_id="limit-test",
            contract=sample_equity_contract,
            action=OrderAction.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("200.00"),
            time_in_force=TimeInForce.GTC
        )
        
        expected_order_id = 1001
        ib_client.next_order_id = expected_order_id
        
        order_id = await ib_client.place_order(limit_order)
        
        assert order_id == expected_order_id
        
        # Verify IB order parameters
        call_args = mock_ib_wrapper.placeOrder.call_args[0]
        ib_order = call_args[2]
        
        assert ib_order.action == "SELL"
        assert ib_order.totalQuantity == 50
        assert ib_order.orderType == "LMT"
        assert ib_order.lmtPrice == 200.00
        assert ib_order.tif == "GTC"
    
    async def test_place_order_stop_order(self, ib_client, sample_equity_contract, mock_ib_wrapper):
        """Test placing a stop order."""
        stop_order = Order(
            client_order_id="stop-test",
            contract=sample_equity_contract,
            action=OrderAction.SELL,
            quantity=100,
            order_type=OrderType.STOP,
            stop_price=Decimal("140.00")
        )
        
        expected_order_id = 1001
        ib_client.next_order_id = expected_order_id
        
        order_id = await ib_client.place_order(stop_order)
        
        assert order_id == expected_order_id
        
        # Verify IB order parameters
        call_args = mock_ib_wrapper.placeOrder.call_args[0]
        ib_order = call_args[2]
        
        assert ib_order.orderType == "STP"
        assert ib_order.auxPrice == 140.00
    
    async def test_place_order_option_contract(self, ib_client, sample_option_contract, mock_ib_wrapper):
        """Test placing order with option contract."""
        option_order = Order(
            client_order_id="option-test",
            contract=sample_option_contract,
            action=OrderAction.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        
        order_id = await ib_client.place_order(option_order)
        
        # Verify IB contract for option
        call_args = mock_ib_wrapper.placeOrder.call_args[0]
        ib_contract = call_args[1]
        
        assert ib_contract.symbol == "AAPL"
        assert ib_contract.secType == "OPT"
        assert ib_contract.strike == 150.0
        assert ib_contract.right == "C"
        assert ib_contract.lastTradeDateOrContractMonth == "20241115"
        assert ib_contract.multiplier == "100"
    
    async def test_cancel_order(self, ib_client, mock_ib_wrapper):
        """Test cancelling an order."""
        order_id = 1001
        
        await ib_client.cancel_order(order_id)
        
        mock_ib_wrapper.cancelOrder.assert_called_once_with(order_id)
    
    async def test_modify_order(self, ib_client, sample_order, mock_ib_wrapper):
        """Test modifying an order."""
        order_id = 1001
        
        # Mock order retrieval
        ib_client._active_orders = {order_id: sample_order}
        
        modified_order = await ib_client.modify_order(
            order_id,
            limit_price=Decimal("155.00"),
            quantity=150
        )
        
        assert modified_order.limit_price == Decimal("155.00")
        assert modified_order.quantity == 150
        
        # Verify place order was called (modification = cancel + new order)
        mock_ib_wrapper.placeOrder.assert_called()
    
    def test_convert_contract_to_ib_equity(self, ib_client, sample_equity_contract):
        """Test converting equity contract to IB contract."""
        ib_contract = ib_client._convert_contract_to_ib(sample_equity_contract)
        
        assert isinstance(ib_contract, IBContract)
        assert ib_contract.symbol == "AAPL"
        assert ib_contract.secType == "STK"
        assert ib_contract.exchange == "SMART"
        assert ib_contract.currency == "USD"
    
    def test_convert_contract_to_ib_option(self, ib_client, sample_option_contract):
        """Test converting option contract to IB contract."""
        ib_contract = ib_client._convert_contract_to_ib(sample_option_contract)
        
        assert isinstance(ib_contract, IBContract)
        assert ib_contract.symbol == "AAPL"
        assert ib_contract.secType == "OPT"
        assert ib_contract.strike == 150.0
        assert ib_contract.right == "C"
        assert ib_contract.lastTradeDateOrContractMonth == "20241115"
        assert ib_contract.multiplier == "100"
    
    def test_convert_order_to_ib_market(self, ib_client, sample_order):
        """Test converting market order to IB order."""
        ib_order = ib_client._convert_order_to_ib(sample_order)
        
        assert isinstance(ib_order, IBOrder)
        assert ib_order.action == "BUY"
        assert ib_order.totalQuantity == 100
        assert ib_order.orderType == "MKT"
        assert ib_order.clientId == 1
        assert ib_order.orderId is None  # Set later by place_order
    
    def test_convert_order_to_ib_limit(self, ib_client, sample_equity_contract):
        """Test converting limit order to IB order."""
        limit_order = Order(
            client_order_id="limit-test",
            contract=sample_equity_contract,
            action=OrderAction.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("200.00"),
            time_in_force=TimeInForce.GTC
        )
        
        ib_order = ib_client._convert_order_to_ib(limit_order)
        
        assert ib_order.action == "SELL"
        assert ib_order.totalQuantity == 50
        assert ib_order.orderType == "LMT"
        assert ib_order.lmtPrice == 200.00
        assert ib_order.tif == "GTC"
    
    def test_convert_order_to_ib_stop(self, ib_client, sample_equity_contract):
        """Test converting stop order to IB order."""
        stop_order = Order(
            client_order_id="stop-test",
            contract=sample_equity_contract,
            action=OrderAction.SELL,
            quantity=100,
            order_type=OrderType.STOP,
            stop_price=Decimal("140.00")
        )
        
        ib_order = ib_client._convert_order_to_ib(stop_order)
        
        assert ib_order.orderType == "STP"
        assert ib_order.auxPrice == 140.00
    
    def test_generate_order_id(self, ib_client):
        """Test order ID generation."""
        initial_id = ib_client.next_order_id
        
        id1 = ib_client._generate_order_id()
        id2 = ib_client._generate_order_id()
        
        assert id1 == initial_id
        assert id2 == initial_id + 1
        assert ib_client.next_order_id == initial_id + 2
    
    def test_is_connected(self, ib_client, mock_ib_wrapper):
        """Test connection status check."""
        mock_ib_wrapper.isConnected.return_value = True
        assert ib_client.is_connected() is True
        
        mock_ib_wrapper.isConnected.return_value = False
        assert ib_client.is_connected() is False
    
    async def test_place_order_connection_error(self, ib_client, sample_order, mock_ib_wrapper):
        """Test placing order when not connected."""
        mock_ib_wrapper.isConnected.return_value = False
        
        with pytest.raises(IBConnectionError) as exc_info:
            await ib_client.place_order(sample_order)
        
        assert "not connected" in str(exc_info.value).lower()
    
    async def test_place_order_ib_error(self, ib_client, sample_order, mock_ib_wrapper):
        """Test handling IB API errors during order placement."""
        mock_ib_wrapper.placeOrder.side_effect = Exception("IB API Error")
        
        with pytest.raises(IBOrderError) as exc_info:
            await ib_client.place_order(sample_order)
        
        assert "ib api error" in str(exc_info.value).lower()


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.order_management
class TestIBContractConversion:
    """Test IB contract conversion functionality."""
    
    def test_ib_contract_creation(self):
        """Test IBContract creation."""
        contract = IBContract()
        contract.symbol = "AAPL"
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        assert contract.symbol == "AAPL"
        assert contract.secType == "STK"
        assert contract.exchange == "SMART"
        assert contract.currency == "USD"
    
    def test_ib_order_creation(self):
        """Test IBOrder creation."""
        order = IBOrder()
        order.action = "BUY"
        order.totalQuantity = 100
        order.orderType = "MKT"
        order.clientId = 1
        
        assert order.action == "BUY"
        assert order.totalQuantity == 100
        assert order.orderType == "MKT"
        assert order.clientId == 1


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.order_management
class TestIBIntegrationErrors:
    """Test IB integration error handling."""
    
    def test_ib_connection_error_creation(self):
        """Test IBConnectionError creation."""
        error = IBConnectionError("Connection failed")
        assert str(error) == "Connection failed"
        assert isinstance(error, Exception)
    
    def test_ib_order_error_creation(self):
        """Test IBOrderError creation."""
        error = IBOrderError("Order rejected")
        assert str(error) == "Order rejected"
        assert isinstance(error, Exception)
    
    def test_ib_connection_error_with_details(self):
        """Test IBConnectionError with additional details."""
        details = {"host": "127.0.0.1", "port": 7497, "client_id": 1}
        error = IBConnectionError("Connection timeout", details=details)
        
        assert "Connection timeout" in str(error)
        assert error.details == details
    
    def test_ib_order_error_with_order_id(self):
        """Test IBOrderError with order ID."""
        error = IBOrderError("Order rejected", order_id=12345)
        
        assert "Order rejected" in str(error)
        assert error.order_id == 12345


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.order_management
class TestIBClientIntegration:
    """Integration tests for IBClient."""
    
    async def test_full_order_flow_simulation(self, ib_client, sample_order, mock_ib_wrapper):
        """Test complete order flow simulation."""
        # Setup order ID sequence
        ib_client.next_order_id = 1001
        
        # 1. Place order
        order_id = await ib_client.place_order(sample_order)
        assert order_id == 1001
        
        # Verify order was placed
        mock_ib_wrapper.placeOrder.assert_called_once()
        
        # 2. Cancel order
        await ib_client.cancel_order(order_id)
        mock_ib_wrapper.cancelOrder.assert_called_once_with(order_id)
    
    async def test_multiple_orders_sequence(self, ib_client, sample_equity_contract, mock_ib_wrapper):
        """Test placing multiple orders in sequence."""
        orders = []
        for i in range(3):
            order = Order(
                client_order_id=f"test-order-{i}",
                contract=sample_equity_contract,
                action=OrderAction.BUY,
                quantity=100 + i * 10,
                order_type=OrderType.MARKET
            )
            orders.append(order)
        
        # Place all orders
        order_ids = []
        for order in orders:
            order_id = await ib_client.place_order(order)
            order_ids.append(order_id)
        
        # Verify sequential order IDs
        for i, order_id in enumerate(order_ids):
            assert order_id == ib_client.next_order_id - len(order_ids) + i
        
        # Verify all orders were placed
        assert mock_ib_wrapper.placeOrder.call_count == 3