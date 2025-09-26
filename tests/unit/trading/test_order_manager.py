"""
Unit tests for OrderManager.
Following TDD approach - these tests will drive the implementation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from decimal import Decimal
from datetime import datetime, date
from pathlib import Path
import pandas as pd

# These imports will fail initially (Red phase)
from common.trading.order_manager import OrderManager, OrderError, OrderValidationError
from common.trading.events import OrderEvent, OrderEventType
from common.schemas.trading import (
    Order, OrderAction, OrderType, OrderStatus, TimeInForce, 
    EquityContract, OptionContract
)
from common.repositories.trading.orders import OrderRepository


@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary directory for test data."""
    return str(tmp_path)


@pytest.fixture
def mock_order_repo():
    """Mock order repository for testing."""
    return Mock(spec=OrderRepository)


@pytest.fixture
def mock_ib_client():
    """Mock IB client for testing."""
    mock_client = Mock()
    mock_client.connect = AsyncMock(return_value=True)
    mock_client.disconnect = AsyncMock()
    mock_client.place_order = AsyncMock()
    mock_client.cancel_order = AsyncMock()
    mock_client.modify_order = AsyncMock()
    return mock_client


@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing."""
    mock_bus = Mock()
    mock_bus.publish = Mock()
    return mock_bus


@pytest.fixture
def order_manager(mock_order_repo, mock_ib_client, mock_event_bus):
    """OrderManager instance for testing."""
    return OrderManager(
        order_repository=mock_order_repo,
        ib_client=mock_ib_client,
        event_bus=mock_event_bus
    )


@pytest.fixture
def sample_equity_order():
    """Sample equity order for testing."""
    contract = EquityContract(symbol="AAPL")
    return Order(
        client_order_id="test-order-123",
        contract=contract,
        action=OrderAction.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
        created_at=datetime(2024, 1, 15, 10, 30, 0)
    )


@pytest.fixture
def sample_limit_order():
    """Sample limit order for testing."""
    contract = EquityContract(symbol="MSFT")
    return Order(
        client_order_id="test-limit-456",
        contract=contract,
        action=OrderAction.SELL,
        quantity=50,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("300.00"),
        time_in_force=TimeInForce.GTC,
        created_at=datetime(2024, 1, 15, 14, 45, 0)
    )


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.order_management
class TestOrderManager:
    """Test OrderManager functionality."""
    
    def test_order_manager_initialization(self, order_manager):
        """Test OrderManager initializes with required dependencies."""
        assert order_manager.order_repository is not None
        assert order_manager.ib_client is not None
        assert order_manager.event_bus is not None
        assert hasattr(order_manager, 'next_order_id')
        assert order_manager.next_order_id >= 1
    
    @pytest.mark.asyncio
    async def test_submit_order_success(self, order_manager, sample_equity_order, mock_ib_client, mock_order_repo, mock_event_bus):
        """Test successful order submission."""
        # Mock IB client response
        ib_order_id = 12345
        mock_ib_client.place_order.return_value = ib_order_id
        
        # Submit order
        result = await order_manager.submit_order(sample_equity_order)
        
        # Verify result
        assert result.order_id == ib_order_id
        assert result.status == OrderStatus.PENDING_SUBMIT
        assert result.client_order_id == sample_equity_order.client_order_id
        
        # Verify order was saved to repository
        mock_order_repo.domain_save.assert_called_once()
        saved_order = mock_order_repo.domain_save.call_args[0][0]
        assert saved_order.order_id == ib_order_id
        assert saved_order.status == OrderStatus.PENDING_SUBMIT
        
        # Verify IB client was called
        mock_ib_client.place_order.assert_called_once()
        
        # Verify event was published
        mock_event_bus.publish.assert_called_once()
        event_args = mock_event_bus.publish.call_args[0]
        assert isinstance(event_args[0], OrderEvent)
        assert event_args[0].event_type == OrderEventType.ORDER_SUBMITTED
    
    @pytest.mark.asyncio
    async def test_submit_order_validation_error(self, order_manager, sample_equity_order):
        """Test order submission with validation errors."""
        # Create invalid order (negative quantity)
        invalid_order = Order(
            client_order_id="invalid-order",
            contract=sample_equity_order.contract,
            action=OrderAction.BUY,
            quantity=-100,  # Invalid
            order_type=OrderType.MARKET
        )
        
        # Should raise validation error
        with pytest.raises(OrderValidationError) as exc_info:
            await order_manager.submit_order(invalid_order)
        
        assert "quantity must be positive" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_submit_order_ib_connection_error(self, order_manager, sample_equity_order, mock_ib_client):
        """Test order submission when IB client connection fails."""
        # Mock IB client connection failure
        mock_ib_client.place_order.side_effect = ConnectionError("IB connection lost")
        
        # Should raise OrderError
        with pytest.raises(OrderError) as exc_info:
            await order_manager.submit_order(sample_equity_order)
        
        assert "ib connection" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_cancel_order_success(self, order_manager, mock_ib_client, mock_order_repo, mock_event_bus):
        """Test successful order cancellation."""
        order_id = 12345
        
        # Mock order retrieval
        mock_order = Mock()
        mock_order.order_id = order_id
        mock_order.status = OrderStatus.SUBMITTED
        mock_order_repo.get_orders.return_value = [mock_order]
        
        # Cancel order
        result = await order_manager.cancel_order(order_id)
        
        # Verify result
        assert result is True
        
        # Verify IB client was called
        mock_ib_client.cancel_order.assert_called_once_with(order_id)
        
        # Verify order status was updated
        mock_order_repo.update_order_status.assert_called_once_with(
            order_id, OrderStatus.PENDING_CANCEL
        )
        
        # Verify event was published
        mock_event_bus.publish.assert_called_once()
        event_args = mock_event_bus.publish.call_args[0]
        assert isinstance(event_args[0], OrderEvent)
        assert event_args[0].event_type == OrderEventType.CANCEL_REQUESTED
    
    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, order_manager, mock_order_repo):
        """Test cancelling non-existent order."""
        order_id = 99999
        
        # Mock empty order retrieval
        mock_order_repo.get_orders.return_value = []
        
        # Should raise OrderError
        with pytest.raises(OrderError) as exc_info:
            await order_manager.cancel_order(order_id)
        
        assert "order not found" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_cancel_order_invalid_status(self, order_manager, mock_order_repo):
        """Test cancelling order with invalid status."""
        order_id = 12345
        
        # Mock order with non-cancellable status
        mock_order = Mock()
        mock_order.order_id = order_id
        mock_order.status = OrderStatus.FILLED
        mock_order_repo.get_orders.return_value = [mock_order]
        
        # Should raise OrderError
        with pytest.raises(OrderError) as exc_info:
            await order_manager.cancel_order(order_id)
        
        assert "cannot cancel" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_modify_order_success(self, order_manager, sample_limit_order, mock_ib_client, mock_order_repo, mock_event_bus):
        """Test successful order modification."""
        order_id = 12345
        new_limit_price = Decimal("310.00")
        new_quantity = 75
        
        # Mock order retrieval
        existing_order = sample_limit_order
        existing_order.order_id = order_id
        existing_order.status = OrderStatus.SUBMITTED
        mock_order_repo.get_orders.return_value = [existing_order]
        
        # Modify order
        result = await order_manager.modify_order(
            order_id, 
            limit_price=new_limit_price,
            quantity=new_quantity
        )
        
        # Verify result
        assert result.order_id == order_id
        assert result.limit_price == new_limit_price
        assert result.quantity == new_quantity
        
        # Verify IB client was called
        mock_ib_client.modify_order.assert_called_once()
        
        # Verify order was saved
        mock_order_repo.domain_save.assert_called_once()
        
        # Verify event was published
        mock_event_bus.publish.assert_called_once()
        event_args = mock_event_bus.publish.call_args[0]
        assert isinstance(event_args[0], OrderEvent)
        assert event_args[0].event_type == OrderEventType.ORDER_MODIFIED
    
    def test_get_orders_by_status(self, order_manager, mock_order_repo):
        """Test retrieving orders by status."""
        status = OrderStatus.SUBMITTED
        
        # Mock repository response
        mock_orders = [Mock(), Mock()]
        mock_order_repo.get_orders.return_value = mock_orders
        
        # Get orders
        result = order_manager.get_orders_by_status(status)
        
        # Verify repository was called with correct filter
        mock_order_repo.get_orders.assert_called_once_with(status=status)
        assert result == mock_orders
    
    def test_get_orders_by_symbol(self, order_manager, mock_order_repo):
        """Test retrieving orders by symbol."""
        symbol = "AAPL"
        
        # Mock repository response
        mock_orders = [Mock(), Mock()]
        mock_order_repo.get_orders_by_symbol.return_value = mock_orders
        
        # Get orders
        result = order_manager.get_orders_by_symbol(symbol)
        
        # Verify repository was called
        mock_order_repo.get_orders_by_symbol.assert_called_once_with(symbol)
        assert result == mock_orders
    
    def test_get_order_by_id(self, order_manager, mock_order_repo):
        """Test retrieving single order by ID."""
        order_id = 12345
        
        # Mock repository response
        mock_order = Mock()
        mock_order.order_id = order_id
        mock_order_repo.get_orders.return_value = [mock_order]
        
        # Get order
        result = order_manager.get_order_by_id(order_id)
        
        # Verify result
        assert result == mock_order
        mock_order_repo.get_orders.assert_called_once()
    
    def test_get_order_by_id_not_found(self, order_manager, mock_order_repo):
        """Test retrieving non-existent order."""
        order_id = 99999
        
        # Mock empty repository response
        mock_order_repo.get_orders.return_value = []
        
        # Get order
        result = order_manager.get_order_by_id(order_id)
        
        # Should return None
        assert result is None
    
    @pytest.mark.asyncio
    async def test_handle_order_status_update(self, order_manager, mock_order_repo, mock_event_bus):
        """Test handling order status updates from IB."""
        order_id = 12345
        new_status = OrderStatus.FILLED
        filled_qty = 100
        avg_fill_price = 150.50
        
        # Handle status update
        await order_manager.handle_order_status_update(
            order_id, new_status, filled_qty, avg_fill_price
        )
        
        # Verify repository was updated
        mock_order_repo.update_order_status.assert_called_once_with(
            order_id, new_status, filled_qty, avg_fill_price
        )
        
        # Verify event was published
        mock_event_bus.publish.assert_called_once()
        event_args = mock_event_bus.publish.call_args[0]
        assert isinstance(event_args[0], OrderEvent)
        assert event_args[0].event_type == OrderEventType.ORDER_FILLED
    
    def test_validate_order_basic_validation(self, order_manager, sample_equity_order):
        """Test basic order validation."""
        # Valid order should not raise
        order_manager._validate_order(sample_equity_order)
        
        # Invalid quantity
        invalid_order = Order(
            client_order_id="invalid",
            contract=sample_equity_order.contract,
            action=OrderAction.BUY,
            quantity=0,  # Invalid
            order_type=OrderType.MARKET
        )
        
        with pytest.raises(OrderValidationError):
            order_manager._validate_order(invalid_order)
    
    def test_validate_order_limit_price_required(self, order_manager, sample_equity_order):
        """Test validation for limit orders requiring limit price."""
        # Limit order without limit price
        invalid_order = Order(
            client_order_id="invalid-limit",
            contract=sample_equity_order.contract,
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            # limit_price missing
        )
        
        with pytest.raises(OrderValidationError) as exc_info:
            order_manager._validate_order(invalid_order)
        
        assert "limit price required" in str(exc_info.value).lower()
    
    def test_validate_order_stop_price_required(self, order_manager, sample_equity_order):
        """Test validation for stop orders requiring stop price."""
        # Stop order without stop price
        invalid_order = Order(
            client_order_id="invalid-stop",
            contract=sample_equity_order.contract,
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.STOP,
            # stop_price missing
        )
        
        with pytest.raises(OrderValidationError) as exc_info:
            order_manager._validate_order(invalid_order)
        
        assert "stop price required" in str(exc_info.value).lower()
    
    def test_generate_order_id(self, order_manager):
        """Test order ID generation."""
        # Should generate sequential IDs
        id1 = order_manager._generate_order_id()
        id2 = order_manager._generate_order_id()
        
        assert isinstance(id1, int)
        assert isinstance(id2, int)
        assert id2 > id1
    
    def test_can_cancel_order_status_validation(self, order_manager):
        """Test order cancellation status validation."""
        # Cancellable statuses
        cancellable_statuses = [
            OrderStatus.PENDING_SUBMIT,
            OrderStatus.SUBMITTED,
            OrderStatus.PENDING_CANCEL
        ]
        
        for status in cancellable_statuses:
            mock_order = Mock()
            mock_order.status = status
            assert order_manager._can_cancel_order(mock_order) is True
        
        # Non-cancellable statuses
        non_cancellable_statuses = [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED
        ]
        
        for status in non_cancellable_statuses:
            mock_order = Mock()
            mock_order.status = status
            assert order_manager._can_cancel_order(mock_order) is False
    
    def test_can_modify_order_status_validation(self, order_manager):
        """Test order modification status validation."""
        # Modifiable statuses
        modifiable_statuses = [
            OrderStatus.PENDING_SUBMIT,
            OrderStatus.SUBMITTED
        ]
        
        for status in modifiable_statuses:
            mock_order = Mock()
            mock_order.status = status
            assert order_manager._can_modify_order(mock_order) is True
        
        # Non-modifiable statuses
        non_modifiable_statuses = [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.PENDING_CANCEL
        ]
        
        for status in non_modifiable_statuses:
            mock_order = Mock()
            mock_order.status = status
            assert order_manager._can_modify_order(mock_order) is False


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.order_management
class TestOrderManagerIntegration:
    """Integration tests for OrderManager with actual dependencies."""
    
    def test_order_manager_with_real_repository(self, temp_data_dir):
        """Test OrderManager with actual OrderRepository."""
        from common.repositories.trading.orders import OrderRepository
        
        # Create real repository
        order_repo = OrderRepository(temp_data_dir)
        
        # Create OrderManager with real repository
        mock_ib_client = Mock()
        mock_event_bus = Mock()
        
        manager = OrderManager(
            order_repository=order_repo,
            ib_client=mock_ib_client,
            event_bus=mock_event_bus
        )
        
        # Verify initialization
        assert manager.order_repository is order_repo
        assert isinstance(manager.order_repository, OrderRepository)
    
    @pytest.mark.asyncio
    async def test_full_order_lifecycle(self, order_manager, sample_equity_order, mock_ib_client, mock_order_repo, mock_event_bus):
        """Test complete order lifecycle: submit -> status updates -> fill."""
        ib_order_id = 12345
        
        # Setup mocks
        mock_ib_client.place_order.return_value = ib_order_id
        mock_order = Mock()
        mock_order.order_id = ib_order_id
        mock_order.status = OrderStatus.SUBMITTED
        mock_order_repo.get_orders.return_value = [mock_order]
        
        # 1. Submit order
        submitted_order = await order_manager.submit_order(sample_equity_order)
        assert submitted_order.order_id == ib_order_id
        assert submitted_order.status == OrderStatus.PENDING_SUBMIT
        
        # 2. Handle status update to SUBMITTED
        await order_manager.handle_order_status_update(ib_order_id, OrderStatus.SUBMITTED)
        
        # 3. Handle status update to FILLED
        await order_manager.handle_order_status_update(
            ib_order_id, OrderStatus.FILLED, filled_qty=100, avg_fill_price=150.25
        )
        
        # Verify all repository interactions
        assert mock_order_repo.domain_save.call_count == 1
        assert mock_order_repo.update_order_status.call_count == 2
        
        # Verify all events published
        assert mock_event_bus.publish.call_count == 3
        
        # Verify IB client interaction
        mock_ib_client.place_order.assert_called_once()


# Helper functions for tests
def create_mock_order_event(event_type: OrderEventType, order_id: int) -> OrderEvent:
    """Create a mock order event for testing."""
    return OrderEvent(
        event_type=event_type,
        order_id=order_id,
        timestamp=datetime.utcnow(),
        data={"test": "data"}
    )