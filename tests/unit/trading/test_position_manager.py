"""
Unit tests for position manager.
Following TDD approach - these tests will drive the implementation.
"""

import pytest
from datetime import datetime, date
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock

# These imports will fail initially (Red phase)
from common.trading.position_manager import PositionManager
from common.schemas.trading import Position, EquityContract, OptionContract, ExecutionReport, OrderAction
from common.constants import SecType, OptionRight


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.position_management
class TestPositionManager:
    """Test PositionManager functionality."""
    
    @pytest.fixture
    def mock_position_repository(self):
        """Create mock position repository."""
        return Mock()
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        return Mock()
    
    @pytest.fixture
    def position_manager(self, mock_position_repository, mock_event_bus):
        """Create position manager for testing."""
        return PositionManager(
            position_repository=mock_position_repository,
            event_bus=mock_event_bus
        )
    
    @pytest.fixture
    def equity_contract(self):
        """Create test equity contract."""
        return EquityContract(symbol="AAPL", sec_type=SecType.STOCK)
    
    @pytest.fixture
    def option_contract(self):
        """Create test option contract."""
        return OptionContract(
            symbol="AAPL",
            sec_type=SecType.OPTION,
            strike=Decimal("150.00"),
            right=OptionRight.CALL,
            expiry=date(2024, 3, 15),
            multiplier="100"
        )
    
    @pytest.fixture
    def equity_execution(self, equity_contract):
        """Create test equity execution."""
        return ExecutionReport(
            execution_id="EXEC_001",
            order_id=12345,
            contract=equity_contract,
            action=OrderAction.BUY,
            quantity=100,
            price=Decimal("150.25"),
            commission=Decimal("1.00"),
            timestamp=datetime.utcnow()
        )
    
    @pytest.fixture
    def option_execution(self, option_contract):
        """Create test option execution."""
        return ExecutionReport(
            execution_id="EXEC_002",
            order_id=12346,
            contract=option_contract,
            action=OrderAction.BUY,
            quantity=2,
            price=Decimal("5.25"),
            commission=Decimal("2.50"),
            timestamp=datetime.utcnow()
        )
    
    def test_position_manager_initialization(self, position_manager, mock_position_repository, mock_event_bus):
        """Test PositionManager initialization."""
        assert position_manager.position_repository == mock_position_repository
        assert position_manager.event_bus == mock_event_bus
    
    def test_process_execution_new_equity_position(self, position_manager, mock_position_repository, equity_execution):
        """Test processing execution for new equity position."""
        # Mock no existing position
        mock_position_repository.get_position_by_contract.return_value = None
        
        # Process execution
        position_manager.process_execution(equity_execution, account="DU123456")
        
        # Verify repository calls
        mock_position_repository.get_position_by_contract.assert_called_once_with(
            equity_execution.contract, "DU123456"
        )
        
        # Verify new position saved
        mock_position_repository.domain_save.assert_called_once()
        saved_position = mock_position_repository.domain_save.call_args[0][0]
        
        assert isinstance(saved_position, Position)
        assert saved_position.contract.symbol == "AAPL"
        assert saved_position.position_size == 100
        assert saved_position.avg_cost == Decimal("150.25")
        assert saved_position.account == "DU123456"
    
    def test_process_execution_existing_equity_position(self, position_manager, mock_position_repository, equity_execution):
        """Test processing execution for existing equity position."""
        # Mock existing position
        existing_position = Position(
            contract=equity_execution.contract,
            position_size=50,
            avg_cost=Decimal("148.00"),
            market_price=Decimal("150.00"),
            market_value=Decimal("7500.00"),
            unrealized_pnl=Decimal("100.00"),
            account="DU123456"
        )
        mock_position_repository.get_position_by_contract.return_value = existing_position
        
        # Process execution (buy 100 more shares)
        position_manager.process_execution(equity_execution, account="DU123456")
        
        # Verify updated position saved
        mock_position_repository.domain_save.assert_called_once()
        saved_position = mock_position_repository.domain_save.call_args[0][0]
        
        assert saved_position.position_size == 150  # 50 + 100
        # Weighted average cost: (50 * 148.00 + 100 * 150.25) / 150 = 149.50
        assert abs(saved_position.avg_cost - Decimal("149.50")) < Decimal("0.01")
    
    def test_process_execution_sell_partial_position(self, position_manager, mock_position_repository, equity_execution):
        """Test processing sell execution for partial position close."""
        # Mock existing position
        existing_position = Position(
            contract=equity_execution.contract,
            position_size=200,
            avg_cost=Decimal("148.00"),
            market_price=Decimal("150.00"),
            market_value=Decimal("30000.00"),
            unrealized_pnl=Decimal("400.00"),
            account="DU123456"
        )
        mock_position_repository.get_position_by_contract.return_value = existing_position
        
        # Create sell execution
        sell_execution = ExecutionReport(
            execution_id="EXEC_SELL",
            order_id=12347,
            contract=equity_execution.contract,
            action=OrderAction.SELL,
            quantity=100,
            price=Decimal("152.00"),
            commission=Decimal("1.00"),
            timestamp=datetime.utcnow()
        )
        
        # Process sell execution
        position_manager.process_execution(sell_execution, account="DU123456")
        
        # Verify updated position saved
        mock_position_repository.domain_save.assert_called_once()
        saved_position = mock_position_repository.domain_save.call_args[0][0]
        
        assert saved_position.position_size == 100  # 200 - 100
        assert saved_position.avg_cost == Decimal("148.00")  # Cost basis unchanged
        # Realized P&L: (152.00 - 148.00) * 100 = 400.00
        assert saved_position.realized_pnl == Decimal("400.00")
    
    def test_process_execution_option_position(self, position_manager, mock_position_repository, option_execution):
        """Test processing execution for option position."""
        # Mock no existing position
        mock_position_repository.get_position_by_contract.return_value = None
        
        # Process execution
        position_manager.process_execution(option_execution, account="DU123456")
        
        # Verify new position saved
        mock_position_repository.domain_save.assert_called_once()
        saved_position = mock_position_repository.domain_save.call_args[0][0]
        
        assert isinstance(saved_position, Position)
        assert saved_position.contract.symbol == "AAPL"
        assert saved_position.position_size == 2
        assert saved_position.avg_cost == Decimal("5.25")
        assert saved_position.account == "DU123456"
    
    def test_update_market_prices(self, position_manager, mock_position_repository, equity_contract):
        """Test updating market prices for positions."""
        market_data = {
            "AAPL": Decimal("155.50")
        }
        
        # Process price updates
        position_manager.update_market_prices(market_data, account="DU123456")
        
        # Verify repository call
        mock_position_repository.update_position_prices.assert_called_once_with(
            equity_contract, Decimal("155.50"), account="DU123456"
        )
    
    def test_get_positions_by_account(self, position_manager, mock_position_repository):
        """Test retrieving positions by account."""
        # Mock repository response
        mock_positions = [
            Position(
                contract=EquityContract(symbol="AAPL", sec_type=SecType.STOCK),
                position_size=100,
                avg_cost=Decimal("150.00"),
                market_price=Decimal("155.00"),
                market_value=Decimal("15500.00"),
                unrealized_pnl=Decimal("500.00"),
                account="DU123456"
            )
        ]
        mock_position_repository.get_positions_by_account.return_value = mock_positions
        
        # Get positions
        positions = position_manager.get_positions_by_account("DU123456")
        
        # Verify result
        assert len(positions) == 1
        assert positions[0].contract.symbol == "AAPL"
        mock_position_repository.get_positions_by_account.assert_called_once_with("DU123456")
    
    def test_get_portfolio_summary(self, position_manager, mock_position_repository):
        """Test getting portfolio summary."""
        # Mock repository response
        mock_summary = {
            "total_positions": 2,
            "total_cost": 30000.00,
            "total_market_value": 31000.00,
            "total_unrealized_pnl": 1000.00,
            "total_realized_pnl": 500.00,
            "total_pnl": 1500.00
        }
        mock_position_repository.get_portfolio_summary.return_value = mock_summary
        
        # Get summary
        summary = position_manager.get_portfolio_summary("DU123456")
        
        # Verify result
        assert summary == mock_summary
        mock_position_repository.get_portfolio_summary.assert_called_once_with("DU123456")
    
    def test_close_position(self, position_manager, mock_position_repository, equity_contract):
        """Test closing a position."""
        realized_pnl = Decimal("525.00")
        
        # Close position
        position_manager.close_position(equity_contract, realized_pnl, account="DU123456")
        
        # Verify repository call
        mock_position_repository.close_position.assert_called_once_with(
            equity_contract, realized_pnl, account="DU123456"
        )
    
    def test_get_position_by_contract(self, position_manager, mock_position_repository, equity_contract):
        """Test getting specific position by contract."""
        # Mock repository response
        mock_position = Position(
            contract=equity_contract,
            position_size=100,
            avg_cost=Decimal("150.00"),
            market_price=Decimal("155.00"),
            market_value=Decimal("15500.00"),
            unrealized_pnl=Decimal("500.00"),
            account="DU123456"
        )
        mock_position_repository.get_position_by_contract.return_value = mock_position
        
        # Get position
        position = position_manager.get_position_by_contract(equity_contract, account="DU123456")
        
        # Verify result
        assert position == mock_position
        mock_position_repository.get_position_by_contract.assert_called_once_with(
            equity_contract, "DU123456"
        )
    
    def test_calculate_weighted_average_cost(self, position_manager):
        """Test weighted average cost calculation."""
        # Test buy into existing position
        existing_cost = Decimal("148.00")
        existing_quantity = 50
        new_cost = Decimal("152.00")
        new_quantity = 100
        
        avg_cost = position_manager._calculate_weighted_average_cost(
            existing_cost, existing_quantity, new_cost, new_quantity
        )
        
        # Expected: (50 * 148 + 100 * 152) / 150 = 150.67
        expected = Decimal("150.67")
        assert abs(avg_cost - expected) < Decimal("0.01")
    
    def test_calculate_realized_pnl(self, position_manager):
        """Test realized P&L calculation."""
        avg_cost = Decimal("148.00")
        sell_price = Decimal("152.00")
        quantity = 100
        
        realized_pnl = position_manager._calculate_realized_pnl(avg_cost, sell_price, quantity)
        
        # Expected: (152 - 148) * 100 = 400.00
        assert realized_pnl == Decimal("400.00")
    
    def test_event_publishing(self, position_manager, mock_event_bus, equity_execution):
        """Test event publishing during position updates."""
        # Mock no existing position
        position_manager.position_repository.get_position_by_contract.return_value = None
        
        # Process execution
        position_manager.process_execution(equity_execution, account="DU123456")
        
        # Verify event published
        mock_event_bus.publish.assert_called_once()
        published_event = mock_event_bus.publish.call_args[0][0]
        
        assert "POSITION_UPDATED" in str(published_event) or hasattr(published_event, 'event_type')
    
    def test_error_handling_invalid_execution(self, position_manager):
        """Test error handling for invalid execution."""
        # Test with None execution
        with pytest.raises(ValueError, match="Execution cannot be None"):
            position_manager.process_execution(None, account="DU123456")
        
        # Test with empty account
        with pytest.raises(ValueError, match="Account cannot be empty"):
            position_manager.process_execution(
                ExecutionReport(
                    execution_id="TEST",
                    order_id=1,
                    contract=EquityContract(symbol="TEST", sec_type=SecType.STOCK),
                    action=OrderAction.BUY,
                    quantity=100,
                    price=Decimal("100.00")
                ),
                account=""
            )