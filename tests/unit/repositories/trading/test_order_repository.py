"""
Unit tests for OrderRepository.
Following TDD approach - these tests will drive the implementation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, date
from pathlib import Path
import pandas as pd
import pyarrow as pa

# These imports will fail initially (Red phase)
from common.repositories.trading.orders import OrderRepository
from common.repositories.trading.base import BaseTradingRepository
from common.schemas.trading import (
    Order, OrderAction, OrderType, OrderStatus, TimeInForce, EquityContract
)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary directory for test data."""
    return str(tmp_path)


@pytest.fixture
def order_repo(temp_data_dir):
    """Order repository instance for testing."""
    return OrderRepository(temp_data_dir)


@pytest.fixture
def sample_order():
    """Sample order for testing."""
    contract = EquityContract(symbol="AAPL")
    return Order(
        order_id=12345,
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
        order_id=12346,
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
class TestOrderRepository:
    """Test OrderRepository functionality."""
    
    def test_repository_initialization(self, order_repo):
        """Test repository initializes with correct paths."""
        assert order_repo.table_name == "orders"
        assert order_repo.dataset_name == "orders"
        assert order_repo.dataset_path.exists()
    
    def test_get_schema_returns_valid_arrow_schema(self, order_repo):
        """Test schema definition is valid PyArrow schema."""
        schema = order_repo.schema
        
        # Verify it's a PyArrow schema
        assert isinstance(schema, pa.Schema)
        
        # Required fields for order data
        required_fields = [
            "order_id", "client_order_id", "symbol", "sec_type", 
            "action", "quantity", "order_type", "created_at", "date"
        ]
        
        schema_field_names = [field.name for field in schema]
        for field in required_fields:
            assert field in schema_field_names, f"Field {field} missing from schema"
    
    def test_normalize_data_converts_order_to_dict(self, order_repo, sample_order):
        """Test order normalization for storage."""
        normalized = order_repo.domain_normalize_data(sample_order)
        
        # Basic fields
        assert normalized["order_id"] == 12345
        assert normalized["client_order_id"] == "test-order-123"
        assert normalized["symbol"] == "AAPL"
        assert normalized["sec_type"] == "STK"
        assert normalized["action"] == "BUY"
        assert normalized["quantity"] == 100
        assert normalized["order_type"] == "MKT"
        
        # Dates and timestamps
        assert isinstance(normalized["created_at"], datetime)
        assert isinstance(normalized["date"], date)
        assert normalized["date"] == date(2024, 1, 15)
        
        # Default values
        assert normalized["status"] == "PendingSubmit"
        assert normalized["updated_at"] == sample_order.created_at
    
    def test_normalize_data_handles_limit_order(self, order_repo, sample_limit_order):
        """Test normalization handles limit orders with prices."""
        normalized = order_repo.domain_normalize_data(sample_limit_order)
        
        assert normalized["order_type"] == "LMT"
        assert normalized["limit_price"] == 300.0
        assert normalized["time_in_force"] == "GTC"
        assert normalized["stop_price"] is None  # Not set for limit order
    
    @patch('common.repositories.trading.orders.OrderRepository.save')
    def test_save_order_calls_parquet_save(self, mock_save, order_repo, sample_order):
        """Test save method calls underlying parquet save."""
        order_repo.domain_save(sample_order)
        
        # Verify save was called
        mock_save.assert_called_once()
        
        # Verify DataFrame argument
        call_args = mock_save.call_args
        df_arg = call_args[0][0]  # First positional argument (DataFrame)
        
        assert isinstance(df_arg, pd.DataFrame)
        assert len(df_arg) == 1  # One row
        assert df_arg.iloc[0]["client_order_id"] == "test-order-123"
        assert df_arg.iloc[0]["symbol"] == "AAPL"
    
    def test_save_batch_orders(self, order_repo, sample_order, sample_limit_order):
        """Test batch saving of multiple orders."""
        orders = [sample_order, sample_limit_order]
        
        with patch.object(order_repo, 'save') as mock_save:
            order_repo.domain_save_batch(orders)
            
            # Verify called once with DataFrame containing both orders
            mock_save.assert_called_once()
            df_arg = mock_save.call_args[0][0]
            
            assert len(df_arg) == 2
            assert set(df_arg["client_order_id"]) == {"test-order-123", "test-limit-456"}
    
    @patch('common.repositories.trading.orders.OrderRepository.load')
    def test_get_orders_with_no_filters(self, mock_load, order_repo):
        """Test retrieving all orders without filters."""
        # Mock Table response
        mock_table = create_mock_orders_table()
        mock_load.return_value = mock_table
        
        orders = order_repo.get_orders()
        
        # Verify load called with empty filters
        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs.get('filters', []) == []
    
    @patch('common.repositories.trading.orders.OrderRepository.load')
    def test_get_orders_with_account_filter(self, mock_load, order_repo):
        """Test retrieving orders filtered by account."""
        mock_table = create_mock_orders_table()
        mock_load.return_value = mock_table
        
        order_repo.get_orders(account="DU123456")
        
        # Verify filters were applied
        mock_load.assert_called_once()
        filters = mock_load.call_args[1]['filters']
        
        filter_dict = {f[0]: (f[1], f[2]) for f in filters}
        assert filter_dict["account"] == ("==", "DU123456")
    
    @patch('common.repositories.trading.orders.OrderRepository.load')
    def test_get_orders_with_status_filter(self, mock_load, order_repo):
        """Test retrieving orders filtered by status."""
        mock_table = create_mock_orders_table()
        mock_load.return_value = mock_table
        
        order_repo.get_orders(status=OrderStatus.FILLED)
        
        filters = mock_load.call_args[1]['filters']
        filter_dict = {f[0]: (f[1], f[2]) for f in filters}
        assert filter_dict["status"] == ("==", "Filled")
    
    @patch('common.repositories.trading.orders.OrderRepository.load')
    def test_get_orders_with_date_range_filter(self, mock_load, order_repo):
        """Test retrieving orders filtered by date range."""
        mock_table = create_mock_orders_table()
        mock_load.return_value = mock_table
        
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        order_repo.get_orders(date_range=(start_date, end_date))
        
        filters = mock_load.call_args[1]['filters']
        filter_dict = {f[0]: (f[1], f[2]) for f in filters}
        assert filter_dict["date"] == ("between", (start_date, end_date))
    
    @patch('common.repositories.trading.orders.OrderRepository.load')
    def test_get_orders_with_multiple_filters(self, mock_load, order_repo):
        """Test retrieving orders with multiple filters applied."""
        mock_table = create_mock_orders_table()
        mock_load.return_value = mock_table
        
        order_repo.get_orders(
            account="DU123456", 
            status=OrderStatus.SUBMITTED,
            date_range=(date(2024, 1, 1), date(2024, 1, 31))
        )
        
        filters = mock_load.call_args[1]['filters']
        filter_dict = {f[0]: (f[1], f[2]) for f in filters}
        
        assert filter_dict["account"] == ("==", "DU123456")
        assert filter_dict["status"] == ("==", "Submitted")
        assert filter_dict["date"][0] == "between"
    
    def test_update_order_status_modifies_record(self, order_repo):
        """Test order status update functionality."""
        # This test will drive the implementation of status updates
        order_id = 12345
        new_status = OrderStatus.FILLED
        filled_qty = 100
        avg_fill_price = 150.50
        
        with patch.object(order_repo, 'load') as mock_load, \
             patch.object(order_repo, 'save') as mock_save:
            
            # Mock existing order data
            mock_table = create_mock_orders_table()
            mock_load.return_value = mock_table
            
            order_repo.update_order_status(
                order_id, new_status, filled_qty, avg_fill_price
            )
            
            # Verify load was called with order_id filter
            filters = mock_load.call_args[1]['filters']
            filter_dict = {f[0]: (f[1], f[2]) for f in filters}
            assert filter_dict["order_id"] == ("==", order_id)
            
            # Verify save was called
            mock_save.assert_called_once()
    
    def test_df_row_to_order_conversion(self, order_repo):
        """Test conversion from DataFrame row back to Order object."""
        # Create sample row data
        row_data = {
            "order_id": 12345,
            "client_order_id": "test-123",
            "symbol": "AAPL",
            "sec_type": "STK",
            "action": "BUY",
            "quantity": 100,
            "order_type": "MKT",
            "limit_price": None,
            "stop_price": None,
            "time_in_force": "DAY",
            "account": "DU123456",
            "parent_order_id": None,
            "created_at": datetime(2024, 1, 15, 10, 30, 0),
            "status": "Submitted"
        }
        
        row = pd.Series(row_data)
        order = order_repo._df_row_to_order(row)
        
        assert isinstance(order, Order)
        assert order.order_id == 12345
        assert order.client_order_id == "test-123"
        assert order.contract.symbol == "AAPL"
        assert order.action == OrderAction.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.order_management
class TestOrderRepositoryIntegration:
    """Integration tests for OrderRepository with actual file operations."""
    
    def test_save_and_retrieve_order_roundtrip(self, order_repo, sample_order):
        """Test complete save and retrieve cycle."""
        # Save order
        order_repo.domain_save(sample_order)
        
        # Retrieve orders
        retrieved_orders = order_repo.get_orders()
        
        assert len(retrieved_orders) == 1
        retrieved = retrieved_orders[0]
        
        assert retrieved.order_id == sample_order.order_id
        assert retrieved.client_order_id == sample_order.client_order_id
        assert retrieved.contract.symbol == sample_order.contract.symbol
        assert retrieved.action == sample_order.action
        assert retrieved.quantity == sample_order.quantity
    
    def test_filter_orders_by_symbol(self, order_repo, sample_order):
        """Test filtering orders by symbol through contract data."""
        # Save order
        order_repo.domain_save(sample_order)
        
        # Create and save order for different symbol
        msft_contract = EquityContract(symbol="MSFT")
        msft_order = Order(
            order_id=99999,
            client_order_id="msft-test",
            contract=msft_contract,
            action=OrderAction.SELL,
            quantity=50,
            order_type=OrderType.MARKET
        )
        order_repo.domain_save(msft_order)
        
        # Test filtering (this will drive implementation of symbol filtering)
        aapl_orders = order_repo.get_orders_by_symbol("AAPL")
        msft_orders = order_repo.get_orders_by_symbol("MSFT")
        
        assert len(aapl_orders) == 1
        assert len(msft_orders) == 1
        assert aapl_orders[0].contract.symbol == "AAPL"
        assert msft_orders[0].contract.symbol == "MSFT"


# Helper functions for tests
def create_mock_orders_dataframe():
    """Create a mock DataFrame with order data."""
    data = {
        "order_id": [12345, 12346],
        "client_order_id": ["test-123", "test-456"],
        "symbol": ["AAPL", "MSFT"],
        "sec_type": ["STK", "STK"],
        "action": ["BUY", "SELL"],
        "quantity": [100, 50],
        "order_type": ["MKT", "LMT"],
        "limit_price": [None, 300.0],
        "stop_price": [None, None],
        "time_in_force": ["DAY", "GTC"],
        "account": ["DU123456", "DU123456"],
        "parent_order_id": [None, None],
        "status": ["Submitted", "Filled"],
        "created_at": [datetime(2024, 1, 15, 10, 30), datetime(2024, 1, 15, 14, 45)],
        "updated_at": [datetime(2024, 1, 15, 10, 30), datetime(2024, 1, 15, 15, 0)],
        "date": [date(2024, 1, 15), date(2024, 1, 15)]
    }
    
    mock_df = Mock(spec=pd.DataFrame)
    mock_df.empty = False
    mock_df.iterrows.return_value = [(0, pd.Series(data={k: v[0] for k, v in data.items()})),
                                    (1, pd.Series(data={k: v[1] for k, v in data.items()}))]
    mock_df.__len__.return_value = 2
    mock_df.iloc = [pd.Series(data={k: v[i] for k, v in data.items()}) for i in range(2)]
    mock_df.loc = Mock()
    
    return mock_df


def create_mock_orders_table():
    """Create a mock PyArrow Table with order data."""
    # Create actual DataFrame with sample data
    data = {
        "order_id": [12345, 12346],
        "client_order_id": ["test-123", "test-456"],
        "symbol": ["AAPL", "MSFT"],
        "sec_type": ["STK", "STK"],
        "action": ["BUY", "SELL"],
        "quantity": [100, 50],
        "order_type": ["MKT", "LMT"],
        "limit_price": [None, 300.0],
        "stop_price": [None, None],
        "time_in_force": ["DAY", "GTC"],
        "account": ["DU123456", "DU123456"],
        "parent_order_id": [None, None],
        "status": ["Submitted", "Filled"],
        "created_at": [datetime(2024, 1, 15, 10, 30), datetime(2024, 1, 15, 14, 45)],
        "updated_at": [datetime(2024, 1, 15, 10, 30), datetime(2024, 1, 15, 15, 0)],
        "date": [date(2024, 1, 15), date(2024, 1, 15)]
    }
    
    df = pd.DataFrame(data)
    
    # Create mock table with to_pandas method
    mock_table = Mock()
    mock_table.num_rows = 2
    mock_table.to_pandas.return_value = df
    
    return mock_table