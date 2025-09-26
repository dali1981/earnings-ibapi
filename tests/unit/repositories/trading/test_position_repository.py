"""
Unit tests for position repository.
Following TDD approach - these tests will drive the implementation.
"""

import pytest
import pandas as pd
from datetime import datetime, date
from decimal import Decimal
from unittest.mock import Mock, patch

# These imports will fail initially (Red phase)
from common.repositories.trading.positions import PositionRepository
from common.schemas.trading import Position, EquityContract, OptionContract
from common.constants import SecType, OptionRight


@pytest.mark.unit
@pytest.mark.trading
@pytest.mark.position_management
class TestPositionRepository:
    """Test PositionRepository functionality."""
    
    @pytest.fixture
    def position_repository(self):
        """Create position repository for testing."""
        return PositionRepository("test_data")
    
    @pytest.fixture
    def equity_position(self):
        """Create test equity position."""
        contract = EquityContract(symbol="AAPL", sec_type=SecType.STOCK)
        return Position(
            contract=contract,
            position_size=100,
            avg_cost=Decimal("150.00"),
            market_price=Decimal("155.25"),
            market_value=Decimal("15525.00"),  # 100 * 155.25
            unrealized_pnl=Decimal("525.00"),  # (155.25 - 150.00) * 100
            account="DU123456"
        )
    
    @pytest.fixture
    def option_position(self):
        """Create test option position."""
        contract = OptionContract(
            symbol="AAPL",
            sec_type=SecType.OPTION,
            strike=Decimal("150.00"),
            right=OptionRight.CALL,
            expiry=date(2024, 3, 15),
            multiplier="100"  # String as required by schema
        )
        return Position(
            contract=contract,
            position_size=2,
            avg_cost=Decimal("5.25"),
            market_price=Decimal("7.50"),
            market_value=Decimal("1500.00"),    # 2 * 7.50 * 100
            unrealized_pnl=Decimal("450.00"),   # (7.50 - 5.25) * 2 * 100
            account="DU123456"
        )
    
    def test_position_repository_initialization(self, position_repository):
        """Test PositionRepository initialization."""
        assert position_repository.table_name == "positions"
        assert str(position_repository.base_path) == "test_data"
    
    def test_domain_normalize_data_equity(self, position_repository, equity_position):
        """Test position data normalization for equity."""
        normalized = position_repository.domain_normalize_data(equity_position)
        
        expected_fields = [
            "account", "symbol", "sec_type", "position_size", "avg_cost",
            "market_price", "unrealized_pnl", "realized_pnl", "total_cost",
            "market_value", "updated_at", "date"
        ]
        
        for field in expected_fields:
            assert field in normalized
        
        # Check specific values
        assert normalized["symbol"] == "AAPL"
        assert normalized["sec_type"] == "STK"
        assert normalized["position_size"] == 100
        assert normalized["avg_cost"] == 150.00
        assert normalized["market_price"] == 155.25
        assert normalized["account"] == "DU123456"
        
        # Check calculated fields
        assert normalized["total_cost"] == 15000.00  # 100 * 150.00
        assert normalized["market_value"] == 15525.00  # 100 * 155.25
        assert normalized["unrealized_pnl"] == 525.00  # (155.25 - 150.00) * 100
    
    def test_domain_normalize_data_option(self, position_repository, option_position):
        """Test position data normalization for option."""
        normalized = position_repository.domain_normalize_data(option_position)
        
        # Check option-specific fields
        assert normalized["symbol"] == "AAPL"
        assert normalized["sec_type"] == "OPT"
        assert normalized["strike"] == 150.00
        assert normalized["right"] == "C"
        assert normalized["expiry"] == "2024-03-15"
        assert normalized["multiplier"] == 100
        
        # Check calculated values for options
        assert normalized["total_cost"] == 1050.00  # 2 * 5.25 * 100
        assert normalized["market_value"] == 1500.00  # 2 * 7.50 * 100
        assert normalized["unrealized_pnl"] == 450.00  # (7.50 - 5.25) * 2 * 100
    
    def test_domain_save_equity_position(self, position_repository, equity_position):
        """Test saving equity position."""
        with patch.object(position_repository, 'save') as mock_save:
            position_repository.domain_save(equity_position)
            
            # Verify save was called with DataFrame
            mock_save.assert_called_once()
            call_args = mock_save.call_args[0]
            assert len(call_args) == 1
            
            df = call_args[0]
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            
            # Check DataFrame has correct data
            row = df.iloc[0]
            assert row["symbol"] == "AAPL"
            assert row["sec_type"] == "STK"
            assert row["position_size"] == 100
    
    def test_domain_save_option_position(self, position_repository, option_position):
        """Test saving option position."""
        with patch.object(position_repository, 'save') as mock_save:
            position_repository.domain_save(option_position)
            
            mock_save.assert_called_once()
            df = mock_save.call_args[0][0]
            row = df.iloc[0]
            
            assert row["symbol"] == "AAPL"
            assert row["sec_type"] == "OPT"
            assert row["strike"] == 150.00
            assert row["right"] == "C"
    
    def test_get_position_by_contract(self, position_repository, equity_position):
        """Test retrieving position by contract."""
        mock_df = pd.DataFrame([{
            "account": "DU123456",
            "symbol": "AAPL",
            "sec_type": "STK",
            "position_size": 100,
            "avg_cost": 150.00,
            "market_price": 155.25,
            "unrealized_pnl": 525.00,
            "realized_pnl": 0.00,
            "total_cost": 15000.00,
            "market_value": 15525.00,
            "updated_at": datetime.utcnow(),
            "date": date.today()
        }])
        
        with patch.object(position_repository, 'read_data') as mock_read:
            mock_read.return_value = mock_df
            
            result = position_repository.get_position_by_contract(
                equity_position.contract, 
                account="DU123456"
            )
            
            assert result is not None
            assert isinstance(result, Position)
            assert result.contract.symbol == "AAPL"
            assert result.position_size == 100
            assert result.avg_cost == Decimal("150.00")
    
    def test_get_positions_by_account(self, position_repository):
        """Test retrieving all positions by account."""
        mock_df = pd.DataFrame([
            {
                "account": "DU123456",
                "symbol": "AAPL",
                "sec_type": "STK",
                "position_size": 100,
                "avg_cost": 150.00,
                "market_price": 155.25,
                "unrealized_pnl": 525.00,
                "realized_pnl": 0.00,
                "total_cost": 15000.00,
                "market_value": 15525.00,
                "updated_at": datetime.utcnow(),
                "date": date.today()
            },
            {
                "account": "DU123456",
                "symbol": "MSFT",
                "sec_type": "STK",
                "position_size": 50,
                "avg_cost": 300.00,
                "market_price": 310.00,
                "unrealized_pnl": 500.00,
                "realized_pnl": 100.00,
                "total_cost": 15000.00,
                "market_value": 15500.00,
                "updated_at": datetime.utcnow(),
                "date": date.today()
            }
        ])
        
        with patch.object(position_repository, 'read_data') as mock_read:
            mock_read.return_value = mock_df
            
            positions = position_repository.get_positions_by_account("DU123456")
            
            assert len(positions) == 2
            assert all(isinstance(p, Position) for p in positions)
            
            symbols = [p.contract.symbol for p in positions]
            assert "AAPL" in symbols
            assert "MSFT" in symbols
    
    def test_update_position_prices(self, position_repository, equity_position):
        """Test updating position with new market prices."""
        new_price = Decimal("160.00")
        
        # Mock existing position data
        mock_existing = pd.DataFrame([{
            "account": "DU123456",
            "symbol": "AAPL",
            "sec_type": "STK",
            "position_size": 100,
            "avg_cost": 150.00,
            "market_price": 155.25,
            "unrealized_pnl": 525.00,
            "realized_pnl": 0.00,
            "total_cost": 15000.00,
            "market_value": 15525.00,
            "updated_at": datetime.utcnow(),
            "date": date.today()
        }])
        
        with patch.object(position_repository, 'read_data') as mock_read, \
             patch.object(position_repository, 'save') as mock_save:
            
            mock_read.return_value = mock_existing
            
            position_repository.update_position_prices(
                equity_position.contract,
                new_price,
                account="DU123456"
            )
            
            # Verify save was called with updated data
            mock_save.assert_called_once()
            updated_df = mock_save.call_args[0][0]
            
            row = updated_df.iloc[0]
            assert row["market_price"] == 160.00
            assert row["market_value"] == 16000.00  # 100 * 160.00
            assert row["unrealized_pnl"] == 1000.00  # (160.00 - 150.00) * 100
    
    def test_get_portfolio_summary(self, position_repository):
        """Test getting portfolio summary statistics."""
        mock_df = pd.DataFrame([
            {
                "account": "DU123456",
                "symbol": "AAPL",
                "sec_type": "STK",
                "position_size": 100,
                "avg_cost": 150.00,
                "market_price": 155.25,
                "unrealized_pnl": 525.00,
                "realized_pnl": 100.00,
                "total_cost": 15000.00,
                "market_value": 15525.00,
                "updated_at": datetime.utcnow(),
                "date": date.today()
            },
            {
                "account": "DU123456",
                "symbol": "MSFT",
                "sec_type": "STK",
                "position_size": 50,
                "avg_cost": 300.00,
                "market_price": 310.00,
                "unrealized_pnl": 500.00,
                "realized_pnl": 200.00,
                "total_cost": 15000.00,
                "market_value": 15500.00,
                "updated_at": datetime.utcnow(),
                "date": date.today()
            }
        ])
        
        with patch.object(position_repository, 'read_data') as mock_read:
            mock_read.return_value = mock_df
            
            summary = position_repository.get_portfolio_summary("DU123456")
            
            expected_summary = {
                "total_positions": 2,
                "total_cost": 30000.00,
                "total_market_value": 31025.00,
                "total_unrealized_pnl": 1025.00,
                "total_realized_pnl": 300.00,
                "total_pnl": 1325.00
            }
            
            assert summary == expected_summary
    
    def test_close_position(self, position_repository, equity_position):
        """Test closing a position (setting position_size to 0)."""
        realized_pnl = Decimal("525.00")
        
        with patch.object(position_repository, 'save') as mock_save:
            position_repository.close_position(
                equity_position.contract,
                realized_pnl,
                account="DU123456"
            )
            
            mock_save.assert_called_once()
            closed_df = mock_save.call_args[0][0]
            
            row = closed_df.iloc[0]
            assert row["position_size"] == 0
            assert row["realized_pnl"] == 525.00
            assert row["market_value"] == 0.00
            assert row["unrealized_pnl"] == 0.00
    
    def test_position_schema_creation(self, position_repository):
        """Test position repository schema creation."""
        schema = position_repository._create_schema()
        
        expected_fields = [
            "account", "symbol", "sec_type", "position_size", "avg_cost",
            "market_price", "unrealized_pnl", "realized_pnl", "total_cost",
            "market_value", "strike", "right", "expiry", "multiplier",
            "updated_at", "date"
        ]
        
        schema_field_names = [field.name for field in schema]
        
        for field_name in expected_fields:
            assert field_name in schema_field_names
    
    def test_get_partition_columns(self, position_repository):
        """Test partition columns for position repository."""
        partition_cols = position_repository._get_partition_columns()
        
        assert "date" in partition_cols
        assert "account" in partition_cols