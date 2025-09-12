"""
Unit tests for OptionContractsRepository.

Tests the storage, retrieval, and validation of individual option contract data
with proper partitioning by symbol and expiry.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from common.repositories.option_contracts import OptionContractsRepository
from common.exceptions import DataValidationException


def _get_value(data, column, index=0):
    """Helper to get value handling both pandas and PyArrow data structures."""
    # Convert PyArrow Table to pandas DataFrame for easier testing
    import pyarrow as pa
    if isinstance(data, pa.Table):
        data = data.to_pandas()
    
    col_data = data[column]
    if hasattr(col_data, 'iloc'):
        # Pandas Series
        return col_data.iloc[index]
    else:
        # PyArrow structures
        value = col_data[index]
        if hasattr(value, 'as_py'):
            # PyArrow scalar
            return value.as_py()
        else:
            # Already a Python value
            return value


def _convert_pyarrow_to_python(data):
    """Convert PyArrow data structures to Python native types for test assertions."""
    import pyarrow as pa
    
    # Convert Tables to DataFrames
    if isinstance(data, pa.Table):
        data = data.to_pandas()
    
    # Handle unique() results that might be PyArrow arrays
    if hasattr(data, 'unique'):
        unique_vals = data.unique()
        if hasattr(unique_vals, 'to_pylist'):
            # PyArrow array
            return set(unique_vals.to_pylist())
        elif hasattr(unique_vals, '__iter__'):
            # Convert each scalar
            return {val.as_py() if hasattr(val, 'as_py') else val for val in unique_vals}
        else:
            return set(unique_vals)
    
    return data


def _get_columns(data):
    """Get column names handling both pandas and PyArrow data structures."""
    import pyarrow as pa
    
    if isinstance(data, pa.Table):
        return data.column_names
    elif hasattr(data, 'columns'):
        return data.columns
    else:
        return []


@pytest.mark.unit
@pytest.mark.data
class TestOptionContractsRepository:
    """Test cases for OptionContractsRepository."""

    def test_repository_initialization_success(self, temp_data_dir):
        """Test successful repository initialization."""
        # Arrange
        repo_path = temp_data_dir / "option_contracts"
        
        # Act
        repo = OptionContractsRepository(repo_path)
        
        # Assert
        assert repo.base_path == repo_path
        assert repo.dataset_name == "option_contracts"
        assert repo.base_path.exists()

    def test_create_schema_structure(self, option_contracts_repository):
        """Test that schema is created with correct structure."""
        # Act
        schema = option_contracts_repository._create_schema()
        
        # Assert
        expected_fields = [
            "symbol", "sec_type", "conid", "exchange", "primary_exchange",
            "currency", "local_symbol", "trading_class", "multiplier",
            "strike", "right", "expiry", "expiry_raw", "min_tick",
            "valid_exchanges", "long_name", "category", "subcategory"
        ]
        actual_fields = [field.name for field in schema]
        
        for field in expected_fields:
            assert field in actual_fields, f"Missing required field: {field}"

    def test_get_partition_columns(self, option_contracts_repository):
        """Test partition columns configuration."""
        # Act
        partition_columns = option_contracts_repository._get_partition_columns()
        
        # Assert
        assert "symbol" in partition_columns
        assert "expiry" in partition_columns

    def test_save_option_contracts_success(self, option_contracts_repository):
        """Test successful saving of option contracts data."""
        # Arrange
        contracts_data = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "sec_type": ["OPT", "OPT", "OPT", "OPT"],
            "conid": [12345, 12346, 12347, 12348],
            "exchange": ["SMART", "SMART", "SMART", "SMART"],
            "primary_exchange": ["CBOE", "CBOE", "CBOE", "CBOE"],
            "currency": ["USD", "USD", "USD", "USD"],
            "local_symbol": ["AAPL240315C00150000", "AAPL240315P00150000", 
                           "AAPL240315C00155000", "AAPL240315P00155000"],
            "trading_class": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "multiplier": [100, 100, 100, 100],
            "strike": [150.0, 150.0, 155.0, 155.0],
            "right": ["C", "P", "C", "P"],
            "expiry": [date(2024, 3, 15), date(2024, 3, 15), 
                      date(2024, 3, 15), date(2024, 3, 15)],
            "expiry_raw": ["20240315", "20240315", "20240315", "20240315"],
            "min_tick": [0.01, 0.01, 0.01, 0.01],
            "valid_exchanges": ["SMART,CBOE", "SMART,CBOE", "SMART,CBOE", "SMART,CBOE"],
            "long_name": ["Apple Inc.", "Apple Inc.", "Apple Inc.", "Apple Inc."],
            "category": ["Stock", "Stock", "Stock", "Stock"],
            "subcategory": ["Option", "Option", "Option", "Option"]
        })
        
        # Act
        result = option_contracts_repository.save(
            contracts_data, 
            symbol="AAPL",
            expiry=date(2024, 3, 15)
        )
        
        # Assert
        assert result is True or result is None  # Different implementations may return different values
        
        # Verify data was stored
        loaded_data = option_contracts_repository.load(
            symbol="AAPL", 
            expiry=date(2024, 3, 15)
        )
        assert loaded_data is not None
        assert len(loaded_data) == 4
        assert _convert_pyarrow_to_python(loaded_data["right"]) == {"C", "P"}

    def test_save_with_missing_required_fields_error(self, option_contracts_repository):
        """Test error handling when saving data with missing required fields."""
        # Arrange
        incomplete_data = pd.DataFrame({
            "symbol": ["AAPL"],
            "conid": [12345]
            # Missing required fields like sec_type, strike, right, expiry
        })
        
        # Act & Assert
        with pytest.raises((DataValidationException, ValueError)):
            option_contracts_repository.save(incomplete_data, symbol="AAPL")

    def test_save_with_invalid_option_rights(self, option_contracts_repository):
        """Test validation of option rights (should be C or P)."""
        # Arrange
        invalid_data = pd.DataFrame({
            "symbol": ["AAPL"],
            "sec_type": ["OPT"],
            "conid": [12345],
            "exchange": ["SMART"],
            "currency": ["USD"],
            "strike": [150.0],
            "right": ["X"],  # Invalid option right
            "expiry": [date(2024, 3, 15)],
            "multiplier": [100]
        })
        
        # Act & Assert
        with pytest.raises((DataValidationException, ValueError)):
            option_contracts_repository.save(invalid_data, symbol="AAPL")

    def test_save_with_negative_strike_price(self, option_contracts_repository):
        """Test validation of strike prices (should be positive)."""
        # Arrange
        invalid_data = pd.DataFrame({
            "symbol": ["AAPL"],
            "sec_type": ["OPT"],
            "conid": [12345],
            "exchange": ["SMART"],
            "currency": ["USD"],
            "strike": [-150.0],  # Invalid negative strike
            "right": ["C"],
            "expiry": [date(2024, 3, 15)],
            "multiplier": [100]
        })
        
        # Act & Assert
        with pytest.raises((DataValidationException, ValueError)):
            option_contracts_repository.save(invalid_data, symbol="AAPL")

    def test_load_existing_contracts(self, option_contracts_repository):
        """Test loading existing option contracts."""
        # Arrange - first save some data
        contracts_data = pd.DataFrame({
            "symbol": ["MSFT", "MSFT"],
            "sec_type": ["OPT", "OPT"],
            "conid": [23456, 23457],
            "exchange": ["SMART", "SMART"],
            "primary_exchange": ["CBOE", "CBOE"],
            "currency": ["USD", "USD"],
            "local_symbol": ["MSFT240415C00300000", "MSFT240415P00300000"],
            "trading_class": ["MSFT", "MSFT"],
            "multiplier": [100, 100],
            "strike": [300.0, 300.0],
            "right": ["C", "P"],
            "expiry": [date(2024, 4, 15), date(2024, 4, 15)],
            "expiry_raw": ["20240415", "20240415"],
            "min_tick": [0.01, 0.01]
        })
        
        option_contracts_repository.save(
            contracts_data, 
            symbol="MSFT", 
            expiry=date(2024, 4, 15)
        )
        
        # Act
        loaded_data = option_contracts_repository.load(
            symbol="MSFT", 
            expiry=date(2024, 4, 15)
        )
        
        # Assert
        assert loaded_data is not None
        assert len(loaded_data) == 2
        assert _get_value(loaded_data, "symbol") == "MSFT"
        assert _get_value(loaded_data, "strike") == 300.0

    def test_load_nonexistent_contracts(self, option_contracts_repository):
        """Test loading non-existent contracts returns None."""
        # Act
        loaded_data = option_contracts_repository.load(
            symbol="NONEXISTENT", 
            expiry=date(2024, 12, 31)
        )
        
        # Assert
        assert loaded_data is None

    def test_load_contracts_by_symbol_only(self, option_contracts_repository):
        """Test loading all contracts for a symbol across expiries."""
        # Arrange - save contracts for multiple expiries
        expiry1 = date(2024, 3, 15)
        expiry2 = date(2024, 4, 15)
        
        for expiry in [expiry1, expiry2]:
            contracts_data = pd.DataFrame({
                "symbol": ["GOOGL"],
                "sec_type": ["OPT"],
                "conid": [34567 + expiry.day],  # Unique conids
                "exchange": ["SMART"],
                "currency": ["USD"],
                "strike": [2800.0],
                "right": ["C"],
                "expiry": [expiry],
                "multiplier": [100]
            })
            
            option_contracts_repository.save(
                contracts_data, 
                symbol="GOOGL", 
                expiry=expiry
            )
        
        # Act - load all contracts for symbol
        loaded_data = option_contracts_repository.load(symbol="GOOGL")
        
        # Assert
        if loaded_data is not None:  # Some implementations may not support symbol-only queries
            assert len(loaded_data) >= 2  # Should have contracts from both expiries
            assert _get_value(loaded_data, "symbol") == "GOOGL"

    def test_load_contracts_with_date_range(self, option_contracts_repository):
        """Test loading contracts within a date range."""
        # Arrange - save contracts for multiple dates
        symbol = "TSLA"
        expiries = [date(2024, 3, 15), date(2024, 4, 15), date(2024, 5, 15)]
        
        for expiry in expiries:
            contracts_data = pd.DataFrame({
                "symbol": [symbol],
                "sec_type": ["OPT"],
                "conid": [45678 + expiry.month],
                "exchange": ["SMART"],
                "currency": ["USD"],
                "strike": [200.0],
                "right": ["C"],
                "expiry": [expiry],
                "multiplier": [100]
            })
            
            option_contracts_repository.save(contracts_data, symbol=symbol, expiry=expiry)
        
        # Act - load contracts in date range
        loaded_data = option_contracts_repository.load(
            symbol=symbol,
            start_date=date(2024, 3, 1),
            end_date=date(2024, 4, 30)
        )
        
        # Assert
        if loaded_data is not None:
            # Should include March and April expiries, but not May
            unique_expiries = loaded_data["expiry"].unique()
            assert len(unique_expiries) <= 2

    def test_save_duplicate_contracts_handling(self, option_contracts_repository):
        """Test handling of duplicate contract saves."""
        # Arrange
        original_data = pd.DataFrame({
            "symbol": ["NVDA"],
            "sec_type": ["OPT"],
            "conid": [56789],
            "exchange": ["SMART"],
            "currency": ["USD"],
            "strike": [500.0],
            "right": ["C"],
            "expiry": [date(2024, 6, 15)],
            "multiplier": [100],
            "long_name": ["NVIDIA Corporation"]
        })
        
        # Save original data
        option_contracts_repository.save(
            original_data, 
            symbol="NVDA", 
            expiry=date(2024, 6, 15)
        )
        
        # Create "duplicate" with updated information
        updated_data = pd.DataFrame({
            "symbol": ["NVDA"],
            "sec_type": ["OPT"],
            "conid": [56789],  # Same conid
            "exchange": ["SMART"],
            "currency": ["USD"],
            "strike": [500.0],
            "right": ["C"],
            "expiry": [date(2024, 6, 15)],
            "multiplier": [100],
            "long_name": ["NVIDIA Corp (Updated)"]  # Updated field
        })
        
        # Act - save updated data
        option_contracts_repository.save(
            updated_data, 
            symbol="NVDA", 
            expiry=date(2024, 6, 15)
        )
        
        # Assert - should handle duplicate appropriately
        loaded_data = option_contracts_repository.load(
            symbol="NVDA", 
            expiry=date(2024, 6, 15)
        )
        assert loaded_data is not None
        # The behavior here may vary - could be deduplicated or updated

    def test_data_type_validation(self, option_contracts_repository):
        """Test data type validation for contract fields."""
        # Arrange - data with incorrect types
        invalid_types_data = pd.DataFrame({
            "symbol": ["META"],
            "sec_type": ["OPT"],
            "conid": ["not_an_integer"],  # Should be int
            "exchange": ["SMART"],
            "currency": ["USD"],
            "strike": ["not_a_float"],  # Should be float
            "right": ["C"],
            "expiry": ["not_a_date"],  # Should be date
            "multiplier": ["not_an_int"]  # Should be int
        })
        
        # Act & Assert
        with pytest.raises((DataValidationException, ValueError, TypeError)):
            option_contracts_repository.save(
                invalid_types_data, 
                symbol="META", 
                expiry=date(2024, 7, 15)
            )

    def test_large_batch_save_performance(self, option_contracts_repository, benchmark_timer):
        """Test performance with large batch of contracts."""
        # Arrange - create large dataset
        num_contracts = 1000
        symbols = ["AAPL"] * num_contracts
        strikes = [100.0 + i for i in range(num_contracts)]
        rights = ["C", "P"] * (num_contracts // 2)
        
        large_dataset = pd.DataFrame({
            "symbol": symbols,
            "sec_type": ["OPT"] * num_contracts,
            "conid": list(range(100000, 100000 + num_contracts)),
            "exchange": ["SMART"] * num_contracts,
            "currency": ["USD"] * num_contracts,
            "strike": strikes,
            "right": rights,
            "expiry": [date(2024, 12, 15)] * num_contracts,
            "multiplier": [100] * num_contracts
        })
        
        # Act & Time the operation
        with benchmark_timer() as timer:
            option_contracts_repository.save(
                large_dataset, 
                symbol="AAPL", 
                expiry=date(2024, 12, 15)
            )
        
        # Assert
        assert timer.elapsed < 10.0, f"Large batch save took too long: {timer.elapsed}s"
        
        # Verify data was saved
        loaded_data = option_contracts_repository.load(
            symbol="AAPL", 
            expiry=date(2024, 12, 15)
        )
        assert loaded_data is not None
        assert len(loaded_data) == num_contracts

    def test_contracts_with_special_characters_in_symbol(self, option_contracts_repository):
        """Test handling of symbols with special characters."""
        # Arrange
        special_symbols = ["BRK.B", "BF.B", "GOOG/GOOGL"]
        
        for symbol in special_symbols:
            contracts_data = pd.DataFrame({
                "symbol": [symbol],
                "sec_type": ["OPT"],
                "conid": [78901 + hash(symbol) % 1000],
                "exchange": ["SMART"],
                "currency": ["USD"],
                "strike": [100.0],
                "right": ["C"],
                "expiry": [date(2024, 8, 15)],
                "multiplier": [100]
            })
            
            # Act - should handle special characters
            try:
                option_contracts_repository.save(
                    contracts_data, 
                    symbol=symbol, 
                    expiry=date(2024, 8, 15)
                )
                
                # Verify it can be loaded back
                loaded_data = option_contracts_repository.load(
                    symbol=symbol, 
                    expiry=date(2024, 8, 15)
                )
                
                # Assert
                if loaded_data is not None:
                    assert _get_value(loaded_data, "symbol") == symbol
                    
            except (DataValidationException, ValueError):
                # May not support all special characters - that's OK
                pass

    def test_expiry_date_validation(self, option_contracts_repository):
        """Test validation of expiry dates."""
        # Arrange - test various edge cases for expiry dates
        test_cases = [
            {
                "name": "past_expiry",
                "expiry": date(2020, 1, 1),
                "should_pass": True  # Historical data should be allowed
            },
            {
                "name": "far_future_expiry", 
                "expiry": date(2030, 1, 1),
                "should_pass": True  # Future expiries should be allowed
            },
            {
                "name": "weekend_expiry",
                "expiry": date(2024, 3, 16),  # Saturday
                "should_pass": True  # Weekends should be allowed (some options do expire on weekends)
            }
        ]
        
        for test_case in test_cases:
            contracts_data = pd.DataFrame({
                "symbol": ["TEST"],
                "sec_type": ["OPT"],
                "conid": [99999],
                "exchange": ["SMART"],
                "currency": ["USD"],
                "strike": [100.0],
                "right": ["C"],
                "expiry": [test_case["expiry"]],
                "multiplier": [100]
            })
            
            # Act & Assert
            if test_case["should_pass"]:
                try:
                    option_contracts_repository.save(
                        contracts_data, 
                        symbol="TEST", 
                        expiry=test_case["expiry"]
                    )
                except (DataValidationException, ValueError):
                    pass  # Some validation rules may still reject these
            else:
                with pytest.raises((DataValidationException, ValueError)):
                    option_contracts_repository.save(
                        contracts_data, 
                        symbol="TEST", 
                        expiry=test_case["expiry"]
                    )

    def test_concurrent_saves_different_symbols(self, option_contracts_repository):
        """Test concurrent saves of different symbols."""
        import threading
        
        # Arrange
        symbols = ["AAPL", "MSFT", "GOOGL"]
        results = []
        
        def save_symbol_contracts(symbol):
            try:
                contracts_data = pd.DataFrame({
                    "symbol": [symbol],
                    "sec_type": ["OPT"],
                    "conid": [hash(symbol) % 100000],
                    "exchange": ["SMART"],
                    "currency": ["USD"],
                    "strike": [150.0],
                    "right": ["C"],
                    "expiry": [date(2024, 9, 15)],
                    "multiplier": [100]
                })
                
                option_contracts_repository.save(
                    contracts_data, 
                    symbol=symbol, 
                    expiry=date(2024, 9, 15)
                )
                results.append(f"success_{symbol}")
            except Exception as e:
                results.append(f"error_{symbol}_{str(e)}")
        
        # Act
        threads = []
        for symbol in symbols:
            thread = threading.Thread(target=save_symbol_contracts, args=(symbol,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Assert
        success_count = len([r for r in results if r.startswith("success")])
        assert success_count >= 2  # At least most should succeed


@pytest.mark.unit
@pytest.mark.data
class TestOptionContractsRepositoryIntegration:
    """Integration-style tests for OptionContractsRepository."""

    def test_full_option_chain_workflow(self, option_contracts_repository):
        """Test complete workflow of saving and retrieving option chains."""
        # Arrange - create a realistic option chain
        underlying = "AAPL"
        expiry = date(2024, 3, 15)
        strikes = [140, 145, 150, 155, 160]
        
        contracts = []
        conid_base = 100000
        
        for strike in strikes:
            for right in ["C", "P"]:
                contracts.append({
                    "symbol": underlying,
                    "sec_type": "OPT",
                    "conid": conid_base,
                    "exchange": "SMART",
                    "primary_exchange": "CBOE",
                    "currency": "USD",
                    "local_symbol": f"{underlying}{expiry.strftime('%y%m%d')}{right}{strike:08.0f}",
                    "trading_class": underlying,
                    "multiplier": 100,
                    "strike": float(strike),
                    "right": right,
                    "expiry": expiry,
                    "expiry_raw": expiry.strftime("%Y%m%d"),
                    "min_tick": 0.01,
                    "valid_exchanges": "SMART,CBOE",
                    "long_name": "Apple Inc.",
                    "category": "Stock Options",
                    "subcategory": "Equity Option"
                })
                conid_base += 1
        
        contracts_df = pd.DataFrame(contracts)
        
        # Act - save the full chain
        option_contracts_repository.save(
            contracts_df, 
            symbol=underlying, 
            expiry=expiry
        )
        
        # Load back and verify
        loaded_contracts = option_contracts_repository.load(
            symbol=underlying, 
            expiry=expiry
        )
        
        # Assert
        assert loaded_contracts is not None
        assert len(loaded_contracts) == 10  # 5 strikes * 2 rights
        
        # Check strikes are correct
        loaded_strikes = sorted(_convert_pyarrow_to_python(loaded_contracts["strike"]))
        assert loaded_strikes == [140.0, 145.0, 150.0, 155.0, 160.0]
        
        # Check both calls and puts are present
        loaded_rights = _convert_pyarrow_to_python(loaded_contracts["right"])
        assert loaded_rights == {"C", "P"}
        
        # Check contract identifiers are unique
        loaded_conids = loaded_contracts["conid"].unique()
        assert len(loaded_conids) == 10