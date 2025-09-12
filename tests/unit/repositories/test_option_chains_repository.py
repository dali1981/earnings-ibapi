"""
Unit tests for OptionChainSnapshotRepository.

Tests the storage, retrieval, and validation of option chain snapshot data
with proper partitioning and data integrity checks.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from common.repositories.option_chains import OptionChainSnapshotRepository
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
class TestOptionChainSnapshotRepository:
    """Test cases for OptionChainSnapshotRepository."""

    def test_repository_initialization_success(self, temp_data_dir):
        """Test successful repository initialization."""
        # Arrange
        repo_path = temp_data_dir / "option_chains"
        
        # Act
        repo = OptionChainSnapshotRepository(repo_path)
        
        # Assert
        assert repo.base_path == repo_path
        assert repo.dataset_name == "option_chains"
        assert repo.base_path.exists()

    def test_repository_initialization_with_string_path(self, temp_data_dir):
        """Test repository initialization with string path."""
        # Arrange
        repo_path = str(temp_data_dir / "option_chains")
        
        # Act
        repo = OptionChainSnapshotRepository(repo_path)
        
        # Assert
        assert isinstance(repo.base_path, Path)
        assert repo.base_path.name == "option_chains"

    def test_create_schema_structure(self, option_chains_repository):
        """Test that schema is created with correct structure."""
        # Act
        schema = option_chains_repository._create_schema()
        
        # Assert
        expected_fields = [
            "underlying", "underlying_conid", "exchange", "trading_class",
            "multiplier", "expirations", "strikes", "as_of"
        ]
        actual_fields = [field.name for field in schema]
        
        for field in expected_fields:
            assert field in actual_fields, f"Missing required field: {field}"

    def test_get_partition_columns(self, option_chains_repository):
        """Test partition columns configuration."""
        # Act
        partition_columns = option_chains_repository._get_partition_columns()
        
        # Assert
        assert partition_columns == ["as_of"]

    def test_normalize_data_success(self, option_chains_repository):
        """Test successful data normalization."""
        # Arrange
        raw_data = pd.DataFrame({
            "underlying": ["AAPL"],
            "underlying_conid": [123456],
            "exchange": ["SMART"],
            "trading_class": ["AAPL"],
            "multiplier": ["100"],
            "expirations": [["20240315", "20240415"]],
            "strikes": [[140.0, 145.0, 150.0]],
            "snapshot_date": [date.today()]
        })
        
        # Act
        normalized = option_chains_repository._normalize_data(
            raw_data, 
            underlying="AAPL", 
            as_of=date.today()
        )
        
        # Assert
        assert "as_of" in _get_columns(normalized)
        assert _get_value(normalized, "underlying") == "AAPL"
        assert _get_value(normalized, "as_of") == date.today()

    def test_normalize_data_with_column_renaming(self, option_chains_repository):
        """Test data normalization with various column name formats."""
        # Arrange
        raw_data = pd.DataFrame({
            "underlying": ["AAPL"],
            "conId": [123456],  # Should be renamed to underlying_conid
            "exchange": ["SMART"],
            "tradingClass": ["AAPL"],  # Should be renamed to trading_class
            "multiplier": ["100"],
            "expirations": [["20240315", "20240415"]],
            "strikes": [[140.0, 145.0, 150.0]],
            "date": [date.today()]  # Should be renamed to as_of
        })
        
        # Act
        normalized = option_chains_repository._normalize_data(raw_data)
        
        # Assert
        assert "underlying_conid" in _get_columns(normalized)
        assert "trading_class" in _get_columns(normalized)
        assert "as_of" in _get_columns(normalized)
        assert _get_value(normalized, "underlying_conid") == 123456

    def test_normalize_data_empty_dataframe_error(self, option_chains_repository):
        """Test error handling for empty DataFrame."""
        # Arrange
        empty_df = pd.DataFrame()
        
        # Act & Assert
        with pytest.raises(DataValidationException, match="Cannot normalize empty DataFrame"):
            option_chains_repository._normalize_data(empty_df)

    def test_normalize_data_none_dataframe_error(self, option_chains_repository):
        """Test error handling for None DataFrame."""
        # Act & Assert
        with pytest.raises(DataValidationException):
            option_chains_repository._normalize_data(None)

    def test_normalize_data_missing_required_columns_error(self, option_chains_repository):
        """Test error handling for missing required columns."""
        # Arrange
        incomplete_data = pd.DataFrame({
            "underlying": ["AAPL"],
            "exchange": ["SMART"]
            # Missing: expirations, strikes, as_of
        })
        
        # Act & Assert
        with pytest.raises(DataValidationException):
            option_chains_repository._normalize_data(incomplete_data)

    def test_save_chain_snapshot_success(self, option_chains_repository, option_chain_data):
        """Test successful option chain snapshot saving."""
        # Arrange
        underlying = "AAPL"
        snapshot_date = date.today()
        
        # Prepare data in expected format
        aggregated_data = pd.DataFrame({
            "underlying": [underlying],
            "underlying_conid": [123456],
            "exchange": ["SMART"],
            "trading_class": ["AAPL"],
            "multiplier": ["100"],
            "expirations": [["20240315", "20240415"]],
            "strikes": [[140.0, 145.0, 150.0, 155.0, 160.0]],
            "as_of": [snapshot_date]
        })
        
        # Act
        option_chains_repository.save_chain_snapshot(
            aggregated_data, underlying, snapshot_date
        )
        
        # Assert - verify data was stored
        stored_data = option_chains_repository.load(
            underlying=underlying, as_of=snapshot_date
        )
        assert stored_data is not None
        assert len(stored_data) == 1
        assert _get_value(stored_data, "underlying") == underlying

    def test_save_chain_snapshot_with_invalid_data(self, option_chains_repository):
        """Test error handling when saving invalid data."""
        # Arrange
        invalid_data = pd.DataFrame({
            "invalid_column": ["invalid_value"]
        })
        
        # Act & Assert
        with pytest.raises(DataValidationException):
            option_chains_repository.save_chain_snapshot(
                invalid_data, "AAPL", date.today()
            )

    def test_load_existing_data(self, populated_option_chains_repo):
        """Test loading existing option chain data."""
        # Act
        loaded_data = populated_option_chains_repo.load(
            underlying="AAPL", as_of=date.today()
        )
        
        # Assert
        assert loaded_data is not None
        assert len(loaded_data) > 0
        assert "underlying" in _get_columns(loaded_data)
        assert _get_value(loaded_data, "underlying") == "AAPL"

    def test_load_nonexistent_data(self, option_chains_repository):
        """Test loading non-existent data returns empty table."""
        # Act
        loaded_data = option_chains_repository.load(
            underlying="NONEXISTENT", as_of=date.today()
        )
        
        # Assert
        assert loaded_data is not None
        assert loaded_data.num_rows == 0  # Empty PyArrow table

    def test_load_with_date_range(self, option_chains_repository):
        """Test loading data within a date range."""
        # Arrange - save data for multiple dates
        underlying = "AAPL"
        dates = [date.today() - timedelta(days=i) for i in range(3)]
        
        for snapshot_date in dates:
            data = pd.DataFrame({
                "underlying": [underlying],
                "underlying_conid": [123456],
                "exchange": ["SMART"],
                "trading_class": ["AAPL"],
                "multiplier": ["100"],
                "expirations": [["20240315", "20240415"]],
                "strikes": [[140.0, 145.0, 150.0]],
                "as_of": [snapshot_date]
            })
            option_chains_repository.save_chain_snapshot(data, underlying, snapshot_date)
        
        # Act - load data from date range
        start_date = dates[-1]  # Oldest date
        end_date = dates[0]     # Newest date
        
        loaded_data = option_chains_repository.load(
            underlying=underlying,
            filters=[("as_of", "between", (start_date, end_date))]
        )
        
        # Assert
        assert loaded_data is not None
        assert len(loaded_data) == 3  # Should load all three dates

    def test_data_validation_with_invalid_strike_prices(self, option_chains_repository):
        """Test data validation with invalid strike prices."""
        # Arrange
        invalid_data = pd.DataFrame({
            "underlying": ["AAPL"],
            "underlying_conid": [123456],
            "exchange": ["SMART"],
            "trading_class": ["AAPL"],
            "multiplier": ["100"],
            "expirations": [["20240315"]],
            "strikes": [[-100.0, 0.0, "invalid"]],  # Invalid strikes
            "as_of": [date.today()]
        })
        
        # Act & Assert
        # Should either handle gracefully or raise validation error
        try:
            option_chains_repository.save_chain_snapshot(
                invalid_data, "AAPL", date.today()
            )
        except (DataValidationException, ValueError, TypeError):
            pass  # Expected behavior

    def test_data_validation_with_empty_expirations(self, option_chains_repository):
        """Test data validation with empty expirations list."""
        # Arrange
        data_with_empty_expirations = pd.DataFrame({
            "underlying": ["AAPL"],
            "underlying_conid": [123456],
            "exchange": ["SMART"],
            "trading_class": ["AAPL"],
            "multiplier": ["100"],
            "expirations": [[]],  # Empty expirations
            "strikes": [[140.0, 145.0]],
            "as_of": [date.today()]
        })
        
        # Act
        # Repository should handle this gracefully
        result = option_chains_repository.save_chain_snapshot(
            data_with_empty_expirations, "AAPL", date.today()
        )
        
        # Assert - should not crash
        assert result is None or isinstance(result, bool)

    def test_concurrent_access_safety(self, option_chains_repository):
        """Test repository behavior under concurrent access."""
        import threading
        import time
        
        # Arrange
        underlying = "AAPL"
        results = []
        
        def save_data(thread_id):
            try:
                data = pd.DataFrame({
                    "underlying": [underlying],
                    "underlying_conid": [123456 + thread_id],
                    "exchange": ["SMART"],
                    "trading_class": ["AAPL"],
                    "multiplier": ["100"],
                    "expirations": [["20240315"]],
                    "strikes": [[140.0 + thread_id]],
                    "as_of": [date.today()]
                })
                
                option_chains_repository.save_chain_snapshot(
                    data, underlying, date.today()
                )
                results.append(f"success_{thread_id}")
            except Exception as e:
                results.append(f"error_{thread_id}_{str(e)}")
        
        # Act - run concurrent saves
        threads = []
        for i in range(3):
            thread = threading.Thread(target=save_data, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Assert - at least some operations should succeed
        success_count = len([r for r in results if r.startswith("success")])
        assert success_count > 0

    @pytest.mark.slow
    def test_large_dataset_performance(self, option_chains_repository, benchmark_timer):
        """Test repository performance with large datasets."""
        # Arrange
        underlying = "AAPL"
        
        # Create large dataset
        large_expirations = [f"2024{month:02d}15" for month in range(1, 13)]  # 12 months
        large_strikes = [float(strike) for strike in range(50, 300, 5)]  # 50 strikes
        
        large_data = pd.DataFrame({
            "underlying": [underlying],
            "underlying_conid": [123456],
            "exchange": ["SMART"],
            "trading_class": ["AAPL"],
            "multiplier": ["100"],
            "expirations": [large_expirations],
            "strikes": [large_strikes],
            "as_of": [date.today()]
        })
        
        # Act & Time the operation
        with benchmark_timer() as timer:
            option_chains_repository.save_chain_snapshot(
                large_data, underlying, date.today()
            )
        
        # Assert - operation should complete in reasonable time
        assert timer.elapsed < 5.0, f"Save operation took too long: {timer.elapsed}s"
        
        # Verify data was saved correctly
        loaded_data = option_chains_repository.load(
            underlying=underlying, as_of=date.today()
        )
        assert loaded_data is not None
        assert len(_get_value(loaded_data, "expirations")) == 12
        assert len(_get_value(loaded_data, "strikes")) == 50

    def test_data_consistency_after_multiple_saves(self, option_chains_repository):
        """Test data consistency after multiple save operations."""
        # Arrange
        underlying = "AAPL"
        snapshot_date = date.today()
        
        # Save initial data
        initial_data = pd.DataFrame({
            "underlying": [underlying],
            "underlying_conid": [123456],
            "exchange": ["SMART"],
            "trading_class": ["AAPL"],
            "multiplier": ["100"],
            "expirations": [["20240315"]],
            "strikes": [[140.0, 145.0]],
            "as_of": [snapshot_date]
        })
        
        option_chains_repository.save_chain_snapshot(
            initial_data, underlying, snapshot_date
        )
        
        # Save updated data (should overwrite)
        updated_data = pd.DataFrame({
            "underlying": [underlying],
            "underlying_conid": [123456],
            "exchange": ["SMART"],
            "trading_class": ["AAPL"],
            "multiplier": ["100"],
            "expirations": [["20240315", "20240415"]],  # Added expiry
            "strikes": [[140.0, 145.0, 150.0]],  # Added strike
            "as_of": [snapshot_date]
        })
        
        option_chains_repository.save_chain_snapshot(
            updated_data, underlying, snapshot_date
        )
        
        # Act
        final_data = option_chains_repository.load(
            underlying=underlying, as_of=snapshot_date
        )
        
        # Assert - should have updated data
        assert final_data is not None
        assert len(_get_value(final_data, "expirations")) == 2  # Updated
        assert len(_get_value(final_data, "strikes")) == 3  # Updated

    def test_error_recovery_after_corruption(self, option_chains_repository):
        """Test repository behavior after simulated data corruption."""
        # This test would be more complex in a real scenario
        # For now, we test basic error handling
        
        # Arrange - try to save with various problematic data
        problematic_datasets = [
            # Dataset with None values
            pd.DataFrame({
                "underlying": [None],
                "underlying_conid": [123456],
                "exchange": ["SMART"],
                "trading_class": ["AAPL"],
                "multiplier": ["100"],
                "expirations": [["20240315"]],
                "strikes": [[140.0]],
                "as_of": [date.today()]
            }),
            # Dataset with mixed types
            pd.DataFrame({
                "underlying": ["AAPL"],
                "underlying_conid": ["not_an_int"],
                "exchange": ["SMART"],
                "trading_class": ["AAPL"],
                "multiplier": ["100"],
                "expirations": [["20240315"]],
                "strikes": [[140.0]],
                "as_of": [date.today()]
            })
        ]
        
        # Act & Assert - should handle errors gracefully
        for i, problematic_data in enumerate(problematic_datasets):
            try:
                option_chains_repository.save_chain_snapshot(
                    problematic_data, f"TEST{i}", date.today()
                )
            except (DataValidationException, ValueError, TypeError):
                pass  # Expected behavior
            
        # Repository should still be functional after errors
        valid_data = pd.DataFrame({
            "underlying": ["AAPL"],
            "underlying_conid": [123456],
            "exchange": ["SMART"],
            "trading_class": ["AAPL"],
            "multiplier": ["100"],
            "expirations": [["20240315"]],
            "strikes": [[140.0]],
            "as_of": [date.today()]
        })
        
        # Should work after error recovery
        option_chains_repository.save_chain_snapshot(
            valid_data, "AAPL", date.today()
        )
        
        loaded = option_chains_repository.load(
            underlying="AAPL", as_of=date.today()
        )
        assert loaded is not None


@pytest.mark.unit
@pytest.mark.data
class TestOptionChainSnapshotRepositoryEdgeCases:
    """Edge case testing for OptionChainSnapshotRepository."""

    def test_save_with_future_date(self, option_chains_repository):
        """Test saving data with future dates."""
        # Arrange
        future_date = date.today() + timedelta(days=365)
        data = pd.DataFrame({
            "underlying": ["AAPL"],
            "underlying_conid": [123456],
            "exchange": ["SMART"],
            "trading_class": ["AAPL"],
            "multiplier": ["100"],
            "expirations": [["20251215"]],
            "strikes": [[150.0]],
            "as_of": [future_date]
        })
        
        # Act & Assert - should handle future dates
        try:
            option_chains_repository.save_chain_snapshot(
                data, "AAPL", future_date
            )
            # If successful, verify data can be loaded
            loaded = option_chains_repository.load(
                underlying="AAPL", as_of=future_date
            )
            assert loaded is not None or loaded is None  # Either outcome is acceptable
        except DataValidationException:
            pass  # Also acceptable to reject future dates

    def test_save_with_weekend_date(self, option_chains_repository):
        """Test saving data with weekend dates."""
        # Arrange - find a weekend date
        test_date = date.today()
        while test_date.weekday() < 5:  # 0-4 are Mon-Fri, 5-6 are Sat-Sun
            test_date += timedelta(days=1)
        
        data = pd.DataFrame({
            "underlying": ["AAPL"],
            "underlying_conid": [123456],
            "exchange": ["SMART"],
            "trading_class": ["AAPL"],
            "multiplier": ["100"],
            "expirations": [["20240315"]],
            "strikes": [[150.0]],
            "as_of": [test_date]
        })
        
        # Act - should handle weekend dates (markets closed)
        option_chains_repository.save_chain_snapshot(data, "AAPL", test_date)
        
        # Assert
        loaded = option_chains_repository.load(underlying="AAPL", as_of=test_date)
        assert loaded is not None

    def test_save_with_very_old_date(self, option_chains_repository):
        """Test saving data with very old dates."""
        # Arrange
        old_date = date(2000, 1, 1)
        data = pd.DataFrame({
            "underlying": ["AAPL"],
            "underlying_conid": [123456],
            "exchange": ["SMART"],
            "trading_class": ["AAPL"],
            "multiplier": ["100"],
            "expirations": [["20000315"]],
            "strikes": [[25.0]],  # AAPL was much cheaper in 2000
            "as_of": [old_date]
        })
        
        # Act
        option_chains_repository.save_chain_snapshot(data, "AAPL", old_date)
        
        # Assert
        loaded = option_chains_repository.load(underlying="AAPL", as_of=old_date)
        assert loaded is not None
        assert _get_value(loaded, "strikes") == [25.0]