"""
Unit tests for EquityBarRepository.
"""
import pytest
import pandas as pd
from datetime import date, datetime
from pathlib import Path

from ibx_repos.equity_bars import EquityBarRepository


@pytest.mark.unit
@pytest.mark.data
class TestEquityBarRepository:
    """Test cases for EquityBarRepository."""
    
    def test_repository_initialization(self, temp_data_dir):
        """Test repository initialization creates necessary directories."""
        repo_path = temp_data_dir / "test_equity_bars"
        repo = EquityBarRepository(repo_path)
        
        assert repo.base_path.exists()
        assert repo.base_path == repo_path
        assert repo.schema is not None
    
    def test_save_daily_bars(self, equity_repository, sample_equity_data):
        """Test saving daily bar data."""
        symbol = "AAPL"
        bar_size = "1 day"
        
        # Save data
        equity_repository.save(sample_equity_data, symbol, bar_size, "TRADES")
        
        # Verify data was saved
        loaded_data = equity_repository.load(symbol=symbol)
        assert not loaded_data.empty
        assert len(loaded_data) == len(sample_equity_data)
        assert "symbol" in loaded_data.columns
        assert all(loaded_data["symbol"] == symbol)
    
    def test_save_intraday_bars(self, equity_repository):
        """Test saving intraday bar data."""
        symbol = "MSFT"
        bar_size = "1 min"
        
        # Create intraday data
        times = pd.date_range("2024-01-01 09:30", periods=100, freq="1min")
        intraday_data = pd.DataFrame({
            "time": times,
            "open": [100.0 + i * 0.01 for i in range(100)],
            "high": [100.5 + i * 0.01 for i in range(100)],
            "low": [99.5 + i * 0.01 for i in range(100)],
            "close": [100.2 + i * 0.01 for i in range(100)],
            "volume": [1000 + i * 10 for i in range(100)],
        })
        
        # Save data
        equity_repository.save(intraday_data, symbol, bar_size, "TRADES")
        
        # Verify data was saved
        loaded_data = equity_repository.load(symbol=symbol)
        assert not loaded_data.empty
        assert "trade_date" in loaded_data.columns
    
    def test_load_with_filters(self, populated_equity_repo):
        """Test loading data with various filters."""
        symbol = "AAPL"
        
        # Load all data for symbol
        all_data = populated_equity_repo.load(symbol=symbol)
        assert not all_data.empty
        assert all(all_data["symbol"] == symbol)
        
        # Load data for specific trade date
        target_date = date(2024, 1, 1)
        filtered_data = populated_equity_repo.load(symbol=symbol, trade_date=target_date)
        
        if not filtered_data.empty:
            assert all(filtered_data["trade_date"] == target_date)
    
    def test_present_dates(self, populated_equity_repo):
        """Test present_dates method."""
        symbol = "AAPL"
        bar_size = "1 day"
        start = pd.Timestamp("2023-12-31")
        end = pd.Timestamp("2024-01-31")
        
        present_dates = populated_equity_repo.present_dates(symbol, bar_size, start, end)
        
        assert isinstance(present_dates, set)
        # Should have some dates from our sample data
        assert len(present_dates) > 0
    
    def test_normalize_data(self, equity_repository):
        """Test data normalization during save."""
        symbol = "TEST"
        bar_size = "1 day"
        
        # Create data with various formats that need normalization
        test_data = pd.DataFrame({
            "date": ["20240101", "20240102", "20240103"],
            "open": ["100.50", "101.00", "101.50"],  # String prices
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.75, 101.25, 102.0],
            "volume": ["1000000", "1100000", "1200000"],  # String volumes
        })
        
        # Save and reload
        equity_repository.save(test_data, symbol, bar_size)
        loaded_data = equity_repository.load(symbol=symbol)
        
        # Verify data types are correct
        assert loaded_data["open"].dtype == "float64"
        assert loaded_data["volume"].dtype == "float64"
        assert pd.api.types.is_datetime64_any_dtype(loaded_data["time"])
    
    def test_empty_dataframe_handling(self, equity_repository):
        """Test handling of empty dataframes."""
        symbol = "EMPTY"
        bar_size = "1 day"
        
        # Create empty dataframe with correct columns
        empty_data = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        
        # Should not raise an error
        equity_repository.save(empty_data, symbol, bar_size)
        
        # Loading should return empty dataframe
        loaded_data = equity_repository.load(symbol=symbol)
        assert loaded_data.empty
    
    def test_schema_compliance(self, equity_repository, sample_equity_data):
        """Test that saved data complies with repository schema."""
        symbol = "SCHEMA_TEST"
        bar_size = "1 day"
        
        equity_repository.save(sample_equity_data, symbol, bar_size)
        loaded_data = equity_repository.load(symbol=symbol)
        
        # Check required columns exist
        required_columns = [
            "time", "date", "trade_date", "symbol", "bar_size",
            "open", "high", "low", "close", "volume"
        ]
        
        for col in required_columns:
            assert col in loaded_data.columns, f"Missing required column: {col}"
    
    @pytest.mark.slow
    def test_large_dataset_performance(self, equity_repository, large_dataset):
        """Test repository performance with large datasets."""
        symbol = "PERF_TEST"
        bar_size = "1 min"
        
        # Time the save operation
        import time
        start_time = time.time()
        equity_repository.save(large_dataset, symbol, bar_size)
        save_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert save_time < 30.0, f"Save took too long: {save_time:.2f}s"
        
        # Time the load operation
        start_time = time.time()
        loaded_data = equity_repository.load(symbol=symbol)
        load_time = time.time() - start_time
        
        assert load_time < 10.0, f"Load took too long: {load_time:.2f}s"
        assert len(loaded_data) == len(large_dataset)
    
    def test_concurrent_access(self, equity_repository, sample_equity_data):
        """Test concurrent access to repository."""
        import threading
        import time
        
        results = []
        errors = []
        
        def save_data(thread_id):
            try:
                symbol = f"THREAD_{thread_id}"
                equity_repository.save(sample_equity_data, symbol, "1 day")
                results.append(thread_id)
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=save_data, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5, f"Not all threads completed: {results}"


@pytest.mark.unit
class TestEquityBarRepositoryEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_path_handling(self):
        """Test repository behavior with invalid paths."""
        # Test with read-only directory (if possible)
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = Path(temp_dir) / "nonexistent" / "deeply" / "nested"
            
            # Should create directories as needed
            repo = EquityBarRepository(invalid_path)
            assert repo.base_path.exists()
    
    def test_corrupted_data_handling(self, equity_repository):
        """Test handling of corrupted or invalid data."""
        symbol = "CORRUPT"
        bar_size = "1 day"
        
        # Create data with invalid values
        corrupt_data = pd.DataFrame({
            "date": ["invalid_date", "20240102", "20240103"],
            "open": [float("inf"), 101.0, 102.0],
            "high": [101.0, float("nan"), 103.0],
            "low": [-1.0, 100.0, 101.0],  # Negative price
            "close": [100.0, 101.0, None],
            "volume": [-1000, 1100000, 1200000],  # Negative volume
        })
        
        # Should handle gracefully without crashing
        try:
            equity_repository.save(corrupt_data, symbol, bar_size)
            loaded_data = equity_repository.load(symbol=symbol)
            # Verify some data was saved (with corrections)
            assert not loaded_data.empty
        except Exception as e:
            pytest.fail(f"Repository should handle corrupted data gracefully: {e}")
    
    def test_memory_efficiency(self, equity_repository):
        """Test memory efficiency with multiple symbols."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Save data for many symbols
        for i in range(100):
            symbol = f"MEM_TEST_{i:03d}"
            small_data = pd.DataFrame({
                "date": ["20240101"],
                "open": [100.0], "high": [101.0], "low": [99.0],
                "close": [100.5], "volume": [1000000]
            })
            equity_repository.save(small_data, symbol, "1 day")
        
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory growth should be reasonable (adjust threshold as needed)
        assert memory_growth < 100, f"Memory growth too high: {memory_growth:.2f}MB"