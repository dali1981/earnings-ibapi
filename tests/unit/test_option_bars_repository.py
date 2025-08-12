"""
Unit tests for OptionBarRepository.
"""
import pytest
import pandas as pd
from datetime import date, datetime
from pathlib import Path

from ibx_repos.option_bars import OptionBarRepository, OptionMeta


@pytest.mark.unit
@pytest.mark.data
class TestOptionBarRepository:
    """Test cases for OptionBarRepository."""
    
    def test_repository_initialization(self, temp_data_dir):
        """Test repository initialization creates necessary directories."""
        repo_path = temp_data_dir / "test_option_bars"
        repo = OptionBarRepository(repo_path)
        
        assert repo.base_path.exists()
        assert repo.base_path == repo_path
        assert repo.schema is not None
    
    def test_option_meta_creation(self):
        """Test OptionMeta dataclass creation."""
        meta = OptionMeta(
            underlying="AAPL",
            expiry=date(2024, 3, 15),
            strike=150.0,
            right="C",
            bar_size="1 min"
        )
        
        assert meta.underlying == "AAPL"
        assert meta.expiry == date(2024, 3, 15)
        assert meta.strike == 150.0
        assert meta.right == "C"
        assert meta.bar_size == "1 min"
    
    def test_expiry_normalization(self):
        """Test expiry date normalization."""
        repo = OptionBarRepository(Path("/tmp/test"))
        
        # Test with date object
        assert repo._norm_expiry(date(2024, 3, 15)) == "20240315"
        
        # Test with string formats
        assert repo._norm_expiry("20240315") == "20240315"
        assert repo._norm_expiry("202403") == "20240301"  # Adds day
    
    def test_save_intraday_option_bars(self, option_repository, sample_option_data):
        """Test saving intraday option bar data."""
        meta = OptionMeta(
            underlying="AAPL",
            expiry=date(2024, 3, 15),
            strike=150.0,
            right="C",
            bar_size="1 min",
            what_to_show="TRADES"
        )
        
        # Save data
        option_repository.save(sample_option_data, meta)
        
        # Verify data was saved
        loaded_data = option_repository.load(
            underlying="AAPL",
            expiry=date(2024, 3, 15),
            right="C",
            strike=150.0
        )
        
        assert not loaded_data.empty
        assert len(loaded_data) == len(sample_option_data)
        assert all(loaded_data["underlying"] == "AAPL")
        assert all(loaded_data["right"] == "C")
        assert all(loaded_data["strike"] == 150.0)
    
    def test_save_daily_option_bars(self, option_repository):
        """Test saving daily option bar data."""
        daily_data = pd.DataFrame({
            "date": ["20240101", "20240102", "20240103"],
            "open": [5.0, 5.1, 5.2],
            "high": [5.1, 5.2, 5.3],
            "low": [4.9, 5.0, 5.1],
            "close": [5.05, 5.15, 5.25],
            "volume": [1000, 1100, 1200],
        })
        
        meta = OptionMeta(
            underlying="MSFT",
            expiry=date(2024, 4, 15),
            strike=200.0,
            right="P",
            bar_size="1 day"
        )
        
        # Save data
        option_repository.save(daily_data, meta)
        
        # Verify data was saved
        loaded_data = option_repository.load(
            underlying="MSFT",
            expiry="20240415",
            right="P",
            strike=200.0
        )
        
        assert not loaded_data.empty
        assert len(loaded_data) == 3
        assert all(loaded_data["underlying"] == "MSFT")
        assert all(loaded_data["expiry"] == "20240415")
    
    def test_load_with_filters(self, option_repository, sample_option_data):
        """Test loading data with various filters."""
        meta = OptionMeta(
            underlying="GOOGL",
            expiry=date(2024, 6, 21),
            strike=300.0,
            right="C",
            bar_size="5 mins"
        )
        
        option_repository.save(sample_option_data, meta)
        
        # Test different filter combinations
        test_cases = [
            {"underlying": "GOOGL"},
            {"underlying": "GOOGL", "right": "C"},
            {"underlying": "GOOGL", "expiry": date(2024, 6, 21)},
            {"underlying": "GOOGL", "strike": 300.0},
            {"underlying": "GOOGL", "expiry": "20240621", "right": "C", "strike": 300.0}
        ]
        
        for filters in test_cases:
            loaded_data = option_repository.load(**filters)
            assert not loaded_data.empty, f"No data for filters: {filters}"
            
            # Verify filter conditions
            if "underlying" in filters:
                assert all(loaded_data["underlying"] == filters["underlying"])
            if "right" in filters:
                assert all(loaded_data["right"] == filters["right"])
            if "strike" in filters:
                assert all(loaded_data["strike"] == filters["strike"])
    
    def test_data_normalization(self, option_repository):
        """Test data normalization during save."""
        # Data with various formats that need normalization
        test_data = pd.DataFrame({
            "time": ["2024-01-01 09:30:00", "2024-01-01 09:31:00"],
            "open": ["5.50", "5.60"],  # String prices
            "high": [5.75, 5.80],
            "low": [5.40, 5.50],
            "close": [5.65, 5.75],
            "volume": ["500", "600"],  # String volumes
        })
        
        meta = OptionMeta(
            underlying="TEST",
            expiry=date(2024, 1, 19),
            strike=100.0,
            right="C",
            bar_size="1 min"
        )
        
        # Save and reload
        option_repository.save(test_data, meta)
        loaded_data = option_repository.load(underlying="TEST")
        
        # Verify data types are correct
        assert loaded_data["open"].dtype == "float64"
        assert loaded_data["volume"].dtype == "float64"
        assert pd.api.types.is_datetime64_any_dtype(loaded_data["time"])
        assert loaded_data["strike"].dtype == "float64"
    
    def test_present_dates_for_contract(self, option_repository, sample_option_data):
        """Test present_dates_for_contract method."""
        meta = OptionMeta(
            underlying="TSLA",
            expiry=date(2024, 5, 17),
            strike=250.0,
            right="P",
            bar_size="1 min"
        )
        
        option_repository.save(sample_option_data, meta)
        
        start = pd.Timestamp("2023-12-31")
        end = pd.Timestamp("2024-01-31")
        
        present_dates = option_repository.present_dates_for_contract(
            underlying="TSLA",
            expiry=date(2024, 5, 17),
            right="P",
            strike=250.0,
            bar_size="1 min",
            start=start,
            end=end
        )
        
        assert isinstance(present_dates, set)
        # Should have some dates from our sample data
        assert len(present_dates) > 0
    
    def test_invalid_meta_handling(self, option_repository):
        """Test handling of invalid OptionMeta."""
        # Test with missing required fields
        with pytest.raises((ValueError, TypeError)):
            OptionMeta(
                underlying="TEST",
                # Missing expiry, strike, right
                bar_size="1 min"
            )
    
    def test_empty_dataframe_handling(self, option_repository):
        """Test handling of empty dataframes."""
        meta = OptionMeta(
            underlying="EMPTY",
            expiry=date(2024, 1, 19),
            strike=100.0,
            right="C",
            bar_size="1 min"
        )
        
        # Create empty dataframe with correct columns
        empty_data = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        
        # Should not raise an error
        option_repository.save(empty_data, meta)
        
        # Loading should return empty dataframe
        loaded_data = option_repository.load(underlying="EMPTY")
        assert loaded_data.empty
    
    def test_schema_compliance(self, option_repository, sample_option_data):
        """Test that saved data complies with repository schema."""
        meta = OptionMeta(
            underlying="SCHEMA_TEST",
            expiry=date(2024, 2, 16),
            strike=175.0,
            right="C",
            bar_size="1 min"
        )
        
        option_repository.save(sample_option_data, meta)
        loaded_data = option_repository.load(underlying="SCHEMA_TEST")
        
        # Check required columns exist
        required_columns = [
            "time", "date", "trade_date", "underlying", "expiry", "right", "strike",
            "bar_size", "open", "high", "low", "close", "volume"
        ]
        
        for col in required_columns:
            assert col in loaded_data.columns, f"Missing required column: {col}"
    
    def test_multiple_contracts_same_underlying(self, option_repository, sample_option_data):
        """Test saving multiple option contracts for same underlying."""
        contracts = [
            OptionMeta("AAPL", date(2024, 3, 15), 140.0, "C", "1 min"),
            OptionMeta("AAPL", date(2024, 3, 15), 150.0, "C", "1 min"),
            OptionMeta("AAPL", date(2024, 3, 15), 160.0, "C", "1 min"),
            OptionMeta("AAPL", date(2024, 3, 15), 150.0, "P", "1 min"),
        ]
        
        # Save data for each contract
        for meta in contracts:
            option_repository.save(sample_option_data, meta)
        
        # Load all data for AAPL
        all_data = option_repository.load(underlying="AAPL")
        assert not all_data.empty
        
        # Should have data for all strikes and rights
        strikes = set(all_data["strike"].unique())
        rights = set(all_data["right"].unique())
        
        assert 140.0 in strikes
        assert 150.0 in strikes
        assert 160.0 in strikes
        assert "C" in rights
        assert "P" in rights
    
    @pytest.mark.slow
    def test_large_dataset_performance(self, option_repository, large_dataset):
        """Test repository performance with large datasets."""
        meta = OptionMeta(
            underlying="PERF_TEST",
            expiry=date(2024, 12, 20),
            strike=500.0,
            right="C",
            bar_size="1 min"
        )
        
        # Time the save operation
        import time
        start_time = time.time()
        option_repository.save(large_dataset, meta)
        save_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert save_time < 30.0, f"Save took too long: {save_time:.2f}s"
        
        # Time the load operation
        start_time = time.time()
        loaded_data = option_repository.load(underlying="PERF_TEST")
        load_time = time.time() - start_time
        
        assert load_time < 10.0, f"Load took too long: {load_time:.2f}s"
        assert len(loaded_data) == len(large_dataset)


@pytest.mark.unit
class TestOptionMetaEdgeCases:
    """Test edge cases for OptionMeta dataclass."""
    
    def test_expiry_string_formats(self):
        """Test various expiry string formats."""
        repo = OptionBarRepository(Path("/tmp/test"))
        
        # Test various formats
        test_cases = [
            ("2024-03-15", "20240315"),
            ("20240315", "20240315"),
            ("202403", "20240301"),
            (date(2024, 3, 15), "20240315"),
        ]
        
        for input_expiry, expected in test_cases:
            result = repo._norm_expiry(input_expiry)
            assert result == expected, f"Failed for input: {input_expiry}"
    
    def test_meta_with_optional_fields(self):
        """Test OptionMeta with optional fields."""
        meta = OptionMeta(
            underlying="TEST",
            expiry="20240315",
            strike=100.0,
            right="C",
            bar_size="1 min",
            what_to_show="TRADES"
        )
        
        assert meta.what_to_show == "TRADES"
    
    def test_meta_without_optional_fields(self):
        """Test OptionMeta without optional fields."""
        meta = OptionMeta(
            underlying="TEST",
            expiry="20240315",
            strike=100.0,
            right="C",
            bar_size="1 min"
        )
        
        assert meta.what_to_show is None


@pytest.mark.integration
class TestOptionBarRepositoryIntegration:
    """Integration tests for option bar repository."""
    
    def test_full_workflow_integration(self, option_repository, test_data_helper):
        """Test complete workflow from save to load."""
        # Create test data for multiple contracts
        base_date = date(2024, 1, 1)
        contracts_data = []
        
        strikes = [140.0, 150.0, 160.0]
        rights = ["C", "P"]
        
        for strike in strikes:
            for right in rights:
                meta = OptionMeta(
                    underlying="INTEGRATION_TEST",
                    expiry=date(2024, 3, 15),
                    strike=strike,
                    right=right,
                    bar_size="1 min"
                )
                
                # Create sample data
                data = test_data_helper.create_bars(
                    f"INTEGRATION_TEST_{strike}_{right}",
                    base_date, 
                    5, 
                    "1 min"
                )
                
                # Save data
                option_repository.save(data, meta)
                contracts_data.append((meta, data))
        
        # Verify all data was saved correctly
        all_data = option_repository.load(underlying="INTEGRATION_TEST")
        assert not all_data.empty
        
        # Should have data for all combinations
        unique_strikes = set(all_data["strike"].unique())
        unique_rights = set(all_data["right"].unique())
        
        assert unique_strikes == set(strikes)
        assert unique_rights == set(rights)
    
    def test_concurrent_access(self, option_repository, sample_option_data):
        """Test concurrent access to repository."""
        import threading
        import time
        
        results = []
        errors = []
        
        def save_data(thread_id):
            try:
                meta = OptionMeta(
                    underlying=f"THREAD_TEST",
                    expiry=date(2024, 3, 15),
                    strike=100.0 + thread_id,  # Different strikes
                    right="C",
                    bar_size="1 min"
                )
                option_repository.save(sample_option_data, meta)
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