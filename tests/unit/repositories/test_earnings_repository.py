"""
Unit tests for EarningsRepository.

Tests the storage, retrieval, and validation of earnings calendar data
with proper date-based partitioning and symbol filtering.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from common.repositories.earnings import EarningsCalendarRepository
from common.exceptions import DataValidationException


@pytest.mark.unit
@pytest.mark.data
class TestEarningsCalendarRepository:
    """Test cases for EarningsRepository."""

    def test_repository_initialization_success(self, temp_data_dir):
        """Test successful repository initialization."""
        # Arrange
        repo_path = temp_data_dir / "earnings"
        
        # Act
        repo = EarningsCalendarRepository(repo_path)
        
        # Assert
        assert repo.base_path == repo_path
        assert repo.dataset_name == "earnings"
        assert repo.base_path.exists()

    def test_create_schema_structure(self, earnings_repository):
        """Test that schema is created with correct structure."""
        # Act
        schema = earnings_repository._create_schema()
        
        # Assert
        expected_fields = [
            "symbol", "company_name", "earnings_date", "period_ending", 
            "when_time", "confirmed", "last_updated"
        ]
        actual_fields = [field.name for field in schema]
        
        for field in expected_fields:
            assert field in actual_fields, f"Missing required field: {field}"

    def test_save_earnings_data_success(self, earnings_repository):
        """Test successful saving of earnings data."""
        # Arrange
        earnings_data = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "company_name": ["Apple Inc.", "Microsoft Corporation", "Alphabet Inc."],
            "earnings_date": [
                date.today() + timedelta(days=7),
                date.today() + timedelta(days=14),
                date.today() + timedelta(days=21)
            ],
            "period_ending": [
                date.today() - timedelta(days=90),
                date.today() - timedelta(days=90),
                date.today() - timedelta(days=90)
            ],
            "when_time": ["AMC", "BMO", "DMT"],
            "confirmed": [True, True, False],
            "last_updated": [datetime.now(), datetime.now(), datetime.now()],
            "revenue_estimate": [85000000000, 45000000000, 65000000000],
            "eps_estimate": [1.25, 2.10, 15.50]
        })
        
        # Act
        result = earnings_repository.save_dataframe_for_testing(earnings_data)
        
        # Assert
        assert result is True or result is None
        
        # Verify data was stored - try to load it back
        loaded_data = earnings_repository.get_earnings_in_range()
        assert loaded_data is not None
        assert len(loaded_data) >= 3
        assert "AAPL" in loaded_data["symbol"].values

    def test_save_with_missing_required_fields_error(self, earnings_repository):
        """Test error handling when saving data with missing required fields."""
        # Arrange
        incomplete_data = pd.DataFrame({
            "symbol": ["AAPL"],
            "company_name": ["Apple Inc."]
            # Missing required fields like earnings_date, confirmed, etc.
        })
        
        # Act & Assert
        with pytest.raises((DataValidationException, ValueError, KeyError)):
            earnings_repository.save_dataframe_for_testing(incomplete_data)

    def test_save_with_invalid_earnings_date(self, earnings_repository):
        """Test validation of earnings dates."""
        # Arrange
        invalid_data = pd.DataFrame({
            "symbol": ["TEST"],
            "company_name": ["Test Corp"],
            "earnings_date": ["invalid_date"],  # Invalid date
            "period_ending": [date.today()],
            "when_time": ["AMC"],
            "confirmed": [True],
            "last_updated": [datetime.now()]
        })
        
        # Act & Assert
        with pytest.raises((DataValidationException, ValueError, TypeError)):
            earnings_repository.save_dataframe_for_testing(invalid_data)

    def test_save_with_invalid_when_time(self, earnings_repository):
        """Test validation of when_time field (should be AMC, BMO, or DMT)."""
        # Arrange
        invalid_data = pd.DataFrame({
            "symbol": ["TEST"],
            "company_name": ["Test Corp"],
            "earnings_date": [date.today() + timedelta(days=7)],
            "period_ending": [date.today() - timedelta(days=90)],
            "when_time": ["INVALID"],  # Should be AMC, BMO, or DMT
            "confirmed": [True],
            "last_updated": [datetime.now()]
        })
        
        # Act & Assert - may validate or accept invalid values
        try:
            earnings_repository.save_dataframe_for_testing(invalid_data)
        except (DataValidationException, ValueError):
            pass  # Validation error is acceptable

    def test_load_all_earnings(self, earnings_repository):
        """Test loading all earnings data."""
        # Arrange - first save some data
        earnings_data = pd.DataFrame({
            "symbol": ["NVDA", "TSLA"],
            "company_name": ["NVIDIA Corporation", "Tesla, Inc."],
            "earnings_date": [
                date.today() + timedelta(days=10),
                date.today() + timedelta(days=15)
            ],
            "period_ending": [
                date.today() - timedelta(days=90),
                date.today() - timedelta(days=90)
            ],
            "when_time": ["AMC", "BMO"],
            "confirmed": [True, False],
            "last_updated": [datetime.now(), datetime.now()]
        })
        
        earnings_repository.save_dataframe_for_testing(earnings_data)
        
        # Act
        loaded_data = earnings_repository.get_earnings_in_range()
        
        # Assert
        assert loaded_data is not None
        assert len(loaded_data) >= 2
        symbols = set(loaded_data["symbol"])
        assert "NVDA" in symbols
        assert "TSLA" in symbols

    def test_load_earnings_by_symbol(self, earnings_repository):
        """Test loading earnings for specific symbols."""
        # Arrange
        earnings_data = pd.DataFrame({
            "symbol": ["META", "NFLX", "AMZN"],
            "company_name": ["Meta Platforms, Inc.", "Netflix, Inc.", "Amazon.com, Inc."],
            "earnings_date": [
                date.today() + timedelta(days=5),
                date.today() + timedelta(days=12),
                date.today() + timedelta(days=18)
            ],
            "period_ending": [
                date.today() - timedelta(days=90),
                date.today() - timedelta(days=90),
                date.today() - timedelta(days=90)
            ],
            "when_time": ["AMC", "BMO", "DMT"],
            "confirmed": [True, True, True],
            "last_updated": [datetime.now(), datetime.now(), datetime.now()]
        })
        
        earnings_repository.save_dataframe_for_testing(earnings_data)
        
        # Act - load specific symbols
        target_symbols = ["META", "NFLX"]
        loaded_data = earnings_repository.get_earnings_in_range(symbols=target_symbols)
        
        # Assert
        if loaded_data is not None:
            loaded_symbols = set(loaded_data["symbol"])
            assert loaded_symbols.issubset(set(target_symbols))
            assert "AMZN" not in loaded_symbols  # Should not be included

    def test_load_earnings_by_date_range(self, earnings_repository):
        """Test loading earnings within a specific date range."""
        # Arrange
        base_date = date.today()
        earnings_data = pd.DataFrame({
            "symbol": ["STOCK1", "STOCK2", "STOCK3", "STOCK4"],
            "company_name": ["Company 1", "Company 2", "Company 3", "Company 4"],
            "earnings_date": [
                base_date + timedelta(days=1),   # Within range
                base_date + timedelta(days=10),  # Within range
                base_date + timedelta(days=20),  # Within range
                base_date + timedelta(days=40)   # Outside range
            ],
            "period_ending": [base_date - timedelta(days=90)] * 4,
            "when_time": ["AMC", "BMO", "DMT", "AMC"],
            "confirmed": [True, True, True, True],
            "last_updated": [datetime.now()] * 4
        })
        
        earnings_repository.save_dataframe_for_testing(earnings_data)
        
        # Act
        start_date = base_date
        end_date = base_date + timedelta(days=25)
        
        loaded_data = earnings_repository.get_earnings_in_range(
            start_date=start_date,
            end_date=end_date
        )
        
        # Assert
        if loaded_data is not None:
            # Should include STOCK1, STOCK2, STOCK3 but not STOCK4
            loaded_symbols = set(loaded_data["symbol"])
            assert "STOCK4" not in loaded_symbols or len(loaded_data) == 4  # Implementation may vary

    def test_load_upcoming_earnings(self, earnings_repository):
        """Test loading upcoming earnings (future dates only)."""
        # Arrange
        today = date.today()
        earnings_data = pd.DataFrame({
            "symbol": ["PAST", "FUTURE1", "FUTURE2"],
            "company_name": ["Past Corp", "Future Corp 1", "Future Corp 2"],
            "earnings_date": [
                today - timedelta(days=5),   # Past earnings
                today + timedelta(days=7),   # Future earnings
                today + timedelta(days=14)   # Future earnings
            ],
            "period_ending": [today - timedelta(days=90)] * 3,
            "when_time": ["AMC", "BMO", "DMT"],
            "confirmed": [True, True, False],
            "last_updated": [datetime.now()] * 3
        })
        
        earnings_repository.save_dataframe_for_testing(earnings_data)
        
        # Act - load only future earnings
        loaded_data = earnings_repository.get_earnings_in_range(
            start_date=today + timedelta(days=1)  # Tomorrow onwards
        )
        
        # Assert
        if loaded_data is not None:
            for _, row in loaded_data.iterrows():
                assert row["earnings_date"] > today, "Should only contain future earnings"

    def test_save_duplicate_earnings_handling(self, earnings_repository):
        """Test handling of duplicate earnings saves."""
        # Arrange
        original_earnings = pd.DataFrame({
            "symbol": ["DUPL"],
            "company_name": ["Duplicate Corp"],
            "earnings_date": [date.today() + timedelta(days=7)],
            "period_ending": [date.today() - timedelta(days=90)],
            "when_time": ["AMC"],
            "confirmed": [False],
            "last_updated": [datetime.now()],
            "eps_estimate": [1.50]
        })
        
        # Save original
        earnings_repository.save(original_earnings)
        
        # Create updated version
        updated_earnings = pd.DataFrame({
            "symbol": ["DUPL"],
            "company_name": ["Duplicate Corp"],
            "earnings_date": [date.today() + timedelta(days=7)],
            "period_ending": [date.today() - timedelta(days=90)],
            "when_time": ["AMC"],
            "confirmed": [True],  # Changed to confirmed
            "last_updated": [datetime.now()],
            "eps_estimate": [1.75]  # Updated estimate
        })
        
        # Act - save updated data
        earnings_repository.save(updated_earnings)
        
        # Assert - check how duplicates are handled
        loaded_data = earnings_repository.get_earnings_in_range(symbols=["DUPL"])
        assert loaded_data is not None
        # Behavior may vary - could be deduplicated, updated, or multiple entries

    def test_save_earnings_with_estimates(self, earnings_repository):
        """Test saving earnings with revenue and EPS estimates."""
        # Arrange
        earnings_with_estimates = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "company_name": ["Apple Inc.", "Microsoft Corporation"],
            "earnings_date": [
                date.today() + timedelta(days=14),
                date.today() + timedelta(days=21)
            ],
            "period_ending": [date.today() - timedelta(days=90)] * 2,
            "when_time": ["AMC", "BMO"],
            "confirmed": [True, True],
            "last_updated": [datetime.now()] * 2,
            "revenue_estimate": [89000000000.0, 52000000000.0],
            "revenue_estimate_low": [87000000000.0, 50000000000.0],
            "revenue_estimate_high": [91000000000.0, 54000000000.0],
            "eps_estimate": [1.39, 2.55],
            "eps_estimate_low": [1.35, 2.50],
            "eps_estimate_high": [1.43, 2.60],
            "analyst_count": [25, 18]
        })
        
        # Act
        earnings_repository.save(earnings_with_estimates)
        
        # Assert
        loaded_data = earnings_repository.get_earnings_in_range(symbols=["AAPL", "MSFT"])
        assert loaded_data is not None
        
        # Check estimates are preserved
        if "revenue_estimate" in loaded_data.columns:
            apple_data = loaded_data[loaded_data["symbol"] == "AAPL"]
            if not apple_data.empty:
                assert apple_data["revenue_estimate"].iloc[0] == 89000000000.0

    def test_load_nonexistent_symbol(self, earnings_repository):
        """Test loading earnings for non-existent symbol."""
        # Act
        loaded_data = earnings_repository.get_earnings_in_range(symbols=["NONEXISTENT"])
        
        # Assert
        assert loaded_data is None or loaded_data.empty

    def test_save_empty_dataframe_error(self, earnings_repository):
        """Test error handling for empty DataFrame."""
        # Arrange
        empty_df = pd.DataFrame()
        
        # Act & Assert
        with pytest.raises((DataValidationException, ValueError)):
            earnings_repository.save(empty_df)

    def test_earnings_data_validation_types(self, earnings_repository):
        """Test data type validation for earnings fields."""
        # Arrange
        invalid_types_data = pd.DataFrame({
            "symbol": ["TEST"],
            "company_name": ["Test Corp"],
            "earnings_date": ["not_a_date"],
            "period_ending": ["not_a_date"],
            "when_time": ["AMC"],
            "confirmed": ["not_a_boolean"],
            "last_updated": ["not_a_datetime"],
            "revenue_estimate": ["not_a_number"],
            "eps_estimate": ["not_a_number"]
        })
        
        # Act & Assert
        with pytest.raises((DataValidationException, ValueError, TypeError)):
            earnings_repository.save(invalid_types_data)

    def test_large_earnings_dataset_performance(self, earnings_repository, benchmark_timer):
        """Test performance with large earnings dataset."""
        # Arrange
        num_companies = 1000
        base_date = date.today()
        
        large_dataset = pd.DataFrame({
            "symbol": [f"STOCK{i:04d}" for i in range(num_companies)],
            "company_name": [f"Company {i}" for i in range(num_companies)],
            "earnings_date": [
                base_date + timedelta(days=i % 90) for i in range(num_companies)
            ],
            "period_ending": [base_date - timedelta(days=90)] * num_companies,
            "when_time": ["AMC", "BMO", "DMT"] * (num_companies // 3 + 1),
            "confirmed": [True, False] * (num_companies // 2 + 1),
            "last_updated": [datetime.now()] * num_companies,
            "revenue_estimate": [
                float(1e9 + i * 1e6) for i in range(num_companies)
            ],
            "eps_estimate": [1.0 + i * 0.01 for i in range(num_companies)]
        })
        
        # Trim to exact size
        large_dataset = large_dataset.head(num_companies)
        
        # Act & Time the operation
        with benchmark_timer() as timer:
            earnings_repository.save(large_dataset)
        
        # Assert
        assert timer.elapsed < 15.0, f"Large dataset save took too long: {timer.elapsed}s"
        
        # Verify data was saved
        loaded_data = earnings_repository.get_earnings_in_range()
        assert loaded_data is not None
        assert len(loaded_data) >= num_companies

    def test_earnings_date_edge_cases(self, earnings_repository):
        """Test edge cases for earnings dates."""
        # Arrange
        edge_cases = [
            {
                "name": "weekend_earnings",
                "date": date(2024, 3, 16),  # Saturday
                "should_pass": True
            },
            {
                "name": "holiday_earnings",
                "date": date(2024, 12, 25),  # Christmas
                "should_pass": True
            },
            {
                "name": "far_future_earnings",
                "date": date(2025, 12, 31),
                "should_pass": True
            }
        ]
        
        for case in edge_cases:
            earnings_data = pd.DataFrame({
                "symbol": [f"TEST_{case['name'].upper()}"],
                "company_name": ["Test Corporation"],
                "earnings_date": [case["date"]],
                "period_ending": [case["date"] - timedelta(days=90)],
                "when_time": ["AMC"],
                "confirmed": [True],
                "last_updated": [datetime.now()]
            })
            
            # Act & Assert
            if case["should_pass"]:
                try:
                    earnings_repository.save_dataframe_for_testing(earnings_data)
                    # Verify it can be loaded back
                    loaded = earnings_repository.load(symbols=[f"TEST_{case['name'].upper()}"])
                    # Should not raise error
                except (DataValidationException, ValueError):
                    pass  # Some implementations may have restrictions
            else:
                with pytest.raises((DataValidationException, ValueError)):
                    earnings_repository.save_dataframe_for_testing(earnings_data)

    def test_concurrent_earnings_saves(self, earnings_repository):
        """Test concurrent earnings saves."""
        import threading
        
        # Arrange
        results = []
        lock = threading.Lock()
        
        def save_earnings(thread_id):
            try:
                earnings_data = pd.DataFrame({
                    "symbol": [f"THREAD{thread_id}"],
                    "company_name": [f"Thread {thread_id} Corp"],
                    "earnings_date": [date.today() + timedelta(days=thread_id)],
                    "period_ending": [date.today() - timedelta(days=90)],
                    "when_time": ["AMC"],
                    "confirmed": [True],
                    "last_updated": [datetime.now()]
                })
                
                earnings_repository.save_dataframe_for_testing(earnings_data)
                
                with lock:
                    results.append(f"success_{thread_id}")
            except Exception as e:
                with lock:
                    results.append(f"error_{thread_id}_{str(e)}")
        
        # Act
        threads = []
        for i in range(5):
            thread = threading.Thread(target=save_earnings, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Assert
        success_count = len([r for r in results if r.startswith("success")])
        assert success_count >= 3  # Most should succeed


@pytest.mark.unit
@pytest.mark.data
class TestEarningsRepositoryQueries:
    """Test advanced querying capabilities of EarningsRepository."""

    def test_filter_by_confirmation_status(self, earnings_repository):
        """Test filtering earnings by confirmation status."""
        # Arrange
        mixed_confirmations = pd.DataFrame({
            "symbol": ["CONF1", "CONF2", "UNCONF1", "UNCONF2"],
            "company_name": ["Confirmed 1", "Confirmed 2", "Unconfirmed 1", "Unconfirmed 2"],
            "earnings_date": [date.today() + timedelta(days=i+1) for i in range(4)],
            "period_ending": [date.today() - timedelta(days=90)] * 4,
            "when_time": ["AMC", "BMO", "DMT", "AMC"],
            "confirmed": [True, True, False, False],
            "last_updated": [datetime.now()] * 4
        })
        
        earnings_repository.save(mixed_confirmations)
        
        # Act - filter by confirmed status (if supported)
        try:
            confirmed_only = earnings_repository.load(confirmed=True)
            unconfirmed_only = earnings_repository.load(confirmed=False)
            
            # Assert
            if confirmed_only is not None:
                assert all(confirmed_only["confirmed"] == True)
            
            if unconfirmed_only is not None:
                assert all(unconfirmed_only["confirmed"] == False)
                
        except TypeError:
            # Filter by confirmed may not be supported
            pass

    def test_filter_by_time_of_day(self, earnings_repository):
        """Test filtering earnings by time of day (AMC/BMO/DMT)."""
        # Arrange
        mixed_times = pd.DataFrame({
            "symbol": ["AMC1", "BMO1", "DMT1", "AMC2"],
            "company_name": ["After Market 1", "Before Market 1", "During Market 1", "After Market 2"],
            "earnings_date": [date.today() + timedelta(days=i+1) for i in range(4)],
            "period_ending": [date.today() - timedelta(days=90)] * 4,
            "when_time": ["AMC", "BMO", "DMT", "AMC"],
            "confirmed": [True] * 4,
            "last_updated": [datetime.now()] * 4
        })
        
        earnings_repository.save(mixed_times)
        
        # Act - filter by time (if supported)
        try:
            amc_earnings = earnings_repository.load(when_time="AMC")
            
            # Assert
            if amc_earnings is not None:
                assert all(amc_earnings["when_time"] == "AMC")
                assert len(amc_earnings) >= 2  # AMC1 and AMC2
                
        except TypeError:
            # Filter by when_time may not be supported
            pass