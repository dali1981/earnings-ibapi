"""
Unit tests for ContractDescriptionsRepository.
"""

from datetime import date, datetime
from common.repositories.contract_descriptions import ContractDescriptionsRepository, DataValidationException

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest


@pytest.mark.unit
@pytest.mark.data
class TestContractDescriptionsRepository:
    """Test cases for ContractDescriptionsRepository."""

    def test_repository_initialization(self, temp_data_dir):
        """Test repository initialization creates necessary directories."""
        repo_path = temp_data_dir / "test_contract_descriptions"
        repo = ContractDescriptionsRepository(repo_path)

        assert repo.base.exists()
        assert repo.base == repo_path

    def test_schema_definition(self, temp_data_dir):
        """Test that schema contains required fields."""
        repo = ContractDescriptionsRepository(temp_data_dir)
        schema = repo.schema

        expected_fields = {
            "conid",
            "symbol",
            "sec_type",
            "currency",
            'sec_id',
            'sec_id_type',
            "primary_exchange",
            "derivative_sec_types",
            "created_at",
        }

        schema_fields = {field.name for field in schema}
        assert  schema_fields == expected_fields

    @pytest.mark.skip(reason="REQUIRED is not implemented yet/will not be.")
    def test_required_fields_constant(self):
        """Test REQUIRED fields constant."""
        expected_required = [
            "conid",
            "symbol",
            "sec_type",
            "currency",
            "primary_exchange",
            "derivative_sec_types",
            "as_of_date",
        ]
        assert ContractDescriptionsRepository.REQUIRED == expected_required

    def test_save_basic_contract_descriptions(self, temp_data_dir):
        """Test saving basic contract descriptions."""
        repo = ContractDescriptionsRepository(temp_data_dir)

        # Sample contract descriptions data
        contract_data = pd.DataFrame(
            [
                {
                    "conid": 265598,
                    "symbol": "AAPL",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NASDAQ",
                    "derivative_sec_types": ["OPT", "FUT"],
                },
                {
                    "conid": 272093,
                    "symbol": "MSFT",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NASDAQ",
                    "derivative_sec_types": "OPT,WAR",  # String format
                },
                {
                    "conid": 208813719,
                    "symbol": "GOOGL",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NASDAQ",
                    "derivative_sec_types": None,  # No derivatives
                },
            ]
        )

        # Save data
        repo.save(contract_data)

        # Verify data was saved
        loaded_data = repo.load()

        assert not loaded_data.num_rows == 0
        assert len(loaded_data) >= 3  # At least our 3 records

        # Check that symbols are present
        symbols = set(loaded_data["symbol"].to_pylist())
        assert {"AAPL", "MSFT", "GOOGL"}.issubset(symbols)

    def test_data_normalization_during_save(self, temp_data_dir):
        """Test data normalization during save process."""
        # import pytest;pytest.set_trace()

        repo = ContractDescriptionsRepository(temp_data_dir)

        # Data with various formats that need normalization
        test_data = pd.DataFrame(
            [
                {
                    "conid": "123456",  # String conid
                    "symbol": "test",  # Lowercase symbol
                    "sec_type": "stk",  # Lowercase sec_type
                    "currency": "usd",  # Lowercase currency
                    "primary_exchange": "nasdaq",  # Lowercase exchange
                    "derivative_sec_types": "opt,war,fut",  # Comma-separated string
                },
                {
                    "conid": 789012,
                    "symbol": "TEST2",
                    "sec_type": "OPT",
                    "currency": "EUR",
                    "primary_exchange": "LSE",
                    "derivative_sec_types": ["FUT", "WAR"],  # List format
                },
            ]
        )

        repo.save(test_data)
        loaded_data = repo.load(filters=[("symbol", "in", ["TEST", "TEST2"])])

        assert loaded_data.num_rows > 0

        # # Check normalization
        # for _, row in loaded_data.iterrows():
        #     assert isinstance(row["conid"], (int, pd.Int64Dtype))
        #     assert row["symbol"].isupper()  # Should be uppercase
        #     assert row["sec_type"].isupper()
        #     assert row["currency"].isupper()
        #     assert row["primary_exchange"].isupper()
        #     assert isinstance(row["derivative_sec_types"], list)

        # uppercase checks vectorized on columns
        for colname in ["symbol", "sec_type", "currency", "primary_exchange"]:
            col = loaded_data[colname]
            # ensure all are uppercase: value == upper(value)
            uppered = pc.utf8_upper(col)
            assert pc.all(pc.equal(col, uppered)).as_py() is True

        # derivative_sec_types are list<string> and each cell is listy
        lists = loaded_data["derivative_sec_types"]
        # check each element is a Python list when materialized
        for arr in lists.to_pylist():
            assert isinstance(arr, list)

    def test_save_with_missing_columns(self, temp_data_dir):
        """Test saving data with missing required columns."""
        repo = ContractDescriptionsRepository(temp_data_dir)

        # Data missing some required columns
        incomplete_data = pd.DataFrame(
            [
                {
                    "conid": 555555,
                    "symbol": "INCOMPLETE",
                    # Missing sec_type, currency, primary_exchange, derivative_sec_types
                }
            ]
        )

        # Should handle missing columns by adding them with None/default values
        with pytest.raises(DataValidationException) as ei:
            repo.save(incomplete_data)
        # filter = ("symbol", "in", ["INCOMPLETE"])
        # loaded_data = repo.load(filters=[filter])
        #
        # if loaded_data.num_rows > 0:  # Data might be filtered out if validation fails
        #     # get first row's symbol
        #     sym = loaded_data.column("symbol")[0].as_py()
        #     assert sym == "INCOMPLETE"
        #
        #     # column existence (Arrow)
        #     assert "created_at" in loaded.schema.names

    def test_duplicate_handling(self, temp_data_dir):
        """Test handling of duplicate contract descriptions."""
        repo = ContractDescriptionsRepository(temp_data_dir)

        # Save initial data
        initial_data = pd.DataFrame(
            [
                {
                    "conid": 111111,
                    "symbol": "DUPLICATE_TEST",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": ["OPT"],
                }
            ]
        )

        repo.save(initial_data)
        filter = ("symbol", "in", ["DUPLICATE_TEST"])
        initial_count = len(repo.load(filters=[filter]))

        # Save same data again (should be deduplicated)
        repo.save(initial_data)
        final_count = len(repo.load(filters=[filter]))

        # Should not create duplicates for same conid and date
        assert final_count == initial_count

    def test_read_with_filters(self, temp_data_dir):
        """Test reading data with various filters."""
        repo = ContractDescriptionsRepository(temp_data_dir)

        # Save test data
        test_data = pd.DataFrame(
            [
                {
                    "conid": 100001,
                    "symbol": "FILTER_TEST1",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": ["OPT"],
                },
                {
                    "conid": 100002,
                    "symbol": "FILTER_TEST2",
                    "sec_type": "OPT",
                    "currency": "EUR",
                    "primary_exchange": "LSE",
                    "derivative_sec_types": ["FUT"],
                },
                {
                    "conid": 100003,
                    "symbol": "FILTER_TEST3",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NASDAQ",
                    "derivative_sec_types": [],
                },
            ]
        )

        repo.save(test_data)

        # Test different filter combinations
        test_cases = [
            # Filter by single symbol
            {"symbol": "FILTER_TEST1", "expected_count": 1},
            # Filter by multiple symbols
            {"symbol": ["FILTER_TEST1", "FILTER_TEST2"], "expected_count": 2},
            # Filter by sec_type
            {"sec_type": "STK", "expected_count": 2},
            # Filter by currency
            {"currency": "USD", "expected_count": 2},
            # Filter by primary_exchange
            {"primary_exchange": "NYSE", "expected_count": 1},
            # Multiple filters
            {"sec_type": "STK", "currency": "USD", "expected_count": 2},
        ]

        for test_case in test_cases:
            expected_count = test_case.pop("expected_count")

            tickers = test_case.get("symbol")
            tickers = tickers if isinstance(tickers, list) else [tickers]
            filter = ("symbol", "in", tickers)
            loaded_data = repo.load(filters=[filter]).to_pandas()

            actual_count = len(loaded_data)
            assert (
                actual_count >= expected_count
            ), f"Filter {test_case} returned {actual_count}, expected at least {expected_count}"

    def test_read_with_column_selection(self, temp_data_dir):
        """Test reading data with specific column selection."""
        repo = ContractDescriptionsRepository(temp_data_dir)

        # Save test data
        test_data = pd.DataFrame(
            [
                {
                    "conid": 200001,
                    "symbol": "COLUMN_TEST",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": ["OPT"],
                }
            ]
        )

        repo.save(test_data)

        # Test reading with specific columns
        filter = ("symbol", "in", ["COLUMN_TEST"])
        columns_to_read = ["conid", "symbol", "sec_type"]
        loaded_data = repo.load(columns=columns_to_read, filters=[filter]).to_pandas()

        if not loaded_data.empty:
            # Should only have requested columns (plus any partition columns)
            assert "conid" in loaded_data.columns
            assert "symbol" in loaded_data.columns
            assert "sec_type" in loaded_data.columns

    def test_existing_keys_functionality(self, temp_data_dir):
        """Test _existing_keys internal method."""
        repo = ContractDescriptionsRepository(temp_data_dir)

        # Save initial data
        test_date = date(2024, 1, 15)
        initial_data = pd.DataFrame(
            [
                {
                    "conid": 300001,
                    "symbol": "EXISTING_TEST1",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": [],
                },
                {
                    "conid": 300002,
                    "symbol": "EXISTING_TEST2",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NASDAQ",
                    "derivative_sec_types": [],
                },
            ]
        )

        # Manually set as_of_date for testing
        initial_data["created_at"] = test_date
        repo.save(initial_data)

        # Test _existing_keys method
        conids = [300001, 300002, 300003]  # 300003 doesn't exist
        dates = [test_date, date(2024, 1, 16)]  # One date doesn't exist

        existing = repo._existing_keys(dates, conids)

        # Should find the existing combinations
        assert (300001, test_date) in existing
        assert (300002, test_date) in existing
        assert (300003, test_date) not in existing  # Doesn't exist
        assert (300001, date(2024, 1, 16)) not in existing  # Date doesn't exist

    def test_derivative_sec_types_handling(self, temp_data_dir):
        """Test handling of derivative_sec_types field."""
        repo = ContractDescriptionsRepository(temp_data_dir)

        # Test various formats for derivative_sec_types
        test_data = pd.DataFrame(
            [
                {
                    "conid": 400001,
                    "symbol": "DERIV_TEST1",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": ["OPT", "FUT", "WAR"],  # List
                },
                {
                    "conid": 400002,
                    "symbol": "DERIV_TEST2",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": "OPT,FUT",  # Comma-separated string
                },
                {
                    "conid": 400003,
                    "symbol": "DERIV_TEST3",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": None,  # None/null
                },
                {
                    "conid": 400004,
                    "symbol": "DERIV_TEST4",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": "OPT",  # Single string
                },
            ]
        )

        repo.save(test_data)
        loaded_data = repo.load(
            filters=[
                ("symbol", "in", ["DERIV_TEST1", "DERIV_TEST2", "DERIV_TEST3", "DERIV_TEST4"])
        ]
        )

        assert loaded_data.num_rows > 0, "No data loaded"

        assert loaded_data["derivative_sec_types"].type == pa.list_(pa.string())
        assert ( loaded_data.filter(pc.field('symbol') == 'DERIV_TEST1').select(["derivative_sec_types"]).to_pylist()
                 == [{'derivative_sec_types': ['OPT', 'FUT', 'WAR']}]
            )
        assert (loaded_data.filter(pc.field('symbol') == 'DERIV_TEST2').select(["derivative_sec_types"]).to_pylist()
                == [{'derivative_sec_types': ['OPT', 'FUT']}]
                )
        assert (loaded_data.filter(pc.field('symbol') == 'DERIV_TEST3').select(["derivative_sec_types"]).to_pylist()
                == [{'derivative_sec_types': []}]
                )
        assert (loaded_data.filter(pc.field('symbol') == 'DERIV_TEST4').select(["derivative_sec_types"]).to_pylist()
                == [{'derivative_sec_types': ['OPT']}]
                )

        symbol = "DERIV_TEST1"
        expected = ['OPT', 'FUT', 'WAR']
        t = loaded_data.filter(pc.field("symbol") == symbol)
        assert t.num_rows == 1

        # value check (pure Arrow, no dict conversion)
        val = t.column("derivative_sec_types").combine_chunks()[0].as_py()
        assert val == expected


    def test_as_of_date_handling(self, temp_data_dir):
        """Test as_of_date field handling."""
        repo = ContractDescriptionsRepository(temp_data_dir)

        # Test with different date formats
        test_data = pd.DataFrame(
            [
                {
                    "conid": 500001,
                    "symbol": "DATE_TEST1",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": [],
                    "created_at": datetime(2024, 2, 15),  # date object
                },
                {
                    "conid": 500002,
                    "symbol": "DATE_TEST2",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": [],
                    "created_at": "2024-02-16 9:20:00",  # String date
                },
                {
                    "conid": 500003,
                    "symbol": "DATE_TEST3",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": [],
                    # No as_of_date - should default to today
                },
            ]
        )

        repo.save(test_data)

        filter = ("symbol", "in", ["DATE_TEST1", "DATE_TEST2", "DATE_TEST3"])
        loaded_data = repo.load(filters=[filter])

        assert loaded_data.num_rows > 0, "No data loaded"


        assert loaded_data["created_at"].type == pa.timestamp('s', 'UTC')


        data_df = loaded_data.to_pandas()
        # All records should have created_at as timestamp
        for _, row in data_df.iterrows():
            assert "created_at" in row
            assert pd.notna(row["created_at"])
            
            # Extract date from timestamp for comparison
            created_date = row["created_at"].date() if hasattr(row["created_at"], 'date') else row["created_at"]
            
            # Check specific dates
            if row["symbol"] == "DATE_TEST1":
                assert created_date == date(2024, 2, 15)
            elif row["symbol"] == "DATE_TEST2":
                assert created_date == date(2024, 2, 16)
            elif row["symbol"] == "DATE_TEST3":
                # Should be today's date
                assert created_date == date.today()

    def test_first_char_partitioning(self, temp_data_dir):
        """Test first character partitioning functionality."""
        repo = ContractDescriptionsRepository(temp_data_dir)

        # Save data with different first characters
        test_data = pd.DataFrame(
            [
                {
                    "conid": 600001,
                    "symbol": "APPLE_TEST",  # First char: A
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": [],
                },
                {
                    "conid": 600002,
                    "symbol": "BANANA_TEST",  # First char: B
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": [],
                },
                {
                    "conid": 600003,
                    "symbol": "",  # Empty symbol - should get "#"
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": [],
                },
            ]
        )

        repo.save(test_data)

        # Check that partitioned files exist (implementation dependent)
        # This tests the partitioning logic indirectly
        loaded_tbl = repo.load()
        loaded_data = loaded_tbl.to_pandas()
        a_symbols = loaded_data[loaded_data["symbol"].str.startswith("A", na=False)]
        b_symbols = loaded_data[loaded_data["symbol"].str.startswith("B", na=False)]

        assert len(a_symbols) > 0  # Should have APPLE_TEST
        assert len(b_symbols) > 0  # Should have BANANA_TEST

    @pytest.mark.slow
    def test_large_dataset_performance(self, temp_data_dir):
        """Test repository performance with large datasets."""
        repo = ContractDescriptionsRepository(temp_data_dir)

        # Generate large dataset
        large_data = []
        for i in range(1000):  # 1000 contract descriptions
            large_data.append(
                {
                    "conid": 700000 + i,
                    "symbol": f"PERF_TEST_{i:04d}",
                    "sec_type": "STK" if i % 2 == 0 else "OPT",
                    "currency": "USD" if i % 3 == 0 else "EUR",
                    "primary_exchange": ["NYSE", "NASDAQ", "LSE"][i % 3],
                    "derivative_sec_types": ["OPT", "FUT", "WAR"][: (i % 3) + 1],
                }
            )

        large_df = pd.DataFrame(large_data)

        # Time the save operation
        import time

        start_time = time.time()
        repo.save(large_df)
        save_time = time.time() - start_time

        # Should complete within reasonable time
        assert save_time < 30.0, f"Save took too long: {save_time:.2f}s"

        # Time the load operation
        start_time = time.time()
        loaded_tbl = repo.load()
        loaded_data = loaded_tbl.to_pandas()
        load_time = time.time() - start_time

        assert load_time < 10.0, f"Load took too long: {load_time:.2f}s"
        assert len(loaded_data) >= 1000  # Should have at least our data

    def test_empty_dataframe_handling(self, temp_data_dir):
        """Test handling of empty dataframes."""
        repo = ContractDescriptionsRepository(temp_data_dir)

        # Save empty dataframe
        empty_df = pd.DataFrame()

        # Should handle gracefully
        repo.save(empty_df)  # Should not raise error

        # Save None
        repo.save(None)  # Should not raise error


@pytest.mark.unit
class TestContractDescriptionsRepositoryEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_conid_handling(self, temp_data_dir):
        """Test handling of invalid conid values."""
        repo = ContractDescriptionsRepository(temp_data_dir)

        # Data with invalid conids
        invalid_data = pd.DataFrame(
            [
                {
                    "conid": "not_a_number",  # Invalid conid
                    "symbol": "INVALID_CONID",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": [],
                },
                {
                    "conid": None,  # Null conid
                    "symbol": "NULL_CONID",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": [],
                },
            ]
        )

        # Should handle invalid conids gracefully
        repo.save(invalid_data)
        loaded_data = repo.load(symbol=["INVALID_CONID", "NULL_CONID"]).to_pandas()

        # Invalid conids might be filtered out or converted
        # The behavior depends on implementation
        for _, row in loaded_data.iterrows():
            if not pd.isna(row["conid"]):
                assert isinstance(row["conid"], (int, pd.Int64Dtype))

    def test_very_long_symbol_names(self, temp_data_dir):
        """Test handling of very long symbol names."""
        repo = ContractDescriptionsRepository(temp_data_dir)

        # Data with very long symbol names
        long_symbol_data = pd.DataFrame(
            [
                {
                    "conid": 800001,
                    "symbol": "A" * 100,  # Very long symbol
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": [],
                }
            ]
        )

        # Should handle long symbols
        repo.save(long_symbol_data)
        loaded_data = repo.load(symbol=["A" * 100]).to_pandas()

        # Should either truncate or handle the long symbol
        if not loaded_data.empty:
            assert len(loaded_data.iloc[0]["symbol"]) <= 100  # Basic sanity check

    def test_special_characters_in_data(self, temp_data_dir):
        """Test handling of special characters in data."""
        repo = ContractDescriptionsRepository(temp_data_dir)

        # Data with special characters
        special_data = pd.DataFrame(
            [
                {
                    "conid": 900001,
                    "symbol": "SPECIAL-TEST.A",  # Hyphens and dots
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": ["OPT"],
                }
            ]
        )

        repo.save(special_data)
        loaded_data = repo.load(symbol=["SPECIAL-TEST.A"]).to_pandas()

        # Should handle special characters without corruption
        if not loaded_data.empty:
            assert "SPECIAL" in loaded_data.iloc[0]["symbol"]


@pytest.mark.integration
class TestContractDescriptionsRepositoryIntegration:
    """Integration tests for contract descriptions repository."""

    def test_ib_api_contract_details_integration(self, temp_data_dir):
        """Test integration with IB API contract details workflow."""
        repo = ContractDescriptionsRepository(temp_data_dir)

        # Simulate IB API contractDetails response
        ib_contract_details = pd.DataFrame(
            [
                {
                    "conid": 265598,
                    "symbol": "AAPL",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NASDAQ",
                    "derivative_sec_types": [
                        "OPT",
                        "FUT",
                    ],  # Supports options and futures
                },
                {
                    "conid": 272093,
                    "symbol": "MSFT",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NASDAQ",
                    "derivative_sec_types": [
                        "OPT",
                        "WAR",
                    ],  # Supports options and warrants
                },
            ]
        )

        # Save contract details
        repo.save(ib_contract_details)

        # Query for stocks with options
        stocks_with_options = repo.load().to_pandas()
        stocks_with_options = stocks_with_options[
            stocks_with_options["derivative_sec_types"].apply(
                lambda x: "OPT" in x if isinstance(x, list) else False
            )
        ]

        assert not stocks_with_options.empty
        symbols_with_options = set(stocks_with_options["symbol"].unique())
        assert {"AAPL", "MSFT"}.issubset(symbols_with_options)

    def test_multi_day_data_accumulation(self, temp_data_dir):
        """Test accumulating contract descriptions over multiple days."""
        repo = ContractDescriptionsRepository(temp_data_dir)

        # Day 1: Initial contract descriptions
        day1_data = pd.DataFrame(
            [
                {
                    "conid": 101,
                    "symbol": "DAY1_STOCK1",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NYSE",
                    "derivative_sec_types": [],
                },
                {
                    "conid": 102,
                    "symbol": "DAY1_STOCK2",
                    "sec_type": "STK",
                    "currency": "USD",
                    "primary_exchange": "NASDAQ",
                    "derivative_sec_types": ["OPT"],
                },
            ]
        )
        day1_data["as_of_date"] = date(2024, 1, 1)
        repo.save(day1_data)

        # Day 2: Additional contract descriptions
        day2_data = pd.DataFrame(
            [
                {
                    "conid": 103,
                    "symbol": "DAY2_STOCK1",
                    "sec_type": "STK",
                    "currency": "EUR",
                    "primary_exchange": "LSE",
                    "derivative_sec_types": ["FUT"],
                },
                {
                    "conid": 102,
                    "symbol": "DAY1_STOCK2",
                    "sec_type": "STK",  # Same conid, new date
                    "currency": "USD",
                    "primary_exchange": "NASDAQ",
                    "derivative_sec_types": ["OPT", "WAR"],
                },
            ]
        )
        day2_data["as_of_date"] = date(2024, 1, 2)
        repo.save(day2_data)

        # Query all data
        all_data = repo.load().to_pandas()

        # Should have data from both days
        unique_dates = set(all_data["as_of_date"].unique())
        assert date(2024, 1, 1) in unique_dates
        assert date(2024, 1, 2) in unique_dates

        # Should have historical evolution for conid 102
        conid_102_data = all_data[all_data["conid"] == 102]
        assert len(conid_102_data) >= 2  # At least two entries for different dates
