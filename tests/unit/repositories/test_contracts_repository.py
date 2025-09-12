"""
Unit tests for ContractRepository.
"""

from common.repositories.contracts import ContractRepository
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import tempfile


def create_sample_contract(
    conid=265598, symbol="AAPL", sec_type="STK", currency="USD", **kwargs
):
    """Helper to create sample contract data with all required fields."""
    defaults = {
        "conid": conid,
        "symbol": symbol,
        "sec_type": sec_type,
        "currency": currency,
        "primary_exchange": "NASDAQ",
        "exchange": "SMART",
        "local_symbol": symbol,
        "trading_class": "NMS",
        "long_name": f"{symbol} Inc",
        "category": "Technology",
        "sub_category": "Software",
        "industry": "Technology",
        "time_zone": "EST",
        "is_us_listing": True,
    }
    defaults.update(kwargs)
    return defaults


@pytest.mark.unit
@pytest.mark.data
class TestContractRepository:
    """Test cases for ContractRepository."""

    def test_repository_initialization(self, temp_data_dir):
        """Test repository initialization creates necessary directories."""
        repo_path = temp_data_dir / "test_contracts"
        repo = ContractRepository(repo_path)

        assert repo.base_path.exists()
        assert repo.base_path == repo_path
        assert repo.base == repo_path  # Test alias

    def test_contracts_schema(self):
        """Test that the contracts schema is properly defined."""
        schema = ContractRepository._contracts_schema()

        expected_fields = {
            "conid",
            "symbol",
            "sec_type",
            "currency",
            "primary_exchange",
            "exchange",
            "local_symbol",
            "trading_class",
            "long_name",
            "category",
            "sub_category",
            "industry",
            "time_zone",
            "is_us_listing",
            "sym_prefix",
        }

        schema_fields = {field.name for field in schema}
        assert schema_fields == expected_fields

    def test_save_and_read_basic_contracts(self, temp_data_dir):
        """Test saving and reading basic contract data."""
        repo = ContractRepository(temp_data_dir / "contracts")

        # Create sample contract data using helper
        contracts_df = pd.DataFrame(
            [
                create_sample_contract(265598, "AAPL"),
                create_sample_contract(272093, "MSFT"),
            ]
        )

        # Save contracts
        repo.save(contracts_df)

        # Read all contracts
        loaded_tbl = repo.read()
        loaded = loaded_tbl.to_pandas()
        assert not loaded.empty
        assert len(loaded) == 2
        assert set(loaded["symbol"]) == {"AAPL", "MSFT"}
        assert all(loaded["sec_type"] == "STK")

    def test_save_missing_required_columns(self, temp_data_dir):
        """Test that saving with missing required columns raises error."""
        repo = ContractRepository(temp_data_dir / "contracts")

        # Missing required columns (no 'currency')
        incomplete_df = pd.DataFrame(
            [
                {
                    "conid": 265598,
                    "symbol": "AAPL",
                    "sec_type": "STK",
                    # Missing 'currency' - should raise error
                }
            ]
        )

        with pytest.raises(ValueError, match="missing columns"):
            repo.save(incomplete_df)

    def test_save_empty_dataframe(self, temp_data_dir):
        """Test saving empty dataframe is handled gracefully."""
        repo = ContractRepository(temp_data_dir / "contracts")

        empty_df = pd.DataFrame()
        repo.save(empty_df)  # Should not raise error

        loaded_tbl = repo.read()
        assert loaded_tbl.num_rows == 0

    def test_save_none_dataframe(self, temp_data_dir):
        """Test saving None dataframe is handled gracefully."""
        repo = ContractRepository(temp_data_dir / "contracts")

        repo.save(None)  # Should not raise error

        loaded_tbl = repo.read()
        assert loaded_tbl.num_rows == 0

    def test_read_with_symbol_filter(self, temp_data_dir):
        """Test reading contracts filtered by symbol."""
        repo = ContractRepository(temp_data_dir / "contracts")

        contracts_df = pd.DataFrame(
            [
                create_sample_contract(265598, "AAPL"),
                create_sample_contract(272093, "MSFT"),
                create_sample_contract(12345, "GOOGL"),
            ]
        )

        repo.save(contracts_df)

        # Filter by single symbol
        aapl_only = repo.read(symbol="AAPL").to_pandas()
        assert len(aapl_only) == 1
        assert aapl_only.iloc[0]["symbol"] == "AAPL"

        # Filter by multiple symbols
        multi_symbols = repo.read(symbol=["AAPL", "MSFT"]).to_pandas()
        assert len(multi_symbols) == 2
        assert set(multi_symbols["symbol"]) == {"AAPL", "MSFT"}

    def test_read_with_sec_type_filter(self, temp_data_dir):
        """Test reading contracts filtered by security type."""
        repo = ContractRepository(temp_data_dir / "contracts")

        contracts_df = pd.DataFrame(
            [
                create_sample_contract(265598, "AAPL", "STK"),
                create_sample_contract(456789, "ES", "FUT", primary_exchange="CME"),
                create_sample_contract(
                    789123,
                    "EURUSD",
                    "CASH",
                    primary_exchange="IDEALPRO",
                    is_us_listing=False,
                ),
            ]
        )

        repo.save(contracts_df)

        # Filter by stock type
        stocks = repo.read(sec_type="STK").to_pandas()
        assert len(stocks) == 1
        assert stocks.iloc[0]["symbol"] == "AAPL"

        # Filter by futures
        futures = repo.read(sec_type="FUT").to_pandas()
        assert len(futures) == 1
        assert futures.iloc[0]["symbol"] == "ES"

    def test_read_with_currency_filter(self, temp_data_dir):
        """Test reading contracts filtered by currency."""
        repo = ContractRepository(temp_data_dir / "contracts")

        contracts_df = pd.DataFrame(
            [
                create_sample_contract(265598, "AAPL", currency="USD"),
                create_sample_contract(
                    12345,
                    "BP",
                    currency="GBP",
                    primary_exchange="LSE",
                    is_us_listing=False,
                ),
            ]
        )

        repo.save(contracts_df)

        # Filter by USD currency
        usd_contracts = repo.read(currency="USD").to_pandas()
        assert len(usd_contracts) == 1
        assert usd_contracts.iloc[0]["symbol"] == "AAPL"

        # Filter by GBP currency
        gbp_contracts = repo.read(currency="GBP").to_pandas()
        assert len(gbp_contracts) == 1
        assert gbp_contracts.iloc[0]["symbol"] == "BP"

    def test_read_with_column_selection(self, temp_data_dir):
        """Test reading contracts with specific column selection."""
        repo = ContractRepository(temp_data_dir / "contracts")

        contracts_df = pd.DataFrame([create_sample_contract(265598, "AAPL")])

        repo.save(contracts_df)

        # Read only specific columns
        selected_tbl = repo.read(columns=["symbol", "sec_type", "currency"])
        selected = selected_tbl.to_pandas()
        assert len(selected.columns) == 3
        assert "symbol" in selected.columns
        assert "sec_type" in selected.columns
        assert "currency" in selected.columns
        assert "long_name" not in selected.columns

    def test_read_empty_repository(self, temp_data_dir):
        """Test reading from empty repository returns empty dataframe."""
        repo = ContractRepository(temp_data_dir / "contracts")

        # Read from empty repo
        result = repo.read()
        assert result.empty

        # Read with filters from empty repo
        filtered = repo.read(symbol="AAPL")
        assert filtered.empty

        # Read with column selection from empty repo
        selected = repo.read(columns=["symbol", "sec_type"])
        assert selected.empty
        assert list(selected.columns) == ["symbol", "sec_type"]

    def test_load_method(self, temp_data_dir):
        """Test the load method (legacy method for backward compatibility)."""
        repo = ContractRepository(temp_data_dir / "contracts")

        contracts_df = pd.DataFrame(
            [
                create_sample_contract(265598, "AAPL", "STK"),
                create_sample_contract(456789, "ES", "FUT", primary_exchange="CME"),
            ]
        )

        repo.save(contracts_df)

        # Test load method with sec_type filter
        stocks = repo.load(sec_type="STK")
        assert len(stocks) == 1
        assert stocks.iloc[0]["symbol"] == "AAPL"

        # Test load method with primary_exchange filter
        cme_contracts = repo.load(primary_exchange="CME")
        assert len(cme_contracts) == 1
        assert cme_contracts.iloc[0]["symbol"] == "ES"

    def test_present_conids(self, temp_data_dir):
        """Test getting present contract IDs."""
        repo = ContractRepository(temp_data_dir / "contracts")

        # Initially empty
        conids = repo.present_conids()
        assert len(conids) == 0

        contracts_df = pd.DataFrame(
            [
                create_sample_contract(265598, "AAPL"),
                create_sample_contract(272093, "MSFT"),
            ]
        )

        repo.save(contracts_df)

        # Check present contract IDs
        conids = repo.present_conids()
        assert conids == {265598, 272093}

    def test_symbol_prefix_partitioning(self, temp_data_dir):
        """Test that symbol prefix partitioning works correctly."""
        repo = ContractRepository(temp_data_dir / "contracts")

        contracts_df = pd.DataFrame(
            [
                create_sample_contract(265598, "AAPL"),  # Should get 'A' prefix
                create_sample_contract(272093, "MSFT"),  # Should get 'M' prefix
                create_sample_contract(
                    12345, "1ABC"
                ),  # Should get '#' prefix (starts with number)
            ]
        )

        repo.save(contracts_df)

        loaded = repo.read()

        # Verify prefix assignment
        aapl_row = loaded[loaded["symbol"] == "AAPL"].iloc[0]
        assert aapl_row["sym_prefix"] == "A"

        msft_row = loaded[loaded["symbol"] == "MSFT"].iloc[0]
        assert msft_row["sym_prefix"] == "M"

        numeric_row = loaded[loaded["symbol"] == "1ABC"].iloc[0]
        assert numeric_row["sym_prefix"] == "#"

    def test_deduplication_by_conid(self, temp_data_dir):
        """Test that duplicate contract IDs are deduplicated."""
        repo = ContractRepository(temp_data_dir / "contracts")

        # Same conid with different data
        contracts_df = pd.DataFrame(
            [
                create_sample_contract(265598, "AAPL", long_name="Apple Inc"),
                create_sample_contract(
                    265598, "AAPL", long_name="Apple Inc - Updated"
                ),  # Duplicate conid
            ]
        )

        repo.save(contracts_df)

        # Should only have one record
        loaded = repo.read()
        assert len(loaded) == 1
        assert loaded.iloc[0]["conid"] == 265598

    def test_mixed_security_types(self, temp_data_dir):
        """Test handling mixed security types."""
        repo = ContractRepository(temp_data_dir / "contracts")

        contracts_df = pd.DataFrame(
            [
                create_sample_contract(265598, "AAPL", "STK"),
                create_sample_contract(456789, "AAPL", "OPT"),
                create_sample_contract(789123, "ES", "FUT", primary_exchange="CME"),
            ]
        )

        repo.save(contracts_df)

        loaded = repo.read()
        assert len(loaded) == 3

        sec_types = set(loaded["sec_type"])
        assert sec_types == {"STK", "OPT", "FUT"}

    def test_international_contracts(self, temp_data_dir):
        """Test handling international (non-US) contracts."""
        repo = ContractRepository(temp_data_dir / "contracts")

        contracts_df = pd.DataFrame(
            [
                create_sample_contract(
                    12345,
                    "BP",
                    currency="GBP",
                    primary_exchange="LSE",
                    is_us_listing=False,
                ),
                create_sample_contract(
                    67890,
                    "SAP",
                    currency="EUR",
                    primary_exchange="XETRA",
                    is_us_listing=False,
                ),
            ]
        )

        repo.save(contracts_df)

        # Filter by non-US listings
        intl_contracts = repo.read(is_us_listing=False)
        assert len(intl_contracts) == 2
        assert set(intl_contracts["currency"]) == {"GBP", "EUR"}

    def test_comprehensive_contract_data(self, temp_data_dir):
        """Test with comprehensive contract data including all fields."""
        repo = ContractRepository(temp_data_dir / "contracts")

        comprehensive_df = pd.DataFrame(
            [
                create_sample_contract(
                    265598,
                    "AAPL",
                    long_name="Apple Inc",
                    category="Technology",
                    sub_category="Consumer Electronics",
                    industry="Technology Hardware",
                    time_zone="EST",
                )
            ]
        )

        repo.save(comprehensive_df)

        loaded = repo.read()
        assert len(loaded) == 1

        record = loaded.iloc[0]
        assert record["conid"] == 265598
        assert record["symbol"] == "AAPL"
        assert record["long_name"] == "Apple Inc"
        assert record["category"] == "Technology"
        assert record["industry"] == "Technology Hardware"
        assert record["time_zone"] == "EST"
        assert record["is_us_listing"] == True

    def test_multiple_operations(self, temp_data_dir):
        """Test multiple save and read operations work correctly."""
        repo = ContractRepository(temp_data_dir / "contracts")

        # First batch
        batch1 = pd.DataFrame(
            [
                create_sample_contract(265598, "AAPL"),
                create_sample_contract(272093, "MSFT"),
            ]
        )
        repo.save(batch1)

        # Second batch
        batch2 = pd.DataFrame(
            [
                create_sample_contract(12345, "GOOGL"),
                create_sample_contract(67890, "AMZN"),
            ]
        )
        repo.save(batch2)

        # Should have all contracts
        all_contracts = repo.read()
        assert len(all_contracts) == 4
        assert set(all_contracts["symbol"]) == {"AAPL", "MSFT", "GOOGL", "AMZN"}

    def test_filtering_combinations(self, temp_data_dir):
        """Test combinations of filters work correctly."""
        repo = ContractRepository(temp_data_dir / "contracts")

        contracts_df = pd.DataFrame(
            [
                create_sample_contract(265598, "AAPL", "STK", "USD"),
                create_sample_contract(272093, "MSFT", "STK", "USD"),
                create_sample_contract(
                    12345,
                    "BP",
                    "STK",
                    "GBP",
                    primary_exchange="LSE",
                    is_us_listing=False,
                ),
                create_sample_contract(
                    67890, "ES", "FUT", "USD", primary_exchange="CME"
                ),
            ]
        )

        repo.save(contracts_df)

        # Filter by multiple criteria
        usd_stocks = repo.read(sec_type="STK", currency="USD")
        assert len(usd_stocks) == 2
        assert set(usd_stocks["symbol"]) == {"AAPL", "MSFT"}

        # Filter by US listings and stock type
        us_stocks = repo.read(sec_type="STK", is_us_listing=True)
        assert len(us_stocks) == 2
        assert set(us_stocks["symbol"]) == {"AAPL", "MSFT"}
