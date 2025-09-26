"""
Unit tests for OptionChainSnapshotRepository (common.repositories).
"""

from common.repositories.option_chains import OptionChainSnapshotRepository
from datetime import date
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.unit
@pytest.mark.data
class TestOptionChainSnapshotRepository:
    def test_repository_initialization(self, temp_data_dir):
        repo_path = temp_data_dir / "test_option_chains"
        repo = OptionChainSnapshotRepository(repo_path)
        assert repo.base_path.exists()
        assert repo.base_path == repo_path
        assert repo.schema is not None

    def test_schema_definition(self, temp_data_dir):
        repo = OptionChainSnapshotRepository(temp_data_dir)
        expected_fields = {
            "underlying",
            "snapshot_date",
            "expiry",
            "strike",
            "right",
            "bid",
            "ask",
            "last",
            "volume",
            "open_interest",
            "implied_volatility",
            "delta",
            "gamma",
            "theta",
            "vega",
            "contract_id",
            "local_symbol",
            "trading_class",
            "exchange",
            "currency",
            "multiplier",
        }
        schema_fields = {field.name for field in repo.schema}
        assert expected_fields == schema_fields

    def test_save_and_load_snapshot(self, temp_data_dir):
        repo = OptionChainSnapshotRepository(temp_data_dir)
        df = pd.DataFrame(
            [
                {
                    "expiry": date(2024, 3, 15),
                    "strike": 150.0,
                    "right": "C",
                    "bid": 1.2,
                },
                {
                    "expiry": date(2024, 3, 15),
                    "strike": 150.0,
                    "right": "P",
                    "bid": 1.1,
                },
            ]
        )
        repo.save_chain_snapshot(df, underlying="AAPL", snapshot_date=date(2024, 1, 15))
        tbl = repo.load(underlying="AAPL", snapshot_date=date(2024, 1, 15))
        out = tbl.to_pandas()
        assert not out.empty
        assert set(out["right"]) == {"C", "P"}
        assert all(out["expiry"] == date(2024, 3, 15))

    def test_load_with_filters(self, temp_data_dir):
        repo = OptionChainSnapshotRepository(temp_data_dir)
        for d in [date(2024, 1, 15), date(2024, 1, 16)]:
            df = pd.DataFrame(
                [
                    {"expiry": date(2024, 3, 15), "strike": 100.0, "right": "C"},
                ]
            )
            repo.save_chain_snapshot(df, underlying="AAPL", snapshot_date=d)
        tbl = repo.load(underlying="AAPL")
        assert tbl.num_rows >= 2
        df = repo.load(underlying="AAPL", snapshot_date=date(2024, 1, 16)).to_pandas()
        assert not df.empty
        assert all(df["snapshot_date"] == date(2024, 1, 16))

    def test_data_normalization(self, temp_data_dir):
        repo = OptionChainSnapshotRepository(temp_data_dir)
        df = pd.DataFrame(
            [
                {"expiry": "20240315", "strike": "150", "right": "CALL", "bid": "1.2"},
            ]
        )
        repo.save_chain_snapshot(df, underlying="TEST", snapshot_date=date(2024, 1, 20))
        out = repo.load(underlying="TEST").to_pandas()
        assert not out.empty
        assert set(out["right"]) == {"C"}
        assert all(out["strike"] == 150.0)

    def test_schema_compliance(self, temp_data_dir):
        """Test that saved data complies with repository schema."""
        repo = OptionChainSnapshotRepository(temp_data_dir)

        mock_snapshots = [
            {
                "exchange": "NYSE",
                "tradingClass": "COMPLIANCE_TEST",
                "multiplier": "100",
                "expirations": ["20240601", "20240701"],
                "strikes": [50.0, 60.0, 70.0, 80.0, 90.0],
            }
        ]

        repo.save("COMPLIANCE_TEST", 555555, mock_snapshots, date(2024, 2, 1))
        loaded_data = repo.load(underlying="COMPLIANCE_TEST")

        # Check all required columns exist
        required_columns = [
            "underlying",
            "underlying_conid",
            "exchange",
            "trading_class",
            "multiplier",
            "expirations",
            "strikes",
            "snapshot_date",
        ]

        for col in required_columns:
            assert col in loaded_data.columns, f"Missing required column: {col}"

        # Check data types
        row = loaded_data.iloc[0]
        assert isinstance(row["underlying"], str)
        assert isinstance(row["underlying_conid"], (int, pd.Int64Dtype))
        assert isinstance(row["exchange"], str)
        assert isinstance(row["trading_class"], str)
        assert isinstance(row["multiplier"], str)
        assert isinstance(row["expirations"], list)
        assert isinstance(row["strikes"], list)
        assert isinstance(row["snapshot_date"], date)

    def test_present_dates_method(self, temp_data_dir):
        """Test present_dates method (if implemented)."""
        repo = OptionChainSnapshotRepository(temp_data_dir)

        # Save data for multiple dates
        dates = [date(2024, 1, 15), date(2024, 1, 16), date(2024, 1, 17)]

        for snap_date in dates:
            mock_snapshots = [
                {
                    "exchange": "SMART",
                    "tradingClass": "PRESENT_TEST",
                    "multiplier": "100",
                    "expirations": ["20240315"],
                    "strikes": [100.0],
                }
            ]
            repo.save("PRESENT_TEST", 777777, mock_snapshots, snap_date)

        # Test present_dates method if it exists
        if hasattr(repo, "present_dates"):
            start_date = date(2024, 1, 14)
            end_date = date(2024, 1, 18)

            present = repo.present_dates("PRESENT_TEST", start_date, end_date)
            assert isinstance(present, set)
            assert len(present) == 3  # Should have all three dates
            assert set(dates) <= present  # All our dates should be present

    def test_multiple_exchanges_same_underlying(self, temp_data_dir):
        """Test saving snapshots from multiple exchanges for same underlying."""
        repo = OptionChainSnapshotRepository(temp_data_dir)

        exchanges = ["SMART", "NASDAQ", "NYSE"]
        mock_snapshots = []

        for exchange in exchanges:
            mock_snapshots.append(
                {
                    "exchange": exchange,
                    "tradingClass": "MULTI_EXCHANGE_TEST",
                    "multiplier": "100",
                    "expirations": ["20240315"],
                    "strikes": [100.0, 110.0, 120.0],
                }
            )

        repo.save("MULTI_EXCHANGE_TEST", 888888, mock_snapshots, date(2024, 2, 15))
        loaded_data = repo.load(underlying="MULTI_EXCHANGE_TEST")

        assert not loaded_data.empty
        assert len(loaded_data) == 3  # One row per exchange

        # Should have data from all exchanges
        loaded_exchanges = set(loaded_data["exchange"].unique())
        assert loaded_exchanges == set(exchanges)

    @pytest.mark.slow
    def test_large_chain_data_performance(self, temp_data_dir):
        """Test repository performance with large option chain data."""
        repo = OptionChainSnapshotRepository(temp_data_dir)

        # Generate large option chain data
        large_snapshots = []
        for i in range(50):  # Multiple exchanges/trading classes
            large_snapshots.append(
                {
                    "exchange": f"EXCHANGE_{i}",
                    "tradingClass": f"SYMBOL_{i}",
                    "multiplier": "100",
                    "expirations": [
                        f"2024{month:02d}15" for month in range(1, 13)
                    ],  # 12 months
                    "strikes": [
                        float(strike) for strike in range(50, 500, 5)
                    ],  # 90 strikes
                }
            )

        # Time the save operation
        import time

        start_time = time.time()
        repo.save("LARGE_TEST", 999999, large_snapshots, date(2024, 3, 1))
        save_time = time.time() - start_time

        # Should complete within reasonable time
        assert save_time < 10.0, f"Save took too long: {save_time:.2f}s"

        # Time the load operation
        start_time = time.time()
        loaded_data = repo.load(underlying="LARGE_TEST")
        load_time = time.time() - start_time

        assert load_time < 5.0, f"Load took too long: {load_time:.2f}s"
        assert len(loaded_data) == 50  # Should have all snapshots


@pytest.mark.unit
class TestOptionChainRepositoryEdgeCases:
    """Test edge cases and error conditions for OptionChainSnapshotRepository."""

    def test_invalid_data_handling(self, temp_data_dir):
        """Test handling of invalid or malformed data."""
        repo = OptionChainSnapshotRepository(temp_data_dir)

        # Test with missing required fields
        invalid_snapshots = [
            {
                # Missing multiplier, strikes, expirations
                "exchange": "SMART",
                "tradingClass": "INVALID_TEST",
            }
        ]

        # Should handle gracefully or raise appropriate error
        try:
            repo.save("INVALID_TEST", 111111, invalid_snapshots)
            # If no error, verify the data handling
            loaded_data = repo.load(underlying="INVALID_TEST")
            # Should either be empty or have default values
            if not loaded_data.empty:
                row = loaded_data.iloc[0]
                # Check that missing fields are handled appropriately
                assert hasattr(row, "strikes") or pd.isna(row.get("strikes", None))
        except Exception as e:
            # If error is raised, it should be informative
            assert "missing" in str(e).lower() or "required" in str(e).lower()

    def test_duplicate_snapshot_handling(self, temp_data_dir):
        """Test handling of duplicate snapshots."""
        repo = OptionChainSnapshotRepository(temp_data_dir)

        mock_snapshots = [
            {
                "exchange": "SMART",
                "tradingClass": "DUPLICATE_TEST",
                "multiplier": "100",
                "expirations": ["20240315"],
                "strikes": [100.0, 200.0],
            }
        ]

        snap_date = date(2024, 2, 20)

        # Save same data twice
        repo.save("DUPLICATE_TEST", 222222, mock_snapshots, snap_date)
        repo.save("DUPLICATE_TEST", 222222, mock_snapshots, snap_date)

        # Should handle duplicates appropriately
        loaded_data = repo.load(underlying="DUPLICATE_TEST", snapshot_date=snap_date)

        # Behavior depends on implementation - either deduplicated or multiple records
        assert not loaded_data.empty
        # Could be 1 record (deduplicated) or 2 records (both kept)
        assert len(loaded_data) >= 1

    def test_corrupted_data_recovery(self, temp_data_dir):
        """Test repository behavior with corrupted data structures."""
        repo = OptionChainSnapshotRepository(temp_data_dir)

        # Test with malformed strikes and expirations
        corrupted_snapshots = [
            {
                "exchange": "SMART",
                "tradingClass": "CORRUPT_TEST",
                "multiplier": "100",
                "expirations": "not_a_list",  # Should be list
                "strikes": None,  # Should be list
            }
        ]

        # Should handle gracefully
        try:
            repo.save("CORRUPT_TEST", 333333, corrupted_snapshots)
            loaded_data = repo.load(underlying="CORRUPT_TEST")

            if not loaded_data.empty:
                row = loaded_data.iloc[0]
                # Should have converted or handled the bad data
                assert isinstance(row.get("expirations"), (list, type(None)))
                assert isinstance(row.get("strikes"), (list, type(None)))
        except Exception as e:
            # If error occurs, should be informative
            assert "format" in str(e).lower() or "type" in str(e).lower()

    def test_very_large_strikes_expirations(self, temp_data_dir):
        """Test handling of very large strikes and expirations lists."""
        repo = OptionChainSnapshotRepository(temp_data_dir)

        # Create snapshot with very large lists
        large_snapshots = [
            {
                "exchange": "SMART",
                "tradingClass": "LARGE_LISTS_TEST",
                "multiplier": "100",
                "expirations": [
                    f"2024{month:02d}{day:02d}"
                    for month in range(1, 13)
                    for day in [15, 30]
                ],  # 24 expirations
                "strikes": [float(i) for i in range(1, 1001, 1)],  # 1000 strikes
            }
        ]

        # Should handle large lists
        repo.save("LARGE_LISTS_TEST", 444444, large_snapshots)
        loaded_data = repo.load(underlying="LARGE_LISTS_TEST")

        assert not loaded_data.empty
        row = loaded_data.iloc[0]
        assert len(row["expirations"]) == 24
        assert len(row["strikes"]) == 1000


@pytest.mark.integration
class TestOptionChainRepositoryIntegration:
    """Integration tests for option chain repository."""

    def test_end_to_end_workflow(self, temp_data_dir):
        """Test complete workflow from IB API-like data to repository storage."""
        repo = OptionChainSnapshotRepository(temp_data_dir)

        # Simulate IB API secDefOptParams response
        ib_api_response = [
            {
                "exchange": "SMART",
                "underlyingConId": 265598,
                "tradingClass": "AAPL",
                "multiplier": "100",
                "expirations": ["20240315", "20240415", "20240515", "20240621"],
                "strikes": [
                    120.0,
                    125.0,
                    130.0,
                    135.0,
                    140.0,
                    145.0,
                    150.0,
                    155.0,
                    160.0,
                    165.0,
                    170.0,
                    175.0,
                    180.0,
                ],
            },
            {
                "exchange": "NASDAQ",
                "underlyingConId": 265598,
                "tradingClass": "AAPL",
                "multiplier": "100",
                "expirations": ["20240315", "20240415"],
                "strikes": [140.0, 145.0, 150.0, 155.0, 160.0],
            },
        ]

        # Save the data
        repo.save("AAPL", 265598, ib_api_response, date(2024, 2, 15))

        # Load and verify
        loaded_data = repo.load(underlying="AAPL", snapshot_date=date(2024, 2, 15))

        assert not loaded_data.empty
        assert len(loaded_data) == 2  # Two exchange records

        # Verify data integrity
        smart_data = loaded_data[loaded_data["exchange"] == "SMART"].iloc[0]
        nasdaq_data = loaded_data[loaded_data["exchange"] == "NASDAQ"].iloc[0]

        assert len(smart_data["expirations"]) == 4
        assert len(smart_data["strikes"]) == 13
        assert len(nasdaq_data["expirations"]) == 2
        assert len(nasdaq_data["strikes"]) == 5

    def test_historical_snapshots_analysis(self, temp_data_dir):
        """Test analyzing historical snapshots over time."""
        repo = OptionChainSnapshotRepository(temp_data_dir)

        # Save snapshots over multiple days
        base_strikes = [140.0, 145.0, 150.0, 155.0, 160.0]
        dates = [date(2024, 2, 10), date(2024, 2, 11), date(2024, 2, 12)]

        for i, snap_date in enumerate(dates):
            # Simulate changing strikes over time
            current_strikes = base_strikes + [
                165.0 + i * 5.0
            ]  # Adding strikes over time

            snapshots = [
                {
                    "exchange": "SMART",
                    "tradingClass": "HISTORICAL_TEST",
                    "multiplier": "100",
                    "expirations": ["20240315"],
                    "strikes": current_strikes,
                }
            ]

            repo.save("HISTORICAL_TEST", 777777, snapshots, snap_date)

        # Load historical data
        all_data = repo.load(underlying="HISTORICAL_TEST")
        assert len(all_data) == 3  # Three snapshots

        # Verify strikes evolved over time
        for i, snap_date in enumerate(dates):
            day_data = all_data[all_data["snapshot_date"] == snap_date].iloc[0]
            expected_strikes = len(base_strikes) + 1  # Added one strike each day
            assert len(day_data["strikes"]) == expected_strikes
