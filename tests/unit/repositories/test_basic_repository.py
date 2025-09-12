"""
Basic repository tests to check setup.
"""

from common.repositories.equity_bars import EquityBarRepository
from datetime import date
from datetime import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
import tempfile


@pytest.mark.unit
@pytest.mark.data
class TestBasicEquityBarRepository:
    """Basic test cases for EquityBarRepository."""

    def test_repository_initialization(self):
        """Test repository initialization creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "test_equity_bars"
            repo = EquityBarRepository(repo_path)

            assert repo.base_path.exists()
            assert repo.base_path == repo_path
            assert repo.schema is not None

    def test_save_and_load_basic_data(self):
        """Test basic save and load functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = EquityBarRepository(temp_dir)

            # Sample data
            test_data = pd.DataFrame(
                {
                    "date": ["20240101", "20240102"],
                    "time": ["09:30:00", "09:30:00"],
                    "open": [100.0, 101.0],
                    "high": [101.5, 102.5],
                    "low": [99.5, 100.5],
                    "close": [101.0, 102.0],
                    "volume": [1000000, 1100000],
                    "symbol": ["TEST", "TEST"],
                    "bar_size": ["1 day", "1 day"],
                    "data_type": ["TRADES", "TRADES"],
                    "created_at": [dt.utcnow(), dt.utcnow()],
                }
            )

            # Save data
            # repo.save(test_data, "TEST", "1 day", "TRADES")
            repo.save(test_data)

            # Load data
            loaded_table = repo.load(symbol="TEST")
            df = loaded_table.to_pandas()

            assert df is not None and not df.empty
            assert len(df) == 2
            assert all(df["symbol"] == "TEST")

    def test_empty_load(self):
        """Test loading from empty repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = EquityBarRepository(temp_dir)

            # Load from empty repo
            loaded_table = repo.load(symbol="NONEXISTENT")

            assert loaded_table.num_rows == 0
