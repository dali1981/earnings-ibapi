"""
Basic repository tests to check setup.
"""
import pytest
import pandas as pd
from datetime import date
from pathlib import Path
import tempfile

from ibx_repos.equity_bars import EquityBarRepository


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
            test_data = pd.DataFrame({
                "date": ["20240101", "20240102"],
                "open": [100.0, 101.0],
                "high": [101.5, 102.5],
                "low": [99.5, 100.5],
                "close": [101.0, 102.0],
                "volume": [1000000, 1100000],
            })
            
            # Save data
            repo.save(test_data, "TEST", "1 day", "TRADES")
            
            # Load data
            loaded_data = repo.load(symbol="TEST")
            
            assert not loaded_data.empty
            assert len(loaded_data) == 2
            assert all(loaded_data["symbol"] == "TEST")
    
    def test_empty_load(self):
        """Test loading from empty repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = EquityBarRepository(temp_dir)
            
            # Load from empty repo
            loaded_data = repo.load(symbol="NONEXISTENT")
            
            assert loaded_data.empty