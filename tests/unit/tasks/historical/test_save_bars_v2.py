"""
Unit tests for save_bars_v2_task.
"""

from tests.mocks.ib_mock_factory import MockDataGenerator
from unittest.mock import Mock, patch

import pytest


@pytest.mark.unit
class TestSaveBarsV2Task:
    """Tests for save_bars_v2_task."""

    def test_save_bars_success(self):
        """Test successful bars saving."""
        from ops.tasks.historical_data import save_bars_v2_task

        # Mock successful fetch result
        mock_bars = MockDataGenerator.generate_stock_price_series("IBM", num_bars=24)
        fetch_result = {
            "bars": mock_bars,
            "contract": {"symbol": "IBM", "secType": "STK"},
            "parameters": {"bar_size_setting": "1 hour"},
        }

        # Mock the save_historical_bars_task
        mock_save_result = {"saved_count": 24, "symbol": "IBM", "bar_size": "1 hour"}

        with patch(
            "ops.tasks.historical_data.save_historical_bars_task",
            return_value=mock_save_result,
        ) as mock_save:
            with patch("ops.tasks.historical_data.current_task") as mock_current_task:
                # Execute task
                result = save_bars_v2_task(
                    fetch_result=fetch_result, repository_type="equity", root=None
                )

                # Verify results
                assert result == mock_save_result

                # Verify save task was called correctly
                mock_save.assert_called_once_with(
                    mock_current_task,
                    fetch_result=fetch_result,
                    repository_type="equity",
                    root=None,
                )

    def test_save_bars_with_fetch_error(self):
        """Test handling fetch result with error."""
        from ops.tasks.historical_data import save_bars_v2_task

        # Fetch result with error
        fetch_result = {
            "error": "Connection failed",
            "symbol": "IBM",
            "bars": [],
            "contract": {"symbol": "IBM"},
        }

        # Execute task
        result = save_bars_v2_task(fetch_result=fetch_result, repository_type="equity")

        # Verify error handling
        assert result["saved"] == 0
        assert result["skipped"] == 1
        assert "Connection failed" in result["error"]
        assert result["symbol"] == "IBM"

    def test_save_bars_empty_bars(self):
        """Test handling empty bars from fetch."""
        from ops.tasks.historical_data import save_bars_v2_task

        # Fetch result with empty bars but no error
        fetch_result = {
            "bars": [],
            "contract": {"symbol": "IBM"},
            "parameters": {"bar_size_setting": "1 hour"},
        }

        mock_save_result = {"saved_count": 0}

        with patch(
            "ops.tasks.historical_data.save_historical_bars_task",
            return_value=mock_save_result,
        ) as mock_save:
            with patch("ops.tasks.historical_data.current_task") as mock_current_task:
                result = save_bars_v2_task(fetch_result=fetch_result)

                assert result == mock_save_result

                # Verify save task was still called
                mock_save.assert_called_once_with(
                    mock_current_task,
                    fetch_result=fetch_result,
                    repository_type="equity",  # default
                    root=None,  # default
                )

    def test_save_bars_with_custom_repository_type(self):
        """Test save with custom repository type."""
        from ops.tasks.historical_data import save_bars_v2_task

        mock_bars = MockDataGenerator.generate_stock_price_series("SPY", num_bars=10)
        fetch_result = {
            "bars": mock_bars,
            "contract": {"symbol": "SPY"},
            "parameters": {"bar_size_setting": "5 mins"},
        }

        mock_save_result = {"saved_count": 10, "symbol": "SPY"}

        with patch(
            "ops.tasks.historical_data.save_historical_bars_task",
            return_value=mock_save_result,
        ) as mock_save:
            with patch("ops.tasks.historical_data.current_task") as mock_current_task:
                # Test with option repository type
                result = save_bars_v2_task(
                    fetch_result=fetch_result,
                    repository_type="option",
                    root="/custom/root",
                )

                assert result == mock_save_result

                mock_save.assert_called_once_with(
                    mock_current_task,
                    fetch_result=fetch_result,
                    repository_type="option",
                    root="/custom/root",
                )

    def test_save_bars_error_in_save_task(self):
        """Test handling error from underlying save task."""
        from ops.tasks.historical_data import save_bars_v2_task

        mock_bars = MockDataGenerator.generate_stock_price_series("AAPL", num_bars=5)
        fetch_result = {
            "bars": mock_bars,
            "contract": {"symbol": "AAPL"},
            "parameters": {"bar_size_setting": "1 day"},
        }

        # Mock save task raising exception
        with patch("ops.tasks.historical_data.save_historical_bars_task") as mock_save:
            mock_save.side_effect = Exception("Repository write failed")
            with patch("ops.tasks.historical_data.current_task"):

                # Should propagate the exception (let Celery handle retry)
                with pytest.raises(Exception, match="Repository write failed"):
                    save_bars_v2_task(fetch_result=fetch_result)

    def test_save_bars_error_missing_symbol_in_error_result(self):
        """Test error handling when symbol is not in error result."""
        from ops.tasks.historical_data import save_bars_v2_task

        # Error result without symbol field
        fetch_result = {"error": "Generic error", "bars": []}

        result = save_bars_v2_task(fetch_result=fetch_result)

        assert result["saved"] == 0
        assert result["skipped"] == 1
        assert result["error"] == "Generic error"
        assert result["symbol"] is None  # Should handle missing symbol gracefully

    def test_save_bars_resolve_error_propagation(self):
        """Test that resolve errors are properly propagated."""
        from ops.tasks.historical_data import save_bars_v2_task

        # Simulate error from resolve task (no contract found)
        resolve_error = {"error": "No contracts found for INVALID", "symbol": "INVALID"}

        # This would be the result from fetch task when it receives resolve error
        fetch_result = resolve_error  # fetch task returns error unchanged

        result = save_bars_v2_task(fetch_result=fetch_result)

        assert result["saved"] == 0
        assert result["skipped"] == 1
        assert "No contracts found for INVALID" in result["error"]
        assert result["symbol"] == "INVALID"
