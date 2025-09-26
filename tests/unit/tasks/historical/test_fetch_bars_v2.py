"""
Unit tests for fetch_bars_for_contract_v2_task.
"""

from tests.mocks.ib_mock_factory import MockDataGenerator
from unittest.mock import Mock, patch

import pytest


@pytest.mark.unit
class TestFetchBarsForContractV2Task:
    """Tests for fetch_bars_for_contract_v2_task."""

    def test_fetch_bars_success(self):
        """Test successful bars fetching."""
        from ops.tasks.historical_data import fetch_bars_for_contract_v2_task
        from unittest.mock import MagicMock

        # Predefined test data
        contract = {
            "conId": 8314,
            "symbol": "IBM",
            "secType": "STK",
            "currency": "USD",
            "exchange": "SMART",
        }

        expected_bars = [
            {
                "time": "20250825 09:30:00",
                "open": 150.0,
                "high": 152.0,
                "low": 149.0,
                "close": 151.5,
                "volume": 1000,
            },
            {
                "time": "20250825 10:30:00",
                "open": 151.5,
                "high": 153.0,
                "low": 150.5,
                "close": 152.0,
                "volume": 1200,
            },
        ]

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.api.services.ib_api.url = "http://localhost:9010"

        # Mock httpx client
        mock_http_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = expected_bars
        mock_http_client.post.return_value = mock_response

        mock_client_context = MagicMock()
        mock_client_context.__enter__.return_value = mock_http_client
        mock_client_context.__exit__.return_value = None

        with patch(
            "ops.tasks.historical_data.get_settings", return_value=mock_settings
        ):
            with patch(
                "ops.tasks.historical_data.httpx.Client",
                return_value=mock_client_context,
            ):
                # Execute task
                result = fetch_bars_for_contract_v2_task(
                    contract=contract,
                    end_date_time="20250825 23:59:59",
                    duration_str="1 D",
                    bar_size_setting="1 hour",
                    what_to_show="TRADES",
                    use_rth=1,
                    timeout_seconds=20.0,
                    api_base=None,
                )

                # Verify results against expected data
                assert result["bars"] == expected_bars
                assert result["contract"] == contract
                assert result["parameters"]["bar_size_setting"] == "1 hour"
                assert result["parameters"]["what_to_show"] == "TRADES"

                # Verify HTTP call was made correctly
                expected_payload = {
                    "contract": contract,
                    "endDateTime": "20250825 23:59:59",
                    "durationStr": "1 D",
                    "barSizeSetting": "1 hour",
                    "whatToShow": "TRADES",
                    "useRTH": 1,
                    "formatDate": 1,
                    "keepUpToDate": False,
                    "timeout": 20,
                }
                mock_http_client.post.assert_called_once_with(
                    "/ib/historical/bars", json=expected_payload
                )

    def test_fetch_bars_with_error_contract(self):
        """Test handling contract with error from previous task."""
        from ops.tasks.historical_data import fetch_bars_for_contract_v2_task

        # Contract with error from previous task
        error_contract = {
            "error": "No contracts found for NOTFOUND",
            "symbol": "NOTFOUND",
        }

        # Execute task
        result = fetch_bars_for_contract_v2_task(
            contract=error_contract,
            end_date_time="20250825 23:59:59",
            duration_str="1 D",
            bar_size_setting="1 hour",
        )

        # Should return the error contract unchanged
        assert result == error_contract

    def test_fetch_bars_http_error(self):
        """Test handling HTTP errors during fetch."""
        from ops.tasks.historical_data import fetch_bars_for_contract_v2_task

        mock_settings = Mock()
        mock_settings.api.services.ib_api.url = "http://localhost:9010"

        contract = {
            "conId": 8314,
            "symbol": "IBM",
            "secType": "STK",
            "currency": "USD",
            "exchange": "SMART",
        }

        with patch(
            "ops.tasks.historical_data.get_settings", return_value=mock_settings
        ):
            with patch("ops.tasks.historical_data.httpx.Client") as mock_client:
                # Mock HTTP error
                mock_http_client = Mock()
                mock_http_client.post.side_effect = Exception("Connection failed")
                mock_client.return_value.__enter__.return_value = mock_http_client

                # Execute task
                result = fetch_bars_for_contract_v2_task(
                    contract=contract,
                    end_date_time="20250825 23:59:59",
                    duration_str="1 D",
                    bar_size_setting="1 hour",
                )

                # Verify error response
                assert "error" in result
                assert "Connection failed" in result["error"]
                assert result["bars"] == []
                assert result["contract"] == contract

    def test_fetch_bars_http_status_error(self):
        """Test handling HTTP status errors."""
        from ops.tasks.historical_data import fetch_bars_for_contract_v2_task

        import httpx

        mock_settings = Mock()
        mock_settings.api.services.ib_api.url = "http://localhost:9010"

        contract = {
            "conId": 8314,
            "symbol": "IBM",
            "secType": "STK",
            "currency": "USD",
            "exchange": "SMART",
        }

        with patch(
            "ops.tasks.historical_data.get_settings", return_value=mock_settings
        ):
            with patch("ops.tasks.historical_data.httpx.Client") as mock_client:
                # Mock HTTP response with error status
                mock_response = Mock()
                mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "503 Service Unavailable", request=Mock(), response=Mock()
                )

                mock_http_client = Mock()
                mock_http_client.post.return_value = mock_response
                mock_client.return_value.__enter__.return_value = mock_http_client

                # Execute task
                result = fetch_bars_for_contract_v2_task(
                    contract=contract,
                    end_date_time="20250825 23:59:59",
                    duration_str="1 D",
                    bar_size_setting="1 hour",
                )

                # Verify error response
                assert "error" in result
                assert result["bars"] == []
                assert result["contract"] == contract

    def test_fetch_bars_custom_api_base(self):
        """Test using custom API base URL."""
        from ops.tasks.historical_data import fetch_bars_for_contract_v2_task

        # Mock settings (shouldn't be used due to custom api_base)
        mock_settings = Mock()
        mock_settings.api.services.ib_api.url = "http://localhost:9010"

        contract = {
            "conId": 8314,
            "symbol": "IBM",
            "secType": "STK",
            "currency": "USD",
            "exchange": "SMART",
        }

        mock_bars = MockDataGenerator.generate_stock_price_series("IBM", num_bars=5)
        custom_api_base = "http://custom.api:8080"

        with patch(
            "ops.tasks.historical_data.get_settings", return_value=mock_settings
        ):
            with patch("ops.tasks.historical_data.httpx.Client") as mock_client:
                mock_response = Mock()
                mock_response.json.return_value = mock_bars
                mock_response.raise_for_status.return_value = None

                mock_http_client = Mock()
                mock_http_client.post.return_value = mock_response
                mock_client.return_value.__enter__.return_value = mock_http_client

                # Execute task with custom API base
                result = fetch_bars_for_contract_v2_task(
                    contract=contract,
                    end_date_time="20250825 23:59:59",
                    duration_str="1 D",
                    bar_size_setting="1 hour",
                    api_base=custom_api_base,
                )

                # Verify custom base was used
                mock_client.assert_called_with(base_url=custom_api_base, timeout=30.0)
                assert len(result["bars"]) == 5

    def test_fetch_bars_default_parameters(self):
        """Test task with default parameter values."""
        from ops.tasks.historical_data import fetch_bars_for_contract_v2_task

        mock_settings = Mock()
        mock_settings.api.services.ib_api.url = "http://localhost:9010"

        contract = {"conId": 8314, "symbol": "IBM", "secType": "STK"}
        mock_bars = []

        with patch(
            "ops.tasks.historical_data.get_settings", return_value=mock_settings
        ):
            with patch("ops.tasks.historical_data.httpx.Client") as mock_client:
                mock_response = Mock()
                mock_response.json.return_value = mock_bars
                mock_response.raise_for_status.return_value = None

                mock_http_client = Mock()
                mock_http_client.post.return_value = mock_response
                mock_client.return_value.__enter__.return_value = mock_http_client

                # Execute task with minimal parameters (using defaults)
                result = fetch_bars_for_contract_v2_task(
                    contract=contract,
                    end_date_time="20250825 23:59:59",
                    duration_str="1 D",
                    bar_size_setting="1 hour",
                )

                # Verify defaults were used in HTTP call
                expected_payload = {
                    "contract": contract,
                    "endDateTime": "20250825 23:59:59",
                    "durationStr": "1 D",
                    "barSizeSetting": "1 hour",
                    "whatToShow": "TRADES",  # default
                    "useRTH": 1,  # default
                    "formatDate": 1,
                    "keepUpToDate": False,
                    "timeout": 20,  # default timeout_seconds converted to int
                }

                mock_http_client.post.assert_called_once_with(
                    "/ib/historical/bars", json=expected_payload
                )

                # Verify parameters in result
                assert result["parameters"]["what_to_show"] == "TRADES"
