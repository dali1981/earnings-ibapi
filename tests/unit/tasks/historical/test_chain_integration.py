"""
Integration tests for the complete historical data task chain.
"""

from tests.mocks.ib_mock_factory import MockDataGenerator
from unittest.mock import Mock, patch

import pandas as pd
import pytest


@pytest.mark.unit
class TestTaskChainIntegration:
    """Integration tests for the complete task chain."""

    def test_task_chain_success_flow(self, tmp_path):
        """Test complete chain: resolve -> fetch -> save."""
        from ops.tasks.historical_data import (
            fetch_bars_for_contract_v2_task,
            resolve_single_contract_v2_task,
            save_bars_v2_task,
        )

        # Step 1: Mock resolve task
        mock_settings = Mock()
        mock_settings.paths.data_root = tmp_path
        mock_settings.api.services.ib_api.url = "http://localhost:9010"

        with patch(
            "ops.tasks.historical_data.get_settings", return_value=mock_settings
        ):
            # Mock contracts repository
            mock_repo = Mock()
            mock_df = pd.DataFrame(
                [{"conid": 8314, "symbol": "IBM", "sec_type": "STK", "currency": "USD"}]
            )
            mock_repo.read.return_value = mock_df

            with patch(
                "ops.tasks.historical_data.ContractRepository", return_value=mock_repo
            ):
                mock_self = Mock()

                # Execute resolve
                contract_result = resolve_single_contract_v2_task(
                    mock_self, symbol="IBM"
                )
                assert "conId" in contract_result
                assert contract_result["symbol"] == "IBM"

                # Step 2: Mock fetch task
                mock_bars = MockDataGenerator.generate_stock_price_series(
                    "IBM", num_bars=24
                )

                with patch("ops.tasks.historical_data.httpx.Client") as mock_client:
                    mock_response = Mock()
                    mock_response.json.return_value = mock_bars
                    mock_response.raise_for_status.return_value = None

                    mock_http_client = Mock()
                    mock_http_client.post.return_value = mock_response
                    mock_client.return_value.__enter__.return_value = mock_http_client

                    # Execute fetch (using result from resolve)
                    fetch_result = fetch_bars_for_contract_v2_task(
                        mock_self,
                        contract=contract_result,
                        end_date_time="20250825 23:59:59",
                        duration_str="1 D",
                        bar_size_setting="1 hour",
                    )

                    assert "bars" in fetch_result
                    assert len(fetch_result["bars"]) == 24
                    assert fetch_result["contract"] == contract_result

                    # Step 3: Mock save task
                    mock_save_result = {"saved_count": 24, "symbol": "IBM"}

                    with patch(
                        "ops.tasks.historical_data.save_historical_bars_task",
                        return_value=mock_save_result,
                    ):
                        with patch("ops.tasks.historical_data.current_task"):
                            # Execute save (using result from fetch)
                            save_result = save_bars_v2_task(
                                fetch_result=fetch_result, repository_type="equity"
                            )

                            assert save_result["saved_count"] == 24
                            assert save_result["symbol"] == "IBM"

    def test_task_chain_error_propagation(self, tmp_path):
        """Test error propagation through the task chain."""
        from ops.tasks.historical_data import (
            fetch_bars_for_contract_v2_task,
            resolve_single_contract_v2_task,
            save_bars_v2_task,
        )

        # Step 1: Mock resolve task with error
        mock_settings = Mock()
        mock_settings.paths.data_root = tmp_path

        with patch(
            "ops.tasks.historical_data.get_settings", return_value=mock_settings
        ):
            mock_repo = Mock()
            mock_repo.read.return_value = pd.DataFrame()  # Empty result

            with patch(
                "ops.tasks.historical_data.ContractRepository", return_value=mock_repo
            ):
                mock_self = Mock()

                # Resolve fails
                error_result = resolve_single_contract_v2_task(
                    mock_self, symbol="NOTFOUND"
                )
                assert "error" in error_result

                # Step 2: Fetch should handle error gracefully
                fetch_result = fetch_bars_for_contract_v2_task(
                    mock_self,
                    contract=error_result,  # Pass error from resolve
                    end_date_time="20250825 23:59:59",
                    duration_str="1 D",
                    bar_size_setting="1 hour",
                )

                # Should return the same error
                assert fetch_result == error_result

                # Step 3: Save should also handle error gracefully
                save_result = save_bars_v2_task(
                    fetch_result=fetch_result,  # Pass error from fetch
                    repository_type="equity",
                )

                # Should return error summary
                assert save_result["saved"] == 0
                assert save_result["skipped"] == 1
                assert "error" in save_result

    def test_task_chain_partial_failure_scenarios(self, tmp_path):
        """Test various failure scenarios in the chain."""
        from ops.tasks.historical_data import (
            fetch_bars_for_contract_v2_task,
            resolve_single_contract_v2_task,
            save_bars_v2_task,
        )

        # Scenario 1: Resolve succeeds, fetch fails
        mock_settings = Mock()
        mock_settings.paths.data_root = tmp_path
        mock_settings.api.services.ib_api.url = "http://localhost:9010"

        with patch(
            "ops.tasks.historical_data.get_settings", return_value=mock_settings
        ):
            # Successful resolve
            mock_repo = Mock()
            mock_df = pd.DataFrame(
                [{"conid": 8314, "symbol": "IBM", "sec_type": "STK", "currency": "USD"}]
            )
            mock_repo.read.return_value = mock_df

            with patch(
                "ops.tasks.historical_data.ContractRepository", return_value=mock_repo
            ):
                mock_self = Mock()
                contract_result = resolve_single_contract_v2_task(
                    mock_self, symbol="IBM"
                )
                assert "conId" in contract_result

                # Failed fetch
                with patch("ops.tasks.historical_data.httpx.Client") as mock_client:
                    mock_http_client = Mock()
                    mock_http_client.post.side_effect = Exception("Network error")
                    mock_client.return_value.__enter__.return_value = mock_http_client

                    fetch_result = fetch_bars_for_contract_v2_task(
                        mock_self,
                        contract=contract_result,
                        end_date_time="20250825 23:59:59",
                        duration_str="1 D",
                        bar_size_setting="1 hour",
                    )

                    # Fetch should return error
                    assert "error" in fetch_result
                    assert "Network error" in fetch_result["error"]

                    # Save should handle fetch error
                    save_result = save_bars_v2_task(
                        fetch_result=fetch_result, repository_type="equity"
                    )

                    assert save_result["saved"] == 0
                    assert save_result["skipped"] == 1
                    assert "Network error" in save_result["error"]

    def test_task_chain_with_empty_bars(self, tmp_path):
        """Test chain when API returns empty bars (no error but no data)."""
        from ops.tasks.historical_data import (
            fetch_bars_for_contract_v2_task,
            resolve_single_contract_v2_task,
            save_bars_v2_task,
        )

        mock_settings = Mock()
        mock_settings.paths.data_root = tmp_path
        mock_settings.api.services.ib_api.url = "http://localhost:9010"

        with patch(
            "ops.tasks.historical_data.get_settings", return_value=mock_settings
        ):
            # Successful resolve
            mock_repo = Mock()
            mock_df = pd.DataFrame(
                [{"conid": 8314, "symbol": "IBM", "sec_type": "STK", "currency": "USD"}]
            )
            mock_repo.read.return_value = mock_df

            with patch(
                "ops.tasks.historical_data.ContractRepository", return_value=mock_repo
            ):
                mock_self = Mock()
                contract_result = resolve_single_contract_v2_task(
                    mock_self, symbol="IBM"
                )

                # Successful fetch but empty bars
                with patch("ops.tasks.historical_data.httpx.Client") as mock_client:
                    mock_response = Mock()
                    mock_response.json.return_value = []  # Empty bars
                    mock_response.raise_for_status.return_value = None

                    mock_http_client = Mock()
                    mock_http_client.post.return_value = mock_response
                    mock_client.return_value.__enter__.return_value = mock_http_client

                    fetch_result = fetch_bars_for_contract_v2_task(
                        mock_self,
                        contract=contract_result,
                        end_date_time="20250825 23:59:59",
                        duration_str="1 D",
                        bar_size_setting="1 hour",
                    )

                    assert "bars" in fetch_result
                    assert len(fetch_result["bars"]) == 0
                    assert "error" not in fetch_result

                    # Save should handle empty bars
                    mock_save_result = {"saved_count": 0}
                    with patch(
                        "ops.tasks.historical_data.save_historical_bars_task",
                        return_value=mock_save_result,
                    ):
                        with patch("ops.tasks.historical_data.current_task"):
                            save_result = save_bars_v2_task(
                                fetch_result=fetch_result, repository_type="equity"
                            )

                            assert save_result["saved_count"] == 0

    def test_task_signatures_for_chaining(self):
        """Test that task signatures are compatible with Celery chaining."""
        from ops.tasks.historical_data import (
            fetch_bars_for_contract_v2_task,
            resolve_single_contract_v2_task,
            save_bars_v2_task,
        )

        import inspect

        # Check resolve task signature
        resolve_sig = inspect.signature(resolve_single_contract_v2_task)
        assert "self" in resolve_sig.parameters
        assert "symbol" in resolve_sig.parameters
        assert "root" in resolve_sig.parameters

        # Check fetch task signature (should accept contract as first param after self)
        fetch_sig = inspect.signature(fetch_bars_for_contract_v2_task)
        params = list(fetch_sig.parameters.keys())
        assert params[0] == "self"
        assert params[1] == "contract"  # Should receive from previous task

        # Check save task signature (should accept fetch_result as first param)
        save_sig = inspect.signature(save_bars_v2_task)
        save_params = list(save_sig.parameters.keys())
        assert save_params[0] == "fetch_result"  # Should receive from previous task

    def test_data_consistency_through_chain(self, tmp_path):
        """Test that data remains consistent as it flows through the chain."""
        from ops.tasks.historical_data import (
            fetch_bars_for_contract_v2_task,
            resolve_single_contract_v2_task,
            save_bars_v2_task,
        )

        symbol = "AAPL"
        expected_conid = 265598

        mock_settings = Mock()
        mock_settings.paths.data_root = tmp_path
        mock_settings.api.services.ib_api.url = "http://localhost:9010"

        with patch(
            "ops.tasks.historical_data.get_settings", return_value=mock_settings
        ):
            # Mock resolve
            mock_repo = Mock()
            mock_df = pd.DataFrame(
                [
                    {
                        "conid": expected_conid,
                        "symbol": symbol,
                        "sec_type": "STK",
                        "currency": "USD",
                    }
                ]
            )
            mock_repo.read.return_value = mock_df

            with patch(
                "ops.tasks.historical_data.ContractRepository", return_value=mock_repo
            ):
                mock_self = Mock()
                contract_result = resolve_single_contract_v2_task(
                    mock_self, symbol=symbol
                )

                # Verify resolve result structure
                assert contract_result["conId"] == expected_conid
                assert contract_result["symbol"] == symbol

                # Mock fetch with realistic bars
                mock_bars = MockDataGenerator.generate_stock_price_series(
                    symbol, num_bars=10
                )

                with patch("ops.tasks.historical_data.httpx.Client") as mock_client:
                    mock_response = Mock()
                    mock_response.json.return_value = mock_bars
                    mock_response.raise_for_status.return_value = None

                    mock_http_client = Mock()
                    mock_http_client.post.return_value = mock_response
                    mock_client.return_value.__enter__.return_value = mock_http_client

                    fetch_result = fetch_bars_for_contract_v2_task(
                        mock_self,
                        contract=contract_result,
                        end_date_time="20250825 23:59:59",
                        duration_str="1 D",
                        bar_size_setting="1 hour",
                    )

                    # Verify fetch result structure and data consistency
                    assert fetch_result["contract"] == contract_result
                    assert len(fetch_result["bars"]) == 10
                    assert fetch_result["parameters"]["bar_size_setting"] == "1 hour"

                    # Mock save
                    mock_save_result = {
                        "saved_count": 10,
                        "symbol": symbol,
                        "bar_size": "1 hour",
                    }

                    with patch(
                        "ops.tasks.historical_data.save_historical_bars_task",
                        return_value=mock_save_result,
                    ) as mock_save:
                        with patch("ops.tasks.historical_data.current_task"):
                            save_result = save_bars_v2_task(
                                fetch_result=fetch_result, repository_type="equity"
                            )

                            # Verify save result and data consistency
                            assert save_result["saved_count"] == 10
                            assert save_result["symbol"] == symbol

                            # Verify save was called with correct data
                            mock_save.assert_called_once()
                            call_args = mock_save.call_args
                            assert call_args[1]["fetch_result"] == fetch_result
