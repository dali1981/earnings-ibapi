"""
Unit tests for resolve_single_contract_v2_task.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest


@pytest.mark.unit
class TestResolveSingleContractV2Task:
    """Tests for resolve_single_contract_v2_task."""

    def test_resolve_single_contract_success(self, tmp_path):
        """Test successful contract resolution."""
        from ops.tasks.historical_data import resolve_single_contract_v2_task

        # Mock settings
        mock_settings = Mock()
        mock_settings.paths.data_root = tmp_path

        with patch(
            "ops.tasks.historical_data.get_settings", return_value=mock_settings
        ):
            # Create mock contracts repository
            mock_repo = Mock()
            mock_df = pd.DataFrame(
                [{"conid": 8314, "symbol": "IBM", "sec_type": "STK", "currency": "USD"}]
            )
            mock_repo.read.return_value = mock_df

            with patch(
                "ops.tasks.historical_data.ContractRepository", return_value=mock_repo
            ):
                # Create mock task instance
                mock_self = Mock()

                # Execute task
                result = resolve_single_contract_v2_task(mock_self, symbol="IBM")

                # Verify results
                assert result["conId"] == 8314
                assert result["symbol"] == "IBM"
                assert result["secType"] == "STK"
                assert result["currency"] == "USD"
                assert result["exchange"] == "SMART"

                # Verify repository was called correctly
                mock_repo.read.assert_called_once_with(
                    columns=["conid", "symbol", "sec_type", "currency"], symbol=["IBM"]
                )

    def test_resolve_single_contract_not_found(self, tmp_path):
        """Test contract not found scenario."""
        from ops.tasks.historical_data import resolve_single_contract_v2_task

        # Mock settings
        mock_settings = Mock()
        mock_settings.paths.data_root = tmp_path

        with patch(
            "ops.tasks.historical_data.get_settings", return_value=mock_settings
        ):
            # Create mock contracts repository with empty result
            mock_repo = Mock()
            mock_repo.read.return_value = pd.DataFrame()  # Empty DataFrame

            with patch(
                "ops.tasks.historical_data.ContractRepository", return_value=mock_repo
            ):
                mock_self = Mock()

                # Execute task
                result = resolve_single_contract_v2_task(mock_self, symbol="NOTFOUND")

                # Verify error response
                assert "error" in result
                assert "No contracts found for NOTFOUND" in result["error"]
                assert result["symbol"] == "NOTFOUND"

    def test_resolve_single_contract_case_insensitive(self, tmp_path):
        """Test case insensitive symbol handling."""
        from ops.tasks.historical_data import resolve_single_contract_v2_task

        mock_settings = Mock()
        mock_settings.paths.data_root = tmp_path

        with patch(
            "ops.tasks.historical_data.get_settings", return_value=mock_settings
        ):
            mock_repo = Mock()
            mock_df = pd.DataFrame(
                [{"conid": 8314, "symbol": "IBM", "sec_type": "STK", "currency": "USD"}]
            )
            mock_repo.read.return_value = mock_df

            with patch(
                "ops.tasks.historical_data.ContractRepository", return_value=mock_repo
            ):
                mock_self = Mock()

                # Test with lowercase
                result = resolve_single_contract_v2_task(mock_self, symbol="ibm")

                # Should still work
                assert result["symbol"] == "IBM"

                # Verify repository was called with uppercase
                mock_repo.read.assert_called_with(
                    columns=["conid", "symbol", "sec_type", "currency"], symbol=["IBM"]
                )

    def test_resolve_single_contract_multiple_contracts(self, tmp_path):
        """Test handling multiple contracts for same symbol (takes first)."""
        from ops.tasks.historical_data import resolve_single_contract_v2_task

        mock_settings = Mock()
        mock_settings.paths.data_root = tmp_path

        with patch(
            "ops.tasks.historical_data.get_settings", return_value=mock_settings
        ):
            mock_repo = Mock()
            # Multiple contracts for same symbol
            mock_df = pd.DataFrame(
                [
                    {
                        "conid": 8314,
                        "symbol": "IBM",
                        "sec_type": "STK",
                        "currency": "USD",
                    },
                    {
                        "conid": 8315,
                        "symbol": "IBM",
                        "sec_type": "STK",
                        "currency": "EUR",
                    },
                ]
            )
            mock_repo.read.return_value = mock_df

            with patch(
                "ops.tasks.historical_data.ContractRepository", return_value=mock_repo
            ):
                mock_self = Mock()

                result = resolve_single_contract_v2_task(mock_self, symbol="IBM")

                # Should take first contract
                assert result["conId"] == 8314
                assert result["currency"] == "USD"

    def test_resolve_single_contract_with_custom_root(self):
        """Test task with custom root path."""
        from ops.tasks.historical_data import resolve_single_contract_v2_task

        custom_root = "/custom/path"

        # Don't mock get_settings since we're providing root
        mock_repo = Mock()
        mock_df = pd.DataFrame(
            [{"conid": 8314, "symbol": "IBM", "sec_type": "STK", "currency": "USD"}]
        )
        mock_repo.read.return_value = mock_df

        with patch(
            "ops.tasks.historical_data.ContractRepository", return_value=mock_repo
        ):
            mock_self = Mock()

            result = resolve_single_contract_v2_task(
                mock_self, symbol="IBM", root=custom_root
            )

            assert result["conId"] == 8314

            # Verify ContractRepository was created with custom path
            from ops.tasks.historical_data import ContractRepository

            ContractRepository.assert_called_with(f"{custom_root}/contracts")

    def test_resolve_single_contract_repository_error(self, tmp_path):
        """Test handling repository errors."""
        from ops.tasks.historical_data import resolve_single_contract_v2_task

        mock_settings = Mock()
        mock_settings.paths.data_root = tmp_path

        with patch(
            "ops.tasks.historical_data.get_settings", return_value=mock_settings
        ):
            mock_repo = Mock()
            mock_repo.read.side_effect = Exception("Database error")

            with patch(
                "ops.tasks.historical_data.ContractRepository", return_value=mock_repo
            ):
                mock_self = Mock()

                # Should raise the exception (let Celery handle retry)
                with pytest.raises(Exception, match="Database error"):
                    resolve_single_contract_v2_task(mock_self, symbol="IBM")
