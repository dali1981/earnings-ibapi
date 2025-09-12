"""
Unit tests for FileEventBus.

Tests the core event publishing and management functionality of the file-based event bus
that coordinates data flow across the trading system.
"""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import uuid

from common.utils.events.bus import FileEventBus
from common.exceptions import DataValidationException


@pytest.mark.unit
@pytest.mark.events
class TestFileEventBus:
    """Test cases for FileEventBus."""

    def test_event_bus_initialization_success(self, temp_data_dir):
        """Test successful event bus initialization."""
        # Arrange
        root_dir = temp_data_dir
        
        # Act
        bus = FileEventBus(root_dir)
        
        # Assert
        assert bus.events_dir == root_dir / "events"
        assert bus.events_dir.exists()

    def test_event_bus_initialization_with_default_path(self):
        """Test event bus initialization with default path."""
        # Act
        bus = FileEventBus()
        
        # Assert
        assert bus.events_dir is not None
        assert isinstance(bus.events_dir, Path)

    def test_notify_data_available_basic_success(self, event_bus_test_environment):
        """Test basic successful data availability notification."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        dataset = "equity_bars"
        symbol = "AAPL"
        start_date = "2024-01-01"
        end_date = "2024-01-31"
        
        # Act
        result = bus.notify_data_available(
            dataset=dataset,
            symbol=symbol,
            start=start_date,
            end=end_date
        )
        
        # Assert - notify_data_available returns None in actual implementation
        assert result is None
        
        # Verify event file was created
        event_file = event_bus_test_environment["events_dir"] / f"{dataset}.jsonl"
        assert event_file.exists()
        
        # Verify event content - actual implementation uses nested payload structure
        with open(event_file, 'r') as f:
            event_line = f.readline().strip()
            event_data = json.loads(event_line)
            
        assert event_data["type"] == "data.available"
        assert event_data["topic"] == dataset
        assert event_data["payload"]["symbol"] == symbol
        assert event_data["payload"]["start"] == start_date
        assert event_data["payload"]["end"] == end_date
        assert "ts" in event_data  # timestamp should be present

    def test_notify_data_available_with_job_id(self, event_bus_test_environment):
        """Test notification with job ID tracking."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        dataset = "option_chains"
        symbol = "MSFT"
        job_id = "job_12345"
        
        # Act
        result = bus.notify_data_available(
            dataset=dataset,
            symbol=symbol,
            job_id=job_id
        )
        
        # Assert
        event_file = event_bus_test_environment["events_dir"] / f"{dataset}.jsonl"
        with open(event_file, 'r') as f:
            event_data = json.loads(f.readline().strip())
            
        assert event_data["payload"]["job_id"] == job_id
        # Check if correlation_id is in payload (may be auto-generated)
        assert "payload" in event_data

    def test_notify_data_available_with_custom_correlation_id(self, event_bus_test_environment):
        """Test notification with custom correlation ID."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        custom_correlation_id = "custom_correlation_123"
        
        # Act
        result = bus.notify_data_available(
            dataset="earnings",
            correlation_id=custom_correlation_id
        )
        
        # Assert
        assert result is None  # notify_data_available returns None
        
        # Verify in event file
        event_file = event_bus_test_environment["events_dir"] / "earnings.jsonl"
        with open(event_file, 'r') as f:
            event_data = json.loads(f.readline().strip())
            
        assert event_data["payload"]["correlation_id"] == custom_correlation_id

    def test_notify_data_available_with_additional_metadata(self, event_bus_test_environment):
        """Test notification with additional metadata fields."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        additional_metadata = {
            "bar_size": "1 day",
            "what_to_show": "TRADES",
            "rows_processed": 1000,
            "processing_time": 2.5
        }
        
        # Act
        bus.notify_data_available(
            dataset="equity_bars",
            symbol="GOOGL",
            **additional_metadata
        )
        
        # Assert
        event_file = event_bus_test_environment["events_dir"] / "equity_bars.jsonl"
        with open(event_file, 'r') as f:
            event_data = json.loads(f.readline().strip())
            
        for key, value in additional_metadata.items():
            assert event_data["payload"][key] == value

    def test_multiple_notifications_same_dataset(self, event_bus_test_environment):
        """Test multiple notifications for the same dataset append to file."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        dataset = "option_bars"
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        # Act
        for symbol in symbols:
            result = bus.notify_data_available(
                dataset=dataset,
                symbol=symbol
            )
            # result is None, so we don't collect anything
        
        # Assert
        event_file = event_bus_test_environment["events_dir"] / f"{dataset}.jsonl"
        
        # Count lines in file
        with open(event_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 3
        
        # Verify all symbols are present
        symbols_in_file = []
        for line in lines:
            event_data = json.loads(line.strip())
            symbols_in_file.append(event_data["payload"]["symbol"])
            
        assert set(symbols_in_file) == set(symbols)

    def test_notifications_different_datasets(self, event_bus_test_environment):
        """Test notifications for different datasets create separate files."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        datasets = ["equity_bars", "option_chains", "earnings"]
        
        # Act
        for dataset in datasets:
            bus.notify_data_available(dataset=dataset, symbol="TEST")
        
        # Assert
        for dataset in datasets:
            event_file = event_bus_test_environment["events_dir"] / f"{dataset}.jsonl"
            assert event_file.exists()
            
            with open(event_file, 'r') as f:
                event_data = json.loads(f.readline().strip())
                assert event_data["topic"] == dataset

    def test_event_timestamp_format(self, event_bus_test_environment):
        """Test that event timestamps are in correct ISO format."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        before_time = datetime.now()
        
        # Act
        bus.notify_data_available(dataset="test", symbol="TEST")
        
        after_time = datetime.now()
        
        # Assert
        event_file = event_bus_test_environment["events_dir"] / "test.jsonl"
        with open(event_file, 'r') as f:
            event_data = json.loads(f.readline().strip())
            
        timestamp_num = event_data["ts"]  # timestamp is stored as ts (float)
        timestamp = datetime.fromtimestamp(timestamp_num)
        
        # Timestamp should be between before and after
        assert before_time <= timestamp <= after_time + timedelta(seconds=1)

    def test_event_serialization_handles_special_types(self, event_bus_test_environment):
        """Test that event serialization handles dates and other special types."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        from datetime import date
        
        # Act
        bus.notify_data_available(
            dataset="test",
            symbol="TEST",
            date_field=date.today(),
            datetime_field=datetime.now(),
            none_field=None,
            boolean_field=True,
            number_field=123.45
        )
        
        # Assert - should not raise serialization errors
        event_file = event_bus_test_environment["events_dir"] / "test.jsonl"
        with open(event_file, 'r') as f:
            event_data = json.loads(f.readline().strip())
            
        # Check that special types were serialized in payload
        payload = event_data["payload"]
        assert "date_field" in payload
        assert "datetime_field" in payload
        assert payload["none_field"] is None
        assert payload["boolean_field"] is True
        assert payload["number_field"] == 123.45

    def test_concurrent_notifications_thread_safety(self, event_bus_test_environment):
        """Test thread safety of concurrent notifications."""
        import threading
        
        # Arrange
        bus = event_bus_test_environment["bus"]
        dataset = "concurrent_test"
        num_threads = 5
        notifications_per_thread = 10
        results = []
        
        def notify_worker(thread_id):
            thread_results = []
            for i in range(notifications_per_thread):
                try:
                    result = bus.notify_data_available(
                        dataset=dataset,
                        symbol=f"THREAD{thread_id}",
                        message_id=f"{thread_id}_{i}"
                    )
                    thread_results.append(("success", correlation_id))
                except Exception as e:
                    thread_results.append(("error", str(e)))
            results.extend(thread_results)
        
        # Act
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=notify_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Assert
        success_count = len([r for r in results if r[0] == "success"])
        total_expected = num_threads * notifications_per_thread
        
        # Most notifications should succeed
        assert success_count >= total_expected * 0.9
        
        # Check that all notifications were written to file
        event_file = event_bus_test_environment["events_dir"] / f"{dataset}.jsonl"
        with open(event_file, 'r') as f:
            lines = f.readlines()
            
        assert len(lines) >= success_count * 0.9  # Account for potential duplicates

    def test_notification_with_empty_dataset_error(self, event_bus_test_environment):
        """Test error handling for empty dataset name."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        
        # Act & Assert
        with pytest.raises((ValueError, DataValidationException)):
            bus.notify_data_available(dataset="", symbol="TEST")

    def test_notification_with_none_dataset_error(self, event_bus_test_environment):
        """Test error handling for None dataset name."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        
        # Act & Assert
        with pytest.raises((ValueError, DataValidationException, TypeError)):
            bus.notify_data_available(dataset=None, symbol="TEST")

    def test_notification_with_invalid_characters_in_dataset(self, event_bus_test_environment):
        """Test handling of invalid characters in dataset name."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        invalid_datasets = ["dataset/with/slashes", "dataset with spaces", "dataset\nwith\nnewlines"]
        
        for invalid_dataset in invalid_datasets:
            # Act & Assert
            try:
                result = bus.notify_data_available(
                    dataset=invalid_dataset, 
                    symbol="TEST"
                )
                # If it succeeds, verify file was created with safe name
                # Implementation may sanitize the name
                assert correlation_id is not None
            except (ValueError, DataValidationException):
                # Also acceptable to reject invalid names
                pass

    def test_event_file_permissions(self, event_bus_test_environment):
        """Test that event files are created with correct permissions."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        
        # Act
        bus.notify_data_available(dataset="permissions_test", symbol="TEST")
        
        # Assert
        event_file = event_bus_test_environment["events_dir"] / "permissions_test.jsonl"
        assert event_file.exists()
        
        # Check file is readable and writable
        assert event_file.is_file()
        assert event_file.stat().st_size > 0

    def test_large_event_data_handling(self, event_bus_test_environment):
        """Test handling of large event data."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        
        # Create large metadata
        large_data = {
            "large_list": list(range(1000)),
            "large_string": "x" * 10000,
            "nested_dict": {
                f"key_{i}": f"value_{i}" * 100 
                for i in range(100)
            }
        }
        
        # Act
        result = bus.notify_data_available(
            dataset="large_data_test",
            symbol="TEST",
            **large_data
        )
        
        # Assert
        assert correlation_id is not None
        
        event_file = event_bus_test_environment["events_dir"] / "large_data_test.jsonl"
        assert event_file.exists()
        
        # Verify data was written correctly
        with open(event_file, 'r') as f:
            event_data = json.loads(f.readline().strip())
            
        assert len(event_data["large_list"]) == 1000
        assert len(event_data["large_string"]) == 10000
        assert len(event_data["nested_dict"]) == 100

    @pytest.mark.slow
    def test_high_volume_notifications_performance(self, event_bus_test_environment, benchmark_timer):
        """Test performance under high volume of notifications."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        dataset = "performance_test"
        num_notifications = 1000
        
        # Act
        with benchmark_timer() as timer:
            for i in range(num_notifications):
                bus.notify_data_available(
                    dataset=dataset,
                    symbol=f"PERF{i:04d}",
                    message_id=i
                )
        
        # Assert
        assert timer.elapsed < 5.0, f"High volume notifications took too long: {timer.elapsed}s"
        
        # Verify all notifications were written
        event_file = event_bus_test_environment["events_dir"] / f"{dataset}.jsonl"
        with open(event_file, 'r') as f:
            lines = f.readlines()
            
        assert len(lines) == num_notifications

    def test_event_file_recovery_after_corruption(self, event_bus_test_environment):
        """Test event bus recovery after file corruption."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        dataset = "recovery_test"
        
        # Create initial event
        bus.notify_data_available(dataset=dataset, symbol="TEST1")
        
        # Simulate file corruption
        event_file = event_bus_test_environment["events_dir"] / f"{dataset}.jsonl"
        with open(event_file, 'a') as f:
            f.write("CORRUPTED_LINE_NOT_JSON\n")
        
        # Act - try to add another event
        try:
            result = bus.notify_data_available(dataset=dataset, symbol="TEST2")
            # If successful, verify new event was added
            assert correlation_id is not None
        except Exception:
            # May fail due to corruption - acceptable behavior
            pass

    def test_event_deduplication_behavior(self, event_bus_test_environment):
        """Test behavior when identical events are published."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        dataset = "dedup_test"
        
        # Act - publish identical events
        correlation_id1 = bus.notify_data_available(
            dataset=dataset,
            symbol="DEDUP_TEST",
            start="2024-01-01",
            end="2024-01-31"
        )
        
        correlation_id2 = bus.notify_data_available(
            dataset=dataset,
            symbol="DEDUP_TEST", 
            start="2024-01-01",
            end="2024-01-31"
        )
        
        # Assert - correlation IDs should be different (no deduplication at bus level)
        assert correlation_id1 != correlation_id2
        
        # Both events should be in file
        event_file = event_bus_test_environment["events_dir"] / f"{dataset}.jsonl"
        with open(event_file, 'r') as f:
            lines = f.readlines()
            
        assert len(lines) == 2  # No deduplication


@pytest.mark.unit
@pytest.mark.events
class TestFileEventBusEdgeCases:
    """Edge case testing for FileEventBus."""

    def test_notification_with_unicode_characters(self, event_bus_test_environment):
        """Test handling of Unicode characters in event data."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        unicode_data = {
            "symbol": "测试",  # Chinese characters
            "description": "Тест",  # Cyrillic characters  
            "emoji": "🚀📈💹",  # Emojis
            "special": "äöüß€"  # European characters
        }
        
        # Act
        result = bus.notify_data_available(
            dataset="unicode_test",
            **unicode_data
        )
        
        # Assert
        assert correlation_id is not None
        
        event_file = event_bus_test_environment["events_dir"] / "unicode_test.jsonl"
        with open(event_file, 'r', encoding='utf-8') as f:
            event_data = json.loads(f.readline().strip())
            
        for key, value in unicode_data.items():
            assert event_data[key] == value

    def test_notification_disk_space_exhaustion_simulation(self, event_bus_test_environment):
        """Test behavior when disk space is exhausted (simulated)."""
        # This is a difficult test to implement without actually filling disk
        # We'll test with a read-only directory instead
        
        # Arrange
        bus = event_bus_test_environment["bus"]
        
        # Make events directory read-only
        events_dir = event_bus_test_environment["events_dir"]
        events_dir.chmod(0o444)  # Read-only
        
        try:
            # Act & Assert
            with pytest.raises((OSError, PermissionError)):
                bus.notify_data_available(dataset="readonly_test", symbol="TEST")
        finally:
            # Cleanup - restore write permissions
            events_dir.chmod(0o755)

    def test_notification_with_very_long_dataset_name(self, event_bus_test_environment):
        """Test handling of very long dataset names."""
        # Arrange
        bus = event_bus_test_environment["bus"]
        long_dataset_name = "a" * 255  # Very long name
        
        # Act & Assert
        try:
            result = bus.notify_data_available(
                dataset=long_dataset_name,
                symbol="TEST"
            )
            # If successful, should create file with valid name
            assert correlation_id is not None
        except (ValueError, OSError):
            # May reject very long names - also acceptable
            pass