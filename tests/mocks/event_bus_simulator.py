"""
Event bus simulator for comprehensive testing of event-driven workflows.
"""

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock
import queue
import uuid


class EventBusSimulator:
    """
    Simulates the file-based event bus for testing purposes.
    Allows injection of events, simulation of delays, and error scenarios.
    """

    def __init__(self, temp_dir: Path = None):
        self.temp_dir = temp_dir or Path("/tmp/test_events")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.events_queue = queue.Queue()
        self.subscribers = {}
        self.published_events = []
        self.processing_delays = {}
        self.error_scenarios = {}
        self._running = False
        self._processing_thread = None

    def start(self):
        """Start the event bus simulation."""
        self._running = True
        self._processing_thread = threading.Thread(target=self._process_events)
        self._processing_thread.daemon = True
        self._processing_thread.start()

    def stop(self):
        """Stop the event bus simulation."""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=1.0)

    def publish_event(
        self,
        topic: str,
        event_data: Dict[str, Any],
        delay: float = 0.0,
        correlation_id: Optional[str] = None
    ):
        """Publish an event to the specified topic."""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())

        event = {
            'timestamp': datetime.now().isoformat(),
            'topic': topic,
            'correlation_id': correlation_id,
            'event_type': event_data.get('event_type', 'data_available'),
            'payload': event_data,
            'delay': delay
        }

        self.events_queue.put(event)
        self.published_events.append(event)

    def subscribe(self, topic: str, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to events on a specific topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def set_processing_delay(self, topic: str, delay: float):
        """Set a processing delay for events on a specific topic."""
        self.processing_delays[topic] = delay

    def set_error_scenario(self, topic: str, error_rate: float = 0.1, error_message: str = "Simulated error"):
        """Configure error scenarios for a topic."""
        self.error_scenarios[topic] = {
            'rate': error_rate,
            'message': error_message
        }

    def _process_events(self):
        """Process events from the queue (runs in background thread)."""
        while self._running:
            try:
                event = self.events_queue.get(timeout=0.1)
                
                # Simulate event delay
                if event.get('delay', 0) > 0:
                    time.sleep(event['delay'])

                # Simulate topic-specific processing delays
                topic = event['topic']
                if topic in self.processing_delays:
                    time.sleep(self.processing_delays[topic])

                # Simulate errors
                if topic in self.error_scenarios:
                    error_config = self.error_scenarios[topic]
                    import random
                    if random.random() < error_config['rate']:
                        # Simulate error by not delivering the event
                        continue

                # Deliver event to subscribers
                if topic in self.subscribers:
                    for callback in self.subscribers[topic]:
                        try:
                            callback(event)
                        except Exception as e:
                            # Log error but continue processing
                            print(f"Error in event subscriber: {e}")

                self.events_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing event: {e}")

    def get_events_for_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get all published events for a specific topic."""
        return [event for event in self.published_events if event['topic'] == topic]

    def get_event_count(self, topic: str = None) -> int:
        """Get count of events (optionally filtered by topic)."""
        if topic is None:
            return len(self.published_events)
        return len(self.get_events_for_topic(topic))

    def clear_events(self):
        """Clear all published events."""
        self.published_events.clear()

    def create_mock_consumer(self, topic: str) -> Mock:
        """Create a mock consumer for a specific topic."""
        consumer = Mock()
        topic_events = self.get_events_for_topic(topic)
        
        def mock_read(service: str, limit: int = None):
            events = topic_events.copy()
            if limit:
                events = events[:limit]
            return iter(events)

        consumer.read = mock_read
        consumer.topic = topic
        return consumer

    def create_mock_processed_index(self) -> Mock:
        """Create a mock processed index for tracking event processing."""
        index = Mock()
        processed_keys = set()

        def mock_is_processed(key: str) -> bool:
            return key in processed_keys

        def mock_mark_processed(key: str):
            processed_keys.add(key)

        def mock_get_processed_keys():
            return list(processed_keys)

        index.is_processed = mock_is_processed
        index.mark_processed = mock_mark_processed
        index.get_processed_keys = mock_get_processed_keys
        
        return index

    def simulate_data_pipeline_events(
        self,
        symbols: List[str] = None,
        datasets: List[str] = None
    ):
        """Simulate a realistic data pipeline event sequence."""
        if symbols is None:
            symbols = ["AAPL", "MSFT", "GOOGL"]
        
        if datasets is None:
            datasets = ["equity_bars", "option_chains", "option_bars"]

        correlation_id = str(uuid.uuid4())
        
        # Simulate earnings announcement
        self.publish_event(
            "earnings",
            {
                "event_type": "data_available",
                "dataset": "earnings",
                "symbols": symbols,
                "period": "Q1_2024"
            },
            correlation_id=correlation_id
        )

        # Simulate equity data ingestion
        for symbol in symbols:
            self.publish_event(
                "equity_bars",
                {
                    "event_type": "data_available",
                    "dataset": "equity_bars",
                    "symbol": symbol,
                    "start": "2024-01-01",
                    "end": "2024-03-31",
                    "bar_size": "1 day"
                },
                delay=0.1,
                correlation_id=correlation_id
            )

        # Simulate option chain snapshots
        for symbol in symbols:
            self.publish_event(
                "option_chains",
                {
                    "event_type": "data_available",
                    "dataset": "option_chains",
                    "symbol": symbol,
                    "snapshot_time": datetime.now().isoformat(),
                    "expiries": ["2024-04-15", "2024-05-15"]
                },
                delay=0.2,
                correlation_id=correlation_id
            )

        # Simulate option bars processing (depends on chains)
        for symbol in symbols:
            self.publish_event(
                "option_bars",
                {
                    "event_type": "data_available",
                    "dataset": "option_bars",
                    "symbol": symbol,
                    "contracts_processed": 50,
                    "start": "2024-01-01",
                    "end": "2024-03-31"
                },
                delay=0.5,  # Longer delay for complex processing
                correlation_id=correlation_id
            )

        return correlation_id


class MockFileEventBus:
    """Mock implementation of the FileEventBus for testing."""

    def __init__(self, events_dir: Path = None):
        self.events_dir = events_dir or Path("/tmp/test_events")
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.published_events = []
        self.error_on_notify = False
        self.notification_delay = 0.0

    def notify_data_available(
        self,
        dataset: str,
        symbol: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        job_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ):
        """Mock implementation of notify_data_available."""
        if self.error_on_notify:
            raise Exception("Simulated notification error")

        if self.notification_delay > 0:
            time.sleep(self.notification_delay)

        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'data_available',
            'dataset': dataset,
            'symbol': symbol,
            'start': start,
            'end': end,
            'job_id': job_id,
            'correlation_id': correlation_id or str(uuid.uuid4()),
            **kwargs
        }

        self.published_events.append(event)

        # Write to file to simulate real behavior
        topic_file = self.events_dir / f"{dataset}.jsonl"
        with open(topic_file, 'a') as f:
            f.write(json.dumps(event) + '\n')

    def get_published_events(self) -> List[Dict[str, Any]]:
        """Get all published events."""
        return self.published_events.copy()

    def clear_events(self):
        """Clear all published events."""
        self.published_events.clear()
        # Also clear files
        for file in self.events_dir.glob("*.jsonl"):
            file.unlink()

    def set_error_mode(self, enabled: bool):
        """Enable/disable error simulation."""
        self.error_on_notify = enabled

    def set_notification_delay(self, delay: float):
        """Set delay for notifications."""
        self.notification_delay = delay


class EventProcessingTestHelper:
    """Helper class for testing event processing scenarios."""

    @staticmethod
    def create_test_event_sequence(
        symbols: List[str],
        start_date: str = "2024-01-01",
        end_date: str = "2024-01-31"
    ) -> List[Dict[str, Any]]:
        """Create a sequence of test events for processing."""
        events = []
        correlation_id = str(uuid.uuid4())

        for i, symbol in enumerate(symbols):
            events.append({
                'timestamp': (datetime.now() + timedelta(minutes=i)).isoformat(),
                'event_type': 'data_available',
                'correlation_id': correlation_id,
                'dataset': 'equity_bars',
                'symbol': symbol,
                'start': start_date,
                'end': end_date,
                'job_id': f'job_{i:03d}'
            })

        return events

    @staticmethod
    def create_error_event() -> Dict[str, Any]:
        """Create an event that should trigger error handling."""
        return {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'data_available',
            'correlation_id': str(uuid.uuid4()),
            'dataset': 'invalid_dataset',
            'symbol': None,  # Missing required field
            'start': 'invalid_date',
            'end': 'invalid_date'
        }

    @staticmethod
    def create_duplicate_event(original_event: Dict[str, Any]) -> Dict[str, Any]:
        """Create a duplicate event for idempotency testing."""
        duplicate = original_event.copy()
        # Same correlation_id but different timestamp
        duplicate['timestamp'] = (datetime.now() + timedelta(seconds=1)).isoformat()
        return duplicate

    @staticmethod
    def verify_event_ordering(events: List[Dict[str, Any]]) -> bool:
        """Verify that events are in chronological order."""
        timestamps = [datetime.fromisoformat(event['timestamp']) for event in events]
        return timestamps == sorted(timestamps)

    @staticmethod
    def group_events_by_correlation_id(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group events by correlation ID for workflow testing."""
        groups = {}
        for event in events:
            corr_id = event.get('correlation_id')
            if corr_id:
                if corr_id not in groups:
                    groups[corr_id] = []
                groups[corr_id].append(event)
        return groups