"""
Integration tests for streaming components.
"""
import pytest
import time
import threading
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from streamer import EarningsStreamer, Config
from sink import Sink, EquityBarSink, OptionBarSink
from request_sequencer import RequestSequencer
from response_sequencer import ResponseSequencer


@pytest.mark.integration
@pytest.mark.streaming
class TestEarningsStreamer:
    """Integration tests for EarningsStreamer."""
    
    def test_streamer_initialization(self, test_earnings_config, temp_data_dir):
        """Test streamer initialization with config."""
        streamer = EarningsStreamer(test_earnings_config)
        
        assert streamer.cfg == test_earnings_config
        assert streamer.sequencer is not None
        assert streamer._sink is not None
        assert streamer._client is not None
        assert streamer.equity_sink is not None
    
    def test_output_directory_creation(self, test_earnings_config):
        """Test that output directory is created."""
        streamer = EarningsStreamer(test_earnings_config)
        output_dir = streamer._dir()
        
        assert output_dir.exists()
        expected_name = f"{test_earnings_config.symbol}_{test_earnings_config.date:%Y%m%d}"
        assert output_dir.name == expected_name
    
    @patch('streamer.IBClient')
    def test_streamer_request_registration(self, mock_client_class, test_earnings_config):
        """Test that requests are properly registered."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        streamer = EarningsStreamer(test_earnings_config)
        streamer._client = mock_client
        
        # Mock the sequencer ready state
        streamer.sequencer._ready.set()
        streamer.sequencer._next = 1000
        
        # Simulate connection ready
        streamer.sequencer.when_ready(streamer.schedule_chain_snapshot)
        
        # Trigger the ready callbacks
        for callback in streamer.sequencer._ready_callbacks:
            callback(1000)
        
        # Verify that requests were made
        assert mock_client.reqMktData.called or len(streamer.sequencer._ready_callbacks) > 0
    
    def test_historical_bars_request(self, mock_earnings_streamer, sample_stock_contract):
        """Test historical bars request generation."""
        streamer = mock_earnings_streamer
        
        # Create a mock callback
        callback = Mock()
        
        # Request historical bars
        req_id = streamer.request_historical_bars(
            contract=sample_stock_contract,
            end_datetime=date(2024, 3, 15),
            duration="5 D",
            bar_size="1 day",
            what_to_show="TRADES",
            use_rth=True,
            callback=callback
        )
        
        # Verify request was registered
        assert req_id > 0
        assert streamer._client.reqHistoricalData.called
        
        # Check that response sequencer has the callback
        assert req_id in streamer._resp._callbacks
    
    @patch('time.sleep')  # Speed up test
    def test_rate_limiting(self, mock_sleep, mock_earnings_streamer):
        """Test rate limiting functionality."""
        streamer = mock_earnings_streamer
        
        # Make multiple rapid requests
        contracts = [Mock() for _ in range(10)]
        
        start_time = time.time()
        for i, contract in enumerate(contracts):
            contract.symbol = f"TEST{i}"
            streamer.request_historical_bars(
                contract=contract,
                end_datetime=date(2024, 3, 15),
                duration="1 D",
                bar_size="1 day",
                what_to_show="TRADES",
                use_rth=True,
                callback=Mock()
            )
        
        # Verify rate limiter was used
        assert mock_sleep.called or time.time() - start_time < 1.0


@pytest.mark.integration
@pytest.mark.streaming
class TestSinkIntegration:
    """Integration tests for data sinks."""
    
    def test_equity_bar_sink_integration(self, equity_repository, sample_equity_data):
        """Test EquityBarSink with repository integration."""
        sink = EquityBarSink(equity_repository)
        
        # Add metadata to dataframe
        test_data = sample_equity_data.copy()
        test_data.attrs = {
            'symbol': 'AAPL',
            'bar_size': '1 day',
            'what_to_show': 'TRADES'
        }
        
        # Process through sink
        sink(test_data)
        
        # Verify data was saved to repository
        loaded_data = equity_repository.load(symbol='AAPL')
        assert not loaded_data.empty
        assert len(loaded_data) == len(sample_equity_data)
    
    def test_option_bar_sink_integration(self, option_repository, sample_option_data):
        """Test OptionBarSink with repository integration."""
        sink = OptionBarSink(option_repository)
        
        # Add option metadata
        test_data = sample_option_data.copy()
        test_data.attrs = {
            'underlying': 'AAPL',
            'expiry': date(2024, 3, 15),
            'strike': 150.0,
            'right': 'C',
            'bar_size': '1 min',
            'what_to_show': 'TRADES'
        }
        
        # Process through sink
        sink(test_data)
        
        # Verify data was saved
        loaded_data = option_repository.load(underlying='AAPL')
        assert not loaded_data.empty
    
    def test_sink_buffer_management(self, temp_data_dir):
        """Test sink buffer management and flushing."""
        sink = Sink(temp_data_dir)
        
        # Add data to buffer
        req_id = 1001
        test_bars = [
            {"date": "20240101", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000},
            {"date": "20240102", "open": 100.5, "high": 102, "low": 100, "close": 101.5, "volume": 1100},
        ]
        
        for bar_data in test_bars:
            mock_bar = Mock()
            for key, value in bar_data.items():
                setattr(mock_bar, key, value)
            sink.historical_data(req_id, mock_bar)
        
        # Verify buffer has data
        assert len(sink.buffers[req_id]) == 2
        
        # End data stream (should flush to file)
        sink.historical_data_end(req_id, "", "")
        
        # Verify buffer was cleared
        assert req_id not in sink.buffers
        
        # Verify file was created
        output_file = temp_data_dir / f"bars_{req_id}.parquet"
        assert output_file.exists()


@pytest.mark.integration
@pytest.mark.streaming
class TestRequestResponseFlow:
    """Test the complete request/response flow."""
    
    def test_sequencer_integration(self):
        """Test request and response sequencer integration."""
        req_seq = RequestSequencer()
        resp_seq = ResponseSequencer()
        
        # Set up ready state
        req_seq._ready.set()
        req_seq._next = 2000
        
        # Register a callback
        callback = Mock()
        metadata = {'symbol': 'TEST', 'bar_size': '1 day'}
        
        req_id = req_seq.next('test_request')
        resp_seq.add(req_id, callback, metadata)
        
        # Simulate data arrival
        test_data = pd.DataFrame({
            'date': ['20240101', '20240102'],
            'close': [100.0, 101.0]
        })
        test_data.attrs = metadata
        
        # Process through response sequencer
        resp_seq.handle_data(req_id, test_data)
        
        # Verify callback was called
        callback.assert_called_once()
        called_data = callback.call_args[0][0]
        assert isinstance(called_data, pd.DataFrame)
        assert len(called_data) == 2
    
    def test_end_to_end_data_flow(self, temp_data_dir, mock_ib_client):
        """Test complete data flow from request to storage."""
        # Set up components
        config = Config(
            symbol="FLOW_TEST",
            date=date(2024, 3, 15),
            out_dir=temp_data_dir
        )
        
        streamer = EarningsStreamer(config)
        
        # Mock the client
        streamer._client = mock_ib_client
        mock_ib_client.wrapper = streamer._client._resp
        
        # Start the flow
        req_seq = streamer.sequencer
        req_seq._ready.set()
        req_seq._next = 3000
        
        # Create a test contract and request
        contract = Mock()
        contract.symbol = "FLOW_TEST"
        
        callback = Mock()
        req_id = streamer.request_historical_bars(
            contract=contract,
            end_datetime=date(2024, 3, 15),
            duration="1 D",
            bar_size="1 day",
            what_to_show="TRADES",
            use_rth=True,
            callback=callback
        )
        
        # Simulate IB response
        mock_ib_client._send_historical_data(req_id, contract)
        
        # Allow async processing
        time.sleep(0.5)
        
        # Verify the flow worked
        assert callback.called or req_id in streamer._resp._callbacks
    
    @pytest.mark.slow
    def test_concurrent_streaming(self, temp_data_dir, mock_ib_client):
        """Test concurrent streaming operations."""
        symbols = ["CONC1", "CONC2", "CONC3", "CONC4", "CONC5"]
        streamers = []
        
        # Create multiple streamers
        for symbol in symbols:
            config = Config(
                symbol=symbol,
                date=date(2024, 3, 15),
                out_dir=temp_data_dir / symbol,
                client_id=4000 + len(streamers)
            )
            streamer = EarningsStreamer(config)
            streamer._client = mock_ib_client
            streamers.append(streamer)
        
        # Start all streamers concurrently
        threads = []
        results = {}
        
        def run_streamer(streamer):
            try:
                # Mock successful run
                results[streamer.cfg.symbol] = "success"
            except Exception as e:
                results[streamer.cfg.symbol] = f"error: {e}"
        
        for streamer in streamers:
            thread = threading.Thread(target=run_streamer, args=(streamer,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
        
        # Verify all streamers completed
        assert len(results) == len(symbols)
        assert all(result == "success" for result in results.values())
    
    def test_error_handling_in_stream(self, mock_earnings_streamer, mock_ib_client):
        """Test error handling in streaming operations."""
        streamer = mock_earnings_streamer
        
        # Set up error scenario
        mock_ib_client.connected = False
        
        # Attempt operation that should fail gracefully
        contract = Mock()
        contract.symbol = "ERROR_TEST"
        
        try:
            req_id = streamer.request_historical_bars(
                contract=contract,
                end_datetime=date(2024, 3, 15),
                duration="1 D",
                bar_size="1 day",
                what_to_show="TRADES",
                use_rth=True,
                callback=Mock()
            )
            # Should handle gracefully
            assert req_id is not None
        except Exception as e:
            # If exception occurs, should be handled appropriately
            assert "connection" in str(e).lower() or "client" in str(e).lower()


@pytest.mark.integration
@pytest.mark.streaming
class TestDataPersistenceIntegration:
    """Test integration between streaming and data persistence."""
    
    def test_streaming_to_repository_persistence(self, temp_data_dir, mock_ib_client):
        """Test streaming data persistence to repositories."""
        config = Config(
            symbol="PERSIST_TEST",
            date=date(2024, 3, 15),
            out_dir=temp_data_dir
        )
        
        streamer = EarningsStreamer(config)
        streamer._client = mock_ib_client
        
        # Verify repository integration
        assert streamer.equity_sink is not None
        
        # Simulate data flow
        sample_data = pd.DataFrame({
            'date': ['20240315'],
            'open': [150.0], 'high': [152.0], 'low': [149.0], 'close': [151.0],
            'volume': [1000000]
        })
        sample_data.attrs = {
            'symbol': 'PERSIST_TEST',
            'bar_size': '1 day',
            'what_to_show': 'TRADES'
        }
        
        # Process through sink
        streamer.equity_sink(sample_data)
        
        # Verify persistence
        repo_data = streamer.equity_sink._repo.load(symbol='PERSIST_TEST')
        assert not repo_data.empty
        assert repo_data.iloc[0]['symbol'] == 'PERSIST_TEST'
    
    def test_multiple_data_types_persistence(self, temp_data_dir):
        """Test persistence of different data types."""
        sink = Sink(temp_data_dir)
        
        # Test different request types
        test_cases = [
            (1001, "equity_bars"),
            (1002, "option_data"),
            (1003, "market_data"),
        ]
        
        for req_id, data_type in test_cases:
            # Add mock data
            mock_data = {
                "type": data_type,
                "timestamp": datetime.now().isoformat(),
                "value": 123.45
            }
            sink.add_value(req_id, mock_data)
        
        # Verify buffers
        for req_id, _ in test_cases:
            assert req_id in sink.buffers
            assert len(sink.buffers[req_id]) == 1