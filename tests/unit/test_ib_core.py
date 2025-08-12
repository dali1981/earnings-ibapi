"""
Unit tests for IB API core components.
"""
import pytest
import threading
import time
from unittest.mock import Mock, MagicMock, patch

from api.ib_core import (
    RequestIdSequencer, FutureRegistry, IBFuture, 
    IBWrapperBridge, IBRuntime, ContractDetailsService,
    SecDefService, HistoricalService, make_stock
)
from ibapi.contract import Contract


@pytest.mark.unit
@pytest.mark.api
class TestRequestIdSequencer:
    """Test cases for RequestIdSequencer."""
    
    def test_sequencer_initialization(self):
        """Test sequencer starts in uninitialized state."""
        seq = RequestIdSequencer()
        assert seq._next is None
        assert not seq._ready.is_set()
    
    def test_set_base_and_ready(self):
        """Test setting base ID and ready state."""
        seq = RequestIdSequencer()
        base_id = 1000
        
        seq.set_base(base_id)
        
        assert seq._next == base_id
        assert seq._ready.is_set()
    
    def test_wait_ready_success(self):
        """Test waiting for ready state."""
        seq = RequestIdSequencer()
        
        # Set ready in another thread after short delay
        def set_ready():
            time.sleep(0.1)
            seq.set_base(1000)
        
        thread = threading.Thread(target=set_ready)
        thread.start()
        
        # Should complete without timeout
        seq.wait_ready(timeout=1.0)
        
        thread.join()
        assert seq._next == 1000
    
    def test_wait_ready_timeout(self):
        """Test timeout when waiting for ready state."""
        seq = RequestIdSequencer()
        
        with pytest.raises(RuntimeError, match="IB not ready"):
            seq.wait_ready(timeout=0.1)
    
    def test_next_single(self):
        """Test getting next request ID."""
        seq = RequestIdSequencer()
        seq.set_base(1000)
        
        assert seq.next() == 1000
        assert seq.next() == 1001
        assert seq.next() == 1002
    
    def test_next_multiple(self):
        """Test getting multiple request IDs at once."""
        seq = RequestIdSequencer()
        seq.set_base(2000)
        
        assert seq.next(3) == 2000
        assert seq.next() == 2003
        assert seq.next(2) == 2004
    
    def test_next_uninitialized(self):
        """Test error when getting ID before initialization."""
        seq = RequestIdSequencer()
        
        with pytest.raises(RuntimeError, match="Sequencer not initialized"):
            seq.next()
    
    def test_thread_safety(self):
        """Test thread safety of sequencer."""
        seq = RequestIdSequencer()
        seq.set_base(5000)
        
        results = []
        
        def get_ids():
            for _ in range(10):
                results.append(seq.next())
        
        threads = [threading.Thread(target=get_ids) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have 30 unique IDs
        assert len(results) == 30
        assert len(set(results)) == 30  # All unique
        assert min(results) >= 5000
        assert max(results) < 5030


@pytest.mark.unit
@pytest.mark.api
class TestIBFuture:
    """Test cases for IBFuture."""
    
    def test_future_single_result(self):
        """Test future expecting single result."""
        future = IBFuture(expect_many=False, timeout=1.0)
        
        # Add item and finish
        future.add("test_result")
        future.finish()
        
        result = future.result()
        assert result == "test_result"
    
    def test_future_multiple_results(self):
        """Test future expecting multiple results."""
        future = IBFuture(expect_many=True, timeout=1.0)
        
        # Add multiple items
        future.add("result1")
        future.add("result2")
        future.add("result3")
        future.finish()
        
        result = future.result()
        assert result == ["result1", "result2", "result3"]
    
    def test_future_timeout(self):
        """Test future timeout."""
        future = IBFuture(expect_many=False, timeout=0.1)
        
        # Don't finish the future
        with pytest.raises(TimeoutError):
            future.result()
    
    def test_future_error(self):
        """Test future with error."""
        future = IBFuture(expect_many=False, timeout=1.0)
        
        future.set_error(100, "Test error message")
        
        with pytest.raises(RuntimeError, match="IB error 100: Test error message"):
            future.result()
    
    def test_future_empty_result(self):
        """Test future with no items."""
        future = IBFuture(expect_many=False, timeout=1.0)
        future.finish()
        
        result = future.result()
        assert result is None
    
    def test_future_concurrent_access(self):
        """Test concurrent access to future."""
        future = IBFuture(expect_many=True, timeout=5.0)
        
        def add_items():
            for i in range(100):
                future.add(f"item_{i}")
            future.finish()
        
        thread = threading.Thread(target=add_items)
        thread.start()
        
        result = future.result()
        thread.join()
        
        assert len(result) == 100
        assert all(f"item_{i}" in result for i in range(100))


@pytest.mark.unit
@pytest.mark.api
class TestFutureRegistry:
    """Test cases for FutureRegistry."""
    
    def test_register_future(self):
        """Test registering a future."""
        registry = FutureRegistry()
        future = IBFuture(expect_many=False, timeout=1.0)
        
        result = registry.register(1001, future)
        assert result is future
    
    def test_add_item(self):
        """Test adding item to registered future."""
        registry = FutureRegistry()
        future = IBFuture(expect_many=False, timeout=1.0)
        
        registry.register(1001, future)
        registry.add_item(1001, "test_item")
        
        future.finish()
        result = future.result()
        assert result == "test_item"
    
    def test_finish_future(self):
        """Test finishing registered future."""
        registry = FutureRegistry()
        future = IBFuture(expect_many=False, timeout=1.0)
        
        registry.register(1001, future)
        future.add("test_result")
        registry.finish(1001)
        
        result = future.result()
        assert result == "test_result"
    
    def test_set_error(self):
        """Test setting error on registered future."""
        registry = FutureRegistry()
        future = IBFuture(expect_many=False, timeout=1.0)
        
        registry.register(1001, future)
        registry.set_error(1001, 200, "Test error")
        
        with pytest.raises(RuntimeError, match="IB error 200: Test error"):
            future.result()
    
    def test_unknown_request_id(self):
        """Test operations with unknown request ID."""
        registry = FutureRegistry()
        
        # Should not raise errors
        registry.add_item(9999, "test")
        registry.finish(9999)
        registry.set_error(9999, 300, "Unknown error")
    
    def test_thread_safety(self):
        """Test thread safety of registry."""
        registry = FutureRegistry()
        futures = {}
        
        # Register futures from multiple threads
        def register_futures():
            for i in range(10):
                req_id = 2000 + i
                future = IBFuture(expect_many=False, timeout=5.0)
                futures[req_id] = future
                registry.register(req_id, future)
                registry.add_item(req_id, f"result_{i}")
                registry.finish(req_id)
        
        threads = [threading.Thread(target=register_futures) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify all futures got results
        for req_id, future in futures.items():
            result = future.result()
            assert result.startswith("result_")


@pytest.mark.unit
@pytest.mark.api
class TestIBWrapperBridge:
    """Test cases for IBWrapperBridge."""
    
    def test_wrapper_initialization(self):
        """Test wrapper initialization."""
        seq = RequestIdSequencer()
        reg = FutureRegistry()
        
        wrapper = IBWrapperBridge(seq, reg)
        
        assert wrapper._seq is seq
        assert wrapper._reg is reg
        assert wrapper._subs is not None
    
    def test_next_valid_id_callback(self):
        """Test nextValidId callback."""
        seq = RequestIdSequencer()
        reg = FutureRegistry()
        wrapper = IBWrapperBridge(seq, reg)
        
        wrapper.nextValidId(1500)
        
        assert seq._next == 1500
        assert seq._ready.is_set()
    
    def test_contract_details_callbacks(self):
        """Test contract details callbacks."""
        seq = RequestIdSequencer()
        reg = FutureRegistry()
        wrapper = IBWrapperBridge(seq, reg)
        
        future = IBFuture(expect_many=True, timeout=1.0)
        reg.register(1001, future)
        
        # Simulate contract details response
        mock_details = Mock()
        wrapper.contractDetails(1001, mock_details)
        wrapper.contractDetailsEnd(1001)
        
        result = future.result()
        assert len(result) == 1
        assert result[0] is mock_details
    
    def test_historical_data_callbacks(self):
        """Test historical data callbacks."""
        seq = RequestIdSequencer()
        reg = FutureRegistry()
        wrapper = IBWrapperBridge(seq, reg)
        
        future = IBFuture(expect_many=True, timeout=1.0)
        reg.register(1002, future)
        
        # Simulate historical data response
        mock_bar = Mock()
        mock_bar.date = "20240101"
        mock_bar.open = 100.0
        mock_bar.high = 101.0
        mock_bar.low = 99.0
        mock_bar.close = 100.5
        mock_bar.volume = 1000000
        mock_bar.wap = 100.25
        
        wrapper.historicalData(1002, mock_bar)
        wrapper.historicalDataEnd(1002, "20240101", "20240105")
        
        result = future.result()
        assert len(result) == 1
        assert result[0]["date"] == "20240101"
        assert result[0]["close"] == 100.5
    
    def test_error_callback(self):
        """Test error callback handling."""
        seq = RequestIdSequencer()
        reg = FutureRegistry()
        wrapper = IBWrapperBridge(seq, reg)
        
        future = IBFuture(expect_many=False, timeout=1.0)
        reg.register(1003, future)
        
        wrapper.error(1003, 404, "Not found", "")
        
        with pytest.raises(RuntimeError, match="IB error 404: Not found"):
            future.result()
    
    def test_error_callback_no_request_id(self, caplog):
        """Test error callback with no request ID."""
        seq = RequestIdSequencer()
        reg = FutureRegistry()
        wrapper = IBWrapperBridge(seq, reg)
        
        # Should log warning but not raise
        wrapper.error(-1, 500, "System error", "")
        
        assert "IB error (no reqId)" in caplog.text


@pytest.mark.unit
@pytest.mark.api
class TestIBRuntime:
    """Test cases for IBRuntime."""
    
    def test_runtime_initialization(self):
        """Test runtime initialization."""
        runtime = IBRuntime(host="localhost", port=4002, client_id=123)
        
        assert runtime.host == "localhost"
        assert runtime.port == 4002
        assert runtime.client_id == 123
        assert runtime.sequencer is not None
        assert runtime.registry is not None
        assert runtime.client is not None
    
    @patch('api.ib_core.EClient')
    def test_context_manager(self, mock_client_class):
        """Test runtime as context manager."""
        mock_client = Mock()
        mock_client.isConnected.return_value = False
        mock_client_class.return_value = mock_client
        
        with IBRuntime() as runtime:
            assert runtime is not None
            mock_client.connect.assert_called_once()
        
        mock_client.disconnect.assert_called_once()
    
    @patch('api.ib_core.EClient')
    def test_start_stop(self, mock_client_class):
        """Test manual start/stop."""
        mock_client = Mock()
        mock_client.isConnected.side_effect = [False, True, True]
        mock_client_class.return_value = mock_client
        
        runtime = IBRuntime()
        runtime.sequencer.set_base(1000)  # Mock ready state
        
        runtime.start(ready_timeout=0.1)
        assert mock_client.connect.called
        
        runtime.stop()
        assert mock_client.disconnect.called


@pytest.mark.unit
@pytest.mark.api
class TestServices:
    """Test cases for IB API services."""
    
    def test_make_stock_contract(self):
        """Test stock contract creation utility."""
        contract = make_stock("AAPL", "SMART", "USD")
        
        assert contract.symbol == "AAPL"
        assert contract.secType == "STK"
        assert contract.exchange == "SMART"
        assert contract.currency == "USD"
    
    def test_contract_details_service(self, mock_ib_runtime):
        """Test ContractDetailsService."""
        service = ContractDetailsService(mock_ib_runtime)
        contract = make_stock("AAPL")
        
        # Start the runtime to enable request processing
        mock_ib_runtime.start(ready_timeout=1.0)
        
        try:
            # This will use the mock client
            result = service.fetch(contract, timeout=2.0)
            assert isinstance(result, list)
        finally:
            mock_ib_runtime.stop()
    
    def test_historical_service(self, mock_ib_runtime):
        """Test HistoricalService."""
        service = HistoricalService(mock_ib_runtime)
        contract = make_stock("AAPL")
        
        mock_ib_runtime.start(ready_timeout=1.0)
        
        try:
            result = service.bars(
                contract=contract,
                endDateTime="20240315",
                durationStr="5 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=1,
                timeout=2.0
            )
            assert isinstance(result, list)
            assert len(result) > 0
        finally:
            mock_ib_runtime.stop()
    
    def test_secdef_service(self, mock_ib_runtime):
        """Test SecDefService."""
        service = SecDefService(mock_ib_runtime)
        
        mock_ib_runtime.start(ready_timeout=1.0)
        
        try:
            result = service.option_params(
                symbol="AAPL",
                conId=12345,
                timeout=2.0
            )
            assert isinstance(result, list)
        finally:
            mock_ib_runtime.stop()