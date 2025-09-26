"""
Test suite for observability implementation.
Tests all major components of the Sentry integration and custom metrics.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd

from common.observability import (
    init_observability,
    capture_trading_event,
    capture_trading_exception,
    track_trading_operation,
    TradingMetrics,
    get_trading_metrics,
    trace_operation,
    ObservabilityContext,
    trading_context,
    monitor_repository_operation,
    monitor_api_endpoint,
)


class TestObservabilityCore:
    """Test core observability functionality."""
    
    @patch('common.observability.core.sentry_sdk')
    def test_init_observability(self, mock_sentry):
        """Test Sentry initialization with proper configuration."""
        # Test with DSN configured
        with patch('common.observability.core.get_settings') as mock_settings:
            mock_obs_config = Mock()
            mock_obs_config.sentry_dsn = "https://test@sentry.io/123"
            mock_obs_config.enable_error_tracking = True
            mock_obs_config.enable_performance_monitoring = True
            mock_obs_config.enable_celery_monitoring = True
            mock_obs_config.sentry_environment = "test"
            mock_obs_config.sentry_release = "1.0.0"
            mock_obs_config.sentry_sample_rate = 1.0
            mock_obs_config.sentry_traces_sample_rate = 0.1
            mock_obs_config.sentry_profiles_sample_rate = 0.1
            mock_obs_config.sentry_max_breadcrumbs = 50
            mock_obs_config.sentry_attach_stacktrace = True
            mock_obs_config.sentry_send_default_pii = False
            mock_obs_config.sentry_debug = False
            
            mock_settings_obj = Mock()
            mock_settings_obj.observability = mock_obs_config
            mock_settings_obj.env.value = "test"
            mock_settings_obj.paths.data_root = "/test/data"
            mock_settings_obj.ib.host = "127.0.0.1"
            mock_settings_obj.ib.port = 4002
            
            mock_settings.return_value = mock_settings_obj
            
            init_observability("test-service", "1.0.0")
            
            # Verify Sentry was initialized
            mock_sentry.init.assert_called_once()
            args, kwargs = mock_sentry.init.call_args
            
            assert kwargs['dsn'] == "https://test@sentry.io/123"
            assert kwargs['environment'] == "test"
            assert kwargs['release'] == "1.0.0"
            assert len(kwargs['integrations']) >= 2  # At least logging and FastAPI
    
    @patch('common.observability.core.sentry_sdk')
    def test_init_observability_no_dsn(self, mock_sentry):
        """Test initialization skips when DSN not configured."""
        with patch('common.observability.core.get_settings') as mock_settings:
            mock_obs_config = Mock()
            mock_obs_config.sentry_dsn = None
            
            mock_settings_obj = Mock()
            mock_settings_obj.observability = mock_obs_config
            
            mock_settings.return_value = mock_settings_obj
            
            init_observability("test-service", "1.0.0")
            
            # Verify Sentry was not initialized
            mock_sentry.init.assert_not_called()
    
    @patch('common.observability.core.sentry_sdk')
    def test_capture_trading_event(self, mock_sentry):
        """Test trading event capture with context."""
        capture_trading_event(
            "Test event",
            level="info",
            extra={"symbol": "AAPL", "operation": "test"},
            tags={"data_source": "test"}
        )
        
        mock_sentry.capture_message.assert_called_once_with("Test event", level="info")
    
    @patch('common.observability.core.sentry_sdk')
    def test_capture_trading_exception(self, mock_sentry):
        """Test trading exception capture with context."""
        test_exception = ValueError("Test error")
        
        capture_trading_exception(
            test_exception,
            extra={"symbol": "AAPL"},
            tags={"operation": "test"}
        )
        
        mock_sentry.capture_exception.assert_called_once_with(test_exception)


class TestTradingMetrics:
    """Test custom trading metrics."""
    
    def test_trading_metrics_initialization(self):
        """Test TradingMetrics initializes correctly."""
        with patch('common.observability.metrics.get_settings') as mock_settings:
            mock_obs_config = Mock()
            mock_obs_config.enable_custom_metrics = True
            mock_obs_config.metrics_sample_rate = 1.0
            
            mock_settings_obj = Mock()
            mock_settings_obj.observability = mock_obs_config
            
            mock_settings.return_value = mock_settings_obj
            
            metrics = TradingMetrics()
            
            assert metrics._enabled is True
            assert metrics._sample_rate == 1.0
    
    def test_record_operation_success(self):
        """Test recording successful operation."""
        from common.observability.metrics import TradingOperationResult
        
        with patch('common.observability.metrics.get_settings') as mock_settings:
            mock_obs_config = Mock()
            mock_obs_config.enable_custom_metrics = True
            mock_obs_config.metrics_sample_rate = 1.0
            
            mock_settings_obj = Mock()
            mock_settings_obj.observability = mock_obs_config
            
            mock_settings.return_value = mock_settings_obj
            
            metrics = TradingMetrics()
            
            result = TradingOperationResult(
                operation="test_operation",
                symbol="AAPL",
                success=True,
                duration_ms=100.0,
                records_processed=50,
                data_source="test"
            )
            
            metrics.record_operation(result)
            
            # Verify local storage
            assert "test_operation:AAPL" in metrics._operation_counts
            assert metrics._operation_counts["test_operation:AAPL"] == 1
            assert len(metrics._recent_operations) == 1
    
    @patch('common.observability.metrics.sentry_sdk')
    def test_track_trading_operation_success(self, mock_sentry):
        """Test context manager for tracking operations."""
        mock_hub = Mock()
        mock_sentry.Hub.current = mock_hub
        mock_hub.client = Mock()
        
        with patch('common.observability.metrics.get_settings') as mock_settings:
            mock_obs_config = Mock()
            mock_obs_config.enable_custom_metrics = True
            mock_obs_config.metrics_sample_rate = 1.0
            
            mock_settings_obj = Mock()
            mock_settings_obj.observability = mock_obs_config
            
            mock_settings.return_value = mock_settings_obj
            
            with track_trading_operation("test_op", symbol="AAPL", data_source="test"):
                # Simulate some work
                pass
            
            # Verify metrics were recorded
            metrics = get_trading_metrics()
            assert len(metrics._recent_operations) >= 1
    
    @patch('common.observability.metrics.sentry_sdk')
    def test_track_trading_operation_exception(self, mock_sentry):
        """Test context manager handles exceptions properly."""
        mock_hub = Mock()
        mock_sentry.Hub.current = mock_hub
        mock_hub.client = Mock()
        
        with patch('common.observability.metrics.get_settings') as mock_settings:
            mock_obs_config = Mock()
            mock_obs_config.enable_custom_metrics = True
            mock_obs_config.metrics_sample_rate = 1.0
            
            mock_settings_obj = Mock()
            mock_settings_obj.observability = mock_obs_config
            
            mock_settings.return_value = mock_settings_obj
            
            with pytest.raises(ValueError):
                with track_trading_operation("test_op", symbol="AAPL"):
                    raise ValueError("Test error")
            
            # Verify error was captured
            mock_sentry.capture_exception.assert_called()
            
            # Verify metrics recorded the failure
            metrics = get_trading_metrics()
            recent_op = metrics._recent_operations[-1]
            assert recent_op['success'] is False
            assert recent_op['error_type'] == 'ValueError'


class TestDistributedTracing:
    """Test distributed tracing functionality."""
    
    @patch('common.observability.tracing.sentry_sdk')
    def test_trace_operation_decorator(self, mock_sentry):
        """Test trace_operation decorator."""
        with patch('common.observability.tracing.get_settings') as mock_settings:
            mock_obs_config = Mock()
            mock_obs_config.enable_distributed_tracing = True
            
            mock_settings_obj = Mock()
            mock_settings_obj.observability = mock_obs_config
            
            mock_settings.return_value = mock_settings_obj
            
            mock_transaction = Mock()
            mock_sentry.start_transaction.return_value.__enter__.return_value = mock_transaction
            
            @trace_operation("test_operation", "Test operation description")
            def test_function(arg1, symbol="AAPL"):
                return f"processed {symbol}"
            
            result = test_function("test", symbol="AAPL")
            
            # Verify transaction was started
            mock_sentry.start_transaction.assert_called_once()
            
            # Verify result passed through
            assert result == "processed AAPL"
            
            # Verify tags were set
            mock_transaction.set_tag.assert_called()
    
    @patch('common.observability.tracing.sentry_sdk')
    def test_trace_span_context_manager(self, mock_sentry):
        """Test trace_span context manager."""
        with patch('common.observability.tracing.get_settings') as mock_settings:
            mock_obs_config = Mock()
            mock_obs_config.enable_distributed_tracing = True
            
            mock_settings_obj = Mock()
            mock_settings_obj.observability = mock_obs_config
            
            mock_settings.return_value = mock_settings_obj
            
            mock_span = Mock()
            mock_sentry.start_span.return_value.__enter__.return_value = mock_span
            
            from common.observability.tracing import trace_span
            
            with trace_span("database_query", tags={"symbol": "AAPL"}):
                # Simulate database work
                pass
            
            # Verify span was created
            mock_sentry.start_span.assert_called_once()
            mock_span.set_tag.assert_called()
            mock_span.set_status.assert_called_with("ok")


class TestObservabilityContext:
    """Test observability context management."""
    
    def test_observability_context_basic(self):
        """Test basic context functionality."""
        context = ObservabilityContext(
            correlation_id="test-123",
            symbol="AAPL",
            operation="test_op"
        )
        
        assert context.correlation_id == "test-123"
        assert context.symbol == "AAPL"
        assert context.operation == "test_op"
    
    @patch('common.observability.context.sentry_sdk')
    def test_trading_context(self, mock_sentry):
        """Test trading context manager."""
        with trading_context("AAPL", "bars_fetch", correlation_id="test-123"):
            # Context should be set
            pass
        
        # Verify Sentry scope was configured
        mock_sentry.configure_scope.assert_called()


class TestDecorators:
    """Test observability decorators."""
    
    @patch('common.observability.decorators.track_trading_operation')
    @patch('common.observability.decorators.trace_operation')
    def test_monitor_repository_operation(self, mock_trace, mock_track):
        """Test repository operation monitoring decorator."""
        mock_track.return_value.__enter__ = Mock()
        mock_track.return_value.__exit__ = Mock()
        mock_trace.return_value.__enter__ = Mock()
        mock_trace.return_value.__exit__ = Mock()
        
        @monitor_repository_operation(
            "test_save",
            extract_symbol=lambda args, kwargs: kwargs.get('symbol')
        )
        def save_data(df, symbol=None):
            return f"saved {symbol}"
        
        result = save_data(pd.DataFrame(), symbol="AAPL")
        
        # Verify tracking was initiated
        mock_track.assert_called_once()
        mock_trace.assert_called_once()
        
        assert result == "saved AAPL"
    
    @patch('common.observability.decorators.track_trading_operation')
    @patch('common.observability.decorators.trace_operation')
    def test_monitor_api_endpoint(self, mock_trace, mock_track):
        """Test API endpoint monitoring decorator."""
        mock_track.return_value.__enter__ = Mock()
        mock_track.return_value.__exit__ = Mock()
        mock_trace.return_value.__enter__ = Mock()
        mock_trace.return_value.__exit__ = Mock()
        
        @monitor_api_endpoint(
            "test_api",
            extract_symbol=lambda args, kwargs: args[0] if args else None
        )
        def api_endpoint(symbol):
            return f"data for {symbol}"
        
        result = api_endpoint("AAPL")
        
        # Verify monitoring was applied
        mock_track.assert_called_once()
        mock_trace.assert_called_once()
        
        assert result == "data for AAPL"


class TestIntegration:
    """Integration tests for observability system."""
    
    @patch('common.observability.core.sentry_sdk')
    def test_full_request_flow(self, mock_sentry):
        """Test complete request flow with observability."""
        # Mock Sentry components
        mock_hub = Mock()
        mock_sentry.Hub.current = mock_hub
        mock_hub.client = Mock()
        
        mock_transaction = Mock()
        mock_sentry.start_transaction.return_value.__enter__.return_value = mock_transaction
        
        with patch('common.observability.core.get_settings') as mock_settings:
            # Configure observability
            mock_obs_config = Mock()
            mock_obs_config.sentry_dsn = "https://test@sentry.io/123"
            mock_obs_config.enable_error_tracking = True
            mock_obs_config.enable_performance_monitoring = True
            mock_obs_config.enable_distributed_tracing = True
            mock_obs_config.enable_custom_metrics = True
            mock_obs_config.sentry_environment = "test"
            mock_obs_config.sentry_release = "1.0.0"
            mock_obs_config.sentry_sample_rate = 1.0
            mock_obs_config.sentry_traces_sample_rate = 1.0
            mock_obs_config.sentry_profiles_sample_rate = 1.0
            mock_obs_config.sentry_max_breadcrumbs = 50
            mock_obs_config.sentry_attach_stacktrace = True
            mock_obs_config.sentry_send_default_pii = False
            mock_obs_config.sentry_debug = False
            mock_obs_config.metrics_sample_rate = 1.0
            
            mock_settings_obj = Mock()
            mock_settings_obj.observability = mock_obs_config
            mock_settings_obj.env.value = "test"
            mock_settings_obj.paths.data_root = "/test"
            mock_settings_obj.ib.host = "127.0.0.1"
            mock_settings_obj.ib.port = 4002
            
            mock_settings.return_value = mock_settings_obj
            
            # Initialize observability
            init_observability("test-service", "1.0.0")
            
            # Simulate a complete request flow
            with trading_context("AAPL", "bars_fetch", correlation_id="req-123"):
                with track_trading_operation("api_request", symbol="AAPL", data_source="test"):
                    # Simulate API work
                    result = "success"
            
            # Verify components were called
            mock_sentry.init.assert_called_once()
            mock_sentry.configure_scope.assert_called()
            
            # Verify metrics were recorded
            metrics = get_trading_metrics()
            assert len(metrics._recent_operations) >= 1
    
    def test_error_handling_flow(self):
        """Test error handling with observability."""
        with patch('common.observability.core.sentry_sdk') as mock_sentry:
            mock_hub = Mock()
            mock_sentry.Hub.current = mock_hub
            mock_hub.client = Mock()
            
            with patch('common.observability.metrics.get_settings') as mock_settings:
                mock_obs_config = Mock()
                mock_obs_config.enable_custom_metrics = True
                mock_obs_config.metrics_sample_rate = 1.0
                
                mock_settings_obj = Mock()
                mock_settings_obj.observability = mock_obs_config
                
                mock_settings.return_value = mock_settings_obj
                
                # Test error capture and metrics
                test_error = ValueError("Test error")
                
                try:
                    with track_trading_operation("error_test", symbol="AAPL"):
                        raise test_error
                except ValueError:
                    pass
                
                # Verify error was captured
                mock_sentry.capture_exception.assert_called_with(test_error)
                
                # Verify failure metrics
                metrics = get_trading_metrics()
                recent_op = metrics._recent_operations[-1]
                assert recent_op['success'] is False
                assert recent_op['error_type'] == 'ValueError'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])