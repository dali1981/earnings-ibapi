"""Tests for IB API connection pooling."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
import time

from common.api.connection_pool import (
    IBConnectionPool,
    ConnectionState,
    ConnectionInfo,
    get_connection_pool,
    shutdown_connection_pool,
    pooled_connection
)
from common.exceptions import APIException


class TestConnectionInfo:
    """Test connection info dataclass."""
    
    def test_connection_info_creation(self):
        """Test ConnectionInfo creation."""
        current_time = time.time()
        info = ConnectionInfo(
            client_id=1,
            created_at=current_time,
            last_used=current_time,
            state=ConnectionState.IDLE
        )
        
        assert info.client_id == 1
        assert info.created_at == current_time
        assert info.last_used == current_time
        assert info.state == ConnectionState.IDLE
        assert info.error_count == 0
        assert info.max_errors == 3
    
    def test_connection_info_defaults(self):
        """Test ConnectionInfo default values."""
        info = ConnectionInfo(
            client_id=1,
            created_at=time.time(),
            last_used=time.time(),
            state=ConnectionState.IDLE
        )
        
        assert info.error_count == 0
        assert info.max_errors == 3


class TestConnectionState:
    """Test connection state enum."""
    
    def test_state_values(self):
        """Test connection state enum values."""
        assert ConnectionState.IDLE.value == "idle"
        assert ConnectionState.ACTIVE.value == "active"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.ERROR.value == "error"


class TestIBConnectionPool:
    """Test IB connection pool functionality."""
    
    @pytest.fixture
    def pool_config(self):
        """Connection pool configuration."""
        return {
            'max_connections': 3,
            'min_connections': 1,
            'connection_timeout': 5.0,
            'idle_timeout': 30.0,
            'health_check_interval': 10.0,
            'host': 'localhost',
            'port': 7497
        }
    
    @pytest.fixture
    def connection_pool(self, pool_config):
        """Create connection pool instance."""
        return IBConnectionPool(**pool_config)
    
    @pytest.fixture
    def mock_ib_client(self):
        """Create mock IB client."""
        client = AsyncMock()
        client.connect = AsyncMock()
        client.disconnect = AsyncMock()
        client.is_connected.return_value = True
        return client
    
    def test_pool_initialization(self, connection_pool):
        """Test connection pool initialization."""
        assert connection_pool.max_connections == 3
        assert connection_pool.min_connections == 1
        assert connection_pool.connection_timeout == 5.0
        assert connection_pool.idle_timeout == 30.0
        assert connection_pool.health_check_interval == 10.0
        assert connection_pool._next_client_id == 1
        assert not connection_pool._shutdown
    
    @pytest.mark.asyncio
    async def test_pool_start_stop(self, connection_pool):
        """Test pool start and stop lifecycle."""
        with patch.object(connection_pool, '_ensure_minimum_connections', new_callable=AsyncMock) as mock_ensure:
            await connection_pool.start()
            
            assert connection_pool._health_check_task is not None
            assert connection_pool._cleanup_task is not None
            mock_ensure.assert_called_once()
            
            # Test stop
            await connection_pool.stop()
            assert connection_pool._shutdown is True
    
    @pytest.mark.asyncio
    async def test_create_connection(self, connection_pool, mock_ib_client):
        """Test creating a new connection."""
        with patch('common.api.connection_pool.IBClient', return_value=mock_ib_client):
            client_id, client = await connection_pool._create_connection()
            
            assert client_id == 1
            assert client == mock_ib_client
            assert client_id in connection_pool._connections
            assert client_id in connection_pool._connection_info
            
            # Verify client configuration
            mock_ib_client.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_connection_failure(self, connection_pool):
        """Test connection creation failure."""
        with patch('common.api.connection_pool.IBClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.connect.side_effect = Exception("Connection failed")
            mock_client_class.return_value = mock_client
            
            with pytest.raises(APIException):
                await connection_pool._create_connection()
            
            # Ensure cleanup happened
            assert 1 not in connection_pool._connections
            assert 1 not in connection_pool._connection_info
    
    @pytest.mark.asyncio
    async def test_connection_health_check(self, connection_pool, mock_ib_client):
        """Test connection health checking."""
        # Add a connection to the pool
        connection_pool._connections[1] = mock_ib_client
        connection_pool._connection_info[1] = ConnectionInfo(
            client_id=1,
            created_at=time.time(),
            last_used=time.time(),
            state=ConnectionState.IDLE
        )
        
        # Test healthy connection
        result = await connection_pool._is_connection_healthy(1, mock_ib_client)
        assert result is True
        
        # Test unhealthy connection (too many errors)
        connection_pool._connection_info[1].error_count = 5
        result = await connection_pool._is_connection_healthy(1, mock_ib_client)
        assert result is False
        
        # Test disconnected client
        connection_pool._connection_info[1].error_count = 0
        mock_ib_client.is_connected.return_value = False
        result = await connection_pool._is_connection_healthy(1, mock_ib_client)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_connection_context_manager(self, connection_pool, mock_ib_client):
        """Test getting connection via context manager."""
        with patch.object(connection_pool, '_get_available_connection', 
                         return_value=(1, mock_ib_client)) as mock_get:
            async with connection_pool.get_connection() as client:
                assert client == mock_ib_client
                assert 1 in connection_pool._active_connections
            
            # After context exit, connection should be returned
            mock_get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_connection_timeout(self, connection_pool):
        """Test connection timeout."""
        connection_pool._shutdown = False
        
        # Mock empty queue (no available connections)
        with patch.object(connection_pool._available, 'get', side_effect=asyncio.TimeoutError):
            with patch.object(connection_pool, '_create_connection', 
                            side_effect=asyncio.TimeoutError):
                with pytest.raises(APIException, match="Connection timeout"):
                    async with connection_pool.get_connection(timeout=0.1):
                        pass
    
    @pytest.mark.asyncio
    async def test_get_connection_pool_shutdown(self, connection_pool):
        """Test getting connection when pool is shutdown."""
        connection_pool._shutdown = True
        
        with pytest.raises(APIException, match="Connection pool is shut down"):
            async with connection_pool.get_connection():
                pass
    
    @pytest.mark.asyncio
    async def test_return_healthy_connection(self, connection_pool, mock_ib_client):
        """Test returning a healthy connection to the pool."""
        # Setup connection info
        connection_pool._connections[1] = mock_ib_client
        connection_pool._connection_info[1] = ConnectionInfo(
            client_id=1,
            created_at=time.time(),
            last_used=time.time(),
            state=ConnectionState.ACTIVE
        )
        connection_pool._active_connections.add(1)
        
        await connection_pool._return_connection(1, mock_ib_client)
        
        # Should be returned to available queue
        assert 1 not in connection_pool._active_connections
        assert connection_pool._connection_info[1].state == ConnectionState.IDLE
    
    @pytest.mark.asyncio
    async def test_return_unhealthy_connection(self, connection_pool, mock_ib_client):
        """Test returning an unhealthy connection."""
        # Setup unhealthy connection
        connection_pool._connections[1] = mock_ib_client
        connection_pool._connection_info[1] = ConnectionInfo(
            client_id=1,
            created_at=time.time(),
            last_used=time.time(),
            state=ConnectionState.ERROR,
            error_count=5  # Exceeds max_errors
        )
        connection_pool._active_connections.add(1)
        
        await connection_pool._return_connection(1, mock_ib_client)
        
        # Should be removed, not returned to pool
        assert 1 not in connection_pool._connections
        assert 1 not in connection_pool._connection_info
        assert 1 not in connection_pool._active_connections
    
    @pytest.mark.asyncio
    async def test_mark_connection_error(self, connection_pool):
        """Test marking connection as having an error."""
        # Add connection info
        connection_pool._connection_info[1] = ConnectionInfo(
            client_id=1,
            created_at=time.time(),
            last_used=time.time(),
            state=ConnectionState.IDLE
        )
        
        await connection_pool._mark_connection_error(1)
        
        assert connection_pool._connection_info[1].error_count == 1
        assert connection_pool._connection_info[1].state == ConnectionState.ERROR
    
    @pytest.mark.asyncio
    async def test_remove_connection(self, connection_pool, mock_ib_client):
        """Test removing a connection from the pool."""
        # Add connection
        connection_pool._connections[1] = mock_ib_client
        connection_pool._connection_info[1] = ConnectionInfo(
            client_id=1,
            created_at=time.time(),
            last_used=time.time(),
            state=ConnectionState.IDLE
        )
        connection_pool._active_connections.add(1)
        
        await connection_pool._remove_connection(1)
        
        # Should be completely removed
        assert 1 not in connection_pool._connections
        assert 1 not in connection_pool._connection_info
        assert 1 not in connection_pool._active_connections
        
        # Should have called disconnect
        mock_ib_client.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ensure_minimum_connections(self, connection_pool, mock_ib_client):
        """Test ensuring minimum number of connections."""
        with patch.object(connection_pool, '_create_connection', 
                         return_value=(1, mock_ib_client)) as mock_create:
            with patch.object(connection_pool._available, 'put', new_callable=AsyncMock) as mock_put:
                await connection_pool._ensure_minimum_connections()
                
                # Should create at least min_connections
                assert mock_create.call_count >= connection_pool.min_connections
                assert mock_put.call_count >= connection_pool.min_connections
    
    @pytest.mark.asyncio
    async def test_get_pool_stats(self, connection_pool, mock_ib_client):
        """Test getting pool statistics."""
        # Add some test connections
        current_time = time.time()
        connection_pool._connections[1] = mock_ib_client
        connection_pool._connection_info[1] = ConnectionInfo(
            client_id=1,
            created_at=current_time,
            last_used=current_time,
            state=ConnectionState.IDLE
        )
        connection_pool._active_connections.add(2)
        
        stats = await connection_pool.get_pool_stats()
        
        assert stats['max_connections'] == 3
        assert stats['min_connections'] == 1
        assert stats['total_connections'] == 1
        assert stats['active_connections'] == 1
        assert 'connection_states' in stats
        assert 'total_errors' in stats
        assert 'oldest_connection' in stats
    
    @pytest.mark.asyncio
    async def test_health_check_loop(self, connection_pool, mock_ib_client):
        """Test health check background loop."""
        # Add unhealthy connection
        connection_pool._connections[1] = mock_ib_client
        connection_pool._connection_info[1] = ConnectionInfo(
            client_id=1,
            created_at=time.time(),
            last_used=time.time(),
            state=ConnectionState.IDLE,
            error_count=5  # Exceeds max_errors
        )
        
        with patch.object(connection_pool, '_ensure_minimum_connections', new_callable=AsyncMock):
            # Run one iteration of health check
            await connection_pool._health_check_loop()
            
            # Unhealthy connection should be removed
            assert 1 not in connection_pool._connections
    
    @pytest.mark.asyncio
    async def test_cleanup_loop(self, connection_pool, mock_ib_client):
        """Test idle connection cleanup loop."""
        # Add old idle connection
        old_time = time.time() - connection_pool.idle_timeout - 10
        connection_pool._connections[1] = mock_ib_client
        connection_pool._connection_info[1] = ConnectionInfo(
            client_id=1,
            created_at=old_time,
            last_used=old_time,
            state=ConnectionState.IDLE
        )
        
        # Run cleanup
        # We need to modify the loop to run once
        original_shutdown = connection_pool._shutdown
        connection_pool._shutdown = False
        
        # Patch sleep to make it return immediately and then set shutdown
        async def mock_sleep(duration):
            connection_pool._shutdown = True
        
        with patch('asyncio.sleep', side_effect=mock_sleep):
            await connection_pool._cleanup_loop()
        
        # Restore original state
        connection_pool._shutdown = original_shutdown


class TestPoolUtilityFunctions:
    """Test connection pool utility functions."""
    
    @pytest.mark.asyncio
    async def test_get_connection_pool_creates_instance(self):
        """Test that get_connection_pool creates new instance."""
        config = {'max_connections': 2, 'host': 'localhost'}
        
        with patch('common.api.connection_pool.IBConnectionPool') as mock_pool_class:
            mock_instance = AsyncMock()
            mock_pool_class.return_value = mock_instance
            
            pool = await get_connection_pool(**config)
            
            assert pool == mock_instance
            mock_pool_class.assert_called_once_with(**config)
            mock_instance.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_connection_pool_returns_existing(self):
        """Test that get_connection_pool returns existing instance."""
        # First call creates instance
        config = {'max_connections': 2}
        pool1 = await get_connection_pool(**config)
        
        # Second call returns same instance
        pool2 = await get_connection_pool(**config)
        
        assert pool1 == pool2
    
    @pytest.mark.asyncio
    async def test_shutdown_connection_pool(self):
        """Test shutting down global connection pool."""
        # Create a pool first
        await get_connection_pool(max_connections=2)
        
        # Now shutdown
        with patch('common.api.connection_pool._connection_pool') as mock_pool:
            mock_pool.stop = AsyncMock()
            await shutdown_connection_pool()
            mock_pool.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pooled_connection_context_manager(self):
        """Test pooled connection context manager."""
        config = {'max_connections': 2}
        mock_client = AsyncMock()
        
        with patch('common.api.connection_pool.get_connection_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.get_connection.return_value.__aenter__.return_value = mock_client
            mock_pool.get_connection.return_value.__aexit__.return_value = None
            mock_get_pool.return_value = mock_pool
            
            async with pooled_connection(**config) as client:
                assert client == mock_client
            
            mock_get_pool.assert_called_once_with(**config)
            mock_pool.get_connection.assert_called_once()


class TestConnectionPoolIntegration:
    """Integration tests for connection pool."""
    
    @pytest.fixture
    def integration_pool(self):
        """Create pool for integration testing."""
        return IBConnectionPool(
            max_connections=2,
            min_connections=1,
            connection_timeout=1.0,
            idle_timeout=5.0,
            health_check_interval=2.0
        )
    
    @pytest.mark.asyncio
    async def test_full_lifecycle_with_mock_clients(self, integration_pool):
        """Test full connection lifecycle with mock clients."""
        mock_clients = []
        
        def create_mock_client(**config):
            client = AsyncMock()
            client.connect = AsyncMock()
            client.disconnect = AsyncMock()  
            client.is_connected.return_value = True
            mock_clients.append(client)
            return client
        
        with patch('common.api.connection_pool.IBClient', side_effect=create_mock_client):
            # Start pool
            await integration_pool.start()
            
            try:
                # Get connection
                async with integration_pool.get_connection() as client1:
                    assert client1 in mock_clients
                    assert client1.connect.called
                
                # Get another connection
                async with integration_pool.get_connection() as client2:
                    assert client2 in mock_clients
                
                # Verify pool statistics
                stats = await integration_pool.get_pool_stats()
                assert stats['total_connections'] >= 1
                
            finally:
                # Stop pool
                await integration_pool.stop()
                
                # Verify all clients were disconnected
                for client in mock_clients:
                    client.disconnect.assert_called()
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self, integration_pool):
        """Test error handling in connection operations."""
        def create_failing_client(**config):
            client = AsyncMock()
            client.connect.side_effect = Exception("Network error")
            return client
        
        with patch('common.api.connection_pool.IBClient', side_effect=create_failing_client):
            await integration_pool.start()
            
            try:
                with pytest.raises(APIException):
                    async with integration_pool.get_connection():
                        pass
            finally:
                await integration_pool.stop()
    
    @pytest.mark.asyncio 
    async def test_concurrent_connection_requests(self, integration_pool):
        """Test handling multiple concurrent connection requests."""
        mock_clients = []
        
        def create_mock_client(**config):
            client = AsyncMock()
            client.connect = AsyncMock()
            client.disconnect = AsyncMock()
            client.is_connected.return_value = True
            mock_clients.append(client)
            return client
        
        async def get_and_use_connection(delay=0.1):
            async with integration_pool.get_connection() as client:
                await asyncio.sleep(delay)
                return client
        
        with patch('common.api.connection_pool.IBClient', side_effect=create_mock_client):
            await integration_pool.start()
            
            try:
                # Make multiple concurrent requests
                tasks = [get_and_use_connection(0.1) for _ in range(3)]
                results = await asyncio.gather(*tasks)
                
                # All should succeed
                assert len(results) == 3
                assert all(client in mock_clients for client in results)
                
            finally:
                await integration_pool.stop()