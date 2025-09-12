"""Connection pooling for IB API clients."""

import asyncio
import logging
import time
from typing import Dict, Optional, Set, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

from ..exceptions import APIException, ConfigurationException
from ..decorators import circuit_breaker

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration."""
    IDLE = "idle"
    ACTIVE = "active"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class ConnectionInfo:
    """Information about a pooled connection."""
    client_id: int
    created_at: float
    last_used: float
    state: ConnectionState
    error_count: int = 0
    max_errors: int = 3


class IBConnectionPool:
    """Pool of IB API connections for reuse and performance optimization."""
    
    def __init__(
        self,
        max_connections: int = 5,
        min_connections: int = 1,
        connection_timeout: float = 30.0,
        idle_timeout: float = 300.0,  # 5 minutes
        health_check_interval: float = 60.0,  # 1 minute
        **ib_config
    ):
        """Initialize connection pool.
        
        Args:
            max_connections: Maximum number of connections in pool
            min_connections: Minimum number of connections to maintain
            connection_timeout: Timeout for establishing connections
            idle_timeout: Time before idle connections are closed
            health_check_interval: Interval for connection health checks
            **ib_config: IB API configuration (host, port, etc.)
        """
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        self.health_check_interval = health_check_interval
        self.ib_config = ib_config
        
        # Connection management
        self._available: asyncio.Queue = asyncio.Queue(maxsize=max_connections)
        self._connections: Dict[int, Any] = {}  # client_id -> client
        self._connection_info: Dict[int, ConnectionInfo] = {}
        self._active_connections: Set[int] = set()
        
        # Pool state
        self._pool_lock = asyncio.Lock()
        self._shutdown = False
        self._next_client_id = 1
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(
            f"Initialized IB connection pool: "
            f"max={max_connections}, min={min_connections}, "
            f"timeout={connection_timeout}s"
        )
    
    async def start(self) -> None:
        """Start the connection pool and background tasks."""
        if self._health_check_task is not None:
            logger.warning("Connection pool already started")
            return
        
        # Create initial minimum connections
        await self._ensure_minimum_connections()
        
        # Start background tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Connection pool started")
    
    async def stop(self) -> None:
        """Stop the connection pool and cleanup all connections."""
        self._shutdown = True
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self._pool_lock:
            for client_id, client in self._connections.items():
                try:
                    await self._disconnect_client(client)
                    logger.debug(f"Closed connection {client_id}")
                except Exception as e:
                    logger.error(f"Error closing connection {client_id}: {e}")
            
            self._connections.clear()
            self._connection_info.clear()
            self._active_connections.clear()
        
        logger.info("Connection pool stopped")
    
    @asynccontextmanager
    async def get_connection(self, timeout: Optional[float] = None):
        """Get a connection from the pool.
        
        Args:
            timeout: Timeout for getting connection (uses default if None)
            
        Yields:
            IB API client connection
            
        Raises:
            APIException: If unable to get connection
        """
        if self._shutdown:
            raise APIException("Connection pool is shut down")
        
        effective_timeout = timeout or self.connection_timeout
        client = None
        client_id = None
        
        try:
            # Get connection from pool
            client_id, client = await self._get_available_connection(effective_timeout)
            
            # Mark as active
            async with self._pool_lock:
                self._active_connections.add(client_id)
                if client_id in self._connection_info:
                    self._connection_info[client_id].last_used = time.time()
                    self._connection_info[client_id].state = ConnectionState.ACTIVE
            
            logger.debug(f"Acquired connection {client_id}")
            yield client
            
        except Exception as e:
            logger.error(f"Error using connection {client_id}: {e}")
            if client_id is not None:
                await self._mark_connection_error(client_id)
            raise
            
        finally:
            # Return connection to pool
            if client_id is not None:
                await self._return_connection(client_id, client)
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics.
        
        Returns:
            Dictionary with pool statistics
        """
        async with self._pool_lock:
            return {
                "max_connections": self.max_connections,
                "min_connections": self.min_connections,
                "total_connections": len(self._connections),
                "active_connections": len(self._active_connections),
                "available_connections": self._available.qsize(),
                "connection_states": {
                    state.value: sum(
                        1 for info in self._connection_info.values()
                        if info.state == state
                    )
                    for state in ConnectionState
                },
                "total_errors": sum(
                    info.error_count for info in self._connection_info.values()
                ),
                "oldest_connection": min(
                    (info.created_at for info in self._connection_info.values()),
                    default=None
                )
            }
    
    async def _get_available_connection(self, timeout: float) -> tuple[int, Any]:
        """Get an available connection from the pool."""
        try:
            # Try to get existing connection
            client_id = await asyncio.wait_for(
                self._available.get(), timeout=min(timeout, 1.0)
            )
            client = self._connections.get(client_id)
            
            if client and await self._is_connection_healthy(client_id, client):
                return client_id, client
            else:
                # Connection is unhealthy, create new one
                await self._remove_connection(client_id)
                
        except asyncio.TimeoutError:
            pass
        
        # Create new connection if pool not full
        async with self._pool_lock:
            if len(self._connections) < self.max_connections:
                return await self._create_connection()
        
        # Wait for available connection with full timeout
        try:
            client_id = await asyncio.wait_for(
                self._available.get(), timeout=timeout
            )
            client = self._connections.get(client_id)
            
            if client and await self._is_connection_healthy(client_id, client):
                return client_id, client
            else:
                await self._remove_connection(client_id)
                raise APIException("No healthy connections available")
                
        except asyncio.TimeoutError:
            raise APIException(f"Connection timeout after {timeout}s")
    
    async def _create_connection(self) -> tuple[int, Any]:
        """Create a new connection."""
        client_id = self._next_client_id
        self._next_client_id += 1
        
        try:
            # Import here to avoid circular imports
            from .ib_client import IBClient
            
            # Create client with pool-specific client ID
            config = self.ib_config.copy()
            config['client_id'] = client_id
            
            client = IBClient(**config)
            
            # Connect with timeout
            await asyncio.wait_for(
                client.connect(),
                timeout=self.connection_timeout
            )
            
            # Store connection info
            self._connections[client_id] = client
            self._connection_info[client_id] = ConnectionInfo(
                client_id=client_id,
                created_at=time.time(),
                last_used=time.time(),
                state=ConnectionState.IDLE
            )
            
            logger.info(f"Created new connection {client_id}")
            return client_id, client
            
        except Exception as e:
            logger.error(f"Failed to create connection {client_id}: {e}")
            # Cleanup partial connection
            if client_id in self._connections:
                del self._connections[client_id]
            if client_id in self._connection_info:
                del self._connection_info[client_id]
            raise APIException(f"Failed to create connection: {e}")
    
    async def _return_connection(self, client_id: int, client: Any) -> None:
        """Return a connection to the pool."""
        async with self._pool_lock:
            self._active_connections.discard(client_id)
            
            if client_id in self._connection_info:
                self._connection_info[client_id].state = ConnectionState.IDLE
            
            # Only return healthy connections to the pool
            if await self._is_connection_healthy(client_id, client):
                try:
                    await self._available.put(client_id)
                    logger.debug(f"Returned connection {client_id} to pool")
                except asyncio.QueueFull:
                    # Pool is full, close connection
                    await self._remove_connection(client_id)
                    logger.debug(f"Pool full, closed connection {client_id}")
            else:
                await self._remove_connection(client_id)
                logger.debug(f"Removed unhealthy connection {client_id}")
    
    async def _is_connection_healthy(self, client_id: int, client: Any) -> bool:
        """Check if a connection is healthy."""
        try:
            info = self._connection_info.get(client_id)
            if not info:
                return False
            
            # Check error count
            if info.error_count >= info.max_errors:
                return False
            
            # Check if client is connected
            if hasattr(client, 'is_connected') and not client.is_connected():
                return False
            
            # Additional health checks can be added here
            return True
            
        except Exception as e:
            logger.debug(f"Health check failed for connection {client_id}: {e}")
            return False
    
    async def _mark_connection_error(self, client_id: int) -> None:
        """Mark a connection as having an error."""
        async with self._pool_lock:
            if client_id in self._connection_info:
                self._connection_info[client_id].error_count += 1
                self._connection_info[client_id].state = ConnectionState.ERROR
                logger.warning(f"Connection {client_id} error count: {self._connection_info[client_id].error_count}")
    
    async def _remove_connection(self, client_id: int) -> None:
        """Remove a connection from the pool."""
        async with self._pool_lock:
            client = self._connections.pop(client_id, None)
            self._connection_info.pop(client_id, None)
            self._active_connections.discard(client_id)
            
            if client:
                try:
                    await self._disconnect_client(client)
                except Exception as e:
                    logger.error(f"Error disconnecting client {client_id}: {e}")
            
            logger.debug(f"Removed connection {client_id}")
    
    async def _disconnect_client(self, client: Any) -> None:
        """Disconnect a client."""
        try:
            if hasattr(client, 'disconnect'):
                await client.disconnect()
            elif hasattr(client, 'close'):
                await client.close()
        except Exception as e:
            logger.debug(f"Error during client disconnect: {e}")
    
    async def _ensure_minimum_connections(self) -> None:
        """Ensure minimum number of connections are available."""
        async with self._pool_lock:
            current_count = len(self._connections)
            needed = max(0, self.min_connections - current_count)
            
            for _ in range(needed):
                try:
                    client_id, client = await self._create_connection()
                    await self._available.put(client_id)
                except Exception as e:
                    logger.error(f"Failed to create minimum connection: {e}")
                    break
    
    async def _health_check_loop(self) -> None:
        """Background task for connection health checks."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                if self._shutdown:
                    break
                
                # Check all connections
                unhealthy_connections = []
                async with self._pool_lock:
                    for client_id, client in self._connections.items():
                        if client_id not in self._active_connections:
                            if not await self._is_connection_healthy(client_id, client):
                                unhealthy_connections.append(client_id)
                
                # Remove unhealthy connections
                for client_id in unhealthy_connections:
                    await self._remove_connection(client_id)
                
                # Ensure minimum connections
                await self._ensure_minimum_connections()
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up idle connections."""
        while not self._shutdown:
            try:
                await asyncio.sleep(60.0)  # Check every minute
                
                if self._shutdown:
                    break
                
                current_time = time.time()
                idle_connections = []
                
                async with self._pool_lock:
                    for client_id, info in self._connection_info.items():
                        if (client_id not in self._active_connections and 
                            current_time - info.last_used > self.idle_timeout):
                            idle_connections.append(client_id)
                
                # Remove idle connections (but keep minimum)
                connections_to_remove = len(idle_connections) - max(0, len(self._connections) - self.min_connections)
                
                for client_id in idle_connections[:connections_to_remove]:
                    await self._remove_connection(client_id)
                    logger.debug(f"Removed idle connection {client_id}")
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")


# Global connection pool instance
_connection_pool: Optional[IBConnectionPool] = None


async def get_connection_pool(**config) -> IBConnectionPool:
    """Get or create global connection pool.
    
    Args:
        **config: Connection pool configuration
        
    Returns:
        Global connection pool instance
    """
    global _connection_pool
    
    if _connection_pool is None:
        _connection_pool = IBConnectionPool(**config)
        await _connection_pool.start()
    
    return _connection_pool


async def shutdown_connection_pool() -> None:
    """Shutdown global connection pool."""
    global _connection_pool
    
    if _connection_pool is not None:
        await _connection_pool.stop()
        _connection_pool = None


@asynccontextmanager
async def pooled_connection(**config):
    """Context manager for getting pooled connection.
    
    Args:
        **config: Connection configuration
        
    Yields:
        IB API client connection
    """
    pool = await get_connection_pool(**config)
    async with pool.get_connection() as connection:
        yield connection