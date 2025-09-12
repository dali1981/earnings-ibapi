"""Tests for Redis caching layer."""

import json
import pickle
import time
from unittest.mock import Mock, patch, MagicMock
import pytest
import pandas as pd

from common.cache.redis_cache import (
    RedisCache,
    CacheConfig,
    SerializationFormat,
    get_cache,
    shutdown_cache,
    make_query_cache_key
)


class TestCacheConfig:
    """Test cache configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()
        
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.socket_timeout == 5.0
        assert config.connection_timeout == 10.0
        assert config.default_ttl == 3600
        assert config.key_prefix == "earnings_ibapi"
        assert config.enable_compression is True
        assert config.compression_level == 6
        assert config.max_connections == 10
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CacheConfig(
            host="redis.example.com",
            port=6380,
            db=1,
            password="secret",
            default_ttl=7200,
            key_prefix="test_app",
            enable_compression=False
        )
        
        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.db == 1
        assert config.password == "secret"
        assert config.default_ttl == 7200
        assert config.key_prefix == "test_app"
        assert config.enable_compression is False


class TestSerializationFormat:
    """Test serialization format enum."""
    
    def test_format_values(self):
        """Test serialization format enum values."""
        assert SerializationFormat.JSON == "json"
        assert SerializationFormat.PICKLE == "pickle"
        assert SerializationFormat.MSGPACK == "msgpack"
        assert SerializationFormat.PARQUET == "parquet"


class TestMakeQueryCacheKey:
    """Test cache key generation."""
    
    def test_simple_query_key(self):
        """Test simple query cache key generation."""
        key = make_query_cache_key(
            dataset="equity_bars",
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        
        assert key.startswith("query:")
        assert "equity_bars" in key
        assert "AAPL" in key
        assert "2024-01-01" in key
    
    def test_complex_query_key(self):
        """Test complex query cache key with multiple parameters."""
        key = make_query_cache_key(
            dataset="option_chains",
            symbol="AAPL",
            expiration="2024-03-15",
            strike_range=(100, 200),
            option_type="CALL"
        )
        
        assert key.startswith("query:")
        assert all(param in key for param in ["option_chains", "AAPL", "2024-03-15", "CALL"])
    
    def test_key_deterministic(self):
        """Test that keys are deterministic for same parameters."""
        key1 = make_query_cache_key(symbol="AAPL", date="2024-01-01", bar_size="1 day")
        key2 = make_query_cache_key(bar_size="1 day", symbol="AAPL", date="2024-01-01")
        
        assert key1 == key2  # Order shouldn't matter


class TestRedisCache:
    """Test Redis cache functionality."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.set.return_value = True
        mock_client.delete.return_value = 1
        mock_client.exists.return_value = 0
        mock_client.ttl.return_value = -1
        mock_client.flushdb.return_value = True
        return mock_client
    
    @pytest.fixture
    def cache_config(self):
        """Create test cache configuration."""
        return CacheConfig(
            host="localhost",
            port=6379,
            default_ttl=300,
            key_prefix="test"
        )
    
    @pytest.fixture
    def cache(self, mock_redis, cache_config):
        """Create Redis cache with mock client."""
        with patch('common.cache.redis_cache.redis.ConnectionPool'), \
             patch('common.cache.redis_cache.redis.Redis', return_value=mock_redis):
            cache = RedisCache(cache_config)
            return cache
    
    def test_cache_initialization(self, cache_config):
        """Test cache initialization."""
        with patch('common.cache.redis_cache.redis.ConnectionPool') as mock_pool, \
             patch('common.cache.redis_cache.redis.Redis') as mock_redis_class:
            
            cache = RedisCache(cache_config)
            
            mock_pool.assert_called_once()
            mock_redis_class.assert_called_once()
            assert cache.config == cache_config
    
    def test_health_check_success(self, cache, mock_redis):
        """Test successful health check."""
        mock_redis.ping.return_value = True
        
        result = cache.health_check()
        
        assert result is True
        mock_redis.ping.assert_called_once()
    
    def test_health_check_failure(self, cache, mock_redis):
        """Test failed health check."""
        mock_redis.ping.side_effect = Exception("Connection failed")
        
        result = cache.health_check()
        
        assert result is False
    
    def test_set_and_get_json(self, cache, mock_redis):
        """Test setting and getting JSON data."""
        test_data = {"symbol": "AAPL", "price": 150.0}
        test_key = "test_key"
        
        # Mock successful set
        mock_redis.set.return_value = True
        
        # Test set
        result = cache.set(test_key, test_data, ttl=300, format=SerializationFormat.JSON)
        assert result is True
        
        # Mock get returning serialized data
        serialized_data = json.dumps(test_data).encode()
        mock_redis.get.return_value = serialized_data
        
        # Test get
        retrieved_data = cache.get(test_key, format=SerializationFormat.JSON)
        assert retrieved_data == test_data
    
    def test_set_and_get_pickle(self, cache, mock_redis):
        """Test setting and getting pickled data."""
        test_data = {"complex": {"nested": [1, 2, 3]}}
        test_key = "pickle_key"
        
        mock_redis.set.return_value = True
        
        # Test set
        result = cache.set(test_key, test_data, format=SerializationFormat.PICKLE)
        assert result is True
        
        # Mock get returning pickled data
        pickled_data = pickle.dumps(test_data)
        mock_redis.get.return_value = pickled_data
        
        # Test get
        retrieved_data = cache.get(test_key, format=SerializationFormat.PICKLE)
        assert retrieved_data == test_data
    
    def test_set_and_get_dataframe(self, cache, mock_redis):
        """Test setting and getting DataFrame with Parquet format."""
        df = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL'],
            'price': [150.0, 2500.0]
        })
        test_key = "df_key"
        
        mock_redis.set.return_value = True
        
        # Test set
        result = cache.set(test_key, df, format=SerializationFormat.PARQUET)
        assert result is True
        
        # Mock get returning parquet bytes
        parquet_bytes = df.to_parquet()
        mock_redis.get.return_value = parquet_bytes
        
        # Test get
        retrieved_df = cache.get(test_key, format=SerializationFormat.PARQUET)
        pd.testing.assert_frame_equal(retrieved_df, df)
    
    def test_get_nonexistent_key(self, cache, mock_redis):
        """Test getting non-existent key."""
        mock_redis.get.return_value = None
        
        result = cache.get("nonexistent_key")
        assert result is None
    
    def test_delete_key(self, cache, mock_redis):
        """Test deleting cache key."""
        mock_redis.delete.return_value = 1
        
        result = cache.delete("test_key")
        assert result is True
        mock_redis.delete.assert_called_with("test:test_key")
    
    def test_exists_key(self, cache, mock_redis):
        """Test checking key existence."""
        mock_redis.exists.return_value = 1
        
        result = cache.exists("test_key")
        assert result is True
        mock_redis.exists.assert_called_with("test:test_key")
    
    def test_expire_key(self, cache, mock_redis):
        """Test setting key expiration."""
        mock_redis.expire.return_value = True
        
        result = cache.expire("test_key", 600)
        assert result is True
        mock_redis.expire.assert_called_with("test:test_key", 600)
    
    def test_get_ttl(self, cache, mock_redis):
        """Test getting key TTL."""
        mock_redis.ttl.return_value = 300
        
        result = cache.get_ttl("test_key")
        assert result == 300
        mock_redis.ttl.assert_called_with("test:test_key")
    
    def test_clear_all(self, cache, mock_redis):
        """Test clearing all cache keys."""
        mock_redis.flushdb.return_value = True
        
        result = cache.clear_all()
        assert result is True
        mock_redis.flushdb.assert_called_once()
    
    def test_get_stats(self, cache, mock_redis):
        """Test getting cache statistics."""
        # Mock Redis info response
        mock_redis.info.return_value = {
            'used_memory': 1000000,
            'connected_clients': 5,
            'total_connections_received': 100,
            'keyspace_hits': 80,
            'keyspace_misses': 20
        }
        mock_redis.dbsize.return_value = 50
        
        stats = cache.get_stats()
        
        assert stats['memory_usage'] == 1000000
        assert stats['connected_clients'] == 5
        assert stats['total_keys'] == 50
        assert stats['hit_rate'] == 0.8  # 80/(80+20)
    
    def test_cached_call_cache_hit(self, cache, mock_redis):
        """Test cached function call with cache hit."""
        # Mock cache hit
        cached_result = json.dumps("cached_result").encode()
        mock_redis.get.return_value = cached_result
        
        def expensive_function(x, y):
            return f"result_{x}_{y}"
        
        result = cache.cached_call(
            "func_key", 
            expensive_function, 
            ttl=300, 
            format=SerializationFormat.JSON,
            x="a", 
            y="b"
        )
        
        assert result == "cached_result"
        # Function should not be called
        mock_redis.get.assert_called_once()
    
    def test_cached_call_cache_miss(self, cache, mock_redis):
        """Test cached function call with cache miss."""
        # Mock cache miss, then successful set
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        
        def expensive_function(x, y):
            return f"result_{x}_{y}"
        
        result = cache.cached_call(
            "func_key",
            expensive_function,
            ttl=300,
            format=SerializationFormat.JSON,
            x="a",
            y="b"
        )
        
        assert result == "result_a_b"
        # Should try to get from cache, then set result
        mock_redis.get.assert_called_once()
        mock_redis.set.assert_called_once()
    
    def test_key_prefixing(self, cache, mock_redis):
        """Test that keys are properly prefixed."""
        cache.get("test_key")
        mock_redis.get.assert_called_with("test:test_key")
        
        cache.set("another_key", "value")
        # Extract the key argument from the call
        call_args = mock_redis.set.call_args[0]
        assert call_args[0] == "test:another_key"
    
    def test_serialization_error_handling(self, cache, mock_redis):
        """Test handling of serialization errors."""
        # Create an object that can't be JSON serialized
        non_serializable = set([1, 2, 3])
        
        result = cache.set("bad_key", non_serializable, format=SerializationFormat.JSON)
        assert result is False
    
    def test_compression_enabled(self, cache_config, mock_redis):
        """Test compression functionality."""
        config = CacheConfig(enable_compression=True, compression_level=9)
        
        with patch('common.cache.redis_cache.redis.ConnectionPool'), \
             patch('common.cache.redis_cache.redis.Redis', return_value=mock_redis), \
             patch('common.cache.redis_cache.gzip.compress') as mock_compress, \
             patch('common.cache.redis_cache.gzip.decompress') as mock_decompress:
            
            cache = RedisCache(config)
            
            # Test set with compression
            test_data = "large data that should be compressed"
            mock_compress.return_value = b"compressed_data"
            mock_redis.set.return_value = True
            
            result = cache.set("key", test_data, format=SerializationFormat.JSON)
            assert result is True
            mock_compress.assert_called_once()
            
            # Test get with decompression
            mock_redis.get.return_value = b"compressed_data"
            mock_decompress.return_value = json.dumps(test_data).encode()
            
            retrieved_data = cache.get("key", format=SerializationFormat.JSON)
            assert retrieved_data == test_data
            mock_decompress.assert_called_once()


class TestCacheUtilityFunctions:
    """Test cache utility functions."""
    
    @patch('common.cache.redis_cache._cache', None)
    def test_get_cache_creates_instance(self):
        """Test that get_cache creates cache instance."""
        with patch('common.cache.redis_cache.RedisCache') as mock_cache_class:
            mock_instance = Mock()
            mock_cache_class.return_value = mock_instance
            
            cache = get_cache(host="localhost", port=6379)
            
            assert cache == mock_instance
            mock_cache_class.assert_called_once()
    
    @patch('common.cache.redis_cache._cache')
    def test_get_cache_returns_existing(self, mock_existing_cache):
        """Test that get_cache returns existing instance."""
        cache = get_cache()
        assert cache == mock_existing_cache
    
    @patch('common.cache.redis_cache._cache')
    def test_shutdown_cache(self, mock_cache):
        """Test cache shutdown."""
        shutdown_cache()
        
        # Check that global cache is set to None
        from common.cache.redis_cache import _cache
        assert _cache is None


class TestCacheIntegration:
    """Integration tests for cache functionality."""
    
    @pytest.fixture
    def cache_with_real_serialization(self):
        """Cache with real serialization (no Redis)."""
        config = CacheConfig(key_prefix="test")
        
        # Mock Redis but allow real serialization
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        
        with patch('common.cache.redis_cache.redis.ConnectionPool'), \
             patch('common.cache.redis_cache.redis.Redis', return_value=mock_redis):
            cache = RedisCache(config)
            cache._redis = mock_redis  # Ensure we use the mock
            return cache, mock_redis
    
    def test_dataframe_roundtrip(self, cache_with_real_serialization):
        """Test DataFrame serialization roundtrip."""
        cache, mock_redis = cache_with_real_serialization
        
        # Create test DataFrame
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'symbol': ['AAPL'] * 5,
            'price': [150.0, 151.0, 149.0, 152.0, 148.0]
        })
        
        # Capture what gets stored
        stored_data = None
        def capture_set(key, value, **kwargs):
            nonlocal stored_data
            stored_data = value
            return True
        
        mock_redis.set.side_effect = capture_set
        
        # Store DataFrame
        result = cache.set("df_test", df, format=SerializationFormat.PARQUET)
        assert result is True
        assert stored_data is not None
        
        # Mock get to return stored data
        mock_redis.get.return_value = stored_data
        
        # Retrieve DataFrame
        retrieved_df = cache.get("df_test", format=SerializationFormat.PARQUET)
        
        # Verify data integrity
        pd.testing.assert_frame_equal(retrieved_df, df)
    
    def test_json_complex_data_roundtrip(self, cache_with_real_serialization):
        """Test JSON serialization with complex data."""
        cache, mock_redis = cache_with_real_serialization
        
        # Complex nested data structure
        complex_data = {
            'metadata': {
                'version': '1.0',
                'timestamp': '2024-01-01T00:00:00'
            },
            'data': [
                {'symbol': 'AAPL', 'price': 150.0, 'volume': 1000},
                {'symbol': 'GOOGL', 'price': 2500.0, 'volume': 500}
            ],
            'stats': {
                'count': 2,
                'avg_price': 1325.0
            }
        }
        
        # Capture and return stored data
        stored_data = None
        def capture_set(key, value, **kwargs):
            nonlocal stored_data
            stored_data = value
            return True
        
        mock_redis.set.side_effect = capture_set
        mock_redis.get.return_value = lambda: stored_data
        
        # Store complex data
        cache.set("complex_test", complex_data, format=SerializationFormat.JSON)
        mock_redis.get.return_value = stored_data
        
        # Retrieve complex data
        retrieved_data = cache.get("complex_test", format=SerializationFormat.JSON)
        
        # Verify data integrity
        assert retrieved_data == complex_data
    
    def test_performance_statistics_tracking(self, cache_with_real_serialization):
        """Test that performance statistics are tracked."""
        cache, mock_redis = cache_with_real_serialization
        
        # Initialize stats
        initial_stats = cache.get_performance_stats()
        assert initial_stats['cache_hits'] == 0
        assert initial_stats['cache_misses'] == 0
        
        # Simulate cache miss
        mock_redis.get.return_value = None
        cache.get("miss_key")
        
        # Simulate cache hit
        mock_redis.get.return_value = b'"hit_value"'
        cache.get("hit_key", format=SerializationFormat.JSON)
        
        # Check updated stats
        updated_stats = cache.get_performance_stats()
        assert updated_stats['cache_hits'] == 1
        assert updated_stats['cache_misses'] == 1