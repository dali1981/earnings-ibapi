"""
Examples demonstrating the comprehensive reliability system for financial trading applications.

This module shows how to use the reliability system components:
- Custom exceptions and error handling
- Retry mechanisms with exponential backoff
- Circuit breaker pattern
- Specialized decorators for IB API calls
- Structured logging with context
"""
import time
import logging
import asyncio
from typing import List, Dict, Any
from datetime import date, datetime

# Import reliability system components
from reliability import (
    # Exceptions
    BaseTradingException, IBConnectionException, MarketDataTimeoutException,
    RateLimitException, DataValidationException,
    
    # Retry and circuit breaker
    retry, ib_api_retry, market_data_retry, circuit_breaker,
    RetryStrategy, JitterType,
    
    # Decorators
    ib_api_call, market_data_request, historical_data_request,
    connection_operation, resilient_operation,
    
    # Logging
    get_trading_logger, log_context, performance_log,
    configure_trading_logging, StructuredFormatter,
    
    # Configuration
    setup_reliability_system
)


# Example 1: Basic Exception Handling and Logging
def example_basic_error_handling():
    """Demonstrate basic error handling with custom exceptions."""
    
    logger = get_trading_logger(__name__)
    
    def problematic_operation():
        """Simulated operation that might fail."""
        import random
        if random.random() < 0.7:  # 70% failure rate
            raise IBConnectionException(
                "Connection to IB TWS failed",
                error_code="502",
                context={'host': '127.0.0.1', 'port': 7497}
            )
        return "Operation successful"
    
    with log_context(operation="basic_operation", symbol="AAPL"):
        try:
            result = problematic_operation()
            logger.info(f"Operation completed: {result}")
            
        except BaseTradingException as e:
            logger.log_exception(e, msg="Trading operation failed")
            print(f"Trading error occurred: {e.message}")
            print(f"Error details: {e.to_dict()}")
            
        except Exception as e:
            logger.log_exception(e, msg="Unexpected error")
            print(f"Unexpected error: {e}")


# Example 2: Retry Mechanisms
def example_retry_mechanisms():
    """Demonstrate different retry strategies."""
    
    logger = get_trading_logger(__name__)
    
    # Simple retry with exponential backoff
    @retry(
        max_attempts=5,
        base_delay=1.0,
        max_delay=30.0,
        strategy=RetryStrategy.EXPONENTIAL,
        jitter=JitterType.EQUAL
    )
    def unreliable_market_data_fetch():
        """Simulated market data fetch that fails randomly."""
        import random
        if random.random() < 0.6:  # 60% failure rate
            raise MarketDataTimeoutException("AAPL", 30.0)
        return {"symbol": "AAPL", "price": 150.25, "timestamp": datetime.now()}
    
    # IB API specific retry
    @ib_api_retry(max_attempts=3, base_delay=2.0)
    def ib_connection_attempt():
        """Simulated IB connection attempt."""
        import random
        if random.random() < 0.4:  # 40% failure rate
            raise IBConnectionException("Failed to connect to TWS")
        return "Connected to IB TWS"
    
    # Market data specific retry
    @market_data_retry(max_attempts=3)
    def get_option_chain():
        """Simulated option chain request."""
        import random
        if random.random() < 0.3:  # 30% failure rate
            raise RateLimitException("Rate limit exceeded", retry_after=1.0)
        return [{"strike": 150, "call_price": 5.25, "put_price": 3.75}]
    
    with log_context(operation="retry_examples"):
        logger.info("Starting retry mechanism examples")
        
        # Test exponential backoff retry
        try:
            data = unreliable_market_data_fetch()
            logger.info(f"Market data retrieved: {data}")
        except Exception as e:
            logger.log_exception(e, msg="Market data fetch failed after all retries")
        
        # Test IB API retry
        try:
            result = ib_connection_attempt()
            logger.info(f"Connection result: {result}")
        except Exception as e:
            logger.log_exception(e, msg="IB connection failed after all retries")
        
        # Test market data retry
        try:
            chain = get_option_chain()
            logger.info(f"Option chain retrieved: {chain}")
        except Exception as e:
            logger.log_exception(e, msg="Option chain request failed after all retries")


# Example 3: Circuit Breaker Pattern
def example_circuit_breaker():
    """Demonstrate circuit breaker pattern."""
    
    logger = get_trading_logger(__name__)
    
    @circuit_breaker(
        name="market_data_service",
        failure_threshold=3,
        timeout_duration=10.0,
        success_threshold=2
    )
    def market_data_service_call():
        """Simulated market data service that might fail."""
        import random
        if random.random() < 0.8:  # 80% failure rate initially
            raise MarketDataTimeoutException("SPY", 30.0)
        return {"symbol": "SPY", "price": 420.50}
    
    with log_context(operation="circuit_breaker_example"):
        logger.info("Starting circuit breaker example")
        
        # Make multiple calls to trigger circuit breaker
        for i in range(10):
            try:
                result = market_data_service_call()
                logger.info(f"Call {i+1} succeeded: {result}")
                time.sleep(0.5)
                
            except Exception as e:
                logger.log_exception(e, msg=f"Call {i+1} failed")
                time.sleep(0.5)
                
                # Simulate service recovery after some failures
                if i >= 5:
                    # Improve success rate to test recovery
                    import random
                    random.seed(12345)  # Fixed seed for predictable recovery


# Example 4: IB API Decorators
def example_ib_api_decorators():
    """Demonstrate specialized IB API decorators."""
    
    logger = get_trading_logger(__name__)
    
    @market_data_request(symbol="AAPL", timeout=30.0, max_retries=3)
    def get_market_data_for_symbol():
        """Get real-time market data."""
        import random
        if random.random() < 0.3:
            raise MarketDataTimeoutException("AAPL", 30.0)
        return {
            "symbol": "AAPL",
            "bid": 150.20,
            "ask": 150.25,
            "last": 150.22,
            "volume": 1000000
        }
    
    @historical_data_request(
        symbol="MSFT",
        duration="5 D",
        bar_size="1 min", 
        timeout=120.0
    )
    def get_historical_bars():
        """Get historical bar data."""
        import random
        if random.random() < 0.2:
            raise IBConnectionException("Historical data request failed")
        
        # Simulate returning bar data
        bars = []
        for i in range(5):
            bars.append({
                "datetime": f"2024-01-{i+1:02d} 09:30:00",
                "open": 420.0 + i,
                "high": 422.0 + i,
                "low": 419.0 + i,
                "close": 421.0 + i,
                "volume": 100000 + i * 1000
            })
        return bars
    
    @connection_operation("connect_to_tws", critical=True)
    def connect_to_tws():
        """Connect to Interactive Brokers TWS."""
        import random
        if random.random() < 0.2:
            raise IBConnectionException("Failed to connect to TWS on port 7497")
        return "Successfully connected to TWS"
    
    @resilient_operation("save_trading_data", max_retries=5)
    def save_data_to_repository(data: Dict[str, Any]):
        """Save data with high resilience."""
        import random
        if random.random() < 0.1:
            raise DataValidationException("Invalid data format", field="timestamp")
        logger.info(f"Data saved successfully: {len(data)} records")
        return True
    
    with log_context(operation="ib_api_examples", symbol="AAPL"):
        logger.info("Starting IB API decorator examples")
        
        # Test market data request
        try:
            market_data = get_market_data_for_symbol()
            logger.info(f"Market data: {market_data}")
        except Exception as e:
            logger.log_exception(e, msg="Market data request failed")
        
        # Test historical data request
        try:
            historical_data = get_historical_bars()
            logger.info(f"Retrieved {len(historical_data)} historical bars")
        except Exception as e:
            logger.log_exception(e, msg="Historical data request failed")
        
        # Test connection operation
        try:
            connection_result = connect_to_tws()
            logger.info(f"Connection: {connection_result}")
        except Exception as e:
            logger.log_exception(e, msg="Connection failed")
        
        # Test resilient data save
        try:
            sample_data = {"records": [1, 2, 3, 4, 5]}
            save_result = save_data_to_repository(sample_data)
            logger.info(f"Data save result: {save_result}")
        except Exception as e:
            logger.log_exception(e, msg="Data save failed")


# Example 5: Advanced Logging and Context
def example_advanced_logging():
    """Demonstrate advanced logging features."""
    
    logger = get_trading_logger(__name__, extra_context={"component": "examples"})
    
    @performance_log(operation="data_processing")
    def process_market_data(symbol: str, data: List[Dict]):
        """Process market data with automatic performance logging."""
        time.sleep(0.1)  # Simulate processing time
        
        with log_context(symbol=symbol, record_count=len(data)):
            logger.info(f"Processing {len(data)} records for {symbol}")
            
            # Simulate some processing work
            processed_records = []
            for record in data:
                # Add some processing logic here
                processed_record = {**record, "processed": True}
                processed_records.append(processed_record)
            
            logger.info(f"Processed {len(processed_records)} records")
            return processed_records
    
    def simulate_trading_session():
        """Simulate a complete trading session with nested contexts."""
        
        session_id = f"session_{int(time.time())}"
        
        with log_context(operation="trading_session", session_id=session_id):
            logger.info("Starting trading session")
            
            symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
            
            for symbol in symbols:
                with log_context(symbol=symbol):
                    logger.info(f"Processing symbol {symbol}")
                    
                    # Simulate market data
                    sample_data = [
                        {"timestamp": "09:30:00", "price": 150.0, "volume": 1000},
                        {"timestamp": "09:31:00", "price": 150.1, "volume": 1500},
                        {"timestamp": "09:32:00", "price": 149.9, "volume": 800},
                    ]
                    
                    try:
                        processed = process_market_data(symbol, sample_data)
                        logger.info(f"Successfully processed {symbol}: {len(processed)} records")
                        
                    except Exception as e:
                        logger.log_exception(e, msg=f"Failed to process {symbol}")
            
            logger.info("Trading session completed")
    
    logger.info("Starting advanced logging example")
    simulate_trading_session()


# Example 6: Async Operations
async def example_async_operations():
    """Demonstrate async operations with reliability features."""
    
    logger = get_trading_logger(__name__)
    
    @ib_api_call(
        operation="async_market_data",
        max_retries=3,
        timeout=30.0,
        enable_circuit_breaker=True
    )
    async def async_get_market_data(symbol: str):
        """Async market data fetch."""
        await asyncio.sleep(0.1)  # Simulate network delay
        
        import random
        if random.random() < 0.3:
            raise MarketDataTimeoutException(symbol, 30.0)
        
        return {
            "symbol": symbol,
            "price": 150.0 + random.uniform(-5, 5),
            "timestamp": datetime.now().isoformat()
        }
    
    @retry(max_attempts=3, base_delay=0.5)
    async def async_save_data(data: Dict[str, Any]):
        """Async data save with retry."""
        await asyncio.sleep(0.05)  # Simulate I/O
        
        import random
        if random.random() < 0.2:
            raise Exception("Database connection failed")
        
        logger.info(f"Data saved: {data['symbol']}")
        return True
    
    with log_context(operation="async_example"):
        logger.info("Starting async operations example")
        
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        # Fetch market data concurrently
        tasks = []
        for symbol in symbols:
            with log_context(symbol=symbol):
                task = async_get_market_data(symbol)
                tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.log_exception(result, msg=f"Failed to fetch data for {symbol}")
                else:
                    logger.info(f"Fetched data for {symbol}: {result}")
                    
                    # Save data asynchronously
                    try:
                        await async_save_data(result)
                    except Exception as e:
                        logger.log_exception(e, msg=f"Failed to save data for {symbol}")
        
        except Exception as e:
            logger.log_exception(e, msg="Async operations failed")


# Example 7: Complete Trading Application Example
def example_complete_trading_app():
    """Demonstrate a complete trading application using the reliability system."""
    
    # Setup the reliability system
    setup_reliability_system(
        log_level="INFO",
        log_file="trading_app.log",
        enable_performance_monitoring=True
    )
    
    logger = get_trading_logger(__name__)
    
    class TradingApplication:
        """Example trading application with full reliability integration."""
        
        def __init__(self):
            self.logger = get_trading_logger(self.__class__.__name__)
            self.connected = False
        
        @connection_operation("initialize_connection", critical=True)
        def connect(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
            """Connect to IB TWS with reliability features."""
            import random
            if random.random() < 0.2:
                raise IBConnectionException(f"Failed to connect to {host}:{port}")
            
            self.connected = True
            return f"Connected to {host}:{port} with client ID {client_id}"
        
        @market_data_request(timeout=30.0, max_retries=3)
        def get_stock_quote(self, symbol: str):
            """Get stock quote with market data reliability."""
            if not self.connected:
                raise IBConnectionException("Not connected to TWS")
            
            import random
            if random.random() < 0.1:
                raise MarketDataTimeoutException(symbol, 30.0)
            
            return {
                "symbol": symbol,
                "bid": 150.0 + random.uniform(-10, 10),
                "ask": 150.0 + random.uniform(-10, 10) + 0.01,
                "last": 150.0 + random.uniform(-10, 10),
                "volume": random.randint(100000, 1000000)
            }
        
        @historical_data_request("PORTFOLIO", "1 D", "1 min", timeout=60.0)
        def get_portfolio_history(self):
            """Get portfolio historical data."""
            if not self.connected:
                raise IBConnectionException("Not connected to TWS")
            
            # Simulate historical data
            return [
                {"datetime": "2024-01-01 09:30", "value": 100000.0},
                {"datetime": "2024-01-01 09:31", "value": 100050.0},
                {"datetime": "2024-01-01 09:32", "value": 99980.0},
            ]
        
        @resilient_operation("process_trading_signals", max_retries=5)
        def process_signals(self, signals: List[Dict]):
            """Process trading signals with high resilience."""
            import random
            if random.random() < 0.05:  # 5% failure rate
                raise DataValidationException("Invalid signal format")
            
            processed_signals = []
            for signal in signals:
                processed_signal = {
                    **signal,
                    "processed_at": datetime.now().isoformat(),
                    "status": "processed"
                }
                processed_signals.append(processed_signal)
            
            return processed_signals
        
        @performance_log("trading_workflow")
        def run_trading_workflow(self, symbols: List[str]):
            """Run complete trading workflow with monitoring."""
            
            workflow_id = f"workflow_{int(time.time())}"
            
            with log_context(operation="trading_workflow", workflow_id=workflow_id):
                self.logger.info(f"Starting trading workflow for {len(symbols)} symbols")
                
                # Connect to IB
                try:
                    connection_result = self.connect()
                    self.logger.info(f"Connection established: {connection_result}")
                except Exception as e:
                    self.logger.log_exception(e, msg="Failed to connect")
                    return False
                
                # Process each symbol
                results = {}
                for symbol in symbols:
                    with log_context(symbol=symbol):
                        try:
                            # Get quote
                            quote = self.get_stock_quote(symbol)
                            self.logger.info(f"Quote for {symbol}: ${quote['last']:.2f}")
                            
                            # Simulate signal generation
                            signal = {
                                "symbol": symbol,
                                "action": "BUY" if quote["last"] < 150 else "SELL",
                                "quantity": 100,
                                "price": quote["last"]
                            }
                            
                            # Process signal
                            processed = self.process_signals([signal])
                            results[symbol] = processed[0]
                            
                        except Exception as e:
                            self.logger.log_exception(e, msg=f"Failed to process {symbol}")
                            results[symbol] = {"error": str(e)}
                
                # Get portfolio history
                try:
                    history = self.get_portfolio_history()
                    self.logger.info(f"Portfolio history: {len(history)} records")
                except Exception as e:
                    self.logger.log_exception(e, msg="Failed to get portfolio history")
                
                self.logger.info(f"Trading workflow completed: {len(results)} symbols processed")
                return results
    
    # Run the complete example
    app = TradingApplication()
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    
    try:
        results = app.run_trading_workflow(symbols)
        logger.info(f"Application completed successfully: {len(results)} results")
        
        # Print results summary
        success_count = sum(1 for r in results.values() if "error" not in r)
        logger.info(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        
    except Exception as e:
        logger.log_exception(e, msg="Trading application failed")


def main():
    """Run all examples."""
    
    # Configure basic logging for examples
    configure_trading_logging(
        level="INFO",
        format_type="json",
        console_logging=True
    )
    
    logger = get_trading_logger(__name__)
    logger.info("Starting reliability system examples")
    
    print("\n=== Example 1: Basic Error Handling ===")
    example_basic_error_handling()
    
    print("\n=== Example 2: Retry Mechanisms ===") 
    example_retry_mechanisms()
    
    print("\n=== Example 3: Circuit Breaker ===")
    example_circuit_breaker()
    
    print("\n=== Example 4: IB API Decorators ===")
    example_ib_api_decorators()
    
    print("\n=== Example 5: Advanced Logging ===")
    example_advanced_logging()
    
    print("\n=== Example 6: Async Operations ===")
    asyncio.run(example_async_operations())
    
    print("\n=== Example 7: Complete Trading App ===")
    example_complete_trading_app()
    
    logger.info("All examples completed")


if __name__ == "__main__":
    main()