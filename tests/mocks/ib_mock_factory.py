"""
Factory for creating IB API mocks with realistic data.
"""
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Union
from unittest.mock import Mock, MagicMock
import random
import pandas as pd
import threading
import time

from ibapi.contract import Contract
from ibapi.wrapper import EWrapper


class MockDataGenerator:
    """Generate realistic mock data for testing."""
    
    @staticmethod
    def generate_stock_price_series(
        symbol: str, 
        start_price: float = 150.0,
        num_bars: int = 100,
        volatility: float = 0.02,
        trend: float = 0.0001
    ) -> List[Dict[str, Any]]:
        """Generate realistic stock price time series."""
        bars = []
        current_price = start_price
        
        for i in range(num_bars):
            # Generate random price movement
            daily_return = random.normalvariate(trend, volatility)
            new_price = current_price * (1 + daily_return)
            
            # Generate OHLC with realistic relationships
            high = new_price * (1 + abs(random.normalvariate(0, volatility/4)))
            low = new_price * (1 - abs(random.normalvariate(0, volatility/4)))
            open_price = current_price
            close_price = new_price
            
            # Ensure OHLC consistency
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Generate volume
            base_volume = 1000000
            volume = int(base_volume * (1 + random.normalvariate(0, 0.5)))
            volume = max(volume, 10000)  # Minimum volume
            
            bar_date = (date.today() - timedelta(days=num_bars-i-1)).strftime("%Y%m%d")
            
            bars.append({
                "date": bar_date,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close_price, 2),
                "volume": volume,
                "wap": round((high + low + close_price) / 3, 2),
                "barCount": random.randint(1000, 5000)
            })
            
            current_price = new_price
            
        return bars
    
    @staticmethod
    def generate_option_price_series(
        underlying_price: float,
        strike: float,
        expiry_days: int,
        option_type: str = "C",
        num_bars: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate realistic option price time series."""
        bars = []
        
        # Simple Black-Scholes approximation for option pricing
        def black_scholes_approx(S, K, T, r=0.05, sigma=0.25):
            if T <= 0:
                return max(0, S - K) if option_type == "C" else max(0, K - S)
            
            # Simplified approximation
            moneyness = S / K
            time_value = sigma * (T ** 0.5) * S * 0.4
            
            if option_type == "C":
                intrinsic = max(0, S - K)
                return intrinsic + time_value * moneyness
            else:
                intrinsic = max(0, K - S)
                return intrinsic + time_value * (2 - moneyness)
        
        current_underlying = underlying_price
        
        for i in range(num_bars):
            # Simulate underlying price movement
            daily_return = random.normalvariate(0, 0.02)
            current_underlying *= (1 + daily_return)
            
            # Calculate option price
            time_to_expiry = (expiry_days - i) / 365.0
            option_price = black_scholes_approx(current_underlying, strike, time_to_expiry)
            
            # Add some noise to option price
            noise = random.normalvariate(0, option_price * 0.05)
            option_price = max(0.01, option_price + noise)
            
            # Generate OHLC for option
            high = option_price * (1 + abs(random.normalvariate(0, 0.03)))
            low = option_price * (1 - abs(random.normalvariate(0, 0.03)))
            open_price = option_price * (1 + random.normalvariate(0, 0.01))
            close_price = option_price
            
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price, 0.01)
            
            volume = random.randint(10, 1000)
            
            bar_time = datetime.now() - timedelta(minutes=num_bars-i)
            
            bars.append({
                "time": bar_time,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close_price, 2),
                "volume": volume
            })
            
        return bars
    
    @staticmethod
    def generate_option_chain(
        underlying_symbol: str,
        underlying_price: float,
        expiry_date: date,
        strikes: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Generate realistic option chain data."""
        if strikes is None:
            # Generate strikes around current price
            strikes = []
            for i in range(-5, 6):
                strike = underlying_price + (i * 5)
                if strike > 0:
                    strikes.append(strike)
        
        chain_data = []
        days_to_expiry = (expiry_date - date.today()).days
        
        for strike in strikes:
            for right in ["C", "P"]:
                # Calculate theoretical option price
                moneyness = underlying_price / strike
                time_value = max(0.01, 0.25 * (days_to_expiry / 365.0) ** 0.5)
                
                if right == "C":
                    intrinsic = max(0, underlying_price - strike)
                    theoretical_price = intrinsic + time_value * underlying_price * 0.1
                else:
                    intrinsic = max(0, strike - underlying_price)
                    theoretical_price = intrinsic + time_value * strike * 0.1
                
                # Add bid-ask spread
                mid_price = theoretical_price
                spread = max(0.01, mid_price * 0.02)
                bid = max(0.01, mid_price - spread/2)
                ask = mid_price + spread/2
                
                # Generate Greeks (simplified)
                if right == "C":
                    delta = max(0.01, min(0.99, moneyness * 0.6))
                else:
                    delta = max(-0.99, min(-0.01, (moneyness - 1) * 0.6))
                
                gamma = 0.02 * (1 - abs(delta))
                theta = -theoretical_price * 0.01
                vega = underlying_price * 0.1 * time_value
                
                chain_data.append({
                    "underlying": underlying_symbol,
                    "expiry": expiry_date,
                    "strike": strike,
                    "right": right,
                    "bid": round(bid, 2),
                    "ask": round(ask, 2),
                    "last": round(mid_price, 2),
                    "volume": random.randint(0, 500),
                    "openInterest": random.randint(100, 5000),
                    "impliedVolatility": round(0.15 + random.random() * 0.3, 3),
                    "delta": round(delta, 3),
                    "gamma": round(gamma, 3),
                    "theta": round(theta, 3),
                    "vega": round(vega, 3)
                })
        
        return chain_data


class MockContractDetailsFactory:
    """Factory for creating mock contract details."""
    
    @staticmethod
    def create_stock_contract_details(symbol: str, conId: int = None) -> Mock:
        """Create mock stock contract details."""
        if conId is None:
            conId = abs(hash(symbol)) % 1000000
        
        details = Mock()
        contract = Mock()
        
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.primaryExchange = "NASDAQ" if symbol in ["AAPL", "MSFT", "GOOGL"] else "NYSE"
        contract.currency = "USD"
        contract.conId = conId
        contract.localSymbol = symbol
        
        details.contract = contract
        details.marketName = f"{symbol} Stock"
        details.minTick = 0.01
        details.orderTypes = ["LMT", "MKT", "STP"]
        details.validExchanges = "SMART,NASDAQ,NYSE"
        
        return details
    
    @staticmethod
    def create_option_contract_details(
        underlying: str, 
        expiry: str, 
        strike: float, 
        right: str
    ) -> Mock:
        """Create mock option contract details."""
        details = Mock()
        contract = Mock()
        
        contract.symbol = underlying
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = expiry
        contract.strike = strike
        contract.right = right
        contract.multiplier = "100"
        contract.conId = abs(hash(f"{underlying}_{expiry}_{strike}_{right}")) % 1000000
        
        details.contract = contract
        details.marketName = f"{underlying} {expiry} {strike} {right}"
        details.minTick = 0.01
        details.orderTypes = ["LMT", "MKT"]
        
        return details


class MockIBClientFactory:
    """Factory for creating comprehensive IB client mocks."""
    
    def __init__(self):
        self.data_generator = MockDataGenerator()
        self.contract_factory = MockContractDetailsFactory()
        
    def create_full_mock_client(
        self, 
        symbols: List[str] = None,
        enable_delays: bool = True,
        error_rate: float = 0.0
    ) -> Mock:
        """Create a comprehensive mock IB client."""
        if symbols is None:
            symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        
        client = Mock()
        client.isConnected.return_value = True
        client.nextOrderId = 1000
        
        # Store request data
        client._requests = {}
        client._wrapper = None
        
        def mock_connect(host, port, clientId):
            client.isConnected.return_value = True
            if client._wrapper:
                # Simulate connection success
                if enable_delays:
                    threading.Timer(0.1, lambda: client._wrapper.nextValidId(1000)).start()
                else:
                    client._wrapper.nextValidId(1000)
        
        def mock_req_contract_details(reqId, contract):
            client._requests[reqId] = {"type": "contractDetails", "contract": contract}
            
            def send_response():
                if random.random() < error_rate:
                    if client._wrapper:
                        client._wrapper.error(reqId, 200, "Contract not found", "")
                    return
                
                symbol = getattr(contract, 'symbol', 'UNKNOWN')
                if symbol in symbols:
                    details = self.contract_factory.create_stock_contract_details(symbol)
                    if client._wrapper:
                        client._wrapper.contractDetails(reqId, details)
                        client._wrapper.contractDetailsEnd(reqId)
                else:
                    if client._wrapper:
                        client._wrapper.error(reqId, 200, "Contract not found", "")
            
            if enable_delays:
                threading.Timer(0.1 + random.random() * 0.2, send_response).start()
            else:
                send_response()
        
        def mock_req_historical_data(reqId, contract, endDateTime, durationStr, 
                                   barSizeSetting, whatToShow, useRTH, formatDate, 
                                   keepUpToDate, chartOptions):
            client._requests[reqId] = {
                "type": "historicalData", 
                "contract": contract,
                "params": {
                    "endDateTime": endDateTime,
                    "durationStr": durationStr,
                    "barSizeSetting": barSizeSetting
                }
            }
            
            def send_response():
                if random.random() < error_rate:
                    if client._wrapper:
                        client._wrapper.error(reqId, 162, "Historical data error", "")
                    return
                
                symbol = getattr(contract, 'symbol', 'UNKNOWN')
                if symbol in symbols:
                    # Generate appropriate data based on bar size
                    if "day" in barSizeSetting.lower():
                        bars = self.data_generator.generate_stock_price_series(
                            symbol, num_bars=int(durationStr.split()[0]) if durationStr.split()[0].isdigit() else 5
                        )
                    else:
                        bars = self.data_generator.generate_stock_price_series(
                            symbol, num_bars=100
                        )
                    
                    if client._wrapper:
                        for bar_data in bars:
                            bar = Mock()
                            for key, value in bar_data.items():
                                setattr(bar, key, value)
                            client._wrapper.historicalData(reqId, bar)
                        
                        client._wrapper.historicalDataEnd(reqId, bars[0]["date"], bars[-1]["date"])
                else:
                    if client._wrapper:
                        client._wrapper.error(reqId, 162, "No data found", "")
            
            delay = 0.2 + random.random() * 0.5 if enable_delays else 0
            threading.Timer(delay, send_response).start()
        
        def mock_req_mkt_data(reqId, contract, genericTicks, snapshot, regulatorySnapshot, mktDataOptions):
            client._requests[reqId] = {"type": "marketData", "contract": contract}
            
            def send_response():
                if random.random() < error_rate:
                    if client._wrapper:
                        client._wrapper.error(reqId, 354, "Requested market data is not subscribed", "")
                    return
                
                symbol = getattr(contract, 'symbol', 'UNKNOWN')
                if client._wrapper and symbol in symbols:
                    # Send market data ticks
                    base_price = 150.0  # Could be made symbol-specific
                    bid = base_price - 0.01
                    ask = base_price + 0.01
                    
                    client._wrapper.tickPrice(reqId, 1, bid, Mock())  # Bid
                    client._wrapper.tickPrice(reqId, 2, ask, Mock())  # Ask
                    client._wrapper.tickSize(reqId, 0, 100)  # Bid size
                    client._wrapper.tickSize(reqId, 3, 200)  # Ask size
            
            delay = 0.1 if enable_delays else 0
            threading.Timer(delay, send_response).start()
        
        # Set up mock methods
        client.connect = mock_connect
        client.disconnect = Mock()
        client.reqContractDetails = mock_req_contract_details
        client.reqHistoricalData = mock_req_historical_data
        client.reqMktData = mock_req_mkt_data
        client.reqSecDefOptParams = Mock()
        
        # Add wrapper property
        def set_wrapper(wrapper):
            client._wrapper = wrapper
        
        client.wrapper = property(lambda self: self._wrapper, set_wrapper)
        
        return client
    
    def create_error_prone_client(self, error_rate: float = 0.3) -> Mock:
        """Create a mock client that generates errors for testing error handling."""
        return self.create_full_mock_client(error_rate=error_rate)
    
    def create_slow_client(self, base_delay: float = 1.0) -> Mock:
        """Create a mock client with configurable delays for performance testing."""
        client = self.create_full_mock_client(enable_delays=True)
        
        # Override delays to be longer
        original_req_historical = client.reqHistoricalData
        
        def slow_req_historical(*args, **kwargs):
            result = original_req_historical(*args, **kwargs)
            # Add extra delay
            time.sleep(base_delay)
            return result
        
        client.reqHistoricalData = slow_req_historical
        return client


# Convenience functions for common use cases
def create_mock_client_with_data(symbols: List[str] = None) -> Mock:
    """Create a mock client pre-loaded with test data."""
    factory = MockIBClientFactory()
    return factory.create_full_mock_client(symbols)


def create_realistic_market_data(symbol: str, num_days: int = 30) -> pd.DataFrame:
    """Create realistic market data for testing."""
    generator = MockDataGenerator()
    bars = generator.generate_stock_price_series(symbol, num_bars=num_days)
    
    return pd.DataFrame(bars)


def create_realistic_option_data(
    underlying: str, 
    strike: float, 
    expiry_days: int = 30
) -> pd.DataFrame:
    """Create realistic option data for testing."""
    generator = MockDataGenerator()
    bars = generator.generate_option_price_series(150.0, strike, expiry_days)
    
    return pd.DataFrame(bars)