"""Tests for trading domain schemas."""

import pytest
from datetime import date, datetime
from decimal import Decimal
import pandas as pd

from common.schemas.trading import (
    ContractBase,
    EquityContract,
    OptionContract,
    EquityBar,
    OptionBar,
    OptionChain,
    OptionQuote
)
from common.constants import SecType, OptionRight, Exchange


class TestContractBase:
    """Test base contract schema."""
    
    def test_contract_base_creation(self):
        """Test ContractBase creation with valid data."""
        contract = ContractBase(
            symbol="AAPL",
            sec_type=SecType.STOCK,
            exchange=Exchange.SMART,
            currency="USD"
        )
        
        assert contract.symbol == "AAPL"
        assert contract.sec_type == SecType.STOCK
        assert contract.exchange == Exchange.SMART
        assert contract.currency == "USD"
    
    def test_contract_validation(self):
        """Test contract validation."""
        # Test invalid currency
        with pytest.raises(ValueError):
            ContractBase(
                symbol="AAPL",
                sec_type=SecType.STOCK,
                exchange=Exchange.SMART,
                currency="INVALID"
            )
        
        # Test invalid symbol (too short)
        with pytest.raises(ValueError):
            ContractBase(
                symbol="A",
                sec_type=SecType.STOCK,
                exchange=Exchange.SMART,
                currency="USD"
            )


class TestEquityContract:
    """Test equity contract schema."""
    
    def test_equity_contract_creation(self):
        """Test EquityContract creation."""
        contract = EquityContract(
            symbol="GOOGL",
            exchange=Exchange.NASDAQ,
            currency="USD"
        )
        
        assert contract.symbol == "GOOGL"
        assert contract.sec_type == SecType.STOCK  # Should default to STOCK
        assert contract.exchange == Exchange.NASDAQ
    
    def test_to_ib_contract_dict(self):
        """Test conversion to IB contract dictionary."""
        contract = EquityContract(
            symbol="MSFT",
            exchange=Exchange.SMART,
            currency="USD"
        )
        
        ib_dict = contract.to_ib_contract_dict()
        
        assert ib_dict["symbol"] == "MSFT"
        assert ib_dict["secType"] == "STK"
        assert ib_dict["exchange"] == "SMART"
        assert ib_dict["currency"] == "USD"
    
    def test_equity_contract_immutable_sec_type(self):
        """Test that sec_type is immutable for equity contracts."""
        contract = EquityContract(symbol="AAPL")
        
        # sec_type should be frozen
        with pytest.raises(ValueError):
            contract.sec_type = SecType.OPTION


class TestOptionContract:
    """Test option contract schema."""
    
    def test_option_contract_creation(self):
        """Test OptionContract creation."""
        contract = OptionContract(
            symbol="AAPL",
            expiry=date(2024, 3, 15),
            strike=150.0,
            right=OptionRight.CALL,
            exchange=Exchange.SMART,
            currency="USD"
        )
        
        assert contract.symbol == "AAPL"
        assert contract.sec_type == SecType.OPTION
        assert contract.expiry == date(2024, 3, 15)
        assert contract.strike == 150.0
        assert contract.right == OptionRight.CALL
    
    def test_option_contract_validation(self):
        """Test option contract validation."""
        # Test negative strike
        with pytest.raises(ValueError):
            OptionContract(
                symbol="AAPL",
                expiry=date(2024, 3, 15),
                strike=-10.0,
                right=OptionRight.CALL
            )
        
        # Test past expiry date
        with pytest.raises(ValueError):
            OptionContract(
                symbol="AAPL", 
                expiry=date(2020, 1, 1),  # Past date
                strike=150.0,
                right=OptionRight.CALL
            )
    
    def test_to_ib_contract_dict(self):
        """Test option contract to IB dictionary conversion."""
        contract = OptionContract(
            symbol="AAPL",
            expiry=date(2024, 3, 15),
            strike=150.0,
            right=OptionRight.PUT,
            exchange=Exchange.SMART,
            currency="USD"
        )
        
        ib_dict = contract.to_ib_contract_dict()
        
        assert ib_dict["symbol"] == "AAPL"
        assert ib_dict["secType"] == "OPT"
        assert ib_dict["lastTradeDateOrContractMonth"] == "20240315"
        assert ib_dict["strike"] == 150.0
        assert ib_dict["right"] == "P"


class TestEquityBar:
    """Test equity bar schema."""
    
    def test_equity_bar_creation(self):
        """Test EquityBar creation."""
        bar = EquityBar(
            symbol="AAPL",
            date=date(2024, 1, 15),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
            bar_size="1 day"
        )
        
        assert bar.symbol == "AAPL"
        assert bar.date == date(2024, 1, 15)
        assert bar.open == 150.0
        assert bar.high == 155.0
        assert bar.low == 149.0
        assert bar.close == 154.0
        assert bar.volume == 1000000
    
    def test_equity_bar_validation(self):
        """Test equity bar validation."""
        # Test invalid OHLC relationship (high < low)
        with pytest.raises(ValueError):
            EquityBar(
                symbol="AAPL",
                date=date(2024, 1, 15),
                open=150.0,
                high=148.0,  # High less than low
                low=149.0,
                close=150.0,
                volume=1000000,
                bar_size="1 day"
            )
        
        # Test negative volume
        with pytest.raises(ValueError):
            EquityBar(
                symbol="AAPL",
                date=date(2024, 1, 15),
                open=150.0,
                high=155.0,
                low=149.0,
                close=154.0,
                volume=-1000,  # Negative volume
                bar_size="1 day"
            )
    
    def test_to_dict(self):
        """Test equity bar to dictionary conversion."""
        bar = EquityBar(
            symbol="AAPL",
            date=date(2024, 1, 15),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
            bar_size="1 day"
        )
        
        bar_dict = bar.to_dict()
        
        assert bar_dict["symbol"] == "AAPL"
        assert bar_dict["date"] == date(2024, 1, 15)
        assert bar_dict["open"] == 150.0
        assert bar_dict["volume"] == 1000000


class TestOptionBar:
    """Test option bar schema."""
    
    def test_option_bar_creation(self):
        """Test OptionBar creation."""
        bar = OptionBar(
            symbol="AAPL",
            expiry=date(2024, 3, 15),
            strike=150.0,
            right=OptionRight.CALL,
            date=date(2024, 1, 15),
            open=5.0,
            high=5.5,
            low=4.8,
            close=5.2,
            volume=500,
            bar_size="1 day"
        )
        
        assert bar.symbol == "AAPL"
        assert bar.expiry == date(2024, 3, 15)
        assert bar.strike == 150.0
        assert bar.right == OptionRight.CALL
        assert bar.open == 5.0
        assert bar.volume == 500
    
    def test_option_key_property(self):
        """Test option key property."""
        bar = OptionBar(
            symbol="AAPL",
            expiry=date(2024, 3, 15),
            strike=150.0,
            right=OptionRight.CALL,
            date=date(2024, 1, 15),
            open=5.0,
            high=5.5,
            low=4.8,
            close=5.2,
            volume=500,
            bar_size="1 day"
        )
        
        expected_key = "AAPL_20240315_150.0_CALL"
        assert bar.option_key == expected_key


class TestOptionChain:
    """Test option chain schema."""
    
    def test_option_chain_creation(self):
        """Test OptionChain creation."""
        quotes = [
            OptionQuote(
                strike=150.0,
                right=OptionRight.CALL,
                bid=5.0,
                ask=5.2,
                last=5.1,
                volume=100,
                open_interest=1000,
                implied_volatility=0.25
            ),
            OptionQuote(
                strike=150.0,
                right=OptionRight.PUT,
                bid=3.0,
                ask=3.2,
                last=3.1,
                volume=50,
                open_interest=800,
                implied_volatility=0.28
            )
        ]
        
        chain = OptionChain(
            symbol="AAPL",
            expiry=date(2024, 3, 15),
            underlying_price=151.0,
            quotes=quotes,
            timestamp=datetime.now()
        )
        
        assert chain.symbol == "AAPL"
        assert chain.expiry == date(2024, 3, 15)
        assert chain.underlying_price == 151.0
        assert len(chain.quotes) == 2
    
    def test_get_calls_and_puts(self):
        """Test filtering calls and puts from chain."""
        quotes = [
            OptionQuote(
                strike=150.0,
                right=OptionRight.CALL,
                bid=5.0,
                ask=5.2,
                last=5.1
            ),
            OptionQuote(
                strike=155.0,
                right=OptionRight.CALL,
                bid=3.0,
                ask=3.2,
                last=3.1
            ),
            OptionQuote(
                strike=150.0,
                right=OptionRight.PUT,
                bid=2.0,
                ask=2.2,
                last=2.1
            )
        ]
        
        chain = OptionChain(
            symbol="AAPL",
            expiry=date(2024, 3, 15),
            underlying_price=152.0,
            quotes=quotes
        )
        
        calls = chain.get_calls()
        puts = chain.get_puts()
        
        assert len(calls) == 2
        assert len(puts) == 1
        assert all(quote.right == OptionRight.CALL for quote in calls)
        assert all(quote.right == OptionRight.PUT for quote in puts)
    
    def test_get_strike_range(self):
        """Test getting strike range from chain."""
        quotes = [
            OptionQuote(strike=140.0, right=OptionRight.CALL, bid=8.0, ask=8.2),
            OptionQuote(strike=150.0, right=OptionRight.CALL, bid=5.0, ask=5.2),
            OptionQuote(strike=160.0, right=OptionRight.CALL, bid=2.0, ask=2.2)
        ]
        
        chain = OptionChain(
            symbol="AAPL",
            expiry=date(2024, 3, 15),
            underlying_price=150.0,
            quotes=quotes
        )
        
        min_strike, max_strike = chain.get_strike_range()
        
        assert min_strike == 140.0
        assert max_strike == 160.0


class TestOptionQuote:
    """Test option quote schema."""
    
    def test_option_quote_creation(self):
        """Test OptionQuote creation."""
        quote = OptionQuote(
            strike=150.0,
            right=OptionRight.CALL,
            bid=5.0,
            ask=5.2,
            last=5.1,
            volume=100,
            open_interest=1000,
            implied_volatility=0.25,
            delta=0.6,
            gamma=0.05,
            theta=-0.02,
            vega=0.15
        )
        
        assert quote.strike == 150.0
        assert quote.right == OptionRight.CALL
        assert quote.bid == 5.0
        assert quote.ask == 5.2
        assert quote.last == 5.1
        assert quote.volume == 100
        assert quote.delta == 0.6
    
    def test_option_quote_validation(self):
        """Test option quote validation."""
        # Test bid > ask
        with pytest.raises(ValueError):
            OptionQuote(
                strike=150.0,
                right=OptionRight.CALL,
                bid=5.2,  # Bid higher than ask
                ask=5.0,
                last=5.1
            )
        
        # Test negative implied volatility
        with pytest.raises(ValueError):
            OptionQuote(
                strike=150.0,
                right=OptionRight.CALL,
                bid=5.0,
                ask=5.2,
                implied_volatility=-0.1  # Negative IV
            )
    
    def test_spread_calculation(self):
        """Test bid-ask spread calculation."""
        quote = OptionQuote(
            strike=150.0,
            right=OptionRight.CALL,
            bid=5.0,
            ask=5.4,
            last=5.2
        )
        
        assert quote.spread == 0.4  # 5.4 - 5.0
    
    def test_mid_price_calculation(self):
        """Test mid price calculation."""
        quote = OptionQuote(
            strike=150.0,
            right=OptionRight.CALL,
            bid=5.0,
            ask=5.4,
            last=5.2
        )
        
        assert quote.mid_price == 5.2  # (5.0 + 5.4) / 2
    
    def test_moneyness_calculation(self):
        """Test moneyness calculation."""
        quote = OptionQuote(
            strike=150.0,
            right=OptionRight.CALL,
            bid=5.0,
            ask=5.2
        )
        
        # ITM call
        assert quote.moneyness(160.0) == "ITM"
        
        # ATM call
        assert quote.moneyness(150.0) == "ATM" 
        
        # OTM call
        assert quote.moneyness(140.0) == "OTM"
        
        # Test put moneyness
        put_quote = OptionQuote(
            strike=150.0,
            right=OptionRight.PUT,
            bid=3.0,
            ask=3.2
        )
        
        # ITM put
        assert put_quote.moneyness(140.0) == "ITM"
        
        # OTM put  
        assert put_quote.moneyness(160.0) == "OTM"


class TestSchemaIntegration:
    """Test integration between different schemas."""
    
    def test_contract_to_bar_consistency(self):
        """Test consistency between contracts and bars."""
        # Create equity contract
        contract = EquityContract(symbol="AAPL", exchange=Exchange.SMART)
        
        # Create corresponding bar
        bar = EquityBar(
            symbol=contract.symbol,
            date=date(2024, 1, 15),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
            bar_size="1 day"
        )
        
        # Symbols should match
        assert contract.symbol == bar.symbol
    
    def test_option_contract_to_chain_consistency(self):
        """Test consistency between option contracts and chains."""
        # Create option contract
        contract = OptionContract(
            symbol="AAPL",
            expiry=date(2024, 3, 15),
            strike=150.0,
            right=OptionRight.CALL
        )
        
        # Create quote for same option
        quote = OptionQuote(
            strike=contract.strike,
            right=contract.right,
            bid=5.0,
            ask=5.2,
            last=5.1
        )
        
        # Create chain
        chain = OptionChain(
            symbol=contract.symbol,
            expiry=contract.expiry,
            underlying_price=151.0,
            quotes=[quote]
        )
        
        # Verify consistency
        assert contract.symbol == chain.symbol
        assert contract.expiry == chain.expiry
        assert contract.strike == quote.strike
        assert contract.right == quote.right
    
    def test_dataframe_conversion(self):
        """Test conversion of schemas to DataFrame format."""
        bars = [
            EquityBar(
                symbol="AAPL",
                date=date(2024, 1, 15),
                open=150.0,
                high=155.0,
                low=149.0,
                close=154.0,
                volume=1000000,
                bar_size="1 day"
            ),
            EquityBar(
                symbol="AAPL", 
                date=date(2024, 1, 16),
                open=154.0,
                high=157.0,
                low=153.0,
                close=156.0,
                volume=1200000,
                bar_size="1 day"
            )
        ]
        
        # Convert to DataFrame
        df_data = [bar.to_dict() for bar in bars]
        df = pd.DataFrame(df_data)
        
        assert len(df) == 2
        assert list(df['symbol'].unique()) == ["AAPL"]
        assert df['close'].tolist() == [154.0, 156.0]