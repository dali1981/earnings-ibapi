"""
Market data generator for comprehensive test data creation.
Provides realistic market data with proper financial relationships and edge cases.
"""

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import random
import math


class MarketDataGenerator:
    """
    Advanced market data generator that creates realistic financial time series data
    with proper price relationships, volatility patterns, and market microstructure.
    """

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducible test data."""
        np.random.seed(seed)
        random.seed(seed)

    def generate_equity_bars(
        self,
        symbol: str,
        periods: int,
        bar_size: str = "1 day",
        start_date: Optional[date] = None,
        base_price: float = 150.0,
        volatility: float = 0.02,
        trend: float = 0.0005,
        include_gaps: bool = False
    ) -> pd.DataFrame:
        """
        Generate realistic equity bar data with proper OHLCV relationships.
        
        Args:
            symbol: Stock symbol
            periods: Number of bars to generate
            bar_size: Bar size (1 day, 1 hour, 1 min, etc.)
            start_date: Starting date (defaults to today - periods)
            base_price: Starting price
            volatility: Daily volatility (as decimal)
            trend: Daily trend (as decimal)
            include_gaps: Whether to include price gaps
        """
        if start_date is None:
            start_date = date.today() - timedelta(days=max(periods, 30))

        # Generate time series based on bar size
        if bar_size == "1 day":
            dates = pd.date_range(start_date, periods=periods, freq="D")
            date_col = "date"
            date_values = dates.strftime("%Y%m%d")
        elif bar_size in ["1 hour", "1 min"]:
            freq_map = {"1 hour": "1H", "1 min": "1min"}
            freq = freq_map[bar_size]
            
            # For intraday, generate during market hours (9:30 AM - 4:00 PM ET)
            market_start = f"{start_date} 09:30"
            if bar_size == "1 hour":
                total_periods = periods
            else:  # 1 min
                total_periods = periods
                
            dates = pd.date_range(market_start, periods=total_periods, freq=freq)
            # Filter to market hours only
            dates = dates[(dates.hour >= 9) & (dates.hour < 16) | 
                         ((dates.hour == 9) & (dates.minute >= 30))][:periods]
            
            date_col = "time"
            date_values = dates
        else:
            raise ValueError(f"Unsupported bar size: {bar_size}")

        # Generate realistic price series using geometric Brownian motion
        prices = self._generate_price_series(
            periods=len(dates),
            base_price=base_price,
            volatility=volatility,
            trend=trend,
            include_gaps=include_gaps
        )

        # Generate OHLCV data with realistic relationships
        bars_data = []
        
        for i, (timestamp, price) in enumerate(zip(date_values, prices)):
            # Generate intraday range based on volatility
            daily_vol = volatility if bar_size == "1 day" else volatility / math.sqrt(390)  # Scale for intraday
            range_factor = np.random.gamma(2, daily_vol * price * 0.5)
            
            # Create high and low
            high = price + range_factor * np.random.uniform(0.3, 1.0)
            low = price - range_factor * np.random.uniform(0.3, 1.0)
            
            # Create open (previous close with some gap)
            if i == 0:
                open_price = price
            else:
                gap_factor = 1 + np.random.normal(0, volatility * 0.3)
                open_price = max(prices[i-1] * gap_factor, low)
                open_price = min(open_price, high)
            
            # Close is the current price, bounded by high/low
            close_price = max(min(price, high), low)
            
            # Ensure OHLC relationships
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Generate volume with realistic patterns
            volume = self._generate_volume(
                bar_size=bar_size,
                price_change=abs(close_price - open_price) / open_price,
                base_volume=1000000 if bar_size == "1 day" else 50000
            )
            
            # Calculate Volume Weighted Average Price
            wap = (high + low + 2 * close_price) / 4
            
            bar_data = {
                date_col: timestamp,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close_price, 2),
                "volume": int(volume),
                "wap": round(wap, 2)
            }
            
            # Add additional fields for intraday data
            if bar_size != "1 day":
                bar_data["barCount"] = random.randint(100, 1000)
            
            bars_data.append(bar_data)

        return pd.DataFrame(bars_data)

    def generate_option_chain(
        self,
        underlying: str,
        expiries: List[date],
        spot_price: float = 150.0,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.02,
        base_vol: float = 0.25
    ) -> pd.DataFrame:
        """
        Generate realistic option chain data with proper Greeks and pricing.
        """
        chain_data = []
        
        for expiry in expiries:
            days_to_expiry = max((expiry - date.today()).days, 1)
            time_to_expiry = days_to_expiry / 365.0
            
            # Generate strikes around the money
            atm_strike = round(spot_price / 5) * 5  # Round to nearest $5
            strikes = [
                atm_strike + offset 
                for offset in [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
                if atm_strike + offset > 0
            ]
            
            for strike in strikes:
                for right in ["C", "P"]:
                    # Calculate implied volatility with smile
                    moneyness = spot_price / strike
                    vol_smile = self._calculate_vol_smile(moneyness, time_to_expiry, base_vol)
                    
                    # Calculate option price and Greeks using Black-Scholes
                    option_data = self._black_scholes_with_greeks(
                        spot_price, strike, time_to_expiry, risk_free_rate,
                        dividend_yield, vol_smile, right
                    )
                    
                    # Add market microstructure (bid-ask spread, volume, OI)
                    market_data = self._generate_option_market_data(
                        option_data["price"], moneyness, time_to_expiry
                    )
                    
                    chain_data.append({
                        "underlying": underlying,
                        "expiry": expiry,
                        "strike": strike,
                        "right": right,
                        "bid": market_data["bid"],
                        "ask": market_data["ask"],
                        "last": market_data["last"],
                        "volume": market_data["volume"],
                        "openInterest": market_data["open_interest"],
                        "impliedVolatility": round(vol_smile, 4),
                        "delta": round(option_data["delta"], 4),
                        "gamma": round(option_data["gamma"], 6),
                        "theta": round(option_data["theta"], 4),
                        "vega": round(option_data["vega"], 4),
                        "rho": round(option_data["rho"], 4)
                    })
        
        return pd.DataFrame(chain_data)

    def generate_earnings_calendar(
        self,
        symbols: List[str],
        days_ahead: int = 90,
        include_estimates: bool = True
    ) -> pd.DataFrame:
        """Generate realistic earnings calendar data."""
        calendar_data = []
        
        # Company data for realistic names and sectors
        company_info = {
            "AAPL": {"name": "Apple Inc.", "sector": "Technology", "market_cap": "3T"},
            "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "market_cap": "2.8T"},
            "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "market_cap": "1.7T"},
            "TSLA": {"name": "Tesla, Inc.", "sector": "Consumer Cyclical", "market_cap": "800B"},
            "NVDA": {"name": "NVIDIA Corporation", "sector": "Technology", "market_cap": "1.1T"},
            "META": {"name": "Meta Platforms, Inc.", "sector": "Technology", "market_cap": "750B"},
            "AMZN": {"name": "Amazon.com, Inc.", "sector": "Consumer Cyclical", "market_cap": "1.5T"},
            "NFLX": {"name": "Netflix, Inc.", "sector": "Communication Services", "market_cap": "200B"}
        }
        
        for i, symbol in enumerate(symbols):
            info = company_info.get(symbol, {
                "name": f"{symbol} Corporation",
                "sector": "Diversified",
                "market_cap": "50B"
            })
            
            # Distribute earnings over the period
            days_offset = (i * days_ahead // len(symbols)) + random.randint(-5, 5)
            earnings_date = date.today() + timedelta(days=days_offset)
            
            # Generate realistic estimates based on company size
            market_cap_multiplier = self._get_market_cap_multiplier(info["market_cap"])
            
            calendar_entry = {
                "symbol": symbol,
                "company_name": info["name"],
                "sector": info["sector"],
                "earnings_date": earnings_date,
                "earnings_time": random.choice(["BMO", "AMC", "DMT"]),  # Before/After/During market
                "quarter": f"Q{((earnings_date.month - 1) // 3) + 1}",
                "fiscal_year": earnings_date.year,
                "confirmed": random.choice([True, True, True, False]),  # 75% confirmed
                "last_updated": datetime.now()
            }
            
            if include_estimates:
                # Generate realistic estimates
                revenue_base = random.uniform(5e9, 100e9) * market_cap_multiplier
                eps_base = random.uniform(0.50, 8.00) * market_cap_multiplier
                
                calendar_entry.update({
                    "revenue_estimate": round(revenue_base, 2),
                    "revenue_estimate_low": round(revenue_base * 0.95, 2),
                    "revenue_estimate_high": round(revenue_base * 1.05, 2),
                    "eps_estimate": round(eps_base, 2),
                    "eps_estimate_low": round(eps_base * 0.9, 2),
                    "eps_estimate_high": round(eps_base * 1.1, 2),
                    "analyst_count": random.randint(5, 25)
                })
            
            calendar_data.append(calendar_entry)
        
        return pd.DataFrame(calendar_data)

    def generate_test_edge_cases(self, data_type: str = "equity") -> pd.DataFrame:
        """Generate edge case data for robust testing."""
        if data_type == "equity":
            return pd.DataFrame({
                "date": ["20240101", "20240102", "20240103", "invalid", "20240105"],
                "open": [0.01, -10.0, None, "invalid", 1e10],  # Min price, negative, null, invalid, huge
                "high": [0.02, 10.0, 150.0, 155.0, 1e10],
                "low": [0.005, -15.0, 148.0, 152.0, -1.0],     # Below open, negative
                "close": [0.015, 5.0, None, 153.0, float('inf')],  # Null, infinite
                "volume": [0, -1000, 1e12, "invalid", None],      # Zero, negative, huge, invalid, null
                "wap": [0.01, 2.5, 149.5, 153.5, float('nan')]   # NaN case
            })
        elif data_type == "options":
            return pd.DataFrame({
                "underlying": ["AAPL", "", "INVALID", "AAPL", None],
                "strike": [0, -100, 1e6, "invalid", None],
                "expiry": [date.today() - timedelta(days=1), date.today(), date.today() + timedelta(days=1), "invalid", None],
                "right": ["C", "P", "X", "", None],  # Invalid right
                "bid": [-1, 0, 1e6, None, "invalid"],
                "ask": [0.5, -1, 1e6, float('inf'), None],
                "volume": [-1, 0, 1e9, None, "invalid"]
            })

    def _generate_price_series(
        self,
        periods: int,
        base_price: float,
        volatility: float,
        trend: float,
        include_gaps: bool
    ) -> np.ndarray:
        """Generate realistic price series using geometric Brownian motion."""
        # Generate returns with autocorrelation (momentum/mean reversion)
        returns = np.random.normal(trend, volatility, periods)
        
        # Add some autocorrelation to make it more realistic
        for i in range(1, periods):
            returns[i] += 0.1 * returns[i-1]  # Mild momentum
        
        # Generate price levels
        prices = [base_price]
        for i in range(periods - 1):
            # Apply return with some bounds checking
            new_price = prices[-1] * (1 + returns[i])
            new_price = max(new_price, 0.01)  # Prevent negative prices
            
            # Add occasional gaps if requested
            if include_gaps and random.random() < 0.05:  # 5% chance of gap
                gap_factor = 1 + np.random.normal(0, volatility * 2)
                new_price *= gap_factor
            
            prices.append(new_price)
        
        return np.array(prices)

    def _generate_volume(self, bar_size: str, price_change: float, base_volume: int) -> int:
        """Generate realistic volume based on price movement and bar size."""
        # Volume tends to increase with larger price movements
        volume_multiplier = 1 + (price_change * 10)  # 10x factor for price change impact
        
        # Add random component with log-normal distribution (realistic for volume)
        random_factor = np.random.lognormal(0, 0.5)
        
        # Scale base volume
        volume = base_volume * volume_multiplier * random_factor
        
        # Add minimum volume constraint
        min_volume = 1000 if bar_size == "1 day" else 100
        return max(int(volume), min_volume)

    def _calculate_vol_smile(
        self, 
        moneyness: float, 
        time_to_expiry: float, 
        base_vol: float
    ) -> float:
        """Calculate implied volatility with smile effect."""
        # Volatility smile: higher vol for OTM options
        smile_factor = 0.05 * (moneyness - 1) ** 2  # Parabolic smile
        
        # Term structure: shorter expiries have higher vol
        term_factor = 0.1 * math.exp(-time_to_expiry * 2)
        
        vol = base_vol + smile_factor + term_factor
        return max(vol, 0.05)  # Minimum 5% vol

    def _black_scholes_with_greeks(
        self,
        S: float,  # Spot price
        K: float,  # Strike price
        T: float,  # Time to expiry
        r: float,  # Risk-free rate
        q: float,  # Dividend yield
        sigma: float,  # Volatility
        option_type: str  # 'C' for call, 'P' for put
    ) -> Dict[str, float]:
        """Calculate Black-Scholes price and Greeks."""
        from scipy.stats import norm
        
        # Prevent division by zero
        if T <= 0:
            if option_type == "C":
                price = max(S - K, 0)
                delta = 1.0 if S > K else 0.0
            else:
                price = max(K - S, 0)
                delta = -1.0 if K > S else 0.0
            
            return {
                "price": price,
                "delta": delta,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "rho": 0.0
            }
        
        # Calculate d1 and d2
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        # Calculate price
        if option_type == "C":
            price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            delta = math.exp(-q * T) * norm.cdf(d1)
            rho = K * T * math.exp(-r * T) * norm.cdf(d2)
        else:  # Put
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
            delta = -math.exp(-q * T) * norm.cdf(-d1)
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)
        
        # Calculate Greeks
        gamma = math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))
        theta = (
            -S * math.exp(-q * T) * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * (norm.cdf(d2) if option_type == "C" else norm.cdf(-d2))
            + q * S * math.exp(-q * T) * (norm.cdf(d1) if option_type == "C" else norm.cdf(-d1))
        ) / 365  # Convert to daily theta
        
        vega = S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T) / 100  # Convert to 1% vol change
        
        return {
            "price": max(price, 0.01),  # Minimum price
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho
        }

    def _generate_option_market_data(
        self, 
        theoretical_price: float, 
        moneyness: float, 
        time_to_expiry: float
    ) -> Dict[str, Any]:
        """Generate realistic market data for options."""
        # Bid-ask spread increases for OTM and short-dated options
        spread_factor = 0.02 + 0.05 * abs(moneyness - 1) + 0.03 * math.exp(-time_to_expiry * 4)
        spread = max(theoretical_price * spread_factor, 0.05)  # Minimum 5 cent spread
        
        # Mid price with some noise
        mid = theoretical_price * (1 + np.random.normal(0, 0.02))
        bid = max(mid - spread / 2, 0.01)
        ask = mid + spread / 2
        
        # Last price within bid-ask
        last = random.uniform(bid, ask)
        
        # Volume inversely related to spread and time to expiry
        base_volume = 1000 / (spread_factor + 0.1)
        volume = int(np.random.exponential(base_volume))
        
        # Open interest higher for ATM options
        base_oi = 5000 * (1 - abs(moneyness - 1) * 2)  # More OI for ATM
        open_interest = max(int(np.random.exponential(max(base_oi, 100))), 0)
        
        return {
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "last": round(last, 2),
            "volume": volume,
            "open_interest": open_interest
        }

    def _get_market_cap_multiplier(self, market_cap_str: str) -> float:
        """Get multiplier based on market cap for realistic estimates."""
        if "T" in market_cap_str:  # Trillion
            return 3.0
        elif "B" in market_cap_str:  # Billion
            value = float(market_cap_str.replace("B", ""))
            if value > 500:
                return 2.0
            elif value > 100:
                return 1.5
            else:
                return 1.0
        else:
            return 0.5  # Small cap