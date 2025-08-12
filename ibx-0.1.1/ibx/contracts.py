
from ibapi.contract import Contract

def make_stock(symbol: str, exch: str = "SMART", curr: str = "USD") -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = exch
    c.currency = curr
    return c

def make_option(symbol: str, last_trade_date: str, strike: float, right: str,
                exch: str = "SMART", curr: str = "USD", multiplier: str = "100") -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "OPT"
    c.exchange = exch
    c.currency = curr
    c.lastTradeDateOrContractMonth = last_trade_date  # YYYYMMDD
    c.strike = float(strike)
    c.right = right  # "C" or "P"
    c.multiplier = multiplier
    return c
