# ib_vol_live.py
# pip install ibapi
import time
import logging
from enum import IntEnum
from dataclasses import dataclass
from typing import Dict, List

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

# -------- Enums (minimal subset you actually use) --------
class TickSizeType(IntEnum):
    BID_SIZE = 0
    ASK_SIZE = 3
    LAST_SIZE = 5
    VOLUME   = 8
    CALL_OPEN_INT = 27
    PUT_OPEN_INT  = 28

class TickPriceType(IntEnum):
    BID = 1
    ASK = 2
    LAST = 4
    CLOSE = 9

class TickStringType(IntEnum):
    LAST_TIMESTAMP = 45

class TickGenericType(IntEnum):
    HISTORICAL_VOLATILITY_30D = 23     # via generic "104"
    OPTION_IMPLIED_VOL_30D    = 24     # via generic "106"
    REALTIME_HISTORICAL_VOL   = 58     # via generic "411"

class TickOptionCompType(IntEnum):
    BID   = 10
    ASK   = 11
    LAST  = 12
    MODEL = 13
    DELAYED_BID   = 80
    DELAYED_ASK   = 81
    DELAYED_LAST  = 82
    DELAYED_MODEL = 83

# -------- Contracts helpers --------
def make_stock(symbol: str, exchange="SMART", currency="USD", primaryExchange: str | None = None) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = exchange
    c.currency = currency
    if primaryExchange:
        c.primaryExchange = primaryExchange  # e.g. "NASDAQ"
    return c

@dataclass
class OptionSpec:
    symbol: str
    yyyymmdd: str
    strike: float
    right: str  # "C" or "P"
    exchange: str = "SMART"
    currency: str = "USD"

def make_option(spec: OptionSpec) -> Contract:
    c = Contract()
    c.symbol = spec.symbol
    c.secType = "OPT"
    c.exchange = spec.exchange
    c.currency = spec.currency
    c.lastTradeDateOrContractMonth = spec.yyyymmdd
    c.strike = float(spec.strike)
    c.right = spec.right
    c.multiplier = "100"
    return c

# -------- App --------
class LiveVolTap(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.log = logging.getLogger("LiveVolTap")
        self._labels: Dict[int, str] = {}   # reqId -> human label

    # housekeeping
    def error(self, reqId:int, code:int, msg:str, advancedOrderRejectJson:str=""):
        self.log.error(f"ERR code={code} reqId={reqId} msg={msg}")

    def marketDataType(self, reqId:int, mktDataType:int):
        # 1=live, 2=frozen, 3=delayed, 4=delayed-frozen
        self.log.info(f"marketDataType for reqId={reqId}: {mktDataType}")

    # sizes, prices, strings
    def tickSize(self, reqId:int, tickType:int, size:int):
        if tickType in (TickSizeType.BID_SIZE, TickSizeType.ASK_SIZE, TickSizeType.LAST_SIZE,
                        TickSizeType.VOLUME, TickSizeType.CALL_OPEN_INT, TickSizeType.PUT_OPEN_INT):
            self.log.info(f"id{reqId} : {self._labels.get(reqId, reqId)} size {TickSizeType(tickType).name} = {size}")

    def tickPrice(self, reqId:int, tickType:int, price:float, attrib):
        if tickType in (TickPriceType.BID, TickPriceType.ASK, TickPriceType.LAST, TickPriceType.CLOSE):
            self.log.info(f"id{reqId} : {self._labels.get(reqId, reqId)} price {TickPriceType(tickType).name} = {price}")

    def tickString(self, reqId:int, tickType:int, value:str):
        if tickType == TickStringType.LAST_TIMESTAMP:
            self.log.info(f"id{reqId} : {self._labels.get(reqId, reqId)} last_ts = {value}")

    # underlying-level vols (not contract-specific)
    def tickGeneric(self, reqId:int, tickType:int, value:float):
        if tickType in (TickGenericType.HISTORICAL_VOLATILITY_30D,
                        TickGenericType.OPTION_IMPLIED_VOL_30D,
                        TickGenericType.REALTIME_HISTORICAL_VOL):
            name = TickGenericType(tickType).name
            self.log.info(f"id{reqId} : {self._labels.get(reqId, reqId)} generic {name} = {value:.6f}")

    # per-contract IV & Greeks
    def tickOptionComputation(self, reqId:int, tickType:int,
                              impliedVol:float, delta:float, optPrice:float, pvDividend:float,
                              gamma:float, vega:float, theta:float, undPrice:float):
        # Handle both live (10-13) and delayed (80-83)
        name = TickOptionCompType(tickType).name if tickType in TickOptionCompType._value2member_map_ else str(tickType)
        self.log.info(
            f"id{reqId} : "
            f"{self._labels.get(reqId, reqId)} {name} "
            f"IV={impliedVol:.6f} Δ={delta:.6f} Γ={gamma:.6f} Θ={theta:.6f} Vega={vega:.6f} "
            f"OptPx={optPrice:.4f} PVdiv={pvDividend:.4f} Und={undPrice:.4f}"
        )

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = LiveVolTap()
    app.connect("127.0.0.1", 4002, clientId=42)
    time.sleep(1)

    # Off-hours friendly:
    # Prefer 2 (frozen) if you have live; otherwise 4 (delayed-frozen) gives static delayed values on weekends.
    app.reqMarketDataType(4)

    # ---- Underlying 30d HV/IV (underlying-level metrics) ----
    rid = 1000
    stk = make_stock("AMD", primaryExchange="NASDAQ")
    app._labels[rid] = "AMD.Underlier"
    # 104=HV30 -> tickGeneric 23; 106=IV30 -> tickGeneric 24; 411=RT HV30 -> tickGeneric 58
    app.reqMktData(rid, stk, "100,101,104,106,411", False, False, [])

    # ---- Multiple option contracts at once ----
    specs: List[OptionSpec] = [
        OptionSpec("AMD", "20250919", 170, "C"),
        OptionSpec("AMD", "20250919", 175, "C"),
        OptionSpec("AMD", "20250919", 175, "P"),
        OptionSpec("AMD", "20251017", 180, "C"),
    ]
    base = 2000
    for i, spec in enumerate(specs):
        req_id = base + i
        app._labels[req_id] = f"{spec.symbol} {spec.yyyymmdd} {spec.right}{spec.strike}"
        app.reqMktData(req_id, make_option(spec), "", False, False, [])
        # No generic ticks needed: tickOptionComputation 10/11/12/13 (or delayed 80..83) will stream
    app.run()
    # Let it stream a bit
    time.sleep(25)
    app.disconnect()

if __name__ == "__main__":
    main()

# from decimal import Decimal
# UNSET_DEC = Decimal("170141183460469231731687303715884105727")  # treat as missing
# def _is_valid_size(x):
#     try: return x != UNSET_DEC and x is not None
#     except: return False
# # before logging sizes:
# if _is_valid_size(size):
#     log(...)
