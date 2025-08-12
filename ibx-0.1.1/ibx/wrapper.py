
import logging
from ibapi.wrapper import EWrapper

from ibapi.common import *

log = logging.getLogger("ibx.wrapper")

def _tick_name(tickType):
    try:
        from ibapi.ticktype import TickTypeEnum
        # Some ibapi versions expose toStr (camel-case)
        return TickTypeEnum.toStr(tickType)
    except Exception:
        try:
            # very old versions or if tickType already a string
            return str(tickType)
        except Exception:
            return repr(tickType)

class IBWrapperBridge(EWrapper):
    """Routes ibapi callbacks to FutureRegistry and SubscriptionRegistry."""
    def __init__(self, sequencer, future_registry, subs_registry):
        super().__init__()
        self._seq = sequencer
        self._reg = future_registry
        self._subs = subs_registry

    # lifecycle
    def nextValidId(self, orderId: int):
        self._seq.set_base(orderId)

    # error routing: forward to futures if any; else to subscriptions
    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        if reqId in (-1, 0):
            log.warning("IB error (no reqId) code=%s msg=%s", errorCode, errorString)
            return
        handled = False
        try:
            handled = self._reg.set_error(reqId, errorCode, errorString)
        except Exception:
            handled = False
        if not handled:
            self._subs.set_error(reqId, errorCode, errorString)

    # ---- Match Symbols    ----
    def symbolSamples(
            self, reqId: int, contractDescriptions: ListOfContractDescription
    ):
        for cd in contractDescriptions:
            self._reg.add_item(reqId, {
                "contract": cd.contract,
                "derivativeSecTypes": cd.derivativeSecTypes,
            })
        self._reg.finish(reqId)

    # ---- Contract Details ----
    def contractDetails(self, reqId, contractDetails):
        self._reg.add_item(reqId, contractDetails)

    def contractDetailsEnd(self, reqId):
        self._reg.finish(reqId)

    # ---- SecDef Option Params ----
    def securityDefinitionOptionParameter(self, reqId, exchange, underlyingConId, tradingClass, multiplier,
                                          expirations, strikes):
        self._reg.add_item(reqId, {
            "exchange": exchange,
            "underlyingConId": underlyingConId,
            "tradingClass": tradingClass,
            "multiplier": multiplier,
            "expirations": list(expirations),
            "strikes": list(strikes),
        })

    def securityDefinitionOptionParameterEnd(self, reqId):
        self._reg.finish(reqId)

    # ---- Historical Data ----
    def historicalData(self, reqId, bar):
        self._reg.add_item(reqId, {
            "date": bar.date, "open": bar.open, "high": bar.high,
            "low": bar.low, "close": bar.close, "volume": bar.volume,
            "wap": getattr(bar, "wap", None), "barCount": getattr(bar, "barCount", None)
        })

    def historicalDataEnd(self, reqId, start, end):
        self._reg.finish(reqId)

    # ---- Streaming Market Data ----
    def marketDataType(self, reqId, marketDataType):
        self._subs.dispatch(reqId, {"type": "marketDataType", "mode": marketDataType})

    def tickPrice(self, reqId, tickType, price, attrib):
        self._subs.dispatch(reqId, {
            "type": "tickPrice",
            "tick": _tick_name(tickType),
            "price": price,
            "attrib": {
                "canAutoExecute": getattr(attrib, "canAutoExecute", None),
                "pastLimit": getattr(attrib, "pastLimit", None),
                "preOpen": getattr(attrib, "preOpen", None),
            }
        })

    def tickSize(self, reqId, tickType, size):
        self._subs.dispatch(reqId, {
            "type": "tickSize",
            "tick": _tick_name(tickType),
            "size": size
        })

    def tickString(self, reqId, tickType, value):
        self._subs.dispatch(reqId, {
            "type": "tickString",
            "tick": _tick_name(tickType),
            "value": value
        })

    def tickGeneric(self, reqId, tickType, value):
        self._subs.dispatch(reqId, {
            "type": "tickGeneric",
            "tick": _tick_name(tickType),
            "value": value
        })

    def tickOptionComputation(self, reqId, tickType, impliedVol, delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
        self._subs.dispatch(reqId, {
            "type": "tickOption",
            "tick": _tick_name(tickType),
            "impliedVol": impliedVol, "delta": delta, "optPrice": optPrice,
            "pvDividend": pvDividend, "gamma": gamma, "vega": vega, "theta": theta,
            "underPrice": undPrice
        })
