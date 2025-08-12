
from typing import Any, Dict, List
from ibapi.contract import Contract
from .futures import IBFuture
from .runtime import IBRuntime

class MatchingSymbolService:
    def __init__(self, rt: IBRuntime):
        self.rt = rt

    def fetch(self, pattern: str, timeout: float = 10.0) -> List[Any]:
        rid = self.rt.sequencer.next()
        fut = self.rt.registry.register(rid, IBFuture(expect_many=True, timeout=timeout))
        self.rt.client.reqMatchingSymbols(rid, pattern)
        return fut.result()

class ContractDetailsService:
    def __init__(self, rt: IBRuntime):
        self.rt = rt

    def fetch(self, contract: Contract, timeout: float = 10.0) -> List[Any]:
        rid = self.rt.sequencer.next()
        fut = self.rt.registry.register(rid, IBFuture(expect_many=True, timeout=timeout))
        self.rt.client.reqContractDetails(rid, contract)
        return fut.result()

class SecDefService:
    def __init__(self, rt: IBRuntime):
        self.rt = rt

    def option_params(self, symbol: str, conid: int, sec_type: str = "STK", exchange: str = "",
                      timeout: float = 10.0) -> List[Dict[str, Any]]:
        rid = self.rt.sequencer.next()
        fut = self.rt.registry.register(rid, IBFuture(expect_many=True, timeout=timeout))
        self.rt.client.reqSecDefOptParams(rid, symbol, exchange, sec_type, conid)
        return fut.result()

class HistoricalService:
    def __init__(self, rt: IBRuntime):
        self.rt = rt

    def bars(self, contract: Contract, endDateTime: str, durationStr: str,
             barSizeSetting: str, whatToShow: str, useRTH: int,
             formatDate: int = 1, keepUpToDate: bool = False, chartOptions=None,
             timeout: float = 20.0) -> List[Dict[str, Any]]:
            # fixed indent in previous version if any
        rid = self.rt.sequencer.next()
        fut = self.rt.registry.register(rid, IBFuture(expect_many=True, timeout=timeout))
        self.rt.client.reqHistoricalData(
            rid, contract, endDateTime, durationStr, barSizeSetting,
            whatToShow, useRTH, formatDate, keepUpToDate, chartOptions or []
        )
        return fut.result()

class SubscriptionService:
    def __init__(self, rt: IBRuntime):
        self.rt = rt

    def market_data(self, contract: Contract, genericTicks: str = "",
                    snapshot: bool = False, regulatorySnapshot: bool = False,
                    mktDataOptions=None, on_event=None, qsize: int = 1000):
        from .subscriptions import Subscription, SubscriptionHandle
        rid = self.rt.sequencer.next()
        sub = Subscription(on_event=on_event, qsize=qsize)
        self.rt.subs.register(rid, sub)
        self.rt.client.reqMktData(
            rid, contract, genericTicks, snapshot, regulatorySnapshot, mktDataOptions or []
        )
        return SubscriptionHandle(self.rt, rid, sub)
