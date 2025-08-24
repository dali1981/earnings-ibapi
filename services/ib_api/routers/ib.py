from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from services.ib_api.service.ib_runtime import get_runtime
from vendors.ibx.services import (
    ContractDetailsService,
    HistoricalService,
    MatchingSymbolService,
    SecDefService,
    SubscriptionService,
)
from ibapi.contract import Contract


router = APIRouter()


def _build_contract(payload: Dict[str, Any]) -> Contract:
    c = Contract()
    for k, v in payload.items():
        if hasattr(c, k):
            setattr(c, k, v)
    return c


def _serialize(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_serialize(x) for x in obj]
    # Try to serialize known IBAPI objects by their __dict__
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        return {k: _serialize(v) for k, v in d.items() if not k.startswith("_")}
    return str(obj)


class MatchingSymbolsRequest(BaseModel):
    pattern: str = Field(..., description="Symbol search pattern", examples=["AAPL"])
    timeout: float = Field(10.0, examples=[10.0])


@router.post("/matching-symbols")
def matching_symbols(req: MatchingSymbolsRequest) -> List[Dict[str, Any]]:
    rt = get_runtime()
    service = MatchingSymbolService(rt)
    res = service.fetch(req.pattern, timeout=req.timeout)
    return [_serialize(x) for x in res]


class ContractDetailsRequest(BaseModel):
    contract: Dict[str, Any] = Field(
        ...,
        description="ibapi.contract.Contract fields or conId",
        examples=[
            {"symbol": "AAPL", "secType": "STK", "exchange": "SMART", "currency": "USD"},
            {"conId": 265598},
        ],
    )
    timeout: float = Field(15.0, examples=[15.0])


@router.post("/contract-details")
def contract_details(req: ContractDetailsRequest) -> List[Dict[str, Any]]:
    rt = get_runtime()
    service = ContractDetailsService(rt)
    contract = _build_contract(req.contract)
    res = service.fetch(contract, timeout=req.timeout)
    return [_serialize(x) for x in res]


class SecDefParamsRequest(BaseModel):
    symbol: str = Field(..., examples=["AAPL"])
    conid: int = Field(..., examples=[265598])
    sec_type: str = Field("STK", examples=["STK"]) 
    exchange: str = Field("", examples=[""])
    timeout: float = Field(10.0, examples=[10.0])


@router.post("/secdef/option-params")
def secdef_option_params(req: SecDefParamsRequest) -> List[Dict[str, Any]]:
    rt = get_runtime()
    service = SecDefService(rt)
    return service.option_params(
        symbol=req.symbol,
        conid=req.conid,
        sec_type=req.sec_type,
        exchange=req.exchange,
        timeout=req.timeout,
    )


class HistoricalBarsRequest(BaseModel):
    contract: Dict[str, Any] = Field(
        ...,
        examples=[{"symbol": "AAPL", "secType": "STK", "exchange": "SMART", "currency": "USD"}],
    )
    endDateTime: str = Field(..., examples=["20240823 23:59:59"])
    durationStr: str = Field(..., examples=["1 D"]) 
    barSizeSetting: str = Field(..., examples=["1 min"]) 
    whatToShow: str = Field(..., examples=["TRADES"]) 
    useRTH: int = Field(1, examples=[1])
    formatDate: int = Field(1, examples=[1])
    keepUpToDate: bool = Field(False, examples=[False])
    timeout: float = Field(20.0, examples=[20.0])


@router.post("/historical/bars")
def historical_bars(req: HistoricalBarsRequest) -> List[Dict[str, Any]]:
    rt = get_runtime()
    service = HistoricalService(rt)
    contract = _build_contract(req.contract)
    rows = service.bars(
        contract=contract,
        endDateTime=req.endDateTime,
        durationStr=req.durationStr,
        barSizeSetting=req.barSizeSetting,
        whatToShow=req.whatToShow,
        useRTH=req.useRTH,
        formatDate=req.formatDate,
        keepUpToDate=req.keepUpToDate,
        timeout=req.timeout,
    )
    return rows


@router.websocket("/ws/market-data")
async def ws_market_data(ws: WebSocket):
    await ws.accept()
    rt = get_runtime()
    service = SubscriptionService(rt)

    # First message must be a JSON config with contract and options
    try:
        init = await ws.receive_json()
    except Exception:
        await ws.close(code=1003)
        return

    contract = _build_contract(init.get("contract", {}))
    genericTicks = str(init.get("genericTicks", ""))
    snapshot = bool(init.get("snapshot", False))
    regulatorySnapshot = bool(init.get("regulatorySnapshot", False))
    qsize = int(init.get("qsize", 1000))

    handle = service.market_data(
        contract,
        genericTicks=genericTicks,
        snapshot=snapshot,
        regulatorySnapshot=regulatorySnapshot,
        on_event=None,
        qsize=qsize,
    )

    try:
        while True:
            try:
                ev = handle.get(timeout=1.0)
            except Exception:
                ev = None
            if ev is not None:
                await ws.send_json(_serialize(ev))
            # Check for incoming client pings or close frames
            try:
                _ = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        handle.cancel()
    except Exception:
        handle.cancel()
        try:
            await ws.close()
        except Exception:
            pass
