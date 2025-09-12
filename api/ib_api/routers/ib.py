from __future__ import annotations

from api.ib_api.service.ib_runtime import get_runtime
from app_settings.settings import get_settings
from fastapi import APIRouter, HTTPException, status, WebSocket, WebSocketDisconnect
from ibapi.contract import Contract
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from vendors.ibx.services import (
    ContractDetailsService,
    HistoricalService,
    MatchingSymbolService,
    SecDefService,
    SubscriptionService,
)

import asyncio


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


def _ensure_connected(rt) -> None:
    try:
        if hasattr(rt, "client") and not rt.client.isConnected():
            s = get_settings()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"IB runtime not connected. Ensure IB Gateway/TWS is running and reachable at "
                    f"{s.ib.host}:{s.ib.port}, or update IBX_IB__HOST/PORT settings."
                ),
            )
    except AttributeError:
        # If client attribute missing, treat as unavailable
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="IB runtime unavailable (client missing).",
        )


class MatchingSymbolsRequest(BaseModel):
    pattern: str = Field(..., description="Symbol search pattern", examples=["AAPL"])
    timeout: float = Field(10.0, examples=[10.0])


@router.post("/matching-symbols")
def matching_symbols(req: MatchingSymbolsRequest) -> List[Dict[str, Any]]:
    rt = get_runtime()
    _ensure_connected(rt)
    service = MatchingSymbolService(rt)
    res = service.fetch(req.pattern, timeout=req.timeout)
    return [_serialize(x) for x in res]


class ContractDetailsRequest(BaseModel):
    contract: Dict[str, Any] = Field(
        ...,
        description="ibapi.contract.Contract fields or conId",
        examples=[
            {
                "symbol": "AAPL",
                "secType": "STK",
                "exchange": "SMART",
                "currency": "USD",
            },
            {"conId": 265598},
        ],
    )
    timeout: float = Field(15.0, examples=[15.0])


@router.post("/contract-details")
def contract_details(req: ContractDetailsRequest) -> List[Dict[str, Any]]:
    rt = get_runtime()
    _ensure_connected(rt)
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
    _ensure_connected(rt)
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
        examples=[
            {"symbol": "AAPL", "secType": "STK", "exchange": "SMART", "currency": "USD"}
        ],
    )
    endDateTime: str = Field(..., examples=["20240823 23:59:59"])
    durationStr: str = Field(..., examples=["1 D"])
    barSizeSetting: str = Field(..., examples=["1 min"])
    whatToShow: str = Field(..., examples=["TRADES"])
    useRTH: int = Field(1, examples=[1])
    formatDate: int = Field(1, examples=[1])
    keepUpToDate: bool = Field(False, examples=[False])
    timeout: float = Field(20.0, examples=[20.0])


class HealthResponse(BaseModel):
    connected: bool
    host: str
    port: int
    client_id: int
    message: Optional[str] = None


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s = get_settings()
    msg: Optional[str] = None
    connected = False
    try:
        rt = get_runtime()
        try:
            connected = bool(getattr(rt.client, "isConnected", lambda: False)())
        except Exception as e:  # pragma: no cover - defensive
            msg = f"error checking connection: {e}"
    except Exception as e:
        msg = str(e)
        connected = False
    return HealthResponse(
        connected=connected,
        host=s.ib.host,
        port=s.ib.port,
        client_id=s.ib.client_id_contracts,
        message=msg,
    )


@router.post("/historical/bars")
def historical_bars(req: HistoricalBarsRequest) -> List[Dict[str, Any]]:
    import logging
    import traceback

    logger = logging.getLogger(__name__)

    rt = get_runtime()
    _ensure_connected(rt)
    service = HistoricalService(rt)
    contract = _build_contract(req.contract)

    logger.info(
        f"📡 IB API: Historical bars request for {contract.symbol if hasattr(contract, 'symbol') else 'N/A'}"
    )

    # Format endDateTime with proper timezone
    from common.datetime.end_datetime import EndDateTime

    formatted_end_time = EndDateTime(req.endDateTime).to_ib_format()

    try:
        rows = service.bars(
            contract=contract,
            endDateTime=formatted_end_time,
            durationStr=req.durationStr,
            barSizeSetting=req.barSizeSetting,
            whatToShow=req.whatToShow,
            useRTH=req.useRTH,
            formatDate=req.formatDate,
            keepUpToDate=req.keepUpToDate,
            timeout=req.timeout,
        )
        logger.info(
            f"📡 IB API: Successfully returned {len(rows)} bars for {contract.symbol if hasattr(contract, 'symbol') else 'N/A'}"
        )
        return rows
    except Exception as e:
        logger.error(f"📡 IB API: Error in historical bars request: {str(e)}")
        logger.error(f"📡 IB API: Full traceback:\n{traceback.format_exc()}")
        logger.error(
            f"📡 IB API: Request details - contract: {req.contract}, endDateTime: {req.endDateTime}, duration: {req.durationStr}, barSize: {req.barSizeSetting}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Historical bars request failed: {str(e)} | Full error: {repr(e)}",
        )


@router.websocket("/ws/market-data")
async def ws_market_data(ws: WebSocket):
    await ws.accept()
    rt = get_runtime()
    try:
        _ensure_connected(rt)
    except HTTPException as exc:
        await ws.send_json({"error": exc.detail})
        await ws.close(code=1013)
        return
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
