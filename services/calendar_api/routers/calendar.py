from __future__ import annotations

import json
from ..schemas.calendar import EarningsEventModel, NewEarningsEvent
from ..service.calendar_service import CalendarService
from ..service.earnings_fetcher_service import EarningsFetcherService
from common.reliability.logging import get_trading_logger
from common.repositories.earnings import EarningsCalendarRepository
from datetime import date, timedelta
from earnings import EarningsSource
from fastapi import APIRouter, HTTPException, Query, Response
from typing import List, Optional


router = APIRouter()
logger = get_trading_logger(__name__)


def safe_convert_to_model(events: list, event_type=EarningsEventModel) -> List[EarningsEventModel]:
    """
    Safely convert earnings events to Pydantic models with JSON serialization validation.
    
    Logs detailed error information for any events that fail to serialize.
    """
    result = []
    for i, event in enumerate(events):
        try:
            # Convert to model
            model = event_type.from_entity(event)
            
            # Validate JSON serialization
            json.dumps(model.dict())
            
            result.append(model)
        except (ValueError, TypeError) as ex:
            # Log the problematic event details
            logger.error(f"Failed to serialize event {i}/{len(events)}: {ex}")
            logger.error(f"Event: symbol={getattr(event, 'symbol', 'N/A')}, "
                        f"date={getattr(event, 'earnings_date', 'N/A')}, "
                        f"eps_est={getattr(event, 'eps_estimate', 'N/A')}, "
                        f"eps_act={getattr(event, 'eps_actual', 'N/A')}, "
                        f"rev_est={getattr(event, 'revenue_estimate', 'N/A')}, "
                        f"rev_act={getattr(event, 'revenue_actual', 'N/A')}")
            
            # Try to log the full event dict if available
            try:
                logger.debug(f"Full event data: {event.__dict__}")
            except:
                pass
                
    if len(result) < len(events):
        logger.warning(f"Serialized {len(result)}/{len(events)} events successfully")
        
    return result


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/calendar/upcoming", response_model=List[EarningsEventModel])
async def upcoming(
    response: Response,
    days: int = Query(14, ge=0, le=180),
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols"),
):
    """Read-only: return upcoming earnings from the repository only.

    Use `/calendar/fetch-upcoming` to fetch from external sources and persist.
    """
    syms: Optional[List[str]] = (
        [s.strip().upper() for s in symbols.split(",")] if symbols else None
    )
    cal = CalendarService()
    events = cal.get_upcoming_events(days_ahead=days, symbols=syms)
    response.headers["X-Calendar-Source"] = "repo"
    latest = cal.get_latest_collection_date_str()
    if latest:
        response.headers["X-Calendar-Collection-Date"] = latest
    return safe_convert_to_model(events)


@router.get("/calendar/get-upcoming", response_model=List[EarningsEventModel])
def get_upcoming(
    response: Response,
    days: int = Query(14, ge=0, le=180),
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols"),
):
    """Read upcoming earnings from the repository only.

    - No external fetches.
    - If `symbols` omitted, returns all upcoming within the window from local repo.
    - Adds headers for transparency: source and latest collection date.
    """
    syms: Optional[List[str]] = (
        [s.strip().upper() for s in symbols.split(",")] if symbols else None
    )
    cal = CalendarService()
    events = cal.get_upcoming_events(days_ahead=days, symbols=syms)
    response.headers["X-Calendar-Source"] = "repo"
    latest = cal.get_latest_collection_date_str()
    if latest:
        response.headers["X-Calendar-Collection-Date"] = latest
    return safe_convert_to_model(events)


@router.post("/calendar/fetch-upcoming", response_model=List[EarningsEventModel])
async def fetch_upcoming(
    response: Response,
    days: int = Query(14, ge=0, le=180),
    source: str = Query(
        "nasdaq", description="nasdaq (default), yahoo, or all (case-insensitive)"
    ),
    symbols: Optional[str] = Query(
        None, description="Comma-separated symbols (required for yahoo)"
    ),
    symbol: Optional[str] = Query(
        None, description="Single symbol (alias for symbols)"
    ),
):
    """Fetch upcoming earnings from external source(s) and persist.

    - `source`: nasdaq (default), yahoo, or all.
    - `symbols`: optional; required if `source=yahoo`.
    - Persists to the repository when possible and returns fetched list.
    """
    fetcher = EarningsFetcherService()

    # Parse symbols from either `symbols` or `symbol` (for convenience)
    syms: Optional[List[str]] = None
    if symbols:
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    elif symbol:
        syms = [symbol.strip().upper()]

    # Case-insensitive source handling
    src = (source or "nasdaq").strip().lower()
    if src not in {"nasdaq", "yahoo", "all"}:
        raise HTTPException(
            status_code=400, detail="Invalid source. Use nasdaq, yahoo, or all."
        )

    sources: List[EarningsSource]
    if src == "yahoo":
        if not syms:
            raise HTTPException(status_code=400, detail="Yahoo requires symbols param.")
        sources = [EarningsSource.YAHOO]
    elif src == "all":
        sources = [EarningsSource.NASDAQ] + ([EarningsSource.YAHOO] if syms else [])
    else:
        sources = [EarningsSource.NASDAQ]

    fetched = await fetcher.get_upcoming(days_ahead=days, symbols=syms, sources=sources)

    cal = CalendarService()
    try:
        cal.save_events(fetched)
        response.headers["X-Calendar-Source"] = "fetched"
        response.headers["X-Calendar-Collection-Date"] = date.today().isoformat()
    except Exception as e:
        logger.error(f"Failed to persist fetched earnings: {e}")
        # Still return fetched results even if persistence fails
        response.headers["X-Calendar-Source"] = "fetched"
    return safe_convert_to_model(fetched)


@router.post("/calendar/fetch-upcoming/yahoo", response_model=List[EarningsEventModel])
async def fetch_upcoming_yahoo(
    response: Response,
    days: int = Query(14, ge=0, le=180),
    symbols: str = Query(..., description="Comma-separated list of symbols (required)"),
):
    """Yahoo-only fetch for upcoming earnings (requires symbols)."""
    syms: List[str] = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise HTTPException(
            status_code=400, detail="symbols must contain at least one ticker"
        )

    fetcher = EarningsFetcherService()
    fetched = await fetcher.get_upcoming(
        days_ahead=days, symbols=syms, sources=[EarningsSource.YAHOO]
    )

    cal = CalendarService()
    try:
        cal.save_events(fetched)
        response.headers["X-Calendar-Source"] = "fetched"
        response.headers["X-Calendar-Collection-Date"] = date.today().isoformat()
    except Exception as e:
        logger.error(f"Failed to persist fetched earnings: {e}")
        response.headers["X-Calendar-Source"] = "fetched"
    return safe_convert_to_model(fetched)


@router.post("/calendar/events")
def add_events(events: list[NewEarningsEvent]):
    """Add earnings events to the repository, deduplicating by (symbol, earnings_date, source).

    - Logs duplicates.
    - Writes only new records.
    - Returns counts and basic info.
    """
    if not events:
        return {"written": 0, "duplicates": 0, "errors": 0}
    cal = CalendarService()
    to_write = []
    duplicates = 0
    errors = 0
    from earnings.fetcher import EarningsEvent

    for e in events:
        try:
            # Check presence for that exact symbol/date/source
            repo = EarningsCalendarRepository()  # direct read for existence
            df = repo.get_earnings_by_date_range(
                e.earnings_date, e.earnings_date, symbols=[e.symbol], sources=[e.source]
            )
            exists = not df.empty
            if exists:
                duplicates += 1
                logger.info(
                    f"Duplicate earnings event received for {e.symbol} on {e.earnings_date} (source={e.source})"
                )
                continue
            to_write.append(
                EarningsEvent(
                    symbol=e.symbol,
                    company_name=e.company_name,
                    earnings_date=e.earnings_date,
                    time=e.time,
                    eps_estimate=e.eps_estimate,
                    eps_actual=e.eps_actual,
                    revenue_estimate=e.revenue_estimate,
                    revenue_actual=e.revenue_actual,
                    source=e.source,
                )
            )
        except Exception as err:
            errors += 1
            logger.error(f"Error processing event {e.symbol} {e.earnings_date}: {err}")

    written = 0
    if to_write:
        try:
            written = cal.save_events(to_write)
        except Exception as err:
            errors += len(to_write)
            logger.error(f"Failed to write earnings batch: {err}")

    return {"written": written, "duplicates": duplicates, "errors": errors}


@router.get("/calendar/present")
def is_present(symbol: str, earnings_date: date, source: str | None = None):
    repo = EarningsCalendarRepository()
    try:
        df = repo.get_earnings_by_date_range(
            earnings_date,
            earnings_date,
            symbols=[symbol],
            sources=[source] if source else None,
        )
        return {
            "symbol": symbol,
            "earnings_date": earnings_date,
            "present": (not df.empty),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calendar/symbol/{symbol}")
def symbol_earnings(symbol: str, days_back: int = Query(365, ge=1, le=3650)):
    repo = EarningsCalendarRepository()
    try:
        df = repo.get_historical_symbol_earnings(symbol, days_back=days_back)
        return {
            "symbol": symbol,
            "rows": int(len(df)),
            "events": df.head(100).to_dict(orient="records") if not df.empty else [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calendar/stats")
def calendar_stats():
    repo = EarningsCalendarRepository()
    try:
        return repo.get_storage_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
