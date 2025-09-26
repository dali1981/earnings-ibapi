"""
Core calendar read operations router.

Handles basic calendar data retrieval operations including health checks,
upcoming earnings queries, presence checks, and symbol-specific lookups.
"""

from __future__ import annotations

from ..schemas.calendar import (
    EarningsEventModel,
    EarningsQueryMetadata,
    EarningsResponseWithMetadata,
)
from ..schemas.responses import HealthResponse, create_success_response, ErrorCodes
from ..service.calendar_service import CalendarService
from ..utils.common import (
    safe_convert_to_model,
    parse_symbols,
    TimingContext,
    validate_date_range,
    add_cache_headers,
    add_metadata_headers,
)
from common.reliability.logging import get_trading_logger
from common.repositories.earnings import EarningsCalendarRepository
from datetime import date, timedelta
from fastapi import APIRouter, HTTPException, Query, Response
from typing import List, Optional

import time

router = APIRouter(tags=["calendar-core"])
logger = get_trading_logger(__name__)


@router.get("/health", response_model=HealthResponse)
def health():
    """
    Service health check endpoint.
    
    Returns basic service status and version information.
    """
    return HealthResponse(
        status="healthy",
        version="1.0",
        dependencies={
            "calendar_service": "available",
            "earnings_repository": "available",
        }
    )


@router.get("/calendar/upcoming", response_model=EarningsResponseWithMetadata)
async def upcoming(
    response: Response,
    days: int = Query(14, ge=0, le=180, description="Days ahead to query"),
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols"),
):
    """
    Get upcoming earnings from repository with enhanced metadata.
    
    This is a read-only endpoint that queries the local repository.
    Use /fetch/upcoming to fetch from external sources and persist.
    
    Args:
        days: Number of days ahead to query (0-180)
        symbols: Optional comma-separated list of symbols to filter
        
    Returns:
        Earnings events with comprehensive metadata including suggestions
    """
    with TimingContext() as timing:
        try:
            # Validate and parse parameters
            validate_date_range(days, max_days=180)
            syms = parse_symbols(symbols)
            
            # Initialize services
            cal = CalendarService()
            repo = EarningsCalendarRepository()

            # Get filtered events
            events = cal.get_upcoming_events(days_ahead=days, symbols=syms)

            # Get unfiltered events for metadata
            start_date = date.today()
            end_date = start_date + timedelta(days=days)
            all_events_df = repo.get_earnings_by_date_range(start_date, end_date)

            # Build metadata
            symbols_found = list(set(e.symbol for e in events)) if events else []
            suggestions = []

            # If no data found for requested symbols, provide suggestions
            if syms and not events and not all_events_df.empty:
                suggestion_symbols = list(all_events_df["symbol"].unique()[:5])
                suggestions = suggestion_symbols

            # Create metadata object
            metadata = EarningsQueryMetadata(
                total_events=len(events),
                filtered_events=len(events),
                date_range_start=start_date.isoformat(),
                date_range_end=end_date.isoformat(),
                symbols_requested=syms or [],
                symbols_found=symbols_found,
                symbol_suggestions=suggestions,
                latest_collection_date=cal.get_latest_collection_date_str(),
                query_timestamp=time.time(),
            )

            # Convert events to models
            event_models = safe_convert_to_model(events)

            # Create response
            result = EarningsResponseWithMetadata(
                events=event_models,
                metadata=metadata,
            )

            # Add response headers
            add_cache_headers(response, max_age_seconds=300)
            add_metadata_headers(response, {
                "source": "repository",
                "collection_date": metadata.latest_collection_date,
                "events_count": len(event_models),
            })

            logger.info(
                f"Retrieved {len(event_models)} upcoming earnings events "
                f"for {days} days ahead"
                + (f" filtered by {len(syms)} symbols" if syms else "")
            )

            return result
            
        except ValueError as e:
            logger.error(f"Parameter validation error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error retrieving upcoming earnings: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/calendar/get-upcoming", response_model=List[EarningsEventModel])
def get_upcoming(
    response: Response,
    days: int = Query(14, ge=0, le=180, description="Days ahead to query"),
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols"),
):
    """
    Get upcoming earnings from repository (simple list response).

    This is a read-only endpoint that returns a simple list without metadata.
    Use this for clients that need just the earnings data without extra metadata.
    
    Args:
        days: Number of days ahead to query (0-180)
        symbols: Optional comma-separated list of symbols to filter
        
    Returns:
        List of earnings events
    """
    try:
        # Validate and parse parameters
        validate_date_range(days, max_days=180)
        syms = parse_symbols(symbols)
        
        # Get events from service
        cal = CalendarService()
        events = cal.get_upcoming_events(days_ahead=days, symbols=syms)
        
        # Add response headers
        add_cache_headers(response, max_age_seconds=300)
        add_metadata_headers(response, {
            "source": "repository",
            "collection_date": cal.get_latest_collection_date_str(),
            "events_count": len(events),
        })
        
        logger.info(
            f"Retrieved {len(events)} upcoming earnings events (simple format) "
            f"for {days} days ahead"
            + (f" filtered by {len(syms)} symbols" if syms else "")
        )
        
        return safe_convert_to_model(events)
        
    except ValueError as e:
        logger.error(f"Parameter validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving upcoming earnings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/calendar/present")
def is_present(
    symbol: str = Query(..., description="Symbol to check"),
    earnings_date: date = Query(..., description="Earnings date to check"),
    source: Optional[str] = Query(None, description="Source to check (optional)"),
):
    """
    Check if a specific earnings event exists in the repository.
    
    Args:
        symbol: Stock symbol to check
        earnings_date: Date of earnings to check
        source: Optional source filter
        
    Returns:
        Boolean indicating if the event exists
    """
    try:
        repo = EarningsCalendarRepository()
        
        # Query repository for the specific event
        df = repo.get_earnings_by_date_range(
            start_date=earnings_date,
            end_date=earnings_date,
            symbols=[symbol.upper()],
            sources=[source] if source else None,
        )
        
        exists = not df.empty
        
        logger.debug(
            f"Presence check for {symbol} on {earnings_date} "
            f"(source: {source or 'any'}): {exists}"
        )
        
        return {"present": exists, "symbol": symbol.upper(), "date": earnings_date}
        
    except Exception as e:
        logger.error(f"Error checking event presence: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/calendar/symbol/{symbol}")
def symbol_earnings(
    symbol: str,
    days_back: int = Query(365, ge=1, le=3650, description="Days back to search"),
):
    """
    Get historical earnings for a specific symbol.
    
    Args:
        symbol: Stock symbol to query
        days_back: Number of days back to search (1-3650, default 365)
        
    Returns:
        List of historical earnings events for the symbol
    """
    try:
        # Validate parameters
        if days_back < 1 or days_back > 3650:
            raise ValueError(f"days_back must be between 1 and 3650: {days_back}")
            
        symbol = symbol.upper()
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        # Query repository
        repo = EarningsCalendarRepository()
        df = repo.get_earnings_by_date_range(
            start_date=start_date,
            end_date=end_date,
            symbols=[symbol],
        )
        
        if df.empty:
            logger.info(f"No historical earnings found for {symbol} in last {days_back} days")
            return []
        
        # Convert DataFrame to events
        events = []
        for _, row in df.iterrows():
            # Create event objects from DataFrame rows
            # This is a simplified version - you might want to create proper event objects
            events.append({
                "symbol": row.get("symbol"),
                "company_name": row.get("company_name", row.get("symbol")),
                "earnings_date": row.get("earnings_date"),
                "time": row.get("time"),
                "eps_estimate": row.get("eps_estimate"),
                "eps_actual": row.get("eps_actual"),
                "revenue_estimate": row.get("revenue_estimate"),
                "revenue_actual": row.get("revenue_actual"),
                "source": row.get("source", "unknown"),
            })
        
        logger.info(
            f"Retrieved {len(events)} historical earnings for {symbol} "
            f"over last {days_back} days"
        )
        
        return events
        
    except ValueError as e:
        logger.error(f"Parameter validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving symbol earnings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")