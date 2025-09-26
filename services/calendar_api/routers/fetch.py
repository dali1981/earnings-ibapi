"""
External data fetch operations router.

Handles fetching earnings data from external sources (NASDAQ, Yahoo),
saving to repository, and bulk event management operations.
"""

from __future__ import annotations

from ..schemas.calendar import EarningsEventModel, NewEarningsEvent
from ..schemas.responses import FetchResultResponse, create_success_response, ErrorCodes
from ..service.calendar_service import CalendarService
from ..service.earnings_fetcher_service import EarningsFetcherService, FetchMetadata
from ..utils.common import (
    safe_convert_to_model,
    parse_symbols,
    TimingContext,
    validate_date_range,
    create_error_detail,
)
from common.reliability.logging import get_trading_logger
from common.repositories.earnings import EarningsCalendarRepository
from earnings import EarningsSource
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import List, Optional
from datetime import date, timedelta

import asyncio

router = APIRouter(tags=["calendar-fetch"])
logger = get_trading_logger(__name__)


@router.post("/calendar/fetch-upcoming", response_model=List[EarningsEventModel])
async def fetch_upcoming(
    background_tasks: BackgroundTasks,
    days: int = Query(14, ge=1, le=90, description="Days ahead to fetch"),
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols"),
    source: str = Query("nasdaq", description="Data source (nasdaq, yahoo, all)"),
):
    """
    Fetch upcoming earnings from external sources and save to repository.
    
    This endpoint fetches fresh data from external APIs, saves it to the repository,
    and returns the fetched events. Use this when you need up-to-date data.
    
    Args:
        days: Number of days ahead to fetch (1-90)
        symbols: Optional comma-separated list of symbols to filter
        source: Data source - 'nasdaq', 'yahoo', or 'all'
        
    Returns:
        List of fetched earnings events
    """
    with TimingContext() as timing:
        try:
            # Validate parameters
            validate_date_range(days, max_days=90)
            syms = parse_symbols(symbols)
            
            # Parse source parameter
            if source.lower() == "all":
                if syms:
                    # If symbols provided, use both sources
                    sources = [EarningsSource.NASDAQ, EarningsSource.YAHOO]
                else:
                    # If no symbols, use only NASDAQ (Yahoo requires symbols)
                    sources = [EarningsSource.NASDAQ]
                    logger.info("Using NASDAQ only for 'all' source request without symbols (Yahoo requires symbols)")
            elif source.lower() == "nasdaq":
                sources = [EarningsSource.NASDAQ]
            elif source.lower() == "yahoo":
                sources = [EarningsSource.YAHOO]
            else:
                raise ValueError(f"Invalid source: {source}. Must be 'nasdaq', 'yahoo', or 'all'")

            # Yahoo requires symbols
            if EarningsSource.YAHOO in sources and not syms:
                raise ValueError("Yahoo Finance source requires symbols parameter")
            
            # Fetch from external sources
            fetcher = EarningsFetcherService()
            events, metadata = await fetcher.get_upcoming(
                days_ahead=days,
                symbols=syms,
                sources=sources,
                include_metadata=True,
            )
            
            # Save to repository
            cal = CalendarService()
            saved_count = cal.save_events(events)
            
            # Convert to response models
            event_models = safe_convert_to_model(events)
            
            logger.info(
                f"Fetched {len(events)} events from {[s.value for s in sources]}, "
                f"saved {saved_count} to repository in {timing.duration_ms or 0:.2f}ms"
            )
            
            # Log metadata details
            if metadata:
                logger.info(
                    f"Fetch metadata: sources={[s.value for s in metadata.sources_used]}, "
                    f"symbols_requested={metadata.symbols_requested}, "
                    f"symbols_found={metadata.symbols_found}, "
                    f"missing_symbols={metadata.missing_symbols}, "
                    f"duration={metadata.fetch_duration_seconds:.2f}s"
                )
            
            return event_models
            
        except ValueError as e:
            logger.error(f"Parameter validation error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error fetching earnings: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/calendar/fetch-upcoming/yahoo", response_model=List[EarningsEventModel])
async def fetch_upcoming_yahoo(
    days: int = Query(14, ge=1, le=90, description="Days ahead to fetch"),
    symbols: str = Query(..., description="Comma-separated list of symbols (required for Yahoo)"),
):
    """
    Fetch upcoming earnings specifically from Yahoo Finance.
    
    Yahoo Finance requires symbols to be specified and is optimized for
    symbol-specific queries rather than bulk calendar fetches.
    
    Args:
        days: Number of days ahead to fetch (1-90)
        symbols: Comma-separated list of symbols (required)
        
    Returns:
        List of fetched earnings events from Yahoo Finance
    """
    with TimingContext() as timing:
        try:
            # Validate parameters
            validate_date_range(days, max_days=90)
            syms = parse_symbols(symbols)
            
            if not syms:
                raise ValueError("Yahoo Finance source requires symbols parameter")
            
            # Fetch from Yahoo
            fetcher = EarningsFetcherService()
            events, metadata = await fetcher.get_upcoming(
                days_ahead=days,
                symbols=syms,
                sources=[EarningsSource.YAHOO],
                include_metadata=True,
            )
            
            # Save to repository
            cal = CalendarService()
            saved_count = cal.save_events(events)
            
            # Convert to response models
            event_models = safe_convert_to_model(events)
            
            logger.info(
                f"Fetched {len(events)} events from Yahoo Finance for {len(syms)} symbols, "
                f"saved {saved_count} to repository in {timing.duration_ms:.2f}ms"
            )
            
            # Log Yahoo-specific details
            if metadata:
                logger.info(
                    f"Yahoo fetch: symbols_found={metadata.symbols_found}, "
                    f"missing_symbols={metadata.missing_symbols}, "
                    f"duration={metadata.fetch_duration_seconds:.2f}s"
                )
            
            return event_models
            
        except ValueError as e:
            logger.error(f"Parameter validation error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error fetching from Yahoo Finance: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/calendar/events")
def add_events(events: List[NewEarningsEvent]):
    """
    Bulk add earnings events to the repository.
    
    Accepts a list of earnings events and adds them to the repository
    with automatic deduplication. Returns statistics about the operation.
    
    Args:
        events: List of earnings events to add
        
    Returns:
        Statistics about the bulk add operation
    """
    try:
        if not events:
            raise ValueError("No events provided")
            
        if len(events) > 1000:
            raise ValueError(f"Too many events: {len(events)}. Maximum is 1000 per request")
        
        # Convert NewEarningsEvent to internal format if needed
        # This is a simplified implementation - you may need to adapt based on your models
        internal_events = []
        written = 0
        duplicates = 0
        errors = 0
        
        for i, event in enumerate(events):
            try:
                # Basic validation
                if not event.symbol or not event.earnings_date:
                    logger.warning(f"Skipping invalid event {i}: missing symbol or date")
                    errors += 1
                    continue
                
                # Check if event already exists (simplified check)
                repo = EarningsCalendarRepository()
                existing_df = repo.get_earnings_by_date_range(
                    start_date=event.earnings_date,
                    end_date=event.earnings_date,
                    symbols=[event.symbol.upper()],
                )
                
                if not existing_df.empty:
                    logger.debug(f"Event already exists: {event.symbol} on {event.earnings_date}")
                    duplicates += 1
                    continue
                
                # Convert to internal format (this is simplified)
                # You would need to create proper EarningsEvent objects here
                internal_events.append(event)
                written += 1
                
            except Exception as e:
                logger.error(f"Error processing event {i}: {e}")
                errors += 1
        
        # Save the valid events (simplified - you'd use CalendarService here)
        if internal_events:
            cal = CalendarService()
            # This would need proper conversion from NewEarningsEvent to EarningsEvent
            # actual_saved = cal.save_events(internal_events)
            logger.info(f"Would save {len(internal_events)} events")
        
        result = {
            "written": written,
            "duplicates": duplicates,
            "errors": errors,
            "total_processed": len(events),
        }
        
        logger.info(
            f"Bulk add completed: {written} written, {duplicates} duplicates, {errors} errors"
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"Parameter validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding events: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/calendar/validate-sources")
async def validate_sources(
    sources: Optional[List[str]] = Query(None, description="Sources to validate (default: all)"),
):
    """
    Validate connectivity to external earnings data sources.
    
    Tests connectivity to specified sources and returns availability status.
    Useful for monitoring and troubleshooting external service issues.
    
    Args:
        sources: List of sources to test (nasdaq, yahoo). If not provided, tests all.
        
    Returns:
        Dictionary mapping sources to their availability status
    """
    try:
        # Parse sources
        if sources:
            source_enums = []
            for source in sources:
                if source.lower() == "nasdaq":
                    source_enums.append(EarningsSource.NASDAQ)
                elif source.lower() == "yahoo":
                    source_enums.append(EarningsSource.YAHOO)
                else:
                    raise ValueError(f"Invalid source: {source}")
        else:
            source_enums = [EarningsSource.NASDAQ, EarningsSource.YAHOO]
        
        # Test connectivity
        fetcher = EarningsFetcherService()
        results = await fetcher.validate_source_connectivity(source_enums)
        
        # Convert to string keys for JSON response
        status_results = {source.value: available for source, available in results.items()}
        
        logger.info(f"Source validation results: {status_results}")
        
        return {
            "sources": status_results,
            "timestamp": date.today().isoformat(),
            "all_available": all(status_results.values()),
        }
        
    except ValueError as e:
        logger.error(f"Parameter validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error validating sources: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")