"""
Analytics and reporting operations router.

Handles statistics, data availability analysis, symbol suggestions,
and other analytics-focused endpoints for the calendar API.
"""

from __future__ import annotations

from ..schemas.responses import StatsResponse, DataAvailabilityResponse
from ..service.calendar_service import CalendarService
from ..utils.common import (
    parse_symbols,
    validate_date_range,
    TimingContext,
    add_cache_headers,
)
from common.reliability.logging import get_trading_logger
from common.repositories.earnings import EarningsCalendarRepository
from datetime import date, timedelta
from fastapi import APIRouter, HTTPException, Query, Response
from typing import List, Optional

import pandas as pd

router = APIRouter(tags=["calendar-analytics"])
logger = get_trading_logger(__name__)


@router.get("/calendar/stats", response_model=StatsResponse)
def calendar_stats(response: Response):
    """
    Get comprehensive repository statistics.
    
    Returns detailed statistics about the earnings data repository including
    event counts, symbol coverage, storage information, and date ranges.
    
    Returns:
        Repository statistics including counts and storage metrics
    """
    try:
        repo = EarningsCalendarRepository()
        
        # Get basic statistics
        stats = repo.get_storage_stats()
        
        # Create response
        result = StatsResponse(
            total_events=stats.get("total_events", 0),
            total_symbols=stats.get("unique_symbols", 0),
            total_sources=stats.get("unique_sources", 0),
            storage_size_mb=stats.get("storage_size_mb"),
            oldest_event_date=stats.get("oldest_date"),
            newest_event_date=stats.get("newest_date"),
            partitions_count=stats.get("partitions_count"),
        )
        
        # Add cache headers (stats change infrequently)
        add_cache_headers(response, max_age_seconds=600)
        
        logger.info(
            f"Repository stats: {result.total_events} events, "
            f"{result.total_symbols} symbols, {result.total_sources} sources"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving calendar stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/calendar/data-availability", response_model=DataAvailabilityResponse)
def data_availability(
    response: Response,
    days: int = Query(30, ge=1, le=365, description="Days ahead to analyze"),
    symbols: Optional[str] = Query(None, description="Comma-separated symbols to check"),
):
    """
    Analyze data availability for upcoming earnings.
    
    Provides analysis of available earnings data including date ranges,
    source coverage, and symbol availability for planning and monitoring.
    
    Args:
        days: Number of days ahead to analyze (1-365)
        symbols: Optional comma-separated symbols to check
        
    Returns:
        Data availability analysis including coverage and recommendations
    """
    try:
        # Validate parameters
        validate_date_range(days, max_days=365)
        syms = parse_symbols(symbols)
        
        # Calculate date range
        start_date = date.today()
        end_date = start_date + timedelta(days=days)
        
        # Query repository
        repo = EarningsCalendarRepository()
        df = repo.get_earnings_by_date_range(
            start_date=start_date,
            end_date=end_date,
            symbols=syms,
        )
        
        if df.empty:
            # No data available
            result = DataAvailabilityResponse(
                total_events=0,
                date_range_start=start_date.isoformat(),
                date_range_end=end_date.isoformat(),
                sources_available=[],
                symbols_available=0,
            )
        else:
            # Analyze available data
            unique_sources = df["source"].unique().tolist() if "source" in df.columns else []
            unique_symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else []
            
            result = DataAvailabilityResponse(
                total_events=len(df),
                date_range_start=start_date.isoformat(),
                date_range_end=end_date.isoformat(),
                sources_available=unique_sources,
                symbols_available=len(unique_symbols),
                last_updated=df["earnings_date"].max() if not df.empty else None,
            )
        
        # Add cache headers
        add_cache_headers(response, max_age_seconds=300)
        
        logger.info(
            f"Data availability for {days} days: {result.total_events} events, "
            f"{result.symbols_available} symbols, {len(result.sources_available)} sources"
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"Parameter validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing data availability: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/calendar/suggest-symbols")
def suggest_symbols(
    response: Response,
    days: int = Query(14, ge=1, le=90, description="Days ahead for suggestions"),
    limit: int = Query(10, ge=1, le=50, description="Number of suggestions to return"),
    min_market_cap: Optional[float] = Query(None, description="Minimum market cap filter"),
):
    """
    Get symbol suggestions for upcoming earnings.
    
    Provides intelligent symbol suggestions based on upcoming earnings,
    market cap, and other criteria. Useful for discovery and planning.
    
    Args:
        days: Days ahead to look for suggestions (1-90)
        limit: Maximum number of suggestions (1-50)
        min_market_cap: Optional minimum market cap filter
        
    Returns:
        List of suggested symbols with metadata
    """
    try:
        # Validate parameters
        validate_date_range(days, max_days=90)
        if limit < 1 or limit > 50:
            raise ValueError(f"limit must be between 1 and 50: {limit}")
        
        # Get upcoming earnings
        start_date = date.today()
        end_date = start_date + timedelta(days=days)
        
        repo = EarningsCalendarRepository()
        df = repo.get_earnings_by_date_range(
            start_date=start_date,
            end_date=end_date,
        )
        
        if df.empty:
            logger.info(f"No earnings data available for suggestions in next {days} days")
            return {"suggestions": [], "total_available": 0}
        
        # Process suggestions
        suggestions = []
        symbols_processed = set()
        
        # Sort by date and prioritize (simplified logic)
        if "earnings_date" in df.columns:
            df = df.sort_values("earnings_date")
        
        for _, row in df.iterrows():
            symbol = row.get("symbol")
            if symbol and symbol not in symbols_processed:
                suggestion = {
                    "symbol": symbol,
                    "company_name": row.get("company_name", symbol),
                    "earnings_date": str(row.get("earnings_date")),
                    "time": row.get("time"),
                    "source": row.get("source"),
                    "has_estimates": bool(
                        row.get("eps_estimate") or row.get("revenue_estimate")
                    ),
                }
                
                suggestions.append(suggestion)
                symbols_processed.add(symbol)
                
                if len(suggestions) >= limit:
                    break
        
        result = {
            "suggestions": suggestions,
            "total_available": len(df["symbol"].unique()) if "symbol" in df.columns else 0,
            "date_range_start": start_date.isoformat(),
            "date_range_end": end_date.isoformat(),
            "criteria": {
                "days_ahead": days,
                "limit": limit,
                "min_market_cap": min_market_cap,
            },
        }
        
        # Add cache headers
        add_cache_headers(response, max_age_seconds=600)
        
        logger.info(
            f"Generated {len(suggestions)} symbol suggestions for {days} days ahead"
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"Parameter validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating symbol suggestions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/calendar/coverage-report")
def coverage_report(
    response: Response,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    sources: Optional[str] = Query(None, description="Comma-separated sources to analyze"),
):
    """
    Generate comprehensive coverage report for earnings data.
    
    Analyzes data coverage across different sources, date ranges, and symbols
    to provide insights into data completeness and quality.
    
    Args:
        start_date: Optional start date for analysis (default: 30 days ago)
        end_date: Optional end date for analysis (default: today)
        sources: Optional comma-separated sources to analyze
        
    Returns:
        Comprehensive coverage report with statistics and insights
    """
    try:
        # Parse date parameters
        if start_date:
            try:
                start = date.fromisoformat(start_date)
            except ValueError:
                raise ValueError(f"Invalid start_date format: {start_date}. Use YYYY-MM-DD")
        else:
            start = date.today() - timedelta(days=30)
            
        if end_date:
            try:
                end = date.fromisoformat(end_date)
            except ValueError:
                raise ValueError(f"Invalid end_date format: {end_date}. Use YYYY-MM-DD")
        else:
            end = date.today()
            
        if start >= end:
            raise ValueError("start_date must be before end_date")
        
        # Parse sources
        source_list = None
        if sources:
            source_list = [s.strip().lower() for s in sources.split(",")]
        
        # Query repository
        repo = EarningsCalendarRepository()
        df = repo.get_earnings_by_date_range(
            start_date=start,
            end_date=end,
            sources=source_list,
        )
        
        if df.empty:
            coverage = {
                "period": {"start": start.isoformat(), "end": end.isoformat()},
                "total_events": 0,
                "sources": {},
                "symbols": {},
                "daily_coverage": {},
                "quality_metrics": {},
            }
        else:
            # Analyze coverage by source
            source_coverage = {}
            if "source" in df.columns:
                source_counts = df["source"].value_counts()
                source_coverage = {source: int(count) for source, count in source_counts.items()}
            
            # Analyze symbol coverage
            symbol_coverage = {}
            if "symbol" in df.columns:
                symbol_counts = df["symbol"].value_counts().head(20)  # Top 20 symbols
                symbol_coverage = {symbol: int(count) for symbol, count in symbol_counts.items()}
            
            # Analyze daily coverage
            daily_coverage = {}
            if "earnings_date" in df.columns:
                daily_counts = df["earnings_date"].value_counts().sort_index()
                daily_coverage = {str(date): int(count) for date, count in daily_counts.items()}
            
            # Calculate quality metrics
            quality_metrics = {}
            if not df.empty:
                total_events = len(df)
                eps_estimate_coverage = df["eps_estimate"].notna().sum() / total_events if "eps_estimate" in df.columns else 0
                revenue_estimate_coverage = df["revenue_estimate"].notna().sum() / total_events if "revenue_estimate" in df.columns else 0
                
                quality_metrics = {
                    "eps_estimate_coverage": float(eps_estimate_coverage),
                    "revenue_estimate_coverage": float(revenue_estimate_coverage),
                    "complete_events": int(df[
                        (df["eps_estimate"].notna()) & (df["revenue_estimate"].notna())
                    ].shape[0]) if all(col in df.columns for col in ["eps_estimate", "revenue_estimate"]) else 0,
                }
            
            coverage = {
                "period": {"start": start.isoformat(), "end": end.isoformat()},
                "total_events": len(df),
                "sources": source_coverage,
                "symbols": symbol_coverage,
                "daily_coverage": daily_coverage,
                "quality_metrics": quality_metrics,
            }
        
        # Add cache headers
        add_cache_headers(response, max_age_seconds=1800)  # 30 minutes
        
        logger.info(
            f"Generated coverage report for {start} to {end}: "
            f"{coverage['total_events']} events across {len(coverage['sources'])} sources"
        )
        
        return coverage
        
    except ValueError as e:
        logger.error(f"Parameter validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating coverage report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")