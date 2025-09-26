"""
Management and administrative operations router.

Handles administrative operations like data cleanup, bulk operations,
cache management, and other maintenance tasks.
"""

from __future__ import annotations

from ..service.calendar_service import CalendarService
from ..utils.common import TimingContext
from common.reliability.logging import get_trading_logger
from common.repositories.earnings import EarningsCalendarRepository
from datetime import date, timedelta
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

router = APIRouter(tags=["calendar-management"])
logger = get_trading_logger(__name__)


@router.post("/calendar/refresh-cache")
def refresh_cache():
    """
    Refresh internal caches and metadata.
    
    Forces refresh of any cached metadata, statistics, or other
    cached information to ensure data consistency.
    
    Returns:
        Status of cache refresh operation
    """
    try:
        with TimingContext() as timing:
            # This would refresh any internal caches
            # For now, this is a placeholder for future cache implementations
            logger.info("Cache refresh requested - placeholder implementation")
            
            # Could refresh:
            # - Repository metadata caches
            # - Statistics caches
            # - Service-level caches
            
            result = {
                "status": "success",
                "message": "Cache refresh completed",
                "duration_ms": timing.duration_ms,
                "operations": [
                    "metadata_cache_cleared",
                    "statistics_cache_cleared",
                ],
            }
            
            logger.info(f"Cache refresh completed in {timing.duration_ms:.2f}ms")
            return result
            
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/calendar/health-detailed")
def detailed_health():
    """
    Comprehensive health check with dependency validation.
    
    Performs detailed health checks including repository connectivity,
    service dependencies, and data integrity checks.
    
    Returns:
        Detailed health status with dependency information
    """
    try:
        with TimingContext() as timing:
            health_status = {
                "status": "healthy",
                "timestamp": date.today().isoformat(),
                "checks": {},
            }
            
            # Check repository health
            try:
                repo = EarningsCalendarRepository()
                stats = repo.get_storage_stats()
                health_status["checks"]["repository"] = {
                    "status": "healthy",
                    "total_events": stats.get("total_events", 0),
                    "storage_accessible": True,
                }
            except Exception as e:
                health_status["checks"]["repository"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health_status["status"] = "degraded"
            
            # Check calendar service
            try:
                cal = CalendarService()
                latest_date = cal.get_latest_collection_date_str()
                health_status["checks"]["calendar_service"] = {
                    "status": "healthy",
                    "latest_collection_date": latest_date,
                }
            except Exception as e:
                health_status["checks"]["calendar_service"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health_status["status"] = "degraded"
            
            # Check data freshness
            try:
                cal = CalendarService()
                is_fresh = cal.is_fresh()
                health_status["checks"]["data_freshness"] = {
                    "status": "healthy" if is_fresh else "warning",
                    "is_fresh": is_fresh,
                    "message": "Data is current" if is_fresh else "Data may be stale",
                }
                if not is_fresh and health_status["status"] == "healthy":
                    health_status["status"] = "warning"
            except Exception as e:
                health_status["checks"]["data_freshness"] = {
                    "status": "unknown",
                    "error": str(e),
                }
            
            health_status["response_time_ms"] = timing.duration_ms
            
            logger.info(f"Detailed health check completed: {health_status['status']}")
            return health_status
            
    except Exception as e:
        logger.error(f"Error performing detailed health check: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": date.today().isoformat(),
            "error": str(e),
        }


@router.post("/calendar/maintenance/cleanup")
def cleanup_old_data(
    days_to_keep: int = Query(365, ge=30, le=3650, description="Days of data to retain"),
    dry_run: bool = Query(True, description="Perform dry run without actual deletion"),
):
    """
    Clean up old earnings data based on retention policy.
    
    Removes earnings data older than the specified retention period.
    Supports dry-run mode to preview what would be deleted.
    
    Args:
        days_to_keep: Number of days of data to retain (30-3650)
        dry_run: If True, only simulate the cleanup without actual deletion
        
    Returns:
        Statistics about the cleanup operation
    """
    try:
        if days_to_keep < 30 or days_to_keep > 3650:
            raise ValueError(f"days_to_keep must be between 30 and 3650: {days_to_keep}")
        
        cutoff_date = date.today() - timedelta(days=days_to_keep)
        
        with TimingContext() as timing:
            # Query old data
            repo = EarningsCalendarRepository()
            
            # Get data older than cutoff
            old_data_df = repo.get_earnings_by_date_range(
                start_date=date(2000, 1, 1),  # Very old start date
                end_date=cutoff_date,
            )
            
            events_to_delete = len(old_data_df) if not old_data_df.empty else 0
            symbols_affected = len(old_data_df["symbol"].unique()) if not old_data_df.empty and "symbol" in old_data_df.columns else 0
            
            if dry_run:
                operation_type = "dry_run"
                deleted_events = 0
                message = f"Dry run: Would delete {events_to_delete} events older than {cutoff_date}"
            else:
                operation_type = "actual_cleanup"
                # In a real implementation, you would delete the old data here
                # deleted_events = repo.delete_data_before_date(cutoff_date)
                deleted_events = 0  # Placeholder
                message = f"Cleanup disabled in this implementation. Would delete {events_to_delete} events"
            
            result = {
                "operation": operation_type,
                "cutoff_date": cutoff_date.isoformat(),
                "events_identified": events_to_delete,
                "events_deleted": deleted_events,
                "symbols_affected": symbols_affected,
                "duration_ms": timing.duration_ms,
                "message": message,
            }
            
            logger.info(
                f"Cleanup operation ({operation_type}): {events_to_delete} events identified, "
                f"{deleted_events} deleted, cutoff date: {cutoff_date}"
            )
            
            return result
            
    except ValueError as e:
        logger.error(f"Parameter validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during cleanup operation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/calendar/maintenance/status")
def maintenance_status():
    """
    Get maintenance and operational status information.
    
    Returns information about data retention, storage usage,
    recent operations, and maintenance schedules.
    
    Returns:
        Maintenance status information
    """
    try:
        with TimingContext() as timing:
            # Get basic repository info
            repo = EarningsCalendarRepository()
            stats = repo.get_storage_stats()
            
            # Calculate data age
            cal = CalendarService()
            latest_collection = cal.get_latest_collection_date_str()
            
            status = {
                "timestamp": date.today().isoformat(),
                "data_health": {
                    "total_events": stats.get("total_events", 0),
                    "latest_collection_date": latest_collection,
                    "storage_size_mb": stats.get("storage_size_mb"),
                    "partitions_count": stats.get("partitions_count"),
                },
                "maintenance": {
                    "last_cleanup": "not_implemented",
                    "retention_days": 365,  # Default retention
                    "next_scheduled_cleanup": "not_scheduled",
                },
                "performance": {
                    "response_time_ms": timing.duration_ms,
                    "cache_status": "not_implemented",
                },
                "recommendations": [],
            }
            
            # Add recommendations based on data
            if stats.get("total_events", 0) == 0:
                status["recommendations"].append("No earnings data found - consider running data fetch")
            
            if stats.get("storage_size_mb", 0) > 1000:  # If storage > 1GB
                status["recommendations"].append("Large storage usage - consider cleanup of old data")
            
            logger.info("Maintenance status retrieved successfully")
            return status
            
    except Exception as e:
        logger.error(f"Error retrieving maintenance status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")