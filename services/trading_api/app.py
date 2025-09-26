"""
Trading API FastAPI application.
"""

from .routers import orders, positions, strategies, risk
from .schemas.trading import HealthResponse, ServiceStatusResponse
from common.observability import init_observability
from common.utils.logging_setup import disable_noisy_loggers, setup_logging
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import time

import logging
import os


# Application startup time for uptime calculation
_startup_time = time.time()

app = FastAPI(
    title="Trading Engine API",
    version="1.0.0",
    description="Comprehensive trading API with order management, positions, strategies, and risk controls"
)

# Include all routers
app.include_router(orders.router)
app.include_router(positions.router)
app.include_router(strategies.router)
app.include_router(risk.router)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        service="trading-api",
        version="1.0.0",
        timestamp=datetime.utcnow(),
        dependencies={
            "order_manager": "ok",
            "position_manager": "ok", 
            "strategy_engine": "ok",
            "risk_manager": "ok"
        }
    )


@app.get("/status", response_model=ServiceStatusResponse)
async def get_service_status():
    """Get detailed service status."""
    uptime = time.time() - _startup_time
    
    # In production, these would be actual counts from managers
    return ServiceStatusResponse(
        service_name="trading-api",
        version="1.0.0", 
        uptime_seconds=uptime,
        connected_services=["ib-api", "market-data", "database"],
        active_orders=0,  # Would be populated from OrderManager
        active_positions=0,  # Would be populated from PositionManager
        enabled_strategies=0,  # Would be populated from StrategyManager
        last_signal_time=None
    )


@app.on_event("startup")
async def configure_observability_and_logging():
    """Configure observability and logging on startup."""
    # Initialize observability first
    init_observability(service_name="trading-api", service_version="1.0.0")
    
    # Then configure logging
    setup_logging(level=os.environ.get("APP_LOG_LEVEL", "INFO"))
    disable_noisy_loggers()
    
    # Configure FastAPI/Uvicorn loggers to use our setup
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    
    logger = logging.getLogger(__name__)
    logger.info("Trading API service started successfully")


@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown."""
    logger = logging.getLogger(__name__)
    logger.info("Trading API service shutting down")