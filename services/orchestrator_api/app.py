from fastapi import FastAPI
import logging
import os
from fastapi.middleware.cors import CORSMiddleware

from .routers.orchestrator import router as orchestrator_router
from .routers.orchestrator import router_public as orchestrator_public_router
from common.utils.logging_setup import setup_logging, disable_noisy_loggers


def create_app() -> FastAPI:
    setup_logging(level=os.environ.get("APP_LOG_LEVEL", "INFO"))
    disable_noisy_loggers()
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    
    logger = logging.getLogger(__name__)
    
    app = FastAPI(title="Orchestrator Service", version="0.1.0")
    
    # Add startup event to log registered routes
    @app.on_event("startup")
    async def startup_event():
        logger.info("=== Orchestrator Service Starting ===")
        logger.info("Registered routes:")
        for route in app.routes:
            if hasattr(route, "methods") and hasattr(route, "path"):
                logger.info(f"  {route.methods} -> {route.path}")
    
    # Prefixed and public routes for convenience
    app.include_router(orchestrator_router)
    app.include_router(orchestrator_public_router)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add a debug endpoint to list all routes
    @app.get("/debug/routes")
    def list_routes():
        routes = []
        for route in app.routes:
            if hasattr(route, "methods") and hasattr(route, "path"):
                routes.append({
                    "path": route.path,
                    "methods": list(route.methods),
                    "name": route.name if hasattr(route, "name") else None
                })
        return {"routes": routes}
    
    return app


app = create_app()
