from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.orchestrator import router as orchestrator_router
from .routers.orchestrator import router_public as orchestrator_public_router


def create_app() -> FastAPI:
    app = FastAPI(title="Orchestrator Service", version="0.1.0")
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
    return app


app = create_app()
