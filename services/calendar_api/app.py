from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.calendar import router as calendar_router


def create_app() -> FastAPI:
    app = FastAPI(title="Calendar Service", version="0.2.0")
    app.include_router(calendar_router)
    # Enable CORS for local UI/testing (adjust origins as needed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()
