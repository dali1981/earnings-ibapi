from fastapi import FastAPI

from .routers.options import router as options_router


def create_app() -> FastAPI:
    app = FastAPI(title="Option History Service", version="0.2.0")
    app.include_router(options_router)
    return app


app = create_app()

