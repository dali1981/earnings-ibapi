from fastapi import FastAPI

from .routers.chains import router as chains_router


def create_app() -> FastAPI:
    app = FastAPI(title="Option Chain Service", version="0.2.0")
    app.include_router(chains_router)
    return app


app = create_app()

