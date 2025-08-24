from fastapi import FastAPI

from .routers.equity import router as equity_router


def create_app() -> FastAPI:
    app = FastAPI(title="Equity History Service", version="0.2.0")
    app.include_router(equity_router)
    return app


app = create_app()

