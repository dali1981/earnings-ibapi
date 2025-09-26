import os
import pytest
import socket
import threading
import time


playwright = pytest.importorskip(
    "playwright.sync_api", reason="Playwright not installed; skip E2E"
)


pytestmark = [pytest.mark.frontend_e2e]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_uvicorn(app, port: int):
    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    server.run()


def _start_app_thread(app, port: int) -> threading.Thread:
    th = threading.Thread(target=_run_uvicorn, args=(app, port), daemon=True)
    th.start()
    # crude wait for server to be up
    time.sleep(0.5)
    return th


def test_frontend_buttons_route_requests():
    # Spin up mock backends for calendar/equity/chains
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    cal = FastAPI()
    eq = FastAPI()
    ch = FastAPI()

    # Allow cross-origin fetches from the frontend port during the test
    for app in (cal, eq, ch):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @cal.get("/calendar/upcoming")
    def cal_upcoming(days: int, symbols: str | None = None):
        return [
            {
                "symbol": "AAPL",
                "company_name": "Apple",
                "earnings_date": "2025-01-01",
                "source": "mock",
            }
        ]

    @eq.get("/equity/{symbol}/bars")
    def eq_bars(symbol: str, start: str, end: str, bar_size: str = "1 day"):
        return {
            "symbol": symbol,
            "bar_size": bar_size,
            "start": start,
            "end": end,
            "rows": 0,
            "sample": [],
        }

    @ch.get("/chains/{symbol}/latest")
    def ch_latest(symbol: str):
        return {"symbol": symbol, "as_of": "2025-01-01"}

    cal_port = _free_port()
    eq_port = _free_port()
    ch_port = _free_port()

    _start_app_thread(cal, cal_port)
    _start_app_thread(eq, eq_port)
    _start_app_thread(ch, ch_port)

    # Configure frontend to point to mocks via app settings env
    os.environ["APP_API__SERVICES__CALENDAR__HOST"] = "127.0.0.1"
    os.environ["APP_API__SERVICES__CALENDAR__PORT"] = str(cal_port)
    os.environ["APP_API__SERVICES__EQUITY__HOST"] = "127.0.0.1"
    os.environ["APP_API__SERVICES__EQUITY__PORT"] = str(eq_port)
    os.environ["APP_API__SERVICES__CHAINS__HOST"] = "127.0.0.1"
    os.environ["APP_API__SERVICES__CHAINS__PORT"] = str(ch_port)

    # Start frontend
    from services.frontend.main import app as frontend

    fe_port = _free_port()
    _start_app_thread(frontend, fe_port)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:  # Requires browsers installed
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{fe_port}")

        # Calendar: submit form and expect JSON to render
        page.fill("#daysGet", "7")
        page.fill("#symbolsGet", "AAPL")
        page.click("#calendarGetForm button[type=submit]")
        page.wait_for_function(
            "document.querySelector('#calendarOut').textContent.includes('AAPL')"
        )

        # Equity: set fields and submit
        page.fill("#eqSymbol", "AAPL")
        page.fill("#eqStart", "2024-01-01")
        page.fill("#eqEnd", "2024-01-31")
        page.click("#equityForm button[type=submit]")
        page.wait_for_function(
            "document.querySelector('#equityOut').textContent.length > 0"
        )

        # Chains: submit
        page.fill("#chSymbol", "AAPL")
        page.click("#chainForm button[type=submit]")
        page.wait_for_function(
            "document.querySelector('#chainOut').textContent.includes('as_of')"
        )

        browser.close()
