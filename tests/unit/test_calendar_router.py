from fastapi.testclient import TestClient
from types import SimpleNamespace

import pytest


pytestmark = [pytest.mark.unit, pytest.mark.calendar]


def _mk_event(symbol: str = "AAPL"):
    from datetime import date

    return SimpleNamespace(
        symbol=symbol,
        company_name="Apple",
        earnings_date=date(2025, 1, 1),
        time="amc",
        eps_estimate=None,
        eps_actual=None,
        revenue_estimate=None,
        revenue_actual=None,
        source="test",
    )


def test_calendar_upcoming_route(monkeypatch):
    # Monkeypatch EarningsService used by the router to avoid network
    from services.calendar_api.app import app
    from services.calendar_api.routers import calendar as cal_module

    class FakeSvc:
        async def get_upcoming(self, *, days_ahead: int, symbols=None):
            assert days_ahead == 14
            assert symbols == ["AAPL", "MSFT"]
            return [_mk_event("AAPL"), _mk_event("MSFT")]

    # Service renamed to EarningsFetcherService
    monkeypatch.setattr(
        cal_module, "EarningsFetcherService", lambda: FakeSvc(), raising=True
    )

    c = TestClient(app)
    r = c.get("/calendar/upcoming", params={"days": 14, "symbols": "AAPL,MSFT"})
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, list) and len(payload) == 2
    assert payload[0]["symbol"] == "AAPL"
    assert payload[1]["symbol"] == "MSFT"
