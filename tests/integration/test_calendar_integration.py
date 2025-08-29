from fastapi.testclient import TestClient
from types import SimpleNamespace

import asyncio
import pytest


pytestmark = [pytest.mark.integration, pytest.mark.calendar]


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


def test_calendar_integration_monkeypatch_fetcher(monkeypatch):
    # Patch the underlying fetcher coroutine used by EarningsService
    import earnings.fetcher as fetch_mod

    async def fake_async_get_upcoming_earnings(
        self, *, symbols=None, days_ahead=14, sources=None
    ):
        return [_mk_event("AAPL"), _mk_event("MSFT")]

    monkeypatch.setattr(
        fetch_mod.EarningsCalendarFetcher,
        "async_get_upcoming_earnings",
        fake_async_get_upcoming_earnings,
        raising=True,
    )

    from services.calendar_api.app import app

    c = TestClient(app)
    r = c.get("/calendar/upcoming", params={"days": 7, "symbols": "AAPL,MSFT"})
    assert r.status_code == 200
    data = r.json()
    assert [e["symbol"] for e in data] == ["AAPL", "MSFT"]
