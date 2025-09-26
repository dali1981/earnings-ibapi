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


def test_calendar_integration_fetch_upcoming(monkeypatch):
    # Patch the underlying async fetcher used by EarningsFetcherService
    import earnings.async_fetcher as fetch_mod
    import pandas as pd

    async def fake_a_get_upcoming_earnings(
        self, *, symbols=None, days_ahead=7, sources=None
    ):
        return [_mk_event("AAPL"), _mk_event("MSFT")]

    monkeypatch.setattr(
        fetch_mod.AsyncEarningsCalendarFetcher,
        "a_get_upcoming_earnings",
        fake_a_get_upcoming_earnings,
        raising=True,
    )

    # Mock repository to avoid file system access
    from services.calendar_api.routers import calendar as cal_mod

    class FakeRepo:
        def get_earnings_by_date_range(self, start_date, end_date, symbols=None, sources=None):
            return pd.DataFrame()  # Return empty to simulate no existing data

    class FakeCalService:
        def save_events(self, events):
            return len(events)

    monkeypatch.setattr(cal_mod, "EarningsCalendarRepository", lambda: FakeRepo(), raising=True)
    monkeypatch.setattr(cal_mod, "CalendarService", lambda: FakeCalService(), raising=True)

    from services.calendar_api.app import app

    c = TestClient(app)
    r = c.post("/calendar/fetch-upcoming", params={"days": 7, "symbols": "AAPL,MSFT"})
    assert r.status_code == 200
    data = r.json()
    assert [e["symbol"] for e in data] == ["AAPL", "MSFT"]
