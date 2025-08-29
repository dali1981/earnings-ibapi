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


def test_upcoming_uses_cache_when_fresh(monkeypatch):
    from services.calendar_api.app import app
    from services.calendar_api.routers import calendar as cal_mod

    class FakeCal:
        def get_upcoming_events(self, *, days_ahead: int, symbols=None):
            return [_mk_event("AAPL")]

        def is_fresh(self) -> bool:
            return True

        def get_latest_collection_date_str(self):
            return "2025-01-01"

    class FailFetcher:
        async def get_upcoming(
            self, *, days_ahead: int, symbols=None
        ):  # pragma: no cover
            raise AssertionError("Fetcher should not be called when repo has data")

    monkeypatch.setattr(cal_mod, "CalendarService", lambda: FakeCal(), raising=True)
    monkeypatch.setattr(
        cal_mod, "EarningsFetcherService", lambda: FailFetcher(), raising=True
    )

    c = TestClient(app)
    r = c.get(
        "/calendar/upcoming",
        params={"days": 14, "symbols": "AAPL", "refresh": "auto", "cache_hours": 6},
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1 and data[0]["symbol"] == "AAPL"


def test_upcoming_force_refresh_ignores_cache(monkeypatch):
    from services.calendar_api.app import app
    from services.calendar_api.routers import calendar as cal_mod

    class EmptyCal:
        def get_upcoming_events(self, *, days_ahead: int, symbols=None):
            return []

        def is_fresh(self) -> bool:
            return False

    class OkFetcher:
        async def get_upcoming(self, *, days_ahead: int, symbols=None):
            return [_mk_event("MSFT")]

    monkeypatch.setattr(cal_mod, "CalendarService", lambda: EmptyCal(), raising=True)
    monkeypatch.setattr(
        cal_mod, "EarningsFetcherService", lambda: OkFetcher(), raising=True
    )

    c = TestClient(app)
    r = c.get(
        "/calendar/upcoming", params={"days": 14, "symbols": "MSFT", "refresh": "force"}
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1 and data[0]["symbol"] == "MSFT"
