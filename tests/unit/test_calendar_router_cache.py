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
    from services.calendar_api.routers import core as core_mod
    import pandas as pd

    class FakeCal:
        def get_upcoming_events(self, *, days_ahead: int, symbols=None):
            return [_mk_event("AAPL")]

        def is_fresh(self) -> bool:
            return True

        def get_latest_collection_date_str(self):
            return "2025-01-01"

    class FakeRepo:
        def get_earnings_by_date_range(self, start_date, end_date):
            # Return empty DataFrame for metadata calculation
            return pd.DataFrame()

    class FailFetcher:
        async def get_upcoming(
            self, *, days_ahead: int, symbols=None
        ):  # pragma: no cover
            raise AssertionError("Fetcher should not be called when repo has data")

    monkeypatch.setattr(core_mod, "CalendarService", lambda: FakeCal(), raising=True)
    monkeypatch.setattr(core_mod, "EarningsCalendarRepository", lambda: FakeRepo(), raising=True)
    # EarningsFetcherService is not used in core module, no need to patch

    c = TestClient(app)
    r = c.get(
        "/calendar/upcoming",
        params={"days": 14, "symbols": "AAPL"},
    )
    assert r.status_code == 200
    data = r.json()
    # The /calendar/upcoming endpoint returns EarningsResponseWithMetadata
    assert "events" in data and "metadata" in data
    assert len(data["events"]) == 1 and data["events"][0]["symbol"] == "AAPL"


def test_fetch_upcoming_gets_external_data(monkeypatch):
    from services.calendar_api.app import app
    from services.calendar_api.routers import fetch as fetch_mod
    import pandas as pd

    class FakeCal:
        def save_events(self, events):
            return len(events)  # Return count of saved events

    class FakeRepo:
        def get_earnings_by_date_range(self, start_date, end_date):
            return pd.DataFrame()  # Empty existing data

    class OkFetcher:
        async def get_upcoming(self, *, days_ahead: int, symbols=None, sources=None, include_metadata=False):
            if include_metadata:
                # Create a proper FetchMetadata mock
                from services.calendar_api.service.earnings_fetcher_service import FetchMetadata
                from earnings import EarningsSource
                
                metadata = FetchMetadata(
                    sources_used=[EarningsSource.NASDAQ],
                    symbols_requested=symbols or [],
                    symbols_found=["MSFT"],
                    missing_symbols=[],
                    events_count=1,
                    fetch_duration_seconds=0.1
                )
                return [_mk_event("MSFT")], metadata
            return [_mk_event("MSFT")]

    # Mock TimingContext to provide proper duration_ms
    class FakeTimingContext:
        def __init__(self):
            self.duration_ms = 100.0
        
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(fetch_mod, "CalendarService", lambda: FakeCal(), raising=True)
    monkeypatch.setattr(fetch_mod, "EarningsCalendarRepository", lambda: FakeRepo(), raising=True)
    monkeypatch.setattr(
        fetch_mod, "EarningsFetcherService", lambda: OkFetcher(), raising=True
    )
    monkeypatch.setattr(
        fetch_mod, "TimingContext", FakeTimingContext, raising=True
    )

    c = TestClient(app)
    r = c.post(
        "/calendar/fetch-upcoming", params={"days": 14, "symbols": "MSFT"}
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1 and data[0]["symbol"] == "MSFT"
