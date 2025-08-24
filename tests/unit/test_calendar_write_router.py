import pytest
from datetime import date
from fastapi.testclient import TestClient


pytestmark = [pytest.mark.unit, pytest.mark.calendar]


def test_add_events_writes_and_dedup(monkeypatch):
    from services.calendar_api.app import app
    from services.calendar_api.routers import calendar as cal_mod

    written_batches = {}

    class FakeRepo:
        def get_earnings_by_date_range(self, s, e, symbols=None, sources=None):
            # Pretend the first event exists (duplicate), second is new
            if symbols and symbols[0] == "AAPL":
                import pandas as pd

                return pd.DataFrame({"symbol": ["AAPL"], "earnings_date": [s]})
            import pandas as pd

            return pd.DataFrame()

        def store_earnings_batch(self, events):
            written_batches["count"] = len(events)
            return {"total_events_stored": len(events)}

    # Patch repository used for duplicate check in the router
    monkeypatch.setattr(cal_mod, "EarningsCalendarRepository", lambda: FakeRepo(), raising=True)

    # Patch CalendarService to ensure writes go through our fake and are observable
    class FakeCalendarService:
        def save_events(self, events):
            written_batches["count"] = len(events)
            return len(events)

    monkeypatch.setattr(cal_mod, "CalendarService", lambda: FakeCalendarService(), raising=True)

    c = TestClient(app)
    payload = [
        {
            "symbol": "AAPL",
            "company_name": "Apple",
            "earnings_date": date(2025, 1, 1).isoformat(),
            "time": "amc",
            "source": "nasdaq",
        },
        {
            "symbol": "MSFT",
            "company_name": "Microsoft",
            "earnings_date": date(2025, 1, 2).isoformat(),
            "time": "amc",
            "source": "nasdaq",
        },
    ]
    r = c.post("/calendar/events", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["duplicates"] == 1
    assert data["written"] == 1
    assert written_batches["count"] == 1
