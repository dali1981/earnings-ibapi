import asyncio
from types import SimpleNamespace

import pytest


pytestmark = [pytest.mark.unit, pytest.mark.calendar]


def _mk_event(symbol: str = "AAPL"):
    return SimpleNamespace(
        symbol=symbol,
        company_name="Apple",
        earnings_date=__import__("datetime").date(2025, 1, 1),
        time="amc",
        eps_estimate=None,
        eps_actual=None,
        revenue_estimate=None,
        revenue_actual=None,
        source="test",
    )


def test_earnings_service_calls_fetcher(monkeypatch):
    from services.calendar_api.service.earnings_service import EarningsService
    import earnings.fetcher as fetch_mod

    calls = {}

    async def fake_async_get_upcoming_earnings(self, *, symbols=None, days_ahead=14, sources=None):
        calls["args"] = {"symbols": symbols, "days_ahead": days_ahead, "sources": sources}
        return [_mk_event("AAPL"), _mk_event("MSFT")]

    monkeypatch.setattr(
        fetch_mod.EarningsCalendarFetcher,
        "async_get_upcoming_earnings",
        fake_async_get_upcoming_earnings,
        raising=True,
    )

    svc = EarningsService()
    events = asyncio.run(svc.get_upcoming(days_ahead=7, symbols=["AAPL", "MSFT"]))
    assert len(events) == 2
    assert calls["args"]["symbols"] == ["AAPL", "MSFT"]
    assert calls["args"]["days_ahead"] == 7
