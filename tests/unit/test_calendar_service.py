from types import SimpleNamespace

import asyncio
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


def test_earnings_fetcher_service_calls_fetcher(monkeypatch):
    from services.calendar_api.service.earnings_fetcher_service import EarningsFetcherService

    import earnings.async_fetcher as fetch_mod

    calls = {}

    async def fake_a_get_upcoming_earnings(
        self, *, symbols=None, days_ahead=7, sources=None
    ):
        calls["args"] = {
            "symbols": symbols,
            "days_ahead": days_ahead,
            "sources": sources,
        }
        return [_mk_event("AAPL"), _mk_event("MSFT")]

    monkeypatch.setattr(
        fetch_mod.AsyncEarningsCalendarFetcher,
        "a_get_upcoming_earnings",
        fake_a_get_upcoming_earnings,
        raising=True,
    )

    svc = EarningsFetcherService()
    events = asyncio.run(svc.get_upcoming(days_ahead=7, symbols=["AAPL", "MSFT"]))
    assert len(events) == 2
    assert calls["args"]["symbols"] == ["AAPL", "MSFT"]
    assert calls["args"]["days_ahead"] == 7
