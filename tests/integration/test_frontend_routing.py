from fastapi.testclient import TestClient

import pytest


pytestmark = pytest.mark.integration


def test_frontend_injects_custom_bases(monkeypatch):
    # Provide custom bases to verify they are injected into the page and used by JS builder
    monkeypatch.setenv("FRONTEND_CALENDAR_BASE", "http://svc-calendar:1234")
    monkeypatch.setenv("FRONTEND_EQUITY_BASE", "http://svc-equity:2345")
    monkeypatch.setenv("FRONTEND_CHAINS_BASE", "http://svc-chains:3456")

    from services.frontend.main import app

    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200
    html = r.text
    assert "http://svc-calendar:1234" in html
    assert "http://svc-equity:2345" in html
    assert "http://svc-chains:3456" in html
    # Verify the fetch templates are constructed from BASES
    assert "`${BASES.calendar}/calendar/upcoming" in html
    assert "`${BASES.equity}/equity/" in html
    assert "`${BASES.chains}/chains/" in html


def test_frontend_buttons_have_handlers():
    from services.frontend.main import app

    c = TestClient(app)
    html = c.get("/").text
    # Check that the script registers event listeners for forms
    assert "document.getElementById('calendarForm').addEventListener('submit'" in html
    assert "document.getElementById('equityForm').addEventListener('submit'" in html
    assert "document.getElementById('chainForm').addEventListener('submit'" in html
