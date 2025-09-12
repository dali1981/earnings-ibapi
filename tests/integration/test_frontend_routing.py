from fastapi.testclient import TestClient

import pytest


pytestmark = pytest.mark.integration


def test_frontend_injects_custom_bases(monkeypatch):
    # Provide custom bases via settings env to verify they are injected
    monkeypatch.setenv("APP_API__SERVICES__CALENDAR__HOST", "svc-calendar")
    monkeypatch.setenv("APP_API__SERVICES__CALENDAR__PORT", "1234")
    monkeypatch.setenv("APP_API__SERVICES__EQUITY__HOST", "svc-equity")
    monkeypatch.setenv("APP_API__SERVICES__EQUITY__PORT", "2345")
    monkeypatch.setenv("APP_API__SERVICES__CHAINS__HOST", "svc-chains")
    monkeypatch.setenv("APP_API__SERVICES__CHAINS__PORT", "3456")

    from services.frontend.main import app

    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200
    html = r.text
    assert "http://svc-calendar:1234" in html
    assert "http://svc-equity:2345" in html
    assert "http://svc-chains:3456" in html
    # Verify the page wires modules and actions (migrated away from inline fetch templates)
    assert "window.__BASES__" in html
    assert 'data-action="calendar.get"' in html
    assert 'data-action="calendar.fetch"' in html
    assert 'data-action="equity.fetch"' in html
    assert 'data-action="chains.latest"' in html
    assert "/static/js/apiClient.js" in html
    assert "/static/js/controllers.js" in html
    assert "/static/js/bootstrap.js" in html


def test_frontend_buttons_have_actions():
    from services.frontend.main import app

    c = TestClient(app)
    html = c.get("/").text
    # Ensure forms/buttons declare actions which bootstrap binds to controllers
    assert 'data-action="calendar.get"' in html
    assert 'data-action="calendar.fetch"' in html
    assert 'data-action="equity.fetch"' in html
    assert 'data-action="chains.latest"' in html
