from fastapi.testclient import TestClient

import pytest


pytestmark = pytest.mark.unit


def test_frontend_index_renders_defaults(monkeypatch):
    # Ensure defaults are used when no env vars present
    for k in [
        "FRONTEND_CALENDAR_BASE",
        "FRONTEND_EQUITY_BASE",
        "FRONTEND_CHAINS_BASE",
    ]:
        monkeypatch.delenv(k, raising=False)

    from services.frontend.main import app

    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200
    html = r.text
    # Check that base URLs default to localhost ports
    assert "http://localhost:9001" in html
    assert "http://localhost:9002" in html
    assert "http://localhost:9004" in html
    # Check that key forms and modules exist (migrated to data-action wiring)
    assert 'id="calendarGetForm"' in html
    assert 'id="calendarFetchForm"' in html
    assert 'id="equityForm"' in html
    assert 'id="chainForm"' in html
    assert '/static/js/apiClient.js' in html
    assert '/static/js/controllers.js' in html
    assert '/static/js/bootstrap.js' in html


def test_static_css_served():
    from services.frontend.main import app

    c = TestClient(app)
    r = c.get("/static/styles.css")
    assert r.status_code == 200
    assert "body" in r.text
