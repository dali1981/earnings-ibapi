from fastapi.testclient import TestClient

def test_calendar_health():
    from services.calendar_api.app import app as cal
    c = TestClient(cal)
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_equity_health():
    from services.equity_api.app import app as eq
    c = TestClient(eq)
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_options_health():
    from services.option_history_api.app import app as opt
    c = TestClient(opt)
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_chains_health():
    from services.option_chain_api.app import app as ch
    c = TestClient(ch)
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
