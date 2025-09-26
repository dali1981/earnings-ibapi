from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
import sys


@pytest.mark.unit
def test_show_upcoming_earnings_reads_local_db_and_prints(
    tmp_path, monkeypatch, capsys
):
    # Arrange repository structure under a temp base
    base = tmp_path
    earnings_dir = base / "earnings"
    today = date.today()
    part = earnings_dir / f"collection_date={today.isoformat()}" / "source=nasdaq"
    part.mkdir(parents=True, exist_ok=True)

    # Create minimal earnings parquet with upcoming dates within next 2 days
    rows = [
        {
            "symbol": "TEST",
            "company_name": "Test Corp",
            "earnings_date": (today + timedelta(days=1)).isoformat(),
            "time": "amc",
            "eps_estimate": None,
            "eps_actual": None,
            "revenue_estimate": None,
            "revenue_actual": None,
            "source": "nasdaq",
            "collection_date": today.isoformat(),
            "days_until_earnings": 1,
            "priority_score": 80.0,
        }
    ]
    df = pd.DataFrame(rows)
    (part / "earnings.parquet").write_bytes(
        b""
    )  # ensure file exists even if empty write fails
    df.to_parquet(part / "earnings.parquet")

    # Monkeypatch config to point DATA_ROOT to tmp base
    import config as cfg

    monkeypatch.setattr(cfg, "DATA_ROOT", base, raising=True)

    # Import the script module and run main with --days 2
    sys.path.append(str(Path(__file__).parents[2]))  # repo root
    from scripts import show_upcoming_earnings as cli

    monkeypatch.setattr(
        sys, "argv", ["show_upcoming_earnings.py", "--days", "1"], raising=True
    )

    # Monkeypatch repository method to avoid ParquetDataset filter quirks in tests
    from common.repositories.earnings import EarningsCalendarRepository

    def _load_range(self, start_date, end_date, symbols=None, sources=None):
        dfp = pd.read_parquet(part / "earnings.parquet")
        dfp["earnings_date"] = pd.to_datetime(dfp["earnings_date"]).dt.date
        return dfp

    monkeypatch.setattr(
        EarningsCalendarRepository,
        "get_earnings_by_date_range",
        _load_range,
        raising=True,
    )

    # Act
    cli.main()

    # Assert
    out = capsys.readouterr().out
    assert "Upcoming earnings" in out
    assert "TEST" in out
    assert str(today + timedelta(days=1)) in out
