import pandas as pd
from datetime import date
from ibx_repos.equity_bars import EquityBarRepository
from ibx_repos.option_bars import OptionBarRepository, OptionMeta


def test_equity_present_dates(tmp_path):
    repo = EquityBarRepository(tmp_path)
    df = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [10, 20],
        }
    )
    repo.save(df, symbol="AAPL", bar_size="1 day")
    present = repo.present_dates(
        "AAPL", "1 day", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-03")
    )
    assert present == {date(2023, 1, 1), date(2023, 1, 2)}


def test_option_present_dates(tmp_path):
    repo = OptionBarRepository(tmp_path)
    meta = OptionMeta(
        underlying="AAPL",
        expiry=date(2023, 1, 20),
        strike=100.0,
        right="C",
        bar_size="1 day",
    )
    df = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [10, 20],
        }
    )
    repo.save(df, meta)
    present = repo.present_dates_for_contract(
        "AAPL",
        date(2023, 1, 20),
        "C",
        100.0,
        "1 day",
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-01-03"),
    )
    assert present == {date(2023, 1, 1), date(2023, 1, 2)}
