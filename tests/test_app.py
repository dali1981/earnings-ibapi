# tests/test_app.py
import pytest
from datetime import date
import pandas as pd
from ibx_flows.windows import missing_windows


def test_missing_windows():
    """Unit test for the missing_windows function."""
    present_dates = {date(2023, 1, 3), date(2023, 1, 4)}
    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 6)

    gaps = missing_windows(present_dates, start_date, end_date)

    assert gaps == [(date(2023, 1, 2), date(2023, 1, 2)), (date(2023, 1, 5), date(2023, 1, 6))]

# Add more unit and integration tests here...