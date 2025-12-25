import pandas as pd

from src.data.download_prices import validate_prices_df


def test_prices_schema():
    df = pd.DataFrame(
        {
            "date": ["2020-01-01"],
            "ticker": ["AAPL"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "adj_close": [100.4],
            "volume": [1000.0],
        }
    )
    ok, errors = validate_prices_df(df)
    assert ok, f"Validation failed: {errors}"


def test_prices_no_duplicate_date_ticker():
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "ticker": ["AAPL", "AAPL"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "adj_close": [100.4, 101.4],
            "volume": [1000.0, 1100.0],
        }
    )
    ok, errors = validate_prices_df(df)
    assert ok, f"Validation failed: {errors}"
