#!/usr/bin/env python3
"""Download daily OHLCV prices from Yahoo Finance and save as parquet."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------
# Paths (ALWAYS relative to project root)
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_OUTPUT_PATH = BASE_DIR / "00_data" / "raw" / "prices.parquet"
REPORT_PATH = BASE_DIR / "05_reports" / "etl_summary.json"

START_DATE = pd.Timestamp("2015-01-01")


def get_universe() -> list[str]:
    """Return the fixed universe of large-cap tickers."""
    return [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
        "JPM", "BAC", "WFC", "GS",
        "JNJ", "PFE", "MRK",
        "XOM", "CVX",
        "KO", "PEP",
        "WMT", "COST",
        "HD", "LOW",
        "DIS", "NFLX",
        "INTC", "AMD",
    ]


def _normalize_ticker_df(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    if "adj_close" not in df.columns:
        return pd.DataFrame()

    df = df.reset_index().rename(columns={"Date": "date"})
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["adj_close"])

    return df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]


def download_one_ticker(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data = yf.download(
        tickers=ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    return _normalize_ticker_df(ticker, data)


def _download_bulk(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    data = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        progress=False,
        auto_adjust=False,
        threads=True,
    )

    results: dict[str, pd.DataFrame] = {}
    if data.empty:
        return results

    for ticker in tickers:
        if ticker in data.columns.get_level_values(0):
            results[ticker] = _normalize_ticker_df(ticker, data[ticker])

    return results


def _ensure_directories() -> None:
    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    start_date = START_DATE
    end_date = pd.Timestamp(date.today().isoformat())
    tickers = get_universe()

    logger.info("Starting download for %d tickers", len(tickers))
    _ensure_directories()

    bulk_data = _download_bulk(tickers, start_date, end_date)

    downloaded = {}
    failed = []

    for ticker in tickers:
        df = bulk_data.get(ticker, pd.DataFrame())
        if df.empty:
            logger.info("Retrying %s individually", ticker)
            df = download_one_ticker(ticker, start_date, end_date)

        if df.empty:
            logger.warning("No data for %s", ticker)
            failed.append(ticker)
            continue

        downloaded[ticker] = df

    if len(downloaded) < 20:
        raise ValueError("Downloaded data for fewer than 20 tickers")

    combined = (
        pd.concat(downloaded.values(), ignore_index=True)
        .drop_duplicates(subset=["date", "ticker"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    if (combined[["open", "high", "low", "close", "adj_close"]] < 0).any().any():
        raise ValueError("Negative prices detected")

    combined.to_parquet(RAW_OUTPUT_PATH, index=False)

    expected_dates = pd.date_range(start=start_date, end=end_date, freq="B")
    per_ticker = {}

    for t, df in downloaded.items():
        actual = pd.DatetimeIndex(df["date"])
        missing = len(expected_dates.difference(actual))
        per_ticker[t] = {
            "first_date": actual.min().date().isoformat(),
            "last_date": actual.max().date().isoformat(),
            "missing_days_count": missing,
            "pct_missing": missing / len(expected_dates),
        }

    report = {
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "tickers_requested": tickers,
        "tickers_downloaded": sorted(downloaded.keys()),
        "rows_total": int(len(combined)),
        "per_ticker": per_ticker,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))

    logger.info("Saved parquet to %s", RAW_OUTPUT_PATH)
    logger.info("Tickers OK / failed: %d / %d", len(downloaded), len(failed))


if __name__ == "__main__":
    main()
