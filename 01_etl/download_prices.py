#!/usr/bin/env python3
"""Download daily OHLCV prices from Yahoo Finance and save as parquet."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf


RAW_OUTPUT_PATH = Path("00_data/raw/prices.parquet")
REPORT_PATH = Path("05_reports/etl_summary.json")
START_DATE = pd.Timestamp("2015-01-01")


def get_universe() -> list[str]:
    """Return the fixed universe of large-cap tickers."""
    return [
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "META",
        "NVDA",
        "TSLA",
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "JNJ",
        "PFE",
        "MRK",
        "XOM",
        "CVX",
        "KO",
        "PEP",
        "WMT",
        "COST",
        "HD",
        "LOW",
        "DIS",
        "NFLX",
        "INTC",
        "AMD",
    ]


def _normalize_ticker_df(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
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

    price_columns = ["open", "high", "low", "close", "adj_close"]
    for column in price_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.dropna(subset=["adj_close"])

    return df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]


def download_one_ticker(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download a single ticker and return long-format data."""
    data = yf.download(
        tickers=ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return _normalize_ticker_df(ticker, data)


def _download_bulk(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    data = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    results: dict[str, pd.DataFrame] = {}
    if data.empty:
        return results

    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker in data.columns.get_level_values(0):
                results[ticker] = _normalize_ticker_df(ticker, data[ticker])
    else:
        if len(tickers) == 1:
            results[tickers[0]] = _normalize_ticker_df(tickers[0], data)

    return results


def _ensure_directories() -> None:
    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    end_date = pd.Timestamp(date.today().isoformat())
    start_date = START_DATE
    tickers = get_universe()

    logger.info("Starting download for %s tickers", len(tickers))
    _ensure_directories()

    bulk_data = _download_bulk(tickers, start_date, end_date)

    downloaded: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for ticker in tickers:
        df = bulk_data.get(ticker, pd.DataFrame())
        if df.empty:
            logger.info("Retrying download for %s", ticker)
            df = download_one_ticker(ticker, start_date, end_date)

        if df.empty:
            logger.warning("No data for %s", ticker)
            failed.append(ticker)
            continue

        downloaded[ticker] = df

    if len(downloaded) < 20:
        raise ValueError("Downloaded data for fewer than 20 tickers.")

    combined = pd.concat(downloaded.values(), ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "ticker"])

    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    combined["ticker"] = combined["ticker"].astype("string")
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()

    price_columns = ["open", "high", "low", "close", "adj_close"]
    for column in price_columns:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")

    combined["volume"] = pd.to_numeric(combined["volume"], errors="coerce")

    if (combined[price_columns] < 0).any().any():
        raise ValueError("Negative prices detected.")

    min_date = combined["date"].min()
    max_date = combined["date"].max()

    if min_date < start_date or max_date > end_date:
        raise ValueError("Dates out of requested range.")

    combined.to_parquet(RAW_OUTPUT_PATH, index=False)

    expected_dates = pd.date_range(start=start_date, end=end_date, freq="B")
    per_ticker_summary = {}
    for ticker, df in downloaded.items():
        dates = df["date"].dropna().unique()
        actual_dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
        missing_days = len(expected_dates.difference(actual_dates))
        pct_missing = missing_days / len(expected_dates) if len(expected_dates) else 0.0
        per_ticker_summary[ticker] = {
            "first_date": pd.Timestamp(df["date"].min()).date().isoformat(),
            "last_date": pd.Timestamp(df["date"].max()).date().isoformat(),
            "missing_days_count": missing_days,
            "pct_missing": pct_missing,
        }

    report = {
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "tickers_requested": tickers,
        "tickers_downloaded": sorted(downloaded.keys()),
        "rows_total": int(len(combined)),
        "per_ticker": per_ticker_summary,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))

    logger.info("Saved data to %s", RAW_OUTPUT_PATH)
    logger.info("Tickers ok: %s; failed: %s", len(downloaded), len(failed))
    logger.info("Date range: %s to %s", min_date.date().isoformat(), max_date.date().isoformat())

    print(f"Saved parquet: {RAW_OUTPUT_PATH}")
    print(f"Tickers ok/failed: {len(downloaded)}/{len(failed)}")
    print(f"Min/Max date: {min_date.date().isoformat()} / {max_date.date().isoformat()}")


if __name__ == "__main__":
    main()
