import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import yfinance as yf

TICKERS = [
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

OUTPUT_PATH = Path("00_data/raw/prices.parquet")


@dataclass
class DownloadResult:
    data: pd.DataFrame
    succeeded: List[str]
    failed: List[str]


def _download_ticker_group(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    ticker_list = list(tickers)
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(
                "Downloading %d tickers (attempt %d/%d)",
                len(ticker_list),
                attempt,
                max_retries,
            )
            data = yf.download(
                tickers=" ".join(ticker_list),
                start=start_date,
                end=end_date,
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            if data.empty:
                raise ValueError("No data returned from yfinance.")
            return data
        except Exception as exc:  # noqa: BLE001 - log and retry
            logging.warning("Download attempt %d failed: %s", attempt, exc)
            if attempt == max_retries:
                raise
    return pd.DataFrame()


def _normalize_ticker_frame(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    frame["ticker"] = ticker
    frame = frame.reset_index().rename(columns={"Date": "date"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    ordered = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    frame = frame[ordered]
    frame = frame.dropna(how="all", subset=["open", "high", "low", "close", "adj_close", "volume"])
    frame["volume"] = frame["volume"].fillna(0)
    price_cols = ["open", "high", "low", "close", "adj_close"]
    frame[price_cols] = (
        frame.sort_values("date")[price_cols]
        .groupby(frame["ticker"], group_keys=False)
        .ffill(limit=5)
    )
    return frame


def _flatten_download(data: pd.DataFrame, tickers: Iterable[str]) -> DownloadResult:
    succeeded: List[str] = []
    failed: List[str] = []
    frames: List[pd.DataFrame] = []

    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in data.columns.levels[0]:
                failed.append(ticker)
                continue
            frame = data[ticker].dropna(how="all")
            if frame.empty:
                failed.append(ticker)
                continue
            normalized = _normalize_ticker_frame(ticker, frame)
            if normalized.empty:
                failed.append(ticker)
                continue
            frames.append(normalized)
            succeeded.append(ticker)
    else:
        if data.empty:
            failed.extend(list(tickers))
        else:
            ticker = list(tickers)[0]
            normalized = _normalize_ticker_frame(ticker, data)
            if normalized.empty:
                failed.append(ticker)
            else:
                frames.append(normalized)
                succeeded.append(ticker)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return DownloadResult(data=combined, succeeded=succeeded, failed=failed)


def validate_prices_df(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    expected = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    if list(df.columns) != expected:
        errors.append("Dataframe columns do not match required schema.")
    if df.duplicated(subset=["date", "ticker"]).any():
        errors.append("Duplicate (date, ticker) rows detected.")
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        errors.append("Invalid date values detected.")
    else:
        if (parsed_dates.dt.normalize() != parsed_dates).any():
            errors.append("Date values include time components.")
    return len(errors) == 0, errors


def download_prices(
    tickers: Iterable[str] = TICKERS,
    output_path: Path = OUTPUT_PATH,
) -> DownloadResult:
    start_date = date(2015, 1, 1)
    end_date = date.today() + timedelta(days=1)
    data = _download_ticker_group(
        tickers=tickers,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )
    result = _flatten_download(data, tickers)
    if result.data.empty:
        logging.error("No data downloaded successfully.")
        return result

    result.data = result.data.sort_values(["ticker", "date"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_parquet(output_path, index=False)

    ok, errors = validate_prices_df(result.data)
    if not ok:
        logging.warning("Validation issues: %s", "; ".join(errors))

    if result.failed:
        logging.warning("Failed tickers: %s", ", ".join(result.failed))
    logging.info(
        "Downloaded %d tickers, saved %d rows to %s",
        len(result.succeeded),
        len(result.data),
        output_path,
    )
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        download_prices()
    except Exception as exc:  # noqa: BLE001 - top-level logging
        logging.error("Download failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
