#!/usr/bin/env python3
"""Build equity factor features with strict no-lookahead."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def _resolve_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_prices(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading prices from %s", path)
    df = pd.read_parquet(path)

    required = {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    date_series = df["date"]
    if pd.api.types.is_integer_dtype(date_series):
        df["date"] = pd.to_datetime(date_series, unit="ns", utc=True).dt.tz_convert(None)
    else:
        df["date"] = pd.to_datetime(date_series, utc=True, errors="coerce").dt.tz_convert(None)

    if df["date"].isna().any():
        raise ValueError("Found invalid dates after conversion")

    df["date"] = df["date"].dt.normalize()

    return df


def _validate_prices(df: pd.DataFrame) -> None:
    if df.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("Duplicate (date, ticker) rows found")

    monotonic = (
        df.groupby("ticker", sort=False)["date"]
        .apply(lambda s: s.is_monotonic_increasing)
        .all()
    )
    if not monotonic:
        raise ValueError("Dates are not strictly increasing per ticker")


def _build_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"], kind="mergesort").copy()
    grouped = df.groupby("ticker", sort=False)

    df["ret_1d"] = grouped["adj_close"].pct_change()
    df["ret_fwd_5d"] = grouped["adj_close"].shift(-5) / df["adj_close"] - 1.0
    df["mom_20d"] = grouped["adj_close"].pct_change(periods=20)
    df["mom_60d"] = grouped["adj_close"].pct_change(periods=60)
    df["vol_20d"] = grouped["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)

    rolling_mean_20 = grouped["adj_close"].rolling(20).mean().reset_index(level=0, drop=True)
    rolling_std_20 = grouped["adj_close"].rolling(20).std().reset_index(level=0, drop=True)
    df["zscore_20d_price"] = (df["adj_close"] - rolling_mean_20) / rolling_std_20

    df["adv_20d"] = grouped["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    df["dollar_vol_20d"] = df["adv_20d"] * df["adj_close"]

    factor_cols = [
        "ret_1d",
        "ret_fwd_5d",
        "mom_20d",
        "mom_60d",
        "vol_20d",
        "zscore_20d_price",
        "adv_20d",
        "dollar_vol_20d",
    ]

    df = df.dropna(subset=factor_cols).copy()

    return df[["date", "ticker", *factor_cols]]


def _validate_factors(df: pd.DataFrame) -> None:
    if (df["vol_20d"] < 0).any():
        raise ValueError("vol_20d contains negative values")

    if not np.isfinite(df["zscore_20d_price"]).all():
        raise ValueError("zscore_20d_price contains non-finite values")

    ticker_count = df["ticker"].nunique()
    if ticker_count < 10:
        raise ValueError("Fewer than 10 tickers remain after warm-up drop")


def _missingness(df: pd.DataFrame) -> dict[str, float]:
    return (df.isna().mean() * 100).round(4).to_dict()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = _resolve_root()
    prices_path = root / "00_data" / "raw" / "prices.parquet"
    output_path = root / "00_data" / "features" / "factors.parquet"
    summary_path = root / "05_reports" / "factors_summary.json"

    df_prices = _load_prices(prices_path)
    _validate_prices(df_prices)

    rows_in = len(df_prices)
    tickers_in = df_prices["ticker"].nunique()

    factors = _build_factors(df_prices)
    _validate_factors(factors)

    rows_out = len(factors)
    tickers_out = factors["ticker"].nunique()

    min_date = factors["date"].min()
    max_date = factors["date"].max()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Writing factors to %s", output_path)
    factors.to_parquet(output_path, index=False)

    summary = {
        "rows_in": int(rows_in),
        "rows_out": int(rows_out),
        "tickers_in": int(tickers_in),
        "tickers_out": int(tickers_out),
        "min_date": None if pd.isna(min_date) else min_date.strftime("%Y-%m-%d"),
        "max_date": None if pd.isna(max_date) else max_date.strftime("%Y-%m-%d"),
        "missingness_pct": _missingness(factors),
    }

    LOGGER.info("Writing summary to %s", summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
