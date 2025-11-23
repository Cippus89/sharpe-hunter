"""Data ingestion utilities for the multi-factor framework.

Assumptions about fundamentals CSVs:
- Must include columns: ``date`` (YYYY-MM-DD), ``ticker``.
- Optional but recommended metric columns (names can be adapted by user):
    * ebit, market_cap, total_debt, cash_and_equiv, book_equity, free_cash_flow
    * gross_profit, total_assets, roic, roa, debt_to_equity, interest_coverage
    * shares_outstanding (if market_cap not given) and any pre-computed ratios.
- Additional columns (e.g., sector, industry) are preserved and used in
  portfolio construction if available.

This module keeps data loading flexible: fundamentals can be supplied by the
user as pre-computed ratios; price data is downloaded via ``yfinance`` and
cached locally to avoid repeated API calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import yfinance as yf


@dataclass
class Universe:
    tickers: list[str]
    metadata: pd.DataFrame  # columns: ticker, sector, industry (optional)


def load_universe(tickers: Iterable[str] | None, universe_path: Path | None) -> Universe:
    """Load the investable universe from a CSV or explicit list."""
    if universe_path:
        df = pd.read_csv(universe_path)
        if "ticker" not in df.columns:
            raise ValueError("Universe CSV must contain a 'ticker' column")
        df = df.rename(columns={"ticker": "Ticker", "sector": "Sector", "industry": "Industry"})
        tickers_list = df["Ticker"].dropna().astype(str).str.upper().unique().tolist()
    else:
        if not tickers:
            raise ValueError("No tickers provided and no universe file supplied")
        tickers_list = [t.upper() for t in tickers]
        df = pd.DataFrame({"Ticker": tickers_list})
    df["Sector"] = df.get("Sector")
    df["Industry"] = df.get("Industry")
    return Universe(tickers_list, df)


def download_price_history(tickers: list[str], start: date, end: date, cache_dir: Path) -> pd.DataFrame:
    """Download daily OHLCV price data for all tickers and return tidy DataFrame."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"prices_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    price_df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False, group_by="ticker")
    if price_df.empty:
        raise ValueError("No price data downloaded; check tickers or dates")

    tidy_frames = []
    for ticker in tickers:
        if ticker not in price_df.columns.get_level_values(0):
            continue
        ticker_df = price_df[ticker].copy()
        ticker_df.columns = [c.lower() for c in ticker_df.columns]
        ticker_df["ticker"] = ticker
        ticker_df["date"] = ticker_df.index
        tidy_frames.append(ticker_df.reset_index(drop=True))

    tidy = pd.concat(tidy_frames, ignore_index=True)
    tidy = tidy.rename(columns={"adj close": "adj_close"})
    tidy.to_parquet(cache_path)
    return tidy


def download_benchmark(ticker: str, start: date, end: date, cache_dir: Path) -> pd.Series:
    """Download benchmark adjusted close series."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"benchmark_{ticker}_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path, columns=["adj_close"], index_col="date")["adj_close"]

    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    ser = data["Adj Close"].rename("adj_close")
    ser.to_frame().to_parquet(cache_path)
    ser.index.name = "date"
    return ser


def load_fundamentals(paths: Iterable[Path]) -> pd.DataFrame:
    """Load and concatenate fundamentals/ratio CSV files."""
    frames = []
    for path in paths:
        df = pd.read_csv(path, parse_dates=["date"])
        if "ticker" not in df.columns:
            raise ValueError(f"Fundamentals file {path} missing 'ticker' column")
        frames.append(df)
    if not frames:
        raise ValueError("No fundamentals data provided")
    fundamentals = pd.concat(frames, ignore_index=True)
    fundamentals["ticker"] = fundamentals["ticker"].str.upper()
    fundamentals = fundamentals.drop_duplicates(subset=["date", "ticker"]).sort_values(["ticker", "date"])
    return fundamentals


def pivot_prices(price_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return wide pivoted daily close and adj-close DataFrames (index=date, columns=tickers)."""
    close = price_df.pivot(index="date", columns="ticker", values="close").sort_index()
    adj_close = price_df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return close, adj_close


__all__ = [
    "Universe",
    "load_universe",
    "download_price_history",
    "download_benchmark",
    "load_fundamentals",
    "pivot_prices",
]
