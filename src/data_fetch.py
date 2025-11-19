"""Data download and caching utilities."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - yfinance is expected to be installed
    raise RuntimeError("yfinance is required for price downloads") from exc


try:  # pragma: no cover - optional dependency
    import simfin as sf
except ImportError:  # pragma: no cover - optional dependency
    sf = None


PRICE_DIR = Path("data/prices")
FUNDAMENTALS_DIR = Path("data/fundamentals")


def _is_cache_valid(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.utcnow() - modified < timedelta(days=max_age_days)


def download_price_history(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
    interval: str = "1d",
    cache_days: int = 1,
) -> Dict[str, pd.DataFrame]:
    """Download price history for tickers using yfinance with caching."""
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    results: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        ticker = ticker.upper()
        file_path = PRICE_DIR / f"{ticker}.csv"
        if _is_cache_valid(file_path, cache_days):
            logging.info("Using cached prices for %s", ticker)
            results[ticker] = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
            continue

        logging.info("Downloading prices for %s", ticker)
        try:
            data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        except Exception as err:  # pragma: no cover - network issues
            logging.error("Failed downloading %s: %s", ticker, err)
            continue
        if data.empty:
            logging.warning("No price data for %s", ticker)
            continue
        data.index.name = "Date"
        data.to_csv(file_path)
        results[ticker] = data

    return results


def download_fundamentals_simfin(
    tickers: Iterable[str],
    market: str = "us",
    cache_days: int = 30,
) -> Dict[str, pd.DataFrame]:
    """Download fundamentals from SimFin if the package and API key are available."""
    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    api_key = os.getenv("SIMFIN_API_KEY")
    fundamentals: Dict[str, pd.DataFrame] = {}

    if sf is None or not api_key:
        logging.info("SimFin not configured. Set SIMFIN_API_KEY to enable fundamentals download.")
        return fundamentals

    sf.set_api_key(api_key)
    sf.set_data_dir(str(FUNDAMENTALS_DIR))

    for dataset in ("derived", "ratios"):
        try:
            sf.load(dataset=dataset, variant="quarterly", market=market)
        except Exception as err:  # pragma: no cover - network issues
            logging.error("Unable to load SimFin dataset %s: %s", dataset, err)
            return fundamentals

    derived = sf.load(dataset="derived", variant="quarterly", market=market)
    ratios = sf.load(dataset="ratios", variant="quarterly", market=market)

    for ticker in tickers:
        ticker = ticker.upper()
        derived_rows = derived[derived["Ticker"] == ticker]
        ratio_rows = ratios[ratios["Ticker"] == ticker]
        if derived_rows.empty and ratio_rows.empty:
            logging.warning("Ticker %s not found in SimFin datasets", ticker)
            continue
        df = pd.merge(derived_rows, ratio_rows, how="outer")
        fundamentals[ticker] = df.sort_values("Report Date")
        cache_path = FUNDAMENTALS_DIR / f"{ticker}_simfin.csv"
        df.to_csv(cache_path, index=False)

    return fundamentals
