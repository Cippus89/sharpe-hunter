"""Simple backtesting utilities."""
from __future__ import annotations

import logging
from statistics import mean, pstdev
from typing import Dict

import pandas as pd


def _future_return(series: pd.Series, holding_period: int) -> float | None:
    if series.empty:
        return None
    if holding_period <= 0:
        return None
    sorted_series = series.sort_index()
    if len(sorted_series) <= holding_period:
        return None
    start_price = sorted_series.iloc[-holding_period - 1]
    end_price = sorted_series.iloc[-1]
    if start_price == 0:
        return None
    return end_price / start_price - 1


def run_mispricing_backtest(
    results_df: pd.DataFrame,
    price_history: Dict[str, pd.DataFrame],
    holding_period_days: int = 126,
    top_quantile: float = 0.2,
) -> dict:
    """Run a naive long-short backtest using mispricing scores."""
    if results_df.empty:
        return {}

    scores = results_df[["Ticker", "mispricing_pct"]].dropna()
    long_threshold = scores["mispricing_pct"].quantile(1 - top_quantile)
    short_threshold = scores["mispricing_pct"].quantile(top_quantile)

    longs = scores[scores["mispricing_pct"] >= long_threshold]["Ticker"].tolist()
    shorts = scores[scores["mispricing_pct"] <= short_threshold]["Ticker"].tolist()

    long_returns = []
    for ticker in longs:
        df = price_history.get(ticker)
        if df is None or "Adj Close" not in df.columns:
            continue
        ret = _future_return(df["Adj Close"], holding_period_days)
        if ret is not None:
            long_returns.append(ret)

    short_returns = []
    for ticker in shorts:
        df = price_history.get(ticker)
        if df is None or "Adj Close" not in df.columns:
            continue
        ret = _future_return(df["Adj Close"], holding_period_days)
        if ret is not None:
            short_returns.append(-ret)

    if not long_returns and not short_returns:
        logging.warning("Backtest skipped because of insufficient price history.")
        return {}

    portfolio_returns = long_returns + short_returns
    total_return = mean(portfolio_returns) if portfolio_returns else 0.0
    volatility = pstdev(portfolio_returns) if len(portfolio_returns) > 1 else 0.0
    sharpe = total_return / volatility if volatility else 0.0

    return {
        "long_count": len(long_returns),
        "short_count": len(short_returns),
        "mean_return": total_return,
        "volatility": volatility,
        "sharpe": sharpe,
    }
