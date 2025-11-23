"""Reporting utilities for performance statistics and optional plots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class PerformanceStats:
    cagr: float
    vol: float
    sharpe: float
    max_drawdown: float
    turnover: float


def compute_performance(returns: pd.Series, turnover: pd.Series, risk_free_rate: float = 0.0) -> PerformanceStats:
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = returns - daily_rf
    cagr = (1 + returns).prod() ** (252 / len(returns)) - 1 if len(returns) > 0 else np.nan
    vol = excess.std() * np.sqrt(252)
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() != 0 else np.nan
    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum / roll_max - 1).min()
    avg_turnover = turnover.mean() if not turnover.empty else 0.0
    return PerformanceStats(cagr, vol, sharpe, drawdown, avg_turnover)


def plot_equity_curve(portfolio: pd.Series, benchmark: pd.Series | None = None) -> None:
    plt.figure(figsize=(10, 5))
    (portfolio / portfolio.iloc[0]).plot(label="Strategy")
    if benchmark is not None and not benchmark.empty:
        (benchmark / benchmark.iloc[0]).plot(label="Benchmark")
    plt.legend()
    plt.title("Equity Curve")
    plt.ylabel("Cumulative Return")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()


def plot_drawdown(returns: pd.Series) -> None:
    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    dd = cum / roll_max - 1
    dd.plot(figsize=(10, 4), title="Drawdown")
    plt.tight_layout()
    plt.show()


__all__ = ["PerformanceStats", "compute_performance", "plot_equity_curve", "plot_drawdown"]
