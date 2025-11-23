"""Backtest engine for the multi-factor strategy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .multi_factor_config import BacktestConfig
from .multi_factor_data import Universe
from .multi_factor_factors import build_factor_snapshot
from .multi_factor_portfolio import PortfolioSelection, construct_portfolio
from .multi_factor_selection import (
    apply_anti_value_trap_filters,
    attach_layer_info,
    compute_layers_of_maxima,
)


@dataclass
class BacktestResult:
    portfolio_history: pd.Series
    daily_returns: pd.Series
    holdings: Dict[pd.Timestamp, PortfolioSelection]
    turnover: pd.Series
    benchmark: pd.Series | None = None


def _compute_rebalance_dates(price_index: pd.DatetimeIndex, months: int) -> List[pd.Timestamp]:
    schedule = pd.date_range(price_index.min(), price_index.max(), freq=pd.DateOffset(months=months))
    rebal_dates: List[pd.Timestamp] = []
    for dt in schedule:
        candidates = price_index[price_index <= dt]
        if len(candidates) == 0:
            continue
        rebal_dates.append(candidates[-1])
    if not rebal_dates or rebal_dates[-1] != price_index[-1]:
        rebal_dates.append(price_index[-1])
    return rebal_dates


def _portfolio_daily_returns(weights: pd.Series, prices: pd.DataFrame) -> pd.Series:
    pct = prices.pct_change().fillna(0)
    aligned = pct[weights.index].fillna(0)
    return (aligned * weights).sum(axis=1)


def run_backtest(
    config: BacktestConfig,
    universe: Universe,
    fundamentals: pd.DataFrame,
    adj_close: pd.DataFrame,
    benchmark: pd.Series | None = None,
) -> BacktestResult:
    """Execute the backtest and return portfolio performance series."""
    rebalance_dates = _compute_rebalance_dates(adj_close.index, config.rebalance_months)
    capital = 1_000_000.0
    daily_returns: List[pd.Series] = []
    holdings: Dict[pd.Timestamp, PortfolioSelection] = {}
    turnover_list: List[float] = []

    prev_weights = pd.Series(dtype=float)

    for i, date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]
        snapshot = build_factor_snapshot(fundamentals, adj_close, date)
        scores = snapshot.block_scores.dropna()
        eps = {"Quality": config.eps_q, "Value": config.eps_v, "Momentum": config.eps_m}
        layers = compute_layers_of_maxima(scores[["Quality", "Value", "Momentum"]], eps)
        scores_with_layer = attach_layer_info(scores, layers)

        candidates = scores_with_layer.loc[scores_with_layer["layer"] <= 2].copy()
        surviving = apply_anti_value_trap_filters(
            candidates,
            snapshot.indicators[["mom_12_1", "mom_6_1"]],
            snapshot.indicators[["gross_prof_assets", "roic", "debt_to_equity"]],
        )
        candidates = candidates.loc[candidates.index.isin(surviving)]

        ranked = candidates.sort_values(["layer", "Value", "Quality", "Momentum"], ascending=[True, False, False, False])
        selection = construct_portfolio(
            ranked,
            universe.metadata,
            max_positions=config.max_positions,
            sector_max_weight=config.sector_max_weight,
        )
        holdings[date] = selection

        period_prices = adj_close.loc[date:next_date]
        period_returns = _portfolio_daily_returns(selection.weights, period_prices)
        if i > 0:
            # Avoid double-counting the rebalance date when concatenating periods
            period_returns = period_returns.iloc[1:]

        weight_change = selection.weights - prev_weights.reindex(selection.weights.index).fillna(0)
        turnover = 0.5 * weight_change.abs().sum()
        turnover_list.append(turnover)
        cost = turnover * (config.transaction_cost_bps / 10000)

        period_returns.iloc[0] -= cost
        daily_returns.append(period_returns)
        prev_weights = selection.weights

    daily_returns_series = pd.concat(daily_returns).sort_index()
    portfolio_history = (1 + daily_returns_series).cumprod() * capital

    turnover_series = pd.Series(turnover_list, index=rebalance_dates[:-1])
    return BacktestResult(portfolio_history, daily_returns_series, holdings, turnover_series, benchmark)


__all__ = ["run_backtest", "BacktestResult"]
