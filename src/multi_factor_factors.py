"""Factor construction for Value, Quality, and Momentum blocks.

The functions in this module compute indicator-level z-scores and aggregate them
into block scores. They assume the caller provides price data (adjusted close)
and fundamentals aligned to dates no later than the rebalance date to avoid
look-ahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass
class FactorSnapshot:
    date: pd.Timestamp
    indicators: pd.DataFrame  # index=ticker, columns raw indicators
    zscores: pd.DataFrame  # index=ticker, columns standardized indicators
    block_scores: pd.DataFrame  # columns [Value, Quality, Momentum]


def _winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lower_val = series.quantile(lower)
    upper_val = series.quantile(upper)
    return series.clip(lower=lower_val, upper=upper_val)


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def standardize_indicators(indicators: pd.DataFrame, winsor_limits: tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
    """Cross-sectional winsorization and z-score standardization."""
    standardized = {}
    for col in indicators.columns:
        col_data = indicators[col].dropna()
        if col_data.empty:
            standardized[col] = pd.Series(index=indicators.index, dtype=float)
            continue
        w = _winsorize(col_data, winsor_limits[0], winsor_limits[1])
        z = _zscore(w)
        standardized[col] = z.reindex(indicators.index)
    return pd.DataFrame(standardized)


def aggregate_blocks(zscores: pd.DataFrame, block_map: Dict[str, Iterable[str]]) -> pd.DataFrame:
    """Aggregate indicator z-scores into block scores by simple average."""
    block_scores = {}
    for block, cols in block_map.items():
        cols_list = [c for c in cols if c in zscores.columns]
        block_scores[block] = zscores[cols_list].mean(axis=1)
    return pd.DataFrame(block_scores)


def compute_momentum_indicators(adj_close: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.DataFrame:
    """Compute momentum indicators using data up to the rebalance date."""
    hist = adj_close.loc[:rebalance_date]
    monthly = hist.resample("M").last()

    def total_return(window: int, skip_months: int = 1) -> pd.Series:
        if len(monthly) <= window:
            return pd.Series(index=adj_close.columns, dtype=float)
        past = monthly.iloc[-(window + skip_months)]
        end = monthly.iloc[-(skip_months)]
        return end / past - 1

    m12_1 = total_return(12, skip_months=1)
    m6_1 = total_return(6, skip_months=1)

    if len(monthly) > 12:
        twelve_month_return = monthly.iloc[-1] / monthly.iloc[-13] - 1
    else:
        twelve_month_return = pd.Series(index=adj_close.columns, dtype=float)

    m12_vol = hist.pct_change().rolling(252).std().iloc[-1] if len(hist) >= 252 else pd.Series(index=adj_close.columns, dtype=float)
    risk_adj = twelve_month_return / m12_vol.replace(0, np.nan)

    indicators = pd.DataFrame({
        "mom_12_1": m12_1,
        "mom_6_1": m6_1,
        "mom_risk_adj": risk_adj,
    })
    return indicators


def compute_value_quality_indicators(
    fundamentals: pd.DataFrame,
    prices: pd.Series,
    rebalance_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute Value and Quality indicators using the latest fundamentals up to the rebalance date."""
    upto = fundamentals[fundamentals["date"] <= rebalance_date]
    latest = (
        upto.sort_values(["ticker", "date"])
        .groupby("ticker")
        .tail(1)
        .set_index("ticker")
    )

    if latest.empty:
        return pd.DataFrame(index=prices.index)

    if "market_cap" not in latest.columns:
        if "shares_outstanding" in latest.columns:
            latest["market_cap"] = latest["shares_outstanding"] * prices
        else:
            raise ValueError("market_cap or shares_outstanding must be provided in fundamentals")

    ev = latest["market_cap"] + latest.get("total_debt", 0) - latest.get("cash_and_equiv", 0)
    latest["earnings_yield"] = latest.get("ebit") / ev.replace(0, np.nan)
    latest["book_to_market"] = latest.get("book_equity") / latest["market_cap"].replace(0, np.nan)
    latest["cash_flow_yield"] = latest.get("free_cash_flow") / ev.replace(0, np.nan)

    latest["gross_prof_assets"] = latest.get("gross_profit") / latest.get("total_assets")
    latest["roic"] = latest.get("roic", latest.get("roa"))
    latest["debt_to_equity"] = latest.get("debt_to_equity", np.nan)
    latest["neg_leverage"] = -latest["debt_to_equity"]
    if "interest_coverage" not in latest.columns:
        latest["interest_coverage"] = np.nan

    indicators = latest[[
        "earnings_yield",
        "book_to_market",
        "cash_flow_yield",
        "gross_prof_assets",
        "roic",
        "neg_leverage",
        "debt_to_equity",
        "interest_coverage",
    ]].copy()
    indicators.index.name = "ticker"
    return indicators


def build_factor_snapshot(
    fundamentals: pd.DataFrame,
    adj_close: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    winsor_limits: tuple[float, float] = (0.01, 0.99),
) -> FactorSnapshot:
    """Create standardized factor snapshot at a given rebalance date."""
    momentum_ind = compute_momentum_indicators(adj_close, rebalance_date)
    if rebalance_date not in adj_close.index:
        raise ValueError(f"Rebalance date {rebalance_date} not found in price data index")
    price_on_date = adj_close.loc[rebalance_date].dropna()
    value_quality_ind = compute_value_quality_indicators(fundamentals, price_on_date, rebalance_date)

    common_tickers = momentum_ind.index.intersection(value_quality_ind.index)
    indicators = pd.concat([value_quality_ind.loc[common_tickers], momentum_ind.loc[common_tickers]], axis=1)

    zscores = standardize_indicators(indicators, winsor_limits)
    block_map = {
        "Value": ["earnings_yield", "book_to_market", "cash_flow_yield"],
        "Quality": ["gross_prof_assets", "roic", "neg_leverage"],
        "Momentum": ["mom_12_1", "mom_6_1", "mom_risk_adj"],
    }
    block_scores = aggregate_blocks(zscores, block_map)
    return FactorSnapshot(rebalance_date, indicators, zscores, block_scores)


__all__ = [
    "FactorSnapshot",
    "build_factor_snapshot",
    "compute_momentum_indicators",
    "compute_value_quality_indicators",
    "standardize_indicators",
    "aggregate_blocks",
]
