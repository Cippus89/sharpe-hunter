"""Portfolio construction utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


def enforce_sector_constraints(
    candidates: pd.DataFrame,
    metadata: pd.DataFrame,
    max_weight: float,
    max_positions: int,
) -> pd.DataFrame:
    """Greedy selection subject to sector caps."""
    sectors = metadata.set_index("Ticker")["Sector"] if not metadata.empty else pd.Series(dtype=str)
    holdings: List[str] = []
    sector_weights: Dict[str, float] = {}

    for ticker, row in candidates.iterrows():
        sector = sectors.get(ticker, None)
        if len(holdings) >= max_positions:
            break
        if sector:
            if sector_weights.get(sector, 0) + (1 / max_positions) > max_weight + 1e-9:
                continue
            sector_weights[sector] = sector_weights.get(sector, 0) + (1 / max_positions)
        holdings.append(ticker)
    return candidates.loc[holdings]


def equal_weight(candidates: pd.DataFrame) -> pd.Series:
    weights = pd.Series(1 / len(candidates), index=candidates.index)
    return weights


@dataclass
class PortfolioSelection:
    tickers: pd.Index
    weights: pd.Series
    layer_info: pd.Series


def construct_portfolio(
    ranked_candidates: pd.DataFrame,
    metadata: pd.DataFrame,
    max_positions: int,
    sector_max_weight: float,
) -> PortfolioSelection:
    constrained = enforce_sector_constraints(ranked_candidates, metadata, sector_max_weight, max_positions)
    weights = equal_weight(constrained)
    return PortfolioSelection(constrained.index, weights, constrained["layer"])


__all__ = ["PortfolioSelection", "construct_portfolio", "enforce_sector_constraints", "equal_weight"]
