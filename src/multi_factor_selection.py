"""Selection logic including Layers of Maxima with soft dominance and anti-trap filters."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def soft_dominates(a: pd.Series, b: pd.Series, eps: Dict[str, float]) -> bool:
    """Return True if vector ``a`` softly dominates vector ``b`` under eps thresholds."""
    factors = eps.keys()
    not_worse = all(a[f] >= b[f] - eps[f] for f in factors)
    strictly_better = any(a[f] >= b[f] + eps[f] for f in factors)
    return bool(not_worse and strictly_better)


def compute_layers_of_maxima(scores: pd.DataFrame, eps: Dict[str, float]) -> List[list[str]]:
    """Compute layers of maxima using soft dominance in Q,V,M space."""
    remaining = scores.copy()
    layers: List[list[str]] = []
    while not remaining.empty:
        layer = []
        tickers = remaining.index.tolist()
        for i, t_i in enumerate(tickers):
            dominated = False
            for j, t_j in enumerate(tickers):
                if i == j:
                    continue
                if soft_dominates(remaining.loc[t_j], remaining.loc[t_i], eps):
                    dominated = True
                    break
            if not dominated:
                layer.append(t_i)
        layers.append(layer)
        remaining = remaining.drop(index=layer)
    return layers


def apply_anti_value_trap_filters(
    snapshot: pd.DataFrame,
    momentum: pd.DataFrame,
    quality: pd.DataFrame,
    min_gross_profit_percentile: float = 0.3,
    max_debt_to_equity: float = 2.0,
) -> pd.Index:
    """Apply default anti-value-trap filters and return surviving tickers."""
    gross_profit_metric = quality["gross_prof_assets"]
    threshold = gross_profit_metric.quantile(min_gross_profit_percentile)

    conds = (
        (quality.get("roic", pd.Series(dtype=float)) > 0)
        & (gross_profit_metric >= threshold)
        & (quality.get("debt_to_equity", pd.Series(dtype=float)) < max_debt_to_equity)
        & (momentum.get("mom_12_1", pd.Series(dtype=float)) > momentum["mom_12_1"].quantile(0.3))
        & (momentum.get("mom_6_1", pd.Series(dtype=float)) >= 0)
    )
    return snapshot.index[conds.fillna(False)]


def rank_candidates(
    scores: pd.DataFrame,
    layers: List[list[str]],
    max_positions: int,
    weights: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rank securities from L1 and L2 using a composite score."""
    weights = weights or {"Value": 0.4, "Quality": 0.3, "Momentum": 0.3}
    selected_layers = [t for layer in layers[:2] for t in layer]
    subset = scores.loc[selected_layers].copy()
    subset["composite"] = (
        weights["Value"] * subset["Value"]
        + weights["Quality"] * subset["Quality"]
        + weights["Momentum"] * subset["Momentum"]
    )
    subset = subset.sort_values(["layer", "composite"], ascending=[True, False])
    return subset.head(max_positions)


def attach_layer_info(scores: pd.DataFrame, layers: List[list[str]]) -> pd.DataFrame:
    """Add a 'layer' column to scores based on computed layers."""
    layer_map = {}
    for i, layer in enumerate(layers, start=1):
        for ticker in layer:
            layer_map[ticker] = i
    scores = scores.copy()
    scores["layer"] = scores.index.map(layer_map)
    return scores


__all__ = [
    "soft_dominates",
    "compute_layers_of_maxima",
    "apply_anti_value_trap_filters",
    "rank_candidates",
    "attach_layer_info",
]
