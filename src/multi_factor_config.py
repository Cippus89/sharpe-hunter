"""Configuration utilities for the multi-factor portfolio backtest.

This module defines a typed configuration object with sensible defaults. Users
can either instantiate :class:`BacktestConfig` programmatically or load settings
from a simple YAML/JSON file. Only free/open-source packages are used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Optional
import json
import yaml


@dataclass
class BacktestConfig:
    """Container for strategy and backtest parameters.

    Attributes
    ----------
    start_date : date
        First date (inclusive) for pulling price data and running the backtest.
    end_date : date
        Last date (inclusive) for pulling price data and running the backtest.
    tickers : list[str]
        Universe of tickers. Can be loaded from a CSV via ``universe_path`` if
        not provided programmatically.
    universe_path : Optional[Path]
        Path to a CSV with at least a ``ticker`` column and optionally ``sector``
        and ``industry`` columns. If provided, overrides ``tickers`` when
        loading the universe.
    fundamentals_path : Iterable[Path]
        One or more CSV files containing fundamentals/ratios. Each file must
        include a ``date`` column and a ``ticker`` column, along with the
        metrics used in the factor definitions (see documentation in
        ``multi_factor_factors.py``).
    eps_q : float
        Soft-dominance epsilon for the Quality dimension.
    eps_v : float
        Soft-dominance epsilon for the Value dimension.
    eps_m : float
        Soft-dominance epsilon for the Momentum dimension.
    rebalance_months : int
        Number of months between rebalances (default 6 for semi-annual).
    max_positions : int
        Maximum number of positions in the portfolio.
    transaction_cost_bps : float
        Round-trip transaction cost in basis points applied on turnover.
    sector_max_weight : float
        Maximum portfolio weight allowed per sector (0-1 scale). Default 0.15.
    risk_free_rate : float
        Annualized risk-free rate used for Sharpe calculations.
    cache_dir : Path
        Directory where downloaded price data will be cached.
    benchmark : str
        Ticker for the benchmark series (e.g., ``"SPY"``). Downloaded via
        ``yfinance`` for plotting and relative performance statistics.
    """

    start_date: date
    end_date: date
    tickers: list[str] = field(default_factory=list)
    universe_path: Optional[Path] = None
    fundamentals_path: Iterable[Path] = field(default_factory=list)
    eps_q: float = 0.15
    eps_v: float = 0.15
    eps_m: float = 0.15
    rebalance_months: int = 6
    max_positions: int = 30
    transaction_cost_bps: float = 10.0
    sector_max_weight: float = 0.15
    risk_free_rate: float = 0.0
    cache_dir: Path = Path("data/cache")
    benchmark: str = "SPY"

    @staticmethod
    def from_file(path: Path) -> "BacktestConfig":
        """Load configuration from a YAML or JSON file.

        Parameters
        ----------
        path : Path
            Path to a YAML or JSON configuration file.
        """
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in {".yml", ".yaml"}:
                raw = yaml.safe_load(f)
            else:
                raw = json.load(f)
        return BacktestConfig.from_dict(raw)

    @staticmethod
    def from_dict(cfg: dict) -> "BacktestConfig":
        """Instantiate a config object from a plain dictionary."""
        start = cfg.get("start_date")
        end = cfg.get("end_date")
        if isinstance(start, str):
            start = date.fromisoformat(start)
        if isinstance(end, str):
            end = date.fromisoformat(end)

        fundamentals = cfg.get("fundamentals_path", [])
        fundamentals_paths = [Path(p) for p in fundamentals] if fundamentals else []

        universe_path = cfg.get("universe_path")
        tickers = cfg.get("tickers", [])

        return BacktestConfig(
            start_date=start,
            end_date=end,
            tickers=list(tickers),
            universe_path=Path(universe_path) if universe_path else None,
            fundamentals_path=fundamentals_paths,
            eps_q=float(cfg.get("eps_q", 0.15)),
            eps_v=float(cfg.get("eps_v", 0.15)),
            eps_m=float(cfg.get("eps_m", 0.15)),
            rebalance_months=int(cfg.get("rebalance_months", 6)),
            max_positions=int(cfg.get("max_positions", 30)),
            transaction_cost_bps=float(cfg.get("transaction_cost_bps", 10.0)),
            sector_max_weight=float(cfg.get("sector_max_weight", 0.15)),
            risk_free_rate=float(cfg.get("risk_free_rate", 0.0)),
            cache_dir=Path(cfg.get("cache_dir", "data/cache")),
            benchmark=cfg.get("benchmark", "SPY"),
        )


__all__ = ["BacktestConfig"]
