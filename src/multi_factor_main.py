"""Entry point for the multi-factor LoM backtest framework.

Example usage (after providing fundamentals CSVs and optional universe file)::

    python -m src.multi_factor_main --config config/multi_factor.yml

"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .multi_factor_backtest import run_backtest
from .multi_factor_config import BacktestConfig
from .multi_factor_data import (
    Universe,
    download_benchmark,
    download_price_history,
    load_fundamentals,
    load_universe,
    pivot_prices,
)
from .multi_factor_reporting import compute_performance, plot_drawdown, plot_equity_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-factor Layers-of-Maxima backtest")
    parser.add_argument("--config", type=Path, required=False, help="Path to YAML/JSON config")
    return parser.parse_args()


def load_configuration(path: Path | None) -> BacktestConfig:
    if path and path.exists():
        return BacktestConfig.from_file(path)
    raise ValueError("Configuration path required. Provide via --config.")


def main() -> None:
    args = parse_args()
    config = load_configuration(args.config)

    universe = load_universe(config.tickers, config.universe_path)
    price_df = download_price_history(universe.tickers, config.start_date, config.end_date, config.cache_dir)
    close, adj_close = pivot_prices(price_df)

    fundamentals = load_fundamentals(config.fundamentals_path)

    benchmark = download_benchmark(config.benchmark, config.start_date, config.end_date, config.cache_dir)

    result = run_backtest(config, universe, fundamentals, adj_close, benchmark)
    stats = compute_performance(result.daily_returns, result.turnover, risk_free_rate=config.risk_free_rate)

    print("=== Performance ===")
    print(f"CAGR: {stats.cagr:.2%}  Vol: {stats.vol:.2%}  Sharpe: {stats.sharpe:.2f}")
    print(f"Max Drawdown: {stats.max_drawdown:.2%}  Avg Turnover: {stats.turnover:.2%}")

    plot_equity_curve(result.portfolio_history, result.benchmark)
    plot_drawdown(result.daily_returns)


if __name__ == "__main__":  # pragma: no cover
    main()
