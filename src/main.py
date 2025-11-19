"""Entry point for running the cross-sectional valuation pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from .backtest import run_mispricing_backtest
from .config_loader import load_config
from .data_fetch import download_price_history
from .dataset import prepare_cross_sectional_dataset
from .regression import run_cross_sectional_regression, save_regression_output


LOG_PATH = Path("logs/run.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)


def _get_features(config: dict) -> Sequence[str]:
    features = config.get("features", {})
    core = features.get("core", [])
    optional = features.get("optional", [])
    return list(dict.fromkeys(core + optional))


def main() -> None:
    config = load_config()
    snapshot_path = config.get("snapshot_path")
    if not snapshot_path:
        raise ValueError("snapshot_path missing from configuration")

    feature_list = _get_features(config)
    dataset = prepare_cross_sectional_dataset(snapshot_path, feature_list)

    price_config = config.get("price_history", {})
    tickers = dataset["Ticker"].unique().tolist()
    price_history = download_price_history(
        tickers,
        start=price_config.get("start", "2015-01-01"),
        end=price_config.get("end"),
        interval=price_config.get("interval", "1d"),
        cache_days=int(price_config.get("cache_days", 1)),
    )

    log_price = config.get("regression", {}).get("log_price", False)
    model, results_df = run_cross_sectional_regression(dataset, feature_list, log_price=log_price)

    industry_name = config.get("industry_name", "industry")
    save_regression_output(industry_name, results_df, model)

    logging.info("Regression finished with R^2=%.3f and adj.R^2=%.3f", model.rsquared, model.rsquared_adj)

    # Print console summary
    significant = [name for name, pval in model.pvalues.items() if name != "const" and pval < 0.05]
    print("\n=== Regression Summary ===")
    print(f"Tickers: {len(dataset)}")
    print(f"R^2: {model.rsquared:.3f}  Adj. R^2: {model.rsquared_adj:.3f}")
    print(f"Significant variables (p<0.05): {', '.join(significant) if significant else 'None'}")

    sorted_results = results_df.sort_values("mispricing_pct", ascending=False)
    print("\nTop undervalued:")
    print(sorted_results.head(10).to_string(index=False))
    print("\nTop overvalued:")
    print(sorted_results.tail(10).to_string(index=False))

    backtest_cfg = config.get("backtest", {})
    stats = run_mispricing_backtest(
        results_df,
        price_history,
        holding_period_days=int(backtest_cfg.get("holding_period_days", 126)),
        top_quantile=float(backtest_cfg.get("top_quantile", 0.2)),
    )
    if stats:
        print("\n=== Backtest (naive long-short) ===")
        print(
            f"Long positions: {stats['long_count']}  Short positions: {stats['short_count']}  "
            f"Mean return: {stats['mean_return']:.2%}  Sharpe (simple): {stats['sharpe']:.2f}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
