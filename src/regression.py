"""Regression helpers for cross-sectional analysis."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_cross_sectional_regression(
    dataset: pd.DataFrame,
    features: Sequence[str],
    log_price: bool = False,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    """Run OLS on the supplied dataset."""
    if dataset.empty:
        raise ValueError("Dataset is empty; cannot run regression")
    missing = [col for col in features if col not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset missing required features: {missing}")

    y = dataset["Price"].astype(float)
    if log_price:
        y = np.log(y)

    X = dataset[list(features)].astype(float)
    X = sm.add_constant(X)

    model = sm.OLS(y, X, hasconst=True).fit()

    predictions = model.predict(X)
    if log_price:
        fair_values = np.exp(predictions)
    else:
        fair_values = predictions

    results_df = dataset[["Ticker", "Price"]].copy()
    results_df["fair_value"] = fair_values
    results_df["mispricing_pct"] = (results_df["fair_value"] / results_df["Price"] - 1.0) * 100

    return model, results_df


def save_regression_output(industry_name: str, results_df: pd.DataFrame, model: sm.regression.linear_model.RegressionResultsWrapper) -> Path:
    """Persist regression outputs to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = RESULTS_DIR / f"{industry_name}_regression.csv"
    results_df.to_csv(file_path, index=False)

    text_path = RESULTS_DIR / f"{industry_name}_regression_summary.txt"
    with text_path.open("w", encoding="utf-8") as handle:
        handle.write(model.summary().as_text())

    logging.info("Saved regression output to %s", file_path)
    return file_path
