"""Dataset preparation utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .data_utils import clean_numeric


REQUIRED_COLUMNS = {"Ticker", "Price"}


def read_finviz_snapshot(snapshot_path: str | Path, required_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Read and clean a Finviz snapshot CSV file."""
    path = Path(snapshot_path)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [col.strip().replace(" ", "") for col in df.columns]

    required = set(REQUIRED_COLUMNS)
    if required_columns:
        required.update(required_columns)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Snapshot missing required columns: {missing}")

    numeric_cols = [col for col in df.columns if col != "Ticker"]
    for col in numeric_cols:
        df[col] = df[col].apply(clean_numeric)

    return df


def prepare_cross_sectional_dataset(
    snapshot_path: str | Path,
    feature_set: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a cleaned dataset ready for regression."""
    df = read_finviz_snapshot(snapshot_path, required_columns=feature_set)

    if feature_set is None:
        feature_set = [col for col in df.columns if col not in {"Ticker", "Price"}]

    columns = ["Ticker", "Price", *feature_set]
    dataset = df[columns].dropna()

    logging.info("Prepared dataset with %s rows and %s features", len(dataset), len(feature_set))
    return dataset
