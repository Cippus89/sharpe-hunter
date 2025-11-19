"""Generic helpers shared across modules."""
from __future__ import annotations

import logging
import math
from typing import Iterable


PERCENTAGE_SIGNS = {"%", "pct", "percent"}


def clean_numeric(value: str | float | int | None) -> float | None:
    """Convert snapshot strings into floats.

    Handles commas, percent signs and missing values gracefully.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value):
            return None
        return float(value)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped in {"-", "n/a", "NA"}:
            return None
        lowered = stripped.lower()
        is_percent = any(lowered.endswith(s) for s in PERCENTAGE_SIGNS)
        cleaned = stripped.replace(",", "").replace("%", "").replace("$", "")
        if cleaned.endswith("x"):
            cleaned = cleaned[:-1]
        try:
            val = float(cleaned)
        except ValueError:
            logging.debug("Unable to parse numeric value '%s'", value)
            return None
        if is_percent:
            return val / 100
        return val

    return None


def normalize_columns(columns: Iterable[str]) -> list[str]:
    """Normalize snapshot columns by stripping spaces and replacing spaces."""
    return [col.strip().replace(" ", "_") for col in columns]
