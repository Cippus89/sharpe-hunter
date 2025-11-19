"""Utility helpers for loading configuration files."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path = "config/config.yml") -> Dict[str, Any]:
    """Load YAML configuration returning a dictionary.

    Parameters
    ----------
    path: str | Path
        Path to a YAML configuration file.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    return data
