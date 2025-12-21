"""
Configuration loader for monthly pipeline runs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file with environment variable substitution.

    Args:
        config_path: Path to config file. If None, uses default config/monthly_run.yaml

    Returns:
        Dictionary with configuration values
    """
    if config_path is None:
        # Default to config/monthly_run.yaml relative to project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "monthly_run.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Substitute environment variables (simple ${VAR} syntax)
    config = _substitute_env_vars(config)

    return config


def _substitute_env_vars(obj: Any) -> Any:
    """Recursively substitute ${VAR} patterns with environment variables."""
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        var_name = obj[2:-1]
        return os.getenv(var_name, obj)  # Return original if env var not set
    else:
        return obj


def get_year_month(config: dict[str, Any] | None = None) -> tuple[int, int]:
    """
    Extract year and month from config.

    Args:
        config: Config dict. If None, loads default config.

    Returns:
        Tuple of (year, month)
    """
    if config is None:
        config = load_config()

    year = config.get("year", 2023)
    month = config.get("month", 7)

    if not (1 <= month <= 12):
        raise ValueError(f"Month must be between 1 and 12, got {month}")

    return (year, month)


def get_year_month_str(config: dict[str, Any] | None = None) -> str:
    """
    Get year-month string in YYYYMM format.

    Args:
        config: Config dict. If None, loads default config.

    Returns:
        String like "202307" for July 2023
    """
    year, month = get_year_month(config)
    return f"{year}{month:02d}"
