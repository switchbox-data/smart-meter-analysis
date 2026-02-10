#!/usr/bin/env python3
"""Build an hourly tariff price calendar Parquet from a YAML rate structure.

Generates one row per *local* hour in America/Chicago for the requested year,
mapping each hour to its season and TOU period with the associated price.

DST behaviour (v1)
──────────────────
The script builds a timezone-aware hourly range in America/Chicago, then strips
the timezone to produce naive local timestamps (``datetime_chicago``).  A UTC
column (``datetime_utc``) is also emitted for auditability.

• Spring-forward (Mar): the 2 AM hour does not exist locally → 8 759 rows.
• Fall-back (Nov): the 1 AM hour occurs twice → two distinct UTC instants map
  to the *same* naive local timestamp.

Because downstream joins key on ``datetime_chicago``-naive, duplicate values
would silently corrupt results.  **v1 policy:** when a fall-back duplicate is
detected, the script keeps the *first* UTC occurrence (the CDT instant before
the clock falls back) and drops the second (the CST instant).  A warning is
logged listing every dropped row.  The ``datetime_utc`` column is included so
that the deduplication is auditable.

This means the output always has exactly one row per unique ``datetime_chicago``
value, with either 8 759 (spring-forward year) or 8 760 rows for a full year.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required but not installed. Install it with:\n  uv add pyyaml")

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the YAML rate structure."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError("YAML root must be a mapping")
    for key in ("name", "timezone", "unit", "seasons"):
        if key not in cfg:
            raise ValueError(f"Missing required key: {key}")
    if not cfg["seasons"]:
        raise ValueError("At least one season is required")
    return cfg


def _parse_mmdd(mmdd: str) -> tuple[int, int]:
    """Parse 'MM-DD' → (month, day)."""
    parts = mmdd.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid MM-DD: {mmdd!r}")
    return int(parts[0]), int(parts[1])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_period_coverage(seasons: list[dict[str, Any]]) -> None:
    """Ensure every hour 0-23 maps to exactly one period per season."""
    for season in seasons:
        name = season["name"]
        hour_map: dict[int, str] = {}
        for p in season["periods"]:
            start_h = p["start_hour"]
            end_h = p["end_hour"]
            hours = _expand_hour_range(start_h, end_h)
            for h in hours:
                if h in hour_map:
                    raise ValueError(f"Season '{name}': hour {h} mapped to both '{hour_map[h]}' and '{p['period']}'")
                hour_map[h] = p["period"]
        missing = sorted(set(range(24)) - set(hour_map))
        if missing:
            raise ValueError(f"Season '{name}': hours {missing} have no period mapping")


def _expand_hour_range(start: int, end: int) -> list[int]:
    """Expand [start, end) with midnight wrap-around into a list of hours."""
    if end > start:
        return list(range(start, end))
    # wrap: e.g. 21->6 means [21..24) U [0..6)
    return list(range(start, 24)) + list(range(0, end))


# ---------------------------------------------------------------------------
# Season / period resolution
# ---------------------------------------------------------------------------


def _mmdd_to_dayofyear(month: int, day: int, year: int) -> int:
    return datetime(year, month, day).timetuple().tm_yday


def _in_season(month: int, day: int, start_mmdd: str, end_mmdd: str, year: int) -> bool:
    """Check if month/day falls within [start_mmdd, end_mmdd] inclusive, with wrap."""
    sm, sd = _parse_mmdd(start_mmdd)
    em, ed = _parse_mmdd(end_mmdd)
    s_doy = _mmdd_to_dayofyear(sm, sd, year)
    e_doy = _mmdd_to_dayofyear(em, ed, year)
    cur = _mmdd_to_dayofyear(month, day, year)
    if s_doy <= e_doy:
        return s_doy <= cur <= e_doy
    # wrap-around (e.g. Oct-May)
    return cur >= s_doy or cur <= e_doy


def resolve_season(month: int, day: int, year: int, seasons: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the season dict for a given date."""
    for s in seasons:
        if _in_season(month, day, s["start_mmdd"], s["end_mmdd"], year):
            return s
    raise ValueError(f"No season covers {month:02d}-{day:02d}")


def resolve_period(hour: int, season: dict[str, Any]) -> tuple[str, float]:
    """Return (period_name, price) for a given hour within a season."""
    for p in season["periods"]:
        hours = _expand_hour_range(p["start_hour"], p["end_hour"])
        if hour in hours:
            return p["period"], p["price"]
    raise ValueError(f"Season '{season['name']}': no period covers hour {hour}")


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_hourly_prices(cfg: dict[str, Any], year: int) -> pl.DataFrame:
    """Build the hourly price DataFrame for *year*."""
    tz = cfg["timezone"]
    seasons = cfg["seasons"]

    # Timezone-aware hourly range: start of year → start of next year
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    dt_range = pl.datetime_range(
        start,
        end,
        interval="1h",
        time_zone=tz,
        eager=True,
        closed="left",
    )

    # Build a minimal frame with the aware column
    df = pl.DataFrame({"datetime_aware": dt_range})

    # UTC column (for auditability)
    df = df.with_columns(
        pl.col("datetime_aware").dt.convert_time_zone("UTC").dt.replace_time_zone(None).alias("datetime_utc"),
    )

    # Naive local column
    df = df.with_columns(
        pl.col("datetime_aware").dt.replace_time_zone(None).alias("datetime_chicago"),
    )

    # DST fall-back deduplication: keep earliest UTC occurrence per local hour
    n_total = df.height
    n_unique = df.select(pl.col("datetime_chicago").n_unique()).item()
    if n_unique != n_total:
        n_dupes = n_total - n_unique
        dupes = df.group_by("datetime_chicago").len().filter(pl.col("len") > 1).sort("datetime_chicago")
        log.warning(
            "DST fall-back: %d duplicate naive timestamp(s) detected. "
            "Keeping first UTC occurrence, dropping later:\n%s",
            n_dupes,
            dupes,
        )
        # Keep first UTC occurrence (sort by datetime_utc, then unique on datetime_chicago)
        df = df.sort("datetime_utc").unique(subset=["datetime_chicago"], keep="first")
        log.info("After dedup: %d rows (dropped %d).", df.height, n_dupes)

    # Extract helper columns
    df = df.with_columns(
        pl.col("datetime_chicago").dt.year().cast(pl.Int32).alias("year"),
        pl.col("datetime_chicago").dt.month().cast(pl.Int32).alias("month"),
        pl.col("datetime_chicago").dt.day().cast(pl.Int32).alias("day"),
        pl.col("datetime_chicago").dt.hour().cast(pl.Int32).alias("hour"),
        pl.col("datetime_chicago").dt.weekday().cast(pl.Int32).alias("day_of_week"),  # Mon=1 in Polars
    )

    # Polars weekday: Monday=1 … Sunday=7  →  convert to Monday=0 … Sunday=6
    df = df.with_columns(
        (pl.col("day_of_week") - 1).alias("day_of_week"),
    )

    df = df.with_columns(
        (pl.col("day_of_week") >= 5).alias("is_weekend"),
    )

    # Map each row to season / period / price via Python UDF (simple & correct)
    season_names: list[str] = []
    period_names: list[str] = []
    prices: list[float] = []

    months = df["month"].to_list()
    days = df["day"].to_list()
    hours = df["hour"].to_list()

    for m, d, h in zip(months, days, hours):
        s = resolve_season(m, d, year, seasons)
        pname, price = resolve_period(h, s)
        season_names.append(s["name"])
        period_names.append(pname)
        prices.append(price)

    df = df.with_columns(
        pl.Series("season", season_names, dtype=pl.Utf8),
        pl.Series("period", period_names, dtype=pl.Utf8),
        pl.Series("price_cents_per_kwh", prices, dtype=pl.Float64),
    )

    # Final column selection & sort
    df = df.select(
        "datetime_chicago",
        "datetime_utc",
        "year",
        "month",
        "hour",
        "day_of_week",
        "is_weekend",
        "season",
        "period",
        "price_cents_per_kwh",
    ).sort("datetime_chicago")

    # Null check
    null_counts = df.null_count()
    total_nulls = sum(null_counts.row(0))
    if total_nulls > 0:
        raise ValueError(f"Output contains nulls:\n{null_counts}")
    if df.height == 0:
        raise ValueError("Output DataFrame is empty")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build hourly tariff price calendar from YAML rate structure.",
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML rate structure file.",
    )
    p.add_argument(
        "--year",
        type=int,
        required=True,
        help="Calendar year to generate.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path (default: data/reference/comed_stou_hourly_prices_{YEAR}.parquet).",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Validate config and build DataFrame, but do not write output.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    log.info("Loading config: %s", args.config)
    cfg = load_config(args.config)
    log.info("Rate structure: %s  tz=%s", cfg["name"], cfg["timezone"])

    log.info("Validating period coverage …")
    validate_period_coverage(cfg["seasons"])
    log.info("Period coverage OK — all hours 0-23 mapped in every season.")

    log.info("Building hourly prices for year %d …", args.year)
    df = build_hourly_prices(cfg, args.year)
    log.info("Built %d rows.", df.height)

    # Summary stats
    summary = (
        df.group_by("season", "period")
        .agg(pl.len().alias("hours"), pl.col("price_cents_per_kwh").first())
        .sort("season", "period")
    )
    log.info("Summary:\n%s", summary)

    if args.check:
        log.info("--check mode: skipping write.")
        return

    out_path = args.output or Path(f"data/reference/comed_stou_hourly_prices_{args.year}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    log.info("Wrote %s (%d rows)", out_path, df.height)


if __name__ == "__main__":
    main()
