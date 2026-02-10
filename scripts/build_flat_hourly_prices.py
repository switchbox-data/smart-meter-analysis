#!/usr/bin/env python3
"""Build an hourly flat-rate price calendar Parquet from a CSV of monthly rates.

Generates one row per *local* hour in America/Chicago for the requested year,
assigning the flat supply rate for each hour's month.  Output is suitable for
direct comparison against STOU/DTOU hourly tariffs produced by
``build_tariff_hourly_prices.py``.

DST behaviour (matches build_tariff_hourly_prices.py exactly)
─────────────────────────────────────────────────────────────
• Spring-forward (Mar): the skipped local hour is absent from the output.
• Fall-back (Nov): the ambiguous hour appears twice in the tz-aware range;
  the script keeps the *first* UTC occurrence (CDT) and drops the second (CST),
  logging a warning.
• ``datetime_chicago`` uniqueness is enforced; the script fails loudly on
  any violation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

TZ = "America/Chicago"


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_hourly_prices(rates: pl.DataFrame, year: int) -> pl.DataFrame:
    """Build the hourly flat-price DataFrame for *year*."""

    # --- Generate tz-aware hourly range for the full year ----------------
    dt_range = pl.datetime_range(
        datetime(year, 1, 1),
        datetime(year + 1, 1, 1),
        interval="1h",
        time_zone=TZ,
        eager=True,
        closed="left",
    )
    df = pl.DataFrame({"datetime_aware": dt_range})

    # Naive local column
    df = df.with_columns(
        pl.col("datetime_aware").dt.replace_time_zone(None).alias("datetime_chicago"),
    )

    # --- DST fall-back deduplication (keep earliest UTC per local hour) ---
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
        # Sort by the aware column (which sorts by UTC instant) then dedup
        df = df.sort("datetime_aware").unique(subset=["datetime_chicago"], keep="first")
        log.info("After dedup: %d rows (dropped %d).", df.height, n_dupes)

    # Drop the tz-aware helper column
    df = df.drop("datetime_aware")

    # --- Enforce uniqueness -----------------------------------------------
    final_unique = df.select(pl.col("datetime_chicago").n_unique()).item()
    if final_unique != df.height:
        raise ValueError(
            f"datetime_chicago is not unique after dedup: {df.height} rows but {final_unique} distinct values"
        )

    # --- Extract helper columns -------------------------------------------
    df = df.with_columns(
        pl.col("datetime_chicago").dt.year().cast(pl.Int32).alias("year"),
        pl.col("datetime_chicago").dt.month().cast(pl.Int32).alias("month"),
        pl.col("datetime_chicago").dt.hour().cast(pl.Int32).alias("hour"),
        # Polars weekday: Mon=1 … Sun=7 → convert to Mon=0 … Sun=6
        (pl.col("datetime_chicago").dt.weekday() - 1).cast(pl.Int32).alias("day_of_week"),
    )
    df = df.with_columns((pl.col("day_of_week") >= 5).alias("is_weekend"))

    # --- Join flat rates --------------------------------------------------
    rates_join = rates.select(
        pl.col("year").cast(pl.Int32),
        pl.col("month").cast(pl.Int32),
        pl.col("flat_price_cents").cast(pl.Float64).alias("price_cents_per_kwh"),
    )
    df = df.join(rates_join, on=["year", "month"], how="left")

    # --- Fail-loud checks -------------------------------------------------
    null_prices = df.filter(pl.col("price_cents_per_kwh").is_null()).height
    if null_prices > 0:
        missing = df.filter(pl.col("price_cents_per_kwh").is_null()).select("year", "month").unique()
        raise ValueError(f"{null_prices} rows have null price_cents_per_kwh. Missing rates for:\n{missing}")
    if df.height == 0:
        raise ValueError("Output DataFrame is empty")

    # --- Final select & sort ----------------------------------------------
    df = df.select(
        "datetime_chicago",
        "year",
        "month",
        "price_cents_per_kwh",
        "hour",
        "day_of_week",
        "is_weekend",
    ).sort("datetime_chicago")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build hourly flat-rate price calendar from monthly CSV.",
    )
    p.add_argument("--year", type=int, required=True, help="Calendar year to generate.")
    p.add_argument(
        "--flat-rates-csv",
        type=Path,
        required=True,
        help="Path to CSV with columns: year, month, flat_price_cents.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path (default: data/reference/comed_flat_hourly_prices_{YEAR}.parquet).",
    )
    p.add_argument("--check", action="store_true", help="Build and validate but do not write output.")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    # --- Load and validate flat rates -------------------------------------
    csv_path: Path = args.flat_rates_csv
    if not csv_path.exists():
        sys.exit(f"Flat rates CSV not found: {csv_path}")

    rates = pl.read_csv(csv_path)
    log.info("Loaded %d rows from %s", rates.height, csv_path)

    # Ensure every month 1-12 has a rate for the requested year
    expected_months = set(range(1, 13))
    year_rates = rates.filter(pl.col("year") == args.year)
    present_months = set(year_rates["month"].to_list())
    missing_months = sorted(expected_months - present_months)
    if missing_months:
        sys.exit(f"Missing flat rates for year={args.year}, months={missing_months}")

    # --- Build hourly calendar --------------------------------------------
    log.info("Building hourly flat prices for year %d …", args.year)
    df = build_hourly_prices(year_rates, args.year)
    log.info("Built %d rows.", df.height)

    # Summary
    summary = df.group_by("month").agg(pl.len().alias("hours"), pl.col("price_cents_per_kwh").first()).sort("month")
    log.info("Monthly summary:\n%s", summary)

    if args.check:
        log.info("--check mode: skipping write.")
        return

    # --- Write output -----------------------------------------------------
    out_path = args.output or Path(f"data/reference/comed_flat_hourly_prices_{args.year}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    log.info("Wrote %s (%d rows)", out_path, df.height)


if __name__ == "__main__":
    main()
