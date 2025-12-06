#!/usr/bin/env python3
"""
Build hourly "spread" between ComEd flat default supply rate and RTP prices.

Given:
  1) Hourly RTP prices from the ComEd Hourly Pricing API
     (output of scripts/fetch_rtp_prices.py), and
  2) A table of monthly flat rates (Price-to-Compare / default supply) in cents/kWh,

this script creates an hourly dataset with:
  - RTP hourly price (cents/kWh)
  - Flat monthly price (cents/kWh)
  - Hourly spread: flat - RTP (cents/kWh)

Example flat-rate CSV schema (data/reference/comed_flat_rates_2023.csv):
    year,month,flat_price_cents
    2023,1,XX.XXX
    2023,2,XX.XXX
    ...
    2023,12,XX.XXX

Typical usage:
    python scripts/build_rtp_spreads.py \
        --year 2023 \
        --rtp-file data/reference/comed_hourly_prices_2023.parquet \
        --flat-rates-csv data/reference/comed_flat_rates_2023.csv \
        --output data/reference/comed_hourly_spreads_2023.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_hourly_rtp(path: Path) -> pl.DataFrame:
    """
    Load hourly RTP prices.

    Expects columns:
        - datetime_chicago (datetime)
        - datetime_utc (datetime)
        - price_cents (float)
        - n_intervals (int)
    """
    if not path.exists():
        raise FileNotFoundError(f"RTP file not found: {path}")

    logger.info(f"Loading hourly RTP prices from {path}")
    df = pl.read_parquet(path)

    required = {"datetime_chicago", "datetime_utc", "price_cents"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"RTP file missing required columns: {sorted(missing)}")

    return df


def load_flat_rates(flat_rates_csv: Path, year: int) -> pl.DataFrame:
    """
    Load monthly flat rates for a given year from CSV.

    CSV schema must include:
        - year (int)
        - month (1-12, int)
        - flat_price_cents (float)

    Returns:
        DataFrame with columns: year, month, flat_price_cents
        filtered to the requested year.
    """
    if not flat_rates_csv.exists():
        raise FileNotFoundError(
            f"Flat-rate CSV not found: {flat_rates_csv}\n"
            "Create it from the Historical Price Comparison PDF with columns "
            "[year, month, flat_price_cents]."
        )

    logger.info(f"Loading flat rates from {flat_rates_csv}")
    df = pl.read_csv(flat_rates_csv)

    required = {"year", "month", "flat_price_cents"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Flat-rate CSV missing columns: {sorted(missing)}")

    df = df.with_columns(
        pl.col("year").cast(pl.Int32),
        pl.col("month").cast(pl.Int32),
        pl.col("flat_price_cents").cast(pl.Float64),
    )

    df_year = df.filter(pl.col("year") == year).select(["year", "month", "flat_price_cents"])

    if df_year.is_empty():
        raise ValueError(f"No flat-rate rows found for year={year} in {flat_rates_csv}")

    # Sanity check: expect up to 12 months
    months_present = sorted(df_year["month"].unique().to_list())
    logger.info(f"  Loaded flat rates for months: {months_present}")

    return df_year


def build_spreads(
    rtp_df: pl.DataFrame,
    flat_df: pl.DataFrame,
    year: int,
) -> pl.DataFrame:
    """
    Join hourly RTP data with monthly flat rates and compute hourly spreads.

    Args:
        rtp_df: Hourly RTP DataFrame.
        flat_df: Monthly flat-rate DataFrame with [year, month, flat_price_cents].
        year: Year for which we're computing spreads.

    Returns:
        DataFrame with columns:
            - datetime_chicago
            - datetime_utc
            - year
            - month
            - rtp_price_cents
            - flat_price_cents
            - spread_cents
            - n_intervals
    """
    logger.info("Building hourly spread dataset...")

    # Attach year/month to RTP data (local time)
    rtp_aug = rtp_df.with_columns(
        pl.col("datetime_chicago").dt.year().alias("year"),
        pl.col("datetime_chicago").dt.month().alias("month"),
    )

    # Filter to target year (safety)
    rtp_aug = rtp_aug.filter(pl.col("year") == year)

    logger.info(f"  RTP rows for {year}: {len(rtp_aug):,}")

    # Join on (year, month)
    joined = rtp_aug.join(
        flat_df,
        on=["year", "month"],
        how="left",
    )

    # Check for unmatched rows
    unmatched = joined.filter(pl.col("flat_price_cents").is_null())
    n_unmatched = len(unmatched)
    if n_unmatched > 0:
        months_unmatched = unmatched.select(["year", "month"]).unique().sort(["year", "month"]).to_dicts()
        raise ValueError(f"Some hours have no matching flat rate. Examples (year, month): {months_unmatched[:5]}")

    spreads = joined.with_columns(
        pl.col("price_cents").alias("rtp_price_cents"),
        (pl.col("flat_price_cents") - pl.col("price_cents")).alias("spread_cents"),
    )

    # Select and order final columns
    spreads = spreads.select([
        "datetime_chicago",
        "datetime_utc",
        "year",
        "month",
        "rtp_price_cents",
        "flat_price_cents",
        "spread_cents",
        "n_intervals",
    ]).sort("datetime_chicago")

    # Basic summary
    logger.info(
        "Spread summary (cents/kWh): "
        f"min={spreads['spread_cents'].min():.3f}, "
        f"max={spreads['spread_cents'].max():.3f}, "
        f"mean={spreads['spread_cents'].mean():.3f}"
    )

    return spreads


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build hourly spread between flat ComEd default rate and RTP prices.")
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Year to process (default: 2023)",
    )
    parser.add_argument(
        "--rtp-file",
        type=Path,
        default=Path("data/reference/comed_hourly_prices_2023.parquet"),
        help="Hourly RTP parquet file (default: data/reference/comed_hourly_prices_2023.parquet)",
    )
    parser.add_argument(
        "--flat-rates-csv",
        type=Path,
        default=Path("data/reference/comed_flat_rates_2023.csv"),
        help="CSV with monthly flat rates (year, month, flat_price_cents).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reference/comed_hourly_spreads_2023.parquet"),
        help="Output parquet path (default: data/reference/comed_hourly_spreads_2023.parquet)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(f"BUILDING FLAT VS RTP SPREADS FOR {args.year}")
    print("=" * 70)

    try:
        rtp_df = load_hourly_rtp(args.rtp_file)
        flat_df = load_flat_rates(args.flat_rates_csv, args.year)
        spreads_df = build_spreads(rtp_df, flat_df, args.year)
    except Exception as e:
        logger.error(f"Failed to build spreads: {e}")
        return 1

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spreads_df.write_parquet(args.output)

    print("\n" + "-" * 70)
    print("OUTPUT")
    print("-" * 70)
    print(f"  Saved {len(spreads_df):,} hourly spread rows to {args.output}")

    # Show a small sample
    print("\n  Sample data:")
    print(spreads_df.head(5))

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
