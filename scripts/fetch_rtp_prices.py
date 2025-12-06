#!/usr/bin/env python3
"""
Fetch ComEd Real-Time Pricing (RTP) hourly prices for a given year.

Uses the ComEd Hourly Pricing API to download 5-minute prices,
then aggregates to hourly averages in local (Chicago) time.

API Documentation: https://hourlypricing.comed.com/hp-api/

Typical output (for 2023):
data/reference/comed_hourly_prices_2023.parquet
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
COMED_API_BASE = "https://hourlypricing.comed.com/api"
CHICAGO_TZ = ZoneInfo("America/Chicago")
UTC_TZ = ZoneInfo("UTC")

# Rate limiting
REQUEST_DELAY_SECONDS = 1.0


def fetch_5min_prices(start_date: str, end_date: str) -> list[dict]:
    """
    Fetch 5-minute prices from ComEd API for a date range.

    Args:
        start_date: Start datetime in format YYYYMMDDhhmm (local Chicago time)
        end_date: End datetime in format YYYYMMDDhhmm (local Chicago time)

    Returns:
        List of dicts with at least 'millisUTC' and 'price' keys.
    """
    url = f"{COMED_API_BASE}?type=5minutefeed&datestart={start_date}&dateend={end_date}"
    logger.info(f"Requesting 5-minute prices: {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.RequestException as e:
        logger.error(f"API request failed for URL {url}: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from URL {url}: {e}")
        return []


def fetch_month(year: int, month: int) -> pl.DataFrame:
    """
    Fetch all 5-minute prices for a given month.

    Args:
        year: Year (e.g., 2023)
        month: Month (1-12)

    Returns:
        DataFrame with columns:
            - datetime_utc (naive; UTC)
            - datetime_chicago (naive; local)
            - price_cents
    """
    # Start of month in local time
    start_dt = datetime(year, month, 1, 0, 0, tzinfo=CHICAGO_TZ)

    # End-of-month last 5-minute interval in local time
    if month == 12:
        end_dt = datetime(year + 1, 1, 1, 0, 0, tzinfo=CHICAGO_TZ) - timedelta(minutes=5)
    else:
        end_dt = datetime(year, month + 1, 1, 0, 0, tzinfo=CHICAGO_TZ) - timedelta(minutes=5)

    # Format for API (local time, YYYYMMDDhhmm)
    start_str = start_dt.strftime("%Y%m%d%H%M")
    end_str = end_dt.strftime("%Y%m%d%H%M")

    logger.info(f"Fetching {year}-{month:02d}: {start_str} to {end_str}")

    data = fetch_5min_prices(start_str, end_str)

    if not data:
        logger.warning(f"No data returned for {year}-{month:02d}")
        return pl.DataFrame()

    logger.info(f"  Retrieved {len(data)} 5-minute intervals")

    # Parse into DataFrame
    records: list[dict] = []
    for item in data:
        try:
            millis = int(item["millisUTC"])
            price = float(item["price"])
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Skipping malformed item {item}: {e}")
            continue

        utc_dt = datetime.fromtimestamp(millis / 1000, tz=UTC_TZ)
        chicago_dt = utc_dt.astimezone(CHICAGO_TZ)

        records.append({
            # Strip tzinfo; already baked into the wall clock time
            "datetime_utc": utc_dt.replace(tzinfo=None),
            "datetime_chicago": chicago_dt.replace(tzinfo=None),
            "price_cents": price,
        })

    if not records:
        logger.warning(f"All records malformed / skipped for {year}-{month:02d}")
        return pl.DataFrame()

    df = pl.DataFrame(records)
    return df


def aggregate_to_hourly(df_5min: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate 5-minute prices to hourly averages.

    The hourly price is the average of the twelve 5-minute prices in that hour.

    Args:
        df_5min: DataFrame with 5-minute prices

    Returns:
        DataFrame with columns:
            - hour_chicago
            - hour_utc
            - price_cents (hourly average)
            - n_intervals (number of 5-minute points used)
    """
    if df_5min.is_empty():
        return pl.DataFrame()

    df_hourly = (
        df_5min.with_columns(
            pl.col("datetime_chicago").dt.truncate("1h").alias("hour_chicago"),
            pl.col("datetime_utc").dt.truncate("1h").alias("hour_utc"),
        )
        .group_by("hour_chicago")
        .agg(
            pl.col("hour_utc").first(),
            pl.col("price_cents").mean().alias("price_cents"),
            pl.col("price_cents").count().alias("n_intervals"),
        )
        .sort("hour_chicago")
    )

    return df_hourly


def fetch_year(year: int, cache_dir: Path) -> pl.DataFrame:
    """
    Fetch all hourly prices for a year.

    Args:
        year: Year to fetch
        cache_dir: Directory to store/load monthly 5-minute parquet caches

    Returns:
        DataFrame with all hourly prices for the year.
    """
    all_hourly: list[pl.DataFrame] = []

    for month in range(1, 13):
        # Check for cached monthly file
        cache_file = cache_dir / f"rtp_5min_{year}{month:02d}.parquet"

        if cache_file.exists():
            logger.info(f"Loading cached data for {year}-{month:02d}")
            df_5min = pl.read_parquet(cache_file)
        else:
            df_5min = fetch_month(year, month)

            if not df_5min.is_empty():
                # Cache the raw 5-minute data
                df_5min.write_parquet(cache_file)
                logger.info(f"  Cached 5-minute data to {cache_file}")

            # Rate limiting between API calls
            time.sleep(REQUEST_DELAY_SECONDS)

        if not df_5min.is_empty():
            df_hourly = aggregate_to_hourly(df_5min)
            all_hourly.append(df_hourly)
            logger.info(f"  Aggregated to {len(df_hourly)} hourly prices")

    if not all_hourly:
        logger.error(f"No hourly data fetched for year {year}")
        return pl.DataFrame()

    # Combine all months
    df_year = pl.concat(all_hourly).sort("hour_chicago")

    return df_year


def validate_hourly_data(df: pl.DataFrame, year: int) -> dict:
    """
    Validate the hourly price data.

    Args:
        df: Hourly price DataFrame
        year: Year being validated

    Returns:
        Dictionary with validation statistics.
    """
    if df.is_empty():
        return {
            "year": year,
            "total_hours": 0,
            "expected_hours": None,
            "completeness_pct": 0.0,
            "gaps_found": None,
            "price_min": None,
            "price_max": None,
            "price_mean": None,
            "price_median": None,
            "negative_price_hours": None,
        }

    # For a non-leap year we expect ~8760 local hours
    expected_hours = 8760
    actual_hours = len(df)

    # Check for gaps > 1 hour in local time
    df_sorted = df.sort("hour_chicago")
    hour_diffs = df_sorted.select(pl.col("hour_chicago").diff().dt.total_hours().alias("gap_hours")).drop_nulls()

    gaps = hour_diffs.filter(pl.col("gap_hours") > 1)

    stats = {
        "year": year,
        "total_hours": actual_hours,
        "expected_hours": expected_hours,
        "completeness_pct": round(actual_hours / expected_hours * 100, 2),
        "gaps_found": len(gaps),
        "price_min": df["price_cents"].min(),
        "price_max": df["price_cents"].max(),
        "price_mean": round(df["price_cents"].mean(), 3),
        "price_median": df["price_cents"].median(),
        "negative_price_hours": len(df.filter(pl.col("price_cents") < 0)),
    }

    return stats


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch ComEd RTP hourly prices via the Hourly Pricing API.")
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Year to fetch (default: 2023)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reference"),
        help="Output directory (default: data/reference)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for intermediate 5-minute files (default: OUTPUT_DIR/cache)",
    )

    args = parser.parse_args()

    # Setup directories
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir or args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"FETCHING COMED RTP PRICES FOR {args.year}")
    print("=" * 70)

    # Fetch the year
    df_hourly = fetch_year(args.year, cache_dir)

    if df_hourly.is_empty():
        logger.error("No data fetched!")
        return 1

    # Validate
    print("\n" + "-" * 70)
    print("VALIDATION")
    print("-" * 70)

    stats = validate_hourly_data(df_hourly, args.year)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save final output
    output_file = args.output_dir / f"comed_hourly_prices_{args.year}.parquet"

    df_final = df_hourly.select(
        pl.col("hour_chicago").alias("datetime_chicago"),
        pl.col("hour_utc").alias("datetime_utc"),
        pl.col("price_cents"),
        pl.col("n_intervals"),
    )

    df_final.write_parquet(output_file)

    print("\n" + "-" * 70)
    print("OUTPUT")
    print("-" * 70)
    print(f"  Saved {len(df_final)} hourly prices to {output_file}")

    # Show sample
    print("\n  Sample data:")
    print(df_final.head(5))

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
