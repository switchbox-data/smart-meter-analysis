#!/usr/bin/env python3
"""
Step 1, Part 1 of the ComEd Smart Meter Clustering Pipeline: Prepare household-level
data for clustering analysis.

Transforms interval-level energy data into daily load profiles at the
HOUSEHOLD (account) level for k-means clustering.

Pipeline:
    1. Validate schema (no data scan)
    2. Scan for metadata and sample households + dates
    3. Create daily 48-point load profiles per household
    4. Output profiles ready for clustering

Design notes:
    - Minimizes full-file scans (summary + unique accounts + unique dates only)
    - Standard mode: single filtered collect() for selected households/dates
    * good for up to ~50k households sampled
    - Streaming mode: sink_parquet() pre-filter, then aggregate
    * safer for 100k+ households
    - Profiles are 48 half-hourly kWh values in chronological order (00:30-24:00)

Output files:
    - sampled_profiles.parquet:
        One row per (account_identifier, date) with 'profile' = list[float] of length 48
    - household_zip4_map.parquet:
        Unique account_identifier → zip_code mapping for later joins

Usage:
    # Standard (5000 households, 20 days)
    python prepare_clustering_data_households.py \
        --input data/processed/comed_202308.parquet \
        --output-dir data/clustering \
        --sample-households 5000 \
        --sample-days 20

    # Large dataset with streaming
    python prepare_clustering_data_households.py \
        --input data/processed/comed_202308.parquet \
        --output-dir data/clustering \
        --sample-households 100000 \
        --streaming
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Literal

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Required columns for household-level clustering
REQUIRED_ENERGY_COLS = ["zip_code", "account_identifier", "datetime", "kwh"]
REQUIRED_TIME_COLS = ["date", "hour", "is_weekend", "weekday"]


# =============================================================================
# DATA INSPECTION & SAMPLING
# =============================================================================


def validate_schema(path: Path) -> dict[str, Any]:
    """
    Validate input data has required columns (schema-only check, no full data scan).

    Args:
        path: Path to input parquet file.

    Returns:
        Dictionary with:
            - valid: bool
            - errors: list[str]
            - columns: list[str]
    """
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()

    errors: list[str] = []

    missing_energy = [c for c in REQUIRED_ENERGY_COLS if c not in schema.names()]
    missing_time = [c for c in REQUIRED_TIME_COLS if c not in schema.names()]

    if missing_energy:
        errors.append(f"Missing energy columns: {missing_energy}")
    if missing_time:
        errors.append(f"Missing time columns: {missing_time}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "columns": schema.names(),
    }


def get_metadata_and_samples(
    input_path: Path,
    sample_households: int | None,
    sample_days: int,
    day_strategy: Literal["stratified", "random"],
    seed: int = 42,
) -> dict[str, Any]:
    """
    Get summary statistics and sample households + dates using minimal scans.

    This function performs:
    - Summary stats (row counts, unique counts, date range)
    - Unique households
    - Unique dates with weekend flags
    - Sampling of households and dates

    Args:
        input_path: Path to input parquet file.
        sample_households: Number of households to sample (None = all).
        sample_days: Number of days to sample.
        day_strategy: 'stratified' (70/30 weekday/weekend) or 'random'.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with:
            - summary: dict with n_rows, n_accounts, n_zip_codes, min_date, max_date
            - accounts: list[str] of sampled account identifiers
            - dates: list[date] of sampled dates
    """
    logger.info("Scanning input data for metadata and sampling...")

    lf = pl.scan_parquet(input_path)

    # Summary stats
    summary_df = lf.select([
        pl.len().alias("n_rows"),
        pl.col("zip_code").n_unique().alias("n_zip_codes"),
        pl.col("account_identifier").n_unique().alias("n_accounts"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
    ]).collect()
    summary = summary_df.to_dicts()[0]

    # Early checks
    if summary["n_rows"] == 0:
        raise ValueError(f"Input file {input_path} contains no rows.")

    logger.info(f"  {summary['n_rows']:,} rows")
    logger.info(f"  {summary['n_accounts']:,} households")
    logger.info(f"  {summary['n_zip_codes']:,} ZIP+4 codes")
    logger.info(f"  Date range: {summary['min_date']} to {summary['max_date']}")

    # Unique households
    accounts_df = pl.scan_parquet(input_path).select("account_identifier").unique().collect()
    if accounts_df.height == 0:
        raise ValueError("No account_identifier values found in input data.")

    # Unique dates with weekend flag
    dates_df = pl.scan_parquet(input_path).select(["date", "is_weekend"]).unique().collect()
    if dates_df.height == 0:
        raise ValueError("No dates found in input data.")

    # Sample households
    if sample_households is not None and sample_households < len(accounts_df):
        accounts_df = accounts_df.sample(n=sample_households, shuffle=True, seed=seed)
        logger.info(f"  Sampled {len(accounts_df):,} households")
    else:
        logger.info(f"  Using all {len(accounts_df):,} households")

    accounts = accounts_df["account_identifier"].to_list()

    # Sample dates
    if day_strategy == "stratified":
        weekday_df = dates_df.filter(~pl.col("is_weekend"))
        weekend_df = dates_df.filter(pl.col("is_weekend"))

        if weekday_df.height == 0:
            logger.warning("  No weekdays found; falling back to random day sampling.")
            day_strategy = "random"
        elif weekend_df.height == 0:
            logger.warning("  No weekends found; falling back to random day sampling.")
            day_strategy = "random"

    if day_strategy == "stratified":
        n_weekdays = int(sample_days * 0.7)
        n_weekends = sample_days - n_weekdays

        n_weekdays = min(n_weekdays, len(weekday_df))
        n_weekends = min(n_weekends, len(weekend_df))

        sampled_weekdays = (
            weekday_df.sample(n=n_weekdays, shuffle=True, seed=seed)["date"].to_list() if n_weekdays > 0 else []
        )
        sampled_weekends = (
            weekend_df.sample(n=n_weekends, shuffle=True, seed=seed + 1)["date"].to_list() if n_weekends > 0 else []
        )

        dates = sampled_weekdays + sampled_weekends
        logger.info(f"  Sampled {len(sampled_weekdays)} weekdays + {len(sampled_weekends)} weekend days (stratified)")
    else:
        n_sample = min(sample_days, len(dates_df))
        dates = dates_df.sample(n=n_sample, shuffle=True, seed=seed)["date"].to_list()
        logger.info(f"  Sampled {len(dates)} days (random)")

    if not dates:
        raise ValueError("No dates were sampled; check input data and sampling settings.")

    return {
        "summary": summary,
        "accounts": accounts,
        "dates": dates,
    }


# =============================================================================
# PROFILE CREATION
# =============================================================================


def create_household_profiles(
    input_path: Path,
    accounts: list[str],
    dates: list[Any],
) -> pl.DataFrame:
    """
    Create daily load profiles for selected households and dates (in-memory).

    Each profile is a 48-point vector (list) of half-hourly kWh values for one
    household on one day. The profile list is in chronological order
    (00:30 to 24:00).

    This uses a single filtered collect() over the full file, which is efficient
    when sampling a subset of households and dates.

    Args:
        input_path: Path to input parquet file.
        accounts: List of account_identifier values to include.
        dates: List of dates to include.

    Returns:
        DataFrame with columns:
            - account_identifier
            - zip_code
            - date
            - profile        (list[float], length 48)
            - is_weekend
            - weekday
    """
    logger.info(f"Creating profiles for {len(accounts):,} households X {len(dates)} days...")

    if not accounts:
        raise ValueError("No accounts provided for profile creation.")
    if not dates:
        raise ValueError("No dates provided for profile creation.")

    df = (
        pl.scan_parquet(input_path)
        .filter(pl.col("account_identifier").is_in(accounts) & pl.col("date").is_in(dates))
        # Ensures profile list is chronological within each (account, date)
        .sort(["account_identifier", "datetime"])
        .collect()
    )

    if df.is_empty():
        logger.warning("  No data found for selected households/dates")
        return pl.DataFrame()

    logger.info(f"  Loaded {len(df):,} interval records")

    profiles_df = df.group_by(["account_identifier", "zip_code", "date"]).agg([
        pl.col("kwh").alias("profile"),
        pl.col("is_weekend").first(),
        pl.col("weekday").first(),
        pl.len().alias("num_intervals"),
    ])

    n_before = len(profiles_df)
    profiles_df = profiles_df.filter(pl.col("num_intervals") == 48)
    n_dropped = n_before - len(profiles_df)

    if n_dropped > 0:
        logger.info(f"  Dropped {n_dropped:,} incomplete profiles (DST days, missing data)")

    logger.info(f"  Created {len(profiles_df):,} complete profiles")

    return profiles_df.drop("num_intervals")


def create_household_profiles_streaming(
    input_path: Path,
    accounts: list[str],
    dates: list[Any],
    output_path: Path,
) -> int:
    """
    Create daily load profiles using a streaming-friendly two-pass approach.

    Pass 1:
        Filter to selected households/dates and stream to a temp parquet file.
    Pass 2:
        Sort by (account_identifier, datetime), aggregate to daily profiles
        with list-of-48 kWh values, and write final output.

    This avoids loading the entire original parquet into memory, but the
    aggregated profiles are still collected in-memory before final write.

    Args:
        input_path: Path to input parquet file.
        accounts: List of account_identifier values to include.
        dates: List of dates to include.
        output_path: Path to write final profiles parquet.

    Returns:
        Number of complete daily profiles created.
    """
    logger.info(f"Creating profiles (streaming) for {len(accounts):,} households X {len(dates)} days...")

    if not accounts:
        raise ValueError("No accounts provided for streaming profile creation.")
    if not dates:
        raise ValueError("No dates provided for streaming profile creation.")

    temp_path = output_path.parent / "_temp_filtered.parquet"

    try:
        # Pass 1: Stream filtered data to temp file
        logger.info("  Pass 1: Streaming filtered data to temp parquet...")
        (
            pl.scan_parquet(input_path)
            .filter(pl.col("account_identifier").is_in(accounts) & pl.col("date").is_in(dates))
            .sink_parquet(temp_path)
        )

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            logger.warning("  No data found for selected households/dates in streaming mode")
            return 0

        # Pass 2: Sort, aggregate, and write final output
        logger.info("  Pass 2: Sorting and aggregating to daily profiles...")

        profiles_df = (
            pl.scan_parquet(temp_path)
            .sort(["account_identifier", "datetime"])
            .collect()
            .group_by(["account_identifier", "zip_code", "date"])
            .agg([
                pl.col("kwh").alias("profile"),
                pl.col("is_weekend").first(),
                pl.col("weekday").first(),
                pl.len().alias("num_intervals"),
            ])
            .filter(pl.col("num_intervals") == 48)
            .drop("num_intervals")
        )

        n_profiles = len(profiles_df)
        logger.info(f"  Created {n_profiles:,} complete profiles")

        profiles_df.write_parquet(output_path)
        logger.info(f"  Saved profiles (streaming) to {output_path}")

        return n_profiles

    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================


def prepare_clustering_data(
    input_path: Path,
    output_dir: Path,
    sample_households: int | None = None,
    sample_days: int = 20,
    day_strategy: Literal["stratified", "random"] = "stratified",
    streaming: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Prepare household-level clustering data from interval parquet.

    Args:
        input_path: Path to processed interval parquet file.
        output_dir: Output directory for clustering files.
        sample_households: Number of households to sample (None = all).
        sample_days: Number of days to sample.
        day_strategy: 'stratified' (70% weekdays) or 'random'.
        streaming: If True, use streaming-friendly mode for large samples.
        seed: Random seed for reproducibility.

    Returns:
        Statistics dictionary:
            - n_profiles
            - n_households
            - n_zip4s
            - n_dates
    """
    logger.info("=" * 70)
    logger.info("PREPARING HOUSEHOLD-LEVEL CLUSTERING DATA")
    if streaming:
        logger.info("(STREAMING MODE ENABLED)")
    logger.info("=" * 70)

    # 1. Schema validation (cheap, no data load)
    validation = validate_schema(input_path)
    if not validation["valid"]:
        raise ValueError(f"Input validation failed: {validation['errors']}")

    # 2. Metadata + sampling
    metadata = get_metadata_and_samples(
        input_path=input_path,
        sample_households=sample_households,
        sample_days=sample_days,
        day_strategy=day_strategy,
        seed=seed,
    )

    accounts = metadata["accounts"]
    dates = metadata["dates"]

    # 3. Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = output_dir / "sampled_profiles.parquet"

    # 4. Create profiles (in-memory or streaming)
    if streaming:
        n_profiles = create_household_profiles_streaming(
            input_path=input_path,
            accounts=accounts,
            dates=dates,
            output_path=profiles_path,
        )

        if n_profiles == 0:
            raise ValueError("No profiles created in streaming mode - check input data.")

        profiles_df = pl.read_parquet(profiles_path)
    else:
        profiles_df = create_household_profiles(
            input_path=input_path,
            accounts=accounts,
            dates=dates,
        )

        if profiles_df.is_empty():
            raise ValueError("No profiles created - check input data and sampling settings.")

        (
            profiles_df.select([
                "account_identifier",
                "zip_code",
                "date",
                "profile",
                "is_weekend",
                "weekday",
            ]).write_parquet(profiles_path)
        )
        logger.info(f"  Saved profiles: {profiles_path}")

    # 5. Save household → ZIP+4 mapping
    household_map = profiles_df.select(["account_identifier", "zip_code"]).unique()
    map_path = output_dir / "household_zip4_map.parquet"
    household_map.write_parquet(map_path)
    logger.info(f"  Saved household-ZIP+4 map: {map_path} ({len(household_map):,} households)")

    # 6. Stats
    stats = {
        "n_profiles": len(profiles_df),
        "n_households": profiles_df["account_identifier"].n_unique(),
        "n_zip4s": profiles_df["zip_code"].n_unique(),
        "n_dates": profiles_df["date"].n_unique(),
    }

    logger.info("")
    logger.info("=" * 70)
    logger.info("CLUSTERING DATA READY")
    logger.info("=" * 70)
    logger.info(f"  Profiles: {stats['n_profiles']:,}")
    logger.info(f"  Households: {stats['n_households']:,}")
    logger.info(f"  ZIP+4s represented: {stats['n_zip4s']:,}")
    logger.info(f"  Days: {stats['n_dates']}")
    logger.info(f"  Output: {output_dir}")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare household-level data for clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard run (5000 households, 20 days)
    python prepare_clustering_data_households.py \
        --input data/processed/comed_202308.parquet \
        --output-dir data/clustering \
        --sample-households 5000 \
        --sample-days 20

    # Large dataset with streaming
    python prepare_clustering_data_households.py \
        --input data/processed/comed_202308.parquet \
        --output-dir data/clustering \
        --sample-households 100000 \
        --streaming

    # All households, fewer days
    python prepare_clustering_data_households.py \
        --input data/processed/comed_202308.parquet \
        --output-dir data/clustering \
        --sample-days 10

    # Quick test
    python prepare_clustering_data_households.py \
        --input data/processed/comed_202308.parquet \
        --output-dir data/clustering \
        --sample-households 500 \
        --sample-days 5
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to processed interval parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clustering"),
        help="Output directory (default: data/clustering).",
    )
    parser.add_argument(
        "--sample-households",
        type=int,
        default=None,
        help="Number of households to sample (default: all).",
    )
    parser.add_argument(
        "--sample-days",
        type=int,
        default=20,
        help="Number of days to sample (default: 20).",
    )
    parser.add_argument(
        "--day-strategy",
        choices=["stratified", "random"],
        default="stratified",
        help="Day sampling: stratified (70% weekday) or random.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming-friendly mode for large household samples.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    try:
        prepare_clustering_data(
            input_path=args.input,
            output_dir=args.output_dir,
            sample_households=args.sample_households,
            sample_days=args.sample_days,
            day_strategy=args.day_strategy,
            streaming=args.streaming,
            seed=args.seed,
        )
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        # Re-raise so stack traces are visible in logs when run via a pipeline
        raise


if __name__ == "__main__":
    raise SystemExit(main())
