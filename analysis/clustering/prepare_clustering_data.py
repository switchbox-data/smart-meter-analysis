"""
Phase 1: Prepare data for DTW clustering analysis.

Transforms interval-level energy data into daily load profiles suitable for
DTW k-means clustering. Uses disk-based chunking and lazy evaluation to handle
large datasets without excessive memory consumption.

Pipeline:
    1. Load processed interval data (lazy scan)
    2. Aggregate customer usage to ZIP+4 level (chunked, written to disk)
    3. Create daily 48-point load profiles
    4. Sample representative days and ZIP codes (lazy)
    5. Output profiles ready for DTW clustering

Usage:
    python prepare_clustering_data.py \\
        --input data/processed/comed_202308.parquet \\
        --output-dir data/clustering \\
        --sample-days 20 \\
        --sample-zips 500
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import Literal

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA INSPECTION
# =============================================================================


def get_data_summary(path: Path) -> dict:
    """
    Get summary statistics from parquet file without loading into memory.

    Args:
        path: Path to parquet file

    Returns:
        Dictionary with row count, unique counts, and date range
    """
    lf = pl.scan_parquet(path)

    summary = lf.select([
        pl.len().alias("n_rows"),
        pl.col("zip_code").n_unique().alias("n_zip_codes"),
        pl.col("account_identifier").n_unique().alias("n_accounts"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
    ]).collect()

    return summary.to_dicts()[0]


def validate_input_data(path: Path) -> dict:
    """
    Validate input data has required columns for clustering.

    Args:
        path: Path to input parquet file

    Returns:
        Validation results dictionary
    """
    logger.info("Validating input data...")

    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()

    errors = []
    warnings = []

    # Required columns for clustering
    required_energy = ["zip_code", "account_identifier", "datetime", "kwh"]
    required_time = ["date", "hour", "is_weekend", "weekday"]

    missing_energy = [c for c in required_energy if c not in schema.names()]
    missing_time = [c for c in required_time if c not in schema.names()]

    if missing_energy:
        errors.append(f"Missing energy columns: {missing_energy}")
    if missing_time:
        errors.append(f"Missing time columns: {missing_time}")

    # Data quality check
    sample_stats = (
        lf.select([
            pl.col("kwh").min().alias("kwh_min"),
            pl.col("kwh").max().alias("kwh_max"),
            pl.col("kwh").null_count().alias("kwh_nulls"),
            pl.len().alias("total_rows"),
        ])
        .collect()
        .to_dicts()[0]
    )

    if sample_stats["kwh_min"] is not None and sample_stats["kwh_min"] < 0:
        warnings.append(f"Negative kWh values detected: min={sample_stats['kwh_min']:.4f}")

    if sample_stats["kwh_nulls"] > 0:
        null_pct = sample_stats["kwh_nulls"] / sample_stats["total_rows"] * 100
        if null_pct > 5:
            errors.append(f"High null rate in kWh: {null_pct:.1f}%")
        else:
            warnings.append(f"Some null kWh values: {null_pct:.2f}%")

    status = "PASS" if not errors else "FAIL"

    print(f"\n{'=' * 60}")
    print("INPUT DATA VALIDATION")
    print("=" * 60)
    if errors:
        print("\n❌ FAILED:")
        for e in errors:
            print(f"   • {e}")
    else:
        print("\n✅ PASSED")
    if warnings:
        print("\n⚠️  Warnings:")
        for w in warnings:
            print(f"   • {w}")

    return {"status": status, "errors": errors, "warnings": warnings}


# =============================================================================
# ZIP CODE HANDLING
# =============================================================================


def get_zip_codes(input_path: Path, sample_n: int | None = None, seed: int = 42) -> list[str]:
    """
    Get list of unique ZIP+4 codes from input data.

    Args:
        input_path: Path to processed parquet file
        sample_n: If provided, randomly sample this many ZIP codes
        seed: Random seed for reproducibility

    Returns:
        List of ZIP+4 code strings
    """
    logger.info("Identifying ZIP codes...")

    all_zips = pl.scan_parquet(input_path).select("zip_code").unique().collect()["zip_code"].to_list()

    logger.info(f"  Found {len(all_zips):,} unique ZIP+4 codes")

    if sample_n is not None and sample_n < len(all_zips):
        random.seed(seed)
        all_zips = random.sample(all_zips, sample_n)
        logger.info(f"  Sampled to {len(all_zips):,} ZIP+4 codes")

    return all_zips


# =============================================================================
# PROFILE CREATION
# =============================================================================


def process_chunk_to_profiles(
    input_path: Path,
    zip_codes: list[str],
) -> pl.DataFrame:
    """
    Process a chunk of ZIP codes into daily profiles.

    Aggregates interval data to ZIP+4 level, then creates daily 48-point
    profiles. Only keeps complete profiles (exactly 48 intervals).

    Args:
        input_path: Path to processed parquet file
        zip_codes: List of ZIP codes to process in this chunk

    Returns:
        DataFrame with daily profiles for this chunk
    """
    # Load only this chunk's data
    chunk_df = pl.scan_parquet(input_path).filter(pl.col("zip_code").is_in(zip_codes)).collect()

    # Aggregate to ZIP+4 level (sum across all accounts in each ZIP+4)
    zip4_df = (
        chunk_df.group_by(["zip_code", "datetime", "date", "hour"])
        .agg([
            pl.col("kwh").sum().alias("kwh"),
            pl.col("is_weekend").first(),
            pl.col("weekday").first(),
        ])
        .sort(["zip_code", "datetime"])
    )

    # Create daily profiles (list of 48 kWh values per day)
    profiles_df = zip4_df.group_by(["zip_code", "date"]).agg([
        pl.col("kwh").alias("profile"),
        pl.col("is_weekend").first(),
        pl.col("weekday").first(),
        pl.len().alias("num_intervals"),
    ])

    # Keep only complete 48-interval profiles
    # This excludes DST transition days (46 or 50 intervals)
    profiles_df = profiles_df.filter(pl.col("num_intervals") == 48)

    return profiles_df


def process_all_chunks_to_disk(
    input_path: Path,
    zip_codes: list[str],
    tmp_dir: Path,
    chunk_size: int = 50,
) -> int:
    """
    Process all ZIP codes in chunks, writing each chunk to disk.
    Only one chunk is in memory at a time, and results are written to disk
    immediately.

    Args:
        input_path: Path to processed parquet file
        zip_codes: List of all ZIP codes to process
        tmp_dir: Directory to write temporary chunk files
        chunk_size: Number of ZIP codes per chunk

    Returns:
        Number of chunks written
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = (len(zip_codes) + chunk_size - 1) // chunk_size
    logger.info(f"Processing {len(zip_codes):,} ZIP codes in {n_chunks} chunks of {chunk_size}...")

    for i in range(0, len(zip_codes), chunk_size):
        chunk_zips = zip_codes[i : i + chunk_size]
        chunk_num = i // chunk_size + 1

        logger.info(f"  Chunk {chunk_num}/{n_chunks} ({len(chunk_zips)} ZIP codes)")

        # Process this chunk
        profiles_df = process_chunk_to_profiles(input_path, chunk_zips)

        # Write to disk immediately
        chunk_path = tmp_dir / f"profiles_chunk_{chunk_num:04d}.parquet"
        profiles_df.write_parquet(chunk_path)

        # Free memory
        del profiles_df

    return n_chunks


# =============================================================================
# DAY SAMPLING
# =============================================================================


def sample_days_stratified(
    profiles_lf: pl.LazyFrame,
    n_days: int,
    weekday_ratio: float = 0.7,
    seed: int = 42,
) -> list:
    """
    Sample days with stratified weekday/weekend ratio.

    Args:
        profiles_lf: Lazy frame of profiles
        n_days: Total number of days to sample
        weekday_ratio: Proportion of weekdays (default: 70%)
        seed: Random seed

    Returns:
        List of sampled dates
    """
    # Collect only unique dates (small)
    unique_dates = profiles_lf.select(["date", "is_weekend"]).unique().collect()

    n_weekdays = int(n_days * weekday_ratio)
    n_weekends = n_days - n_weekdays

    weekday_dates = unique_dates.filter(~pl.col("is_weekend"))["date"]
    weekend_dates = unique_dates.filter(pl.col("is_weekend"))["date"]

    # Don't sample more than available
    n_weekdays = min(n_weekdays, len(weekday_dates))
    n_weekends = min(n_weekends, len(weekend_dates))

    sampled = pl.concat([
        weekday_dates.sample(n=n_weekdays, seed=seed),
        weekend_dates.sample(n=n_weekends, seed=seed + 1),
    ])

    return sampled.to_list()


def sample_days_random(
    profiles_lf: pl.LazyFrame,
    n_days: int,
    seed: int = 42,
) -> list:
    """
    Sample days randomly without stratification.

    Args:
        profiles_lf: Lazy frame of profiles
        n_days: Number of days to sample
        seed: Random seed

    Returns:
        List of sampled dates
    """
    unique_dates = profiles_lf.select("date").unique().collect()["date"]

    n_days = min(n_days, len(unique_dates))
    sampled = unique_dates.sample(n=n_days, seed=seed)

    return sampled.to_list()


# =============================================================================
# OUTPUT
# =============================================================================


def write_final_outputs(
    profiles_lf: pl.LazyFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Write final profile and demographics files.

    Args:
        profiles_lf: Lazy frame of profiles to write
        output_dir: Output directory

    Returns:
        Tuple of (profiles_path, demographics_path)
    """
    profiles_path = output_dir / "sampled_profiles.parquet"
    demographics_path = output_dir / "zip4_demographics.parquet"

    # Stream profiles to disk
    profiles_lf.sink_parquet(profiles_path)

    # Create demographics placeholder
    demographics_df = pl.scan_parquet(profiles_path).select("zip_code").unique().collect()
    demographics_df.write_parquet(demographics_path)

    return profiles_path, demographics_path


def get_output_stats(profiles_path: Path) -> dict:
    """
    Get statistics about the output profiles.

    Args:
        profiles_path: Path to profiles parquet

    Returns:
        Dictionary with profile counts
    """
    stats = (
        pl.scan_parquet(profiles_path)
        .select([
            pl.len().alias("n_profiles"),
            pl.col("zip_code").n_unique().alias("n_zips"),
            pl.col("date").n_unique().alias("n_dates"),
        ])
        .collect()
        .to_dicts()[0]
    )
    return stats


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================


def create_daily_profiles(
    input_path: Path,
    output_dir: Path,
    sample_days: int | None = None,
    sample_zips: int | None = None,
    day_strategy: Literal["random", "stratified"] = "stratified",
    chunk_size: int = 50,
    cleanup_temp: bool = True,
) -> tuple[Path, Path]:
    """
    Create daily load profiles from interval data.

    Main orchestration function that coordinates the chunked processing
    pipeline. Uses disk-based chunking to handle large datasets.

    Args:
        input_path: Path to processed parquet file
        output_dir: Directory for output files
        sample_days: Number of days to sample (None = all)
        sample_zips: Number of ZIP codes to sample (None = all)
        day_strategy: 'stratified' (70/30 weekday/weekend) or 'random'
        chunk_size: Number of ZIP codes per chunk
        cleanup_temp: Delete temporary files after combining

    Returns:
        Tuple of (profiles_path, demographics_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_profiles_chunks"

    # -------------------------------------------------------------------------
    # Step 1: Get ZIP codes
    # -------------------------------------------------------------------------
    logger.info("Step 1: Getting ZIP codes...")
    zip_codes = get_zip_codes(input_path, sample_n=sample_zips)

    # -------------------------------------------------------------------------
    # Step 2: Process chunks to disk
    # -------------------------------------------------------------------------
    logger.info("Step 2: Processing chunks to disk...")
    process_all_chunks_to_disk(
        input_path=input_path,
        zip_codes=zip_codes,
        tmp_dir=tmp_dir,
        chunk_size=chunk_size,
    )

    # -------------------------------------------------------------------------
    # Step 3: Combine chunks lazily
    # -------------------------------------------------------------------------
    logger.info("Step 3: Combining chunks lazily...")
    profiles_lf = pl.scan_parquet(str(tmp_dir / "profiles_chunk_*.parquet"))

    # -------------------------------------------------------------------------
    # Step 4: Sample days (optional)
    # -------------------------------------------------------------------------
    if sample_days is not None:
        logger.info(f"Step 4: Sampling {sample_days} days ({day_strategy})...")

        # Check available days
        n_available = profiles_lf.select("date").unique().collect().height

        if sample_days >= n_available:
            logger.warning(f"  Requested {sample_days} days but only {n_available} available - using all")
        else:
            if day_strategy == "stratified":
                sampled_dates = sample_days_stratified(profiles_lf, sample_days)
            else:
                sampled_dates = sample_days_random(profiles_lf, sample_days)

            profiles_lf = profiles_lf.filter(pl.col("date").is_in(sampled_dates))
            logger.info(f"  Sampled to {len(sampled_dates)} days")
    else:
        logger.info("Step 4: Keeping all days (no sampling)")

    # -------------------------------------------------------------------------
    # Step 5: Write outputs
    # -------------------------------------------------------------------------
    logger.info("Step 5: Writing outputs...")
    profiles_path, demographics_path = write_final_outputs(profiles_lf, output_dir)

    # Get and log stats
    stats = get_output_stats(profiles_path)
    logger.info(f"  ✓ Profiles: {profiles_path}")
    logger.info(f"    {stats['n_profiles']:,} profiles ({stats['n_zips']} ZIP+4s × {stats['n_dates']} days)")
    logger.info(f"  ✓ Demographics: {demographics_path}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    if cleanup_temp:
        logger.info("  Cleaning up temporary files...")
        shutil.rmtree(tmp_dir)
    else:
        logger.info(f"  Temporary chunks retained: {tmp_dir}")

    return profiles_path, demographics_path


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare data for DTW clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (all ZIP codes, all days)
    python prepare_clustering_data.py \\
        --input data/processed/comed_202308.parquet

    # With sampling (recommended for large datasets)
    python prepare_clustering_data.py \\
        --input data/processed/comed_202308.parquet \\
        --sample-days 20 --sample-zips 500

    # Fast test run
    python prepare_clustering_data.py \\
        --input data/processed/comed_202308.parquet \\
        --sample-days 10 --sample-zips 100

    # Keep temporary files for debugging
    python prepare_clustering_data.py \\
        --input data/processed/comed_202308.parquet \\
        --keep-temp
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to processed parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clustering"),
        help="Output directory (default: data/clustering)",
    )

    # Sampling options
    sampling = parser.add_argument_group("Sampling Options")
    sampling.add_argument(
        "--sample-days",
        type=int,
        default=None,
        help="Number of days to sample (default: all)",
    )
    sampling.add_argument(
        "--sample-zips",
        type=int,
        default=None,
        help="Number of ZIP+4 codes to sample (default: all)",
    )
    sampling.add_argument(
        "--day-strategy",
        choices=["random", "stratified"],
        default="stratified",
        help="Day sampling: 'stratified' = 70%% weekday/30%% weekend (default)",
    )

    # Performance options
    performance = parser.add_argument_group("Performance Options")
    performance.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="ZIP codes per chunk (default: 50, lower = less memory)",
    )
    performance.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary chunk files for debugging",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("PHASE 1: DATA PREPARATION FOR DTW CLUSTERING")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Validate input
    # -------------------------------------------------------------------------
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return

    # Get data summary
    logger.info(f"\nInput: {args.input}")
    summary = get_data_summary(args.input)
    logger.info(f"  Rows: {summary['n_rows']:,}")
    logger.info(f"  ZIP+4 codes: {summary['n_zip_codes']:,}")
    logger.info(f"  Accounts: {summary['n_accounts']:,}")
    logger.info(f"  Date range: {summary['min_date']} to {summary['max_date']}")

    # Validate
    validation = validate_input_data(args.input)
    if validation["status"] == "FAIL":
        logger.error("Input validation failed - aborting")
        return

    # -------------------------------------------------------------------------
    # Run pipeline
    # -------------------------------------------------------------------------
    print()  # Blank line before steps

    profiles_path, demographics_path = create_daily_profiles(
        input_path=args.input,
        output_dir=args.output_dir,
        sample_days=args.sample_days,
        sample_zips=args.sample_zips,
        day_strategy=args.day_strategy,
        chunk_size=args.chunk_size,
        cleanup_temp=not args.keep_temp,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print("\nOutputs:")
    print(f"  Profiles:     {profiles_path}")
    print(f"  Demographics: {demographics_path}")
    print("\nNext: Run DTW clustering")
    print("  python analysis/clustering/dtw_clustering.py \\")
    print(f"      --input {profiles_path} \\")
    print(f"      --output-dir {args.output_dir}/results \\")
    print("      --k-range 3 6 --find-optimal-k --normalize")
    print("=" * 70)


if __name__ == "__main__":
    main()
