#!/usr/bin/env python3
"""
Step 1, Part 1 of the ComEd Smart Meter Clustering Pipeline: Prepare household-level
data for clustering analysis.

Transforms interval-level energy data into daily load profiles at the
HOUSEHOLD (account) level for k-means clustering.

What this does:
    1. Validate schema (no full data scan)
    2. Build/load manifests for accounts and dates (streaming, memory-safe)
    3. Sample households + dates from manifests
    4. Create daily 48-point load profiles per household
    5. Output profiles ready for clustering

Design notes:
    - Uses MANIFESTS to avoid OOM on unique().collect() for large files
    - Manifests are built once via sink_parquet (streaming) and cached
    - Standard mode: single filtered collect() for selected households/dates
      * good for up to ~5k-10k households sampled (depending on memory)
    - Chunked streaming mode: fully streaming pipeline with per-chunk sinks
      * required for 10k+ households on constrained hardware
    - Profiles are 48 half-hourly kWh values in chronological order (00:30-24:00)
    - Incomplete days (not 48 intervals) are dropped as “missing/irregular data”

Output files:
    - sampled_profiles.parquet:
        One row per (account_identifier, date) with 'profile' = list[float] of length 48
    - household_zip4_map.parquet:
        Unique account_identifier → zip_code mapping for later joins

Manifest files (auto-generated alongside input, via smart_meter_analysis.manifests):
    - {input_stem}_accounts.parquet: unique (account_identifier, zip_code) pairs
    - {input_stem}_dates.parquet: unique (date, is_weekend, weekday) tuples

Usage:
    # Standard (5,000 households, 20 days)
    python prepare_clustering_data_households.py \
        --input data/processed/comed_202308.parquet \
        --output-dir data/clustering \
        --sample-households 5000 \
        --sample-days 20

    # Large dataset with chunked streaming (20,000 households)
    python prepare_clustering_data_households.py \
        --input data/processed/comed_202308.parquet \
        --output-dir data/clustering \
        --sample-households 20000 \
        --sample-days 31 \
        --streaming \
        --chunk-size 2000
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Any, Literal

import polars as pl

from smart_meter_analysis.manifests import (
    ensure_account_manifest,
    ensure_date_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Required columns for household-level clustering
REQUIRED_ENERGY_COLS = ["zip_code", "account_identifier", "datetime", "kwh"]
REQUIRED_TIME_COLS = ["date", "hour", "is_weekend", "weekday"]


# =============================================================================
# MEMORY INSTRUMENTATION
# =============================================================================


def log_memory(label: str) -> None:
    """Log current RSS memory usage (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    mem_mb = int(line.split()[1]) / 1024
                    logger.info("[MEMORY] %s: %.0f MB", label, mem_mb)
                    break
    except Exception as exc:
        # Best-effort only; ignore on non-Linux or restricted environments
        logger.debug("Skipping memory log for %s: %s", label, exc)


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
    Get summary statistics and sample households + dates using MANIFESTS.

    This function:
    - Computes summary stats via streaming (safe for large files)
    - Builds/loads account and date manifests (streaming sink, memory-safe)
    - Samples households and dates from the small manifest files

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
    log_memory("start of get_metadata_and_samples")

    lf = pl.scan_parquet(input_path)

    # Summary stats (streaming-safe: single row output)
    logger.info("  Computing summary statistics...")
    summary_df = lf.select([
        pl.len().alias("n_rows"),
        pl.col("zip_code").n_unique().alias("n_zip_codes"),
        pl.col("account_identifier").n_unique().alias("n_accounts"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
    ]).collect(engine="streaming")
    summary = summary_df.to_dicts()[0]

    # Early checks
    if summary["n_rows"] == 0:
        raise ValueError(f"Input file {input_path} contains no rows.")

    logger.info("  %s rows", f"{summary['n_rows']:,}")
    logger.info("  %s households", f"{summary['n_accounts']:,}")
    logger.info("  %s ZIP+4 codes", f"{summary['n_zip_codes']:,}")
    logger.info("  Date range: %s to %s", summary["min_date"], summary["max_date"])

    # ==========================================================================
    # KEY CHANGE: Use manifests instead of unique().collect()
    # ==========================================================================

    logger.info("  Loading manifests...")
    account_manifest = ensure_account_manifest(input_path)
    date_manifest = ensure_date_manifest(input_path)

    # Read from small manifest files (fits easily in memory)
    accounts_df = pl.read_parquet(account_manifest)
    dates_df = pl.read_parquet(date_manifest)

    log_memory("after loading manifests")

    if accounts_df.height == 0:
        raise ValueError("No account_identifier values found in manifest.")
    if dates_df.height == 0:
        raise ValueError("No dates found in manifest.")

    # Sample households
    if sample_households is not None and sample_households < len(accounts_df):
        accounts_df = accounts_df.sample(n=sample_households, shuffle=True, seed=seed)
        logger.info("  Sampled %s households", f"{len(accounts_df):,}")
    else:
        logger.info("  Using all %s households", f"{len(accounts_df):,}")

    accounts = accounts_df["account_identifier"].to_list()

    # Sample dates
    if day_strategy == "stratified":
        weekday_df = dates_df.filter(~pl.col("is_weekend"))
        weekend_df = dates_df.filter(pl.col("is_weekend"))

        if weekday_df.height == 0 or weekend_df.height == 0:
            logger.warning("  Missing weekdays or weekends; falling back to random day sampling.")
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
        logger.info(
            "  Sampled %d weekdays + %d weekend days (stratified)",
            len(sampled_weekdays),
            len(sampled_weekends),
        )
    else:
        n_sample = min(sample_days, len(dates_df))
        dates = dates_df.sample(n=n_sample, shuffle=True, seed=seed)["date"].to_list()
        logger.info("  Sampled %d days (random)", len(dates))

    if not dates:
        raise ValueError("No dates were sampled; check input data and sampling settings.")

    log_memory("end of get_metadata_and_samples")

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
    when sampling a moderate subset of households and dates.

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
    logger.info(
        "Creating profiles for %s households x %d days...",
        f"{len(accounts):,}",
        len(dates),
    )
    log_memory("start of create_household_profiles")

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

    logger.info("  Loaded %s interval records", f"{len(df):,}")
    log_memory("after loading filtered intervals")

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
        logger.info(
            "  Dropped %s incomplete profiles (missing or irregular data)",
            f"{n_dropped:,}",
        )

    logger.info("  Created %s complete profiles", f"{len(profiles_df):,}")
    log_memory("end of create_household_profiles")

    return profiles_df.drop("num_intervals")


def _create_profiles_for_chunk_streaming(
    input_path: Path,
    accounts_chunk: list[str],
    dates: list[Any],
    chunk_idx: int,
    total_chunks: int,
    tmp_dir: Path,
) -> Path:
    """
    Create profiles for a single chunk of households and write directly to parquet.

    Uses within-group sort (struct.sort_by) to preserve chronological order
    without a global sort, which keeps the lazy plan streaming-compatible.

    Returns:
        Path to the chunk parquet file.
    """
    logger.info(
        "  Chunk %d/%d: %s households...",
        chunk_idx + 1,
        total_chunks,
        f"{len(accounts_chunk):,}",
    )
    log_memory(f"chunk {chunk_idx + 1} start")

    chunk_path = tmp_dir / f"sampled_profiles_chunk_{chunk_idx:03d}.parquet"

    (
        pl.scan_parquet(input_path)
        .filter(pl.col("account_identifier").is_in(accounts_chunk) & pl.col("date").is_in(dates))
        .group_by(["account_identifier", "zip_code", "date"])
        .agg([
            # Sort by datetime within group, then extract kwh values
            pl.struct(["datetime", "kwh"]).sort_by("datetime").struct.field("kwh").alias("profile"),
            pl.col("is_weekend").first(),
            pl.col("weekday").first(),
            pl.len().alias("num_intervals"),
        ])
        .filter(pl.col("num_intervals") == 48)
        .drop("num_intervals")
        .sink_parquet(chunk_path)
    )

    log_memory(f"chunk {chunk_idx + 1} done")
    logger.info("    Wrote chunk parquet: %s", chunk_path)

    return chunk_path


def create_household_profiles_chunked_streaming(
    input_path: Path,
    accounts: list[str],
    dates: list[Any],
    output_path: Path,
    chunk_size: int = 5000,
) -> int:
    """
    Create daily load profiles using chunked streaming.

    - Splits households into chunks.
    - For each chunk, runs a streaming filter → group_by → aggregate → sink.
    - Concatenates all chunk files using a streaming concat.
    - Deletes temporary chunk files.

    This avoids ever materializing profiles for all households in memory.

    Args:
        input_path: Path to interval-level parquet file.
        accounts: List of account_identifier values to include.
        dates: List of dates to include.
        output_path: Final output parquet path for all profiles.
        chunk_size: Number of households per chunk.

    Returns:
        Number of complete daily profiles created.
    """
    if not accounts:
        raise ValueError(
            "No accounts provided for chunked streaming profile creation.",
        )
    if not dates:
        raise ValueError("No dates provided for chunked streaming profile creation.")

    n_accounts = len(accounts)
    n_chunks = (n_accounts + chunk_size - 1) // chunk_size

    logger.info(
        "Creating profiles in %d chunks of up to %s households each (total: %s households x %d days)...",
        n_chunks,
        f"{chunk_size:,}",
        f"{n_accounts:,}",
        len(dates),
    )
    log_memory("before chunked streaming")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_path.parent

    chunk_paths: list[Path] = []

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_accounts)
        accounts_chunk = accounts[start_idx:end_idx]

        chunk_path = _create_profiles_for_chunk_streaming(
            input_path=input_path,
            accounts_chunk=accounts_chunk,
            dates=dates,
            chunk_idx=i,
            total_chunks=n_chunks,
            tmp_dir=tmp_dir,
        )

        # Only keep non-empty chunk files
        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            chunk_paths.append(chunk_path)

        gc.collect()

    if not chunk_paths:
        logger.warning("No profiles created in chunked streaming mode!")
        return 0

    # Stream-concatenate all chunk parquet files
    logger.info("Combining %d chunk files into %s", len(chunk_paths), output_path)
    log_memory("before streaming concat")

    combined_lf = pl.concat([pl.scan_parquet(p) for p in chunk_paths])
    combined_lf.sink_parquet(output_path)

    log_memory("after streaming concat")

    # Count rows in final output
    n_profiles = pl.scan_parquet(output_path).select(pl.len()).collect()[0, 0]
    logger.info("  Created %s complete profiles (chunked streaming)", f"{n_profiles:,}")
    logger.info("  Saved to %s", output_path)

    # Clean up temporary chunk files
    for p in chunk_paths:
        try:
            p.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to delete temp chunk file %s: %s", p, exc)

    return int(n_profiles)


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
    chunk_size: int = 5000,
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
        streaming: If True, use chunked streaming mode for large samples.
        chunk_size: Households per chunk when streaming is enabled.
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
        logger.info("(STREAMING MODE ENABLED, chunk_size=%d)", chunk_size)
    logger.info("=" * 70)
    log_memory("start of prepare_clustering_data")

    # 1. Schema validation (cheap, no data load)
    validation = validate_schema(input_path)
    if not validation["valid"]:
        raise ValueError(f"Input validation failed: {validation['errors']}")

    # 2. Metadata + sampling (uses manifests for memory safety)
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

    # 4. Create profiles (in-memory or chunked streaming)
    if streaming:
        n_profiles = create_household_profiles_chunked_streaming(
            input_path=input_path,
            accounts=accounts,
            dates=dates,
            output_path=profiles_path,
            chunk_size=chunk_size,
        )

        if n_profiles == 0:
            raise ValueError(
                "No profiles created in chunked streaming mode - check input data and sampling settings.",
            )

        profiles_df = pl.read_parquet(profiles_path)
    else:
        profiles_df = create_household_profiles(
            input_path=input_path,
            accounts=accounts,
            dates=dates,
        )

        if profiles_df.is_empty():
            raise ValueError(
                "No profiles created - check input data and sampling settings.",
            )

        profiles_df.select([
            "account_identifier",
            "zip_code",
            "date",
            "profile",
            "is_weekend",
            "weekday",
        ]).write_parquet(profiles_path)
        logger.info("  Saved profiles: %s", profiles_path)

    # 5. Save household → ZIP+4 mapping
    household_map = profiles_df.select(["account_identifier", "zip_code"]).unique()
    map_path = output_dir / "household_zip4_map.parquet"
    household_map.write_parquet(map_path)
    logger.info(
        "  Saved household-ZIP+4 map: %s (%s households)",
        map_path,
        f"{len(household_map):,}",
    )

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
    logger.info("  Profiles: %s", f"{stats['n_profiles']:,}")
    logger.info("  Households: %s", f"{stats['n_households']:,}")
    logger.info("  ZIP+4s represented: %s", f"{stats['n_zip4s']:,}")
    logger.info("  Days: %d", stats["n_dates"])
    logger.info("  Output: %s", output_dir)
    log_memory("end of prepare_clustering_data")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare household-level data for clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard run (5,000 households, 20 days)
    python prepare_clustering_data_households.py \\
        --input data/processed/comed_202308.parquet \\
        --output-dir data/clustering \\
        --sample-households 5000 \\
        --sample-days 20

    # Large dataset with chunked streaming
    python prepare_clustering_data_households.py \\
        --input data/processed/comed_202308.parquet \\
        --output-dir data/clustering \\
        --sample-households 20000 \\
        --sample-days 31 \\
        --streaming \\
        --chunk-size 2000

    # All households, fewer days
    python prepare_clustering_data_households.py \\
        --input data/processed/comed_202308.parquet \\
        --output-dir data/clustering \\
        --sample-days 10

    # Quick test
    python prepare_clustering_data_households.py \\
        --input data/processed/comed_202308.parquet \\
        --output-dir data/clustering \\
        --sample-households 500 \\
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
        help="Use chunked streaming mode for large household samples (10k+).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Households per chunk when --streaming is enabled (default: 5000).",
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    try:
        prepare_clustering_data(
            input_path=args.input,
            output_dir=args.output_dir,
            sample_households=args.sample_households,
            sample_days=args.sample_days,
            day_strategy=args.day_strategy,
            streaming=args.streaming,
            chunk_size=args.chunk_size,
            seed=args.seed,
        )
        return 0
    except Exception as exc:
        logger.error("Failed: %s", exc)
        # Re-raise so stack traces are visible in logs when run via a pipeline
        raise


if __name__ == "__main__":
    raise SystemExit(main())
