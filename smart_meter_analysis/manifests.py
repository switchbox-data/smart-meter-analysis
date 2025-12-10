"""
Manifest utilities for memory-safe processing of large parquet files.

These are small parquet files containing unique values extracted from
large source files using streaming (sink_parquet). This avoids OOM when
calling unique().collect() on files with hundreds of millions of rows.

Usage:
    from smart_meter_analysis.manifests import (
        ensure_account_manifest,
        ensure_date_manifest,
        load_account_sample,
        load_date_sample,
    )

    # First call builds manifest via streaming (may take a few minutes)
    # Subsequent calls return instantly from cache
    account_manifest = ensure_account_manifest(
        Path("data/processed/comed_202308.parquet")
    )
    accounts_df = pl.read_parquet(account_manifest)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import polars as pl

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def log_memory(label: str) -> None:
    """Log current RSS memory usage (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    mem_mb = int(line.split()[1]) / 1024
                    logger.debug("[MEMORY] %s: %.0f MB", label, mem_mb)
                    break
    except FileNotFoundError as exc:
        # Likely not on Linux; /proc may not exist
        logger.debug("Skipping memory logging (%s): %s", type(exc).__name__, exc)
    except OSError as exc:
        # Permission issues or other OS-level problems
        logger.debug("Could not read /proc/self/status for memory logging: %s", exc)


def _validate_input_has_columns(input_path: Path, required: list[str]) -> None:
    """Ensure the input parquet exists and has the required columns."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    schema = pl.scan_parquet(input_path).collect_schema()
    missing = [c for c in required if c not in schema.names()]
    if missing:
        raise ValueError(f"Input parquet {input_path} missing required columns: {missing}")


# =============================================================================
# Manifest creation
# =============================================================================


def ensure_account_manifest(input_path: Path) -> Path:
    """
    Create or load account manifest in a memory-safe way.

    The manifest contains one row per account_identifier with an associated
    zip_code. Memory cost scales with the number of accounts, not the number
    of interval-level rows.
    """
    _validate_input_has_columns(input_path, ["account_identifier", "zip_code"])

    manifest_path = input_path.parent / f"{input_path.stem}_accounts.parquet"

    # Check for existing valid manifest
    if manifest_path.exists():
        try:
            n = pl.scan_parquet(manifest_path).select(pl.len()).collect()[0, 0]
            if n > 0:
                logger.info(
                    "Using existing account manifest: %s (%s accounts)",
                    manifest_path,
                    f"{n:,}",
                )
                return manifest_path
        except Exception:
            logger.warning(
                "Account manifest is corrupt or unreadable, rebuilding: %s",
                manifest_path,
            )

    # Build manifest using streaming-friendly group_by
    logger.info("Building account manifest from %s (streaming group_by)...", input_path)
    log_memory("before account manifest")

    lf = pl.scan_parquet(input_path)

    # One row per account_identifier, keep a representative zip_code.
    # group_by + agg with collect(streaming=True) is streaming-safe; we only
    # hold ~#accounts rows in memory.
    accounts_df = (
        lf.select(["account_identifier", "zip_code"])
        .filter(pl.col("account_identifier").is_not_null())
        .group_by("account_identifier")
        .agg(pl.first("zip_code").alias("zip_code"))
        .collect(streaming=True)
        .sort("account_identifier")
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    accounts_df.write_parquet(manifest_path)

    log_memory("after account manifest")

    n = accounts_df.height
    logger.info(
        "Account manifest complete (rebuilt): %s (%s accounts)",
        manifest_path,
        f"{n:,}",
    )

    return manifest_path


def ensure_date_manifest(input_path: Path) -> Path:
    """
    Create or load date manifest in a memory-safe way.

    The manifest contains one row per date with representative weekend/weekday
    flags. This is small in practice (~31 rows for one month), but we still
    build it via streaming group_by for consistency.
    """
    _validate_input_has_columns(input_path, ["date", "is_weekend", "weekday"])

    manifest_path = input_path.parent / f"{input_path.stem}_dates.parquet"

    # Check for existing valid manifest
    if manifest_path.exists():
        try:
            n = pl.scan_parquet(manifest_path).select(pl.len()).collect()[0, 0]
            if n > 0:
                logger.info(
                    "Using existing date manifest: %s (%s dates)",
                    manifest_path,
                    f"{n:,}",
                )
                return manifest_path
        except Exception:
            logger.warning(
                "Date manifest is corrupt or unreadable, rebuilding: %s",
                manifest_path,
            )

    # Build manifest using streaming-friendly group_by
    logger.info("Building date manifest from %s (streaming group_by)...", input_path)
    log_memory("before date manifest")

    lf = pl.scan_parquet(input_path)

    dates_df = (
        lf.select(["date", "is_weekend", "weekday"])
        .filter(pl.col("date").is_not_null())
        .group_by("date")
        .agg(
            pl.first("is_weekend").alias("is_weekend"),
            pl.first("weekday").alias("weekday"),
        )
        .collect(streaming=True)
        .sort("date")
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    dates_df.write_parquet(manifest_path)

    log_memory("after date manifest")

    n = dates_df.height
    logger.info(
        "Date manifest complete (rebuilt): %s (%s dates)",
        manifest_path,
        f"{n:,}",
    )

    return manifest_path


# =============================================================================
# Sampling helpers
# =============================================================================


def load_account_sample(
    manifest_path: Path,
    sample_n: int | None = None,
    seed: int = 42,
) -> list[str]:
    """
    Load account identifiers from manifest, optionally sampling.

    Args:
        manifest_path:
            Path to account manifest parquet.
        sample_n:
            Number of accounts to sample (None = all).
        seed:
            Random seed for reproducible sampling.

    Returns:
        List of account_identifier strings.
    """
    df = pl.read_parquet(manifest_path)

    if df.is_empty():
        logger.warning("Account manifest %s is empty; returning no accounts", manifest_path)
        return []

    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, shuffle=True, seed=seed)
        logger.info("Sampled %s accounts from manifest", f"{len(df):,}")
    else:
        logger.info("Using all %s accounts from manifest", f"{len(df):,}")

    return df["account_identifier"].to_list()


def load_date_sample(
    manifest_path: Path,
    sample_n: int,
    strategy: Literal["stratified", "random"] = "stratified",
    seed: int = 42,
) -> list[Any]:
    """
    Load dates from manifest with stratified or random sampling.

    Args:
        manifest_path:
            Path to date manifest parquet.
        sample_n:
            Number of dates to sample.
        strategy:
            "stratified" (70% weekday, 30% weekend) or "random".
        seed:
            Random seed for reproducible sampling.

    Returns:
        List of date values (type depends on input schema).
    """
    df = pl.read_parquet(manifest_path)

    if df.is_empty():
        logger.warning("Date manifest %s is empty; returning no dates", manifest_path)
        return []

    if sample_n <= 0:
        logger.info("Requested sample_n <= 0; returning empty date list")
        return []

    # Stratified sampling if both weekday and weekend exist
    if strategy == "stratified":
        weekday_df = df.filter(~pl.col("is_weekend"))
        weekend_df = df.filter(pl.col("is_weekend"))

        if weekday_df.height == 0 or weekend_df.height == 0:
            logger.warning("Missing weekdays or weekends in manifest; falling back to random sampling")
            strategy = "random"

    if strategy == "stratified":
        n_weekdays = int(sample_n * 0.7)
        n_weekends = sample_n - n_weekdays

        n_weekdays = min(n_weekdays, len(weekday_df))
        n_weekends = min(n_weekends, len(weekend_df))

        sampled_weekdays = (
            weekday_df.sample(n=n_weekdays, shuffle=True, seed=seed)["date"].to_list() if n_weekdays > 0 else []
        )
        sampled_weekends = (
            weekend_df.sample(
                n=n_weekends,
                shuffle=True,
                seed=seed + 1,
            )["date"].to_list()
            if n_weekends > 0
            else []
        )

        dates = sampled_weekdays + sampled_weekends
        logger.info(
            "Sampled %d weekdays + %d weekend days (stratified)",
            len(sampled_weekdays),
            len(sampled_weekends),
        )
    else:
        n_sample = min(sample_n, len(df))
        dates = df.sample(n=n_sample, shuffle=True, seed=seed)["date"].to_list()
        logger.info("Sampled %d dates (random)", len(dates))

    return dates
