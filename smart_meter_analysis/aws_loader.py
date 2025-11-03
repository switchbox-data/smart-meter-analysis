# smart_meter_analysis/aws_loader.py
"""
AWS S3 utilities for batch processing ComEd smart meter data.

Key improvements:
- Uses lazy semantics (scan_csv instead of read_csv)
- Hardcoded schema to avoid inference overhead
- Direct S3 path scanning (no manual get_object calls)
- Deferred sorting until after concatenation
- Minimizes materialization points
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# S3 Configuration
S3_BUCKET = "smart-meter-data-sb"
S3_PREFIX = "sharepoint-files/Zip4/"

# Error messages (for TRY003)
ERR_NO_FILES_FOUND = "No files found for month: {}"
ERR_NO_SUCCESSFUL_PROCESS = "No files were successfully processed"

# Hardcoded schema overrides for ComEd smart meter data
# We hardcode the ID columns to avoid inference overhead
# Interval columns (INTERVAL_HR0000, HR0030, etc.) are inferred on first scan
# This is a good compromise: fast inference for varying interval columns,
# no inference needed for consistent ID columns
COMED_SCHEMA_OVERRIDES = {
    # ID columns (always present, always these types)
    "ZIP_CODE": pl.Utf8,
    "DELIVERY_SERVICE_CLASS": pl.Utf8,
    "DELIVERY_SERVICE_NAME": pl.Utf8,
    "ACCOUNT_IDENTIFIER": pl.Utf8,
    "INTERVAL_READING_DATE": pl.Utf8,  # Will parse to date later
    "INTERVAL_LENGTH": pl.Utf8,
    "TOTAL_REGISTERED_ENERGY": pl.Float64,
    "PLC_VALUE": pl.Utf8,
    "NSPL_VALUE": pl.Utf8,
    # Interval columns not listed - will be inferred as Float64
    # Format: INTERVAL_HR0000, INTERVAL_HR0030, ..., INTERVAL_HR2430
    # OR: HR0000, HR0030, ..., HR2430
}


def list_s3_files(
    year_month: str,
    bucket: str = S3_BUCKET,
    prefix: str = S3_PREFIX,
    max_files: int | None = None,
) -> list[str]:
    """
    List CSV files in S3 for a given year-month.

    Args:
        year_month: Format 'YYYYMM' (e.g., '202308')
        max_files: Limit number of files (useful for testing)

    Returns:
        List of S3 URIs (s3://bucket/key format)
    """
    import boto3

    s3 = boto3.client("s3")
    full_prefix = f"{prefix}{year_month}/"

    logger.info(f"Listing files from s3://{bucket}/{full_prefix}")

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=full_prefix)

    s3_uris = []
    for page in pages:
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith(".csv"):
                # Return as S3 URI for polars scan_csv
                s3_uri = f"s3://{bucket}/{key}"
                s3_uris.append(s3_uri)
                if max_files and len(s3_uris) >= max_files:
                    logger.info(f"Limited to {max_files} files for testing")
                    return s3_uris

    logger.info(f"Found {len(s3_uris)} CSV files")
    return s3_uris


def scan_single_csv_lazy(s3_uri: str, schema_overrides: dict | None = None) -> pl.LazyFrame:
    """
    Lazily scan a single CSV from S3 and apply transformations.

    This function keeps everything lazy - no materialization happens here.
    All operations will be optimized and executed when .collect() is called.

    Args:
        s3_uri: S3 URI (s3://bucket/key)
        schema_overrides: Optional schema overrides for known columns
                         If None, uses inference (slower but works with varying schemas)

    Returns:
        Lazy DataFrame with transformations applied
    """
    # For ComEd data, we know the ID columns but interval columns vary by file
    # We'll provide schema overrides for the ID columns only
    if schema_overrides is None:
        schema_overrides = {
            "ZIP_CODE": pl.Utf8,
            "DELIVERY_SERVICE_CLASS": pl.Utf8,
            "DELIVERY_SERVICE_NAME": pl.Utf8,
            "ACCOUNT_IDENTIFIER": pl.Utf8,
            "INTERVAL_READING_DATE": pl.Utf8,
            "INTERVAL_LENGTH": pl.Utf8,
            "TOTAL_REGISTERED_ENERGY": pl.Float64,
            "PLC_VALUE": pl.Utf8,
            "NSPL_VALUE": pl.Utf8,
        }

    # Scan CSV lazily
    # We use schema_overrides instead of full schema to let Polars infer interval columns
    # This is still much faster than inferring everything
    lf = pl.scan_csv(
        s3_uri,
        schema_overrides=schema_overrides,
        ignore_errors=True,
        infer_schema_length=1000,  # Small inference for interval columns only
    )

    # Apply wide-to-long transformation (stays lazy)
    lf_long = transform_wide_to_long_lazy(lf)

    # Add time columns (stays lazy)
    lf_time = add_time_columns_lazy(lf_long)

    return lf_time


def detect_interval_columns_lazy(lf: pl.LazyFrame) -> list[str]:
    """
    Detect interval columns from schema without materializing.
    Returns column names like INTERVAL_HR0000, HR0030, etc.
    """
    INTERVAL_PREFIXES = ("INTERVAL_HR", "HR")
    cols: list[str] = []

    # Get columns from schema (no materialization needed)
    for c in lf.collect_schema().names():
        for p in INTERVAL_PREFIXES:
            if c.startswith(p):
                cols.append(c)
                break

    if not cols:
        raise ValueError(f"No interval columns found. Looked for prefixes: {INTERVAL_PREFIXES}")

    return sorted(cols)


def transform_wide_to_long_lazy(
    lf: pl.LazyFrame,
    date_col: str = "INTERVAL_READING_DATE",
) -> pl.LazyFrame:
    """
    Transform wide ComEd interval file to long format with proper timestamps.

    Handles DST transitions correctly:
    - Spring Forward: Filters nulls in HR0200/HR0230 (missing hour) + HR2430/HR2500
    - Fall Back: Keeps all 50 intervals (HR2430/HR2500 contain repeated 1 AM hour)
    - Normal days: Filters nulls in HR2430/HR2500 (unused DST placeholders)

    All operations stay LAZY - no materialization.

    Returns: LazyFrame with columns [zip_code, delivery_service_class,
             delivery_service_name, account_identifier, datetime, kwh]
    """
    # ID columns we want to keep (if they exist in the data)
    ID_COLS = [
        "ZIP_CODE",
        "DELIVERY_SERVICE_CLASS",
        "DELIVERY_SERVICE_NAME",
        "ACCOUNT_IDENTIFIER",
        "INTERVAL_READING_DATE",
        "INTERVAL_LENGTH",
        "TOTAL_REGISTERED_ENERGY",
        "PLC_VALUE",
        "NSPL_VALUE",
    ]

    # Get schema to check which columns exist (no materialization)
    schema = lf.collect_schema()
    keep_ids = [c for c in ID_COLS if c in schema.names()]

    # Detect interval columns (no materialization)
    interval_cols = detect_interval_columns_lazy(lf)

    lf_long = (
        lf
        # Select only columns we need
        .select(keep_ids + interval_cols)
        # Unpivot to long format (stays lazy)
        .unpivot(
            index=keep_ids,
            on=interval_cols,
            variable_name="interval_col",
            value_name="kwh",
        )
        # CRITICAL: Filter null kWh values (handles DST automatically)
        # This is lazy - Polars will push this predicate down
        .filter(pl.col("kwh").is_not_null())
        # Extract time string from interval column name
        .with_columns([pl.col("interval_col").str.extract(r"HR(\d{4})", 1).alias("time_str")])
        # Parse date and time components
        .with_columns([
            pl.col(date_col).str.strptime(pl.Date, format="%m/%d/%Y", strict=False).alias("service_date"),
            pl.col("time_str").str.slice(0, 2).cast(pl.Int16).alias("hour_raw"),
            pl.col("time_str").str.slice(2, 2).cast(pl.Int16).alias("minute"),
        ])
        # Calculate hour and day offset (for HR2430/HR2500 which go into next day)
        .with_columns([
            (pl.col("hour_raw") // 24).alias("days_offset"),
            (pl.col("hour_raw") % 24).alias("hour"),
        ])
        # Construct final datetime
        .with_columns([
            (
                pl.col("service_date").cast(pl.Datetime)
                + pl.duration(days=pl.col("days_offset"), hours=pl.col("hour"), minutes=pl.col("minute"))
            ).alias("datetime")
        ])
        # NOTE: Removed .sort() here - will sort at end if needed
        # Select and rename final columns
        .select([
            pl.col("ZIP_CODE").alias("zip_code") if "ZIP_CODE" in schema.names() else pl.lit(None).alias("zip_code"),
            pl.col("DELIVERY_SERVICE_CLASS").alias("delivery_service_class")
            if "DELIVERY_SERVICE_CLASS" in schema.names()
            else pl.lit(None).alias("delivery_service_class"),
            pl.col("DELIVERY_SERVICE_NAME").alias("delivery_service_name")
            if "DELIVERY_SERVICE_NAME" in schema.names()
            else pl.lit(None).alias("delivery_service_name"),
            pl.col("ACCOUNT_IDENTIFIER").alias("account_identifier")
            if "ACCOUNT_IDENTIFIER" in schema.names()
            else pl.lit(None).alias("account_identifier"),
            pl.col("datetime"),
            pl.col("kwh").cast(pl.Float64),
        ])
    )

    return lf_long


def add_time_columns_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add date/hour/weekday/is_weekend columns AND DST flags.
    All operations stay lazy.
    """
    from datetime import date

    # DST transition dates for 2023
    DST_SPRING_2023 = date(2023, 3, 12)  # Spring Forward
    DST_FALL_2023 = date(2023, 11, 5)  # Fall Back

    return lf.with_columns([
        pl.col("datetime").dt.date().alias("date"),
        pl.col("datetime").dt.hour().alias("hour"),
        pl.col("datetime").dt.weekday().alias("weekday"),
        (pl.col("datetime").dt.weekday() >= 5).alias("is_weekend"),
    ]).with_columns([
        # Flag DST transition days
        (pl.col("date") == DST_SPRING_2023).alias("is_spring_forward_day"),
        (pl.col("date") == DST_FALL_2023).alias("is_fall_back_day"),
        ((pl.col("date") == DST_SPRING_2023) | (pl.col("date") == DST_FALL_2023)).alias("is_dst_day"),
    ])


def process_month_batch(
    year_month: str,
    output_path: Path,
    max_files: int | None = None,
    bucket: str = S3_BUCKET,
    prefix: str = S3_PREFIX,
    sort_output: bool = False,
) -> None:
    """
    Process all CSVs for a month and save as single Parquet file.

    Uses lazy semantics throughout - only materializes at the final .collect().
    Polars will automatically:
    - Parallelize reading from S3
    - Optimize the query plan
    - Stream data in chunks to minimize memory usage

    Args:
        year_month: Format 'YYYYMM' (e.g., '202308')
        output_path: Where to save combined Parquet
        max_files: Limit for testing (None = process all)
        sort_output: Whether to sort by timestamp (adds overhead, usually not needed)
    """
    logger.info(f"Processing month: {year_month}")

    # List files (returns S3 URIs)
    s3_uris = list_s3_files(year_month, bucket, prefix, max_files)

    if not s3_uris:
        raise ValueError(ERR_NO_FILES_FOUND.format(year_month))

    logger.info(f"Scanning {len(s3_uris)} files lazily...")

    # Scan all files lazily
    lazy_frames = []
    for i, s3_uri in enumerate(s3_uris, 1):
        filename = s3_uri.split("/")[-1]
        logger.debug(f"Scanning {i}/{len(s3_uris)}: {filename}")
        try:
            lf = scan_single_csv_lazy(s3_uri)
            lazy_frames.append(lf)
        except Exception:
            logger.exception(f"Failed to scan {s3_uri}")
            continue

    if not lazy_frames:
        raise ValueError(ERR_NO_SUCCESSFUL_PROCESS)

    logger.info(f"Concatenating {len(lazy_frames)} lazy frames...")

    # Concatenate lazily
    lf_combined = pl.concat(lazy_frames, how="diagonal_relaxed")

    # Optional: sort at the end (your boss suggested thinking about whether this is needed)
    # Sorting forces materialization, so only do if necessary
    # Often you can just use .group_by() in your analysis instead
    if sort_output:
        logger.info("Sorting by timestamp (this will materialize data)...")
        lf_combined = lf_combined.sort("timestamp")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # THIS IS THE ONLY MATERIALIZATION POINT
    # Everything up to here has been lazy - polars will now:
    # 1. Optimize the entire query plan
    # 2. Parallelize S3 reads
    # 3. Stream data in chunks
    # 4. Write directly to parquet
    logger.info("Collecting and writing to Parquet (this is where execution happens)...")
    lf_combined.sink_parquet(output_path)

    logger.info(f"Successfully wrote data to {output_path}")

    # Optional: log row count (requires a separate scan)
    # Uncomment if you want to know the count, but it adds overhead
    # row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    # logger.info(f"Total rows: {row_count:,}")


def process_month_batch_streaming(
    year_month: str,
    output_path: Path,
    max_files: int | None = None,
    bucket: str = S3_BUCKET,
    prefix: str = S3_PREFIX,
) -> None:
    """
    Alternative implementation using sink_parquet for true streaming.

    This version uses .sink_parquet() instead of .collect().write_parquet(),
    which means the data streams directly to disk without ever fully materializing
    in memory. This is ideal for very large datasets.

    Args:
        year_month: Format 'YYYYMM' (e.g., '202308')
        output_path: Where to save combined Parquet
        max_files: Limit for testing (None = process all)
    """
    logger.info(f"Processing month (streaming mode): {year_month}")

    # List files
    s3_uris = list_s3_files(year_month, bucket, prefix, max_files)

    if not s3_uris:
        raise ValueError(ERR_NO_FILES_FOUND.format(year_month))

    # Scan and transform lazily
    lazy_frames = []
    for s3_uri in s3_uris:
        try:
            lf = scan_single_csv_lazy(s3_uri)
            lazy_frames.append(lf)
        except Exception:
            logger.exception(f"Failed to scan {s3_uri}")
            continue

    if not lazy_frames:
        raise ValueError(ERR_NO_SUCCESSFUL_PROCESS)

    # Concatenate and stream to disk
    lf_combined = pl.concat(lazy_frames, how="diagonal_relaxed")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Streams directly to disk - never fully materializes
    logger.info("Streaming to Parquet...")
    lf_combined.sink_parquet(output_path)

    logger.info(f"Successfully streamed data to {output_path}")


# Example main function for testing
def main():
    """Example usage - can be called from command line."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python step0_aws.py YYYYMM [max_files]")
        sys.exit(1)

    year_month = sys.argv[1]
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else None

    output_path = Path(f"data/processed/comed_{year_month}.parquet")

    process_month_batch(
        year_month=year_month,
        output_path=output_path,
        max_files=max_files,
        sort_output=False,  # Usually not needed - can group_by instead
    )


if __name__ == "__main__":
    main()
