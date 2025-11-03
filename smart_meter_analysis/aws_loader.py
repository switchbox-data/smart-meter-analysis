# smart_meter_analysis/aws_loader.py
"""
AWS S3 utilities for batch processing ComEd smart meter data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

# Local dtype alias for mypy
DType = Any

logger = logging.getLogger(__name__)

# S3 Configuration
S3_BUCKET = "smart-meter-data-sb"
S3_PREFIX = "sharepoint-files/Zip4/"

# Error messages
ERR_NO_FILES_FOUND = "No files found for month: {}"
ERR_NO_SUCCESSFUL_PROCESS = "No files were successfully processed"

# ID columns with explicit dtypes
COMED_SCHEMA_OVERRIDES: dict[str, DType] = {
    "ZIP_CODE": pl.Utf8,
    "DELIVERY_SERVICE_CLASS": pl.Utf8,
    "DELIVERY_SERVICE_NAME": pl.Utf8,
    "ACCOUNT_IDENTIFIER": pl.Utf8,
    "INTERVAL_READING_DATE": pl.Utf8,  # parsed later
    "INTERVAL_LENGTH": pl.Utf8,
    "TOTAL_REGISTERED_ENERGY": pl.Float64,
    "PLC_VALUE": pl.Utf8,
    "NSPL_VALUE": pl.Utf8,
}

COMED_INTERVAL_COLUMNS: list[str] = [
    f"INTERVAL_HR{m // 60:02d}{m % 60:02d}_ENERGY_QTY" for m in range(30, 24 * 60 + 1, 30)
]


_INTERVAL_SCHEMA: dict[str, DType] = dict.fromkeys(COMED_INTERVAL_COLUMNS, pl.Float64)
COMED_SCHEMA: dict[str, DType] = {**COMED_SCHEMA_OVERRIDES, **_INTERVAL_SCHEMA}


def list_s3_files(
    year_month: str,
    bucket: str = S3_BUCKET,
    prefix: str = S3_PREFIX,
    max_files: int | None = None,
) -> list[str]:
    """
    List CSV files in S3 for a given year-month.

    Args:
        year_month: 'YYYYMM' (e.g., '202308')
        max_files: optional limit for testing

    Returns:
        S3 URIs as s3://bucket/key
    """
    import boto3

    s3 = boto3.client("s3")
    full_prefix = f"{prefix}{year_month}/"

    logger.info(f"Listing files from s3://{bucket}/{full_prefix}")

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=full_prefix)

    s3_uris: list[str] = []
    for page in pages:
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith(".csv"):
                s3_uri = f"s3://{bucket}/{key}"
                s3_uris.append(s3_uri)
                if max_files and len(s3_uris) >= max_files:
                    logger.info(f"Limited to {max_files} files for testing")
                    return s3_uris

    logger.info(f"Found {len(s3_uris)} CSV files")
    return s3_uris


def scan_single_csv_lazy(s3_uri: str, schema: dict[str, DType] | None = None) -> pl.LazyFrame:
    """
    Lazily scan a single CSV from S3 and apply transformations using a fixed schema.
    """
    schema = COMED_SCHEMA if schema is None else schema

    lf = pl.scan_csv(
        s3_uri,
        schema_overrides=schema,  # keep overrides-based approach as-is
        ignore_errors=True,
    )

    lf_long = transform_wide_to_long_lazy(lf)
    lf_time = add_time_columns_lazy(lf_long)
    return lf_time


def transform_wide_to_long_lazy(
    lf: pl.LazyFrame,
    date_col: str = "INTERVAL_READING_DATE",
) -> pl.LazyFrame:
    """
    Transform wide ComEd interval file to long format with timestamps.
    """
    id_cols = [
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

    schema = lf.collect_schema()
    keep_ids = [c for c in id_cols if c in schema.names()]
    interval_cols = [c for c in COMED_INTERVAL_COLUMNS if c in schema.names()]

    lf_long = (
        lf.select(keep_ids + interval_cols)
        .unpivot(
            index=keep_ids,
            on=interval_cols,
            variable_name="interval_col",
            value_name="kwh",
        )
        .filter(pl.col("kwh").is_not_null())
        .with_columns(pl.col("interval_col").str.extract(r"HR(\d{4})", 1).alias("time_str"))
        .with_columns([
            pl.col(date_col).str.strptime(pl.Date, format="%m/%d/%Y", strict=False).alias("service_date"),
            pl.col("time_str").str.slice(0, 2).cast(pl.Int16).alias("hour_raw"),
            pl.col("time_str").str.slice(2, 2).cast(pl.Int16).alias("minute"),
        ])
        .with_columns([
            (pl.col("hour_raw") // 24).alias("days_offset"),
            (pl.col("hour_raw") % 24).alias("hour"),
        ])
        .with_columns([
            (
                pl.col("service_date").cast(pl.Datetime)
                + pl.duration(days=pl.col("days_offset"), hours=pl.col("hour"), minutes=pl.col("minute"))
            ).alias("datetime")
        ])
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
    Add date/hour/weekday/is_weekend columns and DST flags.
    """
    from datetime import date

    DST_SPRING_2023 = date(2023, 3, 12)
    DST_FALL_2023 = date(2023, 11, 5)

    return lf.with_columns([
        pl.col("datetime").dt.date().alias("date"),
        pl.col("datetime").dt.hour().alias("hour"),
        pl.col("datetime").dt.weekday().alias("weekday"),
        (pl.col("datetime").dt.weekday() >= 5).alias("is_weekend"),
    ]).with_columns([
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
    Process all CSVs for a month and save as a single Parquet file.
    """
    logger.info(f"Processing month: {year_month}")

    s3_uris = list_s3_files(year_month, bucket, prefix, max_files)
    if not s3_uris:
        raise ValueError(ERR_NO_FILES_FOUND.format(year_month))

    logger.info(f"Scanning {len(s3_uris)} files lazily...")

    lazy_frames: list[pl.LazyFrame] = []
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
    lf_combined = pl.concat(lazy_frames, how="diagonal_relaxed")

    if sort_output:
        logger.info("Sorting by datetime (this will materialize data)...")
        lf_combined = lf_combined.sort("datetime")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Collecting and writing to Parquet (this is where execution happens)...")
    lf_combined.sink_parquet(output_path)
    logger.info(f"Successfully wrote data to {output_path}")


def process_month_batch_streaming(
    year_month: str,
    output_path: Path,
    max_files: int | None = None,
    bucket: str = S3_BUCKET,
    prefix: str = S3_PREFIX,
) -> None:
    """
    Alternative implementation using sink_parquet for streaming to disk.
    """
    logger.info(f"Processing month (streaming mode): {year_month}")

    s3_uris = list_s3_files(year_month, bucket, prefix, max_files)
    if not s3_uris:
        raise ValueError(ERR_NO_FILES_FOUND.format(year_month))

    lazy_frames: list[pl.LazyFrame] = []
    for s3_uri in s3_uris:
        try:
            lf = scan_single_csv_lazy(s3_uri)
            lazy_frames.append(lf)
        except Exception:
            logger.exception(f"Failed to scan {s3_uri}")
            continue

    if not lazy_frames:
        raise ValueError(ERR_NO_SUCCESSFUL_PROCESS)

    lf_combined = pl.concat(lazy_frames, how="diagonal_relaxed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Streaming to Parquet...")
    lf_combined.sink_parquet(output_path)
    logger.info(f"Successfully streamed data to {output_path}")


def main() -> None:
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
        sort_output=False,
    )


if __name__ == "__main__":
    main()
