# smart_meter_analysis/step0_aws.py
"""
AWS S3 utilities for batch processing ComEd smart meter data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import boto3
import polars as pl

from .transformation import add_time_columns, transform_wide_to_long

logger = logging.getLogger(__name__)

# S3 Configuration
S3_BUCKET = "smart-meter-data-sb"
S3_PREFIX = "sharepoint-files/Zip4/"

# Error messages (for TRY003)
ERR_NO_FILES_FOUND = "No files found for month: {}"
ERR_NO_SUCCESSFUL_PROCESS = "No files were successfully processed"


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
        List of S3 keys
    """
    s3 = boto3.client("s3")
    full_prefix = f"{prefix}{year_month}/"

    logger.info(f"Listing files from s3://{bucket}/{full_prefix}")

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=full_prefix)

    keys = []
    for page in pages:
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith(".csv"):
                keys.append(key)
                if max_files and len(keys) >= max_files:
                    return keys

    logger.info(f"Found {len(keys)} CSV files")
    return keys


def download_s3_file(
    s3_key: str,
    local_path: Path,
    bucket: str = S3_BUCKET,
) -> None:
    """Download single file from S3."""
    s3 = boto3.client("s3")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, s3_key, str(local_path))
    logger.info(f"Downloaded: {local_path.name}")


def process_single_csv(
    s3_key: str,
    bucket: str = S3_BUCKET,
) -> pl.DataFrame:
    """
    Download and transform a single CSV from S3.
    Returns long-format DataFrame with time columns.
    """
    s3 = boto3.client("s3")

    # Read directly from S3 into memory
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    df_raw = pl.read_csv(obj["Body"], ignore_errors=True, infer_schema_length=5000)

    # Transform
    df_long = transform_wide_to_long(df_raw)
    df_time = add_time_columns(df_long)

    return df_time


def process_month_batch(
    year_month: str,
    output_path: Path,
    max_files: int | None = None,
    bucket: str = S3_BUCKET,
    prefix: str = S3_PREFIX,
) -> None:
    """
    Process all CSVs for a month and save as single Parquet file.

    Args:
        year_month: Format 'YYYYMM' (e.g., '202308')
        output_path: Where to save combined Parquet
        max_files: Limit for testing (None = process all)
    """
    logger.info(f"Processing month: {year_month}")

    # List files
    keys = list_s3_files(year_month, bucket, prefix, max_files)

    if not keys:
        raise ValueError(ERR_NO_FILES_FOUND.format(year_month))

    # Process each file
    dfs = []
    for i, key in enumerate(keys, 1):
        logger.info(f"Processing {i}/{len(keys)}: {Path(key).name}")
        try:
            df = process_single_csv(key, bucket)
            dfs.append(df)
        except Exception:
            logger.exception(f"Failed to process {key}")
            continue

    # Combine and save
    if dfs:
        combined = pl.concat(dfs, how="diagonal")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(output_path)
        logger.info(f"Saved {combined.height:,} rows to {output_path}")
    else:
        raise ValueError(ERR_NO_SUCCESSFUL_PROCESS)
