# smart_meter_analysis/aws_loader.py
"""
Batch-load ComEd smart meter CSVs from S3, convert wide intervals to long,
add time features, and write a monthly Parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# S3 configuration
S3_BUCKET = "smart-meter-data-sb"
S3_PREFIX = "sharepoint-files/Zip4/"

# Error messages
ERR_NO_FILES_FOUND = "No files found for month: {}"
ERR_NO_SUCCESSFUL_PROCESS = "No files were successfully processed"

# Fixed ID/metadata columns
COMED_SCHEMA_OVERRIDES = {
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

COMED_INTERVAL_COLUMNS: list[str] = [
    f"INTERVAL_HR{m // 60:02d}{m % 60:02d}_ENERGY_QTY" for m in range(30, 25 * 60 + 1, 30)
]

# Full schema = ID/metadata + interval columns
COMED_SCHEMA: dict[str, pl.DataType] = {
    **COMED_SCHEMA_OVERRIDES,
    **dict.fromkeys(COMED_INTERVAL_COLUMNS, pl.Float64),
}


def list_s3_files(
    year_month: str,
    bucket: str = S3_BUCKET,
    prefix: str = S3_PREFIX,
    max_files: int | None = None,
) -> list[str]:
    """
    Return S3 URIs for CSVs under the given YYYYMM prefix.
    """
    import boto3

    s3 = boto3.client("s3")
    full_prefix = f"{prefix}{year_month}/"
    logger.info(f"Listing files from s3://{bucket}/{full_prefix}")

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=full_prefix)

    s3_uris: list[str] = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".csv"):
                s3_uris.append(f"s3://{bucket}/{key}")
                if max_files and len(s3_uris) >= max_files:
                    logger.info(f"Limited to {max_files} files for testing")
                    return s3_uris

    logger.info(f"Found {len(s3_uris)} CSV files")
    return s3_uris


def scan_single_csv_lazy(
    s3_uri: str,
    schema: dict[str, pl.DataType] | None = None,
) -> pl.LazyFrame:
    """
    Lazily scan one CSV, using fixed typing for known columns,
    then reshape and add time features. Stays lazy until sink/write.
    """
    schema = COMED_SCHEMA if schema is None else schema

    lf = pl.scan_csv(
        s3_uri,
        schema_overrides=schema,
        ignore_errors=True,
    )

    lf_long = transform_wide_to_long_lazy(lf)
    return add_time_columns_lazy(lf_long)


def transform_wide_to_long_lazy(
    lf: pl.LazyFrame,
    date_col: str = "INTERVAL_READING_DATE",
) -> pl.LazyFrame:
    """
    Transform wide ComEd interval file to long format with proper timestamps.
    """
    from datetime import date

    DST_FALL_2023 = date(2023, 11, 5)

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

    schema = lf.collect_schema()
    keep_ids = [c for c in ID_COLS if c in schema.names()]

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
        .with_columns([pl.col("interval_col").str.extract(r"HR(\d{4})", 1).alias("time_str")])
        .with_columns([
            pl.col(date_col).str.strptime(pl.Date, format="%m/%d/%Y", strict=False).alias("service_date"),
            pl.col("time_str").str.slice(0, 2).cast(pl.Int16).alias("hour_raw"),
            pl.col("time_str").str.slice(2, 2).cast(pl.Int16).alias("minute"),
        ])
        # --- DST filter: drop HR2430/HR2500 except on the Fall-Back day ---
        .filter(~(pl.col("time_str").is_in(["2430", "2500"]) & (pl.col("service_date") != pl.lit(DST_FALL_2023))))
        # --------------------------------------------------------------------
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
    Add date/hour/weekday/is_weekend and 2023 DST transition day flags.
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
    Process all CSVs for a month and write a single Parquet file.
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
            lazy_frames.append(scan_single_csv_lazy(s3_uri))
        except Exception:
            logger.exception(f"Failed to scan {s3_uri}")
            continue

    if not lazy_frames:
        raise ValueError(ERR_NO_SUCCESSFUL_PROCESS)

    logger.info(f"Concatenating {len(lazy_frames)} lazy frames...")
    lf_combined = pl.concat(lazy_frames, how="diagonal_relaxed")

    if sort_output:
        logger.info("Sorting by datetime (materializes the data)")
        lf_combined = lf_combined.sort("datetime")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing Parquet...")
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
    Streaming variant that avoids materializing the combined dataset in memory.
    """
    logger.info(f"Processing month (streaming mode): {year_month}")

    s3_uris = list_s3_files(year_month, bucket, prefix, max_files)
    if not s3_uris:
        raise ValueError(ERR_NO_FILES_FOUND.format(year_month))

    lazy_frames: list[pl.LazyFrame] = []
    for s3_uri in s3_uris:
        try:
            lazy_frames.append(scan_single_csv_lazy(s3_uri))
        except Exception:
            logger.exception(f"Failed to scan {s3_uri}")
            continue

    if not lazy_frames:
        raise ValueError(ERR_NO_SUCCESSFUL_PROCESS)

    lf_combined = pl.concat(lazy_frames, how="diagonal_relaxed")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing Parquet (streaming)...")
    lf_combined.sink_parquet(output_path)
    logger.info(f"Successfully streamed data to {output_path}")


def main() -> None:
    """
    CLI entry point.
    Usage: python aws_loader.py YYYYMM [max_files]
    """
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python aws_loader.py YYYYMM [max_files]")
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
