#!/usr/bin/env python3
"""
Memory-optimized CSV processing for large file counts.

Processes CSV files in batches to avoid OOM when concatenating thousands of lazy frames.

Usage:
    python process_csvs_batched_optimized.py \\
        --input-dir data/validation_runs/202308_50k/samples \\
        --output data/validation_runs/202308_50k/processed_combined.parquet \\
        --batch-size 5000
"""

import argparse
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_csv_batch_to_parquet(
    csv_files: list[Path],
    batch_num: int,
    temp_dir: Path,
) -> Path:
    """
    Process a batch of CSV files and write to temporary parquet.

    Args:
        csv_files: List of CSV file paths to process
        batch_num: Batch number (for naming)
        temp_dir: Directory for temporary batch files

    Returns:
        Path to the batch parquet file
    """
    from smart_meter_analysis.aws_loader import (
        COMED_SCHEMA,
        add_time_columns_lazy,
        transform_wide_to_long_lazy,
    )

    logger.info("Processing batch %d: %d files", batch_num, len(csv_files))

    lazy_frames = []
    for i, csv_path in enumerate(csv_files, 1):
        if i % 200 == 0:
            logger.info("  Scanned %d/%d files in batch %d", i, len(csv_files), batch_num)
        try:
            lf = pl.scan_csv(str(csv_path), schema_overrides=COMED_SCHEMA, ignore_errors=True)
            lf = transform_wide_to_long_lazy(lf)
            lf = add_time_columns_lazy(lf, day_mode="calendar")
            lazy_frames.append(lf)
        except Exception as exc:
            logger.warning("Failed to scan %s: %s", csv_path.name, exc)

    if not lazy_frames:
        raise ValueError(f"No files successfully scanned in batch {batch_num}")

    # Write batch to temporary parquet
    batch_output = temp_dir / f"batch_{batch_num:04d}.parquet"
    lf_combined = pl.concat(lazy_frames, how="diagonal_relaxed")
    lf_combined.sink_parquet(batch_output)

    logger.info("  Batch %d complete: %s", batch_num, batch_output)
    return batch_output


def main():
    parser = argparse.ArgumentParser(description="Process CSV files in memory-safe batches")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing CSV files")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet file path")
    parser.add_argument("--batch-size", type=int, default=5000, help="Files per batch (default: 5000)")

    args = parser.parse_args()

    # Get all CSV files
    csv_files = sorted(args.input_dir.glob("*.csv"))
    logger.info("Found %d CSV files", len(csv_files))

    if not csv_files:
        logger.error("No CSV files found in %s", args.input_dir)
        return 1

    # Create temp directory
    temp_dir = args.output.parent / "temp_batches"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Process in batches
    batch_files = []
    for batch_num, i in enumerate(range(0, len(csv_files), args.batch_size), 1):
        batch = csv_files[i : i + args.batch_size]
        batch_file = process_csv_batch_to_parquet(batch, batch_num, temp_dir)
        batch_files.append(batch_file)

    # Concatenate all batches using streaming
    logger.info("Concatenating %d batch files into final output...", len(batch_files))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lf_combined = pl.concat([pl.scan_parquet(str(f)) for f in batch_files], how="diagonal_relaxed")
    lf_combined.sink_parquet(args.output)

    # Verify
    row_count = pl.scan_parquet(args.output).select(pl.len()).collect()[0, 0]
    logger.info("Success! Wrote %s records to %s", f"{row_count:,}", args.output)

    # Clean up temp files
    logger.info("Cleaning up temporary batch files...")
    for batch_file in batch_files:
        batch_file.unlink()
    temp_dir.rmdir()

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
