#!/usr/bin/env python3
"""
Memory-optimized CSV processing for large file counts.

Processes CSV files in batches and sub-batches to avoid OOM / huge lazy plans.

Usage:
    python process_csvs_batched_optimized.py \
        --input-dir data/validation_runs/202308_50k/samples \
        --output data/validation_runs/202308_50k/processed_combined.parquet \
        --batch-size 5000 \
        --sub-batch-size 250
"""

import argparse
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_csv_subbatch_to_parquet(
    csv_files: list[Path],
    batch_num: int,
    sub_num: int,
    temp_dir: Path,
) -> Path:
    """
    Process a sub-batch of CSV files and write to a temporary parquet file.
    """
    from smart_meter_analysis.aws_loader import (
        COMED_SCHEMA,
        add_time_columns_lazy,
        transform_wide_to_long_lazy,
    )

    logger.info("  Sub-batch %d.%d: %d files", batch_num, sub_num, len(csv_files))

    lazy_frames: list[pl.LazyFrame] = []
    for i, csv_path in enumerate(csv_files, 1):
        if i % 200 == 0:
            logger.info("    Scanned %d/%d files in sub-batch %d.%d", i, len(csv_files), batch_num, sub_num)
        try:
            lf = pl.scan_csv(
                str(csv_path),
                schema_overrides=COMED_SCHEMA,
                ignore_errors=True,
            )
            lf = transform_wide_to_long_lazy(lf)

            # IMPORTANT: updated signature (no day_mode)
            lf = add_time_columns_lazy(lf)

            lazy_frames.append(lf)
        except Exception as exc:
            logger.warning("Failed to scan %s: %s", csv_path.name, exc)

    if not lazy_frames:
        raise ValueError(f"No files successfully scanned in sub-batch {batch_num}.{sub_num}")

    sub_output = temp_dir / f"batch_{batch_num:04d}_sub_{sub_num:04d}.parquet"

    # Combine this sub-batch and write immediately
    pl.concat(lazy_frames, how="diagonal_relaxed").sink_parquet(sub_output)

    logger.info("  Sub-batch %d.%d complete: %s", batch_num, sub_num, sub_output)
    return sub_output


def process_csv_batch_to_parquet(
    csv_files: list[Path],
    batch_num: int,
    temp_dir: Path,
    sub_batch_size: int,
) -> Path:
    """
    Process a batch of CSV files by splitting into sub-batches and writing a single
    batch parquet composed from the sub-batch parquets.
    """
    logger.info("Processing batch %d: %d files", batch_num, len(csv_files))

    sub_files: list[Path] = []
    for sub_num, i in enumerate(range(0, len(csv_files), sub_batch_size), 1):
        sub = csv_files[i : i + sub_batch_size]
        sub_file = process_csv_subbatch_to_parquet(sub, batch_num, sub_num, temp_dir)
        sub_files.append(sub_file)

    batch_output = temp_dir / f"batch_{batch_num:04d}.parquet"
    logger.info("  Concatenating %d sub-batches into %s", len(sub_files), batch_output)

    pl.concat([pl.scan_parquet(str(f)) for f in sub_files], how="diagonal_relaxed").sink_parquet(batch_output)

    # Clean up sub-batch files
    for f in sub_files:
        f.unlink(missing_ok=True)

    logger.info("Batch %d complete: %s", batch_num, batch_output)
    return batch_output


def main() -> int:
    parser = argparse.ArgumentParser(description="Process CSV files in memory-safe batches")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing CSV files")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet file path")
    parser.add_argument("--batch-size", type=int, default=5000, help="Files per batch (default: 5000)")
    parser.add_argument(
        "--sub-batch-size",
        type=int,
        default=250,
        help="Files per sub-batch within each batch (default: 250).",
    )

    args = parser.parse_args()

    csv_files = sorted(args.input_dir.glob("*.csv"))
    logger.info("Found %d CSV files", len(csv_files))

    if not csv_files:
        logger.error("No CSV files found in %s", args.input_dir)
        return 1

    temp_dir = args.output.parent / "temp_batches"
    temp_dir.mkdir(parents=True, exist_ok=True)

    batch_files: list[Path] = []
    for batch_num, i in enumerate(range(0, len(csv_files), args.batch_size), 1):
        batch = csv_files[i : i + args.batch_size]
        batch_file = process_csv_batch_to_parquet(
            csv_files=batch,
            batch_num=batch_num,
            temp_dir=temp_dir,
            sub_batch_size=args.sub_batch_size,
        )
        batch_files.append(batch_file)

    logger.info("Concatenating %d batch files into final output...", len(batch_files))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pl.concat([pl.scan_parquet(str(f)) for f in batch_files], how="diagonal_relaxed").sink_parquet(args.output)

    row_count = pl.scan_parquet(args.output).select(pl.len()).collect()[0, 0]
    logger.info("Success! Wrote %s records to %s", f"{row_count:,}", args.output)

    logger.info("Cleaning up temporary batch files...")
    for f in batch_files:
        f.unlink(missing_ok=True)
    temp_dir.rmdir()

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
