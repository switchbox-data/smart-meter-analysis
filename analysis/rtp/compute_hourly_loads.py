#!/usr/bin/env python3
"""
Compute hourly household loads for RTP billing analysis.

Takes interval-level ComEd smart meter data (30-minute intervals in this pipeline)
and aggregates to hourly kWh per household.

If --cluster-assignments is provided, restrict to households that appear
in the clustering assignments using a lazy semi-join (memory-safe).

Expected input columns (from processed interval parquet):
  - account_identifier
  - zip_code
  - datetime       (naive local time, Datetime[us], tz=None)
  - energy_kwh

Output columns:
  - account_identifier
  - zip_code
  - hour_chicago   (datetime truncated to hour)
  - kwh_hour       (sum of energy_kwh within that hour)
"""

from __future__ import annotations

import argparse
import glob as _glob
import logging
import shutil
import tempfile
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_GROUP_KEYS = ["account_identifier", "zip_code", "hour_chicago"]


def _resolve_parquet_paths(input_path: Path) -> list[str]:
    """Resolve input to a list of parquet paths (file, directory, or glob)."""
    input_str = str(input_path)
    if any(ch in input_str for ch in ["*", "?", "["]):
        paths = sorted(_glob.glob(input_str))
        if not paths:
            raise FileNotFoundError(f"Input parquet glob matched 0 files: {input_str}")
        return paths
    if input_path.is_dir():
        paths = sorted(str(p) for p in input_path.glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"Input parquet directory has 0 *.parquet files: {input_path}")
        return paths
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")
    return [str(input_path)]


def _validate_schema(lf: pl.LazyFrame) -> None:
    """Raise ValueError if required columns are missing."""
    required = {"account_identifier", "zip_code", "datetime", "energy_kwh"}
    missing = required - set(lf.collect_schema().names())
    if missing:
        raise ValueError(f"Input file missing required columns: {sorted(missing)}")


def _prepare_assignments_lf(assignments_path: Path) -> pl.LazyFrame:
    """Load and validate cluster assignments, returning a unique-account LazyFrame."""
    if not assignments_path.exists():
        raise FileNotFoundError(f"Cluster assignments not found: {assignments_path}")
    lf_assign = pl.scan_parquet(assignments_path)
    if "account_identifier" not in lf_assign.collect_schema().names():
        raise ValueError("Cluster assignments file has no 'account_identifier' column")
    return lf_assign.select(pl.col("account_identifier")).unique()


def _aggregate_one_file(
    path: str,
    lf_assign: pl.LazyFrame | None,
) -> pl.LazyFrame:
    """Scan a single parquet file, apply optional semi-join, and aggregate to hourly."""
    lf = pl.scan_parquet(path)
    if lf_assign is not None:
        lf = lf.join(lf_assign, on="account_identifier", how="semi")
    return (
        lf.with_columns(pl.col("datetime").dt.truncate("1h").alias("hour_chicago"))
        .group_by(_GROUP_KEYS)
        .agg(pl.col("energy_kwh").sum().alias("kwh_hour"))
    )


def compute_hourly_loads(
    input_path: Path,
    assignments_path: Path | None,
    output_path: Path,
    *,
    sort_output: bool,
) -> None:
    scan_paths = _resolve_parquet_paths(input_path)
    logger.info("Scanning interval data (%d file(s)): %s", len(scan_paths), scan_paths[0])

    # Validate schema from the first file
    _validate_schema(pl.scan_parquet(scan_paths[0]))

    lf_assign: pl.LazyFrame | None = None
    if assignments_path is not None:
        lf_assign = _prepare_assignments_lf(assignments_path)
        logger.info("Restricting to accounts in cluster assignments via semi-join: %s", assignments_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: single file — no temp dir needed
    if len(scan_paths) == 1:
        logger.info("Aggregating hourly loads (single file)")
        lf_hourly = _aggregate_one_file(scan_paths[0], lf_assign)
        if sort_output:
            lf_hourly = lf_hourly.sort(_GROUP_KEYS)
        lf_hourly.sink_parquet(output_path)
        logger.info("Wrote hourly loads to %s", output_path)
        return

    # Chunked path: aggregate each file independently, then re-aggregate
    tmp_dir = Path(tempfile.mkdtemp(prefix="hourly_loads_"))
    try:
        for i, path in enumerate(scan_paths):
            logger.info("Aggregating file %d/%d: %s", i + 1, len(scan_paths), path)
            lf_chunk = _aggregate_one_file(path, lf_assign)
            lf_chunk.sink_parquet(tmp_dir / f"chunk_{i:04d}.parquet")

        # Re-aggregate: same key may span multiple input files
        logger.info("Re-aggregating %d intermediate chunks", len(scan_paths))
        intermediate_paths = sorted(str(p) for p in tmp_dir.glob("*.parquet"))
        lf_merged = pl.scan_parquet(intermediate_paths).group_by(_GROUP_KEYS).agg(pl.col("kwh_hour").sum())

        if sort_output:
            lf_merged = lf_merged.sort(_GROUP_KEYS)

        lf_merged.sink_parquet(output_path)
        logger.info("Wrote hourly loads to %s", output_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate interval-level ComEd data to hourly loads per household.")
    parser.add_argument("--input", type=Path, required=True, help="Path to comed_YYYYMM.parquet (interval-level data).")
    parser.add_argument(
        "--cluster-assignments",
        type=Path,
        default=None,
        help="Optional: cluster_assignments.parquet to restrict to sampled households.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output parquet for hourly loads.")
    parser.add_argument(
        "--sort-output",
        action="store_true",
        help="If set, sort output rows by (zip_code, account_identifier, hour_chicago).",
    )

    args = parser.parse_args()

    try:
        compute_hourly_loads(args.input, args.cluster_assignments, args.output, sort_output=args.sort_output)
    except Exception as e:
        logger.error("Failed to compute hourly loads: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
