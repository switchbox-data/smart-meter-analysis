#!/usr/bin/env python3
"""
Compute hourly household loads for RTP billing analysis.

Takes interval-level ComEd smart meter data (30-minute intervals in this pipeline)
and aggregates to hourly kWh per household.

If --cluster-assignments is provided, restrict to households that appear
in the clustering assignments.

Large files are processed in sub-file row batches via PyArrow's
iter_batches() so peak memory is O(batch_size), not O(file).

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

import gc
import os

# Constrain Polars to a single thread so parallel workers don't each
# allocate full-dataset copies.  Must be set before polars is imported.
os.environ["POLARS_MAX_THREADS"] = "1"

import argparse
import glob as _glob
import logging
import shutil
import tempfile
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.info("Polars thread_pool_size = %d", pl.thread_pool_size())

_GROUP_KEYS = ["account_identifier", "zip_code", "hour_chicago"]
_INPUT_COLS = ["account_identifier", "zip_code", "datetime", "energy_kwh"]
_BATCH_SIZE = 500_000
_MERGE_FAN_IN = 16  # max intermediate files per re-aggregation pass


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


def _prepare_account_filter(assignments_path: Path) -> pl.Series:
    """Load cluster assignments and return account identifiers as a Series.

    Returning a pl.Series (rather than a Python set) lets Polars perform
    ``is_in`` filtering natively without repeated Python→Rust conversion.
    """
    if not assignments_path.exists():
        raise FileNotFoundError(f"Cluster assignments not found: {assignments_path}")
    lf_assign = pl.scan_parquet(assignments_path)
    if "account_identifier" not in lf_assign.collect_schema().names():
        raise ValueError("Cluster assignments file has no 'account_identifier' column")
    return lf_assign.select(pl.col("account_identifier")).unique().collect().to_series()


def _aggregate_file_chunked(
    path: str,
    account_filter: pl.Series | None,
    tmp_dir: Path,
    chunk_offset: int,
    batch_size: int = _BATCH_SIZE,
) -> int:
    """Read a parquet file in row batches, aggregate each, write intermediates.

    Only the four required columns are read from disk (_INPUT_COLS) so extra
    columns in the source file never touch memory.

    Returns the number of intermediate chunks written.
    """
    pf = pq.ParquetFile(path)
    n_written = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=_INPUT_COLS):
        df = pl.from_arrow(batch)
        if account_filter is not None:
            df = df.filter(pl.col("account_identifier").is_in(account_filter.implode()))
        if df.is_empty():
            del df
            continue
        agg = (
            df.select(
                pl.col("account_identifier"),
                pl.col("zip_code"),
                pl.col("datetime").dt.truncate("1h").alias("hour_chicago"),
                pl.col("energy_kwh"),
            )
            .group_by(_GROUP_KEYS)
            .agg(pl.col("energy_kwh").sum().alias("kwh_hour"))
        )
        del df
        agg.write_parquet(tmp_dir / f"chunk_{chunk_offset + n_written:04d}.parquet")
        del agg
        gc.collect()
        n_written += 1
    return n_written


def _merge_aggregate(
    tmp_dir: Path,
    output_path: Path,
    *,
    sort_output: bool,
    fan_in: int = _MERGE_FAN_IN,
) -> None:
    """Re-aggregate intermediate parquets in bounded passes.

    Each pass groups at most ``fan_in`` files, collects, re-aggregates, and
    writes a merged file.  Repeats until the file count is small enough for a
    single final pass that writes ``output_path``.
    """
    current_dir = tmp_dir
    paths = sorted(str(p) for p in current_dir.glob("*.parquet"))

    if not paths:
        logger.warning("No data survived filtering; writing empty output.")
        pl.DataFrame(
            schema={
                "account_identifier": pl.Utf8,
                "zip_code": pl.Utf8,
                "hour_chicago": pl.Datetime("us"),
                "kwh_hour": pl.Float64,
            }
        ).write_parquet(output_path)
        return

    pass_num = 0
    while len(paths) > fan_in:
        pass_num += 1
        next_dir = tmp_dir / f"pass{pass_num}"
        next_dir.mkdir()
        n_groups = (len(paths) + fan_in - 1) // fan_in
        logger.info(
            "Merge pass %d: %d files -> %d groups of <=%d",
            pass_num,
            len(paths),
            n_groups,
            fan_in,
        )
        for g in range(n_groups):
            group = paths[g * fan_in : (g + 1) * fan_in]
            dest = next_dir / f"merged_{g:04d}.parquet"
            pl.scan_parquet(group).group_by(_GROUP_KEYS).agg(pl.col("kwh_hour").sum()).sink_parquet(dest)
            gc.collect()

        # Free previous tier (but not the top-level tmp_dir — caller owns that)
        if current_dir != tmp_dir:
            shutil.rmtree(current_dir, ignore_errors=True)
        current_dir = next_dir
        paths = sorted(str(p) for p in current_dir.glob("*.parquet"))

    logger.info("Final merge of %d files", len(paths))
    lf = pl.scan_parquet(paths).group_by(_GROUP_KEYS).agg(pl.col("kwh_hour").sum())
    if sort_output:
        lf = lf.sort(_GROUP_KEYS)
    lf.sink_parquet(output_path)


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

    account_filter: pl.Series | None = None
    if assignments_path is not None:
        account_filter = _prepare_account_filter(assignments_path)
        logger.info(
            "Restricting to %d accounts from cluster assignments: %s",
            len(account_filter),
            assignments_path,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="hourly_loads_"))
    try:
        chunk_offset = 0
        for i, path in enumerate(scan_paths):
            logger.info("Aggregating file %d/%d in batches: %s", i + 1, len(scan_paths), path)
            n = _aggregate_file_chunked(path, account_filter, tmp_dir, chunk_offset)
            logger.info("  -> wrote %d intermediate chunks", n)
            chunk_offset += n

        _merge_aggregate(tmp_dir, output_path, sort_output=sort_output)
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
