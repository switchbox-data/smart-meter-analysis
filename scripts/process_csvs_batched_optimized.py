#!/usr/bin/env python3
"""Memory-optimized CSV ingestion for large file counts.

Responsibilities (and ONLY these):
- Read local ComEd CSV files from an input directory
- Convert wide-format interval columns to long format
- Add time columns (date/hour/weekday/is_weekend) eagerly
- Write a canonical, analysis-ready interval parquet used downstream
- Write a JSONL processing manifest with per-file outcomes
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from smart_meter_analysis.transformation import COMED_INTERVAL_COLUMNS, add_time_columns, transform_wide_to_long_lf

logger = logging.getLogger(__name__)

# --- Canonical output columns required downstream (Stage 1 profile builder expects these) ---
REQUIRED_ENERGY_COLS = ["zip_code", "account_identifier", "datetime", "kwh"]
REQUIRED_TIME_COLS = ["date", "hour", "weekday", "is_weekend"]

# --- ComEd input schema overrides (keep parsing strict; ignore_errors defaults False) ---
DType = Any

COMED_SCHEMA_OVERRIDES: dict[str, DType] = {
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

_INTERVAL_SCHEMA: dict[str, DType] = dict.fromkeys(COMED_INTERVAL_COLUMNS, pl.Float64)
COMED_SCHEMA: dict[str, DType] = {**COMED_SCHEMA_OVERRIDES, **_INTERVAL_SCHEMA}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _chunked(seq: list[Path], size: int) -> Iterable[list[Path]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _write_manifest_line(fp: Any, record: dict[str, Any]) -> None:
    fp.write(json.dumps(record, sort_keys=True) + "\n")
    fp.flush()


def _ensure_required_cols(df: pl.DataFrame, *, context: str) -> None:
    missing_energy = [c for c in REQUIRED_ENERGY_COLS if c not in df.columns]
    missing_time = [c for c in REQUIRED_TIME_COLS if c not in df.columns]
    if missing_energy or missing_time:
        raise ValueError(
            f"{context}: missing required columns. missing_energy={missing_energy} missing_time={missing_time}",
        )


def _canonicalize_df(df: pl.DataFrame, canonical_cols: list[str]) -> pl.DataFrame:
    """Enforce deterministic column set + order across files/sub-batches.
    If a column is missing, create it as null.
    """
    exprs: list[pl.Expr] = []
    existing = set(df.columns)
    for c in canonical_cols:
        if c in existing:
            exprs.append(pl.col(c))
        else:
            exprs.append(pl.lit(None).alias(c))
    return df.select(exprs)


def _process_one_csv_to_df(
    csv_path: Path,
    *,
    ignore_errors: bool,
    day_mode: str,
) -> pl.DataFrame:
    lf = pl.scan_csv(
        str(csv_path),
        schema_overrides=COMED_SCHEMA,
        ignore_errors=bool(ignore_errors),
    )

    lf_long = transform_wide_to_long_lf(lf)
    df_long = lf_long.collect(engine="streaming")
    df_long = add_time_columns(df_long, day_mode=day_mode)

    _ensure_required_cols(df_long, context=f"{csv_path.name}")

    # Keep only what we promise downstream (canonical interval parquet)
    out = df_long.select(REQUIRED_ENERGY_COLS + REQUIRED_TIME_COLS)
    return out


def process_csv_subbatch_to_parquet(
    csv_files: list[Path],
    *,
    batch_num: int,
    sub_num: int,
    temp_dir: Path,
    canonical_cols: list[str] | None,
    processing_manifest_fp: Any,
    ignore_errors: bool,
    max_errors: int,
    day_mode: str,
    log_every: int,
    errors_so_far: int,
) -> tuple[Path, list[str], int]:
    logger.info("  Sub-batch %d.%d: %d files", batch_num, sub_num, len(csv_files))

    dfs: list[pl.DataFrame] = []
    sub_errors = 0

    for i, csv_path in enumerate(csv_files, 1):
        if log_every > 0 and (i == 1 or i % log_every == 0 or i == len(csv_files)):
            logger.info("    Processing %d/%d in sub-batch %d.%d", i, len(csv_files), batch_num, sub_num)

        ts = _utc_now_iso()
        try:
            df = _process_one_csv_to_df(
                csv_path,
                ignore_errors=ignore_errors,
                day_mode=day_mode,
            )

            if canonical_cols is None:
                canonical_cols = df.columns

            df = _canonicalize_df(df, canonical_cols)
            dfs.append(df)

            _write_manifest_line(
                processing_manifest_fp,
                {"file": csv_path.name, "status": "success", "rows": int(df.height), "timestamp": ts},
            )

        except Exception as exc:
            sub_errors += 1
            total_errors = errors_so_far + sub_errors

            _write_manifest_line(
                processing_manifest_fp,
                {"file": csv_path.name, "status": "error", "error": f"{type(exc).__name__}: {exc}", "timestamp": ts},
            )

            msg = f"Failed to process {csv_path.name}: {type(exc).__name__}: {exc}"

            # Fail-fast unless user explicitly opted into ignore-errors mode.
            if not ignore_errors:
                raise RuntimeError(msg) from exc

            logger.warning("%s", msg)

            if total_errors > max_errors:
                raise RuntimeError(
                    f"Exceeded --max-errors={max_errors}. "
                    f"batch={batch_num} sub_batch={sub_num} total_errors={total_errors}. "
                    f"Last error: {type(exc).__name__}: {exc}",
                ) from exc

    if not dfs:
        raise RuntimeError(f"No files successfully processed in sub-batch {batch_num}.{sub_num}")

    sub_output = temp_dir / f"batch_{batch_num:04d}_sub_{sub_num:04d}.parquet"

    # In-memory concat, then write. No diagonal_relaxed: schemas are already canonicalized.
    pl.concat(dfs, how="vertical").write_parquet(sub_output)

    logger.info("  Sub-batch %d.%d complete: %s", batch_num, sub_num, sub_output)
    return sub_output, (canonical_cols or []), sub_errors


def process_csv_batch_to_parquet(
    csv_files: list[Path],
    *,
    batch_num: int,
    temp_dir: Path,
    sub_batch_size: int,
    canonical_cols: list[str] | None,
    processing_manifest_fp: Any,
    ignore_errors: bool,
    max_errors: int,
    day_mode: str,
    log_every: int,
    errors_so_far: int,
) -> tuple[Path, list[str], int]:
    logger.info("Processing batch %d: %d files", batch_num, len(csv_files))

    sub_files: list[Path] = []
    batch_errors = 0

    for sub_num, sub in enumerate(_chunked(csv_files, sub_batch_size), 1):
        sub_file, canonical_cols, sub_errors = process_csv_subbatch_to_parquet(
            csv_files=sub,
            batch_num=batch_num,
            sub_num=sub_num,
            temp_dir=temp_dir,
            canonical_cols=canonical_cols,
            processing_manifest_fp=processing_manifest_fp,
            ignore_errors=ignore_errors,
            max_errors=max_errors,
            day_mode=day_mode,
            log_every=log_every,
            errors_so_far=errors_so_far + batch_errors,
        )
        sub_files.append(sub_file)
        batch_errors += sub_errors

    batch_output = temp_dir / f"batch_{batch_num:04d}.parquet"
    logger.info("  Concatenating %d sub-batches into %s", len(sub_files), batch_output)

    pl.concat([pl.scan_parquet(str(f)) for f in sub_files], how="vertical").sink_parquet(batch_output)

    for f in sub_files:
        f.unlink(missing_ok=True)

    logger.info("Batch %d complete: %s", batch_num, batch_output)
    return batch_output, (canonical_cols or []), batch_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Process ComEd CSV files in memory-safe batches (local only).")

    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing CSV files")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet file path (interval-level long)")
    parser.add_argument(
        "--processing-manifest",
        type=Path,
        required=True,
        help="Path to write processing_manifest.jsonl (required)",
    )

    parser.add_argument("--batch-size", type=int, default=5000, help="Files per batch (default: 5000)")
    parser.add_argument("--sub-batch-size", type=int, default=100, help="Files per sub-batch (default: 100)")

    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Continue on malformed CSVs up to --max-errors (not recommended). Default: fail-fast.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Maximum errors before aborting (only meaningful with --ignore-errors). Default: 10",
    )

    parser.add_argument(
        "--day-mode",
        type=str,
        choices=["calendar", "billing"],
        default="calendar",
        help="Day attribution mode for time columns (default: calendar)",
    )

    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary parquet batches for debugging")
    parser.add_argument("--no-row-count", action="store_true", help="Skip final output row count scan")
    parser.add_argument("--log-every", type=int, default=200, help="Log progress every N files (default: 200)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    csv_files = sorted(args.input_dir.glob("*.csv"))
    logger.info("Found %d CSV files", len(csv_files))

    if not csv_files:
        logger.error("No CSV files found in %s", args.input_dir)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.processing_manifest.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = args.output.parent / "temp_batches"
    temp_dir.mkdir(parents=True, exist_ok=True)

    canonical_cols: list[str] | None = None
    batch_files: list[Path] = []
    total_errors = 0

    # Overwrite for determinism (byte-identical reruns given identical outcomes)
    with args.processing_manifest.open("w", encoding="utf-8") as mf:
        try:
            for batch_num, batch in enumerate(_chunked(csv_files, args.batch_size), 1):
                batch_file, canonical_cols, batch_errors = process_csv_batch_to_parquet(
                    csv_files=batch,
                    batch_num=batch_num,
                    temp_dir=temp_dir,
                    sub_batch_size=args.sub_batch_size,
                    canonical_cols=canonical_cols,
                    processing_manifest_fp=mf,
                    ignore_errors=bool(args.ignore_errors),
                    max_errors=int(args.max_errors),
                    day_mode=str(args.day_mode),
                    log_every=int(args.log_every),
                    errors_so_far=total_errors,
                )
                batch_files.append(batch_file)
                total_errors += batch_errors

            logger.info("Concatenating %d batch files into final output: %s", len(batch_files), args.output)
            pl.concat([pl.scan_parquet(str(f)) for f in batch_files], how="vertical").sink_parquet(args.output)

            if not args.no_row_count:
                row_count = pl.scan_parquet(args.output).select(pl.len()).collect(streaming=True)[0, 0]  # type: ignore[call-overload]
                logger.info("Success! Wrote %s records to %s", f"{row_count:,}", args.output)
            else:
                logger.info("Success! Wrote output to %s (row count skipped)", args.output)

            logger.info("File-level errors encountered: %d", total_errors)

        finally:
            if not args.keep_temp:
                for f in batch_files:
                    f.unlink(missing_ok=True)
                shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("Wrote processing manifest: %s", args.processing_manifest)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
