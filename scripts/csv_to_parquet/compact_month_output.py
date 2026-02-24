#!/usr/bin/env python3
"""Month-level Parquet compaction for the ComEd CSV→Parquet pipeline.

Compacts all ``batch_*.parquet`` files produced by ``migrate_month_runner.py``
into deterministic ``compacted_NNNN.parquet`` files targeting ~1 GiB each.
Invoked by the runner after all batches for a month complete with zero failures.

Design decisions
----------------
1. **Memory-safe file-by-file streaming** — reads one batch Parquet file at a
   time, accumulates rows until a per-row-count budget is reached, then flushes.
   Maximum in-memory footprint is approximately two batch files simultaneously
   (current file + carry-over slice from the previous boundary).  No global
   collect of the entire month's data is performed.

2. **No re-sort** — relies on the invariant (enforced by the runner) that batch
   files are already globally sorted by ``(zip_code, account_identifier,
   datetime)`` and that lexicographic filename order (batch_0000 < batch_0001
   < …) preserves global sort across file boundaries.  The adjacent-key
   validation pass verifies this contract on the *output* files.

3. **Atomic directory swap** — staging output is written under
   ``<run_dir>/compaction_staging/year=YYYY/month=MM/``.  After validation the
   original month directory is renamed to ``month=MM_precompact_<run_id>`` and
   the staging directory is renamed to the canonical month directory.  Both
   renames use ``os.replace()`` (single rename(2) syscall on Linux), which is
   atomic within the same filesystem.  A failed phase-2 rename triggers an
   automatic rollback of phase 1.

4. **Fail-loud** — any validation failure raises ``RuntimeError`` before the
   atomic swap, leaving the original batch files completely untouched.

5. **Audit trail** — five JSON artifacts are written under
   ``<run_dir>/compaction/`` regardless of outcome (where possible).

6. **Self-contained** — this module does NOT import from
   ``migrate_month_runner`` to avoid circular imports (the runner imports this
   module at its top level).  Shared constants (``FINAL_LONG_COLS``,
   ``SORT_KEYS``) are re-declared here with a cross-reference comment.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Canonical schema contract
# These constants MUST stay in sync with migrate_month_runner.py and
# validate_month_output.py.  They are re-declared here (not imported) to
# prevent a circular import: the runner imports this module at top level, so
# this module cannot import the runner.
# ---------------------------------------------------------------------------

FINAL_LONG_COLS: tuple[str, ...] = (
    "zip_code",
    "delivery_service_class",
    "delivery_service_name",
    "account_identifier",
    "datetime",
    "energy_kwh",
    "plc_value",
    "nspl_value",
    "year",
    "month",
)

SORT_KEYS: tuple[str, str, str] = ("zip_code", "account_identifier", "datetime")

DEFAULT_COMPACT_TARGET_SIZE_BYTES: int = 1_073_741_824  # 1 GiB

JsonDict = dict[str, Any]

STREAMING_VALIDATOR_VERSION: str = "2.0.0-streaming-pyarrow"
DEFAULT_VALIDATION_BATCH_SIZE: int = 1_000_000  # rows per PyArrow iter_batches call


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactionConfig:
    """Immutable configuration for one compaction run.

    Constructed by ``migrate_month_runner.main()`` from parsed CLI args and the
    existing ``RunnerConfig``, then passed to ``run_compaction()``.
    """

    year_month: str  # YYYYMM — must match the runner's target month
    run_id: str  # same run_id as the enclosing migration run
    out_root: Path  # dataset root (Hive partitions live here)
    run_dir: Path  # _runs/YYYYMM/<run_id>/ — audit artifacts go here
    target_size_bytes: int  # target on-disk size per output Parquet file
    max_files: int | None  # optional cap on number of output compacted files
    overwrite: bool  # allow overwriting existing compacted_*.parquet
    dry_run: bool  # plan only: write plan + original inventory + summary; skip write/validate/swap
    no_swap: bool  # run compaction + validation + write artifacts; skip atomic swap
    validation_batch_size: int = DEFAULT_VALIDATION_BATCH_SIZE  # rows per batch for streaming validation


# ---------------------------------------------------------------------------
# Utilities (self-contained; no runner imports)
# ---------------------------------------------------------------------------


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _elapsed_ms(t0: float, t1: float) -> int:
    return round((t1 - t0) * 1000.0)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _try_git_sha() -> str | None:
    """Return the current git HEAD SHA, or None if git is unavailable."""
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        return cp.stdout.strip() if cp.returncode == 0 else None
    except Exception:
        return None


def _year_month_dirs(year_month: str) -> tuple[str, str]:
    y = int(year_month[:4])
    m = int(year_month[4:6])
    return f"year={y:04d}", f"month={m:02d}"


def _file_list_hash(paths: list[Path]) -> str:
    """Stable SHA-256 fingerprint of a sorted list of file paths."""
    content = "\n".join(str(p) for p in paths)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Parquet metadata helpers — row counts from file footer; no data loaded
# ---------------------------------------------------------------------------


def _parquet_row_count(path: Path) -> int:
    """Read row count from Parquet file footer metadata without loading data."""
    return pq.read_metadata(str(path)).num_rows


def _parquet_schema_names(path: Path) -> list[str]:
    """Read column names from Parquet file schema without loading data."""
    arrow_schema = pq.read_schema(str(path))
    return list(arrow_schema.names)


def _file_inventory_entry(path: Path) -> JsonDict:
    """Build a metadata inventory entry for a single Parquet file."""
    meta = pq.read_metadata(str(path))
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "num_rows": int(meta.num_rows),
        "num_row_groups": int(meta.num_row_groups),
    }


# ---------------------------------------------------------------------------
# Core: write one compacted chunk to staging
# ---------------------------------------------------------------------------


def _write_compacted_chunk(
    df: pl.DataFrame,
    staging_month_dir: Path,
    chunk_idx: int,
    logger: Any,
    log_ctx: JsonDict,
) -> Path:
    """Write one compacted Parquet chunk to the staging directory.

    Naming convention: ``compacted_0000.parquet``, ``compacted_0001.parquet``, …
    Uses snappy compression and Parquet statistics to match the runner's
    batch output format.  Caller is responsible for enforcing ``max_files``.
    """
    t0 = time.time()
    fname = f"compacted_{chunk_idx:04d}.parquet"
    out_path = staging_month_dir / fname
    staging_month_dir.mkdir(parents=True, exist_ok=True)

    df.write_parquet(str(out_path), compression="snappy", statistics=True, use_pyarrow=False)
    size = int(out_path.stat().st_size)

    logger.log({
        **log_ctx,
        "event": "compaction_write_file",
        "status": "info",
        "chunk_idx": chunk_idx,
        "chunk_path": str(out_path),
        "num_rows": df.height,
        "size_bytes": size,
        "elapsed_ms": _elapsed_ms(t0, time.time()),
    })
    return out_path


# ---------------------------------------------------------------------------
# Core: memory-safe streaming write (file-by-file accumulation)
# ---------------------------------------------------------------------------


def _stream_write_chunks(
    sorted_input_files: list[Path],
    staging_month_dir: Path,
    rows_per_chunk: int,
    max_files: int | None,
    logger: Any,
    log_ctx: JsonDict,
) -> list[Path]:
    """Stream through sorted batch files and write compacted output chunks.

    Memory-safety guarantee
    -----------------------
    Reads one batch Parquet file at a time.  Maintains a ``carry`` DataFrame
    of rows that didn't fill the previous chunk.  Maximum RSS footprint is
    approximately ``sizeof(current_file_df) + sizeof(carry_df)``.

    No global sort is performed here.  The function relies on the runner's
    invariant that input batch files are already globally sorted by SORT_KEYS
    and that lexicographic filename order preserves global sort order.  The
    post-write adjacent-key validation pass verifies this contract.

    Chunk boundary semantics
    ------------------------
    When ``buffered_rows >= rows_per_chunk``, exactly ``rows_per_chunk`` rows
    are written and the remainder is carried forward.  The final partial chunk
    (if any) is always written regardless of size.

    Parameters
    ----------
    sorted_input_files:
        Input batch Parquet files in lexicographic order (deterministic).
    staging_month_dir:
        Directory to write ``compacted_NNNN.parquet`` files.
    rows_per_chunk:
        Target row count per output file derived from ``target_size_bytes``.
    max_files:
        Optional hard cap on the number of output files written.
    logger:
        JsonlLogger-compatible object (``logger.log(dict)``).
    log_ctx:
        Context fields included in every log event.

    Returns
    -------
    list[Path]
        Paths of written output Parquet files in write order.
    """
    output_files: list[Path] = []
    chunk_idx = 0
    carry: pl.DataFrame | None = None  # rows that didn't fill the last chunk

    logger.log({
        **log_ctx,
        "event": "compaction_scan_start",
        "status": "info",
        "n_input_files": len(sorted_input_files),
        "rows_per_chunk": rows_per_chunk,
    })

    for input_path in sorted_input_files:
        if max_files is not None and chunk_idx >= max_files:
            break

        # Materialize one batch file.  Batch files are bounded in size by
        # batch_size (default 50 CSVs each), so this does not blow up RAM.
        df = pl.read_parquet(str(input_path))

        # Prepend any carry-over rows from the previous file boundary.
        if carry is not None and carry.height > 0:
            df = pl.concat([carry, df], how="vertical", rechunk=False)
            carry = None

        # Flush complete chunks (loop handles rare case where a single file
        # exceeds the row budget, e.g. a very large batch).
        while df.height >= rows_per_chunk:
            if max_files is not None and chunk_idx >= max_files:
                break
            chunk_df = df.slice(0, rows_per_chunk)
            out_path = _write_compacted_chunk(chunk_df, staging_month_dir, chunk_idx, logger, log_ctx)
            output_files.append(out_path)
            chunk_idx += 1
            df = df.slice(rows_per_chunk)  # remainder; may be empty

        # Hold remainder as carry for the next file.
        carry = df if df.height > 0 else None

    # Final partial chunk (always written; may be smaller than rows_per_chunk).
    if carry is not None and carry.height > 0 and (max_files is None or chunk_idx < max_files):
        out_path = _write_compacted_chunk(carry, staging_month_dir, chunk_idx, logger, log_ctx)
        output_files.append(out_path)

    return output_files


# ---------------------------------------------------------------------------
# Validation: pre-flight check for existing compacted files
# ---------------------------------------------------------------------------


def _pre_validate_no_existing_compacted(
    canonical_dir: Path,
    overwrite: bool,
) -> None:
    """Fail-loud if compacted_*.parquet already exist in the canonical dir.

    Called early in run_compaction() before any writes to prevent accidental
    overwrites. Skipped when ``overwrite=True``.
    """
    if overwrite:
        return
    existing = sorted(canonical_dir.glob("compacted_*.parquet"))
    if existing:
        names = [p.name for p in existing[:5]]
        suffix = f" (and {len(existing) - 5} more)" if len(existing) > 5 else ""
        raise RuntimeError(
            f"Canonical directory already contains {len(existing)} compacted_*.parquet "
            f"file(s): {names}{suffix}. Use --overwrite-compact to allow replacement, "
            f"or remove them manually before re-running compaction."
        )


# ---------------------------------------------------------------------------
# Validation: adjacent-key sort order and uniqueness — streaming (no group_by)
# ---------------------------------------------------------------------------


def _validate_adjacent_keys_streaming(
    sorted_files: list[Path],
    label: str,
    batch_size: int = DEFAULT_VALIDATION_BATCH_SIZE,
    logger: Any | None = None,
    log_ctx: JsonDict | None = None,
) -> JsonDict:
    """Validate global sort order and key uniqueness via true streaming batches.

    Algorithm
    ---------
    Uses ``pyarrow.ParquetFile.iter_batches()`` to read only the three key
    columns in fixed-size batches.  Memory is bounded to one batch (~1M rows
    x 3 columns ~30-50 MB) plus a single carry-forward key tuple.

    For each batch:

    1. **Cross-boundary check** — first key of current batch must be strictly
       greater than ``prev_key`` (last key of previous batch/file).
    2. **Within-batch check** — vectorized Polars ``shift(1)`` on the three
       key columns, bounded to ``batch_size`` rows.  No ``group_by``, no
       Python-level row iteration.

    The compound sort key is ``(zip_code, account_identifier, datetime)``
    with lexicographic ordering.

    Parameters
    ----------
    sorted_files :
        Files to validate, in logical sort order (filename order).
    label :
        Human-readable name for the dataset being validated.
    batch_size :
        Rows per PyArrow batch.  Default 1,000,000.
    logger :
        Optional JsonlLogger for structured event logging.
    log_ctx :
        Optional base dict merged into every log event.

    Returns
    -------
    dict with keys: ``passed``, ``error``, ``error_location``, ``n_files``,
    ``total_rows``, ``key_min``, ``key_max``, ``validator_version``,
    ``validator_method``, ``batch_size``.
    """
    KEY_COLS: list[str] = list(SORT_KEYS)
    _log_ctx: JsonDict = log_ctx or {}

    prev_key: tuple[Any, ...] | None = None
    total_rows: int = 0
    n_files: int = 0
    key_min: tuple[Any, ...] | None = None
    key_max: tuple[Any, ...] | None = None
    error: str | None = None
    error_location: JsonDict | None = None

    for file_idx, path in enumerate(sorted_files):
        pf = pq.ParquetFile(str(path))
        n_files += 1
        file_rows: int = 0
        batch_idx: int = 0

        if logger is not None:
            logger.log({
                **_log_ctx,
                "event": "validation_file_start",
                "status": "info",
                "file": path.name,
                "file_idx": file_idx,
                "n_row_groups": pf.metadata.num_row_groups,
            })

        for batch in pf.iter_batches(batch_size=batch_size, columns=KEY_COLS):
            n = batch.num_rows
            if n == 0:
                batch_idx += 1
                continue

            # Convert PyArrow RecordBatch → Polars DataFrame (zero-copy where possible).
            df = pl.from_arrow(batch)

            # First/last key as Python tuples — constant cost per batch.
            first_row = df.row(0)
            last_row = df.row(df.height - 1)
            batch_first_key: tuple[Any, ...] = (first_row[0], first_row[1], first_row[2])
            batch_last_key: tuple[Any, ...] = (last_row[0], last_row[1], last_row[2])

            if key_min is None:
                key_min = batch_first_key
            key_max = batch_last_key

            # ── Cross-boundary check ──────────────────────────────────────
            if prev_key is not None:
                if batch_first_key < prev_key:
                    error = (
                        f"{label}: sort violation at boundary: "
                        f"file={path.name} batch={batch_idx} "
                        f"first_key={batch_first_key!r} < prev_key={prev_key!r}"
                    )
                    error_location = {
                        "file": path.name,
                        "file_idx": file_idx,
                        "batch_idx": batch_idx,
                        "row_offset_in_file": file_rows,
                        "row_offset_global": total_rows + file_rows,
                    }
                    break
                if batch_first_key == prev_key:
                    error = (
                        f"{label}: duplicate key at boundary: "
                        f"file={path.name} batch={batch_idx} "
                        f"key={batch_first_key!r}"
                    )
                    error_location = {
                        "file": path.name,
                        "file_idx": file_idx,
                        "batch_idx": batch_idx,
                        "row_offset_in_file": file_rows,
                        "row_offset_global": total_rows + file_rows,
                    }
                    break

            # ── Within-batch adjacent-pair check (vectorized, bounded) ────
            if n > 1:
                zip_prev = pl.col("zip_code").shift(1)
                acct_prev = pl.col("account_identifier").shift(1)
                dt_prev = pl.col("datetime").shift(1)

                zip_eq = zip_prev == pl.col("zip_code")
                acct_eq = acct_prev == pl.col("account_identifier")
                dt_eq = dt_prev == pl.col("datetime")

                zip_gt = zip_prev > pl.col("zip_code")
                acct_gt = acct_prev > pl.col("account_identifier")
                dt_gt = dt_prev > pl.col("datetime")

                sort_violation = zip_gt | (zip_eq & acct_gt) | (zip_eq & acct_eq & dt_gt)
                dup_violation = zip_eq & acct_eq & dt_eq

                check = (
                    df.with_row_index("_row_idx")
                    .with_columns([
                        sort_violation.alias("_sort_viol"),
                        dup_violation.alias("_dup_viol"),
                    ])
                    .slice(1)  # row 0 has null shift values
                )

                sort_bad = check.filter(pl.col("_sort_viol")).head(1)
                if sort_bad.height > 0:
                    bad = sort_bad.select([*KEY_COLS, "_row_idx"]).to_dicts()[0]
                    row_in_file = file_rows + int(bad["_row_idx"])
                    error = (
                        f"{label}: sort violation in file={path.name} "
                        f"batch={batch_idx} row_in_file={row_in_file} "
                        f"bad_row={bad}"
                    )
                    error_location = {
                        "file": path.name,
                        "file_idx": file_idx,
                        "batch_idx": batch_idx,
                        "row_in_batch": int(bad["_row_idx"]),
                        "row_offset_in_file": row_in_file,
                        "row_offset_global": total_rows + row_in_file,
                        "bad_key": {k: str(bad[k]) for k in KEY_COLS},
                    }
                    break

                dup_bad = check.filter(pl.col("_dup_viol")).head(1)
                if dup_bad.height > 0:
                    bad = dup_bad.select([*KEY_COLS, "_row_idx"]).to_dicts()[0]
                    row_in_file = file_rows + int(bad["_row_idx"])
                    error = (
                        f"{label}: duplicate key in file={path.name} "
                        f"batch={batch_idx} row_in_file={row_in_file} "
                        f"bad_row={bad}"
                    )
                    error_location = {
                        "file": path.name,
                        "file_idx": file_idx,
                        "batch_idx": batch_idx,
                        "row_in_batch": int(bad["_row_idx"]),
                        "row_offset_in_file": row_in_file,
                        "row_offset_global": total_rows + row_in_file,
                        "bad_key": {k: str(bad[k]) for k in KEY_COLS},
                    }
                    break

            prev_key = batch_last_key
            file_rows += n
            batch_idx += 1

        # Count rows scanned even on partial files.
        total_rows += file_rows

        if error is not None:
            break

        if logger is not None:
            logger.log({
                **_log_ctx,
                "event": "validation_file_end",
                "status": "info",
                "file": path.name,
                "file_idx": file_idx,
                "file_rows": file_rows,
                "running_total_rows": total_rows,
            })

    return {
        "passed": error is None,
        "error": error,
        "error_location": error_location,
        "n_files": n_files,
        "total_rows": total_rows,
        "key_min": str(key_min) if key_min is not None else None,
        "key_max": str(key_max) if key_max is not None else None,
        "validator_version": STREAMING_VALIDATOR_VERSION,
        "validator_method": "adjacent_key_streaming_pyarrow_iter_batches",
        "batch_size": batch_size,
    }


# ---------------------------------------------------------------------------
# Validation: schema conformance
# ---------------------------------------------------------------------------


def _validate_schema(files: list[Path], label: str) -> JsonDict:
    """Verify every file has exactly FINAL_LONG_COLS in canonical column order.

    Uses Parquet file footer metadata — no row data is loaded.
    """
    errors: list[str] = []
    for path in files:
        actual = _parquet_schema_names(path)
        # pyarrow may surface internal ``__null_dask_index__`` or similar names;
        # filter them out before comparing.
        actual_clean = [c for c in actual if not c.startswith("__")]
        if tuple(actual_clean) != FINAL_LONG_COLS:
            errors.append(f"{path.name}: expected {list(FINAL_LONG_COLS)} got {actual_clean}")
    return {"passed": len(errors) == 0, "errors": errors, "label": label}


# ---------------------------------------------------------------------------
# Validation: partition uniformity
# ---------------------------------------------------------------------------


def _validate_partition_uniformity(files: list[Path], year_month: str) -> JsonDict:
    """Verify all rows belong to the target (year, month) partition.

    Reads only the ``year`` and ``month`` columns.  A violation here would mean
    cross-month row leakage — a serious data corruption that must block swap.
    """
    y = int(year_month[:4])
    m = int(year_month[4:6])
    errors: list[str] = []
    for path in files:
        df = pl.read_parquet(str(path), columns=["year", "month"])
        bad = df.filter((pl.col("year") != y) | (pl.col("month") != m)).height
        if bad > 0:
            errors.append(f"{path.name}: {bad} rows with wrong (year,month); expected ({y},{m})")
    return {"passed": len(errors) == 0, "errors": errors, "year": y, "month": m}


# ---------------------------------------------------------------------------
# Atomic directory swap
# ---------------------------------------------------------------------------


def _atomic_swap(
    month_canonical_dir: Path,
    staging_month_dir: Path,
    precompact_dir: Path,
    logger: Any,
    log_ctx: JsonDict,
) -> None:
    """Atomically swap staging compacted files into the canonical month directory.

    Two-phase rename sequence
    -------------------------
    Phase 1 (atomic):
        ``os.replace(month_canonical_dir → precompact_dir)``
        The canonical directory disappears; batch files are now in
        ``precompact_dir`` and remain safe.

    Phase 2 (atomic):
        ``os.replace(staging_month_dir → month_canonical_dir)``
        Staging becomes the new canonical directory.

    Rollback:
        If phase 2 fails, phase 1 is reversed by renaming ``precompact_dir``
        back to ``month_canonical_dir``.  If the rollback also fails, a
        ``RuntimeError`` with manual recovery instructions is raised — the
        original files are in ``precompact_dir`` and can be restored manually.

    Filesystem constraint:
        ``os.replace()`` is atomic only within the same filesystem.  We
        assert both directories share the same device before proceeding.
        Cross-device moves are refused with a clear error.

    Note: this function must NEVER touch ``<out_root>/_runs/`` — the run
    artifacts directory is completely separate from Hive partition directories.
    """
    # Guard: same-filesystem requirement for atomic rename(2).
    canonical_dev = os.stat(str(month_canonical_dir.parent)).st_dev
    staging_dev = os.stat(str(staging_month_dir.parent)).st_dev
    if canonical_dev != staging_dev:
        raise RuntimeError(
            f"Staging and canonical parent directories are on different "
            f"filesystems (canonical_dev={canonical_dev}, "
            f"staging_dev={staging_dev}). Atomic rename is not possible. "
            f"Move staging under the same mount point as the output root."
        )

    logger.log({
        **log_ctx,
        "event": "compaction_atomic_swap",
        "status": "start",
        "month_canonical_dir": str(month_canonical_dir),
        "staging_month_dir": str(staging_month_dir),
        "precompact_dir": str(precompact_dir),
    })

    # Phase 1: canonical → precompact  (atomic rename; canonical dir vanishes).
    os.replace(str(month_canonical_dir), str(precompact_dir))

    try:
        # Phase 2: staging → canonical  (atomic rename; staging dir vanishes).
        os.replace(str(staging_month_dir), str(month_canonical_dir))

    except Exception as swap_err:
        # Phase 2 failed.  Attempt rollback of phase 1 to restore original state.
        try:
            os.replace(str(precompact_dir), str(month_canonical_dir))
        except Exception as rollback_err:
            # Both phase-2 and rollback failed.  The original batch files are
            # in precompact_dir; manual intervention is required.
            raise RuntimeError(
                "CRITICAL: compaction phase-2 swap failed AND rollback failed. "
                "Manual recovery required: "
                f"rename {precompact_dir} → {month_canonical_dir}. "
                f"swap_err={swap_err!r} rollback_err={rollback_err!r}"
            ) from swap_err

        # Rollback succeeded; original batch files restored.
        raise RuntimeError(
            f"Compaction phase-2 swap failed (rollback succeeded; original "
            f"batch files are intact): {swap_err!r}. "
            f"Staging files remain at: {staging_month_dir.parent}"
        ) from swap_err

    logger.log({
        **log_ctx,
        "event": "compaction_atomic_swap",
        "status": "success",
        "month_canonical_dir": str(month_canonical_dir),
        "precompact_dir": str(precompact_dir),
    })


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_compaction(cfg: CompactionConfig, logger: Any) -> JsonDict:
    """Run month-level Parquet compaction.

    Caller contract
    ---------------
    - All batches for ``cfg.year_month`` completed with ``total_failure == 0``.
    - The cooperative stop flag is not set.
    - ``logger`` is a ``JsonlLogger``-compatible object exposing ``.log(dict)``.

    Stages
    ------
    1. Locate and sort input ``batch_*.parquet`` files (lexicographic order).
    2. Idempotency guard (fail if ``compacted_*`` files exist without
       ``--overwrite-compact``).
    3. Read pre-compaction row counts from Parquet footer metadata (no I/O).
    4. Validate input schema against ``FINAL_LONG_COLS``.
    5. Estimate ``rows_per_chunk`` from disk size ÷ row count.
    6. Write audit ``compaction_plan.json`` and ``original_file_inventory.json``.
    7. Stream-write compacted chunks to staging directory.
    8. Post-write validations: row count, schema, sort order, duplicates,
       partition uniformity.
    9. Write audit ``compaction_validation.json`` and
       ``compacted_file_inventory.json``.
    10. Atomic directory swap (skip if ``cfg.dry_run``).
    11. Write audit ``compaction_summary.json``.

    Returns
    -------
    Summary dict (also written to ``<run_dir>/compaction/compaction_summary.json``).

    Raises
    ------
    RuntimeError
        On any unrecoverable failure.  Staging files are cleaned up before
        raising.  Original batch files are never modified.
    """
    t_start = time.time()
    git_sha = _try_git_sha()
    ydir, mdir = _year_month_dirs(cfg.year_month)

    month_canonical_dir = cfg.out_root / ydir / mdir
    staging_base = cfg.run_dir / "compaction_staging"
    staging_month_dir = staging_base / ydir / mdir
    audit_dir = cfg.run_dir / "compaction"
    # Precompact backup name: month=MM_precompact_<run_id> — lives adjacent to
    # the canonical month dir so it is on the same filesystem.
    precompact_dir = month_canonical_dir.parent / f"{mdir}_precompact_{cfg.run_id}"

    log_ctx: JsonDict = {
        "ts_utc": _now_utc_iso(),
        "year_month": cfg.year_month,
        "run_id": cfg.run_id,
    }

    logger.log({
        **log_ctx,
        "event": "compaction_start",
        "status": "start",
        "month_canonical_dir": str(month_canonical_dir),
        "staging_month_dir": str(staging_month_dir),
        "audit_dir": str(audit_dir),
        "target_size_bytes": cfg.target_size_bytes,
        "dry_run": cfg.dry_run,
        "overwrite": cfg.overwrite,
    })

    # ── 1. Locate and sort input files ──────────────────────────────────────
    if not month_canonical_dir.exists():
        raise RuntimeError(f"Month directory does not exist: {month_canonical_dir}")

    all_parquet = sorted(month_canonical_dir.glob("*.parquet"))
    if not all_parquet:
        raise RuntimeError(f"No .parquet files found in {month_canonical_dir}")

    # ── 2. Idempotency guard ─────────────────────────────────────────────────
    existing_compacted = [p for p in all_parquet if p.name.startswith("compacted_")]
    if existing_compacted and not cfg.overwrite:
        raise RuntimeError(
            f"Compacted files already exist in {month_canonical_dir}: "
            f"{[p.name for p in existing_compacted]}. "
            f"Pass --overwrite-compact to re-compact."
        )

    # Input files: only batch_*.parquet (never pre-existing compacted_* files).
    input_files: list[Path] = sorted(p for p in all_parquet if p.name.startswith("batch_"))
    if not input_files:
        raise RuntimeError(
            f"No batch_*.parquet input files found in {month_canonical_dir}. "
            f"All files present: {[p.name for p in all_parquet]}"
        )

    # ── 3. Pre-compaction row count from Parquet footer metadata ─────────────
    pre_rows = sum(_parquet_row_count(p) for p in input_files)
    total_input_bytes = sum(p.stat().st_size for p in input_files)

    if pre_rows == 0:
        raise RuntimeError("Input batch files contain zero rows; nothing to compact.")

    # ── 4. Input schema validation ───────────────────────────────────────────
    input_schema_result = _validate_schema(input_files, label="input")
    if not input_schema_result["passed"]:
        raise RuntimeError(f"Input schema validation failed: {input_schema_result['errors']}")

    # ── Pre-flight: check for existing compacted files in canonical dir ──
    _pre_validate_no_existing_compacted(month_canonical_dir, cfg.overwrite)

    # ── 5. Estimate rows_per_chunk ───────────────────────────────────────────
    # We target on-disk chunk size.  bytes_per_row is estimated from the
    # actual compressed Parquet data, so the ratio accurately reflects
    # how many rows fit in target_size_bytes of Parquet output.
    bytes_per_row: float = total_input_bytes / pre_rows
    rows_per_chunk: int = max(1, round(cfg.target_size_bytes / bytes_per_row))

    # ── 6. Write audit plan ──────────────────────────────────────────────────
    audit_dir.mkdir(parents=True, exist_ok=True)
    original_inventory = [_file_inventory_entry(p) for p in input_files]
    compaction_plan: JsonDict = {
        "ts_utc": _now_utc_iso(),
        "git_sha": git_sha,
        "year_month": cfg.year_month,
        "run_id": cfg.run_id,
        "month_canonical_dir": str(month_canonical_dir),
        "staging_month_dir": str(staging_month_dir),
        "precompact_dir": str(precompact_dir),
        "n_input_files": len(input_files),
        "pre_rows": pre_rows,
        "total_input_bytes": total_input_bytes,
        "bytes_per_row_estimate": round(bytes_per_row, 4),
        "rows_per_chunk": rows_per_chunk,
        "target_size_bytes": cfg.target_size_bytes,
        "max_files": cfg.max_files,
        "dry_run": cfg.dry_run,
        "overwrite": cfg.overwrite,
        "sort_keys": list(SORT_KEYS),
        "final_long_cols": list(FINAL_LONG_COLS),
        "input_file_list_hash": _file_list_hash(input_files),
    }
    _write_json(audit_dir / "compaction_plan.json", compaction_plan)
    _write_json(audit_dir / "original_file_inventory.json", original_inventory)

    # ── Dry-run: stop before writing any data ────────────────────────────────
    if cfg.dry_run:
        logger.log({
            **log_ctx,
            "event": "compaction_complete",
            "status": "info",
            "msg": "dry_run=True; validation plan written; swap skipped",
            "pre_rows": pre_rows,
            "rows_per_chunk": rows_per_chunk,
            "n_input_files": len(input_files),
        })
        dry_summary: JsonDict = {
            "ts_utc": _now_utc_iso(),
            "git_sha": git_sha,
            "year_month": cfg.year_month,
            "run_id": cfg.run_id,
            "status": "dry_run",
            "pre_rows": pre_rows,
            "n_input_files": len(input_files),
            "rows_per_chunk": rows_per_chunk,
            "target_size_bytes": cfg.target_size_bytes,
            "elapsed_ms": _elapsed_ms(t_start, time.time()),
        }
        _write_json(audit_dir / "compaction_summary.json", dry_summary)
        return dry_summary

    # ── 7. Stream-write compacted chunks to staging ──────────────────────────
    staging_month_dir.mkdir(parents=True, exist_ok=True)
    try:
        output_files = _stream_write_chunks(
            sorted_input_files=input_files,
            staging_month_dir=staging_month_dir,
            rows_per_chunk=rows_per_chunk,
            max_files=cfg.max_files,
            logger=logger,
            log_ctx=log_ctx,
        )
    except Exception as write_err:
        logger.log({
            **log_ctx,
            "event": "compaction_failure",
            "status": "failure",
            "phase": "write",
            "exception_type": type(write_err).__name__,
            "exception_msg": str(write_err),
            "traceback": traceback.format_exc(),
        })
        shutil.rmtree(str(staging_month_dir), ignore_errors=True)
        raise RuntimeError(f"Compaction write phase failed: {write_err}") from write_err

    if not output_files:
        shutil.rmtree(str(staging_month_dir), ignore_errors=True)
        raise RuntimeError("Compaction produced zero output files; aborting before swap.")

    # ── 8. Post-write validations ────────────────────────────────────────────
    # Row count: sum footer metadata (no data load).
    post_rows = sum(_parquet_row_count(p) for p in output_files)
    row_count_ok = post_rows == pre_rows

    output_schema_result = _validate_schema(output_files, label="output")
    sort_dup_result = _validate_adjacent_keys_streaming(
        output_files,
        label="compacted",
        batch_size=cfg.validation_batch_size,
        logger=logger,
        log_ctx=log_ctx,
    )
    partition_result = _validate_partition_uniformity(output_files, cfg.year_month)

    total_output_bytes_staging = sum(p.stat().st_size for p in output_files)

    # ── 9. Write validation audit artifacts ──────────────────────────────────
    compacted_inventory = [_file_inventory_entry(p) for p in output_files]
    all_passed = (
        row_count_ok and output_schema_result["passed"] and sort_dup_result["passed"] and partition_result["passed"]
    )
    validation_result: JsonDict = {
        "ts_utc": _now_utc_iso(),
        "git_sha": git_sha,
        "year_month": cfg.year_month,
        "run_id": cfg.run_id,
        "pre_rows": pre_rows,
        "post_rows": post_rows,
        "row_count_match": row_count_ok,
        "schema_validation": output_schema_result,
        "sort_dup_validation": sort_dup_result,
        "partition_validation": partition_result,
        "passed": all_passed,
    }
    _write_json(audit_dir / "compaction_validation.json", validation_result)
    _write_json(audit_dir / "compacted_file_inventory.json", compacted_inventory)

    if not all_passed:
        failure_reasons: list[str] = []
        if not row_count_ok:
            failure_reasons.append(f"row_count_mismatch: pre={pre_rows} post={post_rows}")
        if not output_schema_result["passed"]:
            failure_reasons.append(f"schema: {output_schema_result['errors']}")
        if not sort_dup_result["passed"]:
            failure_reasons.append(f"sort_or_dup: {sort_dup_result['error']}")
        if not partition_result["passed"]:
            failure_reasons.append(f"partition: {partition_result['errors']}")

        logger.log({
            **log_ctx,
            "event": "compaction_failure",
            "status": "failure",
            "phase": "validation",
            "reasons": failure_reasons,
        })
        shutil.rmtree(str(staging_month_dir), ignore_errors=True)
        raise RuntimeError(
            f"Compaction post-write validation failed: {failure_reasons}. "
            f"Original batch files are untouched at {month_canonical_dir}."
        )

    logger.log({
        **log_ctx,
        "event": "compaction_validation_pass",
        "status": "success",
        "pre_rows": pre_rows,
        "post_rows": post_rows,
        "n_output_files": len(output_files),
        "total_output_bytes_staging": total_output_bytes_staging,
    })

    # ── No-swap mode: keep staged outputs, skip atomic swap ──────────────────
    if cfg.no_swap:
        t_end = time.time()
        summary: JsonDict = {
            "ts_utc": _now_utc_iso(),
            "git_sha": git_sha,
            "year_month": cfg.year_month,
            "run_id": cfg.run_id,
            "status": "no_swap",
            "n_input_files": len(input_files),
            "n_output_files": len(output_files),
            "pre_rows": pre_rows,
            "post_rows": post_rows,
            "total_input_bytes": total_input_bytes,
            "total_output_bytes_staging": total_output_bytes_staging,
            "rows_per_chunk": rows_per_chunk,
            "target_size_bytes": cfg.target_size_bytes,
            "month_canonical_dir": str(month_canonical_dir),
            "staging_month_dir": str(staging_month_dir),
            "precompact_dir": str(precompact_dir),
            "elapsed_ms": _elapsed_ms(t_start, t_end),
            "sort_keys": list(SORT_KEYS),
            "input_file_list_hash": _file_list_hash(input_files),
            "staged_output_file_list_hash": _file_list_hash(output_files),
        }
        _write_json(audit_dir / "compaction_summary.json", summary)

        logger.log({
            **log_ctx,
            "event": "compaction_complete",
            "status": "info",
            "msg": "no_swap=True; staging+validation complete; swap skipped",
            "pre_rows": pre_rows,
            "post_rows": post_rows,
            "n_input_files": len(input_files),
            "n_output_files": len(output_files),
            "total_output_bytes_staging": total_output_bytes_staging,
            "elapsed_ms": _elapsed_ms(t_start, t_end),
        })
        return summary

    # ── 10. Atomic directory swap ─────────────────────────────────────────────
    try:
        _atomic_swap(
            month_canonical_dir=month_canonical_dir,
            staging_month_dir=staging_month_dir,
            precompact_dir=precompact_dir,
            logger=logger,
            log_ctx=log_ctx,
        )
    except Exception as swap_err:
        logger.log({
            **log_ctx,
            "event": "compaction_failure",
            "status": "failure",
            "phase": "atomic_swap",
            "exception_type": type(swap_err).__name__,
            "exception_msg": str(swap_err),
            "traceback": traceback.format_exc(),
        })
        # Staging files are still in staging_month_dir (swap failed before or
        # during phase-2; _atomic_swap rolls back phase-1 automatically).
        shutil.rmtree(str(staging_month_dir), ignore_errors=True)
        raise

    # After successful swap the output files now live under month_canonical_dir.
    # Compute output bytes from the canonical location (staging dir is gone).
    output_names = [p.name for p in output_files]
    total_output_bytes = sum((month_canonical_dir / name).stat().st_size for name in output_names)

    # ── 11. Final summary ─────────────────────────────────────────────────────
    t_end = time.time()
    summary: JsonDict = {
        "ts_utc": _now_utc_iso(),
        "git_sha": git_sha,
        "year_month": cfg.year_month,
        "run_id": cfg.run_id,
        "status": "success",
        "n_input_files": len(input_files),
        "n_output_files": len(output_files),
        "pre_rows": pre_rows,
        "post_rows": post_rows,
        "total_input_bytes": total_input_bytes,
        "total_output_bytes": total_output_bytes,
        "rows_per_chunk": rows_per_chunk,
        "target_size_bytes": cfg.target_size_bytes,
        "month_canonical_dir": str(month_canonical_dir),
        "precompact_dir": str(precompact_dir),
        "elapsed_ms": _elapsed_ms(t_start, t_end),
        "sort_keys": list(SORT_KEYS),
        "input_file_list_hash": _file_list_hash(input_files),
        "output_file_list_hash": _file_list_hash([month_canonical_dir / n for n in output_names]),
    }
    _write_json(audit_dir / "compaction_summary.json", summary)

    logger.log({
        **log_ctx,
        "event": "compaction_complete",
        "status": "success",
        "pre_rows": pre_rows,
        "post_rows": post_rows,
        "n_input_files": len(input_files),
        "n_output_files": len(output_files),
        "total_output_bytes": total_output_bytes,
        "elapsed_ms": _elapsed_ms(t_start, t_end),
    })

    return summary
