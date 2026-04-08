from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from smart_meter_analysis.wide_to_long import transform_wide_to_long_lf

# --------------------------------------------------------------------------------------
# Batched wide_to_long validator (resumable, JSONL checkpoints)
#
# What is this?
# - Month-scale Zip4 validation in the Docker devcontainer can wedge Docker if we do
#   a global sort or force full materialization of a full month. Many checks are
#   expensive if they require multiple passes over the dataset.
# - This script validates correctness (schema contracts, Daylight Saving Time
#   behavior, datetime bounds, and 48-interval invariants) at full-month scale
#   by processing input CSVs in bounded batches with checkpoints.
#   If the container crashes or Docker becomes unstable, we can resume without
#   redoing completed work.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchResult:
    """
    One record for one batch attempt.

    Stored as JSONL for two reasons:
      1) resume: a successful batch is a stable checkpoint (skip on rerun)
      2) audit: gives an append-only provenance trail of what was validated

    We record both configuration (strict/sort/engine) and outcomes (rows, datetime bounds),
    because "it passed" is not meaningful unless we can tie it to the exact validation mode.
    """

    run_id: str
    batch_id: str
    batch_index: int
    batch_size: int
    n_files: int
    first_path: str
    last_path: str
    started_at_utc: str
    finished_at_utc: str
    elapsed_sec: float

    strict: bool
    sort_output: bool
    engine: str
    infer_schema_length: int

    # Validation outputs
    long_rows: int | None
    long_rows_mod_48: int | None
    min_datetime: str | None
    max_datetime: str | None
    any_null_datetime: bool | None
    schema_fingerprint: str | None

    ok: bool
    error_type: str | None
    error_message: str | None


def _utc_now_iso() -> str:
    """
    Return a stable, timezone-aware UTC timestamp for logs.

    Why: datetime.utcnow() is deprecated in newer Python versions and produces naive
    datetimes. We intentionally write Zulu timestamps into JSONL to keep log output
    consistent across environments and to avoid local-time ambiguity.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_paths(list_path: str) -> list[str]:
    """
    Read an input manifest of CSV paths (S3 or local), one path per line.

    Why:
    - We want the driver/orchestrator to own discovery. This tool should be deterministic
      and replayable: given a manifest, it validates exactly that set of files.
    - We support comment lines (# ...) to allow simple manifest curation.
    """
    p = Path(list_path)
    if not p.exists():
        raise FileNotFoundError(f"Input list file not found: {list_path}")

    paths: list[str] = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        paths.append(s)

    if not paths:
        raise ValueError(f"No usable paths found in list file: {list_path}")

    return paths


def _chunk_paths(paths: Sequence[str], batch_size: int) -> list[list[str]]:
    """
    Partition the manifest into batches of bounded size.

    Why:
    - Memory and swap pressure is the primary failure mode in Docker Desktop devcontainers.
      Batching is the simplest, most reliable control for peak memory use.
    - We prefer deterministic partitioning (simple slicing) so resume behavior is stable:
      batch_00000 always contains the same files for the same manifest and batch_size.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}")
    return [list(paths[i : i + batch_size]) for i in range(0, len(paths), batch_size)]


def _schema_fingerprint(schema: pl.Schema) -> str:
    """
    Compute a stable fingerprint of the output schema (name + dtype).

    Why:
    - When validating at scale, we want a compact way to detect drift across batches.
      If one file has a surprising type coercion behavior, schema fingerprints will diverge.
    - This fingerprint is not meant to be cryptographic security; SHA256 is convenient,
      ubiquitous, and stable.
    """
    pairs = [(name, str(dtype)) for name, dtype in schema.items()]
    payload = json.dumps(pairs, sort_keys=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_completed_batches(checkpoint_jsonl: Path) -> set[str]:
    """
    Read JSONL checkpoints and return the set of batch_ids that completed successfully.

    Why:
    - We treat successful batches as durable checkpoints, allowing safe resume after
      a wedge/crash without reprocessing.
    - We ignore malformed lines rather than failing the resume path; the checkpoint
      file is append-only and may be truncated in a crash scenario.
    """
    completed: set[str] = set()
    if not checkpoint_jsonl.exists():
        return completed

    for line in checkpoint_jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("ok") is True and isinstance(rec.get("batch_id"), str):
            completed.add(rec["batch_id"])

    return completed


def _append_jsonl(checkpoint_jsonl: Path, rec: dict[str, Any]) -> None:
    """
    Append one JSON record to the JSONL checkpoint file.

    Why:
    - Append-only writing is resilient: if a run is interrupted, prior records remain valid.
    - JSONL is easy to inspect, grep, parse, and archive for audit trails.
    """
    checkpoint_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def _validate_batch(
    *,
    run_id: str,
    batch_id: str,
    batch_index: int,
    batch_paths: Sequence[str],
    batch_size: int,
    strict: bool,
    sort_output: bool,
    engine: str,
    infer_schema_length: int,
) -> BatchResult:
    """
    Validate a single batch of CSV files.

    Validation strategy:
    - Build one LazyFrame scan over the batch.
    - Apply wide_to_long (transform-only).
    - Compute all validation metrics in one select + one collect to avoid multiple
      passes over potentially large datasets.

    Why one collect:
    - In Polars, separate .collect() calls can trigger separate executions. At scale,
      multiple passes are expensive and can increase peak memory usage and I/O.
    - We constrain validation to the minimum set of invariants that provide strong
      correctness guarantees without needing to materialize the full long table.

    Note on sorting:
    - sort_output is intentionally configurable. Month-scale semantic validation uses
      sort_output=False to avoid global sorts that can wedge Docker Desktop.
    - Deterministic ordering can be validated separately on bounded samples with
      sort_output=True.
    """
    t0 = time.time()
    started_iso = _utc_now_iso()

    # Batching against resource exhaustion.
    wide_lf = pl.scan_csv(batch_paths, infer_schema_length=infer_schema_length)
    out_lf = transform_wide_to_long_lf(lf=wide_lf, strict=strict, sort_output=sort_output)

    # Single-pass metrics collection (streaming engine recommended).
    metrics_df = out_lf.select([
        pl.len().alias("long_rows"),
        (pl.len() % 48).alias("long_rows_mod_48"),
        pl.col("datetime").min().alias("min_datetime"),
        pl.col("datetime").max().alias("max_datetime"),
        pl.col("datetime").is_null().any().alias("any_null_datetime"),
    ]).collect(engine=engine)

    long_rows = int(metrics_df["long_rows"][0])
    long_rows_mod_48 = int(metrics_df["long_rows_mod_48"][0])
    mn = metrics_df["min_datetime"][0]
    mx = metrics_df["max_datetime"][0]
    any_null = bool(metrics_df["any_null_datetime"][0])

    mn_s = mn.isoformat() if mn is not None else None
    mx_s = mx.isoformat() if mx is not None else None

    # Schema fingerprint is cheap (no data scan); it uses the logical schema post-transform.
    schema_fp = _schema_fingerprint(out_lf.collect_schema())

    # Invariants (fail-loud):
    #
    # These checks are intentionally chosen because they have high diagnostic value:
    # - long_rows % 48 == 0 catches interval count issues (missing/extra intervals).
    # - min/max datetime validate the core datetime semantics and DST folding behavior.
    # - null datetime indicates parsing or datetime math failures (must never happen in strict mode).
    if long_rows == 0:
        raise ValueError("Batch produced 0 long rows (unexpected).")

    if long_rows_mod_48 != 0:
        raise ValueError(f"Batch long_rows not divisible by 48: long_rows={long_rows} mod_48={long_rows_mod_48}")

    if any_null:
        raise ValueError("Batch contains null datetime values.")

    if mn is None or mx is None:
        raise ValueError("Batch min/max datetime is null (unexpected).")

    if (mn.hour, mn.minute) != (0, 0):
        raise ValueError(f"Batch min datetime not at 00:00: {mn!r}")

    if (mx.hour, mx.minute) != (23, 30):
        raise ValueError(f"Batch max datetime not at 23:30: {mx!r}")

    finished_iso = _utc_now_iso()
    elapsed = time.time() - t0

    return BatchResult(
        run_id=run_id,
        batch_id=batch_id,
        batch_index=batch_index,
        batch_size=batch_size,
        n_files=len(batch_paths),
        first_path=batch_paths[0],
        last_path=batch_paths[-1],
        started_at_utc=started_iso,
        finished_at_utc=finished_iso,
        elapsed_sec=elapsed,
        strict=strict,
        sort_output=sort_output,
        engine=engine,
        infer_schema_length=infer_schema_length,
        long_rows=long_rows,
        long_rows_mod_48=long_rows_mod_48,
        min_datetime=mn_s,
        max_datetime=mx_s,
        any_null_datetime=any_null,
        schema_fingerprint=schema_fp,
        ok=True,
        error_type=None,
        error_message=None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entrypoint.

    This tool is intentionally a validator rather than a writer:
    - it proves correctness at scale without entangling file output concerns
    - it is safe to run repeatedly (idempotent resume semantics)
    - it produces a stable, append-only audit log (JSONL checkpoints)
    """
    ap = argparse.ArgumentParser(description="Batched wide_to_long validator with JSONL checkpoints (resumable).")

    ap.add_argument("--input-list", required=True, help="Text file of CSV paths (one per line).")
    ap.add_argument("--batch-size", type=int, default=25, help="Files per batch (e.g., 25, 10).")
    ap.add_argument(
        "--out-dir",
        default="/workspaces/smart-meter-analysis/data/validation",
        help="Directory for checkpoints/logs.",
    )
    ap.add_argument("--run-id", default=None, help="Run identifier (default: timestamp-based).")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip batches already marked ok in checkpoints.jsonl.",
    )

    ap.add_argument("--strict", action="store_true", help="Enable strict wide_to_long validations.")
    ap.add_argument("--no-strict", dest="strict", action="store_false", help="Disable strict mode.")
    ap.set_defaults(strict=True)

    ap.add_argument(
        "--sort-output",
        action="store_true",
        help="Enable global sort inside wide_to_long (use for determinism checks on bounded samples).",
    )
    ap.add_argument(
        "--no-sort-output",
        dest="sort_output",
        action="store_false",
        help="Disable global sort inside wide_to_long (recommended for month-scale semantic validation).",
    )
    ap.set_defaults(sort_output=False)

    ap.add_argument(
        "--engine",
        default="streaming",
        choices=["streaming", "in_memory"],
        help="Polars collect engine.",
    )
    ap.add_argument(
        "--infer-schema-length",
        type=int,
        default=0,
        help="Polars scan_csv infer_schema_length (0 = full scan of types).",
    )

    ap.add_argument("--max-batches", type=int, default=None, help="Process at most this many batches.")
    ap.add_argument("--start-batch", type=int, default=0, help="Start at this batch index (0-based).")
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log failure and continue to next batch.",
    )

    args = ap.parse_args(list(argv) if argv is not None else None)

    paths = _read_paths(args.input_list)
    batches = _chunk_paths(paths, args.batch_size)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("wide_to_long_validate_%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) / run_id
    checkpoint_jsonl = out_dir / "checkpoints.jsonl"

    completed = _load_completed_batches(checkpoint_jsonl) if args.resume else set()

    total_files = len(paths)
    print(f"run_id={run_id}")
    print(f"input_list={args.input_list}")
    print(f"total_files={total_files}")
    print(f"batch_size={args.batch_size}")
    print(f"n_batches={len(batches)}")
    print(f"out_dir={out_dir}")
    print(f"checkpoint_jsonl={checkpoint_jsonl}")
    print(
        f"strict={args.strict} sort_output={args.sort_output} engine={args.engine} infer_schema_length={args.infer_schema_length}"
    )
    print(f"resume={args.resume} completed_batches={len(completed)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    batches_ok = 0
    batches_failed = 0
    files_ok = 0
    files_failed = 0

    for i in range(args.start_batch, len(batches)):
        if args.max_batches is not None and (batches_ok + batches_failed) >= args.max_batches:
            print(f"Reached --max-batches={args.max_batches}; stopping.")
            break

        batch_paths = batches[i]
        batch_id = f"batch_{i:05d}"

        # Resume behavior: skip batches already confirmed OK by prior runs.
        if args.resume and batch_id in completed:
            print(f"[SKIP] {batch_id} already ok (resume).")
            continue

        print(f"[RUN ] {batch_id} n_files={len(batch_paths)} first={batch_paths[0]}")

        try:
            res = _validate_batch(
                run_id=run_id,
                batch_id=batch_id,
                batch_index=i,
                batch_paths=batch_paths,
                batch_size=args.batch_size,
                strict=args.strict,
                sort_output=args.sort_output,
                engine=args.engine,
                infer_schema_length=args.infer_schema_length,
            )
            _append_jsonl(checkpoint_jsonl, asdict(res))
            batches_ok += 1
            files_ok += res.n_files

            print(
                f"[OK  ] {batch_id} files={res.n_files} files_ok={files_ok}/{total_files} "
                f"long_rows={res.long_rows} min={res.min_datetime} max={res.max_datetime} "
                f"elapsed_sec={res.elapsed_sec:.2f}"
            )
        except Exception as e:
            # On failure, we still checkpoint the error. This makes failures reproducible
            # and supports later triage without rerunning the full month.
            batches_failed += 1
            files_failed += len(batch_paths)

            rec: dict[str, Any] = {
                "run_id": run_id,
                "batch_id": batch_id,
                "batch_index": i,
                "batch_size": args.batch_size,
                "n_files": len(batch_paths),
                "first_path": batch_paths[0] if batch_paths else "",
                "last_path": batch_paths[-1] if batch_paths else "",
                "started_at_utc": None,
                "finished_at_utc": _utc_now_iso(),
                "elapsed_sec": None,
                "strict": args.strict,
                "sort_output": args.sort_output,
                "engine": args.engine,
                "infer_schema_length": args.infer_schema_length,
                "ok": False,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
            _append_jsonl(checkpoint_jsonl, rec)

            print(
                f"[FAIL] {batch_id} files={len(batch_paths)} files_failed={files_failed}/{total_files} "
                f"error_type={type(e).__name__} error={e}"
            )

            if not args.continue_on_error:
                print("Stopping on first failure (use --continue-on-error to proceed).")
                print(
                    "Summary: "
                    f"batches_ok={batches_ok} batches_failed={batches_failed} "
                    f"files_ok={files_ok} files_failed={files_failed} total_files={total_files} "
                    f"checkpoint_jsonl={checkpoint_jsonl}"
                )
                return 2

    print(
        "Done. "
        f"batches_ok={batches_ok} batches_failed={batches_failed} "
        f"files_ok={files_ok} files_failed={files_failed} total_files={total_files} "
        f"checkpoint_jsonl={checkpoint_jsonl}"
    )
    return 0 if batches_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
