#!/usr/bin/env python3
"""
CSV→Parquet migration runner (month-scoped), with determinism, parallel execution, observability, and checkpointing.

Determinism
- Inputs are loaded from --input-list, normalized, and sorted lexicographically before any execution.
- Optional --max-files is applied *after* sort (still deterministic).
- Deterministic batch plan: batch_0000..batch_NNNN based solely on sorted input order and --batch-size.
- A plan.json is written before execution capturing inputs, batch ids, config, environment, and git SHA (if available).

Parallel strategy
- Parallelizes at the batch level (each worker handles one batch sequentially).
- Default executor: ThreadPoolExecutor (not ProcessPoolExecutor). Rationale:
  - Polars is a Rust-native library and typically uses internal threading; process-based concurrency can introduce
    extra overhead, oversubscription, and platform-dependent spawn/fork issues.
  - Batch-level thread parallelism gives us concurrency across I/O + compute without pickling large frames.

Observability + error handling
- Structured JSONL run log (run_log.jsonl) with per-batch and per-file events: start/success/failure/skip/warnings.
- Each failure record includes exception type/message + full traceback.
- Per-batch manifest JSONL (one row per input file) + per-batch summary JSON.
- End-of-run stdout summary (counts + paths + sampled failures) for operator ergonomics; JSONL remains authoritative.

Checkpointing/resume
- --resume loads existing manifest_*.jsonl files from manifest_dir and skips inputs already marked success.
- Reruns are deterministic and do not require restarting the entire pipeline.

Writer abstraction (critical)
- WriterBackend isolates Parquet write behavior so the final LazyFrame sink writer can be plugged in later as a
  small, isolated diff.
- This skeleton provides:
  - NoopWriter: used for s3:// out_root in this skeleton (logs warning; does not write).
  - LocalEagerPartitionWriter: writes to local disk using DataFrame.write_parquet(partition_by=[year, month]).
    NOTE: maintain_order=True on *write* is not supported on DataFrame.write_parquet in polars==1.35.2; we log a
    warning but do apply deterministic in-memory sort.

Fail-loud semantics
- Any per-file read/parse/transform/write failure is captured as a failure record and included in manifests.
- The run proceeds unless --fail-fast or per-batch failure count exceeds --max-errors.

Non-goals in this runner
- S3 listing/enumeration via --input-prefix is stubbed (out-of-scope); prefer --input-list.
- Final LazyFrame sink writer is stubbed; plug in later once writer decision is made.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

# -----------------------------
# Constants / typing utilities
# -----------------------------

JsonDict = dict[str, Any]
Status = Literal["start", "success", "failure", "skip", "warning", "info"]

DEFAULT_WORKERS = 4
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_ERRORS = 1000
DEFAULT_PRINT_FAILURES = 10

RUNS_ROOT_REL = Path("data/runs/csv_to_parquet")


# -----------------------------
# Data models
# -----------------------------


@dataclass(frozen=True)
class RunnerConfig:
    year_month: str  # YYYYMM
    input_list: Path | None
    input_prefix: str | None
    out_root: str  # local path or s3://...
    run_id: str
    workers: int
    batch_size: int
    resume: bool
    dry_run: bool
    log_jsonl: Path
    manifest_dir: Path
    fail_fast: bool
    max_errors: int

    # Improvements
    max_files: int | None
    scope_out_root: bool
    print_failures: int


@dataclass(frozen=True)
class BatchPlan:
    batch_id: str
    inputs: list[str]


@dataclass(frozen=True)
class RunPlan:
    year_month: str
    run_id: str
    created_utc: str
    inputs_sorted: list[str]
    batches: list[BatchPlan]
    config: JsonDict
    env: JsonDict
    git: JsonDict


@dataclass
class FileResult:
    input_path: str
    status: Literal["success", "failure", "skip"]
    elapsed_ms: int
    exception_type: str | None = None
    exception_msg: str | None = None
    traceback: str | None = None
    rows_wide: int | None = None
    rows_long: int | None = None
    output_paths: list[str] | None = None


# -----------------------------
# Logging
# -----------------------------


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: JsonDict) -> None:
        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        with self._lock, self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def elapsed_ms(t0: float, t1: float) -> int:
    return round((t1 - t0) * 1000.0)


def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def json_safe(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable objects (e.g., Path) into JSON-safe values."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def compute_effective_out_root(out_root: str, year_month: str, run_id: str, scope: bool) -> str:
    """
    For local runs, default to scoping the output root by year_month/run_id to avoid collisions between runs:
      <out_root>/<YYYYMM>/<run_id>/
    For s3:// out_root (noop writer in this skeleton) or if scope=False, return out_root unchanged.
    """
    if not scope or out_root.startswith("s3://"):
        return out_root
    root = Path(out_root).expanduser().resolve()
    return str(root / year_month / run_id)


def sample_failures_from_log(log_path: Path, n: int) -> list[dict[str, Any]]:
    if n <= 0 or not log_path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if rec.get("event") == "file_failure" or rec.get("status") == "failure":
            out.append({k: rec.get(k) for k in ["batch_id", "input_path", "exception_type", "exception_msg"]})
            if len(out) >= n:
                break
    return out


# -----------------------------
# Git / environment helpers
# -----------------------------


def try_git_info() -> JsonDict:
    def _run(args: list[str]) -> str | None:
        try:
            cp = subprocess.run(args, check=False, capture_output=True, text=True)  # noqa: S603
            if cp.returncode != 0:
                return None
            return cp.stdout.strip()
        except Exception:
            return None

    sha = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    return {
        "sha": sha,
        "is_dirty": bool(dirty) if dirty is not None else None,
    }


def build_env_info() -> JsonDict:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "polars": pl.__version__,
        "cwd": str(Path.cwd()),
    }


# -----------------------------
# Input planning
# -----------------------------


def normalize_input_path(p: str) -> str:
    p = p.strip()
    if not p:
        return p
    if p.startswith("s3://"):
        return p
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:
        return p


def load_inputs(config: RunnerConfig) -> list[str]:
    if config.input_list is None and config.input_prefix is None:
        raise SystemExit("Must provide --input-list or --input-prefix")

    if config.input_list is not None and config.input_prefix is not None:
        raise SystemExit("Provide only one of --input-list or --input-prefix")

    if config.input_list is not None:
        raw = config.input_list.read_text(encoding="utf-8").splitlines()
        inputs = [normalize_input_path(x) for x in raw if x.strip() and not x.strip().startswith("#")]
        inputs_sorted = sorted(inputs)
        return inputs_sorted

    # input_prefix enumeration is out-of-scope; fail loudly.
    raise SystemExit(
        "Input prefix enumeration is out-of-scope in this runner skeleton. "
        "Provide --input-list with one path per line (s3://... or local)."
    )


def make_batches(inputs_sorted: list[str], batch_size: int) -> list[BatchPlan]:
    batches: list[BatchPlan] = []
    n = len(inputs_sorted)
    for i in range(0, n, batch_size):
        j = i // batch_size
        batch_id = f"batch_{j:04d}"
        batches.append(BatchPlan(batch_id=batch_id, inputs=inputs_sorted[i : i + batch_size]))
    return batches


def write_plan(plan: RunPlan, plan_path: Path) -> None:
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    obj: JsonDict = {
        "year_month": plan.year_month,
        "run_id": plan.run_id,
        "created_utc": plan.created_utc,
        "inputs_sorted": plan.inputs_sorted,
        "batches": [{"batch_id": b.batch_id, "inputs": b.inputs} for b in plan.batches],
        "config": plan.config,
        "env": plan.env,
        "git": plan.git,
    }
    plan_path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


# -----------------------------
# Checkpointing
# -----------------------------


def iter_manifest_success_inputs(manifest_dir: Path) -> set[str]:
    """Parse existing manifest JSONL files to find inputs already marked success."""
    if not manifest_dir.exists():
        return set()

    success: set[str] = set()
    for p in sorted(manifest_dir.glob("manifest_*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") == "success" and isinstance(rec.get("input_path"), str):
                    success.add(rec["input_path"])
    return success


# -----------------------------
# Writer backends (abstraction)
# -----------------------------


class WriterBackend:
    """Backend contract: write a long DataFrame to out_root with partitioning."""

    name: str

    def write_long_dataset(
        self,
        *,
        long_df: pl.DataFrame,
        out_root: str,
        partition_cols: Sequence[str],
        compression: str,
        maintain_order: bool,
        storage_options: dict[str, Any] | None,
        logger: JsonlLogger,
        context: JsonDict,
    ) -> list[str]:
        raise NotImplementedError


class NoopWriter(WriterBackend):
    name = "noop"

    def write_long_dataset(
        self,
        *,
        long_df: pl.DataFrame,
        out_root: str,
        partition_cols: Sequence[str],
        compression: str,
        maintain_order: bool,
        storage_options: dict[str, Any] | None,
        logger: JsonlLogger,
        context: JsonDict,
    ) -> list[str]:
        logger.log({
            **context,
            "event": "writer_warning",
            "status": "warning",
            "writer": self.name,
            "msg": "NoopWriter selected: no parquet will be written in this skeleton. "
            "Provide a local out_root for dev or plug in the final LazyFrame sink writer for S3.",
        })
        return []


class LocalEagerPartitionWriter(WriterBackend):
    name = "local_eager_partitioned"

    def write_long_dataset(
        self,
        *,
        long_df: pl.DataFrame,
        out_root: str,
        partition_cols: Sequence[str],
        compression: str,
        maintain_order: bool,
        storage_options: dict[str, Any] | None,
        logger: JsonlLogger,
        context: JsonDict,
    ) -> list[str]:
        if out_root.startswith("s3://"):
            raise ValueError("LocalEagerPartitionWriter cannot write to s3:// out_root")

        if maintain_order:
            logger.log({
                **context,
                "event": "writer_warning",
                "status": "warning",
                "writer": self.name,
                "msg": "maintain_order=True is required by checklist, but polars==1.35.2 does not support "
                "maintain_order on DataFrame.write_parquet; writing without maintain_order on write. "
                "Deterministic sort is still applied in-memory. Plug in LazyFrame sink writer to satisfy "
                "maintain_order-on-write fully.",
            })

        out_path = Path(out_root).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        long_df.write_parquet(
            str(out_path),
            compression=compression,
            statistics=True,
            use_pyarrow=False,
            partition_by=list(partition_cols),
        )
        return [str(out_path)]


def choose_writer(out_root: str) -> WriterBackend:
    if out_root.startswith("s3://"):
        return NoopWriter()
    return LocalEagerPartitionWriter()


def writer_capabilities(writer: WriterBackend) -> JsonDict:
    # In this skeleton, only the eager writer exists; it cannot enforce maintain_order at write time.
    if writer.name == "local_eager_partitioned":
        return {
            "maintain_order_write_supported": False,
            "s3_write_supported": True,
        }
    if writer.name == "noop":
        return {
            "maintain_order_write_supported": False,
            "s3_write_supported": False,
        }
    return {
        "maintain_order_write_supported": False,
        "s3_write_supported": False,
    }


# -----------------------------
# Transform integration
# -----------------------------


def resolve_transform_callable() -> Any:
    """
    wide_to_long is authoritative and must not be modified in this task.

    We locate a callable transform in smart_meter_analysis.wide_to_long. Based on your failure log,
    transform_wide_to_long exists and is used, so it is included in candidates.
    """
    import importlib  # imported here only to avoid module import side-effects at file import time

    mod = importlib.import_module("smart_meter_analysis.wide_to_long")

    candidates = [
        "transform_wide_to_long",
        "wide_to_long",
        "convert_wide_to_long",
        "transform_wide_to_long_df",
        "wide_to_long_df",
    ]
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn

    raise RuntimeError(
        "Could not find a callable transform in smart_meter_analysis.wide_to_long. "
        f"Tried: {candidates}. Update resolve_transform_callable() to match the actual function name."
    )


def read_csv_with_explicit_schema_overrides(input_path: str) -> pl.DataFrame:
    """
    Enforce "no inference" while satisfying strict transform contracts via explicit schema_overrides.

    Strategy:
    - Read header columns via n_rows=0 (fast).
    - Default all columns to Utf8 to avoid inference.
    - Override strict contract columns:
        INTERVAL_LENGTH must be integer seconds (1800) without parsing/coercion by transform.

    This avoids relying on Polars inference while meeting wide_to_long contract expectations.
    """
    hdr_df = pl.read_csv(input_path, n_rows=0)
    cols = hdr_df.columns

    schema_overrides: dict[str, pl.DataType] = dict.fromkeys(cols, pl.Utf8)

    # Strict transform contract: INTERVAL_LENGTH must be integer seconds (no parsing/coercion).
    if "INTERVAL_LENGTH" in schema_overrides:
        schema_overrides["INTERVAL_LENGTH"] = pl.Int32

    df = pl.read_csv(
        input_path,
        schema_overrides=schema_overrides,
        infer_schema_length=0,
        ignore_errors=False,
        try_parse_dates=False,
    )
    return df


def enforce_final_schema(long_df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure the final dataset columns exist and match required names/types.

    Required output columns:
      zip_code: Utf8
      delivery_service_class: Categorical
      delivery_service_name: Categorical
      account_identifier: Utf8
      datetime: Datetime
      energy_kwh: Float64
      plc_value: Float64
      nspl_value: Float64
      year: Int32
      month: Int8

    We refuse to guess semantics if required columns are missing or datetime is not Datetime.
    Deterministic sort requirement: (zip_code, account_identifier, datetime)
    """
    df = long_df

    # Minimal name normalization
    rename_map: dict[str, str] = {}
    if "interval_energy" in df.columns and "energy_kwh" not in df.columns:
        rename_map["interval_energy"] = "energy_kwh"
    if "kwh" in df.columns and "energy_kwh" not in df.columns:
        rename_map["kwh"] = "energy_kwh"
    if rename_map:
        df = df.rename(rename_map)

    required = [
        "zip_code",
        "delivery_service_class",
        "delivery_service_name",
        "account_identifier",
        "datetime",
        "energy_kwh",
        "plc_value",
        "nspl_value",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Transform output is missing required columns. "
            f"Missing={missing}. Present={df.columns}. "
            "Per task constraints, this runner will not invent logic; update the transform or add a validated "
            "bridge layer."
        )

    # datetime must be Datetime
    if df.schema["datetime"] == pl.Utf8:
        raise RuntimeError(
            "Column 'datetime' is Utf8. This runner refuses to guess parsing rules. "
            "Ensure the authoritative transform produces a Datetime typed column."
        )

    df = df.with_columns([
        pl.col("zip_code").cast(pl.Utf8),
        pl.col("account_identifier").cast(pl.Utf8),
        pl.col("delivery_service_class").cast(pl.Categorical),
        pl.col("delivery_service_name").cast(pl.Categorical),
        pl.col("energy_kwh").cast(pl.Float64),
        pl.col("plc_value").cast(pl.Float64),
        pl.col("nspl_value").cast(pl.Float64),
    ])

    df = df.with_columns([
        pl.col("datetime").dt.year().cast(pl.Int32).alias("year"),
        pl.col("datetime").dt.month().cast(pl.Int8).alias("month"),
    ])

    # Deterministic sort (boss-approved) — maintain_order applies to the sort operation
    df = df.sort(["zip_code", "account_identifier", "datetime"], maintain_order=True)

    final_cols = [
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
    ]
    return df.select(final_cols)


def process_one_input(input_path: str) -> tuple[pl.DataFrame, int, int]:
    """
    Placeholder processing function:
    - Read wide CSV with explicit schema_overrides (no inference; strict contracts satisfied).
    - Call authoritative transform (DataFrame -> DataFrame).
    - Enforce final schema.

    Returns: (final_long_df, rows_wide, rows_long)
    """
    wide_df = read_csv_with_explicit_schema_overrides(input_path)
    rows_wide = wide_df.height

    transform_fn = resolve_transform_callable()
    try:
        long_df = transform_fn(wide_df)
    except TypeError as err:
        raise RuntimeError(
            "wide_to_long transform callable exists but could not be called with a single DataFrame argument. "
            "Update process_one_input() to match the actual signature exactly (no semantic changes)."
        ) from err

    if not isinstance(long_df, pl.DataFrame):
        raise TypeError(f"Transform returned type={type(long_df)}, expected polars.DataFrame")

    final_df = enforce_final_schema(long_df)
    rows_long = final_df.height
    return final_df, rows_wide, rows_long


# -----------------------------
# Batch worker
# -----------------------------


def batch_paths(config: RunnerConfig, batch_id: str) -> tuple[Path, Path]:
    manifest_jsonl = config.manifest_dir / f"manifest_{batch_id}.jsonl"
    summary_json = config.manifest_dir / f"summary_{batch_id}.json"
    return manifest_jsonl, summary_json


def write_batch_summary(path: Path, summary: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def run_batch(
    *,
    config: RunnerConfig,
    batch: BatchPlan,
    writer: WriterBackend,
    logger: JsonlLogger,
    skip_set: set[str],
) -> JsonDict:
    """
    Runs one batch sequentially. Returns summary dict.
    Writes:
      - per-file manifest JSONL
      - per-batch summary JSON
    """
    t_batch0 = time.time()
    manifest_jsonl, summary_json = batch_paths(config, batch.batch_id)
    manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)

    batch_ctx_base = {
        "ts_utc": now_utc_iso(),
        "year_month": config.year_month,
        "run_id": config.run_id,
        "batch_id": batch.batch_id,
        "writer": writer.name,
    }

    logger.log({**batch_ctx_base, "event": "batch_start", "status": "start", "n_inputs": len(batch.inputs)})

    n_success = 0
    n_failure = 0
    n_skip = 0
    errors: list[JsonDict] = []

    with manifest_jsonl.open("a", encoding="utf-8") as mf:
        for input_path in batch.inputs:
            if input_path in skip_set:
                n_skip += 1
                rec = dataclasses.asdict(
                    FileResult(
                        input_path=input_path,
                        status="skip",
                        elapsed_ms=0,
                        output_paths=[],
                    )
                )
                mf.write(json.dumps({**batch_ctx_base, **rec}, sort_keys=True) + "\n")
                logger.log({**batch_ctx_base, "event": "file_skip", "status": "skip", "input_path": input_path})
                continue

            t0 = time.time()
            file_ctx = {**batch_ctx_base, "input_path": input_path}

            logger.log({**file_ctx, "event": "file_start", "status": "start"})

            try:
                long_df, rows_wide, rows_long = process_one_input(input_path)

                out_paths = writer.write_long_dataset(
                    long_df=long_df,
                    out_root=config.out_root,
                    partition_cols=["year", "month"],
                    compression="zstd",
                    maintain_order=True,
                    storage_options=None,
                    logger=logger,
                    context=file_ctx,
                )

                t1 = time.time()
                rec = dataclasses.asdict(
                    FileResult(
                        input_path=input_path,
                        status="success",
                        elapsed_ms=elapsed_ms(t0, t1),
                        rows_wide=rows_wide,
                        rows_long=rows_long,
                        output_paths=out_paths,
                    )
                )
                mf.write(json.dumps({**batch_ctx_base, **rec}, sort_keys=True) + "\n")
                logger.log({
                    **file_ctx,
                    "event": "file_success",
                    "status": "success",
                    "elapsed_ms": rec["elapsed_ms"],
                    "rows_wide": rows_wide,
                    "rows_long": rows_long,
                    "output_paths": out_paths,
                })
                n_success += 1

            except Exception as e:
                t1 = time.time()
                tb = traceback.format_exc()
                rec = dataclasses.asdict(
                    FileResult(
                        input_path=input_path,
                        status="failure",
                        elapsed_ms=elapsed_ms(t0, t1),
                        exception_type=type(e).__name__,
                        exception_msg=str(e),
                        traceback=tb,
                        output_paths=[],
                    )
                )
                mf.write(json.dumps({**batch_ctx_base, **rec}, sort_keys=True) + "\n")
                logger.log({
                    **file_ctx,
                    "event": "file_failure",
                    "status": "failure",
                    "elapsed_ms": rec["elapsed_ms"],
                    "exception_type": rec["exception_type"],
                    "exception_msg": rec["exception_msg"],
                    "traceback": rec["traceback"],
                })
                n_failure += 1
                errors.append({
                    "input_path": input_path,
                    "exception_type": rec["exception_type"],
                    "exception_msg": rec["exception_msg"],
                })

                if config.fail_fast:
                    break

                if n_failure >= config.max_errors:
                    break

    t_batch1 = time.time()

    summary: JsonDict = {
        "year_month": config.year_month,
        "run_id": config.run_id,
        "batch_id": batch.batch_id,
        "writer": writer.name,
        "n_inputs": len(batch.inputs),
        "n_success": n_success,
        "n_failure": n_failure,
        "n_skip": n_skip,
        "elapsed_ms": elapsed_ms(t_batch0, t_batch1),
        "errors_sample": errors[:10],
        "manifest_jsonl": str(manifest_jsonl),
    }
    write_batch_summary(summary_json, summary)

    logger.log({**batch_ctx_base, "event": "batch_end", "status": "info", **summary})
    return summary


# -----------------------------
# CLI / main
# -----------------------------


def parse_args(argv: Sequence[str]) -> RunnerConfig:
    ap = argparse.ArgumentParser(
        prog="migrate_month_runner",
        description="CSV→Parquet month migration runner (skeleton) with determinism, parallelism, observability, and resume.",
    )
    ap.add_argument("--year-month", required=True, help="Target month in YYYYMM form, e.g. 202307")
    ap.add_argument("--input-list", type=Path, default=None, help="File of input paths (s3:// or local), one per line.")
    ap.add_argument("--input-prefix", default=None, help="s3:// prefix for enumeration (stubbed; out-of-scope).")
    ap.add_argument(
        "--out-root",
        required=True,
        help="Output root for parquet: local path or s3://... (S3 uses NoopWriter in this skeleton).",
    )
    ap.add_argument("--run-id", default=None, help="Optional run id. Default: deterministic based on UTC timestamp.")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument(
        "--resume", action="store_true", help="Resume by skipping inputs already marked success in manifests."
    )
    ap.add_argument("--dry-run", action="store_true", help="Print plan and exit without processing.")
    ap.add_argument("--log-jsonl", type=Path, default=None, help="Path to JSONL log file (default under run dir).")
    ap.add_argument("--manifest-dir", type=Path, default=None, help="Directory for manifests (default under run dir).")
    ap.add_argument("--fail-fast", action="store_true", help="Stop the run early when failures occur (best-effort).")
    ap.add_argument(
        "--max-errors", type=int, default=DEFAULT_MAX_ERRORS, help="Max failures per batch before stopping that batch."
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional dev cap: process only the first N inputs after deterministic sort.",
    )
    ap.add_argument(
        "--no-scope-out-root",
        action="store_true",
        help="Do not scope local out-root by year_month/run_id (default scopes to avoid collisions).",
    )
    ap.add_argument(
        "--print-failures",
        type=int,
        default=DEFAULT_PRINT_FAILURES,
        help="Print up to N failure records to stdout at end-of-run.",
    )

    ns = ap.parse_args(list(argv))

    ym = ns.year_month.strip()
    if len(ym) != 6 or not ym.isdigit():
        raise SystemExit("--year-month must be YYYYMM (6 digits)")

    if ns.run_id is None:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rid = f"{ts}_{stable_hash(ym + '|' + str(ns.out_root))}"
    else:
        rid = ns.run_id.strip()

    run_dir = RUNS_ROOT_REL / ym / rid
    log_jsonl = ns.log_jsonl or (run_dir / "run_log.jsonl")
    manifest_dir = ns.manifest_dir or (run_dir / "manifests")

    return RunnerConfig(
        year_month=ym,
        input_list=ns.input_list,
        input_prefix=ns.input_prefix,
        out_root=str(ns.out_root),
        run_id=rid,
        workers=ns.workers,
        batch_size=ns.batch_size,
        resume=ns.resume,
        dry_run=ns.dry_run,
        log_jsonl=log_jsonl,
        manifest_dir=manifest_dir,
        fail_fast=ns.fail_fast,
        max_errors=ns.max_errors,
        max_files=ns.max_files,
        scope_out_root=not ns.no_scope_out_root,
        print_failures=ns.print_failures,
    )


def main(argv: Sequence[str]) -> int:
    config = parse_args(argv)

    run_dir = config.manifest_dir.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    config.manifest_dir.mkdir(parents=True, exist_ok=True)

    # Scope local outputs by default to avoid collisions across runs.
    effective_out_root = compute_effective_out_root(
        config.out_root, config.year_month, config.run_id, config.scope_out_root
    )
    config = dataclasses.replace(config, out_root=effective_out_root)

    logger = JsonlLogger(config.log_jsonl)
    writer = choose_writer(config.out_root)

    inputs_sorted = load_inputs(config)
    if config.max_files is not None:
        inputs_sorted = inputs_sorted[: config.max_files]

    batches = make_batches(inputs_sorted, config.batch_size)

    plan = RunPlan(
        year_month=config.year_month,
        run_id=config.run_id,
        created_utc=now_utc_iso(),
        inputs_sorted=inputs_sorted,
        batches=batches,
        config=json_safe(dataclasses.asdict(config)),
        env=build_env_info(),
        git=try_git_info(),
    )
    plan_path = run_dir / "plan.json"
    write_plan(plan, plan_path)

    # Capability transparency (useful for boss checklist review)
    caps = writer_capabilities(writer)
    cap_msgs: list[str] = []
    if writer.name == "local_eager_partitioned" and not caps.get("maintain_order_write_supported", False):
        cap_msgs.append(
            "Eager local parquet writer does not enforce maintain_order on write in polars==1.35.2; deterministic in-memory sort is applied."
        )
    if writer.name == "noop":
        cap_msgs.append(
            "S3 out_root selected but writer is NoopWriter in this skeleton; no parquet outputs will be produced."
        )

    logger.log({
        "ts_utc": now_utc_iso(),
        "event": "run_capabilities",
        "status": "warning" if cap_msgs else "info",
        "year_month": config.year_month,
        "run_id": config.run_id,
        "out_root": config.out_root,
        "writer": writer.name,
        **caps,
        "msg": " ".join(cap_msgs) if cap_msgs else "OK",
    })

    logger.log({
        "ts_utc": now_utc_iso(),
        "event": "run_start",
        "status": "start",
        "year_month": config.year_month,
        "run_id": config.run_id,
        "n_inputs": len(inputs_sorted),
        "n_batches": len(batches),
        "workers": config.workers,
        "batch_size": config.batch_size,
        "resume": config.resume,
        "dry_run": config.dry_run,
        "out_root": config.out_root,
        "writer": writer.name,
        "plan_path": str(plan_path),
        "log_jsonl": str(config.log_jsonl),
        "manifest_dir": str(config.manifest_dir),
        "max_files": config.max_files,
    })

    if config.dry_run:
        print(
            json.dumps(
                {
                    "year_month": plan.year_month,
                    "run_id": plan.run_id,
                    "n_inputs": len(plan.inputs_sorted),
                    "n_batches": len(plan.batches),
                    "first_inputs": plan.inputs_sorted[:5],
                    "first_batches": [{"batch_id": b.batch_id, "n_inputs": len(b.inputs)} for b in plan.batches[:3]],
                    "plan_path": str(plan_path),
                    "log_jsonl": str(config.log_jsonl),
                    "manifest_dir": str(config.manifest_dir),
                    "writer": writer.name,
                    "out_root": config.out_root,
                    "capabilities": caps,
                },
                indent=2,
                sort_keys=True,
            )
        )
        logger.log({"ts_utc": now_utc_iso(), "event": "run_end", "status": "info", "msg": "dry_run complete"})
        return 0

    skip_set: set[str] = set()
    if config.resume:
        skip_set = iter_manifest_success_inputs(config.manifest_dir)
        logger.log({
            "ts_utc": now_utc_iso(),
            "event": "resume_loaded",
            "status": "info",
            "n_success_already": len(skip_set),
        })

    t0 = time.time()
    summaries: list[JsonDict] = []

    with cf.ThreadPoolExecutor(max_workers=config.workers) as ex:
        futs: dict[cf.Future[JsonDict], BatchPlan] = {}
        for b in batches:
            futs[ex.submit(run_batch, config=config, batch=b, writer=writer, logger=logger, skip_set=skip_set)] = b

        for fut in cf.as_completed(futs):
            b = futs[fut]
            try:
                summary = fut.result()
                summaries.append(summary)
                if config.fail_fast and summary.get("n_failure", 0) > 0:
                    for f2 in futs:
                        if not f2.done():
                            f2.cancel()
                    break
            except Exception as e:
                logger.log({
                    "ts_utc": now_utc_iso(),
                    "event": "batch_future_failure",
                    "status": "failure",
                    "year_month": config.year_month,
                    "run_id": config.run_id,
                    "batch_id": b.batch_id,
                    "exception_type": type(e).__name__,
                    "exception_msg": str(e),
                    "traceback": traceback.format_exc(),
                })
                if config.fail_fast:
                    for f2 in futs:
                        if not f2.done():
                            f2.cancel()
                    break

    t1 = time.time()

    total_success = sum(int(s.get("n_success", 0)) for s in summaries)
    total_failure = sum(int(s.get("n_failure", 0)) for s in summaries)
    total_skip = sum(int(s.get("n_skip", 0)) for s in summaries)

    run_summary = {
        "year_month": config.year_month,
        "run_id": config.run_id,
        "writer": writer.name,
        "out_root": config.out_root,
        "n_inputs": len(inputs_sorted),
        "n_batches": len(batches),
        "batches_completed": len(summaries),
        "total_success": total_success,
        "total_failure": total_failure,
        "total_skip": total_skip,
        "elapsed_ms": elapsed_ms(t0, t1),
        "plan_path": str(plan_path),
        "log_jsonl": str(config.log_jsonl),
        "manifest_dir": str(config.manifest_dir),
        "capabilities": caps,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.log({"ts_utc": now_utc_iso(), "event": "run_end", "status": "info", **run_summary})

    # Operator-friendly stdout summary (JSONL remains authoritative)
    print(json.dumps(run_summary, indent=2, sort_keys=True))
    fails = sample_failures_from_log(config.log_jsonl, config.print_failures)
    if fails:
        print("Sample failures:")
        for r in fails:
            print(json.dumps(r, ensure_ascii=False))

    return 1 if total_failure > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
