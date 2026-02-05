#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

from smart_meter_analysis.wide_to_long import transform_wide_to_long, transform_wide_to_long_lf

JsonDict = dict[str, Any]
Status = Literal["start", "success", "failure", "skip", "warning", "info"]

# Deterministic canonical sort order (boss-approved, updated):
# zip_code, account_identifier, datetime
SORT_KEYS: tuple[str, str, str] = ("zip_code", "account_identifier", "datetime")

REQUIRED_WIDE_COLS: tuple[str, ...] = (
    "ZIP_CODE",
    "DELIVERY_SERVICE_CLASS",
    "DELIVERY_SERVICE_NAME",
    "ACCOUNT_IDENTIFIER",
    "INTERVAL_READING_DATE",
    "INTERVAL_LENGTH",
    "TOTAL_REGISTERED_ENERGY",
    "PLC_VALUE",
    "NSPL_VALUE",
)

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

DEFAULT_WORKERS = 4
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_ERRORS = 1000
DEFAULT_PRINT_FAILURES = 10
DEFAULT_EXEC_MODE = "lazy_sink"  # default streaming sink
DEFAULT_SKIP_EXISTING_BATCH_OUTPUTS = True


# -----------------------------
# Data models
# -----------------------------


@dataclass(frozen=True)
class RunnerConfig:
    year_month: str  # YYYYMM
    input_list: Path
    out_root: Path  # dataset root
    run_id: str

    workers: int
    batch_size: int
    resume: bool
    dry_run: bool
    fail_fast: bool
    max_errors: int
    max_files: int | None

    shard_id: int | None

    # Phase A/B ergonomics
    skip_existing_batch_outputs: bool
    overwrite: bool

    run_dir: Path
    log_jsonl: Path
    manifest_dir: Path
    staging_dir: Path

    print_failures: int

    exec_mode: Literal["eager", "lazy_sink"]
    debug_mem: bool
    debug_temp_scan: bool
    polars_temp_dir: str | None


@dataclass(frozen=True)
class BatchPlan:
    batch_id: str
    inputs: list[str]


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
    return {"sha": sha, "is_dirty": bool(dirty) if dirty is not None else None}


def build_env_info() -> JsonDict:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "polars": pl.__version__,
        "cwd": str(Path.cwd()),
    }


# -----------------------------
# Debug helpers (RSS / disk / temp)
# -----------------------------


def _read_rss_bytes() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        kb = int(parts[1])
                        return kb * 1024
    except Exception:
        return None
    return None


def _disk_usage_bytes(path: Path) -> JsonDict:
    try:
        du = shutil.disk_usage(str(path))
        return {"free": int(du.free), "total": int(du.total), "used": int(du.used)}
    except Exception as e:
        return {"error": type(e).__name__, "msg": str(e)}


def _snapshot_dir(path: Path, limit: int = 2000) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        if not path.exists() or not path.is_dir():
            return out
        for i, p in enumerate(path.iterdir()):
            if i >= limit:
                break
            try:
                if p.is_file():
                    out[p.name] = int(p.stat().st_size)
            except Exception:  # noqa: S112
                continue
    except Exception:
        return out
    return out


# -----------------------------
# Planning / inputs
# -----------------------------


def normalize_input_path(p: str) -> str:
    p = p.strip()
    if not p:
        return p
    if p.startswith("s3://"):
        return p
    return str(Path(p).expanduser().resolve())


def load_inputs(input_list: Path) -> list[str]:
    if not input_list.exists():
        raise SystemExit(f"--input-list not found: {input_list}")
    raw = input_list.read_text(encoding="utf-8").splitlines()
    inputs = [normalize_input_path(x) for x in raw if x.strip() and not x.strip().startswith("#")]
    inputs_sorted = sorted(inputs)
    if not inputs_sorted:
        raise SystemExit("No inputs found in --input-list after filtering comments/blank lines.")
    return inputs_sorted


def make_batches(inputs_sorted: list[str], batch_size: int) -> list[BatchPlan]:
    if batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    out: list[BatchPlan] = []
    n = len(inputs_sorted)
    for i in range(0, n, batch_size):
        j = i // batch_size
        out.append(BatchPlan(batch_id=f"batch_{j:04d}", inputs=inputs_sorted[i : i + batch_size]))
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def to_jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return to_jsonable(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


# -----------------------------
# Resume / checkpointing
# -----------------------------


def iter_manifest_success_inputs(manifest_dir: Path) -> set[str]:
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
# Schema / validation helpers
# -----------------------------


def build_wide_schema() -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "ZIP_CODE": pl.Utf8,
        "DELIVERY_SERVICE_CLASS": pl.Utf8,
        "DELIVERY_SERVICE_NAME": pl.Utf8,
        "ACCOUNT_IDENTIFIER": pl.Utf8,
        "INTERVAL_READING_DATE": pl.Utf8,
        "INTERVAL_LENGTH": pl.Int32,
        "TOTAL_REGISTERED_ENERGY": pl.Float64,
        "PLC_VALUE": pl.Float64,
        "NSPL_VALUE": pl.Float64,
    }

    # Standard 0030..2400 (48 cols) + DST extras 2430/2500 (2 cols)
    for minutes in [*list(range(30, 1441, 30)), 1470, 1500]:
        hh, mm = divmod(minutes, 60)
        schema[f"INTERVAL_HR{hh:02d}{mm:02d}_ENERGY_QTY"] = pl.Float64

    return schema


def validate_wide_contract(df: pl.DataFrame) -> None:
    missing = [c for c in REQUIRED_WIDE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required wide columns: {missing}")

    if df.schema.get("INTERVAL_LENGTH") not in {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    }:
        raise ValueError(f"INTERVAL_LENGTH dtype must be integer seconds. observed={df.schema.get('INTERVAL_LENGTH')}")

    bad = df.filter(pl.col("INTERVAL_LENGTH").is_null() | (pl.col("INTERVAL_LENGTH") != 1800)).height
    if bad > 0:
        sample = (
            df.filter(pl.col("INTERVAL_LENGTH").is_null() | (pl.col("INTERVAL_LENGTH") != 1800))
            .select(["ZIP_CODE", "ACCOUNT_IDENTIFIER", "INTERVAL_READING_DATE", "INTERVAL_LENGTH"])
            .head(10)
            .to_dicts()
        )
        raise ValueError(
            f"INTERVAL_LENGTH contract violation: expected 1800 everywhere. bad_rows={bad} sample={sample}"
        )


def validate_wide_contract_lf(lf: pl.LazyFrame) -> None:
    cols = lf.collect_schema().names()
    missing = [c for c in REQUIRED_WIDE_COLS if c not in cols]
    if missing:
        raise ValueError(f"Missing required wide columns: {missing}")

    bad = (
        lf.filter(pl.col("INTERVAL_LENGTH").is_null() | (pl.col("INTERVAL_LENGTH") != 1800))
        .select(pl.len().alias("bad_rows"))
        .collect(engine="streaming")
        .item()
    )
    if int(bad) > 0:
        sample = (
            lf.filter(pl.col("INTERVAL_LENGTH").is_null() | (pl.col("INTERVAL_LENGTH") != 1800))
            .select(["ZIP_CODE", "ACCOUNT_IDENTIFIER", "INTERVAL_READING_DATE", "INTERVAL_LENGTH"])
            .head(10)
            .collect(engine="streaming")
            .to_dicts()
        )
        raise ValueError(
            f"INTERVAL_LENGTH contract violation: expected 1800 everywhere. bad_rows={int(bad)} sample={sample}"
        )


def shape_long_after_transform(df: pl.DataFrame) -> pl.DataFrame:
    out = df
    if "energy_kwh" not in out.columns and "interval_energy" in out.columns:
        out = out.rename({"interval_energy": "energy_kwh"})

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
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Transform output missing required columns: {missing} present_cols={out.columns}")

    if out.schema.get("datetime") == pl.Utf8:
        raise ValueError("datetime is Utf8. transform must output Datetime.")

    out = out.with_columns([
        pl.col("zip_code").cast(pl.Utf8),
        pl.col("account_identifier").cast(pl.Utf8),
        pl.col("delivery_service_class").cast(pl.Categorical),
        pl.col("delivery_service_name").cast(pl.Categorical),
        pl.col("energy_kwh").cast(pl.Float64, strict=False),
        pl.col("plc_value").cast(pl.Float64, strict=False),
        pl.col("nspl_value").cast(pl.Float64, strict=False),
        pl.col("datetime").cast(pl.Datetime("us")),
    ]).with_columns([
        pl.col("datetime").dt.year().cast(pl.Int32).alias("year"),
        pl.col("datetime").dt.month().cast(pl.Int8).alias("month"),
    ])

    return out.select(list(FINAL_LONG_COLS))


def shape_long_after_transform_lf(lf: pl.LazyFrame) -> pl.LazyFrame:
    cols = lf.collect_schema().names()
    if "energy_kwh" not in cols and "interval_energy" in cols:
        lf = lf.rename({"interval_energy": "energy_kwh"})

    cols = lf.collect_schema().names()
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
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Transform output missing required columns: {missing} present_cols={cols}")

    lf = lf.with_columns([
        pl.col("zip_code").cast(pl.Utf8),
        pl.col("account_identifier").cast(pl.Utf8),
        pl.col("delivery_service_class").cast(pl.Categorical),
        pl.col("delivery_service_name").cast(pl.Categorical),
        pl.col("energy_kwh").cast(pl.Float64, strict=False),
        pl.col("plc_value").cast(pl.Float64, strict=False),
        pl.col("nspl_value").cast(pl.Float64, strict=False),
        pl.col("datetime").cast(pl.Datetime("us")),
    ]).with_columns([
        pl.col("datetime").dt.year().cast(pl.Int32).alias("year"),
        pl.col("datetime").dt.month().cast(pl.Int8).alias("month"),
    ])

    return lf.select(list(FINAL_LONG_COLS))


def validate_year_month(df: pl.DataFrame, year_month: str) -> None:
    y = int(year_month[:4])
    m = int(year_month[4:6])
    bad = df.filter((pl.col("year") != y) | (pl.col("month") != m)).height
    if bad > 0:
        raise ValueError(f"--year-month {year_month} validation failed: bad_rows={bad}")


def validate_year_month_lf(lf: pl.LazyFrame, year_month: str) -> None:
    y = int(year_month[:4])
    m = int(year_month[4:6])
    bad = (
        lf.filter((pl.col("year") != y) | (pl.col("month") != m))
        .select(pl.len().alias("bad_rows"))
        .collect(engine="streaming")
        .item()
    )
    if int(bad) > 0:
        raise ValueError(f"--year-month {year_month} validation failed: bad_rows={int(bad)}")


# -----------------------------
# Paths / deterministic output naming
# -----------------------------


def year_month_dirs(year_month: str) -> tuple[str, str]:
    y = int(year_month[:4])
    m = int(year_month[4:6])
    return f"year={y:04d}", f"month={m:02d}"


def batch_output_filename(batch_id: str, shard_id: int | None) -> str:
    if shard_id is None:
        return f"{batch_id}.parquet"
    return f"shard_{shard_id:02d}_{batch_id}.parquet"


def canonical_batch_out_path(cfg: RunnerConfig, batch_id: str) -> Path:
    ydir, mdir = year_month_dirs(cfg.year_month)
    return cfg.out_root / ydir / mdir / batch_output_filename(batch_id, cfg.shard_id)


def staging_batch_out_path(cfg: RunnerConfig, batch_id: str) -> Path:
    ydir, mdir = year_month_dirs(cfg.year_month)
    return cfg.staging_dir / batch_id / ydir / mdir / batch_output_filename(batch_id, cfg.shard_id)


def atomic_publish(staging_path: Path, final_path: Path, overwrite: bool) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        os.replace(str(staging_path), str(final_path))
        return
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {final_path}")
    os.replace(str(staging_path), str(final_path))


# -----------------------------
# Batch execution
# -----------------------------


def batch_manifest_paths(cfg: RunnerConfig, batch_id: str) -> tuple[Path, Path]:
    manifest = cfg.manifest_dir / f"manifest_{batch_id}.jsonl"
    summary = cfg.manifest_dir / f"summary_{batch_id}.json"
    return manifest, summary


def _raise_batch_multi_year_month(uniq: pl.DataFrame) -> None:
    """Raise ValueError with batch (year,month) values for diagnostics."""
    raise ValueError(f"Batch contains multiple (year,month) values: {uniq.sort(['year', 'month']).to_dicts()}")


def run_batch(
    *,
    cfg: RunnerConfig,
    batch: BatchPlan,
    logger: JsonlLogger,
    skip_set: set[str],
    stop_flag: threading.Event,
) -> JsonDict:
    t_batch0 = time.time()
    manifest_path, summary_path = batch_manifest_paths(cfg, batch.batch_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    batch_ctx: JsonDict = {
        "ts_utc": now_utc_iso(),
        "year_month": cfg.year_month,
        "run_id": cfg.run_id,
        "batch_id": batch.batch_id,
        "shard_id": cfg.shard_id,
    }

    final_out = canonical_batch_out_path(cfg, batch.batch_id)

    # Batch-level checkpoint (default ON):
    if cfg.skip_existing_batch_outputs and final_out.exists() and not cfg.overwrite:
        summary: JsonDict = {
            **batch_ctx,
            "status": "skip",
            "skip_reason": "existing_batch_output",
            "n_inputs": len(batch.inputs),
            "n_success": 0,
            "n_failure": 0,
            "n_skip": 0,
            "elapsed_ms": elapsed_ms(t_batch0, time.time()),
            "manifest_jsonl": str(manifest_path),
            "final_out_path": str(final_out),
            "wrote_file": False,
            "exec_mode": cfg.exec_mode,
            "sort_keys": list(SORT_KEYS),
        }
        write_json(summary_path, summary)
        logger.log({
            **batch_ctx,
            "event": "batch_skip_existing_output",
            "status": "skip",
            "final_out_path": str(final_out),
        })
        return summary

    logger.log({
        **batch_ctx,
        "event": "batch_start",
        "status": "start",
        "n_inputs": len(batch.inputs),
        "final_out_path": str(final_out),
    })

    tmp_dir = Path(tempfile.gettempdir())
    polars_tmp = os.environ.get("POLARS_TEMP_DIR")
    wide_schema = build_wide_schema()

    frames: list[pl.DataFrame] = []
    lfs: list[pl.LazyFrame] = []

    n_success = 0
    n_failure = 0
    n_skip = 0
    errors: list[JsonDict] = []

    with manifest_path.open("a", encoding="utf-8") as mf:
        for input_path in batch.inputs:
            if stop_flag.is_set():
                break

            if input_path in skip_set:
                n_skip += 1
                mf.write(
                    json.dumps(
                        {**batch_ctx, "input_path": input_path, "status": "skip", "reason": "resume_success"},
                        sort_keys=True,
                    )
                    + "\n"
                )
                logger.log({
                    **batch_ctx,
                    "event": "file_skip",
                    "status": "skip",
                    "input_path": input_path,
                    "reason": "resume_success",
                })
                continue

            t0 = time.time()
            file_ctx: JsonDict = {**batch_ctx, "input_path": input_path}
            logger.log({**file_ctx, "event": "file_start", "status": "start"})

            try:
                if cfg.exec_mode == "eager":
                    df_wide = pl.read_csv(
                        input_path,
                        schema=wide_schema,
                        has_header=True,
                        infer_schema_length=0,
                        ignore_errors=False,
                        try_parse_dates=False,
                    )
                    validate_wide_contract(df_wide)

                    df_long = transform_wide_to_long(df_wide, strict=True, sort_output=False)
                    df_long = shape_long_after_transform(df_long)
                    validate_year_month(df_long, cfg.year_month)

                    frames.append(df_long)
                    rows_wide = int(df_wide.height)
                    rows_long = int(df_long.height)
                else:
                    lf_wide = pl.scan_csv(
                        input_path,
                        schema=wide_schema,
                        has_header=True,
                        ignore_errors=False,
                        try_parse_dates=False,
                    )
                    validate_wide_contract_lf(lf_wide)

                    lf_long = transform_wide_to_long_lf(lf_wide, strict=True, sort_output=False)
                    lf_long = shape_long_after_transform_lf(lf_long)
                    validate_year_month_lf(lf_long, cfg.year_month)

                    rows_wide = int(lf_wide.select(pl.len()).collect(engine="streaming").item())
                    rows_long = int(lf_long.select(pl.len()).collect(engine="streaming").item())
                    lfs.append(lf_long)

                n_success += 1
                t1 = time.time()
                mf.write(
                    json.dumps(
                        {
                            **file_ctx,
                            "status": "success",
                            "elapsed_ms": elapsed_ms(t0, t1),
                            "rows_wide": rows_wide,
                            "rows_long": rows_long,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                logger.log({
                    **file_ctx,
                    "event": "file_success",
                    "status": "success",
                    "elapsed_ms": elapsed_ms(t0, t1),
                    "rows_wide": rows_wide,
                    "rows_long": rows_long,
                })

            except Exception as e:
                n_failure += 1
                t1 = time.time()
                mf.write(
                    json.dumps(
                        {
                            **file_ctx,
                            "status": "failure",
                            "elapsed_ms": elapsed_ms(t0, t1),
                            "exception_type": type(e).__name__,
                            "exception_msg": str(e),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                logger.log({
                    **file_ctx,
                    "event": "file_failure",
                    "status": "failure",
                    "elapsed_ms": elapsed_ms(t0, t1),
                    "exception_type": type(e).__name__,
                    "exception_msg": str(e),
                    "traceback": traceback.format_exc(),
                })
                errors.append({"input_path": input_path, "exception_type": type(e).__name__, "exception_msg": str(e)})

                if cfg.fail_fast or n_failure >= cfg.max_errors:
                    break

    wrote_file = False
    write_bytes = 0
    staging_out = staging_batch_out_path(cfg, batch.batch_id)
    staging_out.parent.mkdir(parents=True, exist_ok=True)

    if cfg.debug_mem:
        logger.log({
            **batch_ctx,
            "event": "debug_env",
            "status": "info",
            "exec_mode": cfg.exec_mode,
            "tmp_dir": str(tmp_dir),
            "polars_temp_dir_env": polars_tmp,
            "rss_bytes": _read_rss_bytes(),
            "disk_tmp": _disk_usage_bytes(tmp_dir),
            "disk_out_root": _disk_usage_bytes(cfg.out_root),
            "final_out_path": str(final_out),
            "staging_out_path": str(staging_out),
        })

    try:
        if cfg.exec_mode == "eager" and frames:
            df_batch = pl.concat(frames, how="vertical", rechunk=False)
            df_batch = df_batch.sort(list(SORT_KEYS), maintain_order=True)

            uniq = df_batch.select(["year", "month"]).unique()
            if uniq.height != 1:
                _raise_batch_multi_year_month(uniq)

            df_batch.write_parquet(str(staging_out), compression="snappy", statistics=True, use_pyarrow=False)
            wrote_file = True

        if cfg.exec_mode == "lazy_sink" and lfs:
            lf_batch = pl.concat(lfs, how="vertical")
            uniq = lf_batch.select(["year", "month"]).unique().collect(engine="streaming")
            if uniq.height != 1:
                _raise_batch_multi_year_month(uniq)

            # Collect before sort+write: sink_parquet uses the streaming engine
            # which does not honor .sort() — it processes data in unordered chunks,
            # silently producing unsorted output. Materializing first guarantees
            # write_parquet emits rows in sorted order.
            df_batch = lf_batch.collect(engine="streaming")
            df_batch = df_batch.sort(list(SORT_KEYS), maintain_order=True)
            df_batch.write_parquet(str(staging_out), compression="snappy", statistics=True, use_pyarrow=False)
            wrote_file = True

        if wrote_file:
            write_bytes = staging_out.stat().st_size
            atomic_publish(staging_out, final_out, overwrite=cfg.overwrite)

            try:
                staging_batch_root = cfg.staging_dir / batch.batch_id
                if staging_batch_root.exists():
                    shutil.rmtree(staging_batch_root, ignore_errors=True)
            except Exception:  # noqa: S110
                pass

    except FileExistsError as e:
        logger.log({
            **batch_ctx,
            "event": "batch_publish_collision",
            "status": "warning",
            "exception_msg": str(e),
            "final_out_path": str(final_out),
            "staging_out_path": str(staging_out),
        })
        n_failure += 1
        wrote_file = False
    except Exception as e:
        logger.log({
            **batch_ctx,
            "event": "batch_write_failure",
            "status": "failure",
            "exception_type": type(e).__name__,
            "exception_msg": str(e),
            "traceback": traceback.format_exc(),
        })
        n_failure += 1
        wrote_file = False

    t_batch1 = time.time()
    batch_summary: JsonDict = {
        "ts_utc": now_utc_iso(),
        "year_month": cfg.year_month,
        "run_id": cfg.run_id,
        "batch_id": batch.batch_id,
        "shard_id": cfg.shard_id,
        "n_inputs": len(batch.inputs),
        "n_success": n_success,
        "n_failure": n_failure,
        "n_skip": n_skip,
        "elapsed_ms": elapsed_ms(t_batch0, t_batch1),
        "errors_sample": errors[:10],
        "manifest_jsonl": str(manifest_path),
        "final_out_path": str(final_out),
        "staging_out_path": str(staging_out),
        "wrote_file": wrote_file,
        "write_bytes": write_bytes,
        "sort_keys": list(SORT_KEYS),
        "exec_mode": cfg.exec_mode,
        "tmp_dir": str(tmp_dir),
        "polars_temp_dir_env": polars_tmp,
    }
    write_json(summary_path, batch_summary)
    logger.log({**batch_ctx, "event": "batch_end", "status": "info", **batch_summary})
    return batch_summary


# -----------------------------
# CLI / main
# -----------------------------


def parse_args(argv: Sequence[str]) -> RunnerConfig:
    ap = argparse.ArgumentParser(
        prog="migrate_month_runner",
        description="Deterministic, resumable CSV→Parquet month runner (single-file per batch; shard-safe filenames).",
    )
    ap.add_argument("--input-list", required=True, type=Path, help="Newline-delimited input paths (local or s3://).")
    ap.add_argument("--out-root", required=True, type=Path, help="Output dataset root (Hive partitions).")
    ap.add_argument("--year-month", required=True, help="Target month in YYYYMM, e.g. 202307")
    ap.add_argument("--run-id", default=None, help="Optional run id. Default: UTC timestamp + stable hash.")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-errors", type=int, default=DEFAULT_MAX_ERRORS)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--shard-id", type=int, default=None, help="Shard identifier (used in output filenames).")
    ap.add_argument(
        "--skip-existing-batch-outputs",
        action="store_true",
        default=DEFAULT_SKIP_EXISTING_BATCH_OUTPUTS,
        help="Skip a batch if its expected output file already exists (default: on).",
    )
    ap.add_argument(
        "--no-skip-existing-batch-outputs",
        action="store_false",
        dest="skip_existing_batch_outputs",
        help="Disable skip-existing behavior (not recommended).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing batch output files (dangerous; opt-in).",
    )
    ap.add_argument(
        "--exec-mode",
        choices=["eager", "lazy_sink"],
        default=DEFAULT_EXEC_MODE,
        help="Execution mode. Default is lazy_sink (sort+sink_parquet in streaming).",
    )
    ap.add_argument("--debug-mem", action="store_true", help="Log RSS/disk/timing per batch stage.")
    ap.add_argument("--debug-temp-scan", action="store_true", help="Snapshot temp dir before/after sink.")
    ap.add_argument(
        "--polars-temp-dir",
        default=None,
        help="If set, exports POLARS_TEMP_DIR for this process (helps prove spill location).",
    )
    ap.add_argument("--print-failures", type=int, default=DEFAULT_PRINT_FAILURES)

    ns = ap.parse_args(list(argv))

    ym = ns.year_month.strip()
    if len(ym) != 6 or (not ym.isdigit()):
        raise SystemExit("--year-month must be YYYYMM (6 digits)")

    out_root = ns.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if ns.polars_temp_dir is not None:
        os.environ["POLARS_TEMP_DIR"] = str(Path(ns.polars_temp_dir).expanduser().resolve())

    if ns.run_id is None:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rid = f"{ts}_{stable_hash(ym + '|' + str(out_root))}"
    else:
        rid = ns.run_id.strip()

    run_dir = out_root / "_runs" / ym / rid
    log_jsonl = run_dir / "logs" / "run_log.jsonl"
    manifest_dir = run_dir / "manifests"
    staging_dir = run_dir / "staging"

    return RunnerConfig(
        year_month=ym,
        input_list=ns.input_list.expanduser().resolve(),
        out_root=out_root,
        run_id=rid,
        workers=ns.workers,
        batch_size=ns.batch_size,
        resume=ns.resume,
        dry_run=ns.dry_run,
        fail_fast=ns.fail_fast,
        max_errors=ns.max_errors,
        max_files=ns.max_files,
        shard_id=ns.shard_id,
        skip_existing_batch_outputs=ns.skip_existing_batch_outputs,
        overwrite=ns.overwrite,
        run_dir=run_dir,
        log_jsonl=log_jsonl,
        manifest_dir=manifest_dir,
        staging_dir=staging_dir,
        print_failures=ns.print_failures,
        exec_mode=ns.exec_mode,
        debug_mem=ns.debug_mem,
        debug_temp_scan=ns.debug_temp_scan,
        polars_temp_dir=ns.polars_temp_dir,
    )


def sample_failures_from_log(log_path: Path, n: int) -> list[dict[str, Any]]:
    if n <= 0 or (not log_path.exists()):
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
        if (
            rec.get("event") in ("file_failure", "batch_write_failure", "batch_publish_collision")
            or rec.get("status") == "failure"
        ):
            out.append({
                k: rec.get(k)
                for k in ["batch_id", "shard_id", "input_path", "exception_type", "exception_msg", "final_out_path"]
            })
            if len(out) >= n:
                break
    return out


def main(argv: Sequence[str]) -> int:
    cfg = parse_args(argv)

    stop_flag = threading.Event()

    def _handle_signal(_signum: int, _frame: Any) -> None:
        stop_flag.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    cfg.staging_dir.mkdir(parents=True, exist_ok=True)

    logger = JsonlLogger(cfg.log_jsonl)

    inputs_sorted = load_inputs(cfg.input_list)
    if cfg.max_files is not None:
        inputs_sorted = inputs_sorted[: cfg.max_files]

    batches = make_batches(inputs_sorted, cfg.batch_size)

    plan = {
        "ts_utc": now_utc_iso(),
        "year_month": cfg.year_month,
        "run_id": cfg.run_id,
        "inputs_sorted": inputs_sorted,
        "batches": [{"batch_id": b.batch_id, "n_inputs": len(b.inputs)} for b in batches],
        "config": to_jsonable(cfg) | {"sort_keys": list(SORT_KEYS)},
        "env": build_env_info(),
        "git": try_git_info(),
        "notes": {
            "deterministic_sort_keys": "zip_code, account_identifier, datetime",
            "single_file_per_batch_month": True,
            "lazy_sink_note": "lazy_sink uses LazyFrame.sink_parquet (streaming mode).",
            "skip_existing_batch_outputs_default": DEFAULT_SKIP_EXISTING_BATCH_OUTPUTS,
        },
    }
    plan_path = cfg.run_dir / "plan.json"
    write_json(plan_path, plan)

    logger.log({
        "ts_utc": now_utc_iso(),
        "event": "run_start",
        "status": "start",
        "year_month": cfg.year_month,
        "run_id": cfg.run_id,
        "shard_id": cfg.shard_id,
        "n_inputs": len(inputs_sorted),
        "n_batches": len(batches),
        "workers": cfg.workers,
        "batch_size": cfg.batch_size,
        "resume": cfg.resume,
        "dry_run": cfg.dry_run,
        "out_root": str(cfg.out_root),
        "plan_path": str(plan_path),
        "log_jsonl": str(cfg.log_jsonl),
        "manifest_dir": str(cfg.manifest_dir),
        "staging_dir": str(cfg.staging_dir),
        "sort_keys": list(SORT_KEYS),
        "exec_mode": cfg.exec_mode,
        "polars_temp_dir_env": os.environ.get("POLARS_TEMP_DIR"),
        "skip_existing_batch_outputs": cfg.skip_existing_batch_outputs,
        "overwrite": cfg.overwrite,
    })

    if cfg.dry_run:
        print(
            json.dumps(
                {
                    "year_month": cfg.year_month,
                    "run_id": cfg.run_id,
                    "shard_id": cfg.shard_id,
                    "n_inputs": len(inputs_sorted),
                    "n_batches": len(batches),
                    "first_inputs": inputs_sorted[:5],
                    "first_batches": [{"batch_id": b.batch_id, "n_inputs": len(b.inputs)} for b in batches[:3]],
                    "plan_path": str(plan_path),
                    "log_jsonl": str(cfg.log_jsonl),
                    "manifest_dir": str(cfg.manifest_dir),
                    "out_root": str(cfg.out_root),
                    "sort_keys": list(SORT_KEYS),
                    "exec_mode": cfg.exec_mode,
                    "skip_existing_batch_outputs": cfg.skip_existing_batch_outputs,
                    "overwrite": cfg.overwrite,
                },
                indent=2,
                sort_keys=True,
            )
        )
        logger.log({"ts_utc": now_utc_iso(), "event": "run_end", "status": "info", "msg": "dry_run complete"})
        return 0

    skip_set: set[str] = set()
    if cfg.resume:
        skip_set = iter_manifest_success_inputs(cfg.manifest_dir)
        logger.log({
            "ts_utc": now_utc_iso(),
            "event": "resume_loaded",
            "status": "info",
            "run_id": cfg.run_id,
            "year_month": cfg.year_month,
            "shard_id": cfg.shard_id,
            "n_success_already": len(skip_set),
        })

    t0 = time.time()
    summaries: list[JsonDict] = []

    with cf.ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futs: dict[cf.Future[JsonDict], BatchPlan] = {}
        for b in batches:
            futs[ex.submit(run_batch, cfg=cfg, batch=b, logger=logger, skip_set=skip_set, stop_flag=stop_flag)] = b

        for fut in cf.as_completed(futs):
            b = futs[fut]
            try:
                summary = fut.result()
                summaries.append(summary)

                if cfg.fail_fast and int(summary.get("n_failure", 0)) > 0:
                    stop_flag.set()

            except Exception as e:
                logger.log({
                    "ts_utc": now_utc_iso(),
                    "event": "batch_future_failure",
                    "status": "failure",
                    "year_month": cfg.year_month,
                    "run_id": cfg.run_id,
                    "shard_id": cfg.shard_id,
                    "batch_id": b.batch_id,
                    "exception_type": type(e).__name__,
                    "exception_msg": str(e),
                    "traceback": traceback.format_exc(),
                })
                if cfg.fail_fast:
                    stop_flag.set()

    try:
        if cfg.staging_dir.exists() and not any(cfg.staging_dir.iterdir()):
            cfg.staging_dir.rmdir()
    except Exception:  # noqa: S110
        pass

    t1 = time.time()
    total_success = sum(int(x.get("n_success", 0)) for x in summaries)
    total_failure = sum(int(x.get("n_failure", 0)) for x in summaries)
    total_skip = sum(int(x.get("n_skip", 0)) for x in summaries)

    batches_written = sum(1 for x in summaries if x.get("wrote_file") is True)
    batches_skipped_existing_output = sum(1 for x in summaries if x.get("skip_reason") == "existing_batch_output")
    batches_with_failures = sum(1 for x in summaries if int(x.get("n_failure", 0)) > 0)

    run_summary = {
        "ts_utc": now_utc_iso(),
        "year_month": cfg.year_month,
        "run_id": cfg.run_id,
        "shard_id": cfg.shard_id,
        "out_root": str(cfg.out_root),
        "n_inputs": len(inputs_sorted),
        "n_batches_planned": len(batches),
        "n_batches_completed": len(summaries),
        "batches_written": batches_written,
        "batches_skipped_existing_output": batches_skipped_existing_output,
        "batches_with_failures": batches_with_failures,
        "total_success": total_success,
        "total_failure": total_failure,
        "total_skip": total_skip,
        "elapsed_ms": elapsed_ms(t0, t1),
        "plan_path": str(plan_path),
        "log_jsonl": str(cfg.log_jsonl),
        "manifest_dir": str(cfg.manifest_dir),
        "stop_requested": stop_flag.is_set(),
        "sort_keys": list(SORT_KEYS),
        "exec_mode": cfg.exec_mode,
        "polars_temp_dir_env": os.environ.get("POLARS_TEMP_DIR"),
        "skip_existing_batch_outputs": cfg.skip_existing_batch_outputs,
        "overwrite": cfg.overwrite,
    }
    write_json(cfg.run_dir / "run_summary.json", run_summary)
    logger.log({"ts_utc": now_utc_iso(), "event": "run_end", "status": "info", **run_summary})

    print(json.dumps(run_summary, indent=2, sort_keys=True))
    fails = sample_failures_from_log(cfg.log_jsonl, cfg.print_failures)
    if fails:
        print("Sample failures:")
        for r in fails:
            print(json.dumps(r, ensure_ascii=False))

    return 1 if total_failure > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
