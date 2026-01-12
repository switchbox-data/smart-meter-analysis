#!/usr/bin/env python3
"""
Step 1, Part 1 of the ComEd Smart Meter Clustering Pipeline: Prepare household-level
data for clustering analysis.

Transforms interval-level energy data into daily load profiles at the
HOUSEHOLD (account) level for k-means clustering.

MODIFIED: Accepts multiple input parquet files (e.g., split by time period).
ROBUST: Also accepts *_accounts.parquet and *_dates.parquet as inputs (as produced
by the pipeline), and will not attempt to treat them as interval data.

What this does:
    1. Validate schema (schema-only checks; no full scan)
    2. Build/load manifests for accounts and dates (memory-safe)
    3. Sample households + dates from manifests
    4. Create daily 48-point load profiles per household
    5. Output profiles ready for clustering

Design notes:
    - Uses MANIFESTS to avoid OOM on unique().collect() for large files
    - Chunked streaming mode writes per-chunk parquet files and then performs a
      bounded-memory merge via PyArrow iter_batches + ParquetWriter.
    - In streaming mode we DO NOT read the full merged sampled_profiles.parquet
      into memory; we derive the household map and stats via lazy scans.

Output files:
    - sampled_profiles.parquet:
        One row per (account_identifier, date) with 'profile' = list[float] length 48
    - household_zip4_map.parquet:
        Unique account_identifier → zip_code mapping for later joins
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from calendar import monthrange
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import polars as pl
import pyarrow.parquet as pq

from smart_meter_analysis.manifests import ensure_account_manifest, ensure_date_manifest

try:
    import psutil  # optional
except Exception:
    psutil = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_INTERVAL_COLS = ["zip_code", "account_identifier", "datetime", "kwh", "date", "hour", "is_weekend", "weekday"]
REQUIRED_ACCOUNT_MANIFEST_COLS = ["account_identifier", "zip_code"]
REQUIRED_DATE_MANIFEST_COLS = ["date", "is_weekend", "weekday"]

CGROUP_ROOT = Path("/sys/fs/cgroup")

# No CLI flags per requirement: keep bounded-memory merge settings internal.
MERGE_BATCH_SIZE_ROWS = 65_536
FINAL_PARQUET_COMPRESSION = "snappy"


# =============================================================================
# MEMORY TELEMETRY (unprivileged container-safe)
# =============================================================================
_BASELINE_CGROUP_EVENTS: dict[str, int] = {}


def _read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _read_int_file(path: Path) -> int | None:
    s = _read_text_file(path)
    if s is None:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _get_rss_bytes() -> int | None:
    try:
        if psutil is not None:
            return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:  # noqa: S110
        pass

    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except Exception:
        return None
    return None


def _read_cgroup_memory_bytes() -> dict[str, int | None]:
    cur = _read_int_file(CGROUP_ROOT / "memory.current")
    peak = _read_int_file(CGROUP_ROOT / "memory.peak")
    maxv = _read_text_file(CGROUP_ROOT / "memory.max")
    limit = None
    if maxv is not None and maxv != "max":
        try:
            limit = int(maxv)
        except Exception:
            limit = None
    return {"current": cur, "peak": peak, "limit": limit}


def _read_cgroup_memory_events() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        txt = (CGROUP_ROOT / "memory.events").read_text(encoding="utf-8")
        for line in txt.splitlines():
            parts = line.split()
            if len(parts) == 2:
                out[parts[0]] = int(parts[1])
    except Exception:  # noqa: S110
        pass
    return out


def _mb(x: int | None) -> float | None:
    if x is None:
        return None
    return round(x / (1024.0 * 1024.0), 3)


def log_memory(label: str, extra: dict[str, Any] | None = None) -> None:
    global _BASELINE_CGROUP_EVENTS
    if not _BASELINE_CGROUP_EVENTS:
        _BASELINE_CGROUP_EVENTS = _read_cgroup_memory_events()

    rss_b = _get_rss_bytes()
    cg = _read_cgroup_memory_bytes()
    ev = _read_cgroup_memory_events()
    keys = set(_BASELINE_CGROUP_EVENTS) | set(ev)
    delta = {k: ev.get(k, 0) - _BASELINE_CGROUP_EVENTS.get(k, 0) for k in keys}

    payload: dict[str, Any] = {
        "ts": round(time.time(), 3),
        "event": "mem",
        "stage": label,
        "rss_mb": _mb(rss_b),
        "cgroup_current_mb": _mb(cg.get("current")),
        "cgroup_peak_mb": _mb(cg.get("peak")),
        "cgroup_limit_mb": _mb(cg.get("limit")),
        "cgroup_oom_kill_delta": delta.get("oom_kill", 0),
        "cgroup_events_delta": delta,
    }
    if extra:
        payload.update(extra)

    logger.info("[MEMORY] %s", json.dumps(payload, separators=(",", ":"), sort_keys=True))


# =============================================================================
# INPUT CLASSIFICATION
# =============================================================================
def _schema_names_for_path(p: Path) -> list[str]:
    try:
        return pl.scan_parquet(p).collect_schema().names()
    except Exception as exc:
        raise RuntimeError(f"Failed to read parquet schema for {p}: {exc}") from exc


def classify_inputs(paths: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Classify inputs into:
      - interval_paths: full interval dataset parts
      - account_manifest_paths: *_accounts.parquet (or schema matches)
      - date_manifest_paths: *_dates.parquet (or schema matches)

    This makes the script robust to the orchestrator passing mixed inputs.
    """
    interval_paths: list[Path] = []
    account_manifest_paths: list[Path] = []
    date_manifest_paths: list[Path] = []

    for p in paths:
        name = p.name.lower()

        # Fast filename heuristics first
        if name.endswith("_accounts.parquet"):
            account_manifest_paths.append(p)
            continue
        if name.endswith("_dates.parquet"):
            date_manifest_paths.append(p)
            continue

        cols = set(_schema_names_for_path(p))

        if set(REQUIRED_INTERVAL_COLS).issubset(cols):
            interval_paths.append(p)
            continue

        # Schema-based fallback for manifests
        if set(REQUIRED_ACCOUNT_MANIFEST_COLS).issubset(cols) and "datetime" not in cols and "kwh" not in cols:
            account_manifest_paths.append(p)
            continue
        if set(REQUIRED_DATE_MANIFEST_COLS).issubset(cols) and "datetime" not in cols and "kwh" not in cols:
            date_manifest_paths.append(p)
            continue

        raise ValueError(
            "Unrecognized parquet input (neither interval data nor expected manifest schema): "
            f"{p} with columns={sorted(cols)}"
        )

    if not interval_paths:
        raise ValueError("No interval parquet inputs detected. Expected at least one file with interval schema.")

    logger.info(
        "Classified inputs: interval=%d, accounts_manifest=%d, dates_manifest=%d",
        len(interval_paths),
        len(account_manifest_paths),
        len(date_manifest_paths),
    )
    return interval_paths, account_manifest_paths, date_manifest_paths


def scan_multiple_parquet(paths: list[Path]) -> pl.LazyFrame:
    """Create a unified LazyFrame from multiple interval parquet files."""
    if len(paths) == 1:
        return pl.scan_parquet(paths[0])
    logger.info("Combining %d interval input files...", len(paths))
    return pl.concat([pl.scan_parquet(p) for p in paths])


# =============================================================================
# VALIDATION
# =============================================================================
def validate_interval_schema(interval_paths: list[Path]) -> dict[str, Any]:
    """
    Validate required columns exist and all interval parts share the same schema.
    Schema-only checks; does not scan full data.
    """
    errors: list[str] = []

    first_schema = pl.scan_parquet(interval_paths[0]).collect_schema()
    first_names = first_schema.names()
    missing = [c for c in REQUIRED_INTERVAL_COLS if c not in first_names]
    if missing:
        errors.append(f"Missing interval columns in first file {interval_paths[0]}: {missing}")

    # Ensure all interval parts match schema (fail-loud with useful diagnostics)
    for p in interval_paths[1:]:
        s = pl.scan_parquet(p).collect_schema()
        if s.names() != first_names:
            errors.append(
                "Interval schema mismatch:\n"
                f"  first={interval_paths[0]} cols={first_names}\n"
                f"  this ={p} cols={s.names()}"
            )

    return {"valid": len(errors) == 0, "errors": errors, "columns": first_names}


# =============================================================================
# SAMPLING HELPERS
# =============================================================================
def _as_polars_date_list(dates: list[Any]) -> list[Any]:
    if not dates:
        return dates
    return pl.Series("dates", dates).cast(pl.Date).to_list()


def _sum_parquet_rows(paths: Sequence[Path]) -> int:
    total = 0
    for p in paths:
        pf = pq.ParquetFile(str(p))
        md = pf.metadata
        if md is None:
            raise RuntimeError(f"Missing parquet metadata for input: {p}")
        total += int(md.num_rows)
    return total


def _load_union_account_manifest(paths: list[Path]) -> pl.DataFrame:
    return (
        pl.concat([pl.scan_parquet(p) for p in paths])
        .unique(subset=["account_identifier", "zip_code"])
        .collect(engine="streaming")
    )


def _load_union_date_manifest(paths: list[Path]) -> pl.DataFrame:
    return (
        pl.concat([pl.scan_parquet(p) for p in paths])
        .unique(subset=["date", "is_weekend", "weekday"])
        .collect(engine="streaming")
    )


def get_metadata_and_samples(  # noqa: C901
    interval_paths: list[Path],
    account_manifest_paths: list[Path],
    date_manifest_paths: list[Path],
    sample_households: int | None,
    sample_days: int,
    day_strategy: Literal["stratified", "random"],
    seed: int = 42,
    year: int | None = None,
    month: int | None = None,
) -> dict[str, Any]:
    """
    Get summary statistics and sample households + dates using manifests.

    If manifests are provided via CLI inputs (as the orchestrator does), use them directly.
    Otherwise, fall back to ensure_*_manifest(interval_file) for each interval file.
    """
    logger.info("Sampling from manifests...")
    log_memory("start_of_get_metadata_and_samples")

    if account_manifest_paths:
        accounts_df = _load_union_account_manifest(account_manifest_paths)
    else:
        manifests = [ensure_account_manifest(p) for p in interval_paths]
        accounts_df = _load_union_account_manifest([Path(m) for m in manifests])

    if date_manifest_paths:
        dates_df = _load_union_date_manifest(date_manifest_paths)
    else:
        manifests = [ensure_date_manifest(p) for p in interval_paths]
        dates_df = _load_union_date_manifest([Path(m) for m in manifests])

    if year is not None and month is not None:
        _, last_day = monthrange(year, month)
        start_date = pl.date(year, month, 1)
        end_date = pl.date(year, month, last_day)
        dates_df = dates_df.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
        logger.info("  Dates available after %d-%02d filter: %d", year, month, dates_df.height)
    else:
        logger.info("  No month filter applied (using all available dates): %d", dates_df.height)

    if accounts_df.height == 0:
        raise ValueError("No account_identifier values found in account manifest(s).")
    if dates_df.height == 0:
        raise ValueError("No dates found in date manifest(s).")

    summary = {
        "n_rows": _sum_parquet_rows(interval_paths),
        "n_accounts": int(accounts_df.height),
        "n_zip_codes": int(accounts_df.select(pl.col("zip_code").n_unique()).item()),
        "min_date": dates_df.select(pl.col("date").min()).item(),
        "max_date": dates_df.select(pl.col("date").max()).item(),
    }

    logger.info("  %s rows (from interval parquet metadata)", f"{summary['n_rows']:,}")
    logger.info("  %s households (from manifests)", f"{summary['n_accounts']:,}")
    logger.info("  %s ZIP+4 codes (from manifests)", f"{summary['n_zip_codes']:,}")
    logger.info("  Date range: %s to %s (from manifests)", summary["min_date"], summary["max_date"])

    log_memory("after_loading_manifests", {"accounts_rows": accounts_df.height, "dates_rows": dates_df.height})

    # Sample households
    if sample_households is not None and sample_households < accounts_df.height:
        accounts_df = accounts_df.sample(n=sample_households, shuffle=True, seed=seed)
        logger.info("  Sampled %s households", f"{accounts_df.height:,}")
    else:
        logger.info("  Using all %s households", f"{accounts_df.height:,}")

    accounts = accounts_df["account_identifier"].to_list()

    # Sample dates
    if day_strategy == "stratified":
        weekday_df = dates_df.filter(~pl.col("is_weekend"))
        weekend_df = dates_df.filter(pl.col("is_weekend"))
        if weekday_df.height == 0 or weekend_df.height == 0:
            logger.warning("  Missing weekdays or weekends; falling back to random day sampling.")
            day_strategy = "random"

    if day_strategy == "stratified":
        weekday_df = dates_df.filter(~pl.col("is_weekend"))
        weekend_df = dates_df.filter(pl.col("is_weekend"))

        n_weekdays = int(sample_days * 0.7)
        n_weekends = sample_days - n_weekdays

        n_weekdays = min(n_weekdays, weekday_df.height)
        n_weekends = min(n_weekends, weekend_df.height)

        sampled_weekdays = (
            weekday_df.sample(n=n_weekdays, shuffle=True, seed=seed)["date"].to_list() if n_weekdays > 0 else []
        )
        sampled_weekends = (
            weekend_df.sample(n=n_weekends, shuffle=True, seed=seed + 1)["date"].to_list() if n_weekends > 0 else []
        )

        dates = sorted(sampled_weekdays + sampled_weekends)
        logger.info(
            "  Sampled %d weekdays + %d weekend days (stratified)", len(sampled_weekdays), len(sampled_weekends)
        )
    else:
        n_sample = min(sample_days, dates_df.height)
        dates = dates_df.sample(n=n_sample, shuffle=True, seed=seed)["date"].to_list()
        logger.info("  Sampled %d days (random)", len(dates))

    if not dates:
        raise ValueError("No dates were sampled; check input data and sampling settings.")

    dates = _as_polars_date_list(dates)

    del accounts_df, dates_df
    gc.collect()

    log_memory("end_of_get_metadata_and_samples", {"sampled_accounts": len(accounts), "sampled_dates": len(dates)})
    return {"summary": summary, "accounts": accounts, "dates": dates}


# =============================================================================
# PROFILE CREATION (STREAMING)
# =============================================================================
def _create_profiles_for_chunk_streaming(
    interval_paths: list[Path],
    accounts_chunk: list[str],
    dates: list[Any],
    chunk_idx: int,
    total_chunks: int,
    tmp_dir: Path,
) -> Path:
    logger.info("  Chunk %d/%d: %s households...", chunk_idx + 1, total_chunks, f"{len(accounts_chunk):,}")
    log_memory(f"chunk_{chunk_idx + 1}_start", {"chunk_accounts": len(accounts_chunk), "n_chunks": total_chunks})

    dates = _as_polars_date_list(dates)
    chunk_path = tmp_dir / f"sampled_profiles_chunk_{chunk_idx:03d}.parquet"

    (
        scan_multiple_parquet(interval_paths)
        .filter(pl.col("account_identifier").is_in(accounts_chunk) & pl.col("date").is_in(dates))
        .group_by(["account_identifier", "zip_code", "date"])
        .agg([
            pl.struct(["datetime", "kwh"]).sort_by("datetime").struct.field("kwh").alias("profile"),
            pl.col("is_weekend").first(),
            pl.col("weekday").first(),
            pl.len().alias("num_intervals"),
        ])
        .filter(pl.col("num_intervals") == 48)
        .drop("num_intervals")
        .sink_parquet(chunk_path)
    )

    log_memory(f"chunk_{chunk_idx + 1}_done", {"chunk_path": str(chunk_path)})
    logger.info("    Wrote chunk parquet: %s", chunk_path)
    return chunk_path


def _merge_parquet_chunks_bounded_memory(chunk_paths: Sequence[Path], output_path: Path) -> int:
    if not chunk_paths:
        raise ValueError("No chunk files provided for merge.")

    chunk_files = sorted((Path(p) for p in chunk_paths), key=lambda p: p.name)
    output_path = Path(output_path)
    tmp_out = output_path.with_suffix(output_path.suffix + ".tmp")

    if tmp_out.exists():
        tmp_out.unlink()

    log_memory("merge_start", {"n_chunks": len(chunk_files), "batch_rows": MERGE_BATCH_SIZE_ROWS})

    expected_rows = 0
    for p in chunk_files:
        pf = pq.ParquetFile(str(p))
        md = pf.metadata
        if md is None:
            raise RuntimeError(f"Missing parquet metadata for chunk: {p}")
        expected_rows += int(md.num_rows)

    first_pf = pq.ParquetFile(str(chunk_files[0]))
    schema = first_pf.schema_arrow

    writer = pq.ParquetWriter(
        where=str(tmp_out),
        schema=schema,
        compression=FINAL_PARQUET_COMPRESSION,
        use_dictionary=True,
        write_statistics=True,
    )

    rows_written = 0
    t0 = time.time()
    try:
        for i, p in enumerate(chunk_files, start=1):
            pf = pq.ParquetFile(str(p))
            if not pf.schema_arrow.equals(schema, check_metadata=False):
                raise RuntimeError(
                    "Schema mismatch across chunk files; refusing to merge.\n"
                    f"First schema: {schema}\n"
                    f"Mismatch file: {p}\n"
                    f"File schema: {pf.schema_arrow}"
                )

            for rb in pf.iter_batches(batch_size=MERGE_BATCH_SIZE_ROWS):
                writer.write_batch(rb)
                rows_written += int(rb.num_rows)

            del pf
            gc.collect()

            if i == 1 or i == len(chunk_files) or (i % 10 == 0):
                log_memory("merge_progress", {"chunk_i": i, "n_chunks": len(chunk_files), "rows_written": rows_written})
    finally:
        writer.close()

    if rows_written != expected_rows:
        raise RuntimeError(f"Merged row count mismatch: wrote {rows_written} rows but expected {expected_rows} rows.")

    os.replace(str(tmp_out), str(output_path))
    log_memory("merge_done", {"rows_written": rows_written, "seconds": round(time.time() - t0, 3)})
    return rows_written


def create_household_profiles_chunked_streaming(
    interval_paths: list[Path],
    accounts: list[str],
    dates: list[Any],
    output_path: Path,
    chunk_size: int = 5000,
) -> int:
    if not accounts:
        raise ValueError("No accounts provided for chunked streaming profile creation.")
    if not dates:
        raise ValueError("No dates provided for chunked streaming profile creation.")

    dates = _as_polars_date_list(dates)
    n_accounts = len(accounts)
    n_chunks = (n_accounts + chunk_size - 1) // chunk_size

    logger.info(
        "Creating profiles in %d chunks of up to %s households each (total: %s households x %d days)...",
        n_chunks,
        f"{chunk_size:,}",
        f"{n_accounts:,}",
        len(dates),
    )
    log_memory("before_chunked_streaming", {"n_accounts": n_accounts, "n_chunks": n_chunks, "chunk_size": chunk_size})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_path.parent

    chunk_paths: list[Path] = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_accounts)
        accounts_chunk = accounts[start_idx:end_idx]

        chunk_path = _create_profiles_for_chunk_streaming(
            interval_paths=interval_paths,
            accounts_chunk=accounts_chunk,
            dates=dates,
            chunk_idx=i,
            total_chunks=n_chunks,
            tmp_dir=tmp_dir,
        )

        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            chunk_paths.append(chunk_path)

        gc.collect()

    if not chunk_paths:
        logger.warning("No profiles created in chunked streaming mode!")
        return 0

    logger.info("Combining %d chunk files into %s (bounded-memory merge)", len(chunk_paths), output_path)
    gc.collect()
    n_profiles = _merge_parquet_chunks_bounded_memory(chunk_paths=chunk_paths, output_path=output_path)

    logger.info("  Created %s complete profiles (chunked streaming)", f"{n_profiles:,}")
    logger.info("  Saved to %s", output_path)

    for p in chunk_paths:
        try:
            p.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to delete temp chunk file %s: %s", p, exc)

    log_memory("after_chunk_cleanup", {"deleted_chunks": len(chunk_paths)})
    return int(n_profiles)


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
def prepare_clustering_data(
    input_paths: list[Path],
    output_dir: Path,
    sample_households: int | None = None,
    sample_days: int = 20,
    day_strategy: Literal["stratified", "random"] = "stratified",
    streaming: bool = False,
    chunk_size: int = 5000,
    seed: int = 42,
    year: int | None = None,
    month: int | None = None,
) -> dict[str, Any]:
    logger.info("=" * 70)
    logger.info("PREPARING HOUSEHOLD-LEVEL CLUSTERING DATA")
    if len(input_paths) > 1:
        logger.info("(MULTI-FILE MODE: %d input files)", len(input_paths))
    if streaming:
        logger.info("(STREAMING MODE ENABLED, chunk_size=%d)", chunk_size)
    logger.info("=" * 70)
    log_memory("start_of_prepare_clustering_data", {"streaming": streaming, "n_inputs": len(input_paths)})

    interval_paths, account_manifest_paths, date_manifest_paths = classify_inputs(input_paths)

    validation = validate_interval_schema(interval_paths)
    if not validation["valid"]:
        raise ValueError(f"Input validation failed: {validation['errors']}")

    metadata = get_metadata_and_samples(
        interval_paths=interval_paths,
        account_manifest_paths=account_manifest_paths,
        date_manifest_paths=date_manifest_paths,
        sample_households=sample_households,
        sample_days=sample_days,
        day_strategy=day_strategy,
        seed=seed,
        year=year,
        month=month,
    )

    accounts = metadata["accounts"]
    dates = metadata["dates"]

    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = output_dir / "sampled_profiles.parquet"
    map_path = output_dir / "household_zip4_map.parquet"

    if streaming:
        n_profiles = create_household_profiles_chunked_streaming(
            interval_paths=interval_paths,
            accounts=accounts,
            dates=dates,
            output_path=profiles_path,
            chunk_size=chunk_size,
        )
        if n_profiles == 0:
            raise ValueError("No profiles created in chunked streaming mode - check input data and sampling settings.")

        # Memory-safe: do not read entire profiles parquet
        log_memory("build_household_map_start")
        (pl.scan_parquet(profiles_path).select(["account_identifier", "zip_code"]).unique().sink_parquet(map_path))
        log_memory("build_household_map_done", {"map_path": str(map_path)})

        stats_row = (
            pl.scan_parquet(profiles_path)
            .select([
                pl.len().alias("n_profiles"),
                pl.col("account_identifier").n_unique().alias("n_households"),
                pl.col("zip_code").n_unique().alias("n_zip4s"),
                pl.col("date").n_unique().alias("n_dates"),
            ])
            .collect(engine="streaming")
            .to_dicts()[0]
        )
        stats = {k: int(stats_row[k]) for k in ["n_profiles", "n_households", "n_zip4s", "n_dates"]}
    else:
        # Non-streaming mode unchanged from original implementation.
        profiles_df = (
            scan_multiple_parquet(interval_paths)
            .filter(pl.col("account_identifier").is_in(accounts) & pl.col("date").is_in(_as_polars_date_list(dates)))
            .sort(["account_identifier", "datetime"])
            .collect()
        )
        if profiles_df.is_empty():
            raise ValueError("No profiles created - check input data and sampling settings.")

        profiles_out = (
            profiles_df.group_by(["account_identifier", "zip_code", "date"])
            .agg([
                pl.col("kwh").alias("profile"),
                pl.col("is_weekend").first(),
                pl.col("weekday").first(),
                pl.len().alias("num_intervals"),
            ])
            .filter(pl.col("num_intervals") == 48)
            .drop("num_intervals")
        )

        profiles_out.select([
            "account_identifier",
            "zip_code",
            "date",
            "profile",
            "is_weekend",
            "weekday",
        ]).write_parquet(profiles_path)
        logger.info("  Saved profiles: %s", profiles_path)

        household_map = profiles_out.select(["account_identifier", "zip_code"]).unique()
        household_map.write_parquet(map_path)
        logger.info("  Saved household-ZIP+4 map: %s (%s households)", map_path, f"{household_map.height:,}")

        stats = {
            "n_profiles": int(profiles_out.height),
            "n_households": int(profiles_out["account_identifier"].n_unique()),
            "n_zip4s": int(profiles_out["zip_code"].n_unique()),
            "n_dates": int(profiles_out["date"].n_unique()),
        }

    logger.info("")
    logger.info("=" * 70)
    logger.info("CLUSTERING DATA READY")
    logger.info("=" * 70)
    logger.info("  Profiles: %s", f"{stats['n_profiles']:,}")
    logger.info("  Households: %s", f"{stats['n_households']:,}")
    logger.info("  ZIP+4s represented: %s", f"{stats['n_zip4s']:,}")
    logger.info("  Days: %d", stats["n_dates"])
    logger.info("  Output: %s", output_dir)
    log_memory("end_of_prepare_clustering_data", stats)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare household-level data for clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", type=Path, required=True, nargs="+", help="Path(s) to processed interval parquet file(s)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/clustering"), help="Output directory (default: data/clustering)."
    )
    parser.add_argument(
        "--sample-households", type=int, default=None, help="Number of households to sample (default: all)."
    )
    parser.add_argument("--sample-days", type=int, default=20, help="Number of days to sample (default: 20).")
    parser.add_argument(
        "--day-strategy", choices=["stratified", "random"], default="stratified", help="Day sampling strategy."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--streaming", action="store_true", help="Use chunked streaming mode (10k+ households).")
    parser.add_argument(
        "--chunk-size", type=int, default=5000, help="Households per chunk when --streaming is enabled."
    )
    parser.add_argument("--year", type=int, default=None, help="Year to filter dates (e.g., 2023).")
    parser.add_argument("--month", type=int, default=None, help="Month to filter dates (1-12).")

    args = parser.parse_args()

    input_paths = args.input if isinstance(args.input, list) else [args.input]
    for path in input_paths:
        if not path.exists():
            logger.error("Input file not found: %s", path)
            return 1

    prepare_clustering_data(
        input_paths=input_paths,
        output_dir=args.output_dir,
        sample_households=args.sample_households,
        sample_days=args.sample_days,
        day_strategy=args.day_strategy,
        streaming=args.streaming,
        chunk_size=args.chunk_size,
        seed=args.seed,
        year=args.year,
        month=args.month,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
