# scripts/csv_to_parquet/validate_month_output.py
from __future__ import annotations

import argparse
import datetime as dt_mod
import json
import random
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

JsonDict = dict[str, Any]

"""
Month-output Validator (QA + determinism + contract enforcement) for ComEd CSV→Parquet migration.

What this validates (fail-loud; raises ValueError with actionable diagnostics):
1) Discovery:
   - Walks --out-root recursively and discovers Hive partitions year=YYYY/month=MM (no filename/count assumptions).
   - Finds all parquet files under discovered partitions; fails if none found.

2) Schema contract (metadata-first where possible):
   - Required columns exist exactly (no silent passing on missing columns).
   - Dtypes match contract:
       zip_code Utf8
       delivery_service_class Categorical
       delivery_service_name Categorical
       account_identifier Utf8
       datetime Datetime
       energy_kwh/plc_value/nspl_value Float64
       year Int32
       month Int8
     Note: year/month accept Int16/Int32 only if explicitly allowed via flags is NOT implemented; contract default is strict.

3) Partition integrity (per-file):
   - year/month columns exist, non-null, and min/max match partition directory year=... month=...
   - Detects mismatches and reports offending files.

4) Datetime invariants (per partition):
   - No null datetime.
   - min(datetime) has (hour,minute)==(0,0)
   - max(datetime) has (hour,minute)==(23,30)
   - All datetime values fall within (partition year, partition month) (no spillover).

5) DST Option B invariants (optional: --dst-month-check):
   - Exactly 48 distinct time slots per day (no 49/50 slot days).
   - Ensures no timestamps beyond 23:30.
   - Spot-checks that (23:00 and 23:30) exist on at least one day with non-null energy_kwh (coarse sanity).

6) Sortedness (non-tautological):
   - Validates lexicographic non-decreasing order by (zip_code, account_identifier, datetime).
   - Modes:
       --check-mode full   : scans entire selected files (streaming=True) and checks is_sorted on a composite key.
       --check-mode sample : checks first/last K rows and deterministic random windows per file (also checks boundaries).

7) Determinism compare (optional: --compare-root):
   - Compares directory trees (relative paths) and per-file sizes between two outputs.
   - Optionally row-counts for a limited number of files (controlled by --max-files in compare pass).

How to run:
  python scripts/csv_to_parquet/validate_month_output.py --out-root /path/to/month_output_root --check-mode sample
  python scripts/csv_to_parquet/validate_month_output.py --out-root ... --check-mode sample --dst-month-check
  python scripts/csv_to_parquet/validate_month_output.py --out-root run1 --compare-root run2 --check-mode sample
"""


RE_YEAR_DIR = re.compile(r"^year=(?P<year>\d{4})$")
RE_MONTH_DIR = re.compile(r"^month=(?P<month>\d{1,2})$")


REQUIRED_SCHEMA: dict[str, pl.DataType] = {
    "zip_code": pl.Utf8,
    "delivery_service_class": pl.Categorical,
    "delivery_service_name": pl.Categorical,
    "account_identifier": pl.Utf8,
    "datetime": pl.Datetime,
    "energy_kwh": pl.Float64,
    "plc_value": pl.Float64,
    "nspl_value": pl.Float64,
    "year": pl.Int32,
    "month": pl.Int8,
}

SORT_KEY_COLS: tuple[str, str, str] = ("zip_code", "account_identifier", "datetime")


@dataclass(frozen=True)
class Partition:
    year: int
    month: int
    path: Path


def _fail(msg: str) -> None:
    raise ValueError(msg)


def _is_parquet(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".parquet"


def _read_parquet_schema(path: Path) -> dict[str, pl.DataType]:
    # Prefer metadata-only schema extraction if available.
    # Polars versions may differ; we keep a robust fallback.
    try:
        schema = pl.read_parquet_schema(str(path))
        return dict(schema)
    except Exception:
        try:
            return dict(pl.scan_parquet(str(path)).schema)
        except Exception as e:
            _fail(f"Failed to read parquet schema for {path}: {e}")
    return {}


def _dtype_eq(observed: pl.DataType, expected: pl.DataType) -> bool:
    # Datetime may carry time_unit/time_zone; contract is "Datetime" at type level.
    if expected == pl.Datetime:
        return isinstance(observed, pl.Datetime) or observed == pl.Datetime
    return observed == expected


def _discover_partitions(out_root: Path) -> list[Partition]:  # noqa: C901
    if not out_root.exists():
        _fail(f"--out-root does not exist: {out_root}")
    if not out_root.is_dir():
        _fail(f"--out-root is not a directory: {out_root}")

    parts: list[Partition] = []
    # Walk directories; find .../year=YYYY/month=MM
    for year_dir in out_root.rglob("*"):
        if not year_dir.is_dir():
            continue
        # Skip _runs/ artifact directories
        if "_runs" in year_dir.parts:
            continue
        m_y = RE_YEAR_DIR.match(year_dir.name)
        if not m_y:
            continue
        year = int(m_y.group("year"))
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir():
                continue
            m_m = RE_MONTH_DIR.match(month_dir.name)
            if not m_m:
                continue
            month = int(m_m.group("month"))
            if not (1 <= month <= 12):
                _fail(f"Invalid month directory detected: {month_dir} (month={month})")
            parts.append(Partition(year=year, month=month, path=month_dir))

    if not parts:
        _fail(f"No Hive partitions found under out-root={out_root}. Expected directories like year=YYYY/month=MM.")
    # Deterministic ordering
    parts.sort(key=lambda p: (p.year, p.month, str(p.path)))
    return parts


def _discover_parquet_files(partitions: Sequence[Partition]) -> dict[Partition, list[Path]]:
    mapping: dict[Partition, list[Path]] = {}
    total = 0
    for part in partitions:
        files = [p for p in part.path.rglob("*.parquet") if _is_parquet(p)]
        files.sort()
        mapping[part] = files
        total += len(files)

    if total == 0:
        _fail(
            "Discovery succeeded but found zero parquet files under discovered partitions. "
            "Check out-root and conversion output."
        )
    return mapping


def _validate_schema_on_file(path: Path) -> None:
    schema = _read_parquet_schema(path)

    missing = [c for c in REQUIRED_SCHEMA if c not in schema]
    if missing:
        _fail(
            f"Schema missing required columns in file {path}:\n  missing={missing}\n  observed_cols={sorted(schema.keys())}"
        )

    # Exactness: allow extra columns, but report them (do not fail).
    # (If you want strict no-extras later, make that a flag.)
    mismatches: list[str] = []
    for col, expected in REQUIRED_SCHEMA.items():
        observed = schema[col]
        if not _dtype_eq(observed, expected):
            mismatches.append(f"{col}: expected={expected}, observed={observed}")

    if mismatches:
        _fail(f"Dtype mismatches in file {path}:\n  " + "\n  ".join(mismatches))


def _validate_partition_integrity_file(path: Path, part: Partition) -> None:
    # Read tiny aggregates only.
    lf = pl.scan_parquet(str(path)).select([
        pl.col("year").null_count().alias("year_nulls"),
        pl.col("month").null_count().alias("month_nulls"),
        pl.col("year").min().alias("year_min"),
        pl.col("year").max().alias("year_max"),
        pl.col("month").min().alias("month_min"),
        pl.col("month").max().alias("month_max"),
    ])
    try:
        row = lf.collect(streaming=True).row(0)
    except Exception as e:
        _fail(f"Failed to collect partition integrity stats for {path}: {e}")

    year_nulls, month_nulls, year_min, year_max, month_min, month_max = row
    if year_nulls != 0 or month_nulls != 0:
        _fail(f"Null partition keys in file {path}: year_nulls={year_nulls}, month_nulls={month_nulls}")
    if year_min != part.year or year_max != part.year or month_min != part.month or month_max != part.month:
        _fail(
            f"Partition key mismatch in file {path} (dir year={part.year}, month={part.month}) "
            f"but columns have year_min={year_min}, year_max={year_max}, month_min={month_min}, month_max={month_max}"
        )


def _validate_datetime_invariants_partition(part: Partition, files: Sequence[Path]) -> None:
    # Partition-level scan across files (still lazy).
    lf = pl.scan_parquet([str(p) for p in files]).select([
        pl.col("datetime").null_count().alias("dt_nulls"),
        pl.col("datetime").min().alias("dt_min"),
        pl.col("datetime").max().alias("dt_max"),
        pl.col("datetime").dt.year().min().alias("dt_year_min"),
        pl.col("datetime").dt.year().max().alias("dt_year_max"),
        pl.col("datetime").dt.month().min().alias("dt_month_min"),
        pl.col("datetime").dt.month().max().alias("dt_month_max"),
    ])
    try:
        row = lf.collect(streaming=True).row(0)
    except Exception as e:
        _fail(f"Failed to collect datetime invariants for partition {part.path}: {e}")

    dt_nulls, dt_min, dt_max, y_min, y_max, m_min, m_max = row
    if dt_nulls != 0:
        _fail(f"Null datetime found in partition {part.path}: dt_nulls={dt_nulls}")

    if dt_min is None or dt_max is None:
        _fail(f"Datetime min/max unexpectedly None in partition {part.path}")

    # Ensure within partition month
    if y_min != part.year or y_max != part.year or m_min != part.month or m_max != part.month:
        _fail(
            f"Datetime spillover in partition {part.path} (dir year={part.year}, month={part.month}) "
            f"but datetime year range=({y_min},{y_max}) month range=({m_min},{m_max})"
        )

    # Time-of-day checks
    if (dt_min.hour, dt_min.minute) != (0, 0):
        _fail(f"Partition {part.path} has dt_min={dt_min} but expected time-of-day 00:00")
    if (dt_max.hour, dt_max.minute) != (23, 30):
        _fail(f"Partition {part.path} has dt_max={dt_max} but expected time-of-day 23:30")


def _validate_dst_option_b_partition(part: Partition, files: Sequence[Path]) -> None:
    # Exactly 48 unique time slots per day; no times beyond 23:30.
    lf = (
        pl.scan_parquet([str(p) for p in files])
        .select([
            pl.col("datetime"),
            pl.col("energy_kwh"),
        ])
        .with_columns([
            pl.col("datetime").dt.date().alias("d"),
            pl.col("datetime").dt.hour().alias("h"),
            pl.col("datetime").dt.minute().alias("m"),
        ])
    )

    # Count distinct (h,m) per day. Should be 48 everywhere.
    counts = lf.group_by("d").agg([
        pl.struct(["h", "m"]).n_unique().alias("slots"),
        pl.col("datetime").max().alias("dt_max_day"),
        pl.col("datetime").min().alias("dt_min_day"),
        pl.col("energy_kwh").null_count().alias("ekwh_nulls_day"),
        pl.len().alias("rows_day"),
    ])

    try:
        df = counts.collect(streaming=True)
    except Exception as e:
        _fail(f"Failed to collect DST Option B invariants for partition {part.path}: {e}")

    bad = df.filter(pl.col("slots") != 48)
    if bad.height > 0:
        sample = bad.select(["d", "slots", "rows_day"]).head(10).to_dicts()
        _fail(f"DST Option B violation: days with slots!=48 in partition {part.path}. Examples (up to 10): {sample}")

    # No timestamps beyond 23:30
    too_late = lf.filter((pl.col("h") > 23) | ((pl.col("h") == 23) & (pl.col("m") > 30))).select(
        pl.col("datetime").min().alias("first_bad")
    )
    try:
        first_bad = too_late.collect(streaming=True).row(0)[0]
    except Exception as e:
        _fail(f"Failed to check 'no timestamps beyond 23:30' in partition {part.path}: {e}")

    if first_bad is not None:
        _fail(
            f"DST Option B violation: found datetime beyond 23:30 in partition {part.path}. First example: {first_bad}"
        )

    # Coarse folded-energy sanity: ensure at least one day has non-null energy at 23:00 and 23:30.
    # (We avoid deep calendar inference; this is a light guardrail.)
    spot = (
        lf.filter(((pl.col("h") == 23) & (pl.col("m").is_in([0, 30]))) & pl.col("energy_kwh").is_not_null())
        .group_by("d")
        .agg(pl.struct(["h", "m"]).n_unique().alias("unique_slots_nonnull"))
        .filter(pl.col("unique_slots_nonnull") == 2)
        .select(pl.len().alias("days_with_both_slots_nonnull"))
    )
    try:
        days_ok = int(spot.collect(streaming=True).row(0)[0])
    except Exception as e:
        _fail(f"Failed DST spot-check for partition {part.path}: {e}")

    if days_ok == 0:
        _fail(
            f"DST Option B spot-check failed in partition {part.path}: "
            f"did not find any day with non-null energy_kwh at both 23:00 and 23:30."
        )


def _validate_no_duplicates_file(path: Path) -> int:
    """Check for duplicate (zip_code, account_identifier, datetime) within a single parquet file.

    Returns the row count for the file (used by row-count sanity reporting).
    Raises ValueError if duplicates are found.
    """
    lf = pl.scan_parquet(str(path))
    stats = lf.select([
        pl.len().alias("total_rows"),
        pl.struct(["zip_code", "account_identifier", "datetime"]).n_unique().alias("unique_keys"),
    ])
    try:
        row = stats.collect(streaming=True).row(0)
    except Exception as e:
        _fail(f"Failed to collect duplicate-check stats for {path}: {e}")

    total_rows, unique_keys = row
    if total_rows == 0:
        _fail(f"Empty parquet file (0 rows): {path}")

    if unique_keys < total_rows:
        n_dups = total_rows - unique_keys
        # Grab a small sample of duplicate keys for diagnostics.
        dup_sample = (
            lf.group_by(["zip_code", "account_identifier", "datetime"])
            .agg(pl.len().alias("cnt"))
            .filter(pl.col("cnt") > 1)
            .sort("cnt", descending=True)
            .head(5)
        )
        try:
            sample_dicts = dup_sample.collect(streaming=True).to_dicts()
        except Exception:
            sample_dicts = []
        _fail(
            f"Duplicate (zip_code, account_identifier, datetime) rows in {path}: "
            f"total_rows={total_rows}, unique_keys={unique_keys}, duplicates={n_dups}. "
            f"Top duplicates (up to 5): {sample_dicts}"
        )

    return int(total_rows)


def _composite_key_expr() -> pl.Expr:
    # Build a lexicographically comparable composite key without sorting.
    # datetime is included via cast to UTF-8 (stable ordering for ISO-like timestamp representation in Polars).
    return pl.concat_str([
        pl.col("zip_code").cast(pl.Utf8),
        pl.lit("\u001f"),  # unit separator
        pl.col("account_identifier").cast(pl.Utf8),
        pl.lit("\u001f"),
        pl.col("datetime").cast(pl.Utf8),
    ]).alias("_k")


def _check_sorted_full(path: Path) -> None:
    # Collect the composite key then check Series.is_sorted() (Expr.is_sorted was removed in Polars 1.38).
    try:
        k_series = pl.scan_parquet(str(path)).select([_composite_key_expr()]).collect(streaming=True)["_k"]
        is_sorted = bool(k_series.is_sorted())
    except Exception as e:
        _fail(f"Failed full sortedness check for {path}: {e}")
    if not is_sorted:
        # Find first break index with a second pass (still streaming-friendly).
        lf2 = (
            pl.scan_parquet(str(path))
            .select([_composite_key_expr()])
            .with_row_index("idx")
            .with_columns(pl.col("_k").shift(1).alias("_k_prev"))
            .filter(pl.col("_k_prev").is_not_null() & (pl.col("_k") < pl.col("_k_prev")))
            .select(["idx", "_k_prev", "_k"])
            .head(1)
        )
        try:
            df = lf2.collect(streaming=True)
        except Exception as e:
            _fail(f"Sortedness failed for {path}, and failed to locate break index: {e}")
        if df.height == 0:
            _fail(f"Sortedness failed for {path} (is_sorted=False) but could not locate first break (unexpected).")
        r = df.row(0)
        idx, prev_k, k = r
        _fail(f"Sortedness violation in {path} at row idx={idx}: prev_key={prev_k} > key={k}")


def _slice_keys(path: Path, offset: int, length: int) -> pl.DataFrame:
    lf = pl.scan_parquet(str(path)).select([pl.col(c) for c in SORT_KEY_COLS]).slice(offset, length)
    try:
        return lf.collect(streaming=True)
    except Exception as e:
        _fail(f"Failed to slice keys for {path} offset={offset} length={length}: {e}")


def _keys_is_sorted_df(df: pl.DataFrame) -> bool:
    if df.height <= 1:
        return True
    # Composite key as series (in-memory for this small df)
    k = df.select(
        pl.concat_str([
            pl.col("zip_code").cast(pl.Utf8),
            pl.lit("\u001f"),
            pl.col("account_identifier").cast(pl.Utf8),
            pl.lit("\u001f"),
            pl.col("datetime").cast(pl.Utf8),
        ]).alias("_k")
    )["_k"]
    return bool(k.is_sorted())


def _first_last_key(df: pl.DataFrame) -> tuple[str, str]:
    k = df.select(
        pl.concat_str([
            pl.col("zip_code").cast(pl.Utf8),
            pl.lit("\u001f"),
            pl.col("account_identifier").cast(pl.Utf8),
            pl.lit("\u001f"),
            pl.col("datetime").cast(pl.Utf8),
        ]).alias("_k")
    )["_k"]
    return str(k[0]), str(k[-1])


def _check_sorted_sample(path: Path, seed: int, max_windows: int, window_k: int, head_k: int) -> None:
    # Get row count cheaply
    try:
        n = int(pl.scan_parquet(str(path)).select(pl.len().alias("n")).collect(streaming=True).row(0)[0])
    except Exception as e:
        _fail(f"Failed to get row count for {path}: {e}")

    if n <= 1:
        return

    rng = random.Random(seed)  # noqa: S311

    slices: list[tuple[int, int, str]] = []
    # Head and tail
    slices.append((0, min(head_k, n), "head"))
    tail_len = min(head_k, n)
    slices.append((max(0, n - tail_len), tail_len, "tail"))

    # Deterministic random windows
    if n > window_k and max_windows > 0:
        for i in range(max_windows):
            off = rng.randrange(0, n - window_k + 1)
            slices.append((off, window_k, f"win{i}"))

    # Sort slices by offset to allow boundary checks
    slices.sort(key=lambda t: t[0])

    prev_last_key: str | None = None
    prev_tag: str | None = None

    for off, length, tag in slices:
        df = _slice_keys(path, off, length)

        if not _keys_is_sorted_df(df):
            _fail(
                f"Sortedness violation within slice tag={tag} offset={off} length={length} in file {path}. "
                f"Re-run with --check-mode full for exact break index."
            )

        first_k, last_k = _first_last_key(df)
        if prev_last_key is not None and first_k < prev_last_key:
            _fail(
                f"Sortedness violation across slice boundary in file {path}: "
                f"prev_slice={prev_tag} last_key={prev_last_key} > "
                f"slice={tag} first_key={first_k}. "
                f"Re-run with --check-mode full for exact break index."
            )

        prev_last_key = last_k
        prev_tag = tag


def _select_files_for_mode(files: Sequence[Path], mode: str, max_files: int | None, seed: int) -> list[Path]:
    if max_files is None or max_files <= 0 or max_files >= len(files):
        return list(files)

    if mode == "full":
        # Deterministic selection: first max_files by path order.
        return list(files)[:max_files]

    rng = random.Random(seed)  # noqa: S311
    idxs = list(range(len(files)))
    rng.shuffle(idxs)
    chosen = sorted(idxs[:max_files])
    return [files[i] for i in chosen]


def _compare_roots(root_a: Path, root_b: Path, max_files: int | None, seed: int) -> None:  # noqa: C901
    if not root_b.exists() or not root_b.is_dir():
        _fail(f"--compare-root is not a directory: {root_b}")

    def list_rel_files(root: Path) -> list[Path]:
        rels = []
        for p in root.rglob("*"):
            if p.is_file():
                rels.append(p.relative_to(root))
        rels.sort()
        return rels

    a_files = list_rel_files(root_a)
    b_files = list_rel_files(root_b)

    a_set = set(a_files)
    b_set = set(b_files)
    if a_set != b_set:
        only_a = sorted(a_set - b_set)[:20]
        only_b = sorted(b_set - a_set)[:20]
        _fail(
            "Determinism compare failed: directory trees differ.\n"
            f"  only_in_out_root (up to 20): {only_a}\n"
            f"  only_in_compare_root (up to 20): {only_b}"
        )

    # Compare sizes (cheap and stable)
    mismatches: list[str] = []
    for rel in a_files:
        pa = root_a / rel
        pb = root_b / rel
        sa = pa.stat().st_size
        sb = pb.stat().st_size
        if sa != sb:
            mismatches.append(f"{rel}: size_out={sa}, size_compare={sb}")
            if len(mismatches) >= 50:
                break

    if mismatches:
        _fail(
            "Determinism compare failed: file sizes differ (note: writer versions may legitimately differ; "
            "this check is intentionally strict on size).\n  " + "\n  ".join(mismatches)
        )

    # Optional: row counts for up to max_files parquet files (controlled)
    parquet_rels = [rel for rel in a_files if rel.suffix.lower() == ".parquet"]
    if not parquet_rels:
        return

    chosen: list[Path]
    if max_files is None or max_files <= 0 or max_files >= len(parquet_rels):
        chosen = parquet_rels
    else:
        rng = random.Random(seed)  # noqa: S311
        idxs = list(range(len(parquet_rels)))
        rng.shuffle(idxs)
        chosen = [parquet_rels[i] for i in sorted(idxs[:max_files])]

    row_mismatches: list[str] = []
    for rel in chosen:
        pa = root_a / rel
        pb = root_b / rel
        try:
            na = int(pl.scan_parquet(str(pa)).select(pl.len()).collect(streaming=True).row(0)[0])
            nb = int(pl.scan_parquet(str(pb)).select(pl.len()).collect(streaming=True).row(0)[0])
        except Exception as e:
            _fail(f"Determinism compare failed reading row counts for {rel}: {e}")
        if na != nb:
            row_mismatches.append(f"{rel}: rows_out={na}, rows_compare={nb}")
            if len(row_mismatches) >= 50:
                break

    if row_mismatches:
        _fail("Determinism compare failed: row counts differ.\n  " + "\n  ".join(row_mismatches))


def _validate_run_artifacts(run_dir: Path, expected_parquet_count: int | None = None) -> JsonDict:  # noqa: C901
    """Validate runner artifacts under a _runs/<YYYYMM>/<run_id>/ directory.

    Checks:
    - plan.json exists and is valid JSON
    - run_summary.json exists, is valid JSON, and reports total_failure=0
    - Manifest JSONL files exist; all file-level entries are success or skip
    - If expected_parquet_count is provided, cross-checks batches_written

    Returns a dict of artifact-check results for inclusion in the validation report.
    """
    if not run_dir.exists() or not run_dir.is_dir():
        _fail(f"--run-dir does not exist or is not a directory: {run_dir}")

    results: JsonDict = {"run_dir": str(run_dir)}

    # -- plan.json --
    plan_path = run_dir / "plan.json"
    if not plan_path.exists():
        _fail(f"Missing plan.json in run artifacts: {plan_path}")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        _fail(f"Invalid plan.json: {plan_path}: {e}")
    results["plan_n_inputs"] = len(plan.get("inputs_sorted", []))
    results["plan_n_batches"] = len(plan.get("batches", []))

    # -- run_summary.json --
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        _fail(f"Missing run_summary.json in run artifacts: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        _fail(f"Invalid run_summary.json: {summary_path}: {e}")

    total_failure = int(summary.get("total_failure", -1))
    total_success = int(summary.get("total_success", 0))
    total_skip = int(summary.get("total_skip", 0))
    batches_written = int(summary.get("batches_written", 0))
    stop_requested = summary.get("stop_requested", False)

    if total_failure != 0:
        _fail(
            f"run_summary.json reports total_failure={total_failure} (must be 0). "
            f"total_success={total_success}, total_skip={total_skip}. "
            f"Investigate logs at: {run_dir / 'logs' / 'run_log.jsonl'}"
        )

    if stop_requested:
        _fail(f"run_summary.json reports stop_requested=True. Run was interrupted: {summary_path}")

    results["summary_total_success"] = total_success
    results["summary_total_failure"] = total_failure
    results["summary_total_skip"] = total_skip
    results["summary_batches_written"] = batches_written

    if expected_parquet_count is not None and batches_written != expected_parquet_count:
        _fail(
            f"Batch count mismatch: run_summary.json reports batches_written={batches_written} "
            f"but discovered {expected_parquet_count} parquet files on disk."
        )

    # -- manifest JSONL --
    manifest_dir = run_dir / "manifests"
    if not manifest_dir.exists():
        _fail(f"Missing manifests directory: {manifest_dir}")

    manifest_files = sorted(manifest_dir.glob("manifest_*.jsonl"))
    if not manifest_files:
        _fail(f"No manifest_*.jsonl files found in {manifest_dir}")

    manifest_failures: list[str] = []
    manifest_success_count = 0
    manifest_skip_count = 0

    for mf in manifest_files:
        try:
            for line in mf.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                status = rec.get("status", "")
                if status == "success":
                    manifest_success_count += 1
                elif status == "skip":
                    manifest_skip_count += 1
                elif status == "failure":
                    inp = rec.get("input_path", "?")
                    exc = rec.get("exception_msg", "?")
                    manifest_failures.append(f"{inp}: {exc}")
        except (json.JSONDecodeError, OSError) as e:
            _fail(f"Error reading manifest file {mf}: {e}")

    if manifest_failures:
        sample = manifest_failures[:10]
        _fail(f"Manifest contains {len(manifest_failures)} failure entries (must be 0). Sample (up to 10): {sample}")

    results["manifest_files_checked"] = len(manifest_files)
    results["manifest_success_count"] = manifest_success_count
    results["manifest_skip_count"] = manifest_skip_count

    # -- batch summaries --
    summary_files = sorted(manifest_dir.glob("summary_*.json"))
    batch_failures = []
    for sf in summary_files:
        try:
            bs = json.loads(sf.read_text(encoding="utf-8"))
            if int(bs.get("n_failure", 0)) > 0:
                batch_failures.append(f"{sf.name}: n_failure={bs.get('n_failure')}")
        except (json.JSONDecodeError, OSError):
            batch_failures.append(f"{sf.name}: unreadable")

    if batch_failures:
        _fail(f"Batch summary files report failures: {batch_failures[:10]}")

    results["batch_summaries_checked"] = len(summary_files)

    return results


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    p = argparse.ArgumentParser(description="Validate ComEd month-output parquet dataset contract.")
    p.add_argument(
        "--out-root", required=True, help="Converted dataset output root containing year=YYYY/month=MM partitions."
    )
    p.add_argument(
        "--check-mode", choices=["full", "sample"], default="sample", help="Validation intensity for sortedness checks."
    )
    p.add_argument(
        "--dst-month-check", action="store_true", help="Enable DST Option B shape checks (48 slots/day; no extras)."
    )
    p.add_argument(
        "--compare-root", default=None, help="Optional second output root to compare for determinism invariants."
    )
    p.add_argument(
        "--max-files", type=int, default=None, help="Max parquet files to validate (selection depends on mode)."
    )
    p.add_argument("--seed", type=int, default=42, help="Deterministic seed for sampling selection/windows.")
    p.add_argument("--output-report", default=None, help="Write validation report JSON to this path.")
    p.add_argument(
        "--run-dir",
        default=None,
        help="Runner artifact directory (_runs/YYYYMM/<run_id>/) to validate plan.json, run_summary.json, manifests.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    out_root = Path(args.out_root).resolve()

    partitions = _discover_partitions(out_root)
    mapping = _discover_parquet_files(partitions)

    # Compare mode first (structural invariants); fail fast if mismatched.
    if args.compare_root is not None:
        _compare_roots(out_root, Path(args.compare_root).resolve(), args.max_files, args.seed)

    # Validate schema + per-file partition integrity + duplicates on selected files.
    total_files = sum(len(v) for v in mapping.values())
    checked_files = 0
    total_rows = 0
    per_file_rows: list[dict[str, object]] = []

    for part in partitions:
        files = mapping[part]
        if not files:
            # Partition exists but empty: fail-loud.
            _fail(
                f"Discovered partition {part.path} (year={part.year}, month={part.month}) but found zero parquet files under it."
            )

        selected = _select_files_for_mode(files, args.check_mode, args.max_files, args.seed)

        for f in selected:
            _validate_schema_on_file(f)
            _validate_partition_integrity_file(f, part)

            # Duplicate detection + row count (always runs; cheap single-pass aggregate).
            file_rows = _validate_no_duplicates_file(f)
            total_rows += file_rows
            per_file_rows.append({"file": f.name, "rows": file_rows})

            # Sortedness: per-file
            if args.check_mode == "full":
                _check_sorted_full(f)
            else:
                # Windows: modest defaults; tune if needed
                _check_sorted_sample(
                    f,
                    seed=args.seed,
                    max_windows=3,
                    window_k=5_000,
                    head_k=5_000,
                )

            checked_files += 1

        # Partition-level datetime invariants: always across *all* files in partition (cheap aggregates).
        _validate_datetime_invariants_partition(part, files)

        if args.dst_month_check:
            _validate_dst_option_b_partition(part, files)

    if checked_files == 0:
        _fail("No files validated (unexpected). Check --max-files and discovered outputs.")

    # Run artifact integrity (optional).
    run_artifact_results: JsonDict | None = None
    if args.run_dir is not None:
        run_artifact_results = _validate_run_artifacts(
            Path(args.run_dir).resolve(),
            expected_parquet_count=total_files,
        )

    # Build validation report
    checks_passed = [
        "schema_contract",
        "partition_integrity",
        "no_duplicates",
        "datetime_invariants",
        f"sortedness_{args.check_mode}",
    ]

    if args.dst_month_check:
        checks_passed.append("dst_option_b")

    if args.compare_root:
        checks_passed.append("determinism_compare")

    if run_artifact_results is not None:
        checks_passed.append("run_artifact_integrity")

    report: JsonDict = {
        "status": "pass",
        "timestamp": dt_mod.datetime.now(dt_mod.timezone.utc).isoformat(),
        "out_root": str(out_root),
        "partitions_validated": len(partitions),
        "partition_details": [{"year": p.year, "month": p.month, "files": len(mapping[p])} for p in partitions],
        "files_validated": checked_files,
        "total_files_discovered": total_files,
        "total_rows_validated": total_rows,
        "per_file_rows": per_file_rows,
        "check_mode": args.check_mode,
        "dst_month_check": args.dst_month_check,
        "checks_passed": checks_passed,
        "sort_order": list(SORT_KEY_COLS),
    }

    if args.compare_root:
        report["compare_root"] = str(Path(args.compare_root).resolve())

    if run_artifact_results is not None:
        report["run_artifacts"] = run_artifact_results

    # Write report if requested
    if args.output_report:
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as outfile:
            json.dump(report, outfile, indent=2)
        print(f"Validation report written to: {report_path}")

    # Minimal success signal (no prints during failure).
    print(
        f"OK: validated {checked_files} parquet files across {len(partitions)} partitions "
        f"(discovered total parquet files={total_files}, total rows validated={total_rows})."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
