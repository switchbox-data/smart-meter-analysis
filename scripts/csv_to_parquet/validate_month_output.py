# scripts/csv_to_parquet/validate_month_output.py
from __future__ import annotations

import argparse
import datetime as dt_mod
import json
import random
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import polars as pl
import pyarrow.parquet as pq

JsonDict = dict[str, Any]

"""
Month-output Validator (QA + determinism + contract enforcement) for ComEd CSV->Parquet migration.

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

4) Datetime invariants (per partition, collected per-file and merged):
   - No null datetime.
   - min(datetime) has (hour,minute)==(0,0)
   - max(datetime) has (hour,minute)==(23,30)
   - All datetime values fall within (partition year, partition month) (no spillover).

5) DST Option B invariants (optional: --dst-month-check, collected per-file and merged):
   - Exactly 48 distinct time slots per day (no 49/50 slot days).
   - Ensures no timestamps beyond 23:30.
   - Spot-checks that (23:00 and 23:30) exist on at least one day with non-null energy_kwh (coarse sanity).

6) Sortedness + Uniqueness (non-tautological):
   - Validates strict lexicographic ordering by (zip_code, account_identifier, datetime).
   - Modes:
       --check-mode full   : PyArrow streaming pass across files; O(batch_size) memory; checks strictly
                              increasing composite key (sortedness + no duplicates in one pass).
       --check-mode sample : checks first/last K rows and deterministic random windows per file;
                              also checks boundaries and strictly-increasing keys within windows.

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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Partition:
    year: int
    month: int
    path: Path


@dataclass
class _DtStats:
    """Aggregated datetime statistics for a single file or merged partition."""

    dt_nulls: int = 0
    dt_min: dt_mod.datetime | None = None
    dt_max: dt_mod.datetime | None = None
    year_min: int | None = None
    year_max: int | None = None
    month_min: int | None = None
    month_max: int | None = None


@dataclass
class _DstFileStats:
    """Per-file DST statistics for merge across a partition."""

    day_slots: dict[dt_mod.date, set[tuple[int, int]]] = field(default_factory=dict)
    day_nonnull_late_slots: dict[dt_mod.date, set[tuple[int, int]]] = field(default_factory=dict)
    has_beyond_2330: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fail(msg: str) -> NoReturn:
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


def _composite_key_expr() -> pl.Expr:
    # Build a lexicographically comparable composite key without sorting.
    return pl.concat_str([
        pl.col("zip_code").cast(pl.Utf8),
        pl.lit("\u001f"),  # unit separator
        pl.col("account_identifier").cast(pl.Utf8),
        pl.lit("\u001f"),
        pl.col("datetime").cast(pl.Utf8),
    ]).alias("_k")


def _get_row_count_metadata(path: Path) -> int:
    """Get row count from parquet file metadata (O(1), no data scan)."""
    pf = pq.ParquetFile(str(path))
    return pf.metadata.num_rows


# ---------------------------------------------------------------------------
# Phase 1: Discovery
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase 2: Metadata checks (schema + partition integrity)
# ---------------------------------------------------------------------------


def _validate_schema_on_file(path: Path) -> None:
    schema = _read_parquet_schema(path)

    missing = [c for c in REQUIRED_SCHEMA if c not in schema]
    if missing:
        _fail(
            f"Schema missing required columns in file {path}:\n  missing={missing}\n  observed_cols={sorted(schema.keys())}"
        )

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
        row = lf.collect(engine="streaming").row(0)
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


# ---------------------------------------------------------------------------
# Phase 3a: Streaming sort + duplicate check (full mode)
# ---------------------------------------------------------------------------


def _streaming_sort_and_dup_check(
    files: Sequence[Path],
    batch_size: int = 65_536,
) -> tuple[int, list[dict[str, object]]]:
    """Combined streaming sortedness + uniqueness check across ordered files.

    Leverages the global sort order: data sorted by (zip_code, account_identifier, datetime)
    means duplicates are always adjacent.  Checks each composite key is strictly greater than
    the previous (sort order AND uniqueness in a single pass).

    Uses PyArrow iter_batches for O(batch_size) memory per pass.

    Returns (total_rows, per_file_rows).
    """
    prev_key: str | None = None
    total_rows = 0
    per_file_rows: list[dict[str, object]] = []

    for fpath in files:
        pf = pq.ParquetFile(str(fpath))
        file_rows = 0

        for batch in pf.iter_batches(batch_size=batch_size, columns=list(SORT_KEY_COLS)):
            n = batch.num_rows
            if n == 0:
                continue

            # Convert PyArrow batch -> Polars DataFrame for composite key
            df = pl.from_arrow(batch)
            keys = df.select(_composite_key_expr())["_k"]

            # -- Cross-batch/file boundary check --
            first_key = str(keys[0])
            if prev_key is not None:
                if first_key < prev_key:
                    _fail(
                        f"Sort violation at batch boundary (row ~{total_rows + file_rows}) "
                        f"in {fpath}: prev_key={prev_key!r} > first_key={first_key!r}"
                    )
                elif first_key == prev_key:
                    _fail(
                        f"Duplicate key at batch boundary (row ~{total_rows + file_rows}) in {fpath}: key={first_key!r}"
                    )

            # -- Within-batch: strictly increasing check --
            if n > 1:
                violations = (
                    df.select([_composite_key_expr()])
                    .with_row_index("_idx")
                    .with_columns(pl.col("_k").shift(1).alias("_kp"))
                    .filter(pl.col("_kp").is_not_null() & (pl.col("_k") <= pl.col("_kp")))
                    .head(1)
                )
                if violations.height > 0:
                    r = violations.row(0)
                    idx_in_batch, k, kp = r
                    abs_row = total_rows + file_rows + idx_in_batch
                    kind = "Duplicate key" if k == kp else "Sort violation"
                    _fail(f"{kind} at row ~{abs_row} in {fpath}: prev_key={kp!r}, key={k!r}")

            prev_key = str(keys[-1])
            file_rows += n

        if file_rows == 0:
            _fail(f"Empty parquet file (0 rows): {fpath}")

        per_file_rows.append({"file": fpath.name, "rows": file_rows})
        total_rows += file_rows

    return total_rows, per_file_rows


# ---------------------------------------------------------------------------
# Phase 3b: Sample-mode sort + duplicate check
# ---------------------------------------------------------------------------


def _slice_keys(path: Path, offset: int, length: int) -> pl.DataFrame:
    # Do NOT use engine="streaming" here: streaming may reorder rows for
    # sliced reads, which defeats the purpose of sortedness validation.
    # Slices are small (head_k / window_k rows of 3 key cols) so default
    # engine is both correct and fast enough.
    lf = pl.scan_parquet(str(path)).select([pl.col(c) for c in SORT_KEY_COLS]).slice(offset, length)
    try:
        return lf.collect()
    except Exception as e:
        _fail(f"Failed to slice keys for {path} offset={offset} length={length}: {e}")


def _keys_strictly_increasing_df(df: pl.DataFrame) -> bool:
    """Check that composite keys in df are strictly increasing (sorted + unique)."""
    if df.height <= 1:
        return True
    violations = (
        df.select([_composite_key_expr()])
        .with_row_index("_idx")
        .with_columns(pl.col("_k").shift(1).alias("_kp"))
        .filter(pl.col("_kp").is_not_null() & (pl.col("_k") <= pl.col("_kp")))
    )
    return violations.height == 0


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
    # Get row count cheaply from parquet metadata
    n = _get_row_count_metadata(path)
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
    prev_end: int = 0  # offset + length of previous slice

    for off, length, tag in slices:
        df = _slice_keys(path, off, length)

        if not _keys_strictly_increasing_df(df):
            # Locate the violation for diagnostics
            viol = (
                df.select([_composite_key_expr()])
                .with_row_index("_idx")
                .with_columns(pl.col("_k").shift(1).alias("_kp"))
                .filter(pl.col("_kp").is_not_null() & (pl.col("_k") <= pl.col("_kp")))
                .head(1)
            )
            if viol.height > 0:
                r = viol.row(0)
                idx, k, kp = r
                kind = "Duplicate key" if k == kp else "Sort violation"
                _fail(
                    f"{kind} in slice tag={tag} offset={off}+{idx} in file {path}: "
                    f"prev_key={kp!r}, key={k!r}. "
                    f"Re-run with --check-mode full for exact break index."
                )
            _fail(
                f"Strictly-increasing violation in slice tag={tag} offset={off} in file {path}. "
                f"Re-run with --check-mode full for exact break index."
            )

        first_k, last_k = _first_last_key(df)
        # Cross-slice boundary check: only valid when slices do NOT overlap.
        # Random windows can overlap with head/tail or each other; comparing
        # last-key-of-A to first-key-of-B is meaningless if B starts inside A.
        if prev_last_key is not None and off >= prev_end:
            if first_k < prev_last_key:
                _fail(
                    f"Sort violation across slice boundary in file {path}: "
                    f"prev_slice={prev_tag} last_key={prev_last_key!r} > "
                    f"slice={tag} first_key={first_k!r}. "
                    f"Re-run with --check-mode full for exact break index."
                )
            elif first_k == prev_last_key:
                _fail(
                    f"Duplicate key across slice boundary in file {path}: "
                    f"prev_slice={prev_tag} key={first_k!r}. "
                    f"Re-run with --check-mode full for exact break index."
                )

        prev_last_key = last_k
        prev_tag = tag
        prev_end = off + length


# ---------------------------------------------------------------------------
# Phase 4: Datetime invariants (per-file collect + merge)
# ---------------------------------------------------------------------------


def _collect_datetime_stats_file(path: Path) -> _DtStats:
    """Collect datetime aggregate stats from a single file (cheap aggregates)."""
    lf = pl.scan_parquet(str(path)).select([
        pl.col("datetime").null_count().alias("dt_nulls"),
        pl.col("datetime").min().alias("dt_min"),
        pl.col("datetime").max().alias("dt_max"),
        pl.col("datetime").dt.year().min().alias("dt_year_min"),
        pl.col("datetime").dt.year().max().alias("dt_year_max"),
        pl.col("datetime").dt.month().min().alias("dt_month_min"),
        pl.col("datetime").dt.month().max().alias("dt_month_max"),
    ])
    try:
        row = lf.collect(engine="streaming").row(0)
    except Exception as e:
        _fail(f"Failed to collect datetime stats for {path}: {e}")

    return _DtStats(
        dt_nulls=row[0],
        dt_min=row[1],
        dt_max=row[2],
        year_min=row[3],
        year_max=row[4],
        month_min=row[5],
        month_max=row[6],
    )


def _merge_dt_stats(stats_list: Sequence[_DtStats]) -> _DtStats:
    """Merge per-file datetime stats into partition-level stats."""
    merged = _DtStats()
    for s in stats_list:
        merged.dt_nulls += s.dt_nulls
        if s.dt_min is not None:
            merged.dt_min = min(merged.dt_min, s.dt_min) if merged.dt_min is not None else s.dt_min
        if s.dt_max is not None:
            merged.dt_max = max(merged.dt_max, s.dt_max) if merged.dt_max is not None else s.dt_max
        if s.year_min is not None:
            merged.year_min = min(merged.year_min, s.year_min) if merged.year_min is not None else s.year_min
        if s.year_max is not None:
            merged.year_max = max(merged.year_max, s.year_max) if merged.year_max is not None else s.year_max
        if s.month_min is not None:
            merged.month_min = min(merged.month_min, s.month_min) if merged.month_min is not None else s.month_min
        if s.month_max is not None:
            merged.month_max = max(merged.month_max, s.month_max) if merged.month_max is not None else s.month_max
    return merged


def _validate_datetime_stats_for_partition(merged: _DtStats, part: Partition) -> None:
    """Validate merged datetime stats against partition expectations."""
    if merged.dt_nulls != 0:
        _fail(f"Null datetime found in partition {part.path}: dt_nulls={merged.dt_nulls}")

    if merged.dt_min is None or merged.dt_max is None:
        _fail(f"Datetime min/max unexpectedly None in partition {part.path}")

    # Ensure within partition month
    if (
        merged.year_min != part.year
        or merged.year_max != part.year
        or merged.month_min != part.month
        or merged.month_max != part.month
    ):
        _fail(
            f"Datetime spillover in partition {part.path} (dir year={part.year}, month={part.month}) "
            f"but datetime year range=({merged.year_min},{merged.year_max}) "
            f"month range=({merged.month_min},{merged.month_max})"
        )

    # Time-of-day checks
    if (merged.dt_min.hour, merged.dt_min.minute) != (0, 0):
        _fail(f"Partition {part.path} has dt_min={merged.dt_min} but expected time-of-day 00:00")
    if (merged.dt_max.hour, merged.dt_max.minute) != (23, 30):
        _fail(f"Partition {part.path} has dt_max={merged.dt_max} but expected time-of-day 23:30")


# ---------------------------------------------------------------------------
# Phase 5: DST Option B (per-file collect + merge)
# ---------------------------------------------------------------------------


def _collect_dst_stats_file(path: Path) -> _DstFileStats:
    """Collect DST-relevant stats from a single file.

    Returns per-day unique (h,m) slot sets and spot-check data.
    Memory: O(days_in_month * 48) = ~1500 entries max.
    """
    lf_base = (
        pl.scan_parquet(str(path))
        .select(["datetime", "energy_kwh"])
        .with_columns([
            pl.col("datetime").dt.date().alias("d"),
            pl.col("datetime").dt.hour().alias("h"),
            pl.col("datetime").dt.minute().alias("m"),
        ])
    )

    # Unique (d, h, m) — at most 31 * 48 = 1488 rows regardless of account count
    slots_df = lf_base.select(["d", "h", "m"]).unique().collect(engine="streaming")
    day_slots: dict[dt_mod.date, set[tuple[int, int]]] = {}
    for row in slots_df.iter_rows():
        d, h, m = row
        day_slots.setdefault(d, set()).add((h, m))

    # Beyond 23:30 check
    beyond_count = int(
        lf_base.filter((pl.col("h") > 23) | ((pl.col("h") == 23) & (pl.col("m") > 30)))
        .select(pl.len())
        .collect(engine="streaming")
        .row(0)[0]
    )

    # Non-null energy at 23:00 and 23:30 (for spot-check merge)
    late_df = (
        lf_base.filter((pl.col("h") == 23) & pl.col("m").is_in([0, 30]) & pl.col("energy_kwh").is_not_null())
        .select(["d", "h", "m"])
        .unique()
        .collect(engine="streaming")
    )
    day_nonnull: dict[dt_mod.date, set[tuple[int, int]]] = {}
    for row in late_df.iter_rows():
        d, h, m = row
        day_nonnull.setdefault(d, set()).add((h, m))

    return _DstFileStats(
        day_slots=day_slots,
        day_nonnull_late_slots=day_nonnull,
        has_beyond_2330=beyond_count > 0,
    )


def _validate_dst_for_partition(part: Partition, files: Sequence[Path]) -> None:
    """Validate DST Option B by collecting per-file stats and merging."""
    merged_slots: dict[dt_mod.date, set[tuple[int, int]]] = {}
    merged_nonnull: dict[dt_mod.date, set[tuple[int, int]]] = {}
    any_beyond = False

    for f in files:
        stats = _collect_dst_stats_file(f)
        for d, s in stats.day_slots.items():
            merged_slots.setdefault(d, set()).update(s)
        for d, s in stats.day_nonnull_late_slots.items():
            merged_nonnull.setdefault(d, set()).update(s)
        if stats.has_beyond_2330:
            any_beyond = True

    # Check 1: exactly 48 unique time slots per day
    bad_days = [(d, len(s)) for d, s in merged_slots.items() if len(s) != 48]
    if bad_days:
        bad_days.sort()
        sample = [{"date": str(d), "slots": n} for d, n in bad_days[:10]]
        _fail(f"DST Option B violation: days with slots!=48 in partition {part.path}. Examples (up to 10): {sample}")

    # Check 2: no timestamps beyond 23:30
    if any_beyond:
        _fail(f"DST Option B violation: found datetime beyond 23:30 in partition {part.path}.")

    # Check 3: at least one day has non-null energy_kwh at both 23:00 and 23:30
    days_with_both = sum(1 for s in merged_nonnull.values() if (23, 0) in s and (23, 30) in s)
    if days_with_both == 0:
        _fail(
            f"DST Option B spot-check failed in partition {part.path}: "
            f"did not find any day with non-null energy_kwh at both 23:00 and 23:30."
        )


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Determinism compare
# ---------------------------------------------------------------------------


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
            na = int(pl.scan_parquet(str(pa)).select(pl.len()).collect(engine="streaming").row(0)[0])
            nb = int(pl.scan_parquet(str(pb)).select(pl.len()).collect(engine="streaming").row(0)[0])
        except Exception as e:
            _fail(f"Determinism compare failed reading row counts for {rel}: {e}")
        if na != nb:
            row_mismatches.append(f"{rel}: rows_out={na}, rows_compare={nb}")
            if len(row_mismatches) >= 50:
                break

    if row_mismatches:
        _fail("Determinism compare failed: row counts differ.\n  " + "\n  ".join(row_mismatches))


# ---------------------------------------------------------------------------
# Run artifact validation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main (phase-based architecture)
# ---------------------------------------------------------------------------


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

    # ── Phase 1: Discovery ──────────────────────────────────────────────
    partitions = _discover_partitions(out_root)
    mapping = _discover_parquet_files(partitions)

    # ── Phase 1b: Compare mode (structural, fail fast) ──────────────────
    if args.compare_root is not None:
        _compare_roots(out_root, Path(args.compare_root).resolve(), args.max_files, args.seed)

    # ── Phase 2: Metadata checks (schema + partition integrity) ─────────
    total_files = sum(len(v) for v in mapping.values())
    checked_files = 0

    for part in partitions:
        files = mapping[part]
        if not files:
            _fail(
                f"Discovered partition {part.path} (year={part.year}, month={part.month}) "
                f"but found zero parquet files under it."
            )

        selected = _select_files_for_mode(files, args.check_mode, args.max_files, args.seed)
        for f in selected:
            _validate_schema_on_file(f)
            _validate_partition_integrity_file(f, part)
            checked_files += 1

    if checked_files == 0:
        _fail("No files validated (unexpected). Check --max-files and discovered outputs.")

    # ── Phase 3: Sortedness + duplicates + row counts ───────────────────
    total_rows = 0
    per_file_rows: list[dict[str, object]] = []

    for part in partitions:
        files = mapping[part]
        selected = _select_files_for_mode(files, args.check_mode, args.max_files, args.seed)

        if args.check_mode == "full":
            # Combined streaming sort+dup check — O(batch_size) memory
            partition_rows, partition_per_file = _streaming_sort_and_dup_check(selected)
            total_rows += partition_rows
            per_file_rows.extend(partition_per_file)
        else:
            # Sample mode: enhanced strict-increasing check per file
            for f in selected:
                _check_sorted_sample(
                    f,
                    seed=args.seed,
                    max_windows=3,
                    window_k=5_000,
                    head_k=5_000,
                )
                frows = _get_row_count_metadata(f)
                total_rows += frows
                per_file_rows.append({"file": f.name, "rows": frows})

    # ── Phase 4: Datetime invariants (all files, per-file + merge) ──────
    for part in partitions:
        files = mapping[part]
        dt_stats_list = [_collect_datetime_stats_file(f) for f in files]
        merged = _merge_dt_stats(dt_stats_list)
        _validate_datetime_stats_for_partition(merged, part)

    # ── Phase 5: DST Option B (all files, per-file + merge) ────────────
    if args.dst_month_check:
        for part in partitions:
            _validate_dst_for_partition(part, mapping[part])

    # ── Phase 6: Run artifact integrity (optional) ─────────────────────
    run_artifact_results: JsonDict | None = None
    if args.run_dir is not None:
        run_artifact_results = _validate_run_artifacts(
            Path(args.run_dir).resolve(),
            expected_parquet_count=total_files,
        )

    # ── Phase 7: Build validation report ────────────────────────────────
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
