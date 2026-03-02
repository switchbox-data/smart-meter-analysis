#!/usr/bin/env python3
"""Restructure 49 months of compacted Parquet data for export.

Copies files from the blessed-run layout into a clean, Spark-compatible layout:

  SOURCE:  <source_root>/out_YYYYMM_blessed/year=YYYY/month=MM/compacted_NNNN.parquet
  EXPORT:  <export_root>/YYYY/MM/part-NNNNN.parquet
                                 _SUCCESS.json

Three transformations applied during copy:
  1. year=YYYY/month=MM/  →  YYYY/MM/   (drop Hive prefixes)
  2. compacted_NNNN       →  part-NNNNN  (Spark naming, 5-digit zero-pad)
  3. _SUCCESS.json generated per month from pyarrow footer metadata

Source files are NEVER modified or deleted.  The export directory is a fresh
copy.  Use --force to overwrite an existing export.

Usage
-----
    uv run python scripts/csv_to_parquet/restructure_for_export.py \\
        --source-root /ebs/.../runs_bs0500_w2 \\
        --export-root /ebs/.../runs_bs0500_w2/export \\
        [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants — must stay in sync with the pipeline schema contract
# ---------------------------------------------------------------------------

SORT_KEYS: tuple[str, ...] = ("zip_code", "account_identifier", "datetime")

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

EXPECTED_MONTHS: tuple[str, ...] = (
    "202103",
    "202104",
    "202105",
    "202106",
    "202107",
    "202108",
    "202202",
    "202203",
    "202204",
    "202205",
    "202206",
    "202207",
    "202208",
    "202209",
    "202210",
    "202211",
    "202212",
    "202301",
    "202302",
    "202303",
    "202304",
    "202305",
    "202306",
    "202307",
    "202308",
    "202309",
    "202310",
    "202311",
    "202312",
    "202401",
    "202402",
    "202403",
    "202404",
    "202405",
    "202406",
    "202407",
    "202408",
    "202409",
    "202410",
    "202411",
    "202412",
    "202501",
    "202502",
    "202503",
    "202504",
    "202505",
    "202506",
    "202507",
    "202508",
)

EXPECTED_MONTH_COUNT: int = 49

# Pattern: out_YYYYMM_blessed
_BLESSED_DIR_RE = re.compile(r"^out_(\d{6})_blessed$")

# Pattern: compacted_NNNN.parquet  →  capture the 4-digit index
_COMPACTED_RE = re.compile(r"^compacted_(\d{4})\.parquet$")

JsonDict = dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _write_json(path: Path, obj: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _compacted_to_part_name(compacted_name: str) -> str:
    """Convert compacted_NNNN.parquet → part-NNNNN.parquet (4-digit → 5-digit)."""
    m = _COMPACTED_RE.match(compacted_name)
    if not m:
        raise ValueError(f"Unexpected filename (expected compacted_NNNN.parquet): {compacted_name!r}")
    idx = int(m.group(1))
    return f"part-{idx:05d}.parquet"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class MonthPlan:
    year_month: str  # YYYYMM
    year: str  # YYYY
    month: str  # MM
    source_dir: Path  # .../year=YYYY/month=MM/
    dest_dir: Path  # export/YYYY/MM/
    # sorted list of (src_path, dest_name) pairs
    files: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def n_files(self) -> int:
        return len(self.files)


@dataclass
class MonthResult:
    year_month: str
    n_files: int
    total_rows: int
    total_bytes: int
    status: str  # "OK" or "FAIL: <reason>"


# ---------------------------------------------------------------------------
# Discovery: scan source root for blessed dirs
# ---------------------------------------------------------------------------


def discover_months(source_root: Path) -> list[MonthPlan]:
    """Find all out_YYYYMM_blessed dirs and build MonthPlan list."""
    plans: dict[str, MonthPlan] = {}

    blessed_dirs = sorted(d for d in source_root.iterdir() if d.is_dir() and _BLESSED_DIR_RE.match(d.name))

    if not blessed_dirs:
        sys.exit(f"ERROR: No out_*_blessed directories found under {source_root}\n       Check --source-root path.")

    for bdir in blessed_dirs:
        m = _BLESSED_DIR_RE.match(bdir.name)
        if m is None:
            raise RuntimeError(f"BUG: {bdir.name!r} passed filter but didn't match regex")
        year_month = m.group(1)
        year = year_month[:4]
        month = year_month[4:6]

        # Locate year=YYYY/month=MM/ sub-path
        hive_year_dirs = sorted(bdir.glob("year=*"))
        if not hive_year_dirs:
            sys.exit(f"ERROR: {bdir} has no year=* subdirectory.\n       Expected: {bdir}/year={year}/month={month}/")

        found_source: Path | None = None
        for ydir in hive_year_dirs:
            candidate = ydir / f"month={month}"
            if candidate.is_dir():
                found_source = candidate
                break

        if found_source is None:
            sys.exit(
                f"ERROR: {bdir} has no month={month} subdirectory under any year=* dir.\n"
                f"       Dirs present: {[d.name for d in hive_year_dirs]}"
            )

        compacted = sorted(found_source.glob("compacted_*.parquet"))
        if not compacted:
            sys.exit(
                f"ERROR: {found_source} contains no compacted_*.parquet files.\n"
                f"       Files present: {[p.name for p in found_source.iterdir()]}"
            )

        # Validate all filenames match expected pattern
        file_pairs: list[tuple[Path, str]] = []
        for src in compacted:
            dest_name = _compacted_to_part_name(src.name)
            file_pairs.append((src, dest_name))

        if year_month in plans:
            sys.exit(f"ERROR: Duplicate year_month {year_month} found in source root.")

        plans[year_month] = MonthPlan(
            year_month=year_month,
            year=year,
            month=month,
            source_dir=found_source,
            dest_dir=Path("__placeholder__"),  # set below after export_root known
            files=file_pairs,
        )

    return list(plans.values())


def attach_dest_dirs(plans: list[MonthPlan], export_root: Path) -> None:
    for p in plans:
        p.dest_dir = export_root / p.year / p.month


# ---------------------------------------------------------------------------
# Validation: expected months
# ---------------------------------------------------------------------------


def validate_month_set(plans: list[MonthPlan]) -> None:
    found = {p.year_month for p in plans}
    expected = set(EXPECTED_MONTHS)

    missing = sorted(expected - found)
    extra = sorted(found - expected)

    errors: list[str] = []
    if missing:
        errors.append(f"Missing months ({len(missing)}): {missing}")
    if extra:
        errors.append(f"Unexpected months ({len(extra)}): {extra}")
    if len(found) != EXPECTED_MONTH_COUNT:
        errors.append(f"Expected {EXPECTED_MONTH_COUNT} months, found {len(found)}")

    if errors:
        sys.exit("ERROR: Month set mismatch:\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------
# Dry-run: print plan without copying
# ---------------------------------------------------------------------------


def print_dry_run(plans: list[MonthPlan], export_root: Path) -> None:
    total_files = sum(p.n_files for p in plans)
    total_bytes = sum(src.stat().st_size for p in plans for src, _ in p.files)

    print(f"\nDRY-RUN — {len(plans)} months, {total_files} files, {total_bytes / 1_073_741_824:.1f} GiB total\n")
    print(f"  export root: {export_root}\n")
    print(f"  {'MONTH':<8}  {'FILES':>5}  {'BYTES (GiB)':>11}  DEST DIR")
    print(f"  {'-' * 8}  {'-' * 5}  {'-' * 11}  {'-' * 50}")

    for p in sorted(plans, key=lambda x: x.year_month):
        month_bytes = sum(src.stat().st_size for src, _ in p.files)
        print(f"  {p.year_month:<8}  {p.n_files:>5}  {month_bytes / 1_073_741_824:>10.3f}  {p.dest_dir}")
        for src, dest_name in p.files:
            print(f"             {src.name:>20}  →  {dest_name}")

    print(f"\n  TOTAL: {len(plans)} months / {total_files} files / {total_bytes / 1_073_741_824:.2f} GiB\n")


# ---------------------------------------------------------------------------
# Copy: one month
# ---------------------------------------------------------------------------


def copy_month(plan: MonthPlan, force: bool, git_sha: str | None) -> MonthResult:
    dest_dir = plan.dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing part files
    existing_parts = sorted(dest_dir.glob("part-*.parquet"))
    if existing_parts and not force:
        sys.exit(
            f"ERROR: {dest_dir} already contains {len(existing_parts)} part-*.parquet file(s).\n"
            f"       Use --force to overwrite."
        )
    if existing_parts and force:
        for p in existing_parts:
            p.unlink()
        success_marker = dest_dir / "_SUCCESS.json"
        if success_marker.exists():
            success_marker.unlink()

    # Copy files
    for src, dest_name in plan.files:
        dest_path = dest_dir / dest_name
        shutil.copy2(src, dest_path)

    # Read metadata from destination files for _SUCCESS.json
    files_manifest: list[JsonDict] = []
    total_rows = 0
    total_bytes = 0

    for _src, dest_name in plan.files:
        dest_path = dest_dir / dest_name
        meta = pq.read_metadata(str(dest_path))
        size = int(dest_path.stat().st_size)
        rows = int(meta.num_rows)
        total_rows += rows
        total_bytes += size
        files_manifest.append({
            "name": dest_name,
            "size_bytes": size,
            "num_rows": rows,
            "num_row_groups": int(meta.num_row_groups),
        })

    marker: JsonDict = {
        "timestamp": _now_utc_iso(),
        "git_sha": git_sha,
        "year_month": plan.year_month,
        "n_files": plan.n_files,
        "total_rows": total_rows,
        "total_bytes": total_bytes,
        "sort_keys": list(SORT_KEYS),
        "schema": list(FINAL_LONG_COLS),
        "files": files_manifest,
    }
    _write_json(dest_dir / "_SUCCESS.json", marker)

    return MonthResult(
        year_month=plan.year_month,
        n_files=plan.n_files,
        total_rows=total_rows,
        total_bytes=total_bytes,
        status="OK",
    )


# ---------------------------------------------------------------------------
# Post-copy verification
# ---------------------------------------------------------------------------


def verify_export(plans: list[MonthPlan], export_root: Path) -> list[MonthResult]:
    """Verify all destination files: count, size, readability."""
    results: list[MonthResult] = []
    failures: list[str] = []

    for plan in sorted(plans, key=lambda x: x.year_month):
        dest_dir = plan.dest_dir
        dest_parts = sorted(dest_dir.glob("part-*.parquet"))

        # 1. File count
        if len(dest_parts) != plan.n_files:
            msg = f"{plan.year_month}: file count mismatch — expected {plan.n_files}, found {len(dest_parts)}"
            failures.append(msg)
            results.append(MonthResult(plan.year_month, plan.n_files, 0, 0, f"FAIL: {msg}"))
            continue

        month_rows = 0
        month_bytes = 0
        month_ok = True

        for (src, dest_name), dest_path in zip(plan.files, dest_parts):
            # 2. Byte-for-byte size match
            src_size = src.stat().st_size
            dst_size = dest_path.stat().st_size
            if src_size != dst_size:
                msg = f"{plan.year_month}/{dest_name}: size mismatch — src {src_size}, dst {dst_size}"
                failures.append(msg)
                month_ok = False
                continue

            # 3. Parquet readability
            try:
                meta = pq.read_metadata(str(dest_path))
                month_rows += int(meta.num_rows)
                month_bytes += dst_size
            except Exception as exc:
                msg = f"{plan.year_month}/{dest_name}: pq.read_metadata failed — {exc}"
                failures.append(msg)
                month_ok = False

        status = "OK" if month_ok else "FAIL: see above"
        results.append(MonthResult(plan.year_month, plan.n_files, month_rows, month_bytes, status))

    # 4. Total month count
    export_month_dirs = list(export_root.rglob("_SUCCESS.json"))
    if len(export_month_dirs) != EXPECTED_MONTH_COUNT:
        failures.append(
            f"Total month count: expected {EXPECTED_MONTH_COUNT}, found {len(export_month_dirs)} _SUCCESS.json files"
        )

    if failures:
        print("\nVERIFICATION FAILURES:")
        for f in failures:
            print(f"  ✗ {f}")
        return results

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(results: list[MonthResult]) -> bool:
    """Print summary table. Returns True if all OK."""
    header = f"{'MONTH':<8} | {'FILES':>5} | {'ROWS':>16} | {'BYTES':>14} | STATUS"
    sep = "-" * len(header)
    print(f"\n{header}")
    print(sep)

    total_files = 0
    total_rows = 0
    total_bytes = 0
    n_ok = 0

    for r in sorted(results, key=lambda x: x.year_month):
        row = f"{r.year_month:<8} | {r.n_files:>5} | {r.total_rows:>16,} | {r.total_bytes:>14,} | {r.status}"
        print(row)
        total_files += r.n_files
        total_rows += r.total_rows
        total_bytes += r.total_bytes
        if r.status == "OK":
            n_ok += 1

    print(sep)
    all_ok = n_ok == len(results)
    overall = f"{n_ok}/{len(results)} OK"
    print(f"{'TOTAL':<8} | {total_files:>5} | {total_rows:>16,} | {total_bytes:>14,} | {overall}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restructure 49 months of compacted Parquet for export.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Root containing out_YYYYMM_blessed/ directories.",
    )
    parser.add_argument(
        "--export-root",
        required=True,
        type=Path,
        help="Destination root for YYYY/MM/ export layout.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without copying any files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing part-*.parquet files in export dir.",
    )
    args = parser.parse_args()

    source_root: Path = args.source_root.resolve()
    export_root: Path = args.export_root.resolve()

    # Safety guard: export_root must be on /ebs
    if not str(export_root).startswith("/ebs"):
        sys.exit(f"ERROR: export-root must be under /ebs. Got: {export_root}")
    if not str(source_root).startswith("/ebs"):
        sys.exit(f"ERROR: source-root must be under /ebs. Got: {source_root}")

    # Export root must not be the same as source root
    if export_root == source_root:
        sys.exit("ERROR: --export-root must differ from --source-root.")

    # Export root must not be inside a blessed dir
    if any(part.endswith("_blessed") for part in export_root.parts):
        sys.exit("ERROR: --export-root must not be inside a blessed directory.")

    print(f"source-root : {source_root}")
    print(f"export-root : {export_root}")
    print(f"dry-run     : {args.dry_run}")
    print(f"force       : {args.force}")

    # ── 1. Discover months ────────────────────────────────────────────────────
    print("\nDiscovering source months...")
    plans = discover_months(source_root)
    attach_dest_dirs(plans, export_root)

    # ── 2. Validate month set ─────────────────────────────────────────────────
    validate_month_set(plans)
    print(f"Found {len(plans)} months ({sum(p.n_files for p in plans)} files total). ✓")

    # ── 3. Dry-run shortcut ───────────────────────────────────────────────────
    if args.dry_run:
        print_dry_run(plans, export_root)
        return

    # ── 4. Copy months ────────────────────────────────────────────────────────
    git_sha = _git_sha()
    print(f"\nCopying {len(plans)} months to {export_root} ...")
    results: list[MonthResult] = []

    for i, plan in enumerate(sorted(plans, key=lambda x: x.year_month), 1):
        month_bytes = sum(src.stat().st_size for src, _ in plan.files)
        print(
            f"  [{i:>2}/{len(plans)}] {plan.year_month}  "
            f"{plan.n_files} file(s)  {month_bytes / 1_073_741_824:.3f} GiB ...",
            end="",
            flush=True,
        )
        try:
            result = copy_month(plan, force=args.force, git_sha=git_sha)
            print("  OK")
            results.append(result)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            sys.exit(f"\nERROR: Copy failed for {plan.year_month}: {exc}")

    # ── 5. Verify ─────────────────────────────────────────────────────────────
    print("\nVerifying export...")
    verify_results = verify_export(plans, export_root)

    # ── 6. Summary ────────────────────────────────────────────────────────────
    all_ok = print_summary(verify_results)

    if not all_ok:
        sys.exit("EXPORT FAILED — see verification failures above.")

    print(f"Export complete. {len(plans)} months written to {export_root}")


if __name__ == "__main__":
    main()
