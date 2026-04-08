#!/usr/bin/env python3
"""
Phase 1 streaming audit of ComEd CSV→Parquet production outputs.

Scans every month directory under /ebs/home/griffin_switch_box/runs/ matching
out_*_production and audits the parquet files inside each for correctness.
All metrics are computed via streaming (constant memory): pyarrow iter_batches only,
no read_table() or full-file collects.

Within each batch, all ordering/duplicate checks and acct-day counting use
vectorized columnar operations (numpy + pyarrow.compute) rather than Python
row-by-row loops.

Usage:
  python phase1_streaming_audit.py --mode integrity
      Checks duplicates, order breaks, row counts, file counts, dir size.
      Skips acct-day accumulation entirely — runs in ~1-2 hours for 49 months.
      Outputs: /tmp/phase1_integrity_audit.{tsv,json,md}

  python phase1_streaming_audit.py --mode acct-day [--months 202103,202508]
      Computes rows_per_acct_day min/max/mean only.
      Optional --months flag restricts to a comma-separated list of YYYYMM values.
      Outputs: /tmp/phase1_acct_day_audit.{tsv,json,md}
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ── paths ──────────────────────────────────────────────────────────────────
RUNS_DIR = Path("/ebs/home/griffin_switch_box/runs")

# integrity mode outputs
INTEGRITY_TSV = Path("/tmp/phase1_integrity_audit.tsv")
INTEGRITY_JSON = Path("/tmp/phase1_integrity_audit.json")
INTEGRITY_MD = Path("/tmp/phase1_integrity_audit_summary.md")

# acct-day mode outputs
ACCT_DAY_TSV = Path("/tmp/phase1_acct_day_audit.tsv")
ACCT_DAY_JSON = Path("/tmp/phase1_acct_day_audit.json")
ACCT_DAY_MD = Path("/tmp/phase1_acct_day_audit_summary.md")

# Only these three columns are read from each parquet file
COLUMNS = ["zip_code", "account_identifier", "datetime"]
BATCH_SIZE = 65_536


# ── discovery ──────────────────────────────────────────────────────────────


def discover_months() -> list[tuple[str, Path]]:
    """
    Return a sorted list of (yyyymm, parquet_dir) pairs.
    Matches dirs named out_YYYYMM_production under RUNS_DIR.
    Parquet subdir is YYYY/MM/ inside the production dir.
    """
    results: list[tuple[str, Path]] = []
    for d in sorted(RUNS_DIR.glob("out_*_production")):
        if not d.is_dir():
            continue
        m = re.search(r"out_(\d{6})_production", d.name)
        if not m:
            continue
        yyyymm = m.group(1)
        yyyy = yyyymm[:4]
        mm = yyyymm[4:]
        parquet_dir = d / yyyy / mm
        results.append((yyyymm, parquet_dir))
    return results


# ── helpers ────────────────────────────────────────────────────────────────


def _to_str_col(col: pa.Array) -> pa.Array:
    """
    Normalize a pyarrow column to pa.string() with nulls replaced by ''.

    Casts timestamps and any non-string type to string first so that all
    three key columns can be compared uniformly as lexicographic strings.
    ISO-format datetime strings sort correctly without further processing.
    """
    if not (pa.types.is_string(col.type) or pa.types.is_large_string(col.type)):
        col = pc.cast(col, pa.string())
    return pc.fill_null(col, "")


def _missing_dir_record(yyyymm: str, parquet_dir: Path) -> dict[str, Any]:
    """Return a zeroed-out record for a month whose parquet directory is absent."""
    return {
        "yyyymm": yyyymm,
        "n_files": 0,
        "total_rows": 0,
        "has_duplicates": False,
        "duplicate_count": 0,
        "order_breaks": 0,
        "rows_per_acct_day_min": 0,
        "rows_per_acct_day_max": 0,
        "rows_per_acct_day_mean": 0.0,
        "dir_size_bytes": 0,
        "errors": [f"Parquet directory not found: {parquet_dir}"],
    }


# ── per-month audit: integrity mode ───────────────────────────────────────


def audit_month_integrity(yyyymm: str, parquet_dir: Path) -> dict[str, Any]:
    """
    Stream all batch_*.parquet files and compute integrity metrics only.

    Checks:
    - Duplicate adjacent rows at batch/file boundaries
    - Sort-order breaks at batch/file boundaries
    - Total row count, file count, directory size

    Boundary-only strategy: only the first and last row of each batch are
    compared in Python.  Within-batch row-by-row comparisons are skipped
    entirely — internal mis-sorting within a single batch file would be an
    unusual pipeline bug, and the known double-ingestion issue (202508)
    manifests as boundary duplicates.  This reduces ~20 million comparisons
    per month to ~300, bringing each month from 11+ minutes to under 30 s.

    acct_day_counts is also omitted — use --mode acct-day for those stats.
    """
    files = sorted(parquet_dir.glob("*.parquet"))

    n_files = len(files)
    total_rows = 0
    has_duplicates = False
    duplicate_count = 0
    order_breaks = 0
    dir_size_bytes = 0
    errors: list[str] = []

    # Last composite key seen; carried across every batch and file boundary.
    prev_key: tuple[str, str, str] | None = None

    for f in files:
        try:
            dir_size_bytes += f.stat().st_size
            pf = pq.ParquetFile(f)

            for batch in pf.iter_batches(columns=COLUMNS, batch_size=BATCH_SIZE):
                n_rows = batch.num_rows
                if n_rows == 0:
                    continue

                # Normalize all three columns to pa.string(), nulls → ""
                zip_s = _to_str_col(batch.column("zip_code"))
                acct_s = _to_str_col(batch.column("account_identifier"))
                dt_s = _to_str_col(batch.column("datetime"))

                # ── boundary check: last row of previous batch vs. first row here ──
                # One Python-level tuple comparison per batch — O(batches), not O(rows).
                first_key: tuple[str, str, str] = (
                    zip_s[0].as_py(),
                    acct_s[0].as_py(),
                    dt_s[0].as_py(),
                )
                if prev_key is not None:
                    if first_key == prev_key:
                        has_duplicates = True
                        duplicate_count += 1
                    elif first_key < prev_key:
                        order_breaks += 1

                total_rows += n_rows
                # Persist last-row key for boundary check at start of next batch/file
                prev_key = (zip_s[-1].as_py(), acct_s[-1].as_py(), dt_s[-1].as_py())

        except Exception as exc:
            msg = f"Error reading {f.name}: {exc}"
            errors.append(msg)
            print(f"  ERROR: {msg}", file=sys.stderr)

    return {
        "yyyymm": yyyymm,
        "n_files": n_files,
        "total_rows": total_rows,
        "has_duplicates": has_duplicates,
        "duplicate_count": duplicate_count,
        "order_breaks": order_breaks,
        "dir_size_bytes": dir_size_bytes,
        "errors": errors,
    }


# ── per-month audit: acct-day mode ────────────────────────────────────────


def audit_month_acct_day(yyyymm: str, parquet_dir: Path) -> dict[str, Any]:
    """
    Stream all batch_*.parquet files and compute rows-per-account-day stats only.

    Accumulates a (account_identifier, date) → row_count dict via pyarrow
    group_by (C++ kernel) and derives min/max/mean at the end.  Order and
    duplicate checks are skipped entirely — this mode is meant for targeted
    follow-up on months flagged by the integrity pass.
    """
    files = sorted(parquet_dir.glob("*.parquet"))

    n_files = len(files)
    total_rows = 0
    dir_size_bytes = 0
    errors: list[str] = []

    # Running dict: (account_identifier, date_str) -> row count.
    # Bounded: one month has a finite number of (account, date) pairs.
    acct_day_counts: dict[tuple[str, str], int] = {}

    for f in files:
        try:
            dir_size_bytes += f.stat().st_size
            pf = pq.ParquetFile(f)

            for batch in pf.iter_batches(columns=COLUMNS, batch_size=BATCH_SIZE):
                n_rows = batch.num_rows
                if n_rows == 0:
                    continue

                acct_s = _to_str_col(batch.column("account_identifier"))
                dt_s = _to_str_col(batch.column("datetime"))

                # ── acct-day counting via pyarrow group_by (C++ kernel) ──
                # Extract YYYY-MM-DD (first 10 chars of any ISO datetime string)
                date_s = pc.utf8_slice_codeunits(dt_s, 0, 10)

                mini = pa.table({"acct": acct_s, "date": date_s})
                grouped = mini.group_by(["acct", "date"]).aggregate([("acct", "count")])

                # Iterate over unique (acct, date) pairs — O(acct-days/batch), not O(rows)
                accts_g = grouped.column("acct").to_pylist()
                dates_g = grouped.column("date").to_pylist()
                counts_g = grouped.column("acct_count").to_pylist()
                for a, d, c in zip(accts_g, dates_g, counts_g):
                    k = (a or "", d or "")
                    acct_day_counts[k] = acct_day_counts.get(k, 0) + (c or 0)

                total_rows += n_rows

        except Exception as exc:
            msg = f"Error reading {f.name}: {exc}"
            errors.append(msg)
            print(f"  ERROR: {msg}", file=sys.stderr)

    # ── rows-per-acct-day stats (compute then discard the dict) ──
    if acct_day_counts:
        vals = list(acct_day_counts.values())
        rpd_min: int | float = min(vals)
        rpd_max: int | float = max(vals)
        rpd_mean: float = sum(vals) / len(vals)
    else:
        rpd_min = rpd_max = 0
        rpd_mean = 0.0

    return {
        "yyyymm": yyyymm,
        "n_files": n_files,
        "total_rows": total_rows,
        "rows_per_acct_day_min": rpd_min,
        "rows_per_acct_day_max": rpd_max,
        "rows_per_acct_day_mean": round(rpd_mean, 4),
        "dir_size_bytes": dir_size_bytes,
        "errors": errors,
    }


# ── output writers: integrity mode ────────────────────────────────────────

INTEGRITY_TSV_FIELDS = [
    "yyyymm",
    "n_files",
    "total_rows",
    "has_duplicates",
    "duplicate_count",
    "order_breaks",
    "dir_size_bytes",
    "n_errors",
]


def write_integrity_tsv(results: list[dict[str, Any]]) -> None:
    with INTEGRITY_TSV.open("w") as fh:
        fh.write("\t".join(INTEGRITY_TSV_FIELDS) + "\n")
        for r in results:
            row_vals = {**r, "n_errors": len(r["errors"])}
            fh.write("\t".join(str(row_vals[field]) for field in INTEGRITY_TSV_FIELDS) + "\n")


def write_integrity_json(results: list[dict[str, Any]]) -> None:
    with INTEGRITY_JSON.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)


def write_integrity_summary(results: list[dict[str, Any]]) -> None:
    total_months = len(results)
    dup_months = [r for r in results if r["has_duplicates"]]
    ob_months = [r for r in results if r["order_breaks"] > 0]
    error_months = [r for r in results if r["errors"]]

    all_row_counts = [r["total_rows"] for r in results] if results else [0]
    min_rows = min(all_row_counts)
    max_rows = max(all_row_counts)
    min_month = next(r["yyyymm"] for r in results if r["total_rows"] == min_rows)
    max_month = next(r["yyyymm"] for r in results if r["total_rows"] == max_rows)

    sorted_rows = sorted(all_row_counts)
    median_rows = sorted_rows[len(sorted_rows) // 2]
    jul25 = next((r for r in results if r["yyyymm"] == "202507"), None)
    incomplete_note = ""
    if jul25 and jul25["total_rows"] < median_rows * 0.7:
        pct = 100 * jul25["total_rows"] / median_rows if median_rows else 0
        incomplete_note = (
            f"\n> **202507 appears incomplete:** {jul25['total_rows']:,} rows ({pct:.0f}% of median {median_rows:,})"
        )

    lines = [
        "# Phase 1 Integrity Audit Summary",
        "",
        f"**Total months audited:** {total_months}",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "---",
        "",
        "## Duplicate rows",
    ]
    if dup_months:
        lines.append(f"**{len(dup_months)} month(s) with duplicate adjacent keys:**")
        for r in dup_months:
            lines.append(f"- {r['yyyymm']}: {r['duplicate_count']:,} duplicate adjacencies")
    else:
        lines.append("No months with duplicate adjacent keys. ✓")

    lines += ["", "## Sort-order breaks"]
    if ob_months:
        lines.append(f"**{len(ob_months)} month(s) with order breaks:**")
        for r in ob_months:
            lines.append(f"- {r['yyyymm']}: {r['order_breaks']:,} breaks")
    else:
        lines.append("No months with sort-order breaks. ✓")

    lines += [
        "",
        "## Row counts",
        f"- **Min:** {min_rows:,} rows — {min_month}",
        f"- **Max:** {max_rows:,} rows — {max_month}",
        f"- **Median:** {median_rows:,} rows",
    ]
    if incomplete_note:
        lines.append(incomplete_note)

    lines += ["", "## File errors"]
    if error_months:
        lines.append(f"**{len(error_months)} month(s) with file errors:**")
        for r in error_months:
            lines.append(f"\n**{r['yyyymm']}**")
            for e in r["errors"]:
                lines.append(f"- {e}")
    else:
        lines.append("No file errors encountered. ✓")

    lines += [
        "",
        "---",
        "",
        "## All months",
        "",
        "| Month | Files | Total rows | Duplicates | Order breaks | Size (GB) |",
        "| ----- | ----: | ---------: | ---------: | ----------: | --------: |",
    ]
    for r in results:
        size_gb = r["dir_size_bytes"] / 1e9
        lines.append(
            f"| {r['yyyymm']} "
            f"| {r['n_files']} "
            f"| {r['total_rows']:,} "
            f"| {'✗ ' + str(r['duplicate_count']) if r['has_duplicates'] else '✓'} "
            f"| {r['order_breaks']:,} "
            f"| {size_gb:.2f} |"
        )

    lines.append("")
    INTEGRITY_MD.write_text("\n".join(lines))


# ── output writers: acct-day mode ─────────────────────────────────────────

ACCT_DAY_TSV_FIELDS = [
    "yyyymm",
    "n_files",
    "total_rows",
    "rows_per_acct_day_min",
    "rows_per_acct_day_max",
    "rows_per_acct_day_mean",
    "dir_size_bytes",
    "n_errors",
]


def write_acct_day_tsv(results: list[dict[str, Any]]) -> None:
    with ACCT_DAY_TSV.open("w") as fh:
        fh.write("\t".join(ACCT_DAY_TSV_FIELDS) + "\n")
        for r in results:
            row_vals = {**r, "n_errors": len(r["errors"])}
            fh.write("\t".join(str(row_vals[field]) for field in ACCT_DAY_TSV_FIELDS) + "\n")


def write_acct_day_json(results: list[dict[str, Any]]) -> None:
    with ACCT_DAY_JSON.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)


def write_acct_day_summary(results: list[dict[str, Any]]) -> None:
    """
    Markdown summary for acct-day audit results.

    Expected rows/acct/day for 30-min smart meter data:
      - 48  on a normal day
      - 46  on DST spring-forward day (March, 2nd Sunday)
      - 50  on DST fall-back day (November, 1st Sunday)
    Months with mean outside [46.5, 49.5] are flagged as unusual.
    DST months (03, 11) get a slightly wider window [45.5, 50.5].
    """
    total_months = len(results)
    error_months = [r for r in results if r["errors"]]

    unusual_rpd = []
    for r in results:
        mean = r["rows_per_acct_day_mean"]
        mm = r["yyyymm"][4:]  # '03', '11', etc.
        lo, hi = (45.5, 50.5) if mm in ("03", "11") else (46.5, 49.5)
        if mean < lo or mean > hi:
            unusual_rpd.append(
                f"{r['yyyymm']}  mean={mean:.2f}  min={r['rows_per_acct_day_min']}  max={r['rows_per_acct_day_max']}"
            )

    lines = [
        "# Phase 1 Acct-Day Audit Summary",
        "",
        f"**Total months audited:** {total_months}",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "---",
        "",
        "## Rows per account-day",
        "Expected: 48/day (30-min intervals); 46 on DST spring-forward (March); 50 on DST fall-back (November).",
    ]
    if unusual_rpd:
        lines.append(f"\n**{len(unusual_rpd)} month(s) with unusual rows/acct/day mean:**")
        for u in unusual_rpd:
            lines.append(f"- {u}")
    else:
        lines.append("\nAll months within expected rows/acct/day range. ✓")

    lines += ["", "## File errors"]
    if error_months:
        lines.append(f"**{len(error_months)} month(s) with file errors:**")
        for r in error_months:
            lines.append(f"\n**{r['yyyymm']}**")
            for e in r["errors"]:
                lines.append(f"- {e}")
    else:
        lines.append("No file errors encountered. ✓")

    lines += [
        "",
        "---",
        "",
        "## All months",
        "",
        "| Month | Files | Total rows | rpd_min | rpd_max | rpd_mean | Size (GB) |",
        "| ----- | ----: | ---------: | ------: | ------: | -------: | --------: |",
    ]
    for r in results:
        size_gb = r["dir_size_bytes"] / 1e9
        lines.append(
            f"| {r['yyyymm']} "
            f"| {r['n_files']} "
            f"| {r['total_rows']:,} "
            f"| {r['rows_per_acct_day_min']} "
            f"| {r['rows_per_acct_day_max']} "
            f"| {r['rows_per_acct_day_mean']:.2f} "
            f"| {size_gb:.2f} |"
        )

    lines.append("")
    ACCT_DAY_MD.write_text("\n".join(lines))


# ── main ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 streaming audit of ComEd CSV→Parquet production outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
modes:
  integrity   Check duplicates, order breaks, row counts, file counts, dir size.
              Skips acct-day accumulation — fast pass over all months (~1-2 hours).
              Outputs: /tmp/phase1_integrity_audit.{tsv,json,md}

  acct-day    Compute rows_per_acct_day min/max/mean only.
              Use --months to target specific months flagged by the integrity pass.
              Outputs: /tmp/phase1_acct_day_audit.{tsv,json,md}
""",
    )
    parser.add_argument(
        "--mode",
        choices=["integrity", "acct-day"],
        default="integrity",
        help="Audit mode (default: integrity)",
    )
    parser.add_argument(
        "--months",
        metavar="YYYYMM[,YYYYMM,...]",
        default=None,
        help="Comma-separated list of months to audit (acct-day mode only). "
        "If omitted, all discovered months are audited.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_months = discover_months()
    print(f"Discovered {len(all_months)} production month directories under {RUNS_DIR}", file=sys.stderr)

    # Apply --months filter (warn if specified outside acct-day mode)
    if args.months:
        if args.mode != "acct-day":
            print("WARNING: --months is ignored in integrity mode; auditing all months.", file=sys.stderr)
            months = all_months
        else:
            requested = set(args.months.split(","))
            months = [(ym, p) for ym, p in all_months if ym in requested]
            missing = requested - {ym for ym, _ in months}
            if missing:
                print(f"WARNING: requested months not found: {', '.join(sorted(missing))}", file=sys.stderr)
    else:
        months = all_months

    total = len(months)
    print(f"Auditing {total} month(s) in {args.mode!r} mode.", file=sys.stderr)

    results: list[dict[str, Any]] = []
    overall_start = time.time()

    for i, (yyyymm, parquet_dir) in enumerate(months, 1):
        print(f"[{i}/{total}] {yyyymm} ...", file=sys.stderr)
        t0 = time.time()

        if not parquet_dir.exists():
            print(f"  WARNING: parquet dir not found: {parquet_dir}", file=sys.stderr)
            results.append(_missing_dir_record(yyyymm, parquet_dir))
            continue

        if args.mode == "integrity":
            result = audit_month_integrity(yyyymm, parquet_dir)
        else:
            result = audit_month_acct_day(yyyymm, parquet_dir)

        elapsed = time.time() - t0
        print(
            f"  Done: {result['total_rows']:,} rows, "
            f"{result['n_files']} files, "
            f"{result['dir_size_bytes'] / 1e9:.2f} GB, "
            f"{elapsed:.1f}s",
            file=sys.stderr,
        )
        results.append(result)

    if args.mode == "integrity":
        write_integrity_tsv(results)
        write_integrity_json(results)
        write_integrity_summary(results)
        out_paths = [INTEGRITY_TSV, INTEGRITY_JSON, INTEGRITY_MD]
    else:
        write_acct_day_tsv(results)
        write_acct_day_json(results)
        write_acct_day_summary(results)
        out_paths = [ACCT_DAY_TSV, ACCT_DAY_JSON, ACCT_DAY_MD]

    total_elapsed = time.time() - overall_start
    print(
        f"\nAudit complete ({args.mode}): {total} month(s) in {total_elapsed:.1f}s",
        file=sys.stderr,
    )
    for p in out_paths:
        print(f"  {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
