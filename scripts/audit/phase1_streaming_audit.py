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

Outputs:
  /tmp/phase1_streaming_audit.tsv
  /tmp/phase1_streaming_audit.json
  /tmp/phase1_streaming_audit_summary.md
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ── paths ──────────────────────────────────────────────────────────────────
RUNS_DIR = Path("/ebs/home/griffin_switch_box/runs")
OUTPUT_TSV = Path("/tmp/phase1_streaming_audit.tsv")
OUTPUT_JSON = Path("/tmp/phase1_streaming_audit.json")
OUTPUT_MD = Path("/tmp/phase1_streaming_audit_summary.md")

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


# ── per-month audit ────────────────────────────────────────────────────────


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


def audit_month(yyyymm: str, parquet_dir: Path) -> dict[str, Any]:
    """
    Stream all batch_*.parquet files in parquet_dir and compute metrics.

    Streaming guarantees:
    - Uses iter_batches(columns=COLUMNS, batch_size=BATCH_SIZE); no read_table().
    - prev_key (tuple of 3 strings) carries across file boundaries within the month.
    - acct_day_counts accumulates (account, date) → row_count across all batches;
      bounded by unique acct-days in the month, discarded after stats are computed.

    Vectorized inner loop:
    - Ordering/duplicate detection uses numpy sliced-array comparisons over the
      full batch at once; only the single cross-batch boundary uses Python.
    - Acct-day grouping uses pyarrow group_by (C++), iterating only over the
      unique (account, date) pairs per batch (~acct-days/batch), not every row.
    """
    files = sorted(parquet_dir.glob("batch_*.parquet"))

    n_files = len(files)
    total_rows = 0
    has_duplicates = False
    duplicate_count = 0
    order_breaks = 0
    dir_size_bytes = 0
    errors: list[str] = []

    # Running dict: (account_identifier, date_str) -> row count.
    # Bounded: one month has a finite number of (account, date) pairs.
    acct_day_counts: dict[tuple[str, str], int] = {}

    # Carries the composite key of the last row across batch/file boundaries.
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
                # Only one Python-level comparison per batch crossing — acceptable.
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

                # ── within-batch ordering/duplicate detection (vectorized) ──
                if n_rows > 1:
                    # numpy object arrays — string < / == / > works lexicographically
                    zip_np = np.asarray(zip_s.to_pylist(), dtype=object)
                    acct_np = np.asarray(acct_s.to_pylist(), dtype=object)
                    dt_np = np.asarray(dt_s.to_pylist(), dtype=object)

                    # Sliced views: curr[j] = row j+1, prev[j] = row j
                    z_c, z_p = zip_np[1:], zip_np[:-1]
                    a_c, a_p = acct_np[1:], acct_np[:-1]
                    d_c, d_p = dt_np[1:], dt_np[:-1]

                    z_eq = z_c == z_p
                    a_eq = a_c == a_p

                    # Duplicate: all three columns equal on adjacent rows
                    dup_mask = z_eq & a_eq & (d_c == d_p)
                    n_dups = int(np.sum(dup_mask))
                    if n_dups:
                        has_duplicates = True
                        duplicate_count += n_dups

                    # Order break: composite (zip, acct, datetime) tuple less-than
                    # i.e. current row sorts strictly before previous row
                    composite_less = (z_c < z_p) | (z_eq & (a_c < a_p)) | (z_eq & a_eq & (d_c < d_p))
                    order_breaks += int(np.sum(composite_less))

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
                # Persist last-row key for boundary check at start of next batch/file
                prev_key = (zip_s[-1].as_py(), acct_s[-1].as_py(), dt_s[-1].as_py())

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
        "has_duplicates": has_duplicates,
        "duplicate_count": duplicate_count,
        "order_breaks": order_breaks,
        "rows_per_acct_day_min": rpd_min,
        "rows_per_acct_day_max": rpd_max,
        "rows_per_acct_day_mean": round(rpd_mean, 4),
        "dir_size_bytes": dir_size_bytes,
        "errors": errors,
    }


# ── output writers ─────────────────────────────────────────────────────────

# Fields written to TSV (errors list goes to JSON only; n_errors added here)
TSV_FIELDS = [
    "yyyymm",
    "n_files",
    "total_rows",
    "has_duplicates",
    "duplicate_count",
    "order_breaks",
    "rows_per_acct_day_min",
    "rows_per_acct_day_max",
    "rows_per_acct_day_mean",
    "dir_size_bytes",
    "n_errors",
]


def write_tsv(results: list[dict[str, Any]]) -> None:
    with OUTPUT_TSV.open("w") as fh:
        fh.write("\t".join(TSV_FIELDS) + "\n")
        for r in results:
            row_vals = {**r, "n_errors": len(r["errors"])}
            fh.write("\t".join(str(row_vals[field]) for field in TSV_FIELDS) + "\n")


def write_json(results: list[dict[str, Any]]) -> None:
    with OUTPUT_JSON.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)


def write_summary(results: list[dict[str, Any]]) -> None:
    """
    Markdown summary highlighting data quality findings.

    Expected rows/acct/day for 30-min smart meter data:
      - 48  on a normal day
      - 46  on DST spring-forward day (March, 2nd Sunday)
      - 50  on DST fall-back day (November, 1st Sunday)
    Months with mean outside [46.5, 49.5] are flagged as unusual.
    DST months (03, 11) get a slightly wider window [45.5, 50.5].
    """
    total_months = len(results)
    dup_months = [r for r in results if r["has_duplicates"]]
    ob_months = [r for r in results if r["order_breaks"] > 0]
    error_months = [r for r in results if r["errors"]]

    all_row_counts = [r["total_rows"] for r in results] if results else [0]
    min_rows = min(all_row_counts)
    max_rows = max(all_row_counts)
    min_month = next(r["yyyymm"] for r in results if r["total_rows"] == min_rows)
    max_month = next(r["yyyymm"] for r in results if r["total_rows"] == max_rows)

    # Months with unusual rows/acct/day mean
    unusual_rpd = []
    for r in results:
        mean = r["rows_per_acct_day_mean"]
        mm = r["yyyymm"][4:]  # '03', '11', etc.
        lo, hi = (45.5, 50.5) if mm in ("03", "11") else (46.5, 49.5)
        if mean < lo or mean > hi:
            unusual_rpd.append(
                f"{r['yyyymm']}  mean={mean:.2f}  min={r['rows_per_acct_day_min']}  max={r['rows_per_acct_day_max']}"
            )

    # Flag 202507 if it looks incomplete vs. median
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
        "# Phase 1 Streaming Audit Summary",
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

    lines += ["", "## Rows per account-day"]
    lines.append(
        "Expected: 48/day (30-min intervals); 46 on DST spring-forward (March); 50 on DST fall-back (November)."
    )
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
        "| Month | Files | Total rows | Duplicates | Order breaks | rpd_min | rpd_max | rpd_mean | Size (GB) |",
        "| ----- | ----: | ---------: | ---------: | ----------: | ------: | ------: | -------: | --------: |",
    ]
    for r in results:
        size_gb = r["dir_size_bytes"] / 1e9
        lines.append(
            f"| {r['yyyymm']} "
            f"| {r['n_files']} "
            f"| {r['total_rows']:,} "
            f"| {'✗ ' + str(r['duplicate_count']) if r['has_duplicates'] else '✓'} "
            f"| {r['order_breaks']:,} "
            f"| {r['rows_per_acct_day_min']} "
            f"| {r['rows_per_acct_day_max']} "
            f"| {r['rows_per_acct_day_mean']:.2f} "
            f"| {size_gb:.2f} |"
        )

    lines.append("")
    OUTPUT_MD.write_text("\n".join(lines))


# ── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    months = discover_months()
    total = len(months)
    print(f"Discovered {total} production month directories under {RUNS_DIR}", file=sys.stderr)

    results: list[dict[str, Any]] = []
    overall_start = time.time()

    for i, (yyyymm, parquet_dir) in enumerate(months, 1):
        print(f"Auditing month {i}/{total}: {yyyymm} ...", file=sys.stderr)
        t0 = time.time()

        if not parquet_dir.exists():
            print(f"  WARNING: parquet dir not found: {parquet_dir}", file=sys.stderr)
            results.append({
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
            })
            continue

        result = audit_month(yyyymm, parquet_dir)
        elapsed = time.time() - t0

        print(
            f"  Done: {result['total_rows']:,} rows, "
            f"{result['n_files']} files, "
            f"{result['dir_size_bytes'] / 1e9:.2f} GB, "
            f"{elapsed:.1f}s",
            file=sys.stderr,
        )
        results.append(result)

    write_tsv(results)
    write_json(results)
    write_summary(results)

    total_elapsed = time.time() - overall_start
    print(
        f"\nAudit complete: {total} months in {total_elapsed:.1f}s",
        file=sys.stderr,
    )
    print(f"  {OUTPUT_TSV}", file=sys.stderr)
    print(f"  {OUTPUT_JSON}", file=sys.stderr)
    print(f"  {OUTPUT_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
