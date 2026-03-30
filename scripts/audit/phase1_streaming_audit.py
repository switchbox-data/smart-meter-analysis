#!/usr/bin/env python3
"""
Phase 1 streaming audit of ComEd CSV→Parquet production outputs.

Scans every month directory under /ebs/home/griffin_switch_box/runs/ matching
out_*_production and audits the parquet files inside each for correctness.
All metrics are computed via streaming (constant memory): pyarrow iter_batches only,
no read_table() or full-file collects.

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


def _dt_to_str(dt_val: Any) -> str:
    """Convert a datetime column value to a consistent ISO string."""
    if dt_val is None:
        return ""
    if isinstance(dt_val, str):
        return dt_val
    if hasattr(dt_val, "isoformat"):
        return dt_val.isoformat()
    return str(dt_val)


def audit_month(yyyymm: str, parquet_dir: Path) -> dict[str, Any]:
    """
    Stream all batch_*.parquet files in parquet_dir and compute metrics.

    Streaming guarantees:
    - Uses iter_batches(columns=COLUMNS, batch_size=BATCH_SIZE).
    - prev_key carries across file boundaries within the month.
    - acct_day_counts is a running dict bounded by unique (account, date) pairs
      in the month; discarded after stats are computed.
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

    # prev_key carries across file boundaries within this month
    prev_key: tuple[str, str, str] | None = None

    for f in files:
        try:
            dir_size_bytes += f.stat().st_size
            pf = pq.ParquetFile(f)

            for batch in pf.iter_batches(columns=COLUMNS, batch_size=BATCH_SIZE):
                n_rows = batch.num_rows
                if n_rows == 0:
                    continue

                zip_list = batch.column("zip_code").to_pylist()
                acct_list = batch.column("account_identifier").to_pylist()
                dt_list = batch.column("datetime").to_pylist()

                # Convert datetime values to strings once per batch
                dt_str_list = [_dt_to_str(dt) for dt in dt_list]

                # Build composite sort keys for the entire batch
                keys: list[tuple[str, str, str]] = [
                    (
                        str(zip_list[i]) if zip_list[i] is not None else "",
                        str(acct_list[i]) if acct_list[i] is not None else "",
                        dt_str_list[i],
                    )
                    for i in range(n_rows)
                ]

                # ── boundary check: last row of previous batch vs. first row here ──
                if prev_key is not None:
                    first = keys[0]
                    if first == prev_key:
                        has_duplicates = True
                        duplicate_count += 1
                    elif first < prev_key:
                        order_breaks += 1

                # ── within-batch ordering/duplicate check ──
                for i in range(1, n_rows):
                    if keys[i] == keys[i - 1]:
                        has_duplicates = True
                        duplicate_count += 1
                    elif keys[i] < keys[i - 1]:
                        order_breaks += 1

                # ── acct-day accumulation ──
                for i in range(n_rows):
                    date_str = dt_str_list[i][:10]  # first 10 chars = YYYY-MM-DD
                    acct = str(acct_list[i]) if acct_list[i] is not None else ""
                    k = (acct, date_str)
                    acct_day_counts[k] = acct_day_counts.get(k, 0) + 1

                total_rows += n_rows
                prev_key = keys[-1]

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
