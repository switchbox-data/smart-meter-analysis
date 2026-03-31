#!/usr/bin/env python3
"""
Phase 4: publish clean production parquet outputs to S3 with a versioned,
immutable release prefix.

Copies all parquet files from /ebs/home/griffin_switch_box/runs/out_*_production/
to s3://smart-meter-data-sb/parquet_release/{release_tag}/ where release_tag is
YYYYMMDD_HHMM_{git_short_sha} (e.g., 20260331_2100_b5c2915).

After all copies succeed, writes a CURRENT_RELEASE.txt pointer to
s3://smart-meter-data-sb/parquet_release/CURRENT_RELEASE.txt.

Usage:

  # Preview what will be published (default — safe to run any time)
  python scripts/audit/phase4_s3_publish.py --dry-run

  # Actually publish
  python scripts/audit/phase4_s3_publish.py --execute
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

# ── paths & constants ─────────────────────────────────────────────────────
RUNS_DIR = Path("/ebs/home/griffin_switch_box/runs")
GIT_REPO = Path("/ebs/home/griffin_switch_box/smart-meter-analysis")
S3_BUCKET = "smart-meter-data-sb"
S3_RELEASE_PREFIX = "parquet_release"

MANIFEST_PATH = Path("/tmp/s3_release_manifest.json")
COPY_LOG_PATH = Path("/tmp/s3_copy_log.txt")
SUMMARY_PATH = Path("/tmp/s3_release_summary.md")

CHUNK_SIZE = 65_536  # 64 KB for streaming sha256


# ── release tag ───────────────────────────────────────────────────────────


def get_git_short_sha() -> str:
    """Return the short SHA of HEAD in the smart-meter-analysis repo."""
    result = subprocess.run(
        ["git", "-C", str(GIT_REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def make_release_tag() -> str:
    """Generate YYYYMMDD_HHMM_{git_short_sha}."""
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M")
    sha = get_git_short_sha()
    return f"{ts}_{sha}"


# ── discovery ─────────────────────────────────────────────────────────────


def discover_months() -> list[tuple[str, Path]]:
    """
    Return sorted (yyyymm, parquet_dir) pairs.

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


def list_month_files(parquet_dir: Path) -> list[Path]:
    """
    Return all parquet files and _SUCCESS.json (if present) for a month.

    Handles both naming conventions:
      - batch_*.parquet  (47 months)
      - part-*.parquet   (202301, 202307)
    """
    files: list[Path] = sorted(parquet_dir.glob("*.parquet"))
    success = parquet_dir / "_SUCCESS.json"
    if success.exists():
        files.append(success)
    return files


# ── checksums & metadata ──────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    """Stream a file in 64 KB chunks and return its hex sha256."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def parquet_row_count(path: Path) -> int:
    """Read row count from parquet footer metadata (no data scan)."""
    if path.suffix != ".parquet":
        return 0
    try:
        meta = pq.read_metadata(path)
        return meta.num_rows
    except Exception:
        return -1


# ── S3 helpers ────────────────────────────────────────────────────────────


def s3_prefix_exists(prefix: str) -> bool:
    """Return True if any objects exist under the given S3 prefix."""
    result = subprocess.run(
        ["aws", "s3", "ls", f"s3://{S3_BUCKET}/{prefix}/"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def s3_cp(source: Path, s3_key: str) -> tuple[bool, str]:
    """
    Copy a single local file to S3 via `aws s3 cp`.

    Returns (success, message).
    """
    dest = f"s3://{S3_BUCKET}/{s3_key}"
    result = subprocess.run(
        ["aws", "s3", "cp", str(source), dest],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, f"OK: {dest}"
    return False, f"FAILED: {dest}\n  stderr: {result.stderr.strip()}"


def s3_put_text(content: str, s3_key: str) -> tuple[bool, str]:
    """Write a small text string to S3 via stdin pipe."""
    dest = f"s3://{S3_BUCKET}/{s3_key}"
    result = subprocess.run(
        ["aws", "s3", "cp", "-", dest],
        input=content,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, f"OK: {dest}"
    return False, f"FAILED: {dest}\n  stderr: {result.stderr.strip()}"


# ── main publish logic ────────────────────────────────────────────────────


def run_publish(dry_run: bool) -> int:
    """
    Discover, validate, copy, and report.

    Returns exit code (0 = success, 1 = errors).
    """
    # 1. Release tag
    try:
        release_tag = make_release_tag()
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: could not determine git SHA: {exc}", file=sys.stderr)
        return 1

    release_prefix = f"{S3_RELEASE_PREFIX}/{release_tag}"
    print(f"Release tag : {release_tag}", file=sys.stderr)
    print(f"S3 prefix   : s3://{S3_BUCKET}/{release_prefix}/", file=sys.stderr)
    print(f"Dry-run     : {dry_run}", file=sys.stderr)
    print("", file=sys.stderr)

    # 2. Guard: abort if prefix already exists on S3
    if not dry_run and s3_prefix_exists(release_prefix):
        print(
            f"ERROR: S3 prefix already exists: s3://{S3_BUCKET}/{release_prefix}/\n"
            "Refusing to overwrite an existing release. Use a new release tag.",
            file=sys.stderr,
        )
        return 1

    # 3. Discover months
    months = discover_months()
    if not months:
        print(f"ERROR: no out_*_production dirs found under {RUNS_DIR}", file=sys.stderr)
        return 1
    print(f"Discovered {len(months)} production month(s).", file=sys.stderr)

    # 4. Build file list and compute metadata
    manifest: list[dict[str, Any]] = []
    total_size_bytes = 0
    total_rows = 0
    all_files: list[tuple[str, str, Path, str]] = []  # (yyyymm, s3_key, local_path, filename)

    for yyyymm, parquet_dir in months:
        yyyy = yyyymm[:4]
        mm = yyyymm[4:]
        if not parquet_dir.exists():
            print(f"  WARNING: parquet dir missing, skipping {yyyymm}: {parquet_dir}", file=sys.stderr)
            continue
        files = list_month_files(parquet_dir)
        for f in files:
            s3_key = f"{release_prefix}/{yyyy}/{mm}/{f.name}"
            all_files.append((yyyymm, s3_key, f, f.name))

    total_file_count = len(all_files)
    print(f"Total files to publish: {total_file_count}", file=sys.stderr)
    print("", file=sys.stderr)

    # 5. Dry-run: print plan and exit
    if dry_run:
        print("DRY-RUN — the following files would be copied:\n", file=sys.stderr)
        for _yyyymm, s3_key, local_path, _filename in all_files:
            size_mb = local_path.stat().st_size / 1e6
            print(f"  {local_path}  ->  s3://{S3_BUCKET}/{s3_key}  ({size_mb:.1f} MB)")
        print(f"\nTotal: {total_file_count} file(s).")
        print("\nRun with --execute to publish.")
        return 0

    # 6. Execute: copy files, build manifest
    errors: list[str] = []
    copy_log_lines: list[str] = []

    # Organise files by month for progress reporting
    month_files: dict[str, list[tuple[str, Path, str]]] = {}
    for yyyymm, s3_key, local_path, filename in all_files:
        month_files.setdefault(yyyymm, []).append((s3_key, local_path, filename))

    month_list = [(ym, month_files[ym]) for ym, _ in months if ym in month_files]
    n_months = len(month_list)

    for month_idx, (yyyymm, file_tuples) in enumerate(month_list, 1):
        yyyy = yyyymm[:4]
        mm = yyyymm[4:]
        n_files = len(file_tuples)

        for file_idx, (s3_key, local_path, filename) in enumerate(file_tuples, 1):
            print(
                f"Copying month {month_idx}/{n_months}: {yyyymm} (file {file_idx}/{n_files}: {filename}) ...",
                end="",
                file=sys.stderr,
            )
            t0 = time.time()

            size_bytes = local_path.stat().st_size
            sha256 = sha256_file(local_path)
            row_count = parquet_row_count(local_path)

            ok, msg = s3_cp(local_path, s3_key)
            elapsed = time.time() - t0

            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            status = "OK" if ok else "FAILED"
            log_line = f"{ts}\t{status}\t{local_path}\t{s3_key}\t{size_bytes}\t{elapsed:.1f}s"
            copy_log_lines.append(log_line)

            if ok:
                total_size_bytes += size_bytes
                total_rows += max(row_count, 0)
                manifest.append({
                    "month": yyyymm,
                    "filename": filename,
                    "source_path": str(local_path),
                    "s3_key": s3_key,
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                    "row_count": row_count,
                })
                print(f" OK ({elapsed:.1f}s)", file=sys.stderr)
            else:
                errors.append(msg)
                print(f" FAILED ({elapsed:.1f}s)", file=sys.stderr)
                print(f"  {msg}", file=sys.stderr)

    # 7. Write CURRENT_RELEASE.txt pointer (only on full success)
    if errors:
        print(
            f"\n{len(errors)} copy error(s) — skipping CURRENT_RELEASE.txt update.",
            file=sys.stderr,
        )
    else:
        pointer_key = f"{S3_RELEASE_PREFIX}/CURRENT_RELEASE.txt"
        ok, msg = s3_put_text(release_tag + "\n", pointer_key)
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        copy_log_lines.append(f"{ts}\t{'OK' if ok else 'FAILED'}\tCURRENT_RELEASE.txt\t{pointer_key}")
        if ok:
            print(f"\nUpdated CURRENT_RELEASE.txt -> {release_tag}", file=sys.stderr)
        else:
            print(f"\nWARNING: failed to write CURRENT_RELEASE.txt: {msg}", file=sys.stderr)
            errors.append(msg)

    # 8. Write local outputs
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    COPY_LOG_PATH.write_text("\n".join(copy_log_lines) + "\n")
    _write_summary(release_tag, total_file_count, total_size_bytes, total_rows, errors)

    print(f"\nManifest : {MANIFEST_PATH}", file=sys.stderr)
    print(f"Copy log : {COPY_LOG_PATH}", file=sys.stderr)
    print(f"Summary  : {SUMMARY_PATH}", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} error(s) encountered. See copy log for details.", file=sys.stderr)
        return 1

    print(
        f"\nRelease complete: {len(manifest)} file(s), {total_size_bytes / 1e9:.2f} GB, {total_rows:,} rows.",
        file=sys.stderr,
    )
    return 0


# ── summary writer ────────────────────────────────────────────────────────


def _write_summary(
    release_tag: str,
    total_files: int,
    total_size_bytes: int,
    total_rows: int,
    errors: list[str],
) -> None:
    lines = [
        "# S3 Release Summary",
        "",
        f"**Release tag:** `{release_tag}`",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"**Destination:** `s3://{S3_BUCKET}/{S3_RELEASE_PREFIX}/{release_tag}/`",
        "",
        "---",
        "",
        "## Totals",
        "",
        "| Metric | Value |",
        "| ------ | ----: |",
        f"| Files copied | {total_files:,} |",
        f"| Total size | {total_size_bytes / 1e9:.2f} GB |",
        f"| Total rows | {total_rows:,} |",
        f"| Errors | {len(errors)} |",
        "",
    ]

    if errors:
        lines += [
            "## Errors",
            "",
        ]
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")
    else:
        lines += ["## Status", "", "All files copied successfully. ✓", ""]

    SUMMARY_PATH.write_text("\n".join(lines))


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: publish production parquet outputs to S3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Preview what will be published (safe — no S3 writes)
  python scripts/audit/phase4_s3_publish.py --dry-run

  # Actually publish
  python scripts/audit/phase4_s3_publish.py --execute
""",
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Print what would be copied without writing to S3 (default)",
    )
    mode.add_argument(
        "--execute",
        dest="dry_run",
        action="store_false",
        help="Actually copy files to S3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.exit(run_publish(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
