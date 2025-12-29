# smart_meter_analysis/aws_loader.py
"""AWS S3 utilities for deterministic batch download of ComEd smart meter CSV files.

Contract A (S3 Download only)
----------------------------
This module is responsible for:
- Deterministically listing S3 CSV keys for a given year-month (YYYYMM)
- Downloading those keys to local disk with retry/backoff
- Writing a JSONL download manifest for provenance and QA

Downstream processing belongs to scripts/process_csvs_batched_optimized.py and
smart_meter_analysis/transformation.py.

Operational scaling note
------------------------
For full-scale runs (e.g., ~700k files), downloads must be resumable.
Accordingly, this module supports:
- Appending to an existing manifest (default)
- Skipping downloads for files that already exist on disk (default)
- Optional manifest overwrite for clean-room re-runs
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

S3_BUCKET = "smart-meter-data-sb"
S3_PREFIX = "sharepoint-files/Zip4/"

ERR_BAD_YEAR_MONTH = "year_month must be YYYYMM (e.g., 202308); got: {}"
ERR_NO_FILES_FOUND = "No files found for month: {}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _validate_year_month(year_month: str) -> None:
    if not re.fullmatch(r"\d{6}", year_month):
        raise ValueError(ERR_BAD_YEAR_MONTH.format(year_month))


def list_s3_files(
    year_month: str,
    *,
    bucket: str = S3_BUCKET,
    prefix: str = S3_PREFIX,
    max_files: int | None = None,
) -> list[str]:
    """List CSV files in S3 for a given year-month.

    Returns:
        Deterministically sorted S3 *keys* (e.g., "sharepoint-files/Zip4/202307/file.csv"),
        NOT URIs. These keys are intended to be passed directly to download_s3_batch().

    Determinism:
        - Collect ALL keys for the prefix
        - Sort the full list
        - Then apply max_files slicing
    """
    _validate_year_month(year_month)

    s3 = boto3.client("s3")
    full_prefix = f"{prefix}{year_month}/"

    logger.info("Listing files from s3://%s/%s", bucket, full_prefix)

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=full_prefix)

    keys: list[str] = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key and key.endswith(".csv"):
                keys.append(key)

    keys = sorted(keys)

    if max_files is not None:
        keys = keys[: int(max_files)]

    if not keys:
        raise ValueError(ERR_NO_FILES_FOUND.format(year_month))

    logger.info("Found %d CSV files (after limit)", len(keys))
    return keys


def _write_manifest_line(fp: Any, record: dict[str, Any]) -> None:
    fp.write(json.dumps(record, sort_keys=True) + "\n")
    fp.flush()


def _validate_manifest(manifest_path: Path) -> bool:
    """Best-effort validation that the manifest is valid JSONL and contains required fields.
    Non-fatal: returns False on any issue.
    """
    required_fields = {"s3_key", "status", "timestamp"}
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not required_fields.issubset(rec.keys()):
                    raise ValueError(f"Line {i} missing required fields: {required_fields - set(rec.keys())}")
        return True
    except Exception as exc:
        logger.error("Invalid manifest JSONL at %s: %s", manifest_path, exc)
        return False


def _local_path_for_key(*, key: str, output_dir: Path, prefix: str) -> Path:
    """Map an S3 key to a deterministic local path under output_dir.

    We preserve the key's structure beneath the configured prefix to avoid filename collisions.
    Example:
        key:    sharepoint-files/Zip4/202307/foo.csv
        prefix: sharepoint-files/Zip4/
        local:  <output_dir>/202307/foo.csv
    """
    key_path = Path(key)
    prefix_path = Path(prefix.rstrip("/"))

    try:
        rel = key_path.relative_to(prefix_path)
    except ValueError:
        # Fallback if key does not start with prefix as a Path.
        rel = key_path

    return output_dir / rel


def _is_existing_nonempty(path: Path) -> tuple[bool, int | None]:
    """Return (exists_and_nonempty, size_bytes_if_known)."""
    try:
        if path.exists():
            size = path.stat().st_size
            return (size > 0, size)
        return (False, None)
    except OSError:
        return (False, None)


def download_s3_batch(
    *,
    s3_keys: list[str],
    output_dir: Path,
    manifest_path: Path,
    bucket: str = S3_BUCKET,
    prefix: str = S3_PREFIX,
    max_files: int | None = None,
    fail_fast: bool = True,
    max_errors: int = 10,
    retries: int = 3,
    backoff_factor: float = 2.0,
    log_every: int = 100,
    overwrite_manifest: bool = False,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Download S3 keys to local directory with manifest tracking.

    Args:
        s3_keys: S3 keys (NOT URIs). Example: "sharepoint-files/Zip4/202307/file.csv"
        output_dir: Local directory to write downloaded CSVs
        manifest_path: JSONL manifest path (append by default; optionally overwritten)
        bucket: S3 bucket
        prefix: S3 prefix used to compute deterministic local paths
        max_files: Optional cap (applied after s3_keys is already deterministic/sorted upstream)
        fail_fast: Stop immediately on first error
        max_errors: Allowed errors before aborting when fail_fast=False
        retries: Number of retry attempts per file (in addition to the initial attempt)
        backoff_factor: Exponential backoff multiplier (seconds: 1, 2, 4, ...)
        log_every: Progress logging interval
        overwrite_manifest: If True, truncate and rewrite manifest (clean-room rerun)
        skip_existing: If True, skip download when local_path exists and is non-empty

    Returns:
        dict with keys: downloaded, failed, skipped, manifest_path
    """
    if max_files is not None:
        s3_keys = s3_keys[: int(max_files)]

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")

    downloaded = 0
    failed = 0
    skipped = 0

    mode = "w" if overwrite_manifest else "a"
    if overwrite_manifest:
        logger.info("Overwriting manifest (clean run): %s", manifest_path)
    else:
        if manifest_path.exists():
            logger.info("Appending to existing manifest (resume mode): %s", manifest_path)
        else:
            logger.info("Creating new manifest: %s", manifest_path)

    with manifest_path.open(mode, encoding="utf-8") as mf:
        for i, key in enumerate(s3_keys, 1):
            if log_every > 0 and (i == 1 or i % log_every == 0 or i == len(s3_keys)):
                logger.info("Downloading %d/%d", i, len(s3_keys))

            local_path = _local_path_for_key(key=key, output_dir=output_dir, prefix=prefix)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            s3_uri = f"s3://{bucket}/{key}"
            ts = _utc_now_iso()

            if skip_existing:
                exists_nonempty, size_bytes = _is_existing_nonempty(local_path)
                if exists_nonempty:
                    _write_manifest_line(
                        mf,
                        {
                            "s3_key": key,
                            "s3_uri": s3_uri,
                            "local_path": str(local_path),
                            "status": "skipped_exists",
                            "size_bytes": size_bytes,
                            "timestamp": ts,
                            "attempt": 0,
                        },
                    )
                    skipped += 1
                    continue

            attempt = 0
            last_exc: Exception | None = None
            while attempt <= retries:
                try:
                    attempt += 1
                    s3.download_file(bucket, key, str(local_path))

                    size_bytes = None
                    try:
                        size_bytes = local_path.stat().st_size
                    except OSError:
                        size_bytes = None

                    _write_manifest_line(
                        mf,
                        {
                            "s3_key": key,
                            "s3_uri": s3_uri,
                            "local_path": str(local_path),
                            "status": "success",
                            "size_bytes": size_bytes,
                            "timestamp": ts,
                            "attempt": attempt,
                        },
                    )
                    downloaded += 1
                    last_exc = None
                    break

                except (ClientError, BotoCoreError, OSError) as exc:
                    last_exc = exc
                    if attempt > retries:
                        break

                    sleep_s = float(backoff_factor) ** float(attempt - 1)
                    logger.warning(
                        "Download failed (attempt %d/%d) for %s: %s; backing off %.1fs",
                        attempt,
                        retries + 1,
                        key,
                        type(exc).__name__,
                        sleep_s,
                    )
                    time.sleep(sleep_s)

            if last_exc is not None:
                failed += 1
                _write_manifest_line(
                    mf,
                    {
                        "s3_key": key,
                        "s3_uri": s3_uri,
                        "local_path": str(local_path),
                        "status": "error",
                        "error": f"{type(last_exc).__name__}: {last_exc}",
                        "timestamp": ts,
                        "attempt": attempt,
                    },
                )

                msg = f"Failed to download {key} after {attempt} attempt(s): {type(last_exc).__name__}: {last_exc}"
                if fail_fast:
                    raise RuntimeError(msg) from last_exc

                if failed > max_errors:
                    raise RuntimeError(
                        f"Exceeded max_errors={max_errors} during S3 download. "
                        f"Downloaded={downloaded} Skipped={skipped} Failed={failed}. Last error: {msg}",
                    ) from last_exc

    logger.info(
        "Download complete. Success=%d Skipped=%d Failed=%d Manifest=%s",
        downloaded,
        skipped,
        failed,
        manifest_path,
    )

    if not _validate_manifest(manifest_path):
        logger.warning("Manifest validation failed (non-fatal): %s", manifest_path)

    return {"downloaded": downloaded, "skipped": skipped, "failed": failed, "manifest_path": str(manifest_path)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Deterministically list and download ComEd CSVs from S3.")
    parser.add_argument("year_month", help="Month in YYYYMM format (e.g., 202307)")
    parser.add_argument("--bucket", default=S3_BUCKET)
    parser.add_argument("--prefix", default=S3_PREFIX)
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files (testing)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/runs/manual/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/runs/manual/download_manifest.jsonl"))
    parser.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Overwrite (truncate) the manifest instead of appending (clean-room rerun).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip downloads for existing non-empty local files.",
    )
    parser.add_argument("--no-fail-fast", action="store_true", help="Allow errors up to --max-errors")
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff-factor", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=100)

    args = parser.parse_args()
    _validate_year_month(args.year_month)

    keys = list_s3_files(
        args.year_month,
        bucket=args.bucket,
        prefix=args.prefix,
        max_files=args.max_files,
    )

    download_s3_batch(
        s3_keys=keys,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        bucket=args.bucket,
        prefix=args.prefix,
        max_files=args.max_files,
        fail_fast=not args.no_fail_fast,
        max_errors=args.max_errors,
        retries=args.retries,
        backoff_factor=args.backoff_factor,
        log_every=args.log_every,
        overwrite_manifest=bool(args.overwrite_manifest),
        skip_existing=not bool(args.no_skip_existing),
    )


if __name__ == "__main__":
    main()
