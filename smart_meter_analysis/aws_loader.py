# smart_meter_analysis/aws_loader.py
"""AWS S3 utilities for deterministic batch download of ComEd smart meter CSV files.

Contract A (S3 Download only)
----------------------------
This module is responsible  for:
- Deterministically listing S3 CSV keys for a given year-month (YYYYMM)
- Downloading those keys to local disk with retry/backoff
- Writing a JSONL download manifest for provenance and QA

Downstream processing belongs to scripts/process_csvs_batched_optimized.py and
smart_meter_analysis/transformation.py.
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


def download_s3_batch(
    *,
    s3_keys: list[str],
    output_dir: Path,
    manifest_path: Path,
    bucket: str = S3_BUCKET,
    max_files: int | None = None,
    fail_fast: bool = True,
    max_errors: int = 10,
    retries: int = 3,
    backoff_factor: float = 2.0,
    log_every: int = 100,
) -> dict[str, Any]:
    """Download S3 keys to local directory with manifest tracking.

    Args:
        s3_keys: S3 keys (NOT URIs). Example: "sharepoint-files/Zip4/202307/file.csv"
        output_dir: Local directory to write downloaded CSVs
        manifest_path: JSONL manifest path (overwritten for determinism)
        bucket: S3 bucket
        max_files: Optional cap (applied after s3_keys is already deterministic/sorted upstream)
        fail_fast: Stop immediately on first error
        max_errors: Allowed errors before aborting when fail_fast=False
        retries: Number of retry attempts per file (in addition to the initial attempt)
        backoff_factor: Exponential backoff multiplier (seconds: 1, 2, 4, ...)
        log_every: Progress logging interval

    Returns:
        dict with keys: downloaded, failed, manifest_path

    """
    if max_files is not None:
        s3_keys = s3_keys[: int(max_files)]

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")

    downloaded = 0
    failed = 0

    # Overwrite for determinism: reruns produce byte-identical manifests given identical outcomes.
    with manifest_path.open("w", encoding="utf-8") as mf:
        for i, key in enumerate(s3_keys, 1):
            if log_every > 0 and (i == 1 or i % log_every == 0 or i == len(s3_keys)):
                logger.info("Downloading %d/%d", i, len(s3_keys))

            local_path = output_dir / Path(key).name
            s3_uri = f"s3://{bucket}/{key}"
            ts = _utc_now_iso()

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
                        f"Downloaded={downloaded} Failed={failed}. Last error: {msg}",
                    ) from last_exc

    logger.info("Download complete. Success=%d Failed=%d Manifest=%s", downloaded, failed, manifest_path)

    if not _validate_manifest(manifest_path):
        logger.warning("Manifest validation failed (non-fatal): %s", manifest_path)

    return {"downloaded": downloaded, "failed": failed, "manifest_path": str(manifest_path)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Deterministically list and download ComEd CSVs from S3.")
    parser.add_argument("year_month", help="Month in YYYYMM format (e.g., 202307)")
    parser.add_argument("--bucket", default=S3_BUCKET)
    parser.add_argument("--prefix", default=S3_PREFIX)
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files (testing)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/runs/manual/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/runs/manual/download_manifest.jsonl"))
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
        max_files=args.max_files,
        fail_fast=not args.no_fail_fast,
        max_errors=args.max_errors,
        retries=args.retries,
        backoff_factor=args.backoff_factor,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
