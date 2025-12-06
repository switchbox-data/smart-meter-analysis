#!/usr/bin/env python3
"""
Download real sample CSV files from S3 for local testing.

This pulls a small number of actual ComEd files from S3 and saves them locally,
so you can test the transformation pipeline without processing the full dataset.

Usage:
    # Download 5 files from August 2023
    python scripts/testing/download_samples_from_s3.py --year-month 202308 --num-files 5

    # Download specific files
    python scripts/testing/download_samples_from_s3.py --year-month 202308 --num-files 10

    # Download to custom location
    python scripts/testing/download_samples_from_s3.py --year-month 202308 --num-files 5 --output-dir data/test-samples
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# S3 Configuration
DEFAULT_BUCKET = "smart-meter-data-sb"
DEFAULT_PREFIX = "sharepoint-files/Zip4/"
DEFAULT_OUTPUT_DIR = Path("data/samples")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def list_s3_files(
    year_month: str,
    bucket: str,
    prefix: str,
    max_files: int,
) -> list[str]:
    """
    List CSV files in S3 for a given year-month.

    Args:
        year_month: Format 'YYYYMM' (e.g., '202308')
        bucket: S3 bucket name
        prefix: S3 prefix/folder
        max_files: Maximum number of files to return

    Returns:
        List of S3 keys (file paths)
    """
    s3 = boto3.client("s3")
    full_prefix = f"{prefix}{year_month}/"

    logger.info(f"Listing files from s3://{bucket}/{full_prefix}")

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=full_prefix)

    keys = []
    for page in pages:
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith(".csv"):
                keys.append(key)
                if len(keys) >= max_files:
                    logger.info(f"Reached limit of {max_files} files")
                    return keys

    logger.info(f"Found {len(keys)} CSV files")
    return keys


def download_file_from_s3(
    s3_key: str,
    bucket: str,
    output_dir: Path,
) -> Path | None:
    """
    Download a single file from S3 to local directory.

    Args:
        s3_key: S3 key (file path in bucket)
        bucket: S3 bucket name
        output_dir: Local directory to save file

    Returns:
        Path to downloaded file, or None if failed
    """
    s3 = boto3.client("s3")

    # Extract filename from S3 key
    filename = Path(s3_key).name
    local_path = output_dir / filename

    try:
        # Get file size
        response = s3.head_object(Bucket=bucket, Key=s3_key)
        file_size = response["ContentLength"]
        size_mb = file_size / (1024 * 1024)

        logger.info(f"Downloading: {filename} ({size_mb:.2f} MB)")

        # Download file
        s3.download_file(bucket, s3_key, str(local_path))

        logger.info(f"  ✓ Saved to: {local_path}")
        return local_path

    except ClientError as e:
        logger.error(f"  ✗ Failed to download {filename}: {e}")
        return None
    except Exception as e:
        logger.error(f"  ✗ Unexpected error downloading {filename}: {e}")
        return None


def download_sample_files(
    year_month: str,
    num_files: int,
    output_dir: Path,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
) -> list[Path]:
    """
    Download sample CSV files from S3 to local directory.

    Args:
        year_month: Format 'YYYYMM' (e.g., '202308')
        num_files: Number of files to download
        output_dir: Local directory to save files
        bucket: S3 bucket name
        prefix: S3 prefix/folder

    Returns:
        List of paths to downloaded files
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # List files in S3
    s3_keys = list_s3_files(year_month, bucket, prefix, num_files)

    if not s3_keys:
        logger.error(f"No files found in s3://{bucket}/{prefix}{year_month}/")
        return []

    logger.info(f"\nDownloading {len(s3_keys)} files to {output_dir}")
    logger.info("=" * 80)

    # Download each file
    downloaded_files = []
    for i, s3_key in enumerate(s3_keys, 1):
        logger.info(f"\nFile {i}/{len(s3_keys)}")
        local_path = download_file_from_s3(s3_key, bucket, output_dir)
        if local_path:
            downloaded_files.append(local_path)

    return downloaded_files


def main():
    parser = argparse.ArgumentParser(
        description="Download sample CSV files from S3 for local testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 5 files from August 2023
  python scripts/testing/download_samples_from_s3.py --year-month 202308 --num-files 5

  # Download 10 files to custom location
  python scripts/testing/download_samples_from_s3.py --year-month 202308 --num-files 10 --output-dir data/test-samples

  # Use custom S3 bucket
  python scripts/testing/download_samples_from_s3.py --year-month 202308 --num-files 5 --bucket my-bucket
        """,
    )

    parser.add_argument(
        "--year-month",
        required=True,
        help="Year-month to download from (format: YYYYMM, e.g., 202308)",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=5,
        help="Number of files to download (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"S3 bucket name (default: {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"S3 prefix/folder (default: {DEFAULT_PREFIX})",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Download Sample Files from S3")
    logger.info("=" * 80)
    logger.info(f"Year-Month: {args.year_month}")
    logger.info(f"Number of Files: {args.num_files}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"S3 Bucket: {args.bucket}")
    logger.info(f"S3 Prefix: {args.prefix}")
    logger.info("")

    try:
        downloaded = download_sample_files(
            year_month=args.year_month,
            num_files=args.num_files,
            output_dir=args.output_dir,
            bucket=args.bucket,
            prefix=args.prefix,
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"✓ Successfully downloaded {len(downloaded)} files")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Inspect a sample file:")
        logger.info(f"     head -n 3 {downloaded[0] if downloaded else args.output_dir / '*.csv'}")
        logger.info("")
        logger.info("  2. Run pipeline on samples:")
        logger.info("     python run_pipeline.py --source local")
        logger.info("")
        logger.info("  3. Or use just:")
        logger.info("     just test-pipeline-local")
        logger.info("")
        logger.info("  4. Inspect output:")
        logger.info("     just inspect-data data/processed/comed_samples.parquet")

        # Calculate total size
        if downloaded:
            total_size = sum(f.stat().st_size for f in downloaded)
            total_mb = total_size / (1024 * 1024)
            logger.info("")
            logger.info(f"Total size: {total_mb:.2f} MB")

            # Suggest gitignore
            if total_mb > 10:
                logger.info("")
                logger.info("⚠️  Note: Files are > 10 MB total")
                logger.info("Consider adding to .gitignore:")
                logger.info(f"  echo '{args.output_dir}/*.csv' >> .gitignore")

        return 0

    except Exception:
        logger.exception("Failed to download samples")
        logger.info("")
        logger.info("Troubleshooting:")
        logger.info("  - Check AWS credentials: aws s3 ls")
        logger.info("  - Verify bucket access: aws s3 ls s3://smart-meter-data-sb/")
        logger.info(f"  - Check year-month exists: aws s3 ls s3://{args.bucket}/{args.prefix}{args.year_month}/")
        return 1


if __name__ == "__main__":
    exit(main())
