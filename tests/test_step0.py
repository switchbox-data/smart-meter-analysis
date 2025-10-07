#!/usr/bin/env python
"""
Test script for Step 0 transformation pipeline.
Tests both local transformation and AWS batch processing.
"""

import logging
from pathlib import Path

from smart_meter_analysis.step0_aws import (
    list_s3_files,
    process_month_batch,
    process_single_csv,
)
from smart_meter_analysis.step0_transform import (
    daily_interval_qc,
    dst_transition_dates,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def test_aws_listing():
    """Test 1: Can we list files from S3?"""
    print("\n" + "=" * 60)
    print("TEST 1: List S3 Files")
    print("=" * 60)

    files = list_s3_files(year_month="202308", max_files=5)
    print(f"✅ Found {len(files)} files")
    for f in files:
        print(f"  - {Path(f).name}")

    return files


def test_single_file_processing(s3_key: str):
    """Test 2: Process a single file from S3"""
    print("\n" + "=" * 60)
    print("TEST 2: Process Single CSV")
    print("=" * 60)
    print(f"Processing: {Path(s3_key).name}")

    df = process_single_csv(s3_key)

    print("\n✅ Transformation successful!")
    print(f"  Rows: {df.height:,}")
    print(f"  Columns: {df.columns}")
    print("\nFirst few rows:")
    print(df.head())

    # Check data quality
    print("\n📊 Data Quality:")
    print(f"  Unique accounts: {df['account_identifier'].n_unique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total kWh: {df['kwh'].sum():,.2f}")
    print(f"  Null kWh values: {df['kwh'].null_count()}")

    return df


def test_qc_checks(df):
    """Test 3: QC checks for interval counts"""
    print("\n" + "=" * 60)
    print("TEST 3: Quality Control Checks")
    print("=" * 60)

    qc = daily_interval_qc(df)
    print("\nInterval counts by day type:")
    print(
        qc.group_by("day_type")
        .agg([
            pl.len().alias("n_days"),
            pl.col("n_intervals").mean().alias("avg_intervals"),
        ])
        .sort("day_type")
    )

    # Check for DST
    dst_dates = dst_transition_dates(df)
    if dst_dates.height > 0:
        print("\n🕐 DST Transition Dates Found:")
        print(dst_dates)
    else:
        print("\n✅ No DST transitions in this data")

    return qc


def test_batch_processing():
    """Test 4: Process multiple files and save"""
    print("\n" + "=" * 60)
    print("TEST 4: Batch Processing (3 files)")
    print("=" * 60)

    output_path = Path("data/processed/test_august_2023.parquet")

    process_month_batch(
        year_month="202308",
        output_path=output_path,
        max_files=3,  # Just 3 files for testing
    )

    print("\n✅ Batch processing complete!")
    print(f"  Output: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Read it back to verify
    import polars as pl

    df = pl.read_parquet(output_path)
    print(f"  Rows in saved file: {df.height:,}")
    print(f"  Unique accounts: {df['account_identifier'].n_unique()}")

    return df


if __name__ == "__main__":
    import polars as pl

    try:
        # Test 1: List files
        files = test_aws_listing()

        # Test 2: Process single file
        df = test_single_file_processing(files[0])

        # Test 3: QC checks
        qc = test_qc_checks(df)

        # Test 4: Batch processing
        df_batch = test_batch_processing()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Review the output in data/processed/")
        print("  2. Check if timestamps look correct")
        print("  3. Ready to build crosswalk module!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
