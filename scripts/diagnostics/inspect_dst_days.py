#!/usr/bin/env python
"""
Download and inspect DST transition days from AWS. Assumes 2023 data.
"""

from pathlib import Path

import boto3
import polars as pl

# DST dates in 2023
DST_SPRING = "202303"  # March 12, 2023 - Spring Forward
DST_FALL = "202311"  # November 5, 2023 - Fall Back


def download_and_inspect(year_month: str, day_to_check: int):
    """Download one file and check specific day"""
    s3 = boto3.client("s3")

    # List files
    response = s3.list_objects_v2(
        Bucket="smart-meter-data-sb", Prefix=f"sharepoint-files/Zip4/{year_month}/", MaxKeys=1
    )

    if "Contents" not in response:
        print(f"❌ No files found for {year_month}")
        return

    # Download first file
    key = response["Contents"][0]["Key"]
    filename = Path(key).name
    print(f"\n{'=' * 60}")
    print(f"Inspecting: {filename}")
    print(f"Looking for day: {year_month[4:6]}/{day_to_check:02d}/{year_month[:4]}")
    print(f"{'=' * 60}\n")

    obj = s3.get_object(Bucket="smart-meter-data-sb", Key=key)
    df = pl.read_csv(obj["Body"], ignore_errors=True, infer_schema_length=5000)

    # Filter to specific date
    target_date = f"{year_month[4:6]}/{day_to_check:02d}/{year_month[:4]}"
    day_data = df.filter(pl.col("INTERVAL_READING_DATE") == target_date)

    if day_data.height == 0:
        print(f"⚠️  Date {target_date} not found in this file")
        print(f"Available dates: {df['INTERVAL_READING_DATE'].unique().sort().to_list()[:5]}")
        return

    print(f"✅ Found {day_data.height} rows for {target_date}")

    # Check interval columns
    interval_cols = [c for c in df.columns if c.startswith(("INTERVAL_HR", "HR"))]
    print(f"\n📊 Total interval columns: {len(interval_cols)}")

    # Get first customer's data
    first_customer = day_data.head(1)

    print(f"\n�� Checking interval values for Account {first_customer['ACCOUNT_IDENTIFIER'][0]}:")

    # Check each interval for nulls
    null_intervals = []
    non_null_intervals = []

    for col in sorted(interval_cols):
        val = first_customer[col][0]
        if val is None or (isinstance(val, float) and pl.Series([val]).is_null()[0]):
            null_intervals.append(col)
        else:
            non_null_intervals.append(col)

    print(f"\n  Non-null intervals: {len(non_null_intervals)}")
    print(f"  Null intervals: {len(null_intervals)}")

    if null_intervals:
        print("\n  Null columns:")
        for col in null_intervals[:10]:
            print(f"    - {col}")
        if len(null_intervals) > 10:
            print(f"    ... and {len(null_intervals) - 10} more")

    # Show sample values
    print("\n  Sample non-null values:")
    for col in non_null_intervals[:5]:
        val = first_customer[col][0]
        print(f"    {col}: {val}")

    # Check the DST columns specifically
    print("\n🕐 DST-specific columns:")
    dst_cols = ["INTERVAL_HR2400_ENERGY_QTY", "INTERVAL_HR2430_ENERGY_QTY", "INTERVAL_HR2500_ENERGY_QTY"]
    for col in dst_cols:
        if col in df.columns:
            val = first_customer[col][0] if col in first_customer.columns else "N/A"
            status = "NULL" if val is None else f"{val}"
            print(f"    {col}: {status}")

    # Save sample for inspection
    output_path = Path(f"data/raw/dst_sample_{year_month}_{day_to_check:02d}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    day_data.head(5).write_csv(output_path)
    print(f"\n💾 Saved sample to: {output_path}")


print("\n" + "=" * 60)
print("DST DAY INSPECTION")
print("=" * 60)

# Check Spring Forward (March 12, 2023)
print("\n🌸 SPRING FORWARD - March 12, 2023")
print("Expected: 46 intervals (2 AM hour missing)")
download_and_inspect(DST_SPRING, 12)

# Check Fall Back (November 5, 2023)
print("\n\n🍂 FALL BACK - November 5, 2023")
print("Expected: 50 intervals (1 AM hour repeated)")
download_and_inspect(DST_FALL, 5)

# Check a normal day for comparison
print("\n\n📅 NORMAL DAY - August 15, 2023")
print("Expected: 48 intervals")
download_and_inspect("202308", 15)
