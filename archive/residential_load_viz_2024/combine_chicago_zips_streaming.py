#!/usr/bin/env python
"""
Combine all sampled ZIP codes using streaming (lazy frames).
"""

from pathlib import Path

import polars as pl

BASE_DIR = Path("analysis/chicago_2024_full_year")
OUTPUT_DIR = BASE_DIR / "combined"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("COMBINING ALL CHICAGO ZIP CODES - 2024 FULL YEAR (STREAMING)")
print("=" * 80)

# Find all CLIPPED_CM90 files
cm90_files = sorted(BASE_DIR.glob("zip*/final/*_CLIPPED_CM90.parquet"))
print(f"Found {len(cm90_files)} ZIP datasets\n")

if not cm90_files:
    print("❌ No files found!")
    exit(1)

# Add zipcode column to each file and save temporarily
temp_dir = OUTPUT_DIR / "temp"
temp_dir.mkdir(exist_ok=True)

print("Adding zipcode column to each file...")
temp_files = []

for file in cm90_files:
    zipcode = file.parent.parent.name.replace("zip", "")
    temp_file = temp_dir / f"{zipcode}.parquet"

    if not temp_file.exists():
        print(f"  Processing ZIP {zipcode}...", end=" ")
        df = pl.read_parquet(file).with_columns(pl.lit(zipcode).alias("zipcode"))
        df.write_parquet(temp_file)
        print(f"{len(df):,} rows")
    else:
        print(f"  ZIP {zipcode} already processed")

    temp_files.append(temp_file)

# Now use lazy scan to combine WITHOUT loading into memory
print("\nCombining all ZIPs (streaming)...")
output_file = OUTPUT_DIR / "chicago_all_zips_2024_CM90.parquet"

# Scan all temp files lazily and sink to final file
lf = pl.scan_parquet(temp_files)
lf.sink_parquet(output_file)

print(f"✅ Saved: {output_file}")
print(f"   File size: {output_file.stat().st_size / 1e9:.2f} GB")

# Now compute summaries using streaming
print("\n📊 Computing monthly summary (streaming)...")
monthly = (
    pl.scan_parquet(output_file)
    .group_by("sample_month")
    .agg([
        pl.col("account_identifier").n_unique().alias("customers"),
        pl.col("zipcode").n_unique().alias("zips"),
        pl.col("kwh").mean().alias("mean_kwh"),
        pl.len().alias("rows"),
    ])
    .sort("sample_month")
    .collect(streaming=True)
)

print("\nMonthly Summary (ALL 12 months of 2024):")
print(monthly)

# Check April
april = monthly.filter(pl.col("sample_month") == "202404")
if april.height > 0:
    april_customers = april["customers"][0]
    other_months = monthly.filter(pl.col("sample_month") != "202404")
    other_avg = other_months["customers"].mean()

    print("\n⚠️  April 2024 Data Quality Check:")
    print(f"   April customers: {april_customers:,}")
    print(f"   Other months avg: {other_avg:.0f}")

    if april_customers < other_avg * 0.5:
        print(f"   🚨 WARNING: April has {(1 - april_customers / other_avg) * 100:.0f}% fewer customers!")
        print("   Recommendation: Exclude April from analysis")
    else:
        print("   ✅ April looks okay!")

# ZIP summary
print("\n📍 Computing ZIP summary (streaming)...")
zip_summary = (
    pl.scan_parquet(output_file)
    .group_by("zipcode")
    .agg([
        pl.col("sample_month").n_unique().alias("months"),
        pl.col("account_identifier").n_unique().alias("customers_total"),
        pl.col("kwh").mean().alias("mean_kwh"),
    ])
    .sort("zipcode")
    .collect(streaming=True)
)

print("\nZIP Code Summary:")
print(zip_summary)

# Overall stats
print("\n📈 Overall Statistics:")
overall = (
    pl.scan_parquet(output_file)
    .select([
        pl.col("account_identifier").n_unique().alias("unique_customers"),
        pl.len().alias("total_rows"),
        pl.col("zipcode").n_unique().alias("zip_count"),
        pl.col("sample_month").n_unique().alias("month_count"),
        pl.col("kwh").mean().alias("mean_kwh"),
        pl.min("date").alias("start_date"),
        pl.max("date").alias("end_date"),
    ])
    .collect(streaming=True)
)

print(f"   Total customers: {overall['unique_customers'][0]:,}")
print(f"   Total rows: {overall['total_rows'][0]:,}")
print(f"   ZIP codes: {overall['zip_count'][0]}")
print(f"   Months: {overall['month_count'][0]}")
print(f"   Mean kWh: {overall['mean_kwh'][0]:.4f}")
print(f"   Date range: {overall['start_date'][0]} to {overall['end_date'][0]}")

print("\n" + "=" * 80)
print("✅ COMPLETE! Ready for visualization.")
print("=" * 80)
