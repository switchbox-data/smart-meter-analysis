#!/usr/bin/env python
"""
Combine all sampled ZIP codes into one Chicago-wide dataset.
Checks for April 2024 data quality issues.
"""

from pathlib import Path

import polars as pl

BASE_DIR = Path("analysis/chicago_2024_full_year")
OUTPUT_DIR = BASE_DIR / "combined"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("COMBINING ALL CHICAGO ZIP CODES - 2024 FULL YEAR")
print("=" * 80)

# Find all CLIPPED_CM90 files
cm90_files = sorted(BASE_DIR.glob("zip*/final/*_CLIPPED_CM90.parquet"))

print(f"Found {len(cm90_files)} ZIP datasets")

if not cm90_files:
    print("❌ No files found! Run sampling first.")
    exit(1)

# Combine all ZIPs
print("\nLoading and combining...")
all_data = []

for file in cm90_files:
    zipcode = file.parent.parent.name.replace("zip", "")
    print(f"  ZIP {zipcode}...", end=" ")

    df = pl.read_parquet(file)
    df = df.with_columns(pl.lit(zipcode).alias("zipcode"))
    all_data.append(df)
    print(f"{len(df):,} rows")

combined = pl.concat(all_data)

print("\n✅ Combined dataset:")
print(f"   Total rows: {len(combined):,}")
print(f"   Unique customers: {combined['account_identifier'].n_unique():,}")
print(f"   ZIP codes: {combined['zipcode'].n_unique()}")
print(f"   Months: {combined['sample_month'].n_unique()}")
print(f"   Date range: {combined['date'].min()} to {combined['date'].max()}")

# Summary by month
monthly = (
    combined.group_by("sample_month")
    .agg([
        pl.col("account_identifier").n_unique().alias("customers"),
        pl.col("zipcode").n_unique().alias("zips"),
        pl.col("kwh").mean().alias("mean_kwh"),
        pl.len().alias("rows"),
    ])
    .sort("sample_month")
)

print("\n📊 Monthly Summary (ALL 12 months of 2024):")
print(monthly)

# Check April specifically
april = monthly.filter(pl.col("sample_month") == "202404")
if april.height > 0:
    april_customers = april["customers"][0]
    other_avg = monthly.filter(pl.col("sample_month") != "202404")["customers"].mean()
    print("\n⚠️  April 2024 check:")
    print(f"   April customers: {april_customers}")
    print(f"   Other months avg: {other_avg:.0f}")
    if april_customers < other_avg * 0.3:
        print("   🚨 WARNING: April has significantly fewer customers!")
        print("   Consider excluding April from analysis.")

# Summary by ZIP
zip_summary = (
    combined.group_by("zipcode")
    .agg([
        pl.col("sample_month").n_unique().alias("months"),
        pl.col("account_identifier").n_unique().alias("customers"),
        pl.col("kwh").mean().alias("mean_kwh"),
    ])
    .sort("zipcode")
)

print("\n📍 ZIP Code Summary:")
print(zip_summary)

# Save
output_file = OUTPUT_DIR / "chicago_all_zips_2024_CM90.parquet"
combined.write_parquet(output_file)

print(f"\n✅ Saved: {output_file}")
print(f"   File size: {output_file.stat().st_size / 1e9:.2f} GB")
print("=" * 80)
