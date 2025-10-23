#!/usr/bin/env python
"""
Combine original 10-ZIP dataset with April city-wide boost.
"""

from pathlib import Path

import polars as pl

print("=" * 80)
print("COMBINING: Original + April City-Wide Boost")
print("=" * 80)

# Original 10 ZIPs (all 12 months)
original = Path("analysis/chicago_2024_full_year/combined/chicago_all_zips_2024_CM90.parquet")

# April boost from 45 additional ZIPs
april_files = list(Path("analysis/chicago_april_citywide").glob("zip*/final/*_CLIPPED_CM90.parquet"))

print(f"Original dataset: {original}")
print(f"April boost files: {len(april_files)} ZIPs")

if not april_files:
    print("\n❌ No April boost files! Run sample_april_citywide.py first")
    exit(1)

# Load April boost data
print("\nLoading April boost data...")
april_frames = []
for file in april_files:
    zipcode = file.parent.parent.name.replace("zip", "")
    df = pl.read_parquet(file).with_columns(pl.lit(zipcode).alias("zipcode"))
    customers = df["account_identifier"].n_unique()
    print(f"  ZIP {zipcode}: {customers} customers (after CM90)")
    april_frames.append(df)

if april_frames:
    april_boost = pl.concat(april_frames)
    total_april_boost = april_boost["account_identifier"].n_unique()
    print(f"\n✅ April boost total: {total_april_boost:,} customers from {len(april_files)} ZIPs")
else:
    print("\n❌ No April data passed CM90!")
    exit(1)

# Combine
print("\nCombining with original dataset...")
combined = pl.concat([pl.scan_parquet(original), pl.LazyFrame(april_boost)], how="vertical_relaxed")

output = Path("analysis/chicago_2024_full_year/combined/chicago_2024_with_april_boost_CM90.parquet")
combined.sink_parquet(output)

print(f"\n✅ Saved: {output}")
print(f"   Size: {output.stat().st_size / 1e9:.2f} GB")

# Monthly summary
monthly = (
    pl.scan_parquet(output)
    .group_by("sample_month")
    .agg([
        pl.col("account_identifier").n_unique().alias("customers"),
        pl.col("zipcode").n_unique().alias("zips"),
        pl.col("kwh").mean().alias("mean_kwh"),
    ])
    .sort("sample_month")
    .collect(engine="streaming")
)

print("\n📊 FINAL Monthly Summary (with April boost):")
print(monthly)

april_row = monthly.filter(pl.col("sample_month") == "202404")
if april_row.height > 0:
    april_customers = april_row["customers"][0]
    april_zips = april_row["zips"][0]
    other_avg = monthly.filter(pl.col("sample_month") != "202404")["customers"].mean()

    print(f"\n{'=' * 80}")
    print("APRIL 2024 FINAL STATUS:")
    print(f"  Customers: {april_customers:,}")
    print(f"  ZIPs: {april_zips}")
    print(f"  Other months avg: {other_avg:.0f}")
    print(f"  April is {april_customers / other_avg * 100:.0f}% of typical month")

    if april_customers >= other_avg * 0.8:
        print("  ✅ EXCELLENT - April is now comparable!")
    elif april_customers >= other_avg * 0.5:
        print("  ✅ GOOD - April is usable with minor caveat")
    else:
        print("  ⚠️  FAIR - April still lower but much better")
    print(f"{'=' * 80}")

print("\n✅ Ready for visualization!")
