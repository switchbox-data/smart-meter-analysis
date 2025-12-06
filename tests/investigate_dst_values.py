#!/usr/bin/env python3
"""
Investigate HR2430 and HR2500 columns in source data.

Questions:
1. Are these columns always populated or only on DST days?
2. What values do they contain?
3. How do they compare to adjacent intervals?
4. Can we safely ignore them?
"""

from pathlib import Path

import polars as pl


def investigate_dst_columns():
    """Analyze HR2430 and HR2500 in sample CSVs."""

    sample_dir = Path("data/samples")
    csv_files = list(sample_dir.glob("*.csv"))

    if not csv_files:
        print("No sample files found. Run: just download-samples 202308 5")
        return

    print("=" * 80)
    print("INVESTIGATING HR2430 AND HR2500 COLUMNS")
    print("=" * 80)

    for csv_file in csv_files[:3]:  # Check first 3 files
        print(f"\n📁 File: {csv_file.name}")
        print("-" * 80)

        df = pl.read_csv(csv_file)

        # Check if these columns exist
        has_2430 = "INTERVAL_HR2430_ENERGY_QTY" in df.columns
        has_2500 = "INTERVAL_HR2500_ENERGY_QTY" in df.columns

        print(f"Has HR2430: {has_2430}")
        print(f"Has HR2500: {has_2500}")

        if not (has_2430 and has_2500):
            print("⚠️  Expected columns not found!")
            continue

        # Count non-null values
        null_2430 = df["INTERVAL_HR2430_ENERGY_QTY"].null_count()
        null_2500 = df["INTERVAL_HR2500_ENERGY_QTY"].null_count()
        total_rows = df.height

        print(f"\nTotal rows: {total_rows}")
        print(f"HR2430 null count: {null_2430} ({null_2430 / total_rows * 100:.1f}%)")
        print(f"HR2500 null count: {null_2500} ({null_2500 / total_rows * 100:.1f}%)")
        print(f"HR2430 non-null: {total_rows - null_2430} ({(total_rows - null_2430) / total_rows * 100:.1f}%)")
        print(f"HR2500 non-null: {total_rows - null_2500} ({(total_rows - null_2500) / total_rows * 100:.1f}%)")

        # Sample non-null values
        if total_rows - null_2430 > 0:
            print("\n📊 Sample HR2430 values (non-null):")
            sample_2430 = df.filter(pl.col("INTERVAL_HR2430_ENERGY_QTY").is_not_null()).head(5)
            print(
                sample_2430.select([
                    "INTERVAL_READING_DATE",
                    "ACCOUNT_IDENTIFIER",
                    "INTERVAL_HR2330_ENERGY_QTY",
                    "INTERVAL_HR2400_ENERGY_QTY",
                    "INTERVAL_HR2430_ENERGY_QTY",
                ])
            )

        if total_rows - null_2500 > 0:
            print("\n📊 Sample HR2500 values (non-null):")
            sample_2500 = df.filter(pl.col("INTERVAL_HR2500_ENERGY_QTY").is_not_null()).head(5)
            print(
                sample_2500.select([
                    "INTERVAL_READING_DATE",
                    "ACCOUNT_IDENTIFIER",
                    "INTERVAL_HR2400_ENERGY_QTY",
                    "INTERVAL_HR2430_ENERGY_QTY",
                    "INTERVAL_HR2500_ENERGY_QTY",
                ])
            )

        # Statistics on non-null values
        if total_rows - null_2430 > 0:
            print("\n📈 HR2430 Statistics (non-null values):")
            stats = df.filter(pl.col("INTERVAL_HR2430_ENERGY_QTY").is_not_null())[
                "INTERVAL_HR2430_ENERGY_QTY"
            ].describe()
            print(stats)

        if total_rows - null_2500 > 0:
            print("\n📈 HR2500 Statistics (non-null values):")
            stats = df.filter(pl.col("INTERVAL_HR2500_ENERGY_QTY").is_not_null())[
                "INTERVAL_HR2500_ENERGY_QTY"
            ].describe()
            print(stats)

        # Compare to adjacent regular intervals
        print("\n🔍 Comparing to adjacent intervals:")
        comparison = df.select([
            "INTERVAL_READING_DATE",
            pl.col("INTERVAL_HR2330_ENERGY_QTY").alias("23:30"),
            pl.col("INTERVAL_HR2400_ENERGY_QTY").alias("24:00"),
            pl.col("INTERVAL_HR2430_ENERGY_QTY").alias("24:30_DST"),
            pl.col("INTERVAL_HR2500_ENERGY_QTY").alias("25:00_DST"),
            pl.col("INTERVAL_HR0030_ENERGY_QTY").alias("00:30"),
            pl.col("INTERVAL_HR0100_ENERGY_QTY").alias("01:00"),
        ]).head(10)
        print(comparison)

    # Check August dates specifically
    print("\n" + "=" * 80)
    print("AUGUST 2023 DST CHECK")
    print("=" * 80)

    # DST transitions in 2023
    dst_spring = "2023-03-12"  # Spring forward
    dst_fall = "2023-11-05"  # Fall back

    print("\n2023 DST Transitions:")
    print(f"  Spring Forward: {dst_spring} (not in August)")
    print(f"  Fall Back: {dst_fall} (not in August)")
    print("\nAugust 2023 has NO DST transitions!")
    print("Therefore, HR2430 and HR2500 should be NULL for all August dates.")

    # Check if any August date has these values
    for csv_file in csv_files[:1]:
        df = pl.read_csv(csv_file)

        august_dates = (
            df.filter(
                (pl.col("INTERVAL_HR2430_ENERGY_QTY").is_not_null())
                | (pl.col("INTERVAL_HR2500_ENERGY_QTY").is_not_null())
            )
            .select("INTERVAL_READING_DATE")
            .unique()
        )

        if august_dates.height > 0:
            print("\n⚠️  Found non-null HR2430/HR2500 on these dates:")
            print(august_dates)
        else:
            print("\n✅ All HR2430/HR2500 are null (as expected)")

    # RECOMMENDATION
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    print("\nBased on the data above:")
    print("\nIF HR2430/HR2500 are mostly/all NULL:")
    print("  ✅ These are DST-only columns")
    print("  ✅ Safe to exclude from schema (use 48 columns)")
    print("  ✅ Filter would have removed them anyway")

    print("\nIF HR2430/HR2500 have actual values:")
    print("  ⚠️  ComEd may be using these for something else")
    print("  ⚠️  Need to understand what they represent")
    print("  ⚠️  May need special handling")

    print("\nTo decide:")
    print("  1. Check the percentages above")
    print("  2. If >95% null → exclude from schema")
    print("  3. If <95% null → investigate further with ComEd")


if __name__ == "__main__":
    investigate_dst_columns()
