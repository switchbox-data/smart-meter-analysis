#!/usr/bin/env python
"""
Comprehensive validation of AWS/transformeation/census pipeline outputs.
Checks data quality before next steps (crosswalk and clustering).
"""

import sys
from pathlib import Path

import polars as pl


def validate_smart_meter_data(parquet_path: Path) -> dict:
    """Validate Step 0 output: transformed smart meter data"""
    print("\n" + "=" * 60)
    print("VALIDATING SMART METER DATA")
    print("=" * 60)

    if not parquet_path.exists():
        print(f"❌ File not found: {parquet_path}")
        return {"status": "FAIL", "reason": "File not found"}

    df = pl.read_parquet(parquet_path)
    issues = []
    warnings = []

    # Basic structure
    print("\n📊 Basic Info:")
    print(f"  Rows: {df.height:,}")
    print(f"  Columns: {df.columns}")
    print(f"  File size: {parquet_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Check required columns
    required_cols = ["zip_code", "account_identifier", "datetime", "kwh", "date", "hour", "weekday"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    # Check data types
    print("\n🔍 Data Types:")
    print(f"  datetime: {df['datetime'].dtype}")
    print(f"  kwh: {df['kwh'].dtype}")
    print(f"  hour: {df['hour'].dtype}")

    if df["datetime"].dtype != pl.Datetime:
        issues.append(f"datetime should be Datetime, got {df['datetime'].dtype}")

    # Check for nulls
    print("\n🔍 Null Values:")
    null_counts = {col: df[col].null_count() for col in df.columns}
    for col, count in null_counts.items():
        if count > 0:
            pct = 100 * count / df.height
            print(f"  {col}: {count:,} ({pct:.2f}%)")
            if col in ["datetime", "account_identifier"] and count > 0:
                issues.append(f"{col} has {count} nulls (should be 0)")
            elif col == "kwh" and pct > 10:
                warnings.append(f"kwh has {pct:.1f}% nulls (>10%)")

    # Check accounts
    print("\n👥 Accounts:")
    n_accounts = df["account_identifier"].n_unique()
    print(f"  Unique accounts: {n_accounts:,}")
    print(f"  Rows per account: {df.height / n_accounts:.0f}")

    # Check date range
    print("\n📅 Date Range:")
    if df["date"].null_count() == 0:
        min_date = df["date"].min()
        max_date = df["date"].max()
        date_span = (max_date - min_date).days if max_date and min_date else 0
        print(f"  From: {min_date}")
        print(f"  To: {max_date}")
        print(f"  Span: {date_span} days")

        if date_span > 400:
            warnings.append(f"Date span is {date_span} days (expected ~365 for 2023)")
    else:
        issues.append("All dates are null")

    # Check kWh values
    print("\n⚡ kWh Statistics:")
    kwh_stats = df.select([
        pl.col("kwh").min().alias("min"),
        pl.col("kwh").max().alias("max"),
        pl.col("kwh").mean().alias("mean"),
        pl.col("kwh").median().alias("median"),
        pl.col("kwh").std().alias("std"),
    ]).to_dicts()[0]

    for stat, value in kwh_stats.items():
        print(f"  {stat}: {value:.2f}" if value is not None else f"  {stat}: None")

    if kwh_stats["min"] and kwh_stats["min"] < 0:
        issues.append(f"Negative kWh values found (min: {kwh_stats['min']:.2f})")

    if kwh_stats["max"] and kwh_stats["max"] > 50:
        warnings.append(f"Very high kWh value found (max: {kwh_stats['max']:.2f} kWh in 30min)")

    # Check intervals per day
    print("\n📊 Intervals Per Day:")
    intervals_per_day = (
        df.group_by(["account_identifier", "date"])
        .agg(pl.len().alias("n_intervals"))
        .group_by("n_intervals")
        .agg(pl.len().alias("n_days"))
        .sort("n_intervals")
    )
    print(intervals_per_day)

    # Most days should have 48 intervals
    normal_days = intervals_per_day.filter(pl.col("n_intervals") == 48)
    if normal_days.height == 0:
        issues.append("No days with 48 intervals (expected for normal days)")

    # Check for duplicates
    print("\n🔍 Checking for Duplicates:")
    n_duplicates = df.height - df.select(["account_identifier", "datetime"]).unique().height
    print(f"  Duplicate rows: {n_duplicates:,}")
    if n_duplicates > 0:
        warnings.append(f"Found {n_duplicates} duplicate account/datetime combinations")

    # Check ZIP codes
    print("\n📍 ZIP Codes:")
    n_zips = df["zip_code"].n_unique()
    print(f"  Unique ZIP+4s: {n_zips}")
    sample_zips = df["zip_code"].unique()[:5].to_list()
    print(f"  Sample: {sample_zips}")

    # Summary
    print(f"\n{'=' * 60}")
    if issues:
        print("❌ VALIDATION FAILED")
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("✅ VALIDATION PASSED")

    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")

    return {
        "status": "FAIL" if issues else "PASS",
        "issues": issues,
        "warnings": warnings,
        "stats": {
            "n_rows": df.height,
            "n_accounts": n_accounts,
            "n_zips": n_zips,
            "date_range": (min_date, max_date) if df["date"].null_count() == 0 else None,
            "kwh_stats": kwh_stats,
        },
    }


def validate_census_data(parquet_path: Path) -> dict:
    """Validate Step 2 output: census data"""
    print("\n" + "=" * 60)
    print("VALIDATING CENSUS DATA")
    print("=" * 60)

    if not parquet_path.exists():
        print(f"❌ File not found: {parquet_path}")
        return {"status": "SKIP", "reason": "File not found (run census fetch first)"}

    df = pl.read_parquet(parquet_path)
    issues = []
    warnings = []

    # Basic structure
    print("\n📊 Basic Info:")
    print(f"  Block Groups: {df.height:,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  File size: {parquet_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Check GEOID
    print("\n🔍 GEOID Validation:")
    if "GEOID" not in df.columns:
        issues.append("Missing GEOID column")
    else:
        geoid_lengths = df["GEOID"].str.len_chars().unique().sort().to_list()
        print(f"  GEOID lengths: {geoid_lengths}")

        if 12 not in geoid_lengths:
            issues.append(f"No 12-digit GEOIDs found (got: {geoid_lengths})")

        sample_geoids = df["GEOID"].head(5).to_list()
        print(f"  Sample GEOIDs: {sample_geoids}")

        # Check if starts with state FIPS
        if not all(df["GEOID"].str.starts_with("17")):
            non_il = df.filter(~pl.col("GEOID").str.starts_with("17")).height
            warnings.append(f"{non_il} GEOIDs don't start with '17' (Illinois)")

    # Check key demographics exist
    print("\n📋 Key Demographics:")
    key_vars = [
        "Median_Household_Income",
        "Total_Households",
        "Owner_Occupied",
        "Renter_Occupied",
        "Urban_Rural_Classification",
    ]

    for var in key_vars:
        if var in df.columns:
            if df[var].dtype == pl.Utf8:
                unique_vals = df[var].unique().to_list()[:5]
                print(f"  ✅ {var}: {unique_vals}")
            else:
                stats = df[var].describe()
                print(f"  ✅ {var}: min={stats['min'][0]:.0f}, max={stats['max'][0]:.0f}, median={stats['50%'][0]:.0f}")
        else:
            issues.append(f"Missing variable: {var}")

    # Check for reasonable values
    print("\n🔍 Data Quality Checks:")

    if "Median_Household_Income" in df.columns:
        income_stats = df["Median_Household_Income"].filter(pl.col("Median_Household_Income").is_not_null())
        min_income = income_stats.min()
        max_income = income_stats.max()
        print(f"  Income range: ${min_income:,.0f} - ${max_income:,.0f}")

        if min_income < 0:
            issues.append(f"Negative median income: {min_income}")
        if max_income > 500000:
            warnings.append(f"Very high median income: ${max_income:,.0f}")

    # Check null percentages
    print("\n🔍 Completeness:")
    high_null_cols = []
    for col in df.columns:
        if col not in ["NAME", "GEOID"]:
            null_pct = 100 * df[col].null_count() / df.height
            if null_pct > 50:
                high_null_cols.append(f"{col} ({null_pct:.1f}%)")

    if high_null_cols:
        print(f"  Columns with >50% nulls: {len(high_null_cols)}")
        for col in high_null_cols[:5]:
            print(f"    - {col}")
        if len(high_null_cols) > 5:
            print(f"    ... and {len(high_null_cols) - 5} more")
    else:
        print("  ✅ No columns with excessive nulls")

    # Summary
    print(f"\n{'=' * 60}")
    if issues:
        print("❌ VALIDATION FAILED")
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("✅ VALIDATION PASSED")

    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")

    return {
        "status": "FAIL" if issues else "PASS",
        "issues": issues,
        "warnings": warnings,
        "stats": {
            "n_block_groups": df.height,
            "n_columns": len(df.columns),
        },
    }


def main():
    """Run all validations"""
    print("\n🔍 PIPELINE OUTPUT VALIDATION")
    print("=" * 60)

    # Check Step 0 output
    meter_result = validate_smart_meter_data(Path("data/processed/test_august_2023.parquet"))

    # Check Step 2 output (if exists)
    census_result = validate_census_data(Path("data/processed/census_blockgroups_il.parquet"))

    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Smart Meter Data: {meter_result['status']}")
    print(f"Census Data: {census_result['status']}")

    all_pass = meter_result["status"] in ["PASS", "SKIP"] and census_result["status"] in ["PASS", "SKIP"]

    if all_pass:
        print("\n✅ All validations passed!")
        print("Ready to proceed with crosswalk and clustering.")
        return 0
    else:
        print("\n❌ Some validations failed. Review issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
