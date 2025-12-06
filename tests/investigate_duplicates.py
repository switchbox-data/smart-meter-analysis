#!/usr/bin/env python3
"""
Memory-efficient duplicate investigation using streaming.
"""

import polars as pl


def investigate_duplicates_efficient():
    """
    Memory-efficient duplicate detection using lazy evaluation.
    """

    print("=" * 80)
    print("DUPLICATE TIMESTAMP INVESTIGATION (Memory Efficient)")
    print("=" * 80)

    lf = pl.scan_parquet("data/processed/comed_202308.parquet")

    # Strategy 1: Check a larger sample (10M rows)
    print("\n🔍 Strategy 1: Checking 10M row sample...")

    sample_dups = (
        lf.head(10_000_000)
        .group_by(["account_identifier", "datetime"])
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
        .collect()
    )

    print(f"Duplicates in 10M sample: {sample_dups.height:,}")

    if sample_dups.height > 0:
        # Extrapolate to full dataset
        sample_ratio = 10_000_000 / 34_340_581
        estimated_total = int(sample_dups.height / sample_ratio)
        print(f"Estimated total duplicates: ~{estimated_total:,}")

        print("\n📊 Sample of duplicate pairs:")
        print(sample_dups.head(10))

        # Check if same or different kWh values
        print("\n🔬 Checking if duplicates have same kWh...")

        dup_sample = sample_dups.head(20)

        for row in dup_sample.iter_rows(named=True):
            account = row["account_identifier"]
            dt = row["datetime"]

            # Get all rows for this duplicate
            dup_rows = (
                lf.filter((pl.col("account_identifier") == account) & (pl.col("datetime") == dt))
                .select(["account_identifier", "datetime", "zip_code", "kwh"])
                .collect()
            )

            unique_kwh = dup_rows["kwh"].n_unique()

            status = "✅ SAME kWh (true duplicate)" if unique_kwh == 1 else "⚠️  DIFFERENT kWh (data issue!)"

            print(f"\nAccount {account}, {dt}:")
            print(f"  Occurrences: {row['count']}, {status}")
            print(dup_rows)

    # Strategy 2: Check specific accounts that had duplicates in sample
    print("\n" + "=" * 80)
    print("🔍 Strategy 2: Checking specific problematic accounts...")
    print("=" * 80)

    if sample_dups.height > 0:
        # Get accounts with most duplicates
        problem_accounts = sample_dups.sort("count", descending=True).head(5)["account_identifier"]

        for account in problem_accounts:
            print(f"\nAnalyzing account: {account}")

            account_data = (
                lf.filter(pl.col("account_identifier") == account)
                .select(["account_identifier", "datetime", "zip_code", "kwh"])
                .collect()
            )

            # Count duplicates for this account
            account_dups = (
                account_data.group_by(["account_identifier", "datetime"])
                .agg(pl.len().alias("count"))
                .filter(pl.col("count") > 1)
            )

            print(f"  Total rows: {account_data.height}")
            print(f"  Duplicate timestamps: {account_dups.height}")

            if account_dups.height > 0:
                print("  Sample duplicates:")

                for dup_row in account_dups.head(3).iter_rows(named=True):
                    dt = dup_row["datetime"]
                    dup_instances = account_data.filter(pl.col("datetime") == dt)
                    print(f"\n    Datetime: {dt}")
                    print(dup_instances)

    # Strategy 3: Partition-based check (most efficient)
    print("\n" + "=" * 80)
    print("🔍 Strategy 3: Partition-based duplicate count...")
    print("=" * 80)

    # Count unique (account, datetime) pairs vs total rows
    unique_pairs = lf.select(["account_identifier", "datetime"]).unique().select(pl.len()).collect()[0, 0]
    total_rows = lf.select(pl.len()).collect()[0, 0]

    duplicate_rows = total_rows - unique_pairs

    print(f"\nTotal rows: {total_rows:,}")
    print(f"Unique (account, datetime) pairs: {unique_pairs:,}")
    print(f"Duplicate rows: {duplicate_rows:,}")
    print(f"Duplicate percentage: {duplicate_rows / total_rows * 100:.2f}%")

    # RECOMMENDATION
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if duplicate_rows > 0:
        print(f"\n⚠️  You have {duplicate_rows:,} duplicate rows ({duplicate_rows / total_rows * 100:.2f}% of data)")
        print("\nThis is likely caused by:")
        print("  1. Same account appearing in multiple source files")
        print("  2. ZIP+4 overlap causing duplicate account entries")
        print("  3. Data corrections in source system")
        print("\n✅ RECOMMENDED FIX:")
        print("  Add deduplication to your pipeline:")
        print("\n  In aws_loader.py, after concatenation:")
        print("  lf_combined = lf_combined.unique(subset=['account_identifier', 'datetime'], keep='last')")
        print("\n  This will keep the most recent value for each (account, datetime) pair")
    else:
        print("\n✅ No duplicates found!")

    return duplicate_rows


if __name__ == "__main__":
    investigate_duplicates_efficient()
