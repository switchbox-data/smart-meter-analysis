"""Build account → delivery_service_class lookup from raw interval Parquet.

Reads the compacted production output for 202301 and 202307,
extracts the unique (account_identifier, delivery_service_class) pairs,
validates 1:1 mapping within each month and cross-month consistency,
then writes a single deduplicated lookup to Parquet.

Usage (on EC2):
    python scripts/pricing_pilot/build_delivery_class_lookup.py \
        --base-dir /ebs/home/griffin_switch_box/runs \
        --output /ebs/home/griffin_switch_box/runs/billing_output/delivery_class_lookup.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

MONTHS = ["202301", "202307"]


def load_lookup(base_dir: Path, month: str) -> pl.DataFrame:
    """Select account_identifier + delivery_service_class, deduplicate.

    Streams each compacted part file in 500K-row batches via PyArrow's
    iter_batches(), deduplicating per-file so peak memory stays low.
    """
    import pyarrow.parquet as pq

    mm = month[4:]
    parquet_dir = base_dir / f"out_{month}_production"
    part_dir = parquet_dir / "2023" / mm
    files = sorted(part_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No part-*.parquet files in {part_dir}")
    if len(files) != 3:
        print(f"  WARNING: expected 3 part files, found {len(files)} in {part_dir}")
    print(f"  Found {len(files)} compacted part files in {part_dir}")

    schema = {"account_identifier": pl.Utf8, "delivery_service_class": pl.Utf8}
    month_chunks: list[pl.DataFrame] = []
    for i, f in enumerate(files):
        pf = pq.ParquetFile(str(f))
        file_result = pl.DataFrame(schema=schema)
        pending: list[pl.DataFrame] = []
        batch_count = 0
        for batch in pf.iter_batches(
            batch_size=500_000,
            columns=["account_identifier", "delivery_service_class"],
        ):
            pending.append(pl.from_arrow(batch).unique())
            batch_count += 1
            if batch_count % 50 == 0:
                file_result = pl.concat([file_result, *pending]).unique()
                pending.clear()

        if pending:
            file_result = pl.concat([file_result, *pending]).unique()
            pending.clear()

        month_chunks.append(file_result)
        print(f"    File {i + 1}/{len(files)}: {f.name}  ({batch_count} batches, unique pairs: {file_result.height:,})")

    return pl.concat(month_chunks).unique()


def validate_one_class_per_account(df: pl.DataFrame, label: str) -> None:
    """Assert each account maps to exactly one delivery class."""
    multi = (
        df.group_by("account_identifier")
        .agg(pl.col("delivery_service_class").n_unique().alias("n_classes"))
        .filter(pl.col("n_classes") > 1)
    )
    if len(multi) > 0:
        print(f"\nERROR [{label}]: {len(multi)} accounts map to multiple classes:")
        print(multi.head(20))
        raise ValueError(f"{label}: accounts with multiple delivery classes")
    print(f"  [{label}] All {len(df)} account→class pairs are 1:1.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build account → delivery_service_class lookup")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/ebs/home/griffin_switch_box/runs"),
        help="Root directory containing out_YYYYMM_production/ folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/ebs/home/griffin_switch_box/runs/billing_output/delivery_class_lookup.parquet"),
        help="Output path for the lookup Parquet file",
    )
    args = parser.parse_args()

    lookups: dict[str, pl.DataFrame] = {}

    for month in MONTHS:
        print(f"\nLoading {month} from {args.base_dir} ...")
        df = load_lookup(args.base_dir, month)
        print(f"  Raw unique pairs: {len(df):,}")
        validate_one_class_per_account(df, month)

        # Null check
        null_acct = df["account_identifier"].null_count()
        null_class = df["delivery_service_class"].null_count()
        if null_acct > 0 or null_class > 0:
            raise ValueError(
                f"{month}: found nulls — account_identifier: {null_acct}, delivery_service_class: {null_class}"
            )
        print("  No nulls in either column.")

        vc = df["delivery_service_class"].value_counts().sort("delivery_service_class")
        print(f"  Value counts:\n{vc}")
        lookups[month] = df
        del df, null_acct, null_class, vc

    # ------------------------------------------------------------------
    # Cross-month consistency
    # ------------------------------------------------------------------
    jan, jul = lookups["202301"], lookups["202307"]
    overlap = jan.join(jul, on="account_identifier", how="inner", suffix="_jul")

    print("\n--- Cross-month consistency ---")
    print(f"  Accounts in Jan only:  {len(jan) - len(overlap):,}")
    print(f"  Accounts in Jul only:  {len(jul) - len(overlap):,}")
    print(f"  Overlapping accounts:  {len(overlap):,}")

    mismatches = overlap.filter(pl.col("delivery_service_class") != pl.col("delivery_service_class_jul"))
    if len(mismatches) > 0:
        print(f"  MISMATCH: {len(mismatches)} accounts changed class between months:")
        print(mismatches.head(20))
    else:
        print("  All overlapping accounts have consistent delivery class.")

    # ------------------------------------------------------------------
    # Merge and deduplicate
    # ------------------------------------------------------------------
    combined = pl.concat([jan, jul]).unique()
    validate_one_class_per_account(combined, "combined")

    vc_final = combined["delivery_service_class"].value_counts().sort("delivery_service_class")
    print("\n--- Final lookup ---")
    print(f"  Total unique accounts: {len(combined):,}")
    print(f"  Value counts:\n{vc_final}")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.sort("account_identifier").write_parquet(args.output)
    print(f"\n  Saved to {args.output}")
    print(f"  File size: {args.output.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
