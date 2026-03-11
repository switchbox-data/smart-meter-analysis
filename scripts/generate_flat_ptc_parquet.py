#!/usr/bin/env python3
"""Generate the flat PTC (price-to-compare) hourly parquet for 2026 rates.

Reads the existing comed_flat_hourly_prices_2023.parquet, keeps its exact
schema and datetime spine (8759 rows, DST-aware), and replaces the
price_cents_per_kwh column with 2026 PTC values:

  - Nonsummer (months 1-5, 10-12): 9.660 cents/kWh
  - Summer    (months 6-9):       10.028 cents/kWh

Usage:
    python scripts/generate_flat_ptc_parquet.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PARQUET_PATH = Path("data/reference/comed_flat_hourly_prices_2023.parquet")

# 2026 flat PTC rates (cents/kWh)
# Source: Client instruction (Eric, CUB) — current 2026 ComEd PTCs
# No public URL; values provided via email
SUMMER_PTC = 10.028  # months 6-9
NONSUMMER_PTC = 9.660  # months 1-5, 10-12


def main() -> None:
    df = pl.read_parquet(PARQUET_PATH)
    original_shape = df.shape
    original_schema = df.schema

    df = df.with_columns(
        pl.when(pl.col("month").is_between(6, 9))
        .then(pl.lit(SUMMER_PTC))
        .otherwise(pl.lit(NONSUMMER_PTC))
        .alias("price_cents_per_kwh")
    )

    # Verify schema and shape are unchanged
    if df.shape != original_shape:
        raise ValueError(f"Shape changed: {original_shape} -> {df.shape}")
    if df.schema != original_schema:
        raise ValueError(f"Schema changed: {original_schema} -> {df.schema}")

    df.write_parquet(PARQUET_PATH)
    print(f"Wrote {PARQUET_PATH} ({df.shape[0]} rows)")

    # Print verification summary
    summary = df.group_by("month").agg(pl.col("price_cents_per_kwh").first()).sort("month")
    for row in summary.iter_rows():
        print(f"  Month {row[0]:2d}: {row[1]:.3f} cents/kWh")


if __name__ == "__main__":
    main()
