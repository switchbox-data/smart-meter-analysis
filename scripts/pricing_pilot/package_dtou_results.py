#!/usr/bin/env python3
"""Package DTOU (Delivery Time-of-Use) results from existing STOU combined files.

DTOU changes ONLY delivery charges -- supply stays flat. Since DTOU and STOU
use the same TOU delivery rates (Info Sheet 67, same four time blocks), the
DTOU total bill delta equals the delivery delta alone. This script repackages
existing STOU combined data into DTOU-specific output files with correct total
bill columns.

Usage::

    uv run python scripts/pricing_pilot/package_dtou_results.py \\
        --month 202301 \\
        --billing-output-dir ~/pricing_pilot/billing_output

    # Run both months back-to-back:
    uv run python scripts/pricing_pilot/package_dtou_results.py \\
        --both \\
        --billing-output-dir ~/pricing_pilot/billing_output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

MONTHS = ["202301", "202307"]


def _resolve_paths(billing_output_dir: Path, month: str) -> tuple[Path, Path]:
    """Derive input and output file paths from billing-output-dir and month."""
    input_path = billing_output_dir / "stou_combined" / f"stou_combined_{month}.parquet"
    output_dir = billing_output_dir / "dtou_combined"
    return input_path, output_dir


def _build_dtou(stou: pl.DataFrame) -> pl.DataFrame:
    """Derive DTOU-specific columns from the STOU combined data.

    DTOU keeps supply flat on both sides, so:
      - dtou_total_bill_a = flat_supply + flat_delivery  (baseline)
      - dtou_total_bill_b = flat_supply + tou_delivery   (DTOU scenario)
      - dtou_total_delta  = delivery_delta               (supply cancels)
    """
    return stou.select(
        "account_identifier",
        "zip_code",
        "delivery_service_class",
        "total_kwh",
        pl.col("bill_a_dollars").alias("flat_supply_dollars"),
        "flat_delivery_dollars",
        "tou_delivery_dollars",
        "delivery_delta_dollars",
        (pl.col("bill_a_dollars") + pl.col("flat_delivery_dollars")).alias("dtou_total_bill_a_dollars"),
        (pl.col("bill_a_dollars") + pl.col("tou_delivery_dollars")).alias("dtou_total_bill_b_dollars"),
        pl.col("delivery_delta_dollars").alias("dtou_total_delta_dollars"),
        pl.when(pl.col("bill_a_dollars") + pl.col("flat_delivery_dollars") != 0)
        .then(pl.col("delivery_delta_dollars") / (pl.col("bill_a_dollars") + pl.col("flat_delivery_dollars")) * 100)
        .otherwise(None)
        .alias("dtou_total_pct_savings"),
    )


def _validate(dtou: pl.DataFrame) -> None:
    """Print validation checks to stdout."""
    n = len(dtou)

    # 1. Row count
    print("\n  --- Validation ---")
    print(f"  Output rows: {n:,}")

    # 2. Total delta summary
    td = dtou["dtou_total_delta_dollars"]
    print("\n  DTOU total delta ($/month):")
    print(f"    mean={td.mean():.4f}  median={td.median():.4f}  min={td.min():.4f}  max={td.max():.4f}")

    # 3. Percent savings summary
    ps = dtou["dtou_total_pct_savings"].drop_nulls()
    print("\n  DTOU total pct savings:")
    print(f"    mean={ps.mean():.4f}%  median={ps.median():.4f}%")

    # 4. Savings vs paying more
    n_saving = (td > 0).sum()
    n_paying = (td < 0).sum()
    n_even = (td == 0).sum()
    print(f"\n  % households saving:     {100 * n_saving / n:.2f}%")
    print(f"  % households paying more: {100 * n_paying / n:.2f}%")
    print(f"  % households even:        {100 * n_even / n:.2f}%")

    # 5. Delivery class value counts
    vc = dtou["delivery_service_class"].value_counts().sort("delivery_service_class")
    print(f"\n  Delivery class value counts:\n{vc}")

    # 6. Sanity: dtou_total_delta == delivery_delta (must be exact)
    diff = (dtou["dtou_total_delta_dollars"] - dtou["delivery_delta_dollars"]).abs()
    max_diff = diff.max()
    if max_diff > 0:
        raise ValueError(f"dtou_total_delta_dollars != delivery_delta_dollars, max diff = {max_diff}")
    print(f"\n  Sanity check: dtou_total_delta == delivery_delta (max diff: {max_diff:.2e}) OK")


def process_month(month: str, billing_output_dir: Path) -> int:
    """Process a single month end-to-end. Returns 0 on success, 1 on error."""
    print(f"\n{'=' * 60}")
    print(f"Processing {month}")
    print(f"{'=' * 60}")

    input_path, output_dir = _resolve_paths(billing_output_dir, month)

    # Check input exists
    if not input_path.exists():
        print(
            f"ERROR: STOU combined file not found at {input_path}\n"
            "Run scripts/pricing_pilot/compute_delivery_deltas.py first.",
            file=sys.stderr,
        )
        return 1

    # Read STOU combined data
    stou = pl.read_parquet(input_path)
    print(f"  STOU combined input: {len(stou):,} rows")

    # Build DTOU output
    dtou = _build_dtou(stou)

    # Validate
    _validate(dtou)

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"dtou_combined_{month}.parquet"
    dtou.sort("account_identifier").write_parquet(output_path)
    print(f"\n  Saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024 * 1024):.1f} MB")
    print(f"  Rows: {len(dtou):,}")

    return 0


def main() -> int:
    """Parse CLI args and dispatch to process_month for each requested month."""
    default_billing_output = Path.home() / "pricing_pilot" / "billing_output"

    parser = argparse.ArgumentParser(description="Package DTOU results from existing STOU combined files.")
    parser.add_argument(
        "--month",
        type=str,
        help="Month to process in YYYYMM format (e.g. 202301).",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Process both January 2023 and July 2023.",
    )
    parser.add_argument(
        "--billing-output-dir",
        type=Path,
        default=default_billing_output,
        help=f"Root billing output directory (default: {default_billing_output}).",
    )
    args = parser.parse_args()

    if not args.month and not args.both:
        parser.error("Specify --month YYYYMM or --both")
    if args.month and args.both:
        parser.error("Specify --month or --both, not both")

    months = MONTHS if args.both else [args.month]

    for month in months:
        if len(month) != 6 or not month.isdigit():
            print(f"ERROR: Invalid month format '{month}', expected YYYYMM", file=sys.stderr)
            return 1
        rc = process_month(month, args.billing_output_dir)
        if rc != 0:
            return rc

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
