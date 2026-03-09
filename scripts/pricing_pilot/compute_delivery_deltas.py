#!/usr/bin/env python3
"""Compute delivery deltas and combine with supply deltas for STOU analysis.

The v2 STOU pipeline computed supply-only bills. Rate BEST shifts both
supply AND delivery charges by time of day. This script:

1. Loads hourly loads from the v2 pipeline _tmp/ output
2. Assigns each hour to a TOU period
3. Joins the delivery class lookup
4. Computes delivery delta per household (flat delivery - TOU delivery)
5. Joins with existing supply-only household bills
6. Outputs combined total delta files

Usage::

    uv run python scripts/pricing_pilot/compute_delivery_deltas.py \\
        --month 202301 \\
        --billing-output-dir ~/pricing_pilot/billing_output

    # Run both months back-to-back:
    uv run python scripts/pricing_pilot/compute_delivery_deltas.py \\
        --both \\
        --billing-output-dir ~/pricing_pilot/billing_output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# TOU period definitions (Chicago local time, from hour_chicago)
# ---------------------------------------------------------------------------

PERIOD_MAP: dict[int, str] = {
    0: "overnight",
    1: "overnight",
    2: "overnight",
    3: "overnight",
    4: "overnight",
    5: "overnight",
    6: "morning",
    7: "morning",
    8: "morning",
    9: "morning",
    10: "morning",
    11: "morning",
    12: "morning",
    13: "midday_peak",
    14: "midday_peak",
    15: "midday_peak",
    16: "midday_peak",
    17: "midday_peak",
    18: "midday_peak",
    19: "evening",
    20: "evening",
    21: "overnight",
    22: "overnight",
    23: "overnight",
}

PERIODS = ("morning", "midday_peak", "evening", "overnight")

# ---------------------------------------------------------------------------
# Delivery rates (raw, no adjustment factors, no uncollectible multipliers)
# ---------------------------------------------------------------------------

# TOU DFCs by delivery class (Info Sheet 67, cents/kWh)
TOU_DFCS: dict[str, dict[str, float]] = {
    "C23": {"morning": 4.009, "midday_peak": 10.712, "evening": 3.747, "overnight": 2.984},
    "C24": {"morning": 3.073, "midday_peak": 8.689, "evening": 2.856, "overnight": 2.251},
    "C26": {"morning": 1.999, "midday_peak": 5.329, "evening": 1.890, "overnight": 1.550},
    "C28": {"morning": 1.925, "midday_peak": 4.975, "evening": 1.823, "overnight": 1.512},
}

# Flat DFCs by delivery class (Info Sheet 64.1, cents/kWh)
FLAT_DFCS: dict[str, float] = {
    "C23": 5.698,
    "C24": 4.354,
    "C26": 2.712,
    "C28": 2.576,
}

MONTHS = ["202301", "202307"]


def _resolve_paths(billing_output_dir: Path, month: str) -> tuple[Path, Path, Path, Path]:
    """Derive all file paths from billing-output-dir and month."""
    run_name = f"statewide_stou_{month}_v2"
    run_dir = billing_output_dir / run_name / run_name

    hourly_loads_path = run_dir / "_tmp" / f"month={month}" / "hourly_loads.parquet"
    supply_bills_path = run_dir / f"month={month}" / "household_bills.parquet"
    delivery_lookup_path = billing_output_dir / "delivery_class_lookup.parquet"
    output_dir = billing_output_dir / "stou_combined"

    return hourly_loads_path, supply_bills_path, delivery_lookup_path, output_dir


def _aggregate_hourly_to_periods(hourly_loads_path: Path) -> pl.DataFrame:
    """Aggregate hourly kWh into TOU periods per household.

    Returns a DataFrame with columns:
        account_identifier, zip_code, period, kwh_period

    Uses lazy scan → collect. Falls back to PyArrow streaming if OOM.
    """
    print(f"  Scanning hourly loads: {hourly_loads_path}")

    # Build a when/then chain for period assignment from the hour component
    period_expr = (
        pl.when(pl.col("hour_chicago").dt.hour().is_in([6, 7, 8, 9, 10, 11, 12]))
        .then(pl.lit("morning"))
        .when(pl.col("hour_chicago").dt.hour().is_in([13, 14, 15, 16, 17, 18]))
        .then(pl.lit("midday_peak"))
        .when(pl.col("hour_chicago").dt.hour().is_in([19, 20]))
        .then(pl.lit("evening"))
        .otherwise(pl.lit("overnight"))
        .alias("period")
    )

    try:
        result = (
            pl.scan_parquet(hourly_loads_path)
            .with_columns(period_expr)
            .group_by("account_identifier", "zip_code", "period")
            .agg(pl.col("kwh_hour").sum().alias("kwh_period"))
            .collect()
        )
        print(f"  Aggregated to {len(result):,} (account, zip, period) rows via lazy scan")
        return result
    except Exception as e:
        print(f"  Lazy collect failed ({e}), falling back to PyArrow streaming...")
        return _aggregate_hourly_pyarrow(hourly_loads_path)


def _aggregate_hourly_pyarrow(hourly_loads_path: Path) -> pl.DataFrame:
    """Fallback: stream hourly loads via PyArrow iter_batches()."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(hourly_loads_path))
    # Accumulate (account_identifier, zip_code, period) -> kwh sum
    sums: dict[tuple[str, str, str], float] = {}
    batch_count = 0

    for batch in pf.iter_batches(
        batch_size=500_000,
        columns=["account_identifier", "zip_code", "hour_chicago", "kwh_hour"],
    ):
        chunk = pl.from_arrow(batch)
        # Extract hour and assign period
        hours = chunk["hour_chicago"].dt.hour().to_list()
        accts = chunk["account_identifier"].to_list()
        zips = chunk["zip_code"].to_list()
        kwhs = chunk["kwh_hour"].to_list()

        for acct, zc, hr, kwh in zip(accts, zips, hours, kwhs):
            period = PERIOD_MAP[hr]
            key = (acct, zc, period)
            sums[key] = sums.get(key, 0.0) + (kwh or 0.0)

        batch_count += 1
        if batch_count % 100 == 0:
            print(f"    Processed {batch_count} batches, {len(sums):,} unique keys...")

    print(f"  PyArrow streaming complete: {batch_count} batches, {len(sums):,} unique keys")

    rows = [{"account_identifier": k[0], "zip_code": k[1], "period": k[2], "kwh_period": v} for k, v in sums.items()]
    return pl.DataFrame(rows)


def _compute_delivery_deltas(period_kwh: pl.DataFrame, delivery_lookup: pl.DataFrame) -> pl.DataFrame:
    """Compute flat and TOU delivery costs, then delivery delta per household.

    Returns one row per household with columns:
        account_identifier, zip_code, delivery_service_class, total_kwh,
        flat_delivery_dollars, tou_delivery_dollars, delivery_delta_dollars
    """
    # Pivot periods to columns: one row per (account_identifier, zip_code)
    pivoted = period_kwh.pivot(
        on="period",
        index=["account_identifier", "zip_code"],
        values="kwh_period",
    ).fill_null(0.0)

    # Ensure all period columns exist
    for p in PERIODS:
        if p not in pivoted.columns:
            pivoted = pivoted.with_columns(pl.lit(0.0).alias(p))

    # Join delivery class
    pivoted = pivoted.join(delivery_lookup, on="account_identifier", how="inner")

    # Compute total kWh
    pivoted = pivoted.with_columns(
        (pl.col("morning") + pl.col("midday_peak") + pl.col("evening") + pl.col("overnight")).alias("total_kwh")
    )

    # Compute TOU delivery cost in cents using map over delivery classes
    tou_cents_expr = pl.lit(0.0)
    for dc, rates in TOU_DFCS.items():
        dc_contribution = pl.lit(0.0)
        for period, rate in rates.items():
            dc_contribution = dc_contribution + pl.col(period) * rate
        tou_cents_expr = pl.when(pl.col("delivery_service_class") == dc).then(dc_contribution).otherwise(tou_cents_expr)

    # Compute flat delivery cost in cents
    flat_cents_expr = pl.lit(0.0)
    for dc, rate in FLAT_DFCS.items():
        flat_cents_expr = (
            pl.when(pl.col("delivery_service_class") == dc).then(pl.col("total_kwh") * rate).otherwise(flat_cents_expr)
        )

    pivoted = pivoted.with_columns(
        (flat_cents_expr / 100.0).alias("flat_delivery_dollars"),
        (tou_cents_expr / 100.0).alias("tou_delivery_dollars"),
    )

    # Delivery delta: positive = TOU delivery is cheaper
    pivoted = pivoted.with_columns(
        (pl.col("flat_delivery_dollars") - pl.col("tou_delivery_dollars")).alias("delivery_delta_dollars")
    )

    return pivoted.select(
        "account_identifier",
        "zip_code",
        "delivery_service_class",
        "total_kwh",
        "flat_delivery_dollars",
        "tou_delivery_dollars",
        "delivery_delta_dollars",
    )


def _combine_with_supply(delivery: pl.DataFrame, supply: pl.DataFrame) -> pl.DataFrame:
    """Join delivery deltas with supply-only household bills.

    Args:
        delivery: Delivery deltas per household.
        supply: Supply-only household bills DataFrame.

    Returns:
        Combined output with supply + delivery deltas and totals.
    """
    print(f"  Supply bills: {len(supply):,} rows")

    # Identify the supply delta column (bill_diff_dollars in pipeline output)
    if "bill_diff_dollars" in supply.columns:
        supply = supply.rename({"bill_diff_dollars": "supply_delta_dollars"})
    elif "bill_a_dollars" in supply.columns and "bill_b_dollars" in supply.columns:
        supply = supply.with_columns(
            (pl.col("bill_a_dollars") - pl.col("bill_b_dollars")).alias("supply_delta_dollars")
        )
    else:
        raise ValueError(f"Cannot derive supply delta from columns: {supply.columns}")

    # Select only columns we need from supply
    supply_cols = ["account_identifier", "bill_a_dollars", "bill_b_dollars", "supply_delta_dollars"]
    supply = supply.select([c for c in supply_cols if c in supply.columns])

    # Join on account_identifier
    combined = delivery.join(supply, on="account_identifier", how="inner")

    # Compute total bills and total delta
    combined = combined.with_columns(
        (pl.col("bill_a_dollars") + pl.col("flat_delivery_dollars")).alias("total_bill_a_dollars"),
        (pl.col("bill_b_dollars") + pl.col("tou_delivery_dollars")).alias("total_bill_b_dollars"),
        (pl.col("supply_delta_dollars") + pl.col("delivery_delta_dollars")).alias("total_delta_dollars"),
    )

    # Percent savings (null if total_bill_a = 0)
    combined = combined.with_columns(
        pl.when(pl.col("total_bill_a_dollars") != 0)
        .then(pl.col("total_delta_dollars") / pl.col("total_bill_a_dollars") * 100)
        .otherwise(None)
        .alias("total_pct_savings")
    )

    return combined.select(
        "account_identifier",
        "zip_code",
        "delivery_service_class",
        "total_kwh",
        "bill_a_dollars",
        "bill_b_dollars",
        "supply_delta_dollars",
        "flat_delivery_dollars",
        "tou_delivery_dollars",
        "delivery_delta_dollars",
        "total_bill_a_dollars",
        "total_bill_b_dollars",
        "total_delta_dollars",
        "total_pct_savings",
    )


def _validate(combined: pl.DataFrame, n_hourly_hh: int, n_supply_hh: int) -> None:
    """Print validation checks to stdout."""
    n_out = len(combined)

    # 1. Row counts
    print("\n  --- Validation ---")
    print(f"  Hourly loads households:  {n_hourly_hh:,}")
    print(f"  Supply bills households:  {n_supply_hh:,}")
    print(f"  Combined output rows:     {n_out:,}")
    drops_from_delivery = n_supply_hh - n_out if n_supply_hh > n_out else 0
    if drops_from_delivery > 0:
        print(f"  Dropped by delivery class join: {drops_from_delivery:,}")

    # 2. Delivery delta summary
    dd = combined["delivery_delta_dollars"]
    print("\n  Delivery delta ($/month):")
    print(f"    mean={dd.mean():.4f}  median={dd.median():.4f}  min={dd.min():.4f}  max={dd.max():.4f}")
    n_pos = (dd > 0).sum()
    n_neg = (dd < 0).sum()
    print(f"    % positive (TOU cheaper): {100 * n_pos / n_out:.2f}%")
    print(f"    % negative (TOU costlier): {100 * n_neg / n_out:.2f}%")

    # 3. Total delta summary
    td = combined["total_delta_dollars"]
    print("\n  Total delta ($/month):")
    print(f"    mean={td.mean():.4f}  median={td.median():.4f}  min={td.min():.4f}  max={td.max():.4f}")
    n_pos_t = (td > 0).sum()
    n_neg_t = (td < 0).sum()
    print(f"    % positive (TOU cheaper): {100 * n_pos_t / n_out:.2f}%")
    print(f"    % negative (TOU costlier): {100 * n_neg_t / n_out:.2f}%")

    # 4. Sanity: total_delta = supply_delta + delivery_delta
    check = (
        combined["total_delta_dollars"] - combined["supply_delta_dollars"] - combined["delivery_delta_dollars"]
    ).abs()
    max_diff = check.max()
    if max_diff >= 1e-10:
        raise ValueError(f"total_delta != supply_delta + delivery_delta, max diff = {max_diff}")
    print(f"\n  Sanity check: total_delta = supply + delivery (max diff: {max_diff:.2e}) OK")

    # 5. Delivery class value counts
    vc = combined["delivery_service_class"].value_counts().sort("delivery_service_class")
    print(f"\n  Delivery class value counts:\n{vc}")

    # 6. Null checks
    for col in ("delivery_delta_dollars", "total_delta_dollars", "delivery_service_class"):
        nc = combined[col].null_count()
        if nc > 0:
            print(f"  WARNING: {nc} nulls in {col}")
        else:
            print(f"  No nulls in {col}")


def process_month(month: str, billing_output_dir: Path) -> int:
    """Process a single month end-to-end. Returns 0 on success, 1 on error."""
    print(f"\n{'=' * 60}")
    print(f"Processing {month}")
    print(f"{'=' * 60}")

    hourly_loads_path, supply_bills_path, delivery_lookup_path, output_dir = _resolve_paths(billing_output_dir, month)

    # Check delivery class lookup exists
    if not delivery_lookup_path.exists():
        print(
            f"ERROR: delivery_class_lookup.parquet not found at {delivery_lookup_path}\n"
            "Run scripts/pricing_pilot/build_delivery_class_lookup.py first.",
            file=sys.stderr,
        )
        return 1

    # Check input files exist
    if not hourly_loads_path.exists():
        print(f"ERROR: hourly loads not found at {hourly_loads_path}", file=sys.stderr)
        return 1
    if not supply_bills_path.exists():
        print(f"ERROR: supply bills not found at {supply_bills_path}", file=sys.stderr)
        return 1

    # Load delivery class lookup
    delivery_lookup = pl.read_parquet(delivery_lookup_path)
    print(f"  Delivery class lookup: {len(delivery_lookup):,} accounts")

    # Step 1: Aggregate hourly loads to TOU periods
    period_kwh = _aggregate_hourly_to_periods(hourly_loads_path)
    n_hourly_hh = period_kwh["account_identifier"].n_unique()
    print(f"  Unique households in hourly loads: {n_hourly_hh:,}")

    # Step 2: Compute delivery deltas
    delivery = _compute_delivery_deltas(period_kwh, delivery_lookup)
    print(f"  Delivery deltas computed for {len(delivery):,} households")

    # Step 3: Combine with supply bills (single read)
    supply_bills = pl.read_parquet(supply_bills_path)
    n_supply_hh = len(supply_bills)
    combined = _combine_with_supply(delivery, supply_bills)

    # Step 4: Validate
    _validate(combined, n_hourly_hh, n_supply_hh)

    # Step 5: Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"stou_combined_{month}.parquet"
    combined.sort("account_identifier").write_parquet(output_path)
    print(f"\n  Saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024 * 1024):.1f} MB")
    print(f"  Rows: {len(combined):,}")

    return 0


def main() -> int:
    """Parse CLI args and dispatch to process_month for each requested month."""
    default_billing_output = Path.home() / "pricing_pilot" / "billing_output"

    parser = argparse.ArgumentParser(
        description="Compute delivery deltas and combine with supply deltas for STOU analysis."
    )
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
