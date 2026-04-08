#!/usr/bin/env python3
"""Compute delivery + supply deltas for STOU (Rate BEST) analysis.

Rate BEST shifts both supply AND delivery charges by time of day.
This script computes both components inline from hourly load data,
using the current rate constants (no dependency on pre-computed
household_bills.parquet from the core pipeline).

Steps:

1. Loads hourly loads from the v2 pipeline _tmp/ output
2. Assigns each hour to a TOU period
3. Joins the delivery class lookup
4. Computes delivery delta per household (flat delivery - TOU delivery)
5. Computes supply delta per household inline:
   flat supply = total_kwh * flat PTC
   STOU supply = sum(kwh_period * STOU rate[period])
6. Combines delivery + supply into total delta output

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
import yaml

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

# TOU DFCs by delivery class (cents/kWh)
# Source: CUB DTOD Fact Sheet (January 2026), cross-verified against
#         ComEd Info Sheet 67 (2026 column)
# URL (CUB): https://www.citizensutilityboard.org/wp-content/uploads/2025/12/ComEdDeliveryTOD.pdf
# URL (Info Sheet 67, in 2026 Ratebook): https://www.comed.com/cdn/assets/v3/assets/blt3ebb3fed6084be2a/blt86ebee5fe6ed02f8/69ab092748ce0ef3e48504df/2026_Ratebook.pdf
# Note: C28 overnight = 1.512 (CUB PDF has typo of 2.512; Info Sheet 67 confirms 1.512)
TOU_DFCS: dict[str, dict[str, float]] = {
    "C23": {"morning": 4.009, "midday_peak": 10.712, "evening": 3.747, "overnight": 2.984},
    "C24": {"morning": 3.073, "midday_peak": 8.689, "evening": 2.856, "overnight": 2.251},
    "C26": {"morning": 1.999, "midday_peak": 5.329, "evening": 1.890, "overnight": 1.550},
    "C28": {"morning": 1.925, "midday_peak": 4.975, "evening": 1.823, "overnight": 1.512},
}

# Flat DFCs by delivery class (cents/kWh)
# Source: CUB DTOD Fact Sheet (January 2026)
# URL: https://www.citizensutilityboard.org/wp-content/uploads/2025/12/ComEdDeliveryTOD.pdf
FLAT_DFCS: dict[str, float] = {
    "C23": 6.228,
    "C24": 4.791,
    "C26": 3.165,
    "C28": 2.996,
}

# ---------------------------------------------------------------------------
# Supply rates
# ---------------------------------------------------------------------------

# Flat PTCs by season (cents/kWh)
# Source: Client instruction (Eric, CUB) — current 2026 ComEd PTCs
# No public URL; values provided via email
FLAT_PTCS: dict[str, float] = {"summer": 10.028, "nonsummer": 9.660}

# STOU supply rates loaded at runtime from this YAML (avoids rate drift)
STOU_YAML_PATH = Path(__file__).resolve().parent.parent.parent / "rate_structures" / "comed_stou_2026.yaml"

MONTHS = ["202301", "202307"]


def _month_to_season(month: str) -> str:
    """Map YYYYMM to 'summer' (Jun-Sep) or 'nonsummer'."""
    mm = int(month[4:6])
    return "summer" if 6 <= mm <= 9 else "nonsummer"


def _load_stou_supply_rates(yaml_path: Path) -> dict[str, dict[str, float]]:
    """Parse STOU YAML into {season: {period: rate_cents_per_kwh}}."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    rates: dict[str, dict[str, float]] = {}
    for season in data["seasons"]:
        rates[season["name"]] = {p["period"]: p["price"] for p in season["periods"]}
    return rates


def _resolve_paths(billing_output_dir: Path, month: str) -> tuple[Path, Path, Path]:
    """Derive all file paths from billing-output-dir and month."""
    run_name = f"statewide_stou_{month}_v2"
    run_dir = billing_output_dir / run_name / run_name

    hourly_loads_path = run_dir / "_tmp" / f"month={month}" / "hourly_loads.parquet"
    delivery_lookup_path = billing_output_dir / "delivery_class_lookup.parquet"
    output_dir = billing_output_dir / "stou_combined"

    return hourly_loads_path, delivery_lookup_path, output_dir


def _aggregate_hourly_to_periods(hourly_loads_path: Path) -> pl.DataFrame:
    """Aggregate hourly kWh into TOU periods per household.

    Returns a DataFrame with columns:
        account_identifier, zip_code, period, kwh_period

    Uses PyArrow streaming to avoid OOM on large files.
    """
    print(f"  Scanning hourly loads: {hourly_loads_path}")
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


def _compute_supply_inline(
    period_kwh: pl.DataFrame, month: str, stou_rates: dict[str, dict[str, float]]
) -> pl.DataFrame:
    """Compute flat and STOU supply costs per household from period kWh.

    Returns a DataFrame with columns:
        account_identifier, zip_code, bill_a_dollars (flat supply),
        bill_b_dollars (STOU supply), supply_delta_dollars
    """
    season = _month_to_season(month)
    flat_ptc = FLAT_PTCS[season]
    season_rates = stou_rates[season]

    print(f"  Supply season: {season}")
    print(f"  Flat PTC: {flat_ptc} ¢/kWh")
    print(f"  STOU rates: {season_rates}")

    # Build STOU rate expression: map each period to its cents/kWh rate
    stou_rate_expr = pl.lit(0.0)
    for period, rate in season_rates.items():
        stou_rate_expr = pl.when(pl.col("period") == period).then(pl.lit(rate)).otherwise(stou_rate_expr)

    supply = (
        period_kwh.with_columns(
            (pl.col("kwh_period") * flat_ptc / 100.0).alias("flat_supply_contrib"),
            (pl.col("kwh_period") * stou_rate_expr / 100.0).alias("stou_supply_contrib"),
        )
        .group_by("account_identifier", "zip_code")
        .agg(
            pl.col("flat_supply_contrib").sum().alias("bill_a_dollars"),
            pl.col("stou_supply_contrib").sum().alias("bill_b_dollars"),
        )
        .with_columns((pl.col("bill_a_dollars") - pl.col("bill_b_dollars")).alias("supply_delta_dollars"))
    )

    print(f"  Supply computed for {len(supply):,} households")
    return supply


def _combine_delivery_and_supply(delivery: pl.DataFrame, supply: pl.DataFrame) -> pl.DataFrame:
    """Join delivery deltas with inline-computed supply costs.

    Returns combined output with supply + delivery deltas and totals.
    """
    combined = delivery.join(supply, on=["account_identifier", "zip_code"], how="inner")

    combined = combined.with_columns(
        (pl.col("bill_a_dollars") + pl.col("flat_delivery_dollars")).alias("total_bill_a_dollars"),
        (pl.col("bill_b_dollars") + pl.col("tou_delivery_dollars")).alias("total_bill_b_dollars"),
        (pl.col("supply_delta_dollars") + pl.col("delivery_delta_dollars")).alias("total_delta_dollars"),
    )

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


def _validate(combined: pl.DataFrame, n_hourly_hh: int) -> None:
    """Print validation checks to stdout."""
    n_out = len(combined)

    # 1. Row counts
    print("\n  --- Validation ---")
    print(f"  Hourly loads households:  {n_hourly_hh:,}")
    print(f"  Combined output rows:     {n_out:,}")
    drops_from_delivery = n_hourly_hh - n_out if n_hourly_hh > n_out else 0
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

    hourly_loads_path, delivery_lookup_path, output_dir = _resolve_paths(billing_output_dir, month)

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

    # Load STOU supply rates from YAML
    stou_rates = _load_stou_supply_rates(STOU_YAML_PATH)
    print(f"  Loaded STOU supply rates from {STOU_YAML_PATH.name}")

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

    # Step 3: Compute supply deltas inline (no household_bills.parquet dependency)
    supply = _compute_supply_inline(period_kwh, month, stou_rates)

    # Step 4: Combine delivery + supply
    combined = _combine_delivery_and_supply(delivery, supply)

    # Step 5: Validate
    _validate(combined, n_hourly_hh)

    # Step 6: Write output
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
