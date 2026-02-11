#!/usr/bin/env python3
"""
Compute monthly household bills comparing two hourly tariffs for a single month.

Inputs:
  1) Hourly household loads for that month
     - output of analysis/rtp/compute_hourly_loads.py
     - expected columns:
         - account_identifier
         - hour_chicago (local naive datetime, truncated to hour)
         - kwh_hour
         - zip_code (optional but recommended)

  2) Two hourly tariff price files (baseline A and alternative B)
     - output of scripts/build_tariff_hourly_prices.py (or any file with
       the same schema)
     - required columns per file:
         - datetime_chicago (local naive datetime, hourly)
         - price_cents_per_kwh

The script is deliberately generalized to "tariff A vs tariff B" rather
than hard-coded as "flat vs RTP".  This lets us reuse the same billing
logic for any pair of rate structures (e.g., STOU vs DTOU) without code
changes—only the input price files differ.

This script:
  * Joins hourly loads to BOTH tariff price calendars on local time
    (fail-loud: every load hour must match in both tariffs, no silent drops)
  * Computes hourly costs under each tariff and the hourly difference
  * Aggregates to monthly totals per household
  * Optionally applies capacity charge and admin fee as costs specific to
    tariff B (the alternative), reducing the savings from switching A → B

Output schema per household:
  - account_identifier      (str)
  - zip_code                (str, if present in loads)
  - total_kwh               (float)
  - bill_a_dollars          (float, baseline tariff energy cost)
  - bill_b_dollars          (float, alternative tariff energy cost)
  - bill_diff_dollars       (float, bill_a - bill_b; positive = B is cheaper)
  - peak_kwh_hour           (float)
  - capacity_kw             (float, = peak_kwh_hour)
  - capacity_charge_dollars (float, applied to tariff B)
  - admin_fee_dollars       (float, applied to tariff B)
  - net_bill_diff_dollars   (float, bill_diff - capacity - admin)
  - pct_savings             (float, bill_diff / bill_a * 100)
  - net_pct_savings         (float, net_bill_diff / bill_a * 100)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_hourly_loads(path: Path) -> pl.DataFrame:
    """Load hourly household loads."""
    if not path.exists():
        raise FileNotFoundError(f"Hourly loads file not found: {path}")

    logger.info(f"Loading hourly loads from {path}")
    df = pl.read_parquet(path)

    required = {"account_identifier", "hour_chicago", "kwh_hour"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hourly loads missing required columns: {sorted(missing)}")

    return df


def load_tariff_prices(path: Path, label: str) -> pl.DataFrame:
    """Load hourly tariff prices and rename price column with label suffix."""
    if not path.exists():
        raise FileNotFoundError(f"Tariff prices file not found: {path}")

    logger.info(f"Loading tariff prices ({label}) from {path}")
    df = pl.read_parquet(path)

    required = {"datetime_chicago", "price_cents_per_kwh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tariff prices ({label}) missing required columns: {sorted(missing)}")

    return df.select([
        pl.col("datetime_chicago"),
        pl.col("price_cents_per_kwh").alias(f"price_{label}_cents"),
    ])


def _join_tariff(
    df: pl.DataFrame,
    tariff: pl.DataFrame,
    price_col: str,
    label: str,
) -> pl.DataFrame:
    """Left-join a tariff onto df, fail-loud on nulls or row-count changes."""
    n_before = df.height

    # Left join so we can detect unmatched hours explicitly via nulls,
    # rather than silently dropping rows with an inner join.
    joined = df.join(
        tariff,
        left_on="hour_chicago",
        right_on="datetime_chicago",
        how="left",
    )

    # Fail-loud: any null price means the tariff calendar has gaps
    # (e.g. missing DST hours). Better to crash here than produce
    # bills with silently missing hours.
    n_null = joined.select(pl.col(price_col).is_null().sum()).item()
    if n_null > 0:
        unmatched = joined.filter(pl.col(price_col).is_null()).select("hour_chicago").unique().sort("hour_chicago")
        raise ValueError(f"Tariff {label}: {n_null} load rows have no matching price. Unmatched hours:\n{unmatched}")

    # Row-count check catches duplicate datetime_chicago in the tariff,
    # which would silently inflate bills via a many-to-one fan-out.
    if joined.height != n_before:
        raise RuntimeError(
            f"Tariff {label}: join changed row count {n_before} → {joined.height}. "
            "Tariff prices may have duplicate datetime_chicago values."
        )

    return joined


def compute_household_bills(
    hourly_loads: pl.DataFrame,
    prices_a: pl.DataFrame,
    prices_b: pl.DataFrame,
    capacity_rate_dollars_per_kw_month: float = 0.0,
    admin_fee_dollars: float = 0.0,
) -> pl.DataFrame:
    """
    Join hourly loads to two tariff price calendars and compute monthly bills.

    Tariff A is the baseline; tariff B is the alternative.
    bill_diff = bill_a - bill_b (positive means B is cheaper).
    Capacity charge and admin fee are costs specific to tariff B.

    Args:
        hourly_loads: Hourly kWh per household for a single month.
        prices_a: Baseline tariff with price_a_cents column.
        prices_b: Alternative tariff with price_b_cents column.
        capacity_rate_dollars_per_kw_month: Capacity charge rate ($/kW-month).
        admin_fee_dollars: Fixed monthly admin fee per household.

    Returns:
        DataFrame with one row per household and billing/comparison fields.
    """
    n_loads = hourly_loads.height
    logger.info("Joining %d load rows with two tariff price calendars...", n_loads)

    joined = _join_tariff(hourly_loads, prices_a, "price_a_cents", "A")
    joined = _join_tariff(joined, prices_b, "price_b_cents", "B")

    if joined.is_empty():
        raise RuntimeError("Join produced no rows. Check datetime alignment and inputs.")

    # Compute hourly costs and difference.
    # Sign convention: bill_diff = A - B, so positive means B is cheaper.
    # This matches the intuition "savings from switching TO the alternative."
    joined = joined.with_columns(
        (pl.col("kwh_hour") * pl.col("price_a_cents")).alias("bill_a_cents"),
        (pl.col("kwh_hour") * pl.col("price_b_cents")).alias("bill_b_cents"),
    ).with_columns(
        (pl.col("bill_a_cents") - pl.col("bill_b_cents")).alias("bill_diff_cents"),
    )

    # Grouping keys: always by account, include zip_code if present
    group_cols: list[str] = ["account_identifier"]
    if "zip_code" in joined.columns:
        group_cols.append("zip_code")

    logger.info("Aggregating to monthly bills per household...")

    monthly = joined.group_by(group_cols).agg(
        pl.col("kwh_hour").sum().alias("total_kwh"),
        pl.col("bill_a_cents").sum().alias("bill_a_cents"),
        pl.col("bill_b_cents").sum().alias("bill_b_cents"),
        pl.col("bill_diff_cents").sum().alias("bill_diff_cents"),
        pl.col("kwh_hour").max().alias("peak_kwh_hour"),
    )

    # Convert to dollars and define capacity_kw
    monthly = monthly.with_columns(
        (pl.col("bill_a_cents") / 100).alias("bill_a_dollars"),
        (pl.col("bill_b_cents") / 100).alias("bill_b_dollars"),
        (pl.col("bill_diff_cents") / 100).alias("bill_diff_dollars"),
        pl.col("peak_kwh_hour").alias("capacity_kw"),
    ).drop(["bill_a_cents", "bill_b_cents", "bill_diff_cents"])

    # Gross % savings (energy only, relative to baseline A)
    monthly = monthly.with_columns(
        pl.when(pl.col("bill_a_dollars") > 0)
        .then(pl.col("bill_diff_dollars") / pl.col("bill_a_dollars") * 100)
        .otherwise(None)
        .alias("pct_savings"),
    )

    # Capacity + admin adjustments are costs unique to tariff B (the
    # alternative), not present in baseline A. They reduce the savings
    # from switching A→B, modeling real-world RTP program fees.
    apply_capacity = capacity_rate_dollars_per_kw_month > 0
    apply_admin = admin_fee_dollars > 0

    if apply_capacity or apply_admin:
        logger.info(
            "Applying capacity/admin adjustments (tariff B costs): "
            f"capacity_rate=${capacity_rate_dollars_per_kw_month:.3f}/kW-month, "
            f"admin_fee=${admin_fee_dollars:.2f}/month"
        )

        monthly = monthly.with_columns(
            (pl.col("capacity_kw") * capacity_rate_dollars_per_kw_month).alias("capacity_charge_dollars"),
            pl.lit(admin_fee_dollars).alias("admin_fee_dollars"),
        )
    else:
        # Keep columns explicit but zeroed so downstream code never has
        # to branch on "does this column exist?" — stable schema always.
        monthly = monthly.with_columns(
            pl.lit(0.0).alias("capacity_charge_dollars"),
            pl.lit(0.0).alias("admin_fee_dollars"),
        )

    monthly = monthly.with_columns(
        (pl.col("bill_diff_dollars") - pl.col("capacity_charge_dollars") - pl.col("admin_fee_dollars")).alias(
            "net_bill_diff_dollars"
        ),
    )

    monthly = monthly.with_columns(
        pl.when(pl.col("bill_a_dollars") > 0)
        .then(pl.col("net_bill_diff_dollars") / pl.col("bill_a_dollars") * 100)
        .otherwise(None)
        .alias("net_pct_savings"),
    )

    return monthly


def summarize_results(df: pl.DataFrame) -> None:
    """Print a textual summary of the billing comparison."""
    n_households = df.height

    gross_mean = df["bill_diff_dollars"].mean()
    gross_median = df["bill_diff_dollars"].median()

    net_mean = df["net_bill_diff_dollars"].mean()
    net_median = df["net_bill_diff_dollars"].median()

    pct_saving_gross = (df["bill_diff_dollars"] > 0).mean() * 100
    pct_paying_more_gross = (df["bill_diff_dollars"] < 0).mean() * 100

    pct_saving_net = (df["net_bill_diff_dollars"] > 0).mean() * 100
    pct_paying_more_net = (df["net_bill_diff_dollars"] < 0).mean() * 100

    print("\n" + "=" * 70)
    print("HOUSEHOLD BILL SUMMARY (MONTH)")
    print("=" * 70)
    print(f"Households: {n_households:,}")
    print("\nEnergy savings switching A → B (bill_a - bill_b):")
    print(f"  Mean bill difference:   ${gross_mean:8.2f}")
    print(f"  Median bill difference: ${gross_median:8.2f}")
    print(f"  % saving with B:        {pct_saving_gross:6.1f}%")
    print(f"  % paying more on B:     {pct_paying_more_gross:6.1f}%")
    print("\nAfter capacity + admin adjustments (tariff B costs):")
    print(f"  Mean NET bill difference:   ${net_mean:8.2f}")
    print(f"  Median NET bill difference: ${net_median:8.2f}")
    print(f"  % NET saving with B:        {pct_saving_net:6.1f}%")
    print(f"  % NET paying more on B:     {pct_paying_more_net:6.1f}%")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute monthly household bills comparing two hourly tariffs.",
    )
    parser.add_argument(
        "--hourly-loads",
        type=Path,
        required=True,
        help="Parquet file with hourly loads (output of compute_hourly_loads.py)",
    )
    parser.add_argument(
        "--tariff-prices-a",
        type=Path,
        required=True,
        help="Baseline tariff: Parquet with datetime_chicago + price_cents_per_kwh.",
    )
    parser.add_argument(
        "--tariff-prices-b",
        type=Path,
        required=True,
        help="Alternative tariff: Parquet with datetime_chicago + price_cents_per_kwh.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output parquet path for household monthly bills.",
    )
    parser.add_argument(
        "--capacity-rate-dollars-per-kw-month",
        type=float,
        default=0.0,
        help="Capacity charge rate in $/kW-month applied to tariff B (default: 0.0 = disabled).",
    )
    parser.add_argument(
        "--admin-fee-dollars",
        type=float,
        default=0.0,
        help="Fixed monthly admin fee for tariff B in $ (default: 0.0 = disabled).",
    )

    args = parser.parse_args()

    logger.info("Starting household bill computation...")

    hourly_loads = load_hourly_loads(args.hourly_loads)
    prices_a = load_tariff_prices(args.tariff_prices_a, "a")
    prices_b = load_tariff_prices(args.tariff_prices_b, "b")

    bills = compute_household_bills(
        hourly_loads,
        prices_a,
        prices_b,
        capacity_rate_dollars_per_kw_month=args.capacity_rate_dollars_per_kw_month,
        admin_fee_dollars=args.admin_fee_dollars,
    )

    summarize_results(bills)

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bills.write_parquet(args.output)
    logger.info(f"Wrote household bills to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
