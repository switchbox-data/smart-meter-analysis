#!/usr/bin/env python3
"""
Compute monthly household bills under flat vs RTP pricing for a single month.

Inputs:
  1) Hourly household loads for that month
     - output of analysis/rtp/compute_hourly_loads.py
     - expected columns:
         - account_identifier
         - hour_chicago (local naive datetime, truncated to hour)
         - kwh_hour
         - zip_code (optional but recommended)

  2) Hourly flat-vs-RTP spreads for the year
     - output of scripts/build_rtp_spreads.py
     - expected columns:
         - datetime_chicago (local naive datetime, hourly)
         - rtp_price_cents
         - flat_price_cents

This script:
  * Joins hourly loads to hourly prices on local time
  * Computes hourly costs under RTP and flat pricing
  * Aggregates to monthly totals per household
  * Optionally subtracts:
      - a capacity charge based on each household's peak hourly kW
      - a fixed monthly admin fee

Key outputs per household:
  - total_kwh
  - rtp_bill_dollars
  - flat_bill_dollars
  - bill_diff_dollars          (flat - RTP, energy component only)
  - capacity_kw                (approx. peak hourly kWh)
  - capacity_charge_dollars    (if capacity rate > 0)
  - admin_fee_dollars          (if admin fee > 0)
  - net_bill_diff_dollars      (bill_diff_dollars - capacity - admin)
  - pct_savings                (bill_diff_dollars / flat_bill)
  - net_pct_savings            (net_bill_diff_dollars / flat_bill)
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


def load_spreads(path: Path) -> pl.DataFrame:
    """Load hourly flat-vs-RTP spreads."""
    if not path.exists():
        raise FileNotFoundError(f"Spreads file not found: {path}")

    logger.info(f"Loading hourly spreads from {path}")
    df = pl.read_parquet(path)

    required = {"datetime_chicago", "rtp_price_cents", "flat_price_cents"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Spreads file missing required columns: {sorted(missing)}")

    # Only keep what we need
    return df.select([
        pl.col("datetime_chicago"),
        pl.col("rtp_price_cents"),
        pl.col("flat_price_cents"),
    ])


def compute_household_bills(
    hourly_loads: pl.DataFrame,
    spreads: pl.DataFrame,
    capacity_rate_dollars_per_kw_month: float = 0.0,
    admin_fee_dollars: float = 0.0,
) -> pl.DataFrame:
    """
    Join hourly loads to prices and compute monthly bills per household.

    Args:
        hourly_loads: Hourly kWh per household for a single month.
        spreads: Hourly RTP vs flat price data for the year.
        capacity_rate_dollars_per_kw_month: Capacity charge rate ($/kW-month).
        admin_fee_dollars: Fixed monthly admin fee per household.

    Returns:
        DataFrame with one row per household and billing fields.
    """
    logger.info("Joining hourly loads with hourly spreads on local hour...")

    joined = hourly_loads.join(
        spreads,
        left_on="hour_chicago",
        right_on="datetime_chicago",
        how="inner",
    )

    if joined.is_empty():
        raise RuntimeError("Join produced no rows. Check datetime alignment and inputs.")

    # Compute hourly cost under each tariff
    joined = joined.with_columns(
        (pl.col("kwh_hour") * pl.col("rtp_price_cents")).alias("bill_rtp_cents"),
        (pl.col("kwh_hour") * pl.col("flat_price_cents")).alias("bill_flat_cents"),
    ).with_columns((pl.col("bill_flat_cents") - pl.col("bill_rtp_cents")).alias("bill_diff_cents"))

    # Grouping keys: always by account, include zip_code if present
    group_cols: list[str] = ["account_identifier"]
    if "zip_code" in joined.columns:
        group_cols.append("zip_code")

    logger.info("Aggregating to monthly bills per household...")

    monthly = joined.group_by(group_cols).agg(
        pl.col("kwh_hour").sum().alias("total_kwh"),
        pl.col("bill_rtp_cents").sum().alias("rtp_bill_cents"),
        pl.col("bill_flat_cents").sum().alias("flat_bill_cents"),
        pl.col("bill_diff_cents").sum().alias("bill_diff_cents"),
        pl.col("kwh_hour").max().alias("peak_kwh_hour"),
    )

    # Convert to dollars and define capacity_kw
    monthly = monthly.with_columns(
        (pl.col("rtp_bill_cents") / 100).alias("rtp_bill_dollars"),
        (pl.col("flat_bill_cents") / 100).alias("flat_bill_dollars"),
        (pl.col("bill_diff_cents") / 100).alias("bill_diff_dollars"),
        pl.col("peak_kwh_hour").alias("capacity_kw"),
    ).drop(["rtp_bill_cents", "flat_bill_cents", "bill_diff_cents"])

    # Gross % savings (energy spread only)
    monthly = monthly.with_columns(
        pl.when(pl.col("flat_bill_dollars") > 0)
        .then(pl.col("bill_diff_dollars") / pl.col("flat_bill_dollars") * 100)
        .otherwise(None)
        .alias("pct_savings")
    )

    # Capacity + admin adjustments
    apply_capacity = capacity_rate_dollars_per_kw_month > 0
    apply_admin = admin_fee_dollars > 0

    if apply_capacity or apply_admin:
        logger.info(
            "Applying capacity/admin adjustments: "
            f"capacity_rate=${capacity_rate_dollars_per_kw_month:.3f}/kW-month, "
            f"admin_fee=${admin_fee_dollars:.2f}/month"
        )

        monthly = monthly.with_columns(
            (pl.col("capacity_kw") * capacity_rate_dollars_per_kw_month).alias("capacity_charge_dollars"),
            pl.lit(admin_fee_dollars).alias("admin_fee_dollars"),
        )
    else:
        # Keep columns explicit but zeroed so schema is stable
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
        pl.when(pl.col("flat_bill_dollars") > 0)
        .then(pl.col("net_bill_diff_dollars") / pl.col("flat_bill_dollars") * 100)
        .otherwise(None)
        .alias("net_pct_savings")
    )

    return monthly


def summarize_results(df: pl.DataFrame) -> None:
    """Print a textual summary of the billing results."""
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
    print("\nEnergy spread only (flat - RTP):")
    print(f"  Mean bill difference:   ${gross_mean:8.2f}")
    print(f"  Median bill difference: ${gross_median:8.2f}")
    print(f"  % saving with RTP:       {pct_saving_gross:6.1f}%")
    print(f"  % paying more on RTP:    {pct_paying_more_gross:6.1f}%")
    print("\nAfter capacity + admin adjustments:")
    print(f"  Mean NET bill difference:   ${net_mean:8.2f}")
    print(f"  Median NET bill difference: ${net_median:8.2f}")
    print(f"  % NET saving with RTP:       {pct_saving_net:6.1f}%")
    print(f"  % NET paying more on RTP:    {pct_paying_more_net:6.1f}%")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute monthly household bills under flat vs RTP pricing.",
    )
    parser.add_argument(
        "--hourly-loads",
        type=Path,
        required=True,
        help="Parquet file with hourly loads (output of compute_hourly_loads.py)",
    )
    parser.add_argument(
        "--spreads",
        type=Path,
        required=True,
        help="Parquet file with hourly flat vs RTP spreads.",
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
        help="Capacity charge rate in $/kW-month (default: 0.0 = disabled).",
    )
    parser.add_argument(
        "--admin-fee-dollars",
        type=float,
        default=0.0,
        help="Fixed monthly admin fee per household in $ (default: 0.0 = disabled).",
    )

    args = parser.parse_args()

    logger.info("Starting household bill computation...")

    hourly_loads = load_hourly_loads(args.hourly_loads)
    spreads = load_spreads(args.spreads)

    bills = compute_household_bills(
        hourly_loads,
        spreads,
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
