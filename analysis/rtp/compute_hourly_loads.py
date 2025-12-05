#!/usr/bin/env python3
"""
Compute hourly household loads for RTP billing analysis.

Takes interval-level ComEd smart meter data (5- or 30-minute intervals)
and aggregates to hourly kWh per household, *restricted to* the set of
households that appear in the clustering assignments.

Typical usage:

    python analysis/rtp/compute_hourly_loads.py \
        --input data/validation_runs/202308_1000/processed/comed_202308.parquet \
        --cluster-assignments data/validation_runs/202308_1000/clustering/results/cluster_assignments.parquet \
        --output data/validation_runs/202308_1000/rtp/hourly_loads_202308.parquet
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


def load_sampled_accounts(assignments_path: Path) -> pl.Series:
    """
    Load the set of sampled households from cluster_assignments.parquet.

    Returns:
        Series of unique account_identifier values.
    """
    if not assignments_path.exists():
        raise FileNotFoundError(f"Cluster assignments not found: {assignments_path}")

    logger.info("Loading sampled households from %s", assignments_path)
    lf_assign = pl.scan_parquet(assignments_path)

    if "account_identifier" not in lf_assign.collect_schema().names():
        raise ValueError("Cluster assignments file has no 'account_identifier' column")

    df_accounts = lf_assign.select(pl.col("account_identifier").unique()).collect()

    accounts = df_accounts["account_identifier"]
    logger.info("Found %d unique sampled households", len(accounts))
    return accounts


def compute_hourly_loads(
    input_path: Path,
    assignments_path: Path | None,
    output_path: Path,
) -> None:
    """
    Aggregate interval-level kWh to hourly totals per household.

    If assignments_path is provided, restrict to households that appear
    in the clustering assignments (to keep memory manageable).

    Expects input schema to include at least:
        - account_identifier
        - zip_code
        - datetime   (naive local time, at 5- or 30-minute resolution)
        - kwh

    Produces:
        - account_identifier
        - zip_code
        - hour_chicago   (datetime truncated to the top of the hour)
        - kwh_hour       (sum of kWh within that hour)
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    logger.info("Loading interval data from %s", input_path)
    lf = pl.scan_parquet(input_path)

    required_cols = {"account_identifier", "zip_code", "datetime", "kwh"}
    missing = required_cols - set(lf.collect_schema().names())
    if missing:
        raise ValueError(f"Input file missing required columns: {sorted(missing)}")

    # Optionally restrict to sampled / clustered accounts
    if assignments_path is not None:
        accounts = load_sampled_accounts(assignments_path)
        logger.info("Filtering interval data to sampled households only...")
        lf = lf.filter(pl.col("account_identifier").is_in(accounts))

    logger.info("Aggregating to hourly loads per (account_identifier, zip_code, hour)...")

    lf_hourly = (
        lf.with_columns(pl.col("datetime").dt.truncate("1h").alias("hour_chicago"))
        .group_by(["account_identifier", "zip_code", "hour_chicago"])
        .agg(pl.col("kwh").sum().alias("kwh_hour"))
        .sort(["account_identifier", "hour_chicago"])
    )

    # Materialize and write
    df_hourly = lf_hourly.collect()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_hourly.write_parquet(output_path)

    logger.info("Wrote %d hourly rows to %s", len(df_hourly), output_path)
    logger.info(
        "Hourly load summary: kwh_hour min=%.4f, max=%.4f, mean=%.4f",
        df_hourly["kwh_hour"].min(),
        df_hourly["kwh_hour"].max(),
        df_hourly["kwh_hour"].mean(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate interval-level ComEd data to hourly loads per household.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to comed_YYYYMM.parquet (interval-level data).",
    )
    parser.add_argument(
        "--cluster-assignments",
        type=Path,
        default=None,
        help=(
            "Optional: cluster_assignments.parquet to restrict to sampled households (recommended for large months)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output parquet for hourly loads.",
    )

    args = parser.parse_args()

    try:
        compute_hourly_loads(args.input, args.cluster_assignments, args.output)
    except Exception as e:
        logger.error("Failed to compute hourly loads: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
