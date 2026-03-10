#!/usr/bin/env python3
"""Statewide block-group-level pricing analysis: filter, aggregate, regress, export.

Pipeline
--------
1. Load four combined billing Parquets (STOU Jan/Jul, DTOU Jan/Jul).
2. Filter out delivery classes C25 and C27 (keep C23, C24, C26, C28).
3. Inner-join each to the statewide account→BG map.
4. Aggregate to block-group level (mean delta, mean pct savings, median delta,
   mean kWh, household count).
5. Left-join Census block-group demographics; drop BGs with null/non-positive
   median household income.
6. Run OLS regressions (HC1 robust SEs) of mean_delta and mean_pct_savings on
   median_household_income, both pooled across all classes and per-class.
7. Export regression summary CSV, per-scenario BG-level CSVs, and print results.

Sign convention (matches compute_delivery_deltas.py):
    delta = flat - alternative (bill_a - bill_b)
    Positive = customer SAVES under TOU
    A positive beta_1 on income → higher-income BGs save more (regressive in
    absolute terms).  A negative beta_1 → lower-income BGs save more (progressive).

Usage::

    python scripts/pricing_pilot/run_statewide_analysis.py \\
        --billing-output-dir /ebs/home/griffin_switch_box/runs/billing_output \\
        --census data/reference/census_17_2023.parquet \\
        --out-dir /ebs/home/griffin_switch_box/runs/billing_output/statewide_analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_CLASSES = {"C23", "C24", "C26", "C28"}
CLASS_LABELS: dict[str, str] = {
    "C23": "sf_no_esh",
    "C24": "mf_no_esh",
    "C26": "sf_esh",
    "C28": "mf_esh",
}

# (relative path from billing-output-dir, scenario label, rate type, month,
#  delta column, pct savings column)
SCENARIO_SPECS: list[tuple[str, str, str, str, str, str]] = [
    (
        "stou_combined/stou_combined_202301.parquet",
        "stou_jan",
        "stou",
        "202301",
        "total_delta_dollars",
        "total_pct_savings",
    ),
    (
        "stou_combined/stou_combined_202307.parquet",
        "stou_jul",
        "stou",
        "202307",
        "total_delta_dollars",
        "total_pct_savings",
    ),
    (
        "dtou_combined/dtou_combined_202301.parquet",
        "dtou_jan",
        "dtou",
        "202301",
        "dtou_total_delta_dollars",
        "dtou_total_pct_savings",
    ),
    (
        "dtou_combined/dtou_combined_202307.parquet",
        "dtou_jul",
        "dtou",
        "202307",
        "dtou_total_delta_dollars",
        "dtou_total_pct_savings",
    ),
]


# ---------------------------------------------------------------------------
# Step 1 — Filter
# ---------------------------------------------------------------------------


def _filter_classes(df: pl.DataFrame, file_label: str) -> pl.DataFrame:
    """Drop rows where delivery_service_class is not in ALLOWED_CLASSES."""
    n_before = df.height
    df_filtered = df.filter(pl.col("delivery_service_class").is_in(list(ALLOWED_CLASSES)))
    n_after = df_filtered.height
    n_dropped = n_before - n_after
    print(f"  {file_label}: {n_before:,} rows → {n_after:,} after filter ({n_dropped:,} dropped)")
    return df_filtered


# ---------------------------------------------------------------------------
# Step 2 — Join account→BG
# ---------------------------------------------------------------------------


def _join_bg_map(df: pl.DataFrame, bg_map: pl.DataFrame, file_label: str) -> pl.DataFrame:
    """Inner-join to account→BG map on account_identifier. Log match rate."""
    n_before = df.height
    joined = df.join(bg_map, on="account_identifier", how="inner")
    n_after = joined.height
    match_rate = 100 * n_after / n_before if n_before > 0 else 0.0
    print(f"  {file_label}: {n_after:,}/{n_before:,} accounts matched to BG ({match_rate:.2f}%)")
    return joined


# ---------------------------------------------------------------------------
# Step 3 — Aggregate to block group
# ---------------------------------------------------------------------------


def _aggregate_to_bg(
    df: pl.DataFrame,
    delta_col: str,
    pct_col: str,
) -> pl.DataFrame:
    """Group by geoid_bg and compute BG-level metrics."""
    return df.group_by("geoid_bg").agg(
        pl.col(delta_col).mean().alias("mean_delta"),
        pl.col(pct_col).mean().alias("mean_pct_savings"),
        pl.col(delta_col).median().alias("median_delta"),
        pl.col("total_kwh").mean().alias("mean_kwh"),
        pl.len().alias("n_households"),
    )


# ---------------------------------------------------------------------------
# Step 4 — Join Census
# ---------------------------------------------------------------------------


def _join_census(bg_df: pl.DataFrame, census: pl.DataFrame) -> pl.DataFrame:
    """Left-join Census on geoid_bg=GEOID; drop null/non-positive income."""
    n_before = bg_df.height
    joined = bg_df.join(census, left_on="geoid_bg", right_on="GEOID", how="left")
    joined = joined.filter(pl.col("median_household_income").is_not_null() & (pl.col("median_household_income") > 0))
    n_after = joined.height
    print(f"    BGs with valid income: {n_after:,}/{n_before:,}")
    # Census stores income as natural log; exponentiate to recover raw dollars.
    joined = joined.rename({"median_household_income": "log_median_household_income"}).with_columns(
        pl.col("log_median_household_income").exp().alias("median_household_income"),
    )
    return joined


# ---------------------------------------------------------------------------
# Step 5 — OLS regressions
# ---------------------------------------------------------------------------


def _run_regression(
    y: np.ndarray,
    x_income: np.ndarray,
    scenario: str,
    month: str,
    rate: str,
    delivery_class: str,
    dep_var: str,
) -> dict[str, object]:
    """Run OLS with HC1 robust SEs. Returns a dict row for the summary CSV."""
    x = sm.add_constant(x_income)
    model = sm.OLS(y, x).fit(cov_type="HC1")

    return {
        "scenario": scenario,
        "month": month,
        "rate": rate,
        "delivery_class": delivery_class,
        "dep_var": dep_var,
        "beta_0": model.params[0],
        "beta_1": model.params[1],
        "se_beta_1": model.bse[1],
        "t_stat": model.tvalues[1],
        "p_value": model.pvalues[1],
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs),
    }


def _interpret(dep_var: str, beta_1: float) -> str:
    """Return a one-line interpretation of the income coefficient."""
    if dep_var == "mean_delta":
        direction = (
            "regressive (higher-income BGs save more)" if beta_1 > 0 else "progressive (lower-income BGs save more)"
        )
        return f"  → Absolute savings: {direction} (beta_1={beta_1:.4f} $/10k income)"
    else:
        direction = "regressive" if beta_1 > 0 else "progressive"
        return f"  → Pct savings: {direction} (beta_1={beta_1:.4f} pct-pts/10k income)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Statewide block-group-level pricing analysis: filter, aggregate, regress, export.",
    )
    parser.add_argument(
        "--billing-output-dir",
        type=Path,
        required=True,
        help="Root billing output directory containing stou_combined/ and dtou_combined/.",
    )
    parser.add_argument(
        "--census",
        type=Path,
        required=True,
        help="Path to census_17_2023.parquet (block-group demographics).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for regression_summary.csv and bg_level_*.csv files.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not args.billing_output_dir.exists():
        print(f"ERROR: --billing-output-dir not found: {args.billing_output_dir}", file=sys.stderr)
        return 1
    if not args.census.exists():
        print(f"ERROR: --census not found: {args.census}", file=sys.stderr)
        return 1

    bg_map_path = args.billing_output_dir / "account_bg_map_statewide.parquet"
    if not bg_map_path.exists():
        print(
            f"ERROR: account_bg_map_statewide.parquet not found at {bg_map_path}\n"
            "Run scripts/pricing_pilot/build_statewide_account_bg_map.py first.",
            file=sys.stderr,
        )
        return 1

    for rel, _scenario, _rate, _month, _dc, _pc in SCENARIO_SPECS:
        p = args.billing_output_dir / rel
        if not p.exists():
            print(f"ERROR: Input file not found: {p}", file=sys.stderr)
            return 1

    # ------------------------------------------------------------------
    # Load shared data
    # ------------------------------------------------------------------
    print("Loading account→BG map...")
    bg_map = pl.read_parquet(bg_map_path)
    print(f"  {bg_map.height:,} account→BG mappings")

    print("Loading Census block-group demographics...")
    census = pl.read_parquet(args.census).select("GEOID", "median_household_income")
    print(f"  {census.height:,} block groups in Census data")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    regression_rows: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Process each scenario
    # ------------------------------------------------------------------
    for rel, scenario, rate, month, delta_col, pct_col in SCENARIO_SPECS:
        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario} (rate={rate}, month={month})")
        print(f"{'=' * 60}")

        # Load
        input_path = args.billing_output_dir / rel
        df = pl.read_parquet(input_path)
        print(f"  Loaded {df.height:,} rows from {input_path.name}")

        # Step 1: Filter
        df = _filter_classes(df, scenario)

        # Step 2: Join BG
        df = _join_bg_map(df, bg_map, scenario)

        # Step 3: Aggregate to BG
        bg_all = _aggregate_to_bg(df, delta_col, pct_col)
        print(f"  Aggregated to {bg_all.height:,} block groups (all classes)")

        # Step 4: Join Census
        bg_all = _join_census(bg_all, census)

        # Export BG-level CSV
        bg_csv_path = args.out_dir / f"bg_level_{scenario}.csv"
        bg_all.sort("geoid_bg").write_csv(bg_csv_path)
        print(f"  Wrote {bg_csv_path.name} ({bg_all.height:,} rows)")

        # Step 5: OLS regressions — pooled (all classes)
        income_scaled = bg_all["median_household_income"].to_numpy() / 10_000

        for dep_var, col_name in [("mean_delta", "mean_delta"), ("mean_pct_savings", "mean_pct_savings")]:
            y = bg_all[col_name].to_numpy()
            row = _run_regression(y, income_scaled, scenario, month, rate, "all", dep_var)
            regression_rows.append(row)

        # Per-class regressions
        for class_code, class_label in CLASS_LABELS.items():
            df_class = df.filter(pl.col("delivery_service_class") == class_code)
            if df_class.height == 0:
                continue

            bg_class = _aggregate_to_bg(df_class, delta_col, pct_col)
            bg_class = _join_census(bg_class, census)

            if bg_class.height < 3:  # need at least 3 obs for meaningful regression
                print(f"    Skipping {class_label}: only {bg_class.height} BGs with valid income")
                continue

            income_class = bg_class["median_household_income"].to_numpy() / 10_000
            for dep_var, col_name in [("mean_delta", "mean_delta"), ("mean_pct_savings", "mean_pct_savings")]:
                y = bg_class[col_name].to_numpy()
                row = _run_regression(y, income_class, scenario, month, rate, class_label, dep_var)
                regression_rows.append(row)

    # ------------------------------------------------------------------
    # Export regression summary
    # ------------------------------------------------------------------
    reg_df = pl.DataFrame(regression_rows)
    reg_csv_path = args.out_dir / "regression_summary.csv"
    reg_df.write_csv(reg_csv_path)
    print(f"\n{'=' * 60}")
    print(f"Regression summary: {reg_csv_path}")
    print(f"{'=' * 60}")

    # Print full table to stdout
    for row in regression_rows:
        label = f"{row['scenario']}/{row['delivery_class']}/{row['dep_var']}"
        print(
            f"  {label:<40s}  β₁={row['beta_1']:>10.6f}  SE={row['se_beta_1']:>10.6f}  "
            f"t={row['t_stat']:>8.3f}  p={row['p_value']:>8.4f}  R²={row['r_squared']:>8.4f}  n={row['n_obs']}"
        )
        print(_interpret(str(row["dep_var"]), float(row["beta_1"])))

    print(f"\nAll outputs written to {args.out_dir}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
