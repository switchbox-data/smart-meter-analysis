#!/usr/bin/env python3
"""Generate income-vs-savings scatterplots from corrected combined parquets.

Produces 8 scatterplot PNGs (4 scenarios x 2 metrics) and a regression
summary CSV.  Reads from the corrected stou_combined and dtou_combined
parquets (steps 1-2 of the pricing pilot pipeline), NOT from the stale
bills_unscaled/ directory.

Scenarios:
    DTOU January, DTOU July, STOU January, STOU July

Metrics:
    Absolute delta ($) and percent savings (%)

Usage::

    uv run python scripts/pricing_pilot/generate_scatterplots.py \\
        --billing-output-dir ~/pricing_pilot/billing_output \\
        --census ~/pricing_pilot/billing_output/census_17_2023.parquet \\
        --out-dir ~/pricing_pilot/scatterplots

    # With custom options:
    uv run python scripts/pricing_pilot/generate_scatterplots.py \\
        --billing-output-dir ~/pricing_pilot/billing_output \\
        --census ~/pricing_pilot/billing_output/census_17_2023.parquet \\
        --out-dir ~/pricing_pilot/scatterplots \\
        --dpi 300 --min-accounts-per-bg 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
})

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

ALLOWED_CLASSES = {"C23", "C24", "C26", "C28"}

# (relative path, scenario tag, display name, month label,
#  delta column, pct savings column)
SCENARIOS: list[tuple[str, str, str, str, str, str]] = [
    (
        "dtou_combined/dtou_combined_202301.parquet",
        "dtou_jan",
        "DTOU",
        "January",
        "dtou_total_delta_dollars",
        "dtou_total_pct_savings",
    ),
    (
        "dtou_combined/dtou_combined_202307.parquet",
        "dtou_jul",
        "DTOU",
        "July",
        "dtou_total_delta_dollars",
        "dtou_total_pct_savings",
    ),
    (
        "stou_combined/stou_combined_202301.parquet",
        "stou_jan",
        "STOU",
        "January",
        "total_delta_dollars",
        "total_pct_savings",
    ),
    (
        "stou_combined/stou_combined_202307.parquet",
        "stou_jul",
        "STOU",
        "July",
        "total_delta_dollars",
        "total_pct_savings",
    ),
]


# ---------------------------------------------------------------------------
# Data loading + aggregation
# ---------------------------------------------------------------------------


def _load_combined(path: Path, delta_col: str, pct_col: str) -> pl.DataFrame:
    """Load a combined parquet and select/rename to canonical columns."""
    df = pl.read_parquet(path)
    print(f"  Loaded {df.height:,} rows from {path.name}")

    # Filter to allowed delivery classes
    n_before = df.height
    df = df.filter(pl.col("delivery_service_class").is_in(list(ALLOWED_CLASSES)))
    n_dropped = n_before - df.height
    if n_dropped > 0:
        print(f"  Dropped {n_dropped:,} rows with excluded delivery classes")

    return df.select(
        "account_identifier",
        pl.col(delta_col).alias("delta_dollars"),
        pl.col(pct_col).alias("pct_savings"),
    )


def _aggregate_to_bg(
    accounts: pl.DataFrame,
    bg_map: pl.DataFrame,
    census: pl.DataFrame,
    min_accounts: int,
) -> pl.DataFrame:
    """Join accounts → BG map → Census, aggregate to block-group level."""
    # Join to BG map
    joined = accounts.join(bg_map, on="account_identifier", how="inner")
    print(f"  Matched {joined.height:,}/{accounts.height:,} accounts to block groups")

    # Aggregate to block group
    bg = joined.group_by("geoid_bg").agg(
        pl.col("delta_dollars").mean().alias("mean_delta"),
        pl.col("pct_savings").mean().alias("mean_pct_savings"),
        pl.len().alias("n_accounts"),
    )

    # Filter by min accounts
    n_before = bg.height
    bg = bg.filter(pl.col("n_accounts") >= min_accounts)
    print(f"  Block groups: {bg.height:,} (dropped {n_before - bg.height:,} with < {min_accounts} accounts)")

    # Join Census income
    bg = bg.join(census, left_on="geoid_bg", right_on="GEOID", how="left")

    # Census stores income as natural log — exponentiate to raw dollars
    bg = bg.rename({"median_household_income": "log_median_household_income"}).with_columns(
        pl.col("log_median_household_income").exp().alias("median_household_income"),
    )

    # Drop BGs with missing or non-positive income
    n_before = bg.height
    bg = bg.filter(pl.col("median_household_income").is_not_null() & (pl.col("median_household_income") > 0))
    print(f"  Block groups with valid income: {bg.height:,} (dropped {n_before - bg.height:,})")

    total_accounts = bg["n_accounts"].sum()
    print(f"  Total accounts represented: {total_accounts:,}")

    return bg


# ---------------------------------------------------------------------------
# OLS regression
# ---------------------------------------------------------------------------


def _run_ols(
    bg: pl.DataFrame,
    y_col: str,
) -> dict[str, object]:
    """OLS of y_col on median_household_income (scaled by /10,000), HC1 robust SEs."""
    x = bg["median_household_income"].to_numpy() / 10_000
    y = bg[y_col].to_numpy()

    # Drop non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit(cov_type="HC1")

    return {
        "slope": float(model.params[1]),
        "intercept": float(model.params[0]),
        "r_squared": float(model.rsquared),
        "p_value": float(model.pvalues[1]),
        "se_slope": float(model.bse[1]),
        "n_obs": int(model.nobs),
        "model": model,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _dollar_fmt(x: float, _pos: int) -> str:
    if abs(x) >= 1000:
        return f"${x:,.0f}"
    return f"${x:,.0f}"


def _pct_fmt(x: float, _pos: int) -> str:
    return f"{x:.1f}%"


def _income_fmt(x: float, _pos: int) -> str:
    return f"${x:,.0f}"


def _plot_scatter(
    bg: pl.DataFrame,
    ols_result: dict[str, object],
    *,
    rate_name: str,
    month_label: str,
    y_col: str,
    metric_label: str,
    y_axis_label: str,
    out_path: Path,
    dpi: int,
) -> None:
    """Produce one scatterplot with OLS regression line + 95% CI band."""
    fig, ax = plt.subplots(figsize=(10, 7))

    income = bg["median_household_income"].to_numpy()
    y = bg[y_col].to_numpy()
    n_bg = len(income)
    n_accounts = int(bg["n_accounts"].sum())

    # Scatter
    ax.scatter(income, y, s=8, alpha=0.3, color="steelblue", zorder=2)

    # Regression line + CI band
    x_sorted = np.linspace(income.min(), income.max(), 300)
    X_pred = sm.add_constant(x_sorted / 10_000)
    model = ols_result["model"]
    pred = model.get_prediction(X_pred)
    pred_summary = pred.summary_frame(alpha=0.05)

    ax.plot(x_sorted, pred_summary["mean"], color="red", linewidth=2, zorder=3)
    ax.fill_between(
        x_sorted,
        pred_summary["mean_ci_lower"],
        pred_summary["mean_ci_upper"],
        color="red",
        alpha=0.2,
        zorder=1,
    )

    # Stats text box
    r2 = ols_result["r_squared"]
    slope = ols_result["slope"]
    p_val = ols_result["p_value"]
    stats_text = f"R² = {r2:.4f}\nSlope = {slope:.4g}\np = {p_val:.4g}"
    ax.text(
        0.97,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.95},
    )

    # Labels
    is_pct = metric_label == "pct_savings"
    ax.set_xlabel("Median Household Income ($)")
    ax.set_ylabel(y_axis_label)
    ax.set_title(f"{rate_name} — {month_label} 2023", fontsize=14, fontweight="bold")
    ax.text(
        0.5,
        1.01,
        f"{n_bg:,} block groups  |  {n_accounts:,} accounts",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="gray",
    )

    ax.xaxis.set_major_formatter(FuncFormatter(_income_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_pct_fmt if is_pct else _dollar_fmt))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate income-vs-savings scatterplots from corrected combined parquets.",
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
        help="Path to census_17_2023.parquet (block-group demographics, log-transformed income).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for scatterplot PNGs and regression summary CSV.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output PNGs (default: 150).",
    )
    parser.add_argument(
        "--min-accounts-per-bg",
        type=int,
        default=5,
        help="Minimum accounts per block group to include (default: 5).",
    )
    args = parser.parse_args()

    # Validate inputs
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

    for rel, _tag, _name, _month, _dc, _pc in SCENARIOS:
        p = args.billing_output_dir / rel
        if not p.exists():
            print(f"ERROR: Input file not found: {p}", file=sys.stderr)
            return 1

    # Load shared data
    print("Loading account→BG map...")
    bg_map = pl.read_parquet(bg_map_path).select(
        pl.col("account_identifier").cast(pl.Utf8),
        pl.col("geoid_bg").cast(pl.Utf8),
    )
    bg_map = bg_map.unique(subset=["account_identifier"], keep="first")
    print(f"  {bg_map.height:,} unique account→BG mappings")

    print("Loading Census block-group demographics...")
    census = pl.read_parquet(args.census).select("GEOID", "median_household_income")
    print(f"  {census.height:,} block groups in Census data")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    regression_rows: list[dict[str, object]] = []

    # Process each scenario
    for rel, tag, rate_name, month_label, delta_col, pct_col in SCENARIOS:
        print(f"\n{'=' * 60}")
        print(f"Scenario: {tag} ({rate_name} — {month_label})")
        print(f"{'=' * 60}")

        input_path = args.billing_output_dir / rel
        accounts = _load_combined(input_path, delta_col, pct_col)
        bg = _aggregate_to_bg(accounts, bg_map, census, args.min_accounts_per_bg)

        # --- Absolute delta scatterplot ---
        ols_abs = _run_ols(bg, "mean_delta")
        _plot_scatter(
            bg,
            ols_abs,
            rate_name=rate_name,
            month_label=month_label,
            y_col="mean_delta",
            metric_label="absolute_delta",
            y_axis_label=f"Mean Monthly Savings Under {rate_name} ($)",
            out_path=args.out_dir / f"{tag}_absolute_delta.png",
            dpi=args.dpi,
        )
        regression_rows.append({
            "scenario": tag,
            "month": month_label,
            "metric": "absolute_delta",
            "slope": ols_abs["slope"],
            "intercept": ols_abs["intercept"],
            "r_squared": ols_abs["r_squared"],
            "p_value": ols_abs["p_value"],
            "n_block_groups": ols_abs["n_obs"],
            "n_accounts": int(bg["n_accounts"].sum()),
        })

        # --- Percent savings scatterplot ---
        ols_pct = _run_ols(bg, "mean_pct_savings")
        _plot_scatter(
            bg,
            ols_pct,
            rate_name=rate_name,
            month_label=month_label,
            y_col="mean_pct_savings",
            metric_label="pct_savings",
            y_axis_label=f"Mean Monthly Savings Under {rate_name} (%)",
            out_path=args.out_dir / f"{tag}_pct_savings.png",
            dpi=args.dpi,
        )
        regression_rows.append({
            "scenario": tag,
            "month": month_label,
            "metric": "pct_savings",
            "slope": ols_pct["slope"],
            "intercept": ols_pct["intercept"],
            "r_squared": ols_pct["r_squared"],
            "p_value": ols_pct["p_value"],
            "n_block_groups": ols_pct["n_obs"],
            "n_accounts": int(bg["n_accounts"].sum()),
        })

    # Export regression summary CSV
    reg_df = pl.DataFrame(regression_rows)
    csv_path = args.out_dir / "scatterplot_regression_summary.csv"
    reg_df.write_csv(csv_path)
    print(f"\n{'=' * 60}")
    print(f"Regression summary: {csv_path}")
    print(f"{'=' * 60}")

    for row in regression_rows:
        print(
            f"  {row['scenario']:<10s} {row['metric']:<16s}  "
            f"slope={row['slope']:>10.6f}  R²={row['r_squared']:>8.4f}  "
            f"p={row['p_value']:>8.4f}  n_bg={row['n_block_groups']}  n_acct={row['n_accounts']:,}"
        )

    print(f"\nAll outputs written to {args.out_dir}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
