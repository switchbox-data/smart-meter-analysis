#!/usr/bin/env python3
"""Build block-group regression dataset from household bills and run OLS.

Produces three BG-level outcome tables:
  - bg_month_outcomes.parquet   (BG x month)
  - bg_annual_outcomes.parquet  (BG, summed over all months)
  - bg_season_outcomes.parquet  (BG x season)

Then joins the appropriate table to census demographics and fits OLS.

Crosswalk logic mirrors the R script
``analysis/stage2/stage2_multinom_blockgroup_weighted.R`` exactly:

1. Normalize ZIP+4: accept ``#####-####`` or ``#########``; convert to ``#####-####``.
2. Crosswalk TSV with Zip, Zip4, CensusKey2023; derive
   ``block_group_geoid = substr(CensusKey2023, 1, 12)``.
3. Deterministic 1:1 mapping: smallest ``block_group_geoid`` per ``zip4``.
4. Fail-loud if mapping coverage is low or required columns are absent.

"Chicago" throughout this codebase means the IANA timezone **America/Chicago**
(Central Time).  The analysis covers the **full ComEd service territory**
across northern Illinois, not just the City of Chicago.

Typical usage::

    python analysis/rtp/build_regression_dataset.py \\
        --bills data/bills/run123/all_months_household_bills.parquet \\
        --crosswalk data/reference/comed_bg_zip4_crosswalk.txt \\
        --census data/reference/census_17_2023.parquet \\
        --output-dir data/bills/run123/regression

    python analysis/rtp/build_regression_dataset.py \\
        --bills data/bills/run123/all_months_household_bills.parquet \\
        --crosswalk data/reference/comed_bg_zip4_crosswalk.txt \\
        --census data/reference/census_17_2023.parquet \\
        --output-dir data/bills/run123/regression \\
        --regression-level bg_month
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

try:
    import statsmodels.api as sm
except ImportError:
    sys.exit("statsmodels is required but not installed. Install with:\n  uv add statsmodels")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# Preferred and fallback column names for the two OLS outcomes.
# The pipeline orchestrator may or may not add capacity/admin adjustments,
# so we prefer the "net" column (which accounts for those) but fall back
# to the gross column when the caller didn't compute net values.
SAVINGS_COLS = ("net_pct_savings", "pct_savings")
BILL_DIFF_COLS = ("net_bill_diff_dollars", "bill_diff_dollars")
CORE_PREDICTORS = ("median_household_income", "old_building_pct")


# ---------------------------------------------------------------------------
# ZIP+4 normalisation (mirrors R normalize_zip4)
# ---------------------------------------------------------------------------
# ComEd meter data carries 9-digit ZIP+4, not 5-digit ZIP. Using ZIP+4
# gives us ~10x more geographic resolution for the BG crosswalk—most
# ZIP+4s map to a single Census block group, whereas a 5-digit ZIP can
# span dozens of BGs and would require probabilistic allocation.


def _normalize_zip4_expr() -> pl.Expr:
    """Polars expression: derive ``zip4`` (#####-####) from ``zip_code``.

    Handles both ``#####-####`` (already correct) and ``#########`` (9-digit).
    """
    raw = pl.col("zip_code").cast(pl.Utf8).str.strip_chars()
    # Already has dash -> keep as-is; 9-digit -> insert dash; else null
    return (
        pl.when(raw.str.contains(r"^\d{5}-\d{4}$"))
        .then(raw)
        .when(raw.str.contains(r"^\d{9}$"))
        .then(raw.str.slice(0, 5) + pl.lit("-") + raw.str.slice(5, 4))
        .otherwise(pl.lit(None))
        .alias("zip4")
    )


# ---------------------------------------------------------------------------
# Crosswalk loading (duplicated from stage2_logratio_regression.py:184-226)
# ---------------------------------------------------------------------------
# This logic is intentionally duplicated rather than shared: the R-based
# stage-2 code and this Python pipeline must stay 1:1 identical in their
# crosswalk semantics. A shared helper would tempt divergence when one
# side is updated but the other isn't.


def load_crosswalk_one_to_one(
    crosswalk_path: Path,
) -> tuple[pl.LazyFrame, dict[str, Any]]:
    """Load ZIP+4 -> block group crosswalk with deterministic 1:1 linkage.

    Returns:
        mapping_lf: one row per zip4, with ``block_group_geoid`` = smallest BG.
        metrics: size / fanout metrics for provenance.
    """
    log.info("Loading crosswalk: %s", crosswalk_path)

    lf = (
        pl.scan_csv(crosswalk_path, separator="\t", infer_schema_length=10_000)
        .with_columns(
            (pl.col("Zip").cast(pl.Utf8).str.zfill(5) + pl.lit("-") + pl.col("Zip4").cast(pl.Utf8).str.zfill(4)).alias(
                "zip4"
            ),
            pl.col("CensusKey2023").cast(pl.Utf8).str.zfill(15).str.slice(0, 12).alias("block_group_geoid"),
        )
        .select("zip4", "block_group_geoid")
        .drop_nulls()
    )

    metrics = (
        lf.select(
            pl.len().alias("n_rows"),
            pl.col("zip4").n_unique().alias("n_zip4"),
            pl.col("block_group_geoid").n_unique().alias("n_bg"),
        )
        .collect()
        .to_dicts()[0]
    )

    fanout = (
        lf.group_by("zip4")
        .agg(pl.col("block_group_geoid").n_unique().alias("n_bg"))
        .filter(pl.col("n_bg") > 1)
        .select(pl.len().alias("n"))
        .collect()
        .item()
    )
    metrics["n_zip4_multi_bg"] = int(fanout)

    if fanout:
        log.warning(
            "Crosswalk fan-out: %s zip4s map to multiple block groups; using smallest GEOID per zip4.",
            f"{fanout:,}",
        )

    # Deterministic 1:1: smallest GEOID per zip4.
    # min() is arbitrary but reproducible—it guarantees the same mapping
    # across runs without depending on row order in the crosswalk file.
    mapping = lf.group_by("zip4").agg(
        pl.col("block_group_geoid").min().alias("block_group_geoid"),
    )
    return mapping, metrics


# ---------------------------------------------------------------------------
# Join bills -> BG
# ---------------------------------------------------------------------------


def attach_block_groups(
    bills: pl.DataFrame,
    crosswalk_lf: pl.LazyFrame,
    *,
    max_drop_pct: float,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Left-join bills to crosswalk on ``zip4``.  Fail-loud on high drop rate."""
    n_before = bills.height
    crosswalk = crosswalk_lf.collect()

    joined = bills.join(crosswalk, on="zip4", how="left")
    n_matched = joined.filter(pl.col("block_group_geoid").is_not_null()).height
    n_dropped = n_before - n_matched
    pct_dropped = (n_dropped / n_before * 100) if n_before else 0.0

    metrics = {
        "households_before_crosswalk": n_before,
        "households_matched": n_matched,
        "households_dropped": n_dropped,
        "pct_dropped": round(pct_dropped, 3),
    }

    if n_dropped:
        log.warning(
            "Crosswalk join: dropped %s households (%.2f%%) with no BG match.",
            f"{n_dropped:,}",
            pct_dropped,
        )

    if pct_dropped > max_drop_pct:
        raise RuntimeError(
            f"Crosswalk coverage too low: {pct_dropped:.2f}% of households "
            f"have no match (threshold: {max_drop_pct}%). "
            f"Dropped {n_dropped:,} of {n_before:,}."
        )

    return joined.drop_nulls("block_group_geoid"), metrics


# ---------------------------------------------------------------------------
# BG x month outcomes + rollups
# ---------------------------------------------------------------------------


def build_bg_month_outcomes(
    bills_bg: pl.DataFrame,
    *,
    bill_diff_col: str,
) -> pl.DataFrame:
    """Aggregate household bills to BG x month outcomes.

    Returns one row per (block_group_geoid, month) with additive sums and
    bill-weighted savings percentages.  ``n_household_months`` counts unique
    account_identifiers per BG-month (not persistent across months).
    """
    always_sum = ["total_kwh", "bill_a_dollars"]
    optional_sum = ["bill_b_dollars", "bill_diff_dollars", "net_bill_diff_dollars"]
    sum_cols = always_sum + [c for c in optional_sum if c in bills_bg.columns]

    agg_exprs: list[pl.Expr] = [pl.col(c).sum().alias(f"sum_{c}") for c in sum_cols]
    agg_exprs.append(pl.col("account_identifier").n_unique().alias("n_household_months"))

    result = bills_bg.group_by(["block_group_geoid", "month"]).agg(agg_exprs)

    # Bill-weighted pct savings: sum_diff / sum_bill_a * 100
    diff_sum_col = f"sum_{bill_diff_col}"
    result = result.with_columns(
        pl.when(pl.col("sum_bill_a_dollars") > 0)
        .then(pl.col(diff_sum_col) / pl.col("sum_bill_a_dollars") * 100)
        .otherwise(None)
        .alias("pct_savings_weighted"),
    )

    # Also compute net-weighted pct if net is present and wasn't the resolved col
    if "net_bill_diff_dollars" in bills_bg.columns and bill_diff_col != "net_bill_diff_dollars":
        result = result.with_columns(
            pl.when(pl.col("sum_bill_a_dollars") > 0)
            .then(pl.col("sum_net_bill_diff_dollars") / pl.col("sum_bill_a_dollars") * 100)
            .otherwise(None)
            .alias("net_pct_savings_weighted"),
        )

    return result.sort(["block_group_geoid", "month"])


def _rollup_bg_outcomes(
    bg_month: pl.DataFrame,
    group_cols: list[str],
    *,
    bill_diff_col: str,
) -> pl.DataFrame:
    """Roll up BG x month outcomes by summing additive columns and recomputing pct.

    Works for both annual (group by block_group_geoid) and seasonal
    (group by block_group_geoid + season) rollups.
    """
    additive = [c for c in bg_month.columns if c.startswith("sum_")]

    agg_exprs: list[pl.Expr] = [pl.col(c).sum() for c in additive]
    agg_exprs.append(pl.col("n_household_months").sum())

    result = bg_month.group_by(group_cols).agg(agg_exprs)

    # Recompute weighted pct savings from ratio of sums
    diff_sum_col = f"sum_{bill_diff_col}"
    if diff_sum_col in result.columns and "sum_bill_a_dollars" in result.columns:
        result = result.with_columns(
            pl.when(pl.col("sum_bill_a_dollars") > 0)
            .then(pl.col(diff_sum_col) / pl.col("sum_bill_a_dollars") * 100)
            .otherwise(None)
            .alias("pct_savings_weighted"),
        )

    if "sum_net_bill_diff_dollars" in result.columns and bill_diff_col != "net_bill_diff_dollars":
        result = result.with_columns(
            pl.when(pl.col("sum_bill_a_dollars") > 0)
            .then(pl.col("sum_net_bill_diff_dollars") / pl.col("sum_bill_a_dollars") * 100)
            .otherwise(None)
            .alias("net_pct_savings_weighted"),
        )

    return result.sort(group_cols)


def _derive_season_expr() -> pl.Expr:
    """Polars expression: derive season from YYYYMM ``month`` column.

    Winter=12,01,02  Spring=03,04,05  Summer=06,07,08  Fall=09,10,11
    Deterministic; no locale or timezone dependence.
    """
    mm = pl.col("month").str.slice(4, 2)
    return (
        pl.when(mm.is_in(["12", "01", "02"]))
        .then(pl.lit("Winter"))
        .when(mm.is_in(["03", "04", "05"]))
        .then(pl.lit("Spring"))
        .when(mm.is_in(["06", "07", "08"]))
        .then(pl.lit("Summer"))
        .when(mm.is_in(["09", "10", "11"]))
        .then(pl.lit("Fall"))
        .otherwise(pl.lit(None))
        .alias("season")
    )


# ---------------------------------------------------------------------------
# Predictor detection
# ---------------------------------------------------------------------------

EXCLUDE_COLS = {"block_group_geoid", "GEOID", "NAME"}


def detect_predictors(
    census: pl.DataFrame,
    *,
    mode: str,
) -> tuple[list[str], list[str]]:
    """Determine predictor columns from census DataFrame.

    Args:
        census: Census DataFrame (must include block_group_geoid + features).
        mode: ``"auto"`` | ``"core"`` | ``"col1,col2,..."``

    Returns:
        (predictors_used, excluded_all_null)
    """
    if mode == "core":
        preds = [c for c in CORE_PREDICTORS if c in census.columns]
        if not preds:
            raise RuntimeError(
                f"--predictors core requested but none of {CORE_PREDICTORS} "
                f"found in census columns: {sorted(census.columns)}"
            )
        return preds, []

    if mode != "auto":
        # Explicit comma-separated list
        requested = [c.strip() for c in mode.split(",") if c.strip()]
        missing = [c for c in requested if c not in census.columns]
        if missing:
            raise RuntimeError(
                f"Requested predictors not found in census: {missing}. Available: {sorted(census.columns)}"
            )
        return requested, []

    # Auto-infer: all numeric columns minus id/name/all-null.
    # This lets the census file evolve (add new demographic columns)
    # without requiring code changes—new numeric columns are picked up
    # automatically, which is the right default for exploratory modeling.
    numeric_types = {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32}
    candidates = [c for c, dt in census.schema.items() if c not in EXCLUDE_COLS and dt in numeric_types]
    # Drop all-null columns
    null_counts = census.select(candidates).null_count()
    all_null = [c for c in candidates if null_counts[c][0] == census.height]
    predictors = [c for c in candidates if c not in all_null]

    return sorted(predictors), sorted(all_null)


# ---------------------------------------------------------------------------
# OLS fitting
# ---------------------------------------------------------------------------


def fit_ols(
    df: pl.DataFrame,
    *,
    y_col: str,
    predictors: list[str],
    label: str,
    month_fe_col: str | None = None,
) -> dict[str, Any]:
    """Fit OLS: y_col ~ predictors [+ month fixed effects].

    When ``month_fe_col`` is set, month dummies (drop_first=True) are appended
    to the predictor matrix.  Month FE columns are recorded in the result but
    kept separate from the census predictors list.
    """
    select_cols = [y_col, *predictors]
    if month_fe_col:
        select_cols.append(month_fe_col)

    pdf = df.select(select_cols).to_pandas().dropna()
    n_obs = len(pdf)

    month_fe_names: list[str] = []
    if month_fe_col:
        month_dummies = pd.get_dummies(pdf[month_fe_col], drop_first=True, prefix="month")
        month_fe_names = list(month_dummies.columns)
        X_df = pd.concat([pdf[predictors], month_dummies], axis=1)
    else:
        X_df = pdf[predictors]

    all_predictor_names = list(X_df.columns)
    n_total_params = len(all_predictor_names) + 1  # +1 for constant

    if n_obs < n_total_params + 1:
        raise RuntimeError(
            f"OLS '{label}': only {n_obs} complete observations for "
            f"{n_total_params} parameters (incl. constant). "
            f"Need at least {n_total_params + 1}."
        )

    y = pdf[y_col].values
    X = sm.add_constant(X_df.values)
    full_names = ["const", *all_predictor_names]

    model = sm.OLS(y, X).fit()

    coefficients = {}
    for i, name in enumerate(full_names):
        coefficients[name] = {
            "estimate": float(model.params[i]),
            "std_error": float(model.bse[i]),
            "t_stat": float(model.tvalues[i]),
            "p_value": float(model.pvalues[i]),
        }

    return {
        "label": label,
        "outcome": y_col,
        "n_obs": n_obs,
        "n_predictors": len(predictors),
        "n_month_fe": len(month_fe_names),
        "month_fe_columns": month_fe_names,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "condition_number": float(model.condition_number),
        "coefficients": coefficients,
        "summary_text": str(model.summary()),
    }


# ---------------------------------------------------------------------------
# Outcome column resolution
# ---------------------------------------------------------------------------


def _resolve_col(
    columns: set[str],
    preferred: str,
    fallback: str,
    description: str,
) -> tuple[str, bool]:
    """Pick preferred column; fall back if absent.  Fail if neither exists."""
    if preferred in columns:
        return preferred, False
    if fallback in columns:
        log.warning(
            "%s: preferred column '%s' not found; falling back to '%s'.",
            description,
            preferred,
            fallback,
        )
        return fallback, True
    raise RuntimeError(
        f"{description}: neither '{preferred}' nor '{fallback}' found in bills columns: {sorted(columns)}"
    )


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build BG regression dataset from household bills and run OLS.",
    )
    p.add_argument("--bills", type=Path, required=True, help="Household bills parquet.")
    p.add_argument("--crosswalk", type=Path, required=True, help="ZIP+4 -> BG crosswalk TSV.")
    p.add_argument("--census", type=Path, required=True, help="Census demographics parquet.")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    p.add_argument(
        "--predictors",
        type=str,
        default="auto",
        help="'auto' (default) | 'core' (income + building_age) | 'col1,col2,...'",
    )
    p.add_argument(
        "--max-crosswalk-drop-pct",
        type=float,
        default=5.0,
        help="Fail if crosswalk drop rate exceeds this %% (default: 5.0).",
    )
    p.add_argument(
        "--min-obs-per-bg",
        type=int,
        default=3,
        help="Minimum household-months per BG for regression (default: 3).",
    )
    p.add_argument(
        "--regression-level",
        type=str,
        choices=["annual", "bg_month"],
        default="annual",
        help="Regression granularity: 'annual' (BG annual, default) or 'bg_month' (BG x month + month FE).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    args = parse_args(argv)

    # -- Validate input files -------------------------------------------------
    for path, label in [
        (args.bills, "bills"),
        (args.crosswalk, "crosswalk"),
        (args.census, "census"),
    ]:
        if not path.exists():
            log.error("%s file not found: %s", label.title(), path)
            return 1

    # -- Load bills -----------------------------------------------------------
    log.info("Loading bills: %s", args.bills)
    bills = pl.read_parquet(args.bills)
    n_bills_total = bills.height
    log.info("Bills: %d rows, columns: %s", n_bills_total, bills.columns)

    # Validate required columns (including month and bill_a_dollars)
    cols = set(bills.columns)
    for req in ("account_identifier", "zip_code", "total_kwh", "bill_a_dollars", "month"):
        if req not in cols:
            log.error("Bills missing required column: '%s'. Found: %s", req, sorted(cols))
            return 1

    # Validate month format: YYYYMM, 6 digits
    months_unique = bills.select("month").unique().to_series().to_list()
    for m in months_unique:
        if not isinstance(m, str) or len(m) != 6 or not m.isdigit():
            raise ValueError(f"Invalid month value '{m}'; expected YYYYMM (6 digits).")
    log.info("Months in bills: %s", sorted(months_unique))

    # -- Resolve outcome columns ----------------------------------------------
    savings_col, savings_fallback = _resolve_col(
        cols,
        SAVINGS_COLS[0],
        SAVINGS_COLS[1],
        "Savings outcome",
    )
    bill_diff_col, bill_diff_fallback = _resolve_col(
        cols,
        BILL_DIFF_COLS[0],
        BILL_DIFF_COLS[1],
        "Bill diff outcome",
    )
    log.info("Outcome columns: savings='%s', bill_diff='%s'", savings_col, bill_diff_col)

    # -- Derive zip4 from zip_code --------------------------------------------
    bills = bills.with_columns(_normalize_zip4_expr())
    n_null_zip4 = bills.filter(pl.col("zip4").is_null()).height
    if n_null_zip4:
        log.warning("%d rows have un-parseable zip_code -> null zip4; these will be dropped.", n_null_zip4)
        bills = bills.drop_nulls("zip4")

    # -- Load crosswalk -------------------------------------------------------
    crosswalk_lf, crosswalk_metrics = load_crosswalk_one_to_one(args.crosswalk)
    log.info("Crosswalk: %s", crosswalk_metrics)

    # -- Join bills -> BG -----------------------------------------------------
    bills_bg, join_metrics = attach_block_groups(
        bills,
        crosswalk_lf,
        max_drop_pct=args.max_crosswalk_drop_pct,
    )
    log.info("Join metrics: %s", join_metrics)

    # -- Build BG x month outcomes --------------------------------------------
    bg_month = build_bg_month_outcomes(bills_bg, bill_diff_col=bill_diff_col)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bg_month_path = args.output_dir / "bg_month_outcomes.parquet"
    bg_month.write_parquet(bg_month_path)
    log.info("Wrote BG x month outcomes: %s (%d rows)", bg_month_path, bg_month.height)

    # -- BG annual rollup -----------------------------------------------------
    bg_annual = _rollup_bg_outcomes(
        bg_month,
        ["block_group_geoid"],
        bill_diff_col=bill_diff_col,
    )
    bg_annual_path = args.output_dir / "bg_annual_outcomes.parquet"
    bg_annual.write_parquet(bg_annual_path)
    log.info("Wrote BG annual outcomes: %s (%d rows)", bg_annual_path, bg_annual.height)

    # -- BG seasonal rollup ---------------------------------------------------
    bg_month_with_season = bg_month.with_columns(_derive_season_expr())
    bg_season = _rollup_bg_outcomes(
        bg_month_with_season,
        ["block_group_geoid", "season"],
        bill_diff_col=bill_diff_col,
    )
    bg_season_path = args.output_dir / "bg_season_outcomes.parquet"
    bg_season.write_parquet(bg_season_path)
    log.info("Wrote BG seasonal outcomes: %s (%d rows)", bg_season_path, bg_season.height)

    # -- Load census + rename GEOID -------------------------------------------
    log.info("Loading census: %s", args.census)
    census = pl.read_parquet(args.census)
    if "GEOID" in census.columns and "block_group_geoid" not in census.columns:
        census = census.rename({"GEOID": "block_group_geoid"})
    elif "block_group_geoid" not in census.columns:
        log.error("Census has no 'GEOID' or 'block_group_geoid' column.")
        return 1

    # -- Choose regression table based on --regression-level ------------------
    if args.regression_level == "annual":
        bg_data = bg_annual.join(census, on="block_group_geoid", how="left")
    else:
        bg_data = bg_month.join(census, on="block_group_geoid", how="left")

    # -- Census join match rate (robust: check any non-key col is not null) ---
    census_nonkey = [c for c in census.columns if c != "block_group_geoid"]
    n_census_match = bg_data.filter(
        pl.any_horizontal(pl.col(c).is_not_null() for c in census_nonkey),
    ).height
    n_bg_total = bg_data.height
    log.info(
        "Census join: matched %d/%d BGs (%.1f%%)",
        n_census_match,
        n_bg_total,
        n_census_match / n_bg_total * 100 if n_bg_total else 0,
    )

    # -- Detect predictors ----------------------------------------------------
    predictors, excluded_null = detect_predictors(census, mode=args.predictors)
    if not predictors:
        log.error("No usable predictors found (mode='%s').", args.predictors)
        return 1
    log.info("Predictors (%d): %s", len(predictors), predictors)
    if excluded_null:
        log.info("Excluded (all null): %s", excluded_null)

    # -- Filter: min obs + complete case --------------------------------------
    n_before_filter = bg_data.height
    bg_data = bg_data.filter(pl.col("n_household_months") >= args.min_obs_per_bg)
    n_after_min_obs = bg_data.height
    log.info(
        "Min-obs filter (%d): %d -> %d",
        args.min_obs_per_bg,
        n_before_filter,
        n_after_min_obs,
    )

    bg_data = bg_data.drop_nulls(subset=predictors)
    n_complete = bg_data.height
    log.info("Complete-case filter: %d -> %d", n_after_min_obs, n_complete)

    if n_complete == 0:
        log.error("No block groups remain after filtering.")
        return 1

    # -- Write regression dataset ---------------------------------------------
    dataset_path = args.output_dir / "regression_dataset_bg.parquet"
    bg_data.write_parquet(dataset_path)
    log.info("Wrote regression dataset: %s (%d rows)", dataset_path, bg_data.height)

    # -- Fit OLS models -------------------------------------------------------
    diff_sum_col = f"sum_{bill_diff_col}"
    month_fe_col = "month" if args.regression_level == "bg_month" else None

    results: dict[str, Any] = {}
    summary_texts: list[str] = []

    for y_col, label in [
        ("pct_savings_weighted", "model_1_pct_savings_weighted"),
        (diff_sum_col, "model_2_sum_bill_diff"),
    ]:
        log.info(
            "Fitting OLS: %s ~ %d predictors%s",
            y_col,
            len(predictors),
            " + month FE" if month_fe_col else "",
        )
        res = fit_ols(
            bg_data,
            y_col=y_col,
            predictors=predictors,
            label=label,
            month_fe_col=month_fe_col,
        )
        results[label] = {k: v for k, v in res.items() if k != "summary_text"}
        summary_texts.append(f"{'=' * 70}\n{label}: {y_col}\n{'=' * 70}\n{res['summary_text']}\n")
        log.info(
            "  %s: R2=%.4f, adj_R2=%.4f, F=%.2f, n=%d",
            label,
            res["r_squared"],
            res["adj_r_squared"],
            res["f_statistic"],
            res["n_obs"],
        )

    # -- Write results JSON ---------------------------------------------------
    results_path = args.output_dir / "regression_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Wrote regression results: %s", results_path)

    # -- Write summary text ---------------------------------------------------
    summary_path = args.output_dir / "regression_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_texts))
    log.info("Wrote regression summary: %s", summary_path)

    # -- Write metadata JSON --------------------------------------------------
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "regression_level": args.regression_level,
        "months_included": sorted(months_unique),
        "bills_path": str(args.bills),
        "crosswalk_path": str(args.crosswalk),
        "census_path": str(args.census),
        "crosswalk_metrics": crosswalk_metrics,
        "join_metrics": join_metrics,
        "crosswalk_exposure": {
            "n_bills_total": n_bills_total,
            "n_zip4_parse_dropped": n_null_zip4,
            "pct_zip4_parse_dropped": round(n_null_zip4 / n_bills_total * 100, 3) if n_bills_total else 0.0,
            "pct_crosswalk_no_bg_match": join_metrics["pct_dropped"],
        },
        "savings_column_used": savings_col,
        "savings_fallback_used": savings_fallback,
        "bill_diff_column_used": bill_diff_col,
        "bill_diff_fallback_used": bill_diff_fallback,
        "predictor_mode": args.predictors,
        "predictors_used": predictors,
        "predictors_excluded_all_null": excluded_null,
        "min_obs_per_bg": args.min_obs_per_bg,
        "max_crosswalk_drop_pct": args.max_crosswalk_drop_pct,
        "n_bg_before_filter": n_before_filter,
        "n_bg_after_min_obs": n_after_min_obs,
        "n_bg_complete_case": n_complete,
        "files_written": {
            "bg_month_outcomes": str(bg_month_path),
            "bg_annual_outcomes": str(bg_annual_path),
            "bg_season_outcomes": str(bg_season_path),
            "regression_dataset": str(dataset_path),
        },
    }
    metadata_path = args.output_dir / "regression_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Wrote metadata: %s", metadata_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
