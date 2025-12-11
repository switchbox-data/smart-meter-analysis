#!/usr/bin/env python3
"""
Stage 2: Block-Group-Level Regression of Cluster Composition

Goal
-----
Model how Census block-group demographics are associated with the composition
of household-day observations across load-profile clusters.

Unit of Analysis
----------------
Block-group x cluster counts of HOUSEHOLD-DAY observations (not households).

Data Flow
---------
1. Load household-day cluster assignments from Stage 1 (one row per household-day)
2. Join to Census block groups via ZIP+4 → block group crosswalk
3. Aggregate to block-group x cluster counts of household-day observations
4. Join block groups to Census demographics
5. Fit multinomial logistic regression:
   - Outcome: cluster
   - Predictors: demographics
   - Weights: n_obs (household-day count)

Outputs
-------
- regression_data_blockgroups.parquet
- regression_results_blockgroups.json
- statsmodels_summary.txt
- regression_report_blockgroups.txt

Usage
-----
    python stage2_blockgroup_regression.py \\
        --clusters data/clustering/results/cluster_assignments.parquet \\
        --crosswalk data/reference/2023_comed_zip4_census_crosswalk.txt \\
        --census-cache data/reference/census_17_2023.parquet \\
        --output-dir data/clustering/results/stage2_blockgroups
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from smart_meter_analysis.census import fetch_census_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Default predictors (for convergence and interpretability)
DEFAULT_PREDICTORS = [
    "Median_Household_Income",
    "Median_Age",
    "Urban_Percent",
    "Total_Households",
]


# =============================================================================
# 1. LOAD CLUSTER ASSIGNMENTS (HOUSEHOLD-DAY LEVEL)
# =============================================================================


def load_cluster_assignments_household_day(path: Path) -> tuple[pl.DataFrame, dict]:
    """
    Load household-day cluster assignments.

    Returns the raw Stage 1 output: one row per (household, day) with cluster label.

    I still compute "dominance" statistics for reporting purposes, but the
    returned DataFrame keeps all household-day rows.

    Returns
    -------
    df : pl.DataFrame
        One row per household-day with columns:
        - account_identifier
        - zip_code
        - date (if present)
        - cluster

    dominance_stats : dict
        Summary statistics on how consistently households stay in one cluster
        (for reporting/interpretation, not used in regression)
    """
    logger.info("Loading cluster assignments from %s", path)
    raw = pl.read_parquet(path)

    required = ["account_identifier", "zip_code", "cluster"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"cluster_assignments missing required columns: {missing}")

    n_household_days = len(raw)
    n_households = raw["account_identifier"].n_unique()
    n_clusters = raw["cluster"].n_unique()

    logger.info(
        "  Loaded: %s household-day observations, %s households, %s clusters",
        f"{n_household_days:,}",
        f"{n_households:,}",
        n_clusters,
    )

    # This doesn't affect the regression - just useful context
    dominance_stats = _compute_dominance_stats(raw)

    logger.info(
        "  Dominance stats: mean=%.1f%%, median=%.1f%%, >50%%: %.1f%% of households",
        dominance_stats["dominance_mean"] * 100,
        dominance_stats["dominance_median"] * 100,
        dominance_stats["pct_above_50"],
    )

    return raw, dominance_stats


def _compute_dominance_stats(df: pl.DataFrame) -> dict:
    """
    Compute how consistently each household stays in one cluster.

    For each household:
    - dominance = (days in most frequent cluster) / (total days)

    Returns summary statistics across all households.
    """
    # Count days per (household, cluster)
    counts = df.group_by(["account_identifier", "cluster"]).agg(pl.len().alias("days_in_cluster"))

    # Total days per household
    totals = counts.group_by("account_identifier").agg(pl.col("days_in_cluster").sum().alias("n_days"))

    # Max days in any single cluster per household
    max_days = counts.group_by("account_identifier").agg(pl.col("days_in_cluster").max().alias("max_days_in_cluster"))

    # Compute dominance
    dominance_df = max_days.join(totals, on="account_identifier").with_columns(
        (pl.col("max_days_in_cluster") / pl.col("n_days")).alias("dominance")
    )

    dominance_values = dominance_df["dominance"].to_numpy()

    return {
        "n_households": len(dominance_df),
        "dominance_mean": float(dominance_values.mean()),
        "dominance_median": float(np.median(dominance_values)),
        "dominance_std": float(dominance_values.std()),
        "dominance_min": float(dominance_values.min()),
        "dominance_max": float(dominance_values.max()),
        "pct_above_50": float((dominance_values > 0.5).mean() * 100),
        "pct_above_67": float((dominance_values > 0.67).mean() * 100),
        "pct_above_80": float((dominance_values > 0.8).mean() * 100),
    }


# =============================================================================
# 2. CROSSWALK AND BLOCK GROUP MAPPING
# =============================================================================


def load_crosswalk(crosswalk_path: Path, zip_codes: list[str]) -> pl.DataFrame:
    """
    Load ZIP+4 → Census block-group crosswalk for the ZIP+4s in our data.

    Also runs a diagnostic to detect fan-out (ZIP+4 mapping to multiple block groups).
    """
    logger.info("Loading crosswalk from %s", crosswalk_path)

    crosswalk = (
        pl.scan_csv(crosswalk_path, separator="\t", infer_schema_length=10000)
        .with_columns([
            (pl.col("Zip").cast(pl.Utf8).str.zfill(5) + "-" + pl.col("Zip4").cast(pl.Utf8).str.zfill(4)).alias(
                "zip_code"
            ),
            pl.col("CensusKey2023").cast(pl.Utf8).str.zfill(15).str.slice(0, 12).alias("block_group_geoid"),
        ])
        .filter(pl.col("zip_code").is_in(zip_codes))
        .select(["zip_code", "block_group_geoid"])
        .collect()
    )

    logger.info(
        "  Matched %s of %s ZIP+4 codes",
        f"{crosswalk['zip_code'].n_unique():,}",
        f"{len(set(zip_codes)):,}",
    )

    if crosswalk.is_empty():
        logger.warning("  Crosswalk is empty after filtering for sample ZIP+4s.")
        return crosswalk

    # Fan-out diagnostic
    fanout = crosswalk.group_by("zip_code").agg(pl.n_unique("block_group_geoid").alias("n_block_groups"))
    max_fanout = int(fanout["n_block_groups"].max())

    if max_fanout > 1:
        fanout_summary = fanout.group_by("n_block_groups").agg(pl.len().alias("n_zip4")).sort("n_block_groups")
        logger.warning(
            "  WARNING: ZIP+4 → block-group crosswalk has fan-out (some ZIP+4s map to multiple block groups):\n%s",
            fanout_summary,
        )
    else:
        logger.info("  Crosswalk is 1-to-1: each ZIP+4 maps to exactly one block group.")

    return crosswalk


def attach_block_groups_to_household_days(
    household_days: pl.DataFrame,
    crosswalk: pl.DataFrame,
) -> pl.DataFrame:
    """
    Attach block-group GEOIDs to household-day observations via ZIP+4.

    Input: one row per household-day
    Output: one row per household-day with block_group_geoid attached
    """
    logger.info("Joining household-day observations to block groups...")

    df = household_days.join(crosswalk, on="zip_code", how="left")

    n_before = len(df)
    missing = df.filter(pl.col("block_group_geoid").is_null()).height

    if missing > 0:
        pct = missing / n_before * 100
        logger.warning("  %s (%.1f%%) observations missing block_group - dropping", f"{missing:,}", pct)
        df = df.filter(pl.col("block_group_geoid").is_not_null())

    logger.info(
        "  %s household-day observations across %s block groups",
        f"{len(df):,}",
        f"{df['block_group_geoid'].n_unique():,}",
    )

    return df


# =============================================================================
# 3. AGGREGATE TO BLOCK-GROUP x CLUSTER COUNTS
# =============================================================================


def aggregate_blockgroup_cluster_counts(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate household-day observations to block-group x cluster counts.

    Input: one row per household-day with columns:
        - account_identifier
        - block_group_geoid
        - cluster

    Output: one row per (block_group_geoid, cluster) with:
        - n_obs           : count of household-day observations
        - n_households    : count of distinct households (for context)
        - total_obs       : total household-day observations in the block group
        - total_households: total distinct households in the block group
        - cluster_share   : n_obs / total_obs
    """
    logger.info("Aggregating to block-group x cluster counts (household-day units)...")

    # Counts per (block_group, cluster)
    counts = df.group_by(["block_group_geoid", "cluster"]).agg([
        pl.len().alias("n_obs"),
        pl.col("account_identifier").n_unique().alias("n_households"),
    ])

    # Totals per block_group
    totals = df.group_by("block_group_geoid").agg([
        pl.len().alias("total_obs"),
        pl.col("account_identifier").n_unique().alias("total_households"),
    ])

    # Merge and compute cluster shares
    bg_counts = counts.join(totals, on="block_group_geoid", how="left").with_columns(
        (pl.col("n_obs") / pl.col("total_obs")).alias("cluster_share")
    )

    logger.info(
        "  Created %s (block_group, cluster) rows across %s block groups",
        f"{len(bg_counts):,}",
        f"{bg_counts['block_group_geoid'].n_unique():,}",
    )
    logger.info(
        "  Total observations: %s, Total households: %s",
        f"{bg_counts['n_obs'].sum():,}",
        f"{totals['total_households'].sum():,}",
    )

    return bg_counts


# =============================================================================
# 4. CENSUS DATA
# =============================================================================


def fetch_or_load_census(
    cache_path: Path,
    state_fips: str = "17",
    acs_year: int = 2023,
    force_fetch: bool = False,
) -> pl.DataFrame:
    """Fetch Census data from API or load from cache."""
    if cache_path.exists() and not force_fetch:
        logger.info(f"Loading Census data from cache: {cache_path}")
        return pl.read_parquet(cache_path)

    logger.info("Fetching Census data from API (state=%s, year=%s)...", state_fips, acs_year)

    census_df = fetch_census_data(state_fips=state_fips, acs_year=acs_year)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    census_df.write_parquet(cache_path)
    logger.info("  Cached Census data to %s", cache_path)

    return census_df


def attach_census_to_blockgroups(bg_counts: pl.DataFrame, census_df: pl.DataFrame) -> pl.DataFrame:
    """Attach Census demographics to block-group cluster counts."""
    logger.info("Joining Census data to block-group counts...")

    census_df = census_df.with_columns(pl.col("GEOID").cast(pl.Utf8).str.zfill(12).alias("block_group_geoid"))

    demo = bg_counts.join(census_df, on="block_group_geoid", how="left")

    n_before = len(demo)
    missing = demo.filter(pl.col("GEOID").is_null()).height

    if missing > 0:
        pct = missing / n_before * 100
        logger.warning("  %s (%.1f%%) rows missing Census data - dropping", f"{missing:,}", pct)
        demo = demo.filter(pl.col("GEOID").is_not_null())

    logger.info("  Demographics attached for %s block groups", f"{demo['block_group_geoid'].n_unique():,}")

    return demo


# =============================================================================
# 5. PREPARE REGRESSION DATA
# =============================================================================


def prepare_regression_dataset(
    demo_df: pl.DataFrame,
    predictors: list[str],
    min_obs_per_bg: int = 50,
    min_nonzero_clusters_per_bg: int = 2,
) -> tuple[pl.DataFrame, list[str]]:
    """
    Prepare block-group x cluster dataset for regression.

    Filters:
    - Block groups with fewer than min_obs_per_bg household-day observations
    - Block groups with fewer than min_nonzero_clusters_per_bg clusters represented
    """
    logger.info("Preparing regression dataset...")

    # Filter by minimum observations (household-days)
    df = demo_df.filter(pl.col("total_obs") >= min_obs_per_bg)
    logger.info(
        "  After min_obs filter (>=%d): %s block groups",
        min_obs_per_bg,
        f"{df['block_group_geoid'].n_unique():,}",
    )

    # Require block groups to have multiple clusters represented
    nonzero_counts = (
        df.filter(pl.col("n_obs") > 0).group_by("block_group_geoid").agg(pl.len().alias("n_nonzero_clusters"))
    )

    df = (
        df.join(nonzero_counts, on="block_group_geoid", how="left")
        .filter(pl.col("n_nonzero_clusters") >= min_nonzero_clusters_per_bg)
        .drop("n_nonzero_clusters")
    )

    logger.info(
        "  After cluster diversity filter (>=%d clusters): %s block groups",
        min_nonzero_clusters_per_bg,
        f"{df['block_group_geoid'].n_unique():,}",
    )

    # Filter predictors
    available_predictors: list[str] = []
    for p in predictors:
        if p not in df.columns:
            logger.warning("  Predictor not found: %s", p)
            continue
        null_rate = df[p].null_count() / len(df)
        if null_rate > 0.5:
            logger.warning("  Predictor %s has %.0f%% nulls - excluding", p, null_rate * 100)
            continue
        available_predictors.append(p)

    logger.info("  Using %d predictors: %s", len(available_predictors), available_predictors)

    if not available_predictors:
        raise ValueError("No valid predictors available")

    logger.info(
        "  Final dataset: %s rows, %s block groups, %s clusters",
        f"{len(df):,}",
        f"{df['block_group_geoid'].n_unique():,}",
        df["cluster"].n_unique(),
    )

    return df, available_predictors


# =============================================================================
# 6. MULTINOMIAL REGRESSION
# =============================================================================


def run_multinomial_regression(
    reg_df: pl.DataFrame,
    predictors: list[str],
    outcome: str = "cluster",
    weight_col: str = "n_obs",
    standardize: bool = False,
) -> dict[str, object]:
    """
    Run multinomial logistic regression with statsmodels.

    Parameters
    ----------
    reg_df : pl.DataFrame
        Long-form data, one row per (block_group_geoid, cluster).
    predictors : list[str]
        Names of predictor columns.
    outcome : str, default "cluster"
        Name of the outcome column.
    weight_col : str, default "n_obs"
        Column providing frequency weights. Default is n_obs (household-day
        observations), which weights by the number of household-day profiles
        in each block-group x cluster cell.
    standardize : bool, default False
        If True, standardize predictors before regression.
    """
    logger.info("Running multinomial logistic regression...")
    logger.info("  Weighting by: %s (household-day observations)", weight_col)

    # Extract arrays
    X = reg_df.select(predictors).to_numpy().astype(np.float64)
    y = reg_df.get_column(outcome).to_numpy()
    weights = reg_df.get_column(weight_col).to_numpy().astype(np.float64)

    # Drop rows with NaN in predictors
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.sum() > 0:
        logger.warning("  Dropping %s rows with NaN predictors", f"{nan_mask.sum():,}")
        X, y, weights = X[~nan_mask], y[~nan_mask], weights[~nan_mask]

    if len(X) == 0:
        raise ValueError("No observations remaining after dropping NaN rows.")

    n_block_groups = reg_df.filter(~pl.any_horizontal(pl.col(predictors).is_null()))["block_group_geoid"].n_unique()

    # Standardize or use raw units
    if standardize:
        logger.info("  Standardizing predictors...")
        scaler = StandardScaler()
        X_transformed = scaler.fit_transform(X)
    else:
        logger.info("  Using raw predictor units (no standardization).")
        X_transformed = X

    # Add intercept
    X_with_const = sm.add_constant(X_transformed)

    # Expand rows by integer weights
    weight_ints = np.maximum(np.round(weights).astype(int), 1)
    X_expanded = np.repeat(X_with_const, weight_ints, axis=0)
    y_expanded = np.repeat(y, weight_ints)

    logger.info(
        "  Training on %s expanded rows (%s block groups, %s total household-day obs)",
        f"{len(X_expanded):,}",
        n_block_groups,
        f"{int(weights.sum()):,}",
    )

    model = sm.MNLogit(y_expanded, X_expanded)
    result = model.fit(method="newton", maxiter=100, disp=False)

    # Extract results
    classes = sorted(np.unique(y).tolist())
    baseline = classes[0]
    param_names = ["const", *predictors]

    coefficients = {}
    std_errors = {}
    p_values = {}
    odds_ratios = {}

    for i, cls in enumerate(classes[1:]):
        key = f"cluster_{cls}"
        coefficients[key] = {name: float(result.params[j, i]) for j, name in enumerate(param_names)}
        std_errors[key] = {name: float(result.bse[j, i]) for j, name in enumerate(param_names)}
        p_values[key] = {name: float(result.pvalues[j, i]) for j, name in enumerate(param_names)}
        odds_ratios[key] = {name: float(np.exp(result.params[j, i])) for j, name in enumerate(param_names)}

    # Baseline cluster
    baseline_key = f"cluster_{baseline}"
    coefficients[baseline_key] = dict.fromkeys(param_names, 0.0)
    std_errors[baseline_key] = dict.fromkeys(param_names, 0.0)
    p_values[baseline_key] = dict.fromkeys(param_names, 1.0)
    odds_ratios[baseline_key] = dict.fromkeys(param_names, 1.0)

    logger.info("  Baseline cluster: %s", baseline)
    logger.info("  Converged: %s", result.mle_retvals.get("converged", True))
    logger.info("  Pseudo R²: %.4f", result.prsquared)

    return {
        "n_rows": len(X),
        "n_expanded_rows": len(X_expanded),
        "n_block_groups": int(n_block_groups),
        "n_clusters": len(classes),
        "n_predictors": len(predictors),
        "total_household_day_obs": int(weights.sum()),
        "clusters": classes,
        "baseline_cluster": int(baseline),
        "predictors": predictors,
        "weight_col": weight_col,
        "coefficients": coefficients,
        "std_errors": std_errors,
        "p_values": p_values,
        "odds_ratios": odds_ratios,
        "converged": bool(result.mle_retvals.get("converged", True)),
        "pseudo_r2": float(result.prsquared),
        "llf": float(result.llf),
        "model_summary": result.summary().as_text(),
    }


# =============================================================================
# 7. REPORT GENERATION
# =============================================================================


def generate_report(
    results: dict[str, object],
    cluster_distribution: pl.DataFrame,
    dominance_stats: dict,
    output_path: Path,
) -> None:
    """Generate human-readable summary."""
    lines = [
        "=" * 70,
        "STAGE 2: BLOCK-GROUP MULTINOMIAL REGRESSION RESULTS",
        "=" * 70,
        "",
        "ANALYSIS UNIT: HOUSEHOLD-DAY OBSERVATIONS",
        "-" * 70,
        "Each row in the regression represents a (block_group, cluster) pair,",
        "weighted by the number of household-day observations in that cell.",
        "",
        "MODEL SUMMARY",
        "-" * 70,
        f"Block groups: {results['n_block_groups']:,}",
        f"Rows (block_group x cluster): {results['n_rows']:,}",
        f"Total household-day observations: {results['total_household_day_obs']:,}",
        f"Clusters: {results['n_clusters']}",
        f"Predictors: {results['n_predictors']}",
        f"Weight column: {results['weight_col']}",
        f"Baseline cluster: {results['baseline_cluster']}",
        f"Pseudo R²: {results['pseudo_r2']:.4f}",
        f"Converged: {results['converged']}",
        "",
        "HOUSEHOLD CLUSTER CONSISTENCY (for interpretation context)",
        "-" * 70,
        "How consistently do households stay in one cluster across sampled days?",
        "(This doesn't affect the regression - just useful context.)",
        "",
        f"  Households: {dominance_stats['n_households']:,}",
        f"  Mean dominance: {dominance_stats['dominance_mean'] * 100:.1f}%",
        f"  Median dominance: {dominance_stats['dominance_median'] * 100:.1f}%",
        f"  Households >50% in one cluster: {dominance_stats['pct_above_50']:.1f}%",
        f"  Households >67% in one cluster: {dominance_stats['pct_above_67']:.1f}%",
        f"  Households >80% in one cluster: {dominance_stats['pct_above_80']:.1f}%",
        "",
        "CLUSTER DISTRIBUTION (by household-day observations)",
        "-" * 70,
    ]

    for row in cluster_distribution.iter_rows(named=True):
        lines.append(f"  Cluster {row['cluster']}: {row['n_obs']:,} obs ({row['pct']:.1f}%)")

    lines.extend([
        "",
        "TOP PREDICTORS BY CLUSTER (by |coefficient|, *=p<0.05)",
        "-" * 70,
    ])

    for cluster in results["clusters"]:
        key = f"cluster_{cluster}"
        if cluster == results["baseline_cluster"]:
            lines.append(f"\nCluster {cluster} (BASELINE)")
            continue

        lines.append(f"\nCluster {cluster} vs baseline:")
        coefs = results["coefficients"][key]
        pvals = results["p_values"][key]
        ors = results["odds_ratios"][key]

        sorted_preds = sorted(
            [(p, coefs[p]) for p in results["predictors"]],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]

        for pred, coef in sorted_preds:
            star = "*" if pvals[pred] < 0.05 else ""
            arrow = "↑" if coef > 0 else "↓"
            lines.append(f"  {arrow} {pred}: OR={ors[pred]:.2f}, coef={coef:.3f}, p={pvals[pred]:.3f}{star}")

    lines.append("\n" + "=" * 70)

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    logger.info("Report saved to %s", output_path)
    print("\n" + text)


# =============================================================================
# 8. CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2: Block-group-level regression using household-day units.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--clusters", type=Path, required=True, help="cluster_assignments.parquet")
    parser.add_argument("--crosswalk", type=Path, required=True, help="ZIP+4 → block-group crosswalk")
    parser.add_argument(
        "--census-cache",
        type=Path,
        default=Path("data/reference/census_17_2023.parquet"),
    )
    parser.add_argument("--fetch-census", action="store_true", help="Force re-fetch Census data")
    parser.add_argument("--state-fips", default="17")
    parser.add_argument("--acs-year", type=int, default=2023)
    parser.add_argument(
        "--min-obs-per-bg",
        type=int,
        default=50,
        help="Minimum household-day observations per block group (default: 50)",
    )
    parser.add_argument(
        "--min-nonzero-clusters-per-bg",
        type=int,
        default=2,
        help="Minimum clusters represented per block group (default: 2)",
    )
    parser.add_argument("--predictors", nargs="+", default=DEFAULT_PREDICTORS, help="Predictor columns")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clustering/results/stage2_blockgroups"),
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize predictors before regression (default: use raw units).",
    )

    args = parser.parse_args()

    if not args.clusters.exists():
        logger.error("Cluster assignments not found: %s", args.clusters)
        return 1
    if not args.crosswalk.exists():
        logger.error("Crosswalk not found: %s", args.crosswalk)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 2: BLOCK-GROUP REGRESSION (HOUSEHOLD-DAY UNITS)")
    print("=" * 70)

    # 1. Load household-day cluster assignments (NO dominant cluster reduction)
    household_days, dominance_stats = load_cluster_assignments_household_day(args.clusters)

    # 2. Load crosswalk and attach block groups
    zip_codes = household_days["zip_code"].unique().to_list()
    crosswalk = load_crosswalk(args.crosswalk, zip_codes)
    household_days_bg = attach_block_groups_to_household_days(household_days, crosswalk)

    # 3. Aggregate to block-group x cluster counts
    bg_counts = aggregate_blockgroup_cluster_counts(household_days_bg)

    # 4. Load Census and attach demographics
    census_df = fetch_or_load_census(
        cache_path=args.census_cache,
        state_fips=args.state_fips,
        acs_year=args.acs_year,
        force_fetch=args.fetch_census,
    )
    logger.info("  Census: %s block groups, %s columns", f"{len(census_df):,}", len(census_df.columns))

    demo_df = attach_census_to_blockgroups(bg_counts, census_df)

    # 5. Prepare regression dataset
    reg_df, predictors = prepare_regression_dataset(
        demo_df=demo_df,
        predictors=args.predictors,
        min_obs_per_bg=args.min_obs_per_bg,
        min_nonzero_clusters_per_bg=args.min_nonzero_clusters_per_bg,
    )

    if reg_df.is_empty():
        logger.error("No data after filtering")
        return 1

    # Save regression dataset
    reg_df.write_parquet(args.output_dir / "regression_data_blockgroups.parquet")
    logger.info("Saved regression data to %s", args.output_dir / "regression_data_blockgroups.parquet")

    # 6. Run regression (weighted by n_obs = household-day counts)
    results = run_multinomial_regression(
        reg_df=reg_df,
        predictors=predictors,
        outcome="cluster",
        weight_col="n_obs",  # household-day observations
        standardize=args.standardize,
    )

    # Add dominance stats to results for reference
    results["dominance_stats"] = dominance_stats

    # Save results
    model_summary = results.pop("model_summary")
    with open(args.output_dir / "regression_results_blockgroups.json", "w") as f:
        json.dump(results, f, indent=2)
    (args.output_dir / "statsmodels_summary.txt").write_text(model_summary)

    # Generate report
    cluster_dist = (
        reg_df.group_by("cluster")
        .agg(pl.col("n_obs").sum())
        .sort("cluster")
        .with_columns((pl.col("n_obs") / pl.col("n_obs").sum() * 100).alias("pct"))
    )

    generate_report(
        results,
        cluster_dist,
        dominance_stats,
        args.output_dir / "regression_report_blockgroups.txt",
    )

    print(f"\nOutputs saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
