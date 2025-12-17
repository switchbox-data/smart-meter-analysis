#!/usr/bin/env python3
"""
Stage 2: Block-Group-Level Log-Ratio Regression of Cluster Composition (HOUSEHOLD-DAY UNITS)

Goal
-----
Model how Census block-group demographics are associated with the *composition*
of household-day observations across load-profile clusters, using log-ratio
regression (ALR / additive log-ratio).

Unit of Analysis
----------------
One row per Census Block Group

Data Flow
---------
1. Load household-day cluster assignments from Stage 1 (one row per household-day)
2. Join to Census block groups via ZIP+4 → block group crosswalk (1-to-1 enforced)
3. Aggregate to block-group-level cluster composition (wide format)
4. Join block groups to Census demographics
5. Create smoothed proportions and log-ratios vs a baseline cluster:
      y_k = log(p_k / p_base)
6. Fit separate WLS regressions for each non-baseline cluster:
      y_k ~ demographics   with weights = total_obs (household-day count)
7. Fit OLS models (robustness check, unweighted)

Outputs
-------
- regression_data_blockgroups_wide.parquet
- regression_results_logratio_blockgroups.json
- statsmodels_summaries_wls.txt
- statsmodels_summaries_ols.txt
- regression_report_logratio_blockgroups.txt

Usage
-----
    python stage2_logratio_regression.py \
        --clusters data/clustering/results/cluster_assignments.parquet \
        --crosswalk data/reference/2023_comed_zip4_census_crosswalk.txt \
        --census-cache data/reference/census_17_2023.parquet \
        --output-dir data/clustering/results/stage2_blockgroups_logratio \
        --baseline-cluster 1
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

DEFAULT_PREDICTORS = [
    "Owner_Occupied_Pct",
    "Average_Household_Size",
    "Old_Building_Pct",
    "Heat_Electric_Pct",
    "Median_Household_Income",
    "Urban_Percent",
]


def load_cluster_assignments_household_day(path: Path) -> tuple[pl.DataFrame, dict]:
    """
    Load household-day cluster assignments.

    Returns the raw Stage 1 output: one row per (household, day) with cluster label.

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
    counts = df.group_by(["account_identifier", "cluster"]).agg(pl.len().alias("days_in_cluster"))
    totals = counts.group_by("account_identifier").agg(pl.col("days_in_cluster").sum().alias("n_days"))
    max_days = counts.group_by("account_identifier").agg(pl.col("days_in_cluster").max().alias("max_days_in_cluster"))

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


def load_crosswalk_one_to_one(crosswalk_path: Path, zip_codes: list[str]) -> pl.DataFrame:
    """
    Load ZIP+4 → Census block-group crosswalk with deterministic 1-to-1 mapping.

    When fan-out exists (ZIP+4 maps to multiple block groups),
    choose smallest GEOID per ZIP+4 to avoid double-counting household-day observations.

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

    # Check for fan-out
    fanout = crosswalk.group_by("zip_code").agg(pl.n_unique("block_group_geoid").alias("n_block_groups"))
    max_fanout = int(fanout["n_block_groups"].max())

    if max_fanout > 1:
        n_fanout = fanout.filter(pl.col("n_block_groups") > 1).height
        pct_fanout = (n_fanout / len(fanout)) * 100

        logger.warning(
            "  ZIP+4 fan-out detected: %s ZIP+4s (%.1f%%) map to multiple block groups (max=%d per ZIP+4)",
            f"{n_fanout:,}",
            pct_fanout,
            max_fanout,
        )
        logger.warning("  Applying deterministic 1-to-1 mapping: selecting smallest GEOID per ZIP+4")
        logger.warning("  This prevents double-counting household-day observations")

        # Deterministic resolution: smallest GEOID per ZIP+4
        crosswalk = (
            crosswalk.sort(["zip_code", "block_group_geoid"])
            .group_by("zip_code")
            .agg(pl.col("block_group_geoid").first())
        )

        logger.info(
            "  After 1-to-1 resolution: %s ZIP+4 codes → %s unique mappings",
            f"{len(crosswalk):,}",
            f"{len(crosswalk):,}",
        )
    else:
        logger.info("  Crosswalk is already 1-to-1: each ZIP+4 maps to exactly one block group.")

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


def aggregate_blockgroup_cluster_composition(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate household-day observations to block-group-level cluster composition (wide).

    Output: one row per block_group_geoid with:
        - total_obs
        - total_households
        - n_cluster_<k>
        - p_cluster_<k>
    """
    logger.info("Aggregating to block-group cluster composition (wide; household-day units)...")

    totals = df.group_by("block_group_geoid").agg([
        pl.len().alias("total_obs"),
        pl.col("account_identifier").n_unique().alias("total_households"),
    ])

    counts_long = df.group_by(["block_group_geoid", "cluster"]).agg(pl.len().alias("n_obs"))

    counts_wide = (
        counts_long.with_columns(pl.col("cluster").cast(pl.Utf8))
        .pivot(
            values="n_obs",
            index="block_group_geoid",
            columns="cluster",
            aggregate_function="first",
        )
        .fill_null(0)
    )

    cluster_cols = [c for c in counts_wide.columns if c != "block_group_geoid"]
    counts_wide = counts_wide.rename({c: f"n_cluster_{c}" for c in cluster_cols})

    out = totals.join(counts_wide, on="block_group_geoid", how="left").fill_null(0)

    n_cols = [c for c in out.columns if c.startswith("n_cluster_")]
    out = out.with_columns([
        (pl.col(c) / pl.col("total_obs")).alias(c.replace("n_cluster_", "p_cluster_")) for c in n_cols
    ])

    logger.info(
        "  Created %s block-group rows; total obs=%s; total households=%s",
        f"{len(out):,}",
        f"{int(out['total_obs'].sum()):,}",
        f"{int(out['total_households'].sum()):,}",
    )
    return out


def fetch_or_load_census(
    cache_path: Path,
    state_fips: str = "17",
    acs_year: int = 2023,
    force_fetch: bool = False,
) -> pl.DataFrame:
    """Fetch Census data from API or load from cache."""
    if cache_path.exists() and not force_fetch:
        logger.info("Loading Census data from cache: %s", cache_path)
        return pl.read_parquet(cache_path)

    logger.info("Fetching Census data from API (state=%s, year=%s)...", state_fips, acs_year)

    census_df = fetch_census_data(state_fips=state_fips, acs_year=acs_year)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    census_df.write_parquet(cache_path)
    logger.info("  Cached Census data to %s", cache_path)

    return census_df


def create_derived_variables(census_df: pl.DataFrame) -> pl.DataFrame:
    """Create derived percentage variables from raw Census counts."""
    logger.info("Creating derived variables...")

    df = census_df.with_columns([
        (pl.col("Owner_Occupied") / pl.col("Occupied_Housing_Units") * 100).alias("Owner_Occupied_Pct"),
        (pl.col("Heat_Electric") / pl.col("Total_Households") * 100).alias("Heat_Electric_Pct"),
        (
            (
                pl.col("Built_1960_1969")
                + pl.col("Built_1950_1959")
                + pl.col("Built_1940_1949")
                + pl.col("Built_1939_Earlier")
            )
            / pl.col("Total_Housing_Units")
            * 100
        ).alias("Old_Building_Pct"),
    ])

    df = df.with_columns([
        pl.when(pl.col("Owner_Occupied_Pct").is_nan())
        .then(None)
        .otherwise(pl.col("Owner_Occupied_Pct"))
        .alias("Owner_Occupied_Pct"),
        pl.when(pl.col("Heat_Electric_Pct").is_nan())
        .then(None)
        .otherwise(pl.col("Heat_Electric_Pct"))
        .alias("Heat_Electric_Pct"),
        pl.when(pl.col("Old_Building_Pct").is_nan())
        .then(None)
        .otherwise(pl.col("Old_Building_Pct"))
        .alias("Old_Building_Pct"),
    ])

    return df


def attach_census_to_blockgroups(bg_comp: pl.DataFrame, census_df: pl.DataFrame) -> pl.DataFrame:
    """Attach Census demographics to block-group composition (wide)."""
    logger.info("Joining Census data to block-group composition...")

    census_df = census_df.with_columns(pl.col("GEOID").cast(pl.Utf8).str.zfill(12).alias("block_group_geoid"))

    demo = bg_comp.join(census_df, on="block_group_geoid", how="left")

    n_before = len(demo)
    missing = demo.filter(pl.col("GEOID").is_null()).height

    if missing > 0:
        pct = missing / n_before * 100
        logger.warning("  %s (%.1f%%) rows missing Census data - dropping", f"{missing:,}", pct)
        demo = demo.filter(pl.col("GEOID").is_not_null())

    logger.info("  Demographics attached for %s block groups", f"{demo['block_group_geoid'].n_unique():,}")

    return demo


def prepare_regression_dataset_wide(
    demo_df: pl.DataFrame,
    predictors: list[str],
    min_obs_per_bg: int = 50,
) -> tuple[pl.DataFrame, list[str]]:
    """
    Prepare block-group (wide) dataset for log-ratio regression.

    Filters:
    - Block groups with fewer than min_obs_per_bg household-day observations
    - Drops predictors with too many nulls
    - Drops rows with any null predictor values (conservative / statsmodels-friendly)
    """
    logger.info("Preparing regression dataset (wide)...")

    df = demo_df.filter(pl.col("total_obs") >= min_obs_per_bg)
    logger.info(
        "  After min_obs filter (>=%d): %s block groups",
        min_obs_per_bg,
        f"{df['block_group_geoid'].n_unique():,}",
    )

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

    if not available_predictors:
        raise ValueError("No valid predictors available")

    # Drop rows with any null predictor values
    df = df.filter(~pl.any_horizontal(pl.col(available_predictors).is_null()))
    logger.info(
        "  After dropping rows with null predictors: %s block groups",
        f"{df['block_group_geoid'].n_unique():,}",
    )

    logger.info("  Using %d predictors: %s", len(available_predictors), available_predictors)

    return df, available_predictors


def choose_baseline_cluster_from_household_days(household_days: pl.DataFrame) -> str:
    """
    Choose baseline cluster as the most frequent cluster by household-day observations.
    Returns as string (to match pivot-derived cluster column suffixes).
    """
    dist = household_days.group_by("cluster").agg(pl.len().alias("n")).sort("n", descending=True)
    baseline = dist["cluster"][0]
    logger.info("  Auto-selected baseline: cluster %s (most frequent by household-days)", baseline)
    return str(baseline)


def add_smoothed_proportions_and_logratios(
    df: pl.DataFrame,
    baseline_cluster: str,
    alpha: float = 0.5,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """
    Add smoothed proportions and log-ratios vs baseline.

    Smoothing is applied at the count level:
        n_s_k = n_k + alpha
        total_s = total_obs + alpha*K
        p_s_k = n_s_k / total_s

    Then outcomes:
        log_ratio_k = log(p_s_k / p_s_base)
    """
    n_cols = sorted([c for c in df.columns if c.startswith("n_cluster_")])
    if not n_cols:
        raise ValueError("No n_cluster_<k> columns found. Did you run wide aggregation?")

    clusters = [c.replace("n_cluster_", "") for c in n_cols]
    if baseline_cluster not in clusters:
        raise ValueError(f"Baseline cluster {baseline_cluster} not found in clusters={clusters}")

    K = len(clusters)
    nonbase = [k for k in clusters if k != baseline_cluster]

    logger.info("Adding smoothed proportions and log-ratios (alpha=%.2f)...", alpha)
    logger.info("  Clusters: %s (K=%d)", clusters, K)
    logger.info("  Baseline: %s", baseline_cluster)
    logger.info("  Non-baseline: %s", nonbase)

    df2 = df.with_columns([(pl.col(f"n_cluster_{k}") + alpha).alias(f"n_s_{k}") for k in clusters]).with_columns(
        (pl.col("total_obs") + alpha * K).alias("total_obs_s")
    )

    df2 = df2.with_columns([(pl.col(f"n_s_{k}") / pl.col("total_obs_s")).alias(f"p_s_{k}") for k in clusters])

    df2 = df2.with_columns([
        (pl.col(f"p_s_{k}") / pl.col(f"p_s_{baseline_cluster}")).log().alias(f"log_ratio_{k}") for k in nonbase
    ])

    # Diagnostic: check for extreme log-ratios
    for k in nonbase:
        extreme_pos = df2.filter(pl.col(f"log_ratio_{k}") > 5).height
        extreme_neg = df2.filter(pl.col(f"log_ratio_{k}") < -5).height

        if extreme_pos > 0 or extreme_neg > 0:
            logger.warning(
                "  Cluster %s: %d block groups with log_ratio > 5, %d with log_ratio < -5",
                k,
                extreme_pos,
                extreme_neg,
            )
            logger.warning("    This suggests very imbalanced cluster distributions in some block groups")

    return df2, clusters, nonbase


def run_logratio_regressions(
    reg_df: pl.DataFrame,
    predictors: list[str],
    baseline_cluster: str,
    weight_col: str = "total_obs",
    standardize: bool = False,
    include_ols: bool = True,
) -> dict[str, object]:
    """
    Fit separate WLS models for each non-baseline cluster:
        log(p_k / p_base) ~ predictors
    with weights = total_obs (household-day count in block group).

    Also fits OLS models (unweighted) as robustness check.

    Interpretation:
        exp(beta) = multiplicative effect on the proportion ratio p_k/p_base
        for a 1-unit increase in the predictor.
    """
    logger.info("Running log-ratio regressions...")
    logger.info("  Baseline cluster: %s", baseline_cluster)
    logger.info("  Weighting by: %s", weight_col)
    logger.info("  OLS robustness check: %s", include_ols)

    logratio_cols = sorted([c for c in reg_df.columns if c.startswith("log_ratio_")])
    if not logratio_cols:
        raise ValueError("No log_ratio_<k> columns found. Did you call add_smoothed_proportions_and_logratios()?")

    X = reg_df.select(predictors).to_numpy().astype(np.float64)
    w = reg_df.get_column(weight_col).to_numpy().astype(np.float64)

    # Drop invalid rows (NaNs / infs / nonpositive weights)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(w) & (w > 0)
    if valid.sum() == 0:
        raise ValueError("No valid rows after filtering missing predictors / invalid weights.")

    X = X[valid]
    w = w[valid]

    scaler = None
    if standardize:
        logger.info("  Standardizing predictors...")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        logger.info("  Using raw predictor units (no standardization).")

    X = sm.add_constant(X)
    param_names = ["const", *predictors]

    results: dict[str, object] = {
        "n_block_groups": int(reg_df["block_group_geoid"].n_unique()),
        "n_rows": len(reg_df),
        "n_valid_rows": int(valid.sum()),
        "weight_col": weight_col,
        "baseline_cluster": baseline_cluster,
        "predictors": predictors,
        "standardize": bool(standardize),
        "models_wls": {},
        "models_ols": {},
    }

    if scaler is not None:
        results["standardization"] = {
            "type": "zscore",
            "means": {p: float(m) for p, m in zip(predictors, scaler.mean_)},
            "scales": {p: float(s) for p, s in zip(predictors, scaler.scale_)},
        }

    summaries_wls = []
    summaries_ols = []

    for col in logratio_cols:
        k = col.replace("log_ratio_", "")
        y = reg_df.get_column(col).to_numpy().astype(np.float64)[valid]

        if not np.isfinite(y).all():
            raise ValueError(f"Non-finite values in outcome {col}. Check smoothing / inputs.")

        # WLS model
        model_wls = sm.WLS(y, X, weights=w)
        fit_wls = model_wls.fit()

        coef_wls = {name: float(fit_wls.params[i]) for i, name in enumerate(param_names)}
        se_wls = {name: float(fit_wls.bse[i]) for i, name in enumerate(param_names)}
        pvals_wls = {name: float(fit_wls.pvalues[i]) for i, name in enumerate(param_names)}
        mult_wls = {name: float(np.exp(fit_wls.params[i])) for i, name in enumerate(param_names)}

        key = f"cluster_{k}_vs_{baseline_cluster}"
        results["models_wls"][key] = {
            "outcome": f"log(p_{k}/p_{baseline_cluster})",
            "nobs": int(fit_wls.nobs),
            "r2": float(fit_wls.rsquared),
            "adj_r2": float(fit_wls.rsquared_adj),
            "coefficients": coef_wls,
            "std_errors": se_wls,
            "p_values": pvals_wls,
            "multiplicative_effects": mult_wls,
        }

        summaries_wls.append(f"\n{'=' * 80}\nWLS: {key}\n{'=' * 80}\n{fit_wls.summary().as_text()}")
        logger.info("  WLS %s: R²=%.4f", key, float(fit_wls.rsquared))

        # OLS model (robustness check)
        if include_ols:
            model_ols = sm.OLS(y, X)
            fit_ols = model_ols.fit()

            coef_ols = {name: float(fit_ols.params[i]) for i, name in enumerate(param_names)}
            se_ols = {name: float(fit_ols.bse[i]) for i, name in enumerate(param_names)}
            pvals_ols = {name: float(fit_ols.pvalues[i]) for i, name in enumerate(param_names)}
            mult_ols = {name: float(np.exp(fit_ols.params[i])) for i, name in enumerate(param_names)}

            results["models_ols"][key] = {
                "outcome": f"log(p_{k}/p_{baseline_cluster})",
                "nobs": int(fit_ols.nobs),
                "r2": float(fit_ols.rsquared),
                "adj_r2": float(fit_ols.rsquared_adj),
                "coefficients": coef_ols,
                "std_errors": se_ols,
                "p_values": pvals_ols,
                "multiplicative_effects": mult_ols,
            }

            summaries_ols.append(f"\n{'=' * 80}\nOLS: {key}\n{'=' * 80}\n{fit_ols.summary().as_text()}")
            logger.info("  OLS %s: R²=%.4f", key, float(fit_ols.rsquared))

    results["all_model_summaries_wls"] = "\n".join(summaries_wls)
    if include_ols:
        results["all_model_summaries_ols"] = "\n".join(summaries_ols)

    return results


def generate_report_logratio(
    results: dict[str, object],
    dominance_stats: dict,
    cluster_dist: pl.DataFrame,
    output_path: Path,
) -> None:
    """Generate a human-readable report emphasizing log-ratio interpretation."""
    lines = [
        "=" * 80,
        "STAGE 2: BLOCK-GROUP LOG-RATIO REGRESSION RESULTS",
        "=" * 80,
        "",
        "ANALYSIS UNIT: BLOCK GROUPS (HOUSEHOLD-DAY COMPOSITION)",
        "-" * 80,
        "Each row is a block group, with household-day counts aggregated into",
        "cluster composition proportions. Outcomes are log-ratios vs a baseline:",
        "  y_k = log(p_k / p_baseline)",
        "",
        "Models are separate WLS regressions per non-baseline cluster, weighted by total_obs.",
        "OLS models (unweighted) are included as robustness checks.",
        "",
        "MODEL OVERVIEW",
        "-" * 80,
        f"Block groups (total): {results['n_block_groups']:,}",
        f"Block groups (valid): {results['n_valid_rows']:,}",
        f"Rows: {results['n_rows']:,}",
        f"Predictors: {len(results['predictors'])}",
        f"Weight column: {results['weight_col']}",
        f"Baseline cluster: {results['baseline_cluster']}",
        f"Standardized predictors: {results.get('standardize', False)}",
        "",
        "HOUSEHOLD CLUSTER CONSISTENCY (interpretation context)",
        "-" * 80,
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
        "CLUSTER DISTRIBUTION (by household-day observations, overall)",
        "-" * 80,
    ]

    for row in cluster_dist.iter_rows(named=True):
        lines.append(f"  Cluster {row['cluster']}: {row['n_obs']:,} obs ({row['pct']:.1f}%)")

    lines.extend([
        "",
        "TOP PREDICTORS BY MODEL (WLS; by |coef|; *=p<0.05)",
        "-" * 80,
        "Interpretation: exp(coef) multiplies the proportion ratio p_k/p_baseline",
        "for a 1-unit increase in the predictor (holding others constant).",
        "",
    ])

    models_wls = results["models_wls"]
    models_ols = results.get("models_ols", {})
    predictors = results["predictors"]

    for model_key in sorted(models_wls.keys()):
        m_wls = models_wls[model_key]
        coefs_wls = m_wls["coefficients"]
        pvals_wls = m_wls["p_values"]
        mult_wls = m_wls["multiplicative_effects"]

        lines.append(f"\n{model_key}")
        lines.append("-" * 80)
        lines.append(f"WLS R²={m_wls['r2']:.4f}, Adj R²={m_wls['adj_r2']:.4f}")

        if model_key in models_ols:
            m_ols = models_ols[model_key]
            lines.append(f"OLS R²={m_ols['r2']:.4f}, Adj R²={m_ols['adj_r2']:.4f} (robustness check)")

        lines.append("")

        sorted_preds = sorted(
            [(p, coefs_wls[p]) for p in predictors],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]

        for pred, coef in sorted_preds:
            star = "*" if pvals_wls[pred] < 0.05 else ""
            direction = "↑" if coef > 0 else "↓"

            line = f"  {direction} {pred:<30} WLS: mult={mult_wls[pred]:.3f}, coef={coef:>7.4f}, p={pvals_wls[pred]:.3g}{star}"

            # Show OLS comparison if available
            if model_key in models_ols:
                coef_ols = models_ols[model_key]["coefficients"][pred]
                diff = coef - coef_ols
                line += f"  | OLS: coef={coef_ols:>7.4f} (Δ={diff:>6.3f})"

            lines.append(line)

    lines.append("\n" + "=" * 80)
    lines.append("")
    lines.append("NOTES:")
    lines.append("- WLS models weight by total household-day observations per block group")
    lines.append("- OLS models are unweighted (robustness check)")
    lines.append("- Large WLS-OLS differences suggest results driven by large block groups")
    lines.append("- Multiplicative effect > 1.0 means predictor increases p_k/p_baseline")
    lines.append("- Multiplicative effect < 1.0 means predictor decreases p_k/p_baseline")
    lines.append("=" * 80)

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    logger.info("Report saved to %s", output_path)
    print("\n" + text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2: Block-group-level log-ratio regression using household-day units.",
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
    parser.add_argument("--predictors", nargs="+", default=DEFAULT_PREDICTORS, help="Predictor columns")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clustering/results/stage2_blockgroups_logratio"),
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize predictors before regression (default: use raw units).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Pseudocount smoothing parameter for proportions (default: 0.5)",
    )
    parser.add_argument(
        "--baseline-cluster",
        type=str,
        default=None,
        help="Baseline cluster label (default: most frequent cluster by household-day observations)",
    )
    parser.add_argument(
        "--no-ols",
        action="store_true",
        help="Skip OLS robustness check (only run WLS)",
    )

    args = parser.parse_args()

    if not args.clusters.exists():
        logger.error("Cluster assignments not found: %s", args.clusters)
        return 1
    if not args.crosswalk.exists():
        logger.error("Crosswalk not found: %s", args.crosswalk)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 2: BLOCK-GROUP LOG-RATIO REGRESSION (HOUSEHOLD-DAY UNITS)")
    print("=" * 80)

    household_days, dominance_stats = load_cluster_assignments_household_day(args.clusters)

    # Baseline cluster (string form to match wide column naming)
    baseline_cluster = args.baseline_cluster or choose_baseline_cluster_from_household_days(household_days)
    baseline_cluster = str(baseline_cluster)
    logger.info("Using baseline cluster: %s", baseline_cluster)

    zip_codes = household_days["zip_code"].unique().to_list()
    crosswalk = load_crosswalk_one_to_one(args.crosswalk, zip_codes)
    household_days_bg = attach_block_groups_to_household_days(household_days, crosswalk)

    bg_comp = aggregate_blockgroup_cluster_composition(household_days_bg)

    census_df = fetch_or_load_census(
        cache_path=args.census_cache,
        state_fips=args.state_fips,
        acs_year=args.acs_year,
        force_fetch=args.fetch_census,
    )
    logger.info("  Census: %s block groups, %s columns", f"{len(census_df):,}", len(census_df.columns))

    census_df = create_derived_variables(census_df)

    demo_df = attach_census_to_blockgroups(bg_comp, census_df)

    reg_df, predictors = prepare_regression_dataset_wide(
        demo_df=demo_df,
        predictors=args.predictors,
        min_obs_per_bg=args.min_obs_per_bg,
    )

    if reg_df.is_empty():
        logger.error("No data after filtering")
        return 1

    # Add smoothed proportions + log-ratios
    reg_df2, clusters, nonbase = add_smoothed_proportions_and_logratios(
        reg_df,
        baseline_cluster=baseline_cluster,
        alpha=args.alpha,
    )
    logger.info("Clusters detected: %s (baseline=%s, non-baseline=%s)", clusters, baseline_cluster, nonbase)

    # Save regression dataset
    reg_df2.write_parquet(args.output_dir / "regression_data_blockgroups_wide.parquet")
    logger.info("Saved regression data to %s", args.output_dir / "regression_data_blockgroups_wide.parquet")

    # Fit models
    results = run_logratio_regressions(
        reg_df=reg_df2,
        predictors=predictors,
        baseline_cluster=baseline_cluster,
        weight_col="total_obs",
        standardize=args.standardize,
        include_ols=not args.no_ols,
    )

    results["dominance_stats"] = dominance_stats
    results["alpha"] = float(args.alpha)
    results["k"] = len(clusters)
    results["clusters"] = clusters
    results["nonbaseline_clusters"] = nonbase

    # Write outputs
    all_summaries_wls = results.pop("all_model_summaries_wls")
    all_summaries_ols = results.pop("all_model_summaries_ols", None)

    with open(args.output_dir / "regression_results_logratio_blockgroups.json", "w") as f:
        json.dump(results, f, indent=2)

    (args.output_dir / "statsmodels_summaries_wls.txt").write_text(all_summaries_wls, encoding="utf-8")

    if all_summaries_ols:
        (args.output_dir / "statsmodels_summaries_ols.txt").write_text(all_summaries_ols, encoding="utf-8")

    # Cluster distribution overall (by household-day observations)
    cluster_dist = (
        household_days.group_by("cluster")
        .agg(pl.len().alias("n_obs"))
        .sort("cluster")
        .with_columns((pl.col("n_obs") / pl.col("n_obs").sum() * 100).alias("pct"))
    )

    generate_report_logratio(
        results=results,
        dominance_stats=dominance_stats,
        cluster_dist=cluster_dist,
        output_path=args.output_dir / "regression_report_logratio_blockgroups.txt",
    )

    print(f"\nOutputs saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
