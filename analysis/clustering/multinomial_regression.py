"""
Stage 2: Multinomial Logistic Regression for Cluster Prediction

Predicts energy usage cluster membership from Census demographic variables.
This analysis identifies which demographic factors are associated with different
electricity consumption patterns.

This script handles all census data operations internally:
    - Fetches census data from API (or uses cache)
    - Joins with ZIP+4 crosswalk
    - Runs multinomial logistic regression

Pipeline:
    1. Load cluster assignments from Stage 1
    2. Aggregate to one cluster per ZIP+4 (modal cluster)
    3. Fetch/load Census demographics
    4. Join with ZIP+4 crosswalk
    5. Run multinomial logistic regression
    6. Output coefficients, odds ratios, and model diagnostics

Usage:
    python multinomial_regression.py \\
        --clusters data/clustering/results/cluster_assignments.parquet \\
        --output-dir data/clustering/results/stage2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. LOAD & AGGREGATE CLUSTER ASSIGNMENTS
# ---------------------------------------------------------------------------


def load_cluster_assignments(path: Path) -> pl.DataFrame:
    """
    Load cluster assignments and aggregate to one cluster per ZIP+4.

    Each ZIP+4 may have multiple daily profiles assigned to different clusters.
    We take the modal cluster as the ZIP+4's characteristic pattern.

    Args:
        path: Path to cluster_assignments.parquet

    Returns:
        DataFrame with one row per ZIP+4 and its dominant cluster
        Columns: zip_code, cluster, n_profiles, confidence_pct
    """
    logger.info(f"Loading cluster assignments from {path}")

    # Lazy scan for scalability
    assignments = pl.scan_parquet(path)

    # Count profiles per (zip_code, cluster)
    counts = assignments.group_by(["zip_code", "cluster"]).agg(pl.len().alias("n_profiles"))

    # After aggregation, collect (this is now much smaller than raw assignments)
    counts_df = counts.collect(engine="streaming")

    total_profiles = int(counts_df["n_profiles"].sum())
    n_zip_codes = counts_df["zip_code"].n_unique()
    n_clusters = counts_df["cluster"].n_unique()

    logger.info(f"  {total_profiles:,} profile assignments")
    logger.info(f"  {n_zip_codes} unique ZIP+4 codes")
    logger.info(f"  {n_clusters} clusters")

    # Total profiles per ZIP+4
    totals = counts_df.group_by("zip_code").agg(pl.col("n_profiles").sum().alias("total_profiles"))

    # Modal cluster per ZIP+4 (cluster with max n_profiles)
    modal = (
        counts_df.sort(["zip_code", "n_profiles"], descending=[False, True])
        .group_by("zip_code")
        .agg(
            pl.col("cluster").first().alias("cluster"),
            pl.col("n_profiles").first().alias("n_profiles"),
        )
    )

    zip_clusters = (
        modal.join(totals, on="zip_code")
        .with_columns((pl.col("n_profiles") / pl.col("total_profiles") * 100.0).alias("confidence_pct"))
        .select(["zip_code", "cluster", "n_profiles", "confidence_pct"])
    )

    logger.info(f"  Aggregated to {len(zip_clusters)} ZIP+4 codes")
    logger.info(f"  Mean cluster confidence: {zip_clusters['confidence_pct'].mean():.1f}%")

    return zip_clusters


# ---------------------------------------------------------------------------
# 2. CENSUS DATA HANDLING
# ---------------------------------------------------------------------------


def fetch_or_load_census(
    cache_path: Path,
    state_fips: str = "17",
    acs_year: int = 2023,
    force_fetch: bool = False,
) -> pl.DataFrame:
    """
    Fetch census data from API or load from cache.

    Args:
        cache_path: Path to cache file
        state_fips: State FIPS code (default: 17 for Illinois)
        acs_year: ACS year (default: 2023)
        force_fetch: If True, fetch even if cache exists

    Returns:
        DataFrame with census demographics at block group level
    """
    if cache_path.exists() and not force_fetch:
        logger.info(f"Loading census data from cache: {cache_path}")
        return pl.read_parquet(cache_path)

    logger.info(f"Fetching census data from API (state={state_fips}, year={acs_year})...")

    try:
        from smart_meter_analysis.census import fetch_census_data
    except ImportError:
        logger.error("Could not import fetch_census_data from smart_meter_analysis.census")
        logger.error("Make sure the smart_meter_analysis package is installed")
        raise

    census_df = fetch_census_data(state_fips=state_fips, acs_year=acs_year)

    # Cache for future use
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    census_df.write_parquet(cache_path)
    logger.info(f"  Cached census data to {cache_path}")

    return census_df


def load_crosswalk(crosswalk_path: Path, zip_codes: list[str]) -> pl.DataFrame:
    """
    Load ZIP+4 to Census Block Group crosswalk.

    Args:
        crosswalk_path: Path to crosswalk file (tab-separated)
        zip_codes: List of ZIP+4 codes to filter to

    Returns:
        DataFrame with zip_code and block_group_geoid columns
    """
    logger.info(f"Loading crosswalk from {crosswalk_path}")

    crosswalk = pl.read_csv(
        crosswalk_path,
        separator="\t",
        infer_schema_length=10000,
    )

    crosswalk = (
        crosswalk.with_columns([
            (pl.col("Zip").cast(pl.Utf8).str.zfill(5) + "-" + pl.col("Zip4").cast(pl.Utf8).str.zfill(4)).alias(
                "zip_code"
            ),
            pl.col("CensusKey2023").cast(pl.Utf8).str.zfill(15).str.slice(0, 12).alias("block_group_geoid"),
        ])
        .filter(pl.col("zip_code").is_in(zip_codes))
        .select(["zip_code", "block_group_geoid"])
    )

    n_matched = crosswalk["zip_code"].n_unique()
    logger.info(f"  Matched {n_matched} of {len(zip_codes)} ZIP+4 codes")

    return crosswalk


def join_census_to_zip4(
    crosswalk: pl.DataFrame,
    census_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Join census demographics to ZIP+4 codes via crosswalk.

    Args:
        crosswalk: ZIP+4 to block group mapping
        census_df: Census data at block group level

    Returns:
        DataFrame with ZIP+4 codes and census demographics
    """
    logger.info("Joining census data to ZIP+4 codes...")

    # Standardize census GEOID
    census_df = census_df.with_columns(pl.col("GEOID").cast(pl.Utf8).str.zfill(12).alias("block_group_geoid"))

    # Join
    demographics = (
        crosswalk.lazy().join(census_df.lazy(), on="block_group_geoid", how="left").collect(engine="streaming")
    )

    logger.info(f"  Joined {demographics['zip_code'].n_unique()} ZIP+4 codes")

    return demographics


def get_numeric_census_columns(census_df: pl.DataFrame) -> list[str]:
    """
    Get all numeric columns from census data that can be used as predictors.

    Excludes ID columns, geographic identifiers, and cluster assignment columns.

    Args:
        census_df: Census DataFrame

    Returns:
        List of numeric column names suitable as predictors
    """
    exclude_patterns = ["GEOID", "NAME", "state", "county", "tract", "block", "geoid", "fips"]

    # Also exclude cluster assignment columns (not census data)
    exclude_exact = ["cluster", "n_profiles", "confidence_pct", "total_profiles", "zip_code"]

    numeric_cols = []
    for col in census_df.columns:
        # Skip exact matches
        if col in exclude_exact:
            continue
        # Skip non-numeric
        if census_df[col].dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
            continue
        # Skip ID/geographic columns
        if any(pattern.lower() in col.lower() for pattern in exclude_patterns):
            continue
        numeric_cols.append(col)

    return numeric_cols


# ---------------------------------------------------------------------------
# 3. PREPARE REGRESSION DATA
# ---------------------------------------------------------------------------


def prepare_regression_data(
    zip_clusters: pl.DataFrame,
    demographics: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]]:
    """
    Prepare final dataset for regression analysis.

    Joins cluster assignments with demographics and identifies predictor variables
    dynamically from the census data.

    Args:
        zip_clusters: ZIP+4 cluster assignments
        demographics: ZIP+4 demographics

    Returns:
        Tuple of (regression DataFrame, list of predictor column names)
    """
    logger.info("Preparing regression dataset...")

    # Join clusters with demographics
    data = zip_clusters.join(demographics, on="zip_code", how="left")

    # Check match rate
    matched = data.filter(pl.col("block_group_geoid").is_not_null()).height
    match_rate = matched / len(data) * 100 if len(data) > 0 else 0.0
    logger.info(f"  Demographic match rate: {match_rate:.1f}%")

    if match_rate < 80:
        logger.warning("  Low match rate may affect regression results")

    # Get all numeric census columns as potential predictors
    predictors = get_numeric_census_columns(data)
    logger.info(f"  Found {len(predictors)} potential predictor variables")

    if not predictors:
        raise ValueError("No numeric predictor columns found in census data.")

    # Filter to rows with valid cluster and at least some predictor data
    data = data.filter(pl.col("cluster").is_not_null())

    # Drop predictors that are mostly null (>50% missing)
    valid_predictors = []
    for pred in predictors:
        null_rate = data[pred].null_count() / len(data)
        if null_rate < 0.5:
            valid_predictors.append(pred)
        else:
            logger.warning(f"  Dropping {pred}: {null_rate:.0%} null")

    predictors = valid_predictors
    logger.info(f"  Using {len(predictors)} predictor variables after filtering")
    logger.info(f"  Final dataset: {len(data)} observations")

    return data, predictors


# ---------------------------------------------------------------------------
# 4. MULTINOMIAL LOGISTIC REGRESSION
# ---------------------------------------------------------------------------


def run_multinomial_regression(
    data: pl.DataFrame,
    predictors: list[str],
    outcome: str = "cluster",
) -> dict[str, Any]:
    """
    Run multinomial logistic regression.

    Args:
        data: DataFrame with outcome and predictors
        predictors: List of predictor column names
        outcome: Outcome column name

    Returns:
        Dictionary with model results
    """
    logger.info("Running multinomial logistic regression...")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.error("scikit-learn required. Run: pip install scikit-learn")
        raise

    # Polars -> NumPy
    X = data.select(predictors).to_numpy().astype(np.float64)
    y = data.get_column(outcome).to_numpy()

    # Handle missing values (drop rows with NaNs)
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    logger.info(f"  Training on {len(X)} observations with {len(predictors)} predictors")

    if len(X) == 0:
        raise ValueError("No observations left after dropping rows with NaNs.")

    # Standardize predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit multinomial logistic regression
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_scaled, y)

    # Cross-validation accuracy
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, n_jobs=-1)
    logger.info(f"  Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Extract coefficients
    classes = model.classes_.tolist()
    coefficients: dict[str, dict[str, float]] = {}

    for i, cls in enumerate(classes):
        coefficients[f"cluster_{cls}"] = {pred: float(coef) for pred, coef in zip(predictors, model.coef_[i])}

    # Calculate odds ratios
    odds_ratios: dict[str, dict[str, float]] = {}
    for cls, coefs in coefficients.items():
        odds_ratios[cls] = {pred: float(np.exp(coef)) for pred, coef in coefs.items()}

    results: dict[str, Any] = {
        "n_observations": len(X),
        "n_predictors": len(predictors),
        "n_clusters": len(classes),
        "clusters": classes,
        "predictors": predictors,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "coefficients": coefficients,
        "odds_ratios": odds_ratios,
        "intercepts": {f"cluster_{cls}": float(intercept) for cls, intercept in zip(classes, model.intercept_)},
    }

    return results


# ---------------------------------------------------------------------------
# 5. SUMMARIZE / REPORT
# ---------------------------------------------------------------------------


def identify_key_predictors(results: dict[str, Any], top_n: int = 5) -> dict[str, Any]:
    """
    Identify the most important predictors for each cluster.

    Args:
        results: Regression results dictionary
        top_n: Number of top predictors to return per cluster

    Returns:
        Dictionary mapping clusters to their key predictors
    """
    key_predictors: dict[str, Any] = {}

    for cluster in results["clusters"]:
        cluster_key = f"cluster_{cluster}"
        coefs = results["coefficients"][cluster_key]

        # Sort by absolute coefficient value
        sorted_preds = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)

        key_predictors[cluster_key] = [
            {
                "predictor": pred,
                "coefficient": coef,
                "odds_ratio": results["odds_ratios"][cluster_key][pred],
                "direction": "positive" if coef > 0 else "negative",
            }
            for pred, coef in sorted_preds[:top_n]
        ]

    return key_predictors


def generate_report(
    results: dict[str, Any],
    key_predictors: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Generate human-readable regression report.

    Args:
        results: Regression results dictionary
        key_predictors: Key predictors per cluster
        output_path: Path to save report
    """
    lines = [
        "=" * 70,
        "STAGE 2: MULTINOMIAL REGRESSION RESULTS",
        "=" * 70,
        "",
        "MODEL SUMMARY",
        "-" * 70,
        f"Observations: {results['n_observations']}",
        f"Predictors: {results['n_predictors']}",
        f"Clusters: {results['n_clusters']}",
        f"Cross-validation accuracy: {results['cv_accuracy_mean']:.1%} (+/- {results['cv_accuracy_std']:.1%})",
        "",
        "KEY PREDICTORS BY CLUSTER (Top 5 by coefficient magnitude)",
        "-" * 70,
    ]

    for cluster in results["clusters"]:
        cluster_key = f"cluster_{cluster}"
        lines.append(f"\nCluster {cluster}:")

        for pred_info in key_predictors[cluster_key]:
            direction = "↑" if pred_info["direction"] == "positive" else "↓"
            lines.append(
                f"  {direction} {pred_info['predictor']}: "
                f"OR={pred_info['odds_ratio']:.2f}, "
                f"coef={pred_info['coefficient']:.3f}"
            )

    lines.extend([
        "",
        "INTERPRETATION GUIDE",
        "-" * 70,
        "- Odds Ratio (OR) > 1: Higher values increase likelihood of this cluster",
        "- Odds Ratio (OR) < 1: Higher values decrease likelihood of this cluster",
        "- Coefficients are standardized (z-scores)",
        "",
        "ALL PREDICTORS USED",
        "-" * 70,
        ", ".join(results["predictors"]),
        "",
        "=" * 70,
    ])

    report_text = "\n".join(lines)

    output_path.write_text(report_text)
    logger.info(f"Report saved to {output_path}")

    print("\n" + report_text)


# ---------------------------------------------------------------------------
# 6. CLI ENTRYPOINT
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: Multinomial regression for cluster prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (auto-fetches census if needed)
    python multinomial_regression.py \\
        --clusters data/clustering/results/cluster_assignments.parquet

    # Force re-fetch of census data
    python multinomial_regression.py \\
        --clusters data/clustering/results/cluster_assignments.parquet \\
        --fetch-census
        """,
    )

    # Required
    parser.add_argument(
        "--clusters",
        type=Path,
        required=True,
        help="Path to cluster_assignments.parquet from Stage 1",
    )

    # Census options
    census_group = parser.add_argument_group("Census Data")
    census_group.add_argument(
        "--crosswalk",
        type=Path,
        default=Path("data/reference/2023_comed_zip4_census_crosswalk.txt"),
        help="Path to ZIP+4 crosswalk file",
    )
    census_group.add_argument(
        "--census-cache",
        type=Path,
        default=Path("data/reference/census_17_2023.parquet"),
        help="Path to census cache file",
    )
    census_group.add_argument(
        "--fetch-census",
        action="store_true",
        help="Force fetch census data from API (even if cached)",
    )
    census_group.add_argument(
        "--state-fips",
        default="17",
        help="State FIPS code (default: 17 for Illinois)",
    )
    census_group.add_argument(
        "--acs-year",
        type=int,
        default=2023,
        help="ACS year (default: 2023)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clustering/results/stage2"),
        help="Output directory for Stage 2 results",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.clusters.exists():
        logger.error(f"Cluster assignments not found: {args.clusters}")
        logger.error("Run Stage 1 (dtw_clustering.py) first")
        return

    if not args.crosswalk.exists():
        logger.error(f"Crosswalk not found: {args.crosswalk}")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 2: DEMOGRAPHIC PREDICTION OF USAGE CLUSTERS")
    print("=" * 70)

    # Step 1: Load cluster assignments
    zip_clusters = load_cluster_assignments(args.clusters)
    zip_codes = zip_clusters["zip_code"].to_list()

    # Step 2: Fetch/load census data
    census_df = fetch_or_load_census(
        cache_path=args.census_cache,
        state_fips=args.state_fips,
        acs_year=args.acs_year,
        force_fetch=args.fetch_census,
    )
    logger.info(f"  Census data: {len(census_df):,} block groups, {len(census_df.columns)} columns")

    # Step 3: Load crosswalk and join
    crosswalk = load_crosswalk(args.crosswalk, zip_codes)
    demographics = join_census_to_zip4(crosswalk, census_df)

    # Step 4: Prepare regression data
    data, predictors = prepare_regression_data(zip_clusters, demographics)

    if len(data) < 50:
        logger.error(f"Insufficient data for regression: {len(data)} observations")
        return

    # Step 5: Run regression
    results = run_multinomial_regression(data, predictors)

    # Step 6: Identify key predictors
    key_predictors = identify_key_predictors(results)

    # Step 7: Save outputs
    results_path = args.output_dir / "regression_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    report_path = args.output_dir / "regression_report.txt"
    generate_report(results, key_predictors, report_path)

    data_path = args.output_dir / "regression_data.parquet"
    data.write_parquet(data_path)
    logger.info(f"Regression data saved to {data_path}")

    print("\n" + "=" * 70)
    print("STAGE 2 COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
