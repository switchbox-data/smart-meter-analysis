#!/usr/bin/env python3
"""
Diagnostic: Check block-group sampling density for Stage 2 regression.

This script analyzes whether your household sample is dense enough at the
block-group level to detect meaningful demographic patterns.

Usage:
    python diagnose_bg_density.py \
        --clusters data/clustering/results/cluster_assignments.parquet \
        --crosswalk data/reference/2023_comed_zip4_census_crosswalk.txt \
        --output diagnostics.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def inspect_cluster_file(clusters_path: Path) -> None:
    """Inspect what's actually in the cluster assignments file."""
    print("=" * 70)
    print("CLUSTER FILE INSPECTION")
    print("=" * 70)

    df = pl.read_parquet(clusters_path)

    print(f"\nFile: {clusters_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.columns}")
    print("\nSchema:")
    for col, dtype in df.schema.items():
        print(f"  {col:30s} {dtype}")

    print("\nFirst few rows:")
    print(df.head())

    print("\n" + "=" * 70)


def load_and_join_to_blockgroups(
    clusters_path: Path,
    crosswalk_path: Path,
) -> pl.DataFrame:
    """Join cluster assignments to block groups."""
    print("\nLoading cluster assignments...")
    clusters = pl.read_parquet(clusters_path)
    print(f"  {len(clusters):,} household-day observations")

    # Check which ID columns are present
    id_col = None
    if "account_identifier" in clusters.columns:
        id_col = "account_identifier"
    elif "household_id" in clusters.columns:
        id_col = "household_id"
    elif "meter_id" in clusters.columns:
        id_col = "meter_id"
    else:
        print("  ⚠️  WARNING: No household identifier column found!")
        print(f"  Available columns: {clusters.columns}")
        id_col = None

    if id_col:
        print(f"  {clusters[id_col].n_unique():,} unique households (using column: {id_col})")

    if "zip_code" not in clusters.columns:
        raise ValueError(f"'zip_code' column not found in {clusters_path}")

    if "cluster" not in clusters.columns:
        raise ValueError(f"'cluster' column not found in {clusters_path}")

    print(f"  {clusters['cluster'].n_unique()} clusters")

    print("\nLoading crosswalk...")
    zip_codes = clusters["zip_code"].unique().to_list()

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

    print(f"  {crosswalk['zip_code'].n_unique():,} ZIP+4s matched")

    # Check for fan-out
    fanout = crosswalk.group_by("zip_code").agg(pl.n_unique("block_group_geoid").alias("n_bg"))
    max_fanout = fanout["n_bg"].max()
    if max_fanout > 1:
        pct_fanout = (fanout.filter(pl.col("n_bg") > 1).height / len(fanout)) * 100
        print(f"  ⚠️  {pct_fanout:.1f}% of ZIP+4s map to multiple block groups (max={max_fanout})")

    print("\nJoining to block groups...")
    df = clusters.join(crosswalk, on="zip_code", how="left")

    missing = df.filter(pl.col("block_group_geoid").is_null()).height
    if missing > 0:
        print(f"  ⚠️  {missing:,} observations ({missing / len(df) * 100:.1f}%) missing block group")

    df = df.filter(pl.col("block_group_geoid").is_not_null())
    print(f"  ✓ {len(df):,} observations across {df['block_group_geoid'].n_unique():,} block groups")

    # Store the ID column name for later use
    df = df.with_columns(pl.lit(id_col).alias("_id_col_name"))

    return df


def diagnose_household_density(df: pl.DataFrame) -> dict:
    """Check households per block group."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1: HOUSEHOLD DENSITY PER BLOCK GROUP")
    print("=" * 70)

    # Get the ID column name (stored during join)
    id_col = df["_id_col_name"][0] if "_id_col_name" in df.columns else None

    if not id_col or id_col not in df.columns:
        print("\n⚠️  WARNING: Cannot compute household density - no household ID column")
        print("   Skipping this diagnostic.")
        return {}

    hh_per_bg = df.group_by("block_group_geoid").agg(pl.col(id_col).n_unique().alias("n_households"))

    stats = {
        "n_block_groups": hh_per_bg.height,
        "mean_hh_per_bg": hh_per_bg["n_households"].mean(),
        "median_hh_per_bg": hh_per_bg["n_households"].median(),
        "min_hh_per_bg": hh_per_bg["n_households"].min(),
        "max_hh_per_bg": hh_per_bg["n_households"].max(),
        "p10_hh_per_bg": hh_per_bg["n_households"].quantile(0.10),
        "p90_hh_per_bg": hh_per_bg["n_households"].quantile(0.90),
    }

    print(f"\nBlock groups: {stats['n_block_groups']:,}")
    print("Households per block group:")
    print(f"  Mean:   {stats['mean_hh_per_bg']:.1f}")
    print(f"  Median: {stats['median_hh_per_bg']:.1f}")
    print(f"  Min:    {stats['min_hh_per_bg']}")
    print(f"  Max:    {stats['max_hh_per_bg']}")
    print(f"  P10:    {stats['p10_hh_per_bg']:.1f}")
    print(f"  P90:    {stats['p90_hh_per_bg']:.1f}")

    # Distribution
    print("\nDistribution:")
    dist = (
        hh_per_bg.with_columns(
            pl.when(pl.col("n_households") == 1)
            .then(pl.lit("1 household"))
            .when(pl.col("n_households") == 2)
            .then(pl.lit("2 households"))
            .when(pl.col("n_households").is_between(3, 5))
            .then(pl.lit("3-5 households"))
            .when(pl.col("n_households").is_between(6, 10))
            .then(pl.lit("6-10 households"))
            .when(pl.col("n_households").is_between(11, 20))
            .then(pl.lit("11-20 households"))
            .otherwise(pl.lit("20+ households"))
            .alias("bucket")
        )
        .group_by("bucket")
        .agg(pl.len().alias("n_bg"))
        .sort("n_bg", descending=True)
    )

    for row in dist.iter_rows(named=True):
        pct = row["n_bg"] / stats["n_block_groups"] * 100
        print(f"  {row['bucket']:20s}: {row['n_bg']:5,} BGs ({pct:5.1f}%)")

    # Assessment
    print("\nASSESSMENT:")
    if stats["median_hh_per_bg"] < 3:
        print("  ❌ CRITICAL: Median < 3 households per block group")
        print("     → Most block groups have too few households for stable cluster shares")
        print("     → Stage 2 regression will have very weak signal")
    elif stats["median_hh_per_bg"] < 10:
        print("  ⚠️  WARNING: Median < 10 households per block group")
        print("     → Cluster shares will be noisy")
        print("     → Consider increasing sample size or aggregating to ZIP-level")
    else:
        print("  ✓ GOOD: Median ≥ 10 households per block group")
        print("     → Should have reasonable signal for Stage 2")

    return stats


def diagnose_obs_density(df: pl.DataFrame) -> dict:
    """Check household-day observations per block group."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: OBSERVATION DENSITY PER BLOCK GROUP")
    print("=" * 70)

    obs_per_bg = df.group_by("block_group_geoid").agg(pl.len().alias("n_obs"))

    stats = {
        "mean_obs_per_bg": obs_per_bg["n_obs"].mean(),
        "median_obs_per_bg": obs_per_bg["n_obs"].median(),
        "min_obs_per_bg": obs_per_bg["n_obs"].min(),
        "max_obs_per_bg": obs_per_bg["n_obs"].max(),
    }

    print("\nObservations (household-days) per block group:")
    print(f"  Mean:   {stats['mean_obs_per_bg']:.1f}")
    print(f"  Median: {stats['median_obs_per_bg']:.1f}")
    print(f"  Min:    {stats['min_obs_per_bg']}")
    print(f"  Max:    {stats['max_obs_per_bg']}")

    # After Stage 2 filtering (≥50 obs, ≥2 clusters)
    filtered = obs_per_bg.filter(pl.col("n_obs") >= 50)
    n_filtered = filtered.height
    pct_surviving = n_filtered / obs_per_bg.height * 100 if obs_per_bg.height > 0 else 0

    print("\nAfter Stage 2 filtering (≥50 obs per BG):")
    print(f"  {n_filtered:,} block groups ({pct_surviving:.1f}%) survive")

    if pct_surviving < 20:
        print("\n  ⚠️  WARNING: <20% of block groups survive filtering")
        print("     → You're throwing away most of your data")
        print("     → Consider lowering threshold or increasing sample size")

    return stats


def diagnose_cluster_share_variance(df: pl.DataFrame) -> dict:
    """Check variance in cluster shares across block groups."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3: CLUSTER SHARE VARIANCE")
    print("=" * 70)

    # Compute cluster shares per block group
    bg_counts = df.group_by(["block_group_geoid", "cluster"]).agg(pl.len().alias("n_obs"))

    bg_totals = df.group_by("block_group_geoid").agg(pl.len().alias("total_obs"))

    shares = bg_counts.join(bg_totals, on="block_group_geoid").with_columns(
        (pl.col("n_obs") / pl.col("total_obs")).alias("cluster_share")
    )

    # Pivot to wide format for variance calculation
    wide = shares.pivot(
        index="block_group_geoid",
        columns="cluster",
        values="cluster_share",
    ).fill_null(0)

    cluster_cols = [c for c in wide.columns if c != "block_group_geoid"]

    print("\nCluster share variance across block groups:")
    stats = {}
    for col in sorted(cluster_cols):
        if col in wide.columns:
            var = wide[col].var()
            mean = wide[col].mean()
            stats[col] = {"mean": mean, "variance": var}
            print(f"  Cluster {col}: mean={mean:.3f}, variance={var:.4f}")

    if stats:
        avg_var = sum(s["variance"] for s in stats.values()) / len(stats)

        print(f"\nAverage variance: {avg_var:.4f}")

        print("\nASSESSMENT:")
        if avg_var < 0.01:
            print("  ❌ CRITICAL: Variance < 0.01")
            print("     → Cluster shares barely differ across block groups")
            print("     → No demographic signal can be detected")
            print("     → MUST increase sample size or change aggregation level")
        elif avg_var < 0.02:
            print("  ⚠️  WARNING: Variance < 0.02")
            print("     → Weak signal; demographic effects will be hard to detect")
        else:
            print("  ✓ GOOD: Variance ≥ 0.02")
            print("     → Sufficient variation for regression")

    return stats


def generate_recommendations(
    hh_stats: dict,
    obs_stats: dict,
    share_stats: dict,
) -> None:
    """Generate actionable recommendations."""
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if not hh_stats:
        print("\n⚠️  Could not assess household density (no ID column)")
        print("   Assess based on observation density instead.")
        return

    median_hh = hh_stats.get("median_hh_per_bg", 0)

    if median_hh < 3:
        print("\n🔴 CRITICAL ISSUE: Sample too sparse for block-group analysis")
        print("\nOptions:")
        print("  1. Increase household sample to 50k-100k+")
        print("  2. Switch to ZIP-level or ZIP+4-level aggregation")
        print("  3. Use stratified sampling (population-weighted by block group)")
        print("  4. Aggregate to county-level if only interested in broad patterns")

    elif median_hh < 10:
        print("\n⚠️  WARNING: Marginal sample density")
        print("\nOptions:")
        print("  1. Increase sample to 30k-50k households")
        print("  2. Consider ZIP-level aggregation for more stable estimates")
        print("  3. Use hierarchical modeling to pool information across similar BGs")

    else:
        print("\n✓ Sample density looks reasonable")
        print("\nOptional improvements:")
        print("  1. Still consider stratified sampling for better coverage")
        print("  2. Add more days if household-day counts are low")

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose block-group sampling density for Stage 2")
    parser.add_argument("--clusters", type=Path, required=True, help="Path to cluster_assignments.parquet")
    parser.add_argument("--crosswalk", type=Path, required=True, help="Path to ZIP+4 crosswalk file")
    parser.add_argument("--inspect-only", action="store_true", help="Only inspect the cluster file schema and exit")
    parser.add_argument("--output", type=Path, default=None, help="Optional: save report to file")

    args = parser.parse_args()

    # Inspect the cluster file first
    inspect_cluster_file(args.clusters)

    if args.inspect_only:
        return

    # Load and join
    df = load_and_join_to_blockgroups(args.clusters, args.crosswalk)

    # Run diagnostics
    hh_stats = diagnose_household_density(df)
    obs_stats = diagnose_obs_density(df)
    share_stats = diagnose_cluster_share_variance(df)

    # Recommendations
    generate_recommendations(hh_stats, obs_stats, share_stats)

    # TODO: Save to file if requested
    if args.output:
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
