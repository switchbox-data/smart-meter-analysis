#!/usr/bin/env python3
"""
Phase 2: K-Means Clustering for Load Profile Analysis.

Clusters daily electricity usage profiles using standard Euclidean distance
to identify distinct consumption patterns.

Pipeline:
    1. Load daily profiles from Phase 1
    2. Normalize profiles (optional)
    3. Evaluate k values to find optimal k (via silhouette score on a sample)
    4. Run final clustering with optimal k
    5. Output assignments, centroids, and visualizations

Usage:
    # Standard run (evaluates k=3-6 using silhouette on up to 10k samples)
    python euclidean_clustering_fixed.py \\
        --input data/clustering/sampled_profiles.parquet \\
        --output-dir data/clustering/results \\
        --k-range 3 6 \\
        --find-optimal-k \\
        --normalize \\
        --silhouette-sample-size 10000

    # Fixed k (no evaluation)
    python euclidean_clustering.py \\
        --input data/clustering/sampled_profiles.parquet \\
        --output-dir data/clustering/results \\
        --k 4 --normalize
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_NORMALIZATION: str = "minmax"
DEFAULT_NORMALIZE: bool = True


def load_profiles(path: Path) -> tuple[np.ndarray, pl.DataFrame]:
    """
    Load profiles from parquet file.

    Args:
        path: Path to sampled_profiles.parquet

    Returns:
        Tuple of (profile_array, metadata_df)
    """
    logger.info(f"Loading profiles from {path}")

    df = pl.read_parquet(path)

    # Extract profiles as numpy array
    profiles = np.array(df["profile"].to_list(), dtype=np.float64)

    logger.info(
        "  Loaded %s profiles with %s time points each",
        f"{len(profiles):,}",
        profiles.shape[1],
    )
    logger.info("  Data shape: %s", (profiles.shape[0], profiles.shape[1]))
    logger.info("  Data range: [%.2f, %.2f]", profiles.min(), profiles.max())

    return profiles, df


def normalize_profiles(
    df: pl.DataFrame,
    method: str = "minmax",
    profile_col: str = "profile",
    out_col: str | None = None,
) -> pl.DataFrame:
    """
    Normalize per-household-day profiles for clustering.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain a list column with the profile values (e.g. 48-dim vector).
    method : {"none", "zscore", "minmax"}
        - "none": return df unchanged
        - "zscore": per-profile z-score: (x - mean) / std
        - "minmax": per-profile min-max: (x - min) / (max - min)
    profile_col : str
        Name of the list column holding the raw profile.
    out_col : str | None
        If provided, write normalized profile to this column; otherwise overwrite
        `profile_col` in-place.

    Notes
    -----
    - Normalization is done per profile (per row), not globally.
    - For degenerate profiles where max == min, we fall back to all zeros.
    """

    if method == "none":
        return df

    if profile_col not in df.columns:
        raise ValueError(f"normalize_profiles: column '{profile_col}' not found in DataFrame")

    target_col = out_col or profile_col

    expr = pl.col(profile_col)

    if method == "zscore":
        mean_expr = expr.list.mean()
        std_expr = expr.list.std(ddof=0)

        normalized = (expr - mean_expr) / std_expr

        # If std == 0 (flat profile), fall back to zeros
        normalized = pl.when(std_expr != 0).then(normalized).otherwise(expr * 0.0)

    elif method == "minmax":
        min_expr = expr.list.min()
        max_expr = expr.list.max()
        range_expr = max_expr - min_expr

        normalized = (expr - min_expr) / range_expr

        # If range == 0 (flat profile), fall back to zeros
        normalized = pl.when(range_expr != 0).then(normalized).otherwise(expr * 0.0)

    else:
        raise ValueError(f"Unknown normalization method: {method!r}")

    return df.with_columns(normalized.alias(target_col))


def evaluate_clustering(
    X: np.ndarray,
    k_range: range,
    n_init: int = 10,
    random_state: int = 42,
    silhouette_sample_size: int | None = 10_000,
) -> dict:
    """
    Evaluate clustering for different values of k.

    Uses inertia on the full dataset and silhouette score computed on a
    subsample (to avoid O(n^2) cost when n is large).

    Args:
        X: Profile array of shape (n_samples, n_timepoints)
        k_range: Range of k values to test
        n_init: Number of random initializations
        random_state: Random seed for reproducibility
        silhouette_sample_size: Max number of samples for silhouette.
            If None, uses full dataset (NOT recommended for very large n).

    Returns:
        Dictionary with k_values, inertia, and silhouette scores
    """
    n_samples = X.shape[0]
    logger.info("Evaluating clustering for k in %s...", list(k_range))
    logger.info("  Dataset size: %s profiles", f"{n_samples:,}")

    if silhouette_sample_size is None:
        logger.info("  Silhouette: using FULL dataset (may be very slow).")
    elif n_samples > silhouette_sample_size:
        logger.info(
            "  Silhouette: using a random subsample of %s profiles.",
            f"{silhouette_sample_size:,}",
        )
    else:
        logger.info(
            "  Silhouette: using all %s profiles (<= sample size).",
            f"{n_samples:,}",
        )

    results = {
        "k_values": [],
        "inertia": [],
        "silhouette": [],
    }

    for k in k_range:
        logger.info("")
        logger.info("  Testing k=%d...", k)

        model = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=random_state,
        )

        labels = model.fit_predict(X)

        # Inertia on full data
        inertia = float(model.inertia_)

        # Silhouette on sample (or full data if silhouette_sample_size is None)
        sil_kwargs: dict = {"metric": "euclidean"}
        if silhouette_sample_size is not None and n_samples > silhouette_sample_size:
            sil_kwargs["sample_size"] = silhouette_sample_size
            sil_kwargs["random_state"] = random_state

        sil_score = silhouette_score(X, labels, **sil_kwargs)

        results["k_values"].append(k)
        results["inertia"].append(inertia)
        results["silhouette"].append(float(sil_score))

        logger.info("    Inertia: %s", f"{inertia:,.2f}")
        logger.info("    Silhouette: %.3f", sil_score)

    return results


def find_optimal_k(eval_results: dict) -> int:
    """
    Find optimal k based on silhouette score.

    Args:
        eval_results: Results from evaluate_clustering

    Returns:
        Optimal k value
    """
    k_values = eval_results["k_values"]
    silhouettes = eval_results["silhouette"]

    best_idx = int(np.argmax(silhouettes))
    best_k = int(k_values[best_idx])

    logger.info("")
    logger.info(
        "Optimal k=%d (silhouette=%.3f)",
        best_k,
        silhouettes[best_idx],
    )

    return best_k


def run_clustering(
    X: np.ndarray,
    k: int,
    n_init: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run k-means clustering.

    Args:
        X: Profile array
        k: Number of clusters
        n_init: Number of random initializations
        random_state: Random seed

    Returns:
        Tuple of (labels, centroids, inertia)
    """
    logger.info("")
    logger.info(
        "Running k-means with k=%d on %s profiles...",
        k,
        f"{X.shape[0]:,}",
    )

    model = KMeans(
        n_clusters=k,
        n_init=n_init,
        random_state=random_state,
    )

    labels = model.fit_predict(X)
    centroids = model.cluster_centers_
    inertia = float(model.inertia_)

    logger.info("  Inertia: %s", f"{inertia:,.2f}")

    # Log cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        pct = count / len(labels) * 100
        logger.info(
            "  Cluster %d: %s profiles (%.1f%%)",
            cluster,
            f"{count:,}",
            pct,
        )

    return labels, centroids, inertia


def plot_centroids(
    centroids: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot cluster centroids (average load profiles).

    Args:
        centroids: Array of shape (k, n_timepoints)
        output_path: Path to save plot
    """
    k = len(centroids)
    n_timepoints = centroids.shape[1]

    # Create hour labels (assuming 48 half-hourly intervals)
    if n_timepoints == 48:
        hours = np.arange(0.5, 24.5, 0.5)
        xlabel = "Hour of Day"
    elif n_timepoints == 24:
        hours = np.arange(1, 25)
        xlabel = "Hour of Day"
    else:
        hours = np.arange(n_timepoints)
        xlabel = "Time Interval"

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for i, (centroid, color) in enumerate(zip(centroids, colors)):
        ax.plot(hours, centroid, label=f"Cluster {i}", color=color, linewidth=2)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Normalized Usage", fontsize=12)
    ax.set_title("Cluster Centroids (Average Load Profiles)", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    if n_timepoints == 48:
        ax.set_xticks(range(0, 25, 4))
        ax.set_xlim(0, 24)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info("  Saved centroids plot: %s", output_path)


def plot_cluster_samples(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    output_path: Path,
    n_samples: int = 50,
    random_state: int = 42,
) -> None:
    """
    Plot sample profiles from each cluster with centroid overlay.

    Args:
        X: Profile array
        labels: Cluster assignments
        centroids: Cluster centroids
        output_path: Path to save plot
        n_samples: Number of sample profiles per cluster
        random_state: Random seed
    """
    k = len(centroids)
    n_timepoints = X.shape[1]

    # Create hour labels
    if n_timepoints == 48:
        hours = np.arange(0.5, 24.5, 0.5)
    elif n_timepoints == 24:
        hours = np.arange(1, 25)
    else:
        hours = np.arange(n_timepoints)

    fig, axes = plt.subplots(1, k, figsize=(5 * k, 4), sharey=True)
    if k == 1:
        axes = [axes]

    rng = np.random.default_rng(random_state)
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for i, (ax, color) in enumerate(zip(axes, colors)):
        cluster_mask = labels == i
        cluster_profiles = X[cluster_mask]

        n_available = len(cluster_profiles)
        if n_available == 0:
            ax.set_title(f"Cluster {i} (n=0)")
            ax.grid(True, alpha=0.3)
            continue

        n_plot = min(n_samples, n_available)
        idx = rng.choice(n_available, size=n_plot, replace=False)

        # Plot samples with transparency
        for profile in cluster_profiles[idx]:
            ax.plot(hours, profile, color=color, alpha=0.1, linewidth=0.5)

        # Plot centroid
        ax.plot(hours, centroids[i], color="black", linewidth=2, label="Centroid")

        ax.set_title(f"Cluster {i} (n={n_available:,})")
        ax.set_xlabel("Hour")
        if i == 0:
            ax.set_ylabel("Normalized Usage")
        ax.grid(True, alpha=0.3)

        if n_timepoints == 48:
            ax.set_xticks(range(0, 25, 6))
            ax.set_xlim(0, 24)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info("  Saved cluster samples plot: %s", output_path)


def plot_elbow_curve(
    eval_results: dict,
    output_path: Path,
) -> None:
    """
    Plot elbow curve (inertia and silhouette vs k).

    Args:
        eval_results: Results from evaluate_clustering
        output_path: Path to save plot
    """
    k_values = eval_results["k_values"]
    inertia = eval_results["inertia"]
    silhouette = eval_results["silhouette"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Inertia (elbow curve)
    ax1.plot(k_values, inertia, "b-o", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax1.set_ylabel("Inertia", fontsize=12)
    ax1.set_title("Elbow Curve", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)

    # Silhouette score
    ax2.plot(k_values, silhouette, "g-o", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("Silhouette Score", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)

    # Mark optimal k
    best_idx = int(np.argmax(silhouette))
    ax2.axvline(x=k_values[best_idx], color="red", linestyle="--", alpha=0.7)
    ax2.scatter(
        [k_values[best_idx]],
        [silhouette[best_idx]],
        s=200,
        facecolors="none",
        edgecolors="red",
        linewidths=2,
        zorder=5,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info("  Saved elbow curve: %s", output_path)


def analyze_weekday_weekend_distribution(
    df: pl.DataFrame,
    labels: np.ndarray,
) -> dict:
    """
    Analyze weekday vs weekend distribution across clusters.

    This diagnostic checks if certain clusters are dominated by weekdays
    or weekends, which would suggest usage patterns are day-type dependent.

    Args:
        df: Original profile DataFrame with 'is_weekend' column
        labels: Cluster assignments

    Returns:
        Dictionary with distribution statistics
    """
    if "is_weekend" not in df.columns:
        logger.warning("  No 'is_weekend' column found - skipping weekday/weekend analysis")
        return {}

    # Add cluster labels to dataframe
    df_with_clusters = df.with_columns(pl.Series("cluster", labels))

    # Calculate distribution
    dist = (
        df_with_clusters.group_by(["cluster", "is_weekend"])
        .agg(pl.len().alias("count"))
        .sort(["cluster", "is_weekend"])
    )

    # Calculate percentages per cluster
    dist = dist.with_columns([(pl.col("count") / pl.col("count").sum().over("cluster") * 100).alias("pct")])

    # Summary: % weekend by cluster
    summary = (
        df_with_clusters.group_by("cluster")
        .agg([pl.len().alias("total"), (pl.col("is_weekend").sum() / pl.len() * 100).alias("pct_weekend")])
        .sort("cluster")
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("WEEKDAY/WEEKEND DISTRIBUTION BY CLUSTER")
    logger.info("=" * 70)

    for row in summary.iter_rows(named=True):
        cluster = row["cluster"]
        total = row["total"]
        pct_weekend = row["pct_weekend"]
        pct_weekday = 100 - pct_weekend

        logger.info(
            "  Cluster %d: %.1f%% weekday, %.1f%% weekend (n=%s)", cluster, pct_weekday, pct_weekend, f"{total:,}"
        )

        # Flag significant imbalances (>70% one type)
        if pct_weekend > 70:
            logger.warning("    ⚠️  Weekend-dominated cluster")
        elif pct_weekday > 70:
            logger.warning("    ⚠️  Weekday-dominated cluster")

    # Overall distribution
    overall_weekend_pct = float(df_with_clusters["is_weekend"].mean() * 100)
    logger.info("")
    logger.info("  Overall dataset: %.1f%% weekend, %.1f%% weekday", overall_weekend_pct, 100 - overall_weekend_pct)

    # Chi-square test would go here if needed for formal significance testing
    logger.info("=" * 70)

    return {
        "cluster_distribution": summary.to_dicts(),
        "detailed_distribution": dist.to_dicts(),
        "overall_weekend_pct": overall_weekend_pct,
    }


def save_results(
    df: pl.DataFrame,
    labels: np.ndarray,
    centroids: np.ndarray,
    eval_results: dict | None,
    metadata: dict,
    output_dir: Path,
) -> None:
    """
    Save all clustering results to output directory.

    Args:
        df: Original profile DataFrame with metadata
        labels: Cluster assignments
        centroids: Cluster centroids
        eval_results: K evaluation results (if any)
        metadata: Clustering metadata
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which ID columns are present (household vs ZIP+4 level)
    id_cols: list[str] = []
    if "account_identifier" in df.columns:
        id_cols.append("account_identifier")
    if "zip_code" in df.columns:
        id_cols.append("zip_code")
    id_cols.extend(["date", "is_weekend", "weekday"])

    # Only include columns that exist
    available_cols = [c for c in id_cols if c in df.columns]

    # Save cluster assignments
    assignments = df.select(available_cols).with_columns(
        pl.Series("cluster", labels),
    )
    assignments_path = output_dir / "cluster_assignments.parquet"
    assignments.write_parquet(assignments_path)
    logger.info("  Saved assignments: %s", assignments_path)

    # Save centroids as parquet
    centroids_df = pl.DataFrame({
        "cluster": list(range(len(centroids))),
        "centroid": [c.tolist() for c in centroids],
    })
    centroids_path = output_dir / "cluster_centroids.parquet"
    centroids_df.write_parquet(centroids_path)
    logger.info("  Saved centroids: %s", centroids_path)

    # Save k evaluation results
    if eval_results:
        eval_path = output_dir / "k_evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        logger.info("  Saved k evaluation: %s", eval_path)

    # Save metadata
    metadata_path = output_dir / "clustering_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("  Saved metadata: %s", metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="K-Means Clustering for Load Profiles (Euclidean Distance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard run with k evaluation (silhouette on sample)
    python euclidean_clustering.py \\
        --input data/clustering/sampled_profiles.parquet \\
        --output-dir data/clustering/results \\
        --k-range 3 6 --find-optimal-k --normalize \\
        --silhouette-sample-size 10000

    # Fixed k (no evaluation)
    python euclidean_clustering.py \\
        --input data/clustering/sampled_profiles.parquet \\
        --output-dir data/clustering/results \\
        --k 4 --normalize
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to sampled_profiles.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clustering/results"),
        help="Output directory for results",
    )

    # K selection
    k_group = parser.add_argument_group("Cluster Selection")
    k_group.add_argument(
        "--k",
        type=int,
        default=None,
        help="Fixed number of clusters (skip evaluation)",
    )
    k_group.add_argument(
        "--k-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[3, 6],
        help="Range of k values to evaluate (default: 3 6)",
    )
    k_group.add_argument(
        "--find-optimal-k",
        action="store_true",
        help="Evaluate k range and use optimal k",
    )
    k_group.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=10_000,
        help=(
            "Max number of samples for silhouette evaluation "
            "(default: 10000; use -1 to use full dataset, not recommended for large n)."
        ),
    )

    # Clustering parameters
    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="Number of k-means initializations (default: 10)",
    )

    # Preprocessing
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=DEFAULT_NORMALIZE,
        help="Apply normalization to profiles (default: True)",
    )
    parser.add_argument(
        "--normalize-method",
        choices=["zscore", "minmax", "none"],
        default=DEFAULT_NORMALIZATION,
        help="Normalization method (default: minmax)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 2: K-MEANS CLUSTERING (EUCLIDEAN DISTANCE)")
    print("=" * 70)

    # Load profiles
    X, df = load_profiles(args.input)

    logger.info("Loaded %s sampled profiles", f"{len(df):,}")

    # Normalize if requested
    if args.normalize:
        logger.info("Normalizing profiles per household-day (method=%s)", args.normalize_method)
        df = normalize_profiles(
            df,
            method=args.normalize_method,  # ✅ FIXED: was args.normalization_method
            profile_col="profile",
            out_col=None,
        )
        # ✅ CRITICAL FIX: Re-extract normalized profiles as numpy array
        X = np.array(df["profile"].to_list(), dtype=np.float64)
        logger.info("  Normalized data range: [%.2f, %.2f]", X.min(), X.max())
    else:
        logger.info("Profile normalization disabled (using raw kWh values).")

    # Determine k
    eval_results = None

    if args.k is not None:
        # Fixed k
        k = args.k
        logger.info("")
        logger.info("Using fixed k=%d", k)
    elif args.find_optimal_k:
        # Evaluate k range
        k_range = range(args.k_range[0], args.k_range[1] + 1)

        silhouette_sample_size: int | None
        silhouette_sample_size = None if args.silhouette_sample_size < 0 else args.silhouette_sample_size

        eval_results = evaluate_clustering(
            X,
            k_range=k_range,
            n_init=args.n_init,
            random_state=args.random_state,
            silhouette_sample_size=silhouette_sample_size,
        )

        # Save elbow curve
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_elbow_curve(eval_results, args.output_dir / "elbow_curve.png")

        k = find_optimal_k(eval_results)
    else:
        # Default to min of k_range
        k = args.k_range[0]
        logger.info("")
        logger.info("Using default k=%d", k)

    # Run final clustering
    labels, centroids, inertia = run_clustering(
        X,
        k=k,
        n_init=args.n_init,
        random_state=args.random_state,
    )

    # Create visualizations
    logger.info("")
    logger.info("Generating visualizations...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_centroids(centroids, args.output_dir / "cluster_centroids.png")
    plot_cluster_samples(X, labels, centroids, args.output_dir / "cluster_samples.png")

    # Save results
    logger.info("")
    logger.info("Saving results...")

    metadata = {
        "k": int(k),
        "n_profiles": int(X.shape[0]),
        "n_timepoints": int(X.shape[1]),
        "normalized": bool(args.normalize),
        "normalize_method": args.normalize_method if args.normalize else None,
        "n_init": int(args.n_init),
        "random_state": int(args.random_state),
        "inertia": float(inertia),
        "distance_metric": "euclidean",
        "silhouette_sample_size": (None if args.silhouette_sample_size < 0 else int(args.silhouette_sample_size)),
    }

    save_results(df, labels, centroids, eval_results, metadata, args.output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("CLUSTERING COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  • {X.shape[0]:,} profiles clustered into {k} groups")
    print(f"  • Inertia: {inertia:,.2f}")
    if eval_results:
        best_sil = max(eval_results["silhouette"])
        print(f"  • Best silhouette score: {best_sil:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
