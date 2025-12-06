#!/usr/bin/env python3
"""
Phase 2: K-Means Clustering for Load Profile Analysis.

Clusters daily electricity usage profiles using standard Euclidean distance
to identify distinct consumption patterns.

Note: DTW (Dynamic Time Warping) is unnecessary here because all profiles
are aligned to the same 48 half-hourly time grid. Euclidean distance is
appropriate and much faster.

Pipeline:
    1. Load daily profiles from Phase 1
    2. Normalize profiles (optional but recommended)
    3. Evaluate k values to find optimal k (via silhouette score)
    4. Run final clustering with optimal k (or a fixed k)
    5. Output assignments, centroids, diagnostics, and visualizations

Usage:
    # Standard run (evaluates k=3-6, uses best k by silhouette)
    python euclidean_clustering.py \
        --input data/clustering/sampled_profiles.parquet \
        --output-dir data/clustering/results \
        --k-range 3 6 \
        --find-optimal-k \
        --normalize \
        --normalize-method minmax

    # Fixed k (no evaluation)
    python euclidean_clustering.py \
        --input data/clustering/sampled_profiles.parquet \
        --output-dir data/clustering/results \
        --k 4 \
        --normalize \
        --normalize-method minmax
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

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


# =============================================================================
# 1. DATA LOADING & NORMALIZATION
# =============================================================================


def load_profiles(path: Path) -> tuple[np.ndarray, pl.DataFrame]:
    """
    Load profiles from parquet file.

    Args:
        path: Path to sampled_profiles.parquet

    Returns:
        Tuple of (profile_array, metadata_df)
    """
    logger.info("Loading profiles from %s", path)

    df = pl.read_parquet(path)

    if "profile" not in df.columns:
        raise ValueError("Expected a 'profile' column containing the daily load vector.")

    # Extract profiles as numpy array (shape: n_profiles x n_timepoints)
    profiles = np.array(df["profile"].to_list(), dtype=np.float64)

    logger.info("  Loaded %s profiles with %s time points each", f"{len(profiles):,}", profiles.shape[1])
    logger.info("  Data shape: %s", (profiles.shape[0], profiles.shape[1]))
    logger.info("  Data range: [%.2f, %.2f]", profiles.min(), profiles.max())

    return profiles, df


def normalize_profiles(
    X: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """
    Normalize daily load profiles for clustering.

    Two modes:

    - "minmax": per-profile min-max scaling to [0, 1]. This is the
      recommended and default method for nonstationary daily load
      profiles, because it preserves the intraday shape and interprets
      each value as "fraction of that day's peak load."

    - "none": return the raw profiles without any scaling.

    Args:
        X: Profile array of shape (n_samples, n_timepoints).
        method: "minmax" or "none".

    Returns:
        Normalized array with the same shape as X.
    """
    if method == "none":
        logger.info("Skipping profile normalization (method='none').")
        return X

    if method != "minmax":
        raise ValueError(f"Unknown normalization method: {method!r}. Supported methods are 'minmax' and 'none'.")

    logger.info("Normalizing profiles using per-profile min-max scaling...")

    # Per-profile min-max normalization
    mins = X.min(axis=1, keepdims=True)
    maxs = X.max(axis=1, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0  # Avoid division by zero for flat profiles

    X_norm = (X - mins) / ranges

    logger.info("  Normalized data range: [%.2f, %.2f]", X_norm.min(), X_norm.max())
    return X_norm


# =============================================================================
# 2. K SELECTION & CLUSTERING
# =============================================================================


def evaluate_clustering(
    X: np.ndarray,
    k_range: range,
    n_init: int = 10,
    random_state: int = 42,
    keep_best: bool = False,
) -> tuple[dict[str, list[float]], dict[str, Any] | None]:
    """
    Evaluate clustering for different values of k.

    Args:
        X: Profile array of shape (n_samples, n_timepoints)
        k_range: Range of k values to test
        n_init: Number of random initializations
        random_state: Random seed for reproducibility
        keep_best: If True, also return labels/centroids for the
                   best k (by silhouette).

    Returns:
        eval_results: dict with keys
            - "k_values"
            - "inertia"
            - "silhouette"
        best_info: dict with keys
            - "k"
            - "labels"
            - "centroids"
            - "inertia"
            - "silhouette"
          or None if keep_best=False.
    """
    logger.info("Evaluating clustering for k in %s...", list(k_range))
    logger.info("  Dataset size: %s profiles", f"{X.shape[0]:,}")

    results: dict[str, list[float]] = {
        "k_values": [],
        "inertia": [],
        "silhouette": [],
    }

    best_info: dict[str, Any] | None = None

    for k in k_range:
        logger.info("\n  Testing k=%s...", k)

        model = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=random_state,
        )

        labels = model.fit_predict(X)
        inertia = float(model.inertia_)
        sil_score = float(silhouette_score(X, labels, metric="euclidean"))

        results["k_values"].append(k)
        results["inertia"].append(inertia)
        results["silhouette"].append(sil_score)

        logger.info("    Inertia: %s", f"{inertia:,.2f}")
        logger.info("    Silhouette: %.3f", sil_score)

        if keep_best and (best_info is None or sil_score > best_info["silhouette"]):
            best_info = {
                "k": k,
                "labels": labels,
                "centroids": model.cluster_centers_,
                "inertia": inertia,
                "silhouette": sil_score,
            }

    return results, best_info


def find_optimal_k(eval_results: dict[str, list[float]]) -> int:
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

    logger.info("\nOptimal k=%s (silhouette=%.3f)", best_k, silhouettes[best_idx])

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
    logger.info("\nRunning k-means with k=%s on %s profiles...", k, f"{X.shape[0]:,}")

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
        logger.info("  Cluster %s: %s profiles (%.1f%%)", cluster, f"{count:,}", pct)

    return labels, centroids, inertia


# =============================================================================
# 3. PLOTTING
# =============================================================================


def _infer_hours(n_timepoints: int) -> tuple[np.ndarray, str]:
    """Infer x-axis values and label based on number of timepoints."""
    if n_timepoints == 48:
        hours = np.arange(0.5, 24.5, 0.5)
        xlabel = "Hour of Day"
    elif n_timepoints == 24:
        hours = np.arange(1, 25)
        xlabel = "Hour of Day"
    else:
        hours = np.arange(n_timepoints)
        xlabel = "Time Interval"
    return hours, xlabel


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

    hours, xlabel = _infer_hours(n_timepoints)

    _fig, ax = plt.subplots(figsize=(12, 6))

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

    hours, _xlabel = _infer_hours(n_timepoints)

    _fig, axes = plt.subplots(1, k, figsize=(5 * k, 4), sharey=True)
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
    eval_results: dict[str, list[float]],
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

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Inertia (elbow curve)
    ax1.plot(k_values, inertia, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax1.set_ylabel("Inertia", fontsize=12)
    ax1.set_title("Elbow Curve", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)

    # Silhouette score
    ax2.plot(k_values, silhouette, "o-", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("Silhouette Score", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)

    # Mark optimal k
    best_idx = int(np.argmax(silhouette))
    ax2.axvline(x=k_values[best_idx], linestyle="--", alpha=0.7)
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


# =============================================================================
# 4. WEEKDAY/WEEKEND DIAGNOSTICS
# =============================================================================


def compute_cluster_time_diagnostics(assignments: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """
    Compute weekday/weekend and day-of-week breakdowns for each cluster.

    This is aimed at answering:
      - Do we effectively have a "weekday cluster" and a "weekend cluster"?
      - How balanced are clusters across days of week?

    The function expects:
      - a 'cluster' column
      - some date column, preferably 'date_local', otherwise 'date'

    Returns:
        A dict with two lists of dicts, suitable for JSON serialization:
        {
          "cluster_weekend_breakdown": [...],
          "cluster_day_of_week_breakdown": [...]
        }
    """
    if "cluster" not in assignments.columns:
        logger.warning("No 'cluster' column in assignments; skipping weekday/weekend diagnostics.")
        return {}

    date_col = None
    if "date_local" in assignments.columns:
        date_col = "date_local"
    elif "date" in assignments.columns:
        date_col = "date"

    if date_col is None:
        logger.warning("No 'date_local' or 'date' column in assignments; skipping weekday/weekend diagnostics.")
        return {}

    # Add weekday and weekend flags
    df = assignments.with_columns(
        pl.col(date_col).dt.weekday().alias("day_of_week"),
        (pl.col(date_col).dt.weekday() >= 5).alias("is_weekend"),
    )

    # 1) Cluster x weekday/weekend breakdown (by number of days)
    weekday_mix = (
        df.group_by(["cluster", "is_weekend"])
        .agg(pl.len().alias("n_days"))
        .with_columns(
            (pl.col("n_days") / pl.col("n_days").sum().over("cluster") * 100).round(1).alias("pct_of_cluster_days")
        )
        .sort(["cluster", "is_weekend"])
    )

    logger.info("Cluster x weekend breakdown (by days):\n%s", weekday_mix)

    # 2) Cluster x full day-of-week breakdown (0=Mon, 6=Sun)
    dow_mix = (
        df.group_by(["cluster", "day_of_week"])
        .agg(pl.len().alias("n_days"))
        .with_columns(
            (pl.col("n_days") / pl.col("n_days").sum().over("cluster") * 100).round(1).alias("pct_of_cluster_days")
        )
        .sort(["cluster", "day_of_week"])
    )

    logger.info("Cluster x day_of_week breakdown:\n%s", dow_mix)

    # Convert to plain Python for JSON serialization
    diagnostics = {
        "cluster_weekend_breakdown": weekday_mix.to_dicts(),
        "cluster_day_of_week_breakdown": dow_mix.to_dicts(),
    }
    return diagnostics


# =============================================================================
# 5. SAVING RESULTS
# =============================================================================


def save_results(
    df: pl.DataFrame,
    labels: np.ndarray,
    centroids: np.ndarray,
    eval_results: dict[str, list[float]] | None,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Save all clustering results to output directory.

    Args:
        df: Original profile DataFrame with metadata
        labels: Cluster assignments
        centroids: Cluster centroids
        eval_results: K evaluation results (if any)
        metadata: Clustering metadata (will be augmented with diagnostics)
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which ID/date columns are present (household vs ZIP+4 level)
    id_cols: list[str] = []
    if "account_identifier" in df.columns:
        id_cols.append("account_identifier")
    if "zip_code" in df.columns:
        id_cols.append("zip_code")

    # Prefer the more explicit date_local, fall back to date if needed
    date_cols = []
    if "date_local" in df.columns:
        date_cols.append("date_local")
    if "date" in df.columns:
        date_cols.append("date")

    # Additional flags if already present
    extra_cols = [c for c in ["is_weekend", "weekday"] if c in df.columns]

    id_cols.extend(date_cols)
    id_cols.extend(extra_cols)

    available_cols = [c for c in id_cols if c in df.columns]

    # Save cluster assignments (one row per profile/day)
    assignments = df.select(available_cols).with_columns(pl.Series("cluster", labels))
    assignments_path = output_dir / "cluster_assignments.parquet"
    assignments.write_parquet(assignments_path)
    logger.info("  Saved assignments: %s", assignments_path)

    # Compute and save weekday/weekend diagnostics
    diagnostics = compute_cluster_time_diagnostics(assignments)
    if diagnostics:
        # Attach to metadata
        metadata["cluster_time_diagnostics"] = diagnostics

        diag_path = output_dir / "cluster_time_diagnostics.json"
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)
        logger.info("  Saved weekday/weekend diagnostics: %s", diag_path)

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
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)
        logger.info("  Saved k evaluation: %s", eval_path)

    # Save metadata (including diagnostics if computed)
    metadata_path = output_dir / "clustering_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("  Saved metadata: %s", metadata_path)


# =============================================================================
# 6. CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="K-Means Clustering for Load Profiles (Euclidean Distance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard run with k evaluation
    python euclidean_clustering.py \
        --input data/clustering/sampled_profiles.parquet \
        --output-dir data/clustering/results \
        --k-range 3 6 --find-optimal-k --normalize --normalize-method minmax

    # Fixed k (no evaluation)
    python euclidean_clustering.py \
        --input data/clustering/sampled_profiles.parquet \
        --output-dir data/clustering/results \
        --k 4 --normalize --normalize-method minmax
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
        help="Evaluate k range and use optimal k (by silhouette score)",
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
        help="Apply normalization to profiles (see --normalize-method).",
    )
    parser.add_argument(
        "--normalize-method",
        choices=["minmax", "none"],
        default="minmax",
        help=(
            "Normalization method (default: minmax). "
            "Use 'minmax' for per-profile scaling to [0, 1], or 'none' "
            "to use raw kWh values."
        ),
    )

    # Misc
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

    # Normalize if requested
    if args.normalize:
        X = normalize_profiles(X, method=args.normalize_method)

    # Determine k and, if applicable, reuse best model from evaluation
    eval_results: dict[str, list[float]] | None = None
    labels: np.ndarray | None = None
    centroids: np.ndarray | None = None
    inertia: float | None = None

    if args.k is not None:
        # Fixed k
        k = args.k
        logger.info("\nUsing fixed k=%s", k)

        labels, centroids, inertia = run_clustering(
            X,
            k=k,
            n_init=args.n_init,
            random_state=args.random_state,
        )

    elif args.find_optimal_k:
        # Evaluate k range and keep best model
        k_range = range(args.k_range[0], args.k_range[1] + 1)

        eval_results, best_info = evaluate_clustering(
            X,
            k_range=k_range,
            n_init=args.n_init,
            random_state=args.random_state,
            keep_best=True,
        )

        # Save elbow curve
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_elbow_curve(eval_results, args.output_dir / "elbow_curve.png")

        if best_info is None:
            raise RuntimeError("No best model found during k evaluation.")

        k = int(best_info["k"])
        labels = best_info["labels"]
        centroids = best_info["centroids"]
        inertia = float(best_info["inertia"])

        logger.info("\nOptimal k=%s (silhouette=%.3f)", k, best_info["silhouette"])

    else:
        # Default to min of k_range
        k = int(args.k_range[0])
        logger.info("\nUsing default k=%s", k)

        labels, centroids, inertia = run_clustering(
            X,
            k=k,
            n_init=args.n_init,
            random_state=args.random_state,
        )

    # At this point we must have labels/centroids/inertia
    assert labels is not None and centroids is not None and inertia is not None  # noqa: S101

    # Create visualizations
    logger.info("\nGenerating visualizations...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_centroids(centroids, args.output_dir / "cluster_centroids.png")
    plot_cluster_samples(X, labels, centroids, args.output_dir / "cluster_samples.png")

    # Save results (+ diagnostics)
    logger.info("\nSaving results...")

    metadata: dict[str, Any] = {
        "k": int(k),
        "n_profiles": len(X),
        "n_timepoints": int(X.shape[1]),
        "normalized": bool(args.normalize),
        "normalize_method": args.normalize_method if args.normalize else None,
        "n_init": int(args.n_init),
        "random_state": int(args.random_state),
        "inertia": float(inertia),
        "distance_metric": "euclidean",
    }

    save_results(df, labels, centroids, eval_results, metadata, args.output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("CLUSTERING COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  • {len(X):,} profiles clustered into {k} groups")
    print(f"  • Inertia: {inertia:,.2f}")
    if eval_results:
        best_sil = max(eval_results["silhouette"])
        print(f"  • Best silhouette score: {best_sil:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
