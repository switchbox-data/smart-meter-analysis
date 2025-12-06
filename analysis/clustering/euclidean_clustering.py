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
    4. Run final clustering with optimal k
    5. Output assignments, centroids, and visualizations

Usage:
    # Standard run (evaluates k=3-6)
    python euclidean_clustering.py \\
        --input data/clustering/sampled_profiles.parquet \\
        --output-dir data/clustering/results \\
        --k-range 3 6 \\
        --find-optimal-k \\
        --normalize

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

    logger.info(f"  Loaded {len(profiles):,} profiles with {profiles.shape[1]} time points each")
    logger.info(f"  Data shape: {profiles.shape}")
    logger.info(f"  Data range: [{profiles.min():.2f}, {profiles.max():.2f}]")

    return profiles, df


def normalize_profiles(
    X: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """
    Normalize daily load profiles for clustering.

    Two modes:

    - "minmax": per-profile minmax scaling to [0, 1]. This is the
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


def evaluate_clustering(
    X: np.ndarray,
    k_range: range,
    n_init: int = 10,
    random_state: int = 42,
    keep_best: bool = False,
) -> tuple[dict, dict | None]:
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
        eval_results: dict with k_values, inertia, and silhouette scores
        best_info: dict with keys
            - "k"
            - "labels"
            - "centroids"
            - "inertia"
            - "silhouette"
          or None if keep_best=False.
    """
    logger.info(f"Evaluating clustering for k in {list(k_range)}...")
    logger.info(f"  Dataset size: {X.shape[0]:,} profiles")

    results = {
        "k_values": [],
        "inertia": [],
        "silhouette": [],
    }

    best_info: dict | None = None

    for k in k_range:
        logger.info(f"\n  Testing k={k}...")

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

        logger.info(f"    Inertia: {inertia:,.2f}")
        logger.info(f"    Silhouette: {sil_score:.3f}")

        if keep_best and (best_info is None or sil_score > best_info["silhouette"]):
            best_info = {
                "k": k,
                "labels": labels,
                "centroids": model.cluster_centers_,
                "inertia": inertia,
                "silhouette": sil_score,
            }

    return results, best_info


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

    best_idx = np.argmax(silhouettes)
    best_k = k_values[best_idx]

    logger.info(f"\nOptimal k={best_k} (silhouette={silhouettes[best_idx]:.3f})")

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
    logger.info(f"\nRunning k-means with k={k} on {X.shape[0]:,} profiles...")

    model = KMeans(
        n_clusters=k,
        n_init=n_init,
        random_state=random_state,
    )

    labels = model.fit_predict(X)
    centroids = model.cluster_centers_

    logger.info(f"  Inertia: {model.inertia_:,.2f}")

    # Log cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        pct = count / len(labels) * 100
        logger.info(f"  Cluster {cluster}: {count:,} profiles ({pct:.1f}%)")

    return labels, centroids, float(model.inertia_)


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

    logger.info(f"  Saved centroids plot: {output_path}")


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

    _fig, axes = plt.subplots(1, k, figsize=(5 * k, 4), sharey=True)
    if k == 1:
        axes = [axes]

    rng = np.random.default_rng(random_state)
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for i, (ax, color) in enumerate(zip(axes, colors)):
        cluster_mask = labels == i
        cluster_profiles = X[cluster_mask]

        # Sample profiles
        n_available = len(cluster_profiles)
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

    logger.info(f"  Saved cluster samples plot: {output_path}")


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

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

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
    best_idx = np.argmax(silhouette)
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

    logger.info(f"  Saved elbow curve: {output_path}")


def save_results(
    df: pl.DataFrame,
    labels: np.ndarray,
    centroids: np.ndarray,
    eval_results: dict,
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
    id_cols = []
    if "account_identifier" in df.columns:
        id_cols.append("account_identifier")
    if "zip_code" in df.columns:
        id_cols.append("zip_code")
    id_cols.extend(["date", "is_weekend", "weekday"])

    # Only include columns that exist
    available_cols = [c for c in id_cols if c in df.columns]

    # Save cluster assignments
    assignments = df.select(available_cols).with_columns(pl.Series("cluster", labels))
    assignments_path = output_dir / "cluster_assignments.parquet"
    assignments.write_parquet(assignments_path)
    logger.info(f"  Saved assignments: {assignments_path}")

    # Save centroids as parquet
    centroids_df = pl.DataFrame({
        "cluster": list(range(len(centroids))),
        "centroid": [c.tolist() for c in centroids],
    })
    centroids_path = output_dir / "cluster_centroids.parquet"
    centroids_df.write_parquet(centroids_path)
    logger.info(f"  Saved centroids: {centroids_path}")

    # Save k evaluation results
    if eval_results:
        eval_path = output_dir / "k_evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"  Saved k evaluation: {eval_path}")

    # Save metadata
    metadata_path = output_dir / "clustering_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved metadata: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="K-Means Clustering for Load Profiles (Euclidean Distance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard run with k evaluation
    python euclidean_clustering.py \\
        --input data/clustering/sampled_profiles.parquet \\
        --output-dir data/clustering/results \\
        --k-range 3 6 --find-optimal-k --normalize

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
        help="Apply z-score normalization to profiles",
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

    # Determine k
    eval_results = None

    # Determine k and (if available) reuse best model from evaluation
    eval_results: dict | None = None
    labels: np.ndarray | None = None
    centroids: np.ndarray | None = None
    inertia: float | None = None

    if args.k is not None:
        # Fixed k
        k = args.k
        logger.info(f"\nUsing fixed k={k}")

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

        k = best_info["k"]
        labels = best_info["labels"]
        centroids = best_info["centroids"]
        inertia = best_info["inertia"]

        logger.info(f"\nOptimal k={k} (silhouette={best_info['silhouette']:.3f})")

    else:
        # Default to min of k_range
        k = args.k_range[0]
        logger.info(f"\nUsing default k={k}")

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

    # Save results
    logger.info("\nSaving results...")

    metadata = {
        "k": k,
        "n_profiles": len(X),
        "n_timepoints": X.shape[1],
        "normalized": args.normalize,
        "normalize_method": args.normalize_method if args.normalize else None,
        "n_init": args.n_init,
        "random_state": args.random_state,
        "inertia": inertia,
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
