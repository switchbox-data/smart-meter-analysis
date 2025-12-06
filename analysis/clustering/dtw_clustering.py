"""
Phase 2: DTW K-Means Clustering for Load Profile Analysis.

Clusters daily electricity usage profiles using Dynamic Time Warping (DTW)
distance metric to identify consumption patterns.

Performance Optimization:
    - K evaluation uses a subsample (default 2000 profiles) for speed
    - Final clustering runs on full dataset with optimal k
    - Configurable max_iter and n_init for speed vs quality tradeoff

Pipeline:
    1. Load daily profiles from Phase 1
    2. Normalize profiles (optional)
    3. Evaluate k values on subsample to find optimal k
    4. Run final clustering on full dataset with optimal k
    5. Output assignments, centroids, and visualizations

Usage:
    # Standard run (evaluates k=3-6, uses subsample for evaluation)
    python dtw_clustering.py \\
        --input data/clustering/sampled_profiles.parquet \\
        --output-dir data/clustering/results \\
        --k-range 3 6 \\
        --find-optimal-k \\
        --normalize

    # Fast validation run
    python dtw_clustering.py \\
        --input data/clustering/sampled_profiles.parquet \\
        --output-dir data/clustering/results \\
        --k-range 3 4 \\
        --max-eval-samples 1000 \\
        --eval-max-iter 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

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
    method: str = "zscore",
) -> np.ndarray:
    """
    Normalize profiles for clustering.

    Args:
        X: Profile array of shape (n_samples, n_timepoints)
        method: Normalization method ('zscore', 'minmax', 'none')

    Returns:
        Normalized array
    """
    if method == "none":
        return X

    logger.info(f"Normalizing profiles using {method} method...")

    if method == "zscore":
        # Per-profile z-score normalization
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True)
        stds[stds == 0] = 1  # Avoid division by zero
        X_norm = (X - means) / stds
    elif method == "minmax":
        # Per-profile min-max normalization
        mins = X.min(axis=1, keepdims=True)
        maxs = X.max(axis=1, keepdims=True)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        X_norm = (X - mins) / ranges
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    logger.info(f"  Normalized data range: [{X_norm.min():.2f}, {X_norm.max():.2f}]")

    return X_norm


def evaluate_clustering(
    X: np.ndarray,
    k_range: range,
    max_iter: int = 10,
    n_init: int = 3,
    random_state: int = 42,
    max_eval_samples: int = 2000,
) -> dict:
    """
    Evaluate clustering for different values of k using a subsample.

    Uses a random subsample for evaluation to reduce runtime while still
    providing reliable estimates of optimal k.

    Args:
        X: Profile array of shape (n_samples, n_timepoints)
        k_range: Range of k values to test
        max_iter: Maximum iterations per k-means run
        n_init: Number of random initializations
        random_state: Random seed for reproducibility
        max_eval_samples: Maximum profiles to use for evaluation

    Returns:
        Dictionary with k_values, inertia, and silhouette scores
    """
    from sklearn.metrics import silhouette_score
    from tslearn.clustering import TimeSeriesKMeans

    logger.info(f"Evaluating clustering for k in {list(k_range)}...")

    # Subsample for evaluation if dataset is large
    if X.shape[0] > max_eval_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=max_eval_samples, replace=False)
        X_eval = X[idx]
        logger.info(f"  Using subsample of {max_eval_samples:,} profiles for k evaluation")
        logger.info(f"  (Full dataset: {X.shape[0]:,} profiles will be used for final clustering)")
    else:
        X_eval = X
        logger.info(f"  Using all {X_eval.shape[0]:,} profiles for evaluation")

    results = {
        "k_values": [],
        "inertia": [],
        "silhouette": [],
    }

    # Reshape for tslearn (n_samples, n_timepoints, n_features)
    X_reshaped = X_eval.reshape(X_eval.shape[0], X_eval.shape[1], 1)

    for k in k_range:
        logger.info(f"\n  Testing k={k}...")

        model = TimeSeriesKMeans(
            n_clusters=k,
            metric="dtw",
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
        )

        labels = model.fit_predict(X_reshaped)

        # Use Euclidean distance for silhouette (faster, still informative)
        sil_score = silhouette_score(X_eval, labels, metric="euclidean")

        results["k_values"].append(k)
        results["inertia"].append(float(model.inertia_))
        results["silhouette"].append(float(sil_score))

        logger.info(f"    Inertia: {model.inertia_:.2f}")
        logger.info(f"    Silhouette: {sil_score:.3f}")

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

    best_idx = np.argmax(silhouettes)
    best_k = k_values[best_idx]

    logger.info(f"\nOptimal k={best_k} (silhouette={silhouettes[best_idx]:.3f})")

    return best_k


def run_final_clustering(
    X: np.ndarray,
    k: int,
    max_iter: int = 10,
    n_init: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run final clustering on full dataset with chosen k.

    Args:
        X: Full profile array
        k: Number of clusters
        max_iter: Maximum iterations
        n_init: Number of random initializations
        random_state: Random seed

    Returns:
        Tuple of (labels, centroids, inertia)
    """
    from tslearn.clustering import TimeSeriesKMeans

    logger.info(f"\nRunning final clustering with k={k} on {X.shape[0]:,} profiles...")

    X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)

    model = TimeSeriesKMeans(
        n_clusters=k,
        metric="dtw",
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    labels = model.fit_predict(X_reshaped)
    centroids = model.cluster_centers_.squeeze()  # Remove extra dimension

    logger.info(f"  Final inertia: {model.inertia_:.2f}")

    # Log cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        pct = count / len(labels) * 100
        logger.info(f"  Cluster {cluster}: {count:,} profiles ({pct:.1f}%)")

    return labels, centroids, float(model.inertia_)


def plot_centroids(
    centroids: np.ndarray,
    output_path: Path,
    title: str = "Cluster Centroids (Average Load Profiles)",
) -> None:
    """
    Plot cluster centroids showing typical daily patterns.

    Args:
        centroids: Centroid array of shape (k, n_timepoints)
        output_path: Path to save plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    hours = np.arange(0, 24, 0.5)  # 48 half-hour intervals

    colors = plt.cm.tab10(np.linspace(0, 1, len(centroids)))

    for i, (centroid, color) in enumerate(zip(centroids, colors)):
        ax.plot(hours, centroid, label=f"Cluster {i}", color=color, linewidth=2)

    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Normalized Energy Usage", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right")
    ax.set_xticks(range(0, 25, 3))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"  Saved centroid plot: {output_path}")


def plot_cluster_samples(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    output_path: Path,
    n_samples: int = 50,
) -> None:
    """
    Plot sample profiles from each cluster with centroid overlay.

    Args:
        X: Profile array
        labels: Cluster assignments
        centroids: Cluster centroids
        output_path: Path to save plot
        n_samples: Number of sample profiles per cluster
    """
    k = len(centroids)
    fig, axes = plt.subplots(1, k, figsize=(5 * k, 4), sharey=True)

    if k == 1:
        axes = [axes]

    hours = np.arange(0, 24, 0.5)

    for i, ax in enumerate(axes):
        cluster_mask = labels == i
        cluster_profiles = X[cluster_mask]

        # Sample profiles to plot
        n_to_plot = min(n_samples, len(cluster_profiles))
        if n_to_plot > 0:
            idx = np.random.choice(len(cluster_profiles), n_to_plot, replace=False)
            for profile in cluster_profiles[idx]:
                ax.plot(hours, profile, alpha=0.2, color="gray", linewidth=0.5)

        # Plot centroid
        ax.plot(hours, centroids[i], color="red", linewidth=2, label="Centroid")

        ax.set_title(f"Cluster {i} (n={cluster_mask.sum():,})")
        ax.set_xlabel("Hour of Day")
        if i == 0:
            ax.set_ylabel("Normalized Usage")
        ax.set_xticks(range(0, 25, 6))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"  Saved cluster samples plot: {output_path}")


def plot_elbow_curve(
    eval_results: dict,
    output_path: Path,
) -> None:
    """
    Plot elbow curve showing inertia and silhouette vs k.

    Args:
        eval_results: Results from evaluate_clustering
        output_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    k_values = eval_results["k_values"]

    # Inertia plot
    ax1.plot(k_values, eval_results["inertia"], "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Curve")
    ax1.grid(True, alpha=0.3)

    # Silhouette plot
    ax2.plot(k_values, eval_results["silhouette"], "go-", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score vs k")
    ax2.grid(True, alpha=0.3)

    # Mark best k
    best_idx = np.argmax(eval_results["silhouette"])
    ax2.axvline(x=k_values[best_idx], color="red", linestyle="--", alpha=0.7)
    ax2.annotate(
        f"Best k={k_values[best_idx]}",
        xy=(k_values[best_idx], eval_results["silhouette"][best_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        color="red",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"  Saved elbow curve: {output_path}")


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

    # Save cluster assignments
    assignments = df.select(["zip_code", "date", "is_weekend", "weekday"]).with_columns(pl.Series("cluster", labels))
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
        description="DTW K-Means Clustering for Load Profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard run with k evaluation
    python dtw_clustering.py \\
        --input data/clustering/sampled_profiles.parquet \\
        --output-dir data/clustering/results \\
        --k-range 3 6 --find-optimal-k --normalize

    # Fast run for testing
    python dtw_clustering.py \\
        --input data/clustering/sampled_profiles.parquet \\
        --output-dir data/clustering/results \\
        --k-range 3 4 --max-eval-samples 1000 --eval-max-iter 5

    # Fixed k (no evaluation)
    python dtw_clustering.py \\
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

    # Performance tuning
    perf_group = parser.add_argument_group("Performance Tuning")
    perf_group.add_argument(
        "--max-eval-samples",
        type=int,
        default=2000,
        help="Max profiles for k evaluation subsample (default: 2000)",
    )
    perf_group.add_argument(
        "--eval-max-iter",
        type=int,
        default=10,
        help="Max iterations for k evaluation runs (default: 10)",
    )
    perf_group.add_argument(
        "--eval-n-init",
        type=int,
        default=3,
        help="Number of initializations for k evaluation (default: 3)",
    )
    perf_group.add_argument(
        "--final-max-iter",
        type=int,
        default=10,
        help="Max iterations for final clustering (default: 10)",
    )
    perf_group.add_argument(
        "--final-n-init",
        type=int,
        default=3,
        help="Number of initializations for final clustering (default: 3)",
    )

    # Preprocessing
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply z-score normalization to profiles",
    )
    parser.add_argument(
        "--normalize-method",
        choices=["zscore", "minmax", "none"],
        default="zscore",
        help="Normalization method (default: zscore)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 2: DTW K-MEANS CLUSTERING")
    print("=" * 70)

    # Load profiles
    X, df = load_profiles(args.input)

    # Normalize if requested
    if args.normalize:
        X = normalize_profiles(X, method=args.normalize_method)

    # Determine k
    eval_results = None

    if args.k is not None:
        # Fixed k
        k = args.k
        logger.info(f"\nUsing fixed k={k}")
    elif args.find_optimal_k:
        # Evaluate k range on subsample
        k_range = range(args.k_range[0], args.k_range[1] + 1)

        eval_results = evaluate_clustering(
            X,
            k_range=k_range,
            max_iter=args.eval_max_iter,
            n_init=args.eval_n_init,
            random_state=args.random_state,
            max_eval_samples=args.max_eval_samples,
        )

        # Save elbow curve
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_elbow_curve(eval_results, args.output_dir / "elbow_curve.png")

        k = find_optimal_k(eval_results)
    else:
        # Default to min of k_range
        k = args.k_range[0]
        logger.info(f"\nUsing default k={k}")

    # Run final clustering on full dataset
    labels, centroids, inertia = run_final_clustering(
        X,
        k=k,
        max_iter=args.final_max_iter,
        n_init=args.final_n_init,
        random_state=args.random_state,
    )

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
        "max_iter": args.final_max_iter,
        "n_init": args.final_n_init,
        "random_state": args.random_state,
        "inertia": inertia,
        "eval_max_samples": args.max_eval_samples if args.find_optimal_k else None,
    }

    save_results(df, labels, centroids, eval_results, metadata, args.output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("CLUSTERING COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  • {len(X):,} profiles clustered into {k} groups")
    print(f"  • Inertia: {inertia:.2f}")
    if eval_results:
        best_sil = max(eval_results["silhouette"])
        print(f"  • Best silhouette score: {best_sil:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
