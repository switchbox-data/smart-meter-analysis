#!/usr/bin/env python3
"""
MiniBatch K-Means clustering for large load-profile datasets (PyArrow-batched).

Reads an input Parquet file in streaming batches via PyArrow, optionally normalizes
each profile row on-the-fly, fits MiniBatchKMeans with partial_fit, then predicts and
writes cluster assignments back to Parquet incrementally (original columns + `cluster`).

Outputs:
- cluster_assignments.parquet
- cluster_centroids.parquet
- cluster_centroids.png
- clustering_metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------
# IO + batch utilities
# ----------------------------


def parquet_num_rows(path: Path) -> int:
    """Return Parquet row count from file metadata (no full scan)."""
    pf = pq.ParquetFile(path)
    md = pf.metadata
    if md is None:
        return sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
    return int(md.num_rows)


def iter_profile_batches(
    path: Path,
    batch_size: int,
    columns: list[str] | None = None,
) -> Iterable[pa.RecordBatch]:
    """Yield record batches from a Parquet file."""
    pf = pq.ParquetFile(path)
    yield from pf.iter_batches(batch_size=batch_size, columns=columns)


def recordbatch_profiles_to_numpy(rb: pa.RecordBatch, profile_col: str = "profile") -> np.ndarray:
    """Convert RecordBatch `profile` list column into a 2D float64 NumPy array."""
    idx = rb.schema.get_field_index(profile_col)
    if idx < 0:
        raise ValueError(f"RecordBatch missing required column '{profile_col}'")
    profiles = rb.column(idx).to_pylist()
    X = np.asarray(profiles, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D profile array; got shape={X.shape}")
    return X


# ----------------------------
# Normalization
# ----------------------------


def normalize_batch(X: np.ndarray, method: str) -> np.ndarray:
    """Row-wise normalization: minmax, zscore, or none. Constant rows -> zeros."""
    if method in ("none", "", None):
        Xn = X
    elif method == "minmax":
        mins = np.min(X, axis=1, keepdims=True)
        maxs = np.max(X, axis=1, keepdims=True)
        denom = maxs - mins
        denom_safe = np.where(denom == 0, 1.0, denom)
        Xn = (X - mins) / denom_safe
        Xn = np.where(denom == 0, 0.0, Xn)
    elif method == "zscore":
        means = np.mean(X, axis=1, keepdims=True)
        stds = np.std(X, axis=1, keepdims=True)
        std_safe = np.where(stds == 0, 1.0, stds)
        Xn = (X - means) / std_safe
        Xn = np.where(stds == 0, 0.0, Xn)
    else:
        raise ValueError(f"Unknown normalize method: {method}")

    if not np.isfinite(Xn).all():
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    return Xn


# ----------------------------
# Clustering
# ----------------------------


def fit_minibatch_kmeans(
    input_path: Path,
    k: int,
    batch_size: int,
    n_init: int,
    random_state: int,
    normalize: bool,
    normalize_method: str,
) -> MiniBatchKMeans:
    """Fit MiniBatchKMeans by streaming over `profile` batches and calling partial_fit()."""
    logger.info("Fitting MiniBatchKMeans (k=%d, batch_size=%s, n_init=%d)...", k, f"{batch_size:,}", n_init)

    model = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        n_init=n_init,
        random_state=random_state,
        verbose=0,
    )

    total_rows = parquet_num_rows(input_path)
    n_batches = (total_rows + batch_size - 1) // batch_size
    logger.info("  Training on %s profiles in %d batches", f"{total_rows:,}", n_batches)

    seen = 0
    for bi, rb in enumerate(iter_profile_batches(input_path, batch_size=batch_size, columns=["profile"]), start=1):
        X = recordbatch_profiles_to_numpy(rb, profile_col="profile")
        if normalize and normalize_method != "none":
            X = normalize_batch(X, normalize_method)
        model.partial_fit(X)
        seen += X.shape[0]
        if bi % 10 == 0 or bi == n_batches:
            logger.info("    Trained batch %d/%d (seen=%s)", bi, n_batches, f"{seen:,}")

    logger.info("  Training complete. Inertia: %s", f"{float(model.inertia_):,.2f}")
    return model


# ruff: noqa: C901
def predict_and_write_assignments_streaming(
    model: MiniBatchKMeans,
    input_path: Path,
    output_path: Path,
    batch_size: int,
    normalize: bool,
    normalize_method: str,
    silhouette_sample_idx: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Predict labels in batches, write output Parquet incrementally, and optionally
    compute silhouette on a sampled set of global row indices.
    """
    logger.info("Predicting labels + writing assignments streaming...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(input_path)
    out_schema = pf.schema_arrow.append(pa.field("cluster", pa.int32()))
    writer = pq.ParquetWriter(output_path, out_schema, compression="zstd")

    k = int(model.n_clusters)
    counts = np.zeros(k, dtype=np.int64)

    sample_X: list[np.ndarray] = None
    sample_y: list[int] = None
    sample_pos = 0
    if silhouette_sample_idx is not None and len(silhouette_sample_idx) > 0:
        sample_X, sample_y = [], []

    global_row = 0
    try:
        for rb in iter_profile_batches(input_path, batch_size=batch_size, columns=None):
            X = recordbatch_profiles_to_numpy(rb, profile_col="profile")
            if normalize and normalize_method != "none":
                X = normalize_batch(X, normalize_method)

            labels = model.predict(X).astype(np.int32)
            counts += np.bincount(labels, minlength=k)

            if sample_X is not None and sample_y is not None:
                n = X.shape[0]
                while sample_pos < len(silhouette_sample_idx) and silhouette_sample_idx[sample_pos] < global_row:
                    sample_pos += 1
                start = sample_pos
                while sample_pos < len(silhouette_sample_idx) and silhouette_sample_idx[sample_pos] < global_row + n:
                    sample_pos += 1
                if sample_pos > start:
                    idx_in_batch = silhouette_sample_idx[start:sample_pos] - global_row
                    for j in idx_in_batch:
                        jj = int(j)
                        sample_X.append(X[jj])
                        sample_y.append(int(labels[jj]))

            out_rb = rb.append_column("cluster", pa.array(labels, type=pa.int32()))
            writer.write_batch(out_rb)
            global_row += labels.shape[0]
    finally:
        writer.close()

    logger.info("  Cluster distribution:")
    total = int(counts.sum())
    for c, n in enumerate(counts.tolist()):
        pct = (n / total * 100.0) if total > 0 else 0.0
        logger.info("    Cluster %d: %s profiles (%.1f%%)", c, f"{n:,}", pct)

    sil = None
    if sample_X is not None and sample_y is not None and len(sample_y) >= 2:
        Xs = np.asarray(sample_X, dtype=np.float64)
        ys = np.asarray(sample_y, dtype=np.int32)
        logger.info("Computing silhouette on sample_size=%s ...", f"{len(ys):,}")
        sil = float(silhouette_score(Xs, ys, metric="euclidean"))
        logger.info("  Silhouette score (sample): %.3f", sil)

    return counts, sil


# ----------------------------
# Plotting + outputs
# ----------------------------


def plot_centroids(centroids: np.ndarray, output_path: Path) -> None:
    """Save a line plot of cluster centroids."""
    k = int(centroids.shape[0])
    n_timepoints = int(centroids.shape[1])

    if n_timepoints == 48:
        x = np.arange(0.5, 24.5, 0.5)
        xlabel = "Hour of Day"
    else:
        x = np.arange(n_timepoints)
        xlabel = "Time Interval"

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(k):
        ax.plot(x, centroids[i], label=f"Cluster {i}", linewidth=2)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Usage (normalized)" if n_timepoints else "Usage", fontsize=12)
    ax.set_title("Cluster Centroids (MiniBatch K-Means)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if n_timepoints == 48:
        ax.set_xticks(range(0, 25, 4))
        ax.set_xlim(0, 24)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("  Saved centroids plot: %s", output_path)


def save_centroids_parquet(centroids: np.ndarray, output_path: Path) -> None:
    """Write centroids to Parquet as (cluster, centroid[list[float]])."""
    centroids_df = pl.DataFrame({
        "cluster": list(range(int(centroids.shape[0]))),
        "centroid": [c.tolist() for c in centroids],
    })
    centroids_df.write_parquet(output_path)
    logger.info("  Saved centroids parquet: %s", output_path)


def save_metadata(metadata: dict, output_path: Path) -> None:
    """Write run metadata and summary metrics to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("  Saved metadata: %s", output_path)


# ----------------------------
# Main
# ----------------------------


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Memory-Efficient K-Means Clustering (MiniBatch, PyArrow-batched)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="sampled_profiles.parquet")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters")
    parser.add_argument("--normalize", action="store_true", help="Normalize profiles per row")
    parser.add_argument("--normalize-method", choices=["minmax", "zscore", "none"], default="minmax")
    parser.add_argument("--batch-size", type=int, default=50_000, help="Batch size (default: 50k)")
    parser.add_argument("--n-init", type=int, default=3, help="Number of initializations (default: 3)")
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=5_000,
        help="Sample size for silhouette (default: 5000; set 0 to skip)",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MINIBATCH K-MEANS CLUSTERING (PYARROW-BATCHED)")
    logger.info("=" * 70)
    logger.info("Input: %s", args.input)
    logger.info("k: %d", args.k)
    logger.info("Batch size: %s", f"{args.batch_size:,}")

    eff_norm = args.normalize and args.normalize_method != "none"
    logger.info("Normalize: %s (method=%s)", eff_norm, args.normalize_method if eff_norm else "none")

    n_profiles = parquet_num_rows(args.input)
    logger.info("Profiles (from parquet metadata): %s", f"{n_profiles:,}")

    model = fit_minibatch_kmeans(
        input_path=args.input,
        k=args.k,
        batch_size=args.batch_size,
        n_init=args.n_init,
        random_state=args.random_state,
        normalize=bool(eff_norm),
        normalize_method=args.normalize_method,
    )
    centroids = model.cluster_centers_

    silhouette_sample_idx = None
    if args.silhouette_sample_size and args.silhouette_sample_size > 0:
        if args.silhouette_sample_size >= n_profiles:
            silhouette_sample_idx = np.arange(n_profiles, dtype=np.int64)
        else:
            rng = np.random.default_rng(args.random_state)
            silhouette_sample_idx = rng.choice(n_profiles, size=args.silhouette_sample_size, replace=False)
            silhouette_sample_idx.sort()

    assignments_path = args.output_dir / "cluster_assignments.parquet"
    counts, sil_score = predict_and_write_assignments_streaming(
        model=model,
        input_path=args.input,
        output_path=assignments_path,
        batch_size=args.batch_size,
        normalize=bool(eff_norm),
        normalize_method=args.normalize_method,
        silhouette_sample_idx=silhouette_sample_idx,
    )

    plot_centroids(centroids, args.output_dir / "cluster_centroids.png")
    save_centroids_parquet(centroids, args.output_dir / "cluster_centroids.parquet")

    metadata = {
        "k": int(args.k),
        "n_profiles": int(n_profiles),
        "n_timepoints": int(centroids.shape[1]),
        "normalized": bool(eff_norm),
        "normalize_method": args.normalize_method if eff_norm else "none",
        "batch_size": int(args.batch_size),
        "n_init": int(args.n_init),
        "random_state": int(args.random_state),
        "algorithm": "MiniBatchKMeans",
        "inertia": float(model.inertia_),
        "silhouette_score_sample": float(sil_score) if sil_score is not None else None,
        "cluster_counts": {str(i): int(c) for i, c in enumerate(counts.tolist())},
    }
    save_metadata(metadata, args.output_dir / "clustering_metadata.json")

    logger.info("=" * 70)
    logger.info("CLUSTERING COMPLETE")
    logger.info("=" * 70)
    logger.info("Profiles: %s", f"{n_profiles:,}")
    logger.info("Clusters: %d", args.k)
    if sil_score is not None:
        logger.info("Silhouette (sample): %.3f", sil_score)
    logger.info("Output: %s", args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
