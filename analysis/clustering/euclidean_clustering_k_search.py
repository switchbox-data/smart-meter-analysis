#!/usr/bin/env python3
"""
MiniBatch K-Means clustering for large load-profile datasets (PyArrow-batched).

This script is the Stage 1 clustering step in the ComEd pipeline. It reads
`sampled_profiles.parquet` in streaming batches via PyArrow, optionally normalizes
profiles row-wise, fits MiniBatchKMeans with partial_fit, then predicts and writes
cluster assignments back to Parquet incrementally.

Modes:
- k-range evaluation (default): evaluate k in [k_min, k_max], compute silhouette on a
  deterministic sample, select best k, then write final artifacts for the selected k.

Outputs (written to --output-dir):
- cluster_assignments.parquet          (for selected/best k)
- cluster_centroids.parquet
- cluster_centroids.png
- clustering_metadata.json
- k_evaluation.json                    (per-k summary metrics)
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# IO + batch utilities
# =============================================================================


def parquet_num_rows(path: Path) -> int:
    """Return Parquet row count from file metadata (no full scan)."""
    pf = pq.ParquetFile(path)
    md = pf.metadata
    if md is None:
        # Extremely rare; ParquetFile.metadata is normally present.
        total = 0
        for i in range(pf.num_row_groups):
            rg = pf.metadata.row_group(i)  # type: ignore[union-attr]
            total += int(rg.num_rows)
        return total
    return int(md.num_rows)


def iter_profile_batches(
    path: Path,
    batch_size: int,
    columns: list[str] | None = None,
) -> Iterable[pa.RecordBatch]:
    """Yield RecordBatches from a Parquet file."""
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


# =============================================================================
# Normalization
# =============================================================================


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


# =============================================================================
# Clustering primitives
# =============================================================================


def fit_minibatch_kmeans(
    input_path: Path,
    k: int,
    batch_size: int,
    n_init: int,
    random_state: int,
    normalize: bool,
    normalize_method: str,
) -> MiniBatchKMeans:
    """Fit MiniBatchKMeans by streaming over batches and calling partial_fit()."""
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
        seen += int(X.shape[0])
        if bi % 10 == 0 or bi == n_batches:
            logger.info("    Trained batch %d/%d (seen=%s)", bi, n_batches, f"{seen:,}")

    logger.info("  Training complete. Inertia: %s", f"{float(model.inertia_):,.2f}")
    return model


def compute_silhouette_on_sample(  # noqa: C901
    model: MiniBatchKMeans,
    input_path: Path,
    batch_size: int,
    normalize: bool,
    normalize_method: str,
    sample_idx: np.ndarray,
) -> float | None:
    """
    Compute silhouette on a deterministic set of global row indices, streaming through the file.

    Returns None if sample size < 2 or if all sampled points end up in one cluster.
    """
    if sample_idx.size < 2:
        return None

    sample_X: list[np.ndarray] = []
    sample_y: list[int] = []

    global_row = 0
    sample_pos = 0
    for rb in iter_profile_batches(input_path, batch_size=batch_size, columns=["profile"]):
        X = recordbatch_profiles_to_numpy(rb, profile_col="profile")
        if normalize and normalize_method != "none":
            X = normalize_batch(X, normalize_method)

        labels = model.predict(X).astype(np.int32)

        n = int(X.shape[0])
        while sample_pos < sample_idx.size and int(sample_idx[sample_pos]) < global_row:
            sample_pos += 1

        start = sample_pos
        while sample_pos < sample_idx.size and int(sample_idx[sample_pos]) < global_row + n:
            sample_pos += 1

        if sample_pos > start:
            idx_in_batch = sample_idx[start:sample_pos] - global_row
            for j in idx_in_batch:
                jj = int(j)
                sample_X.append(X[jj])
                sample_y.append(int(labels[jj]))

        global_row += n
        if sample_pos >= sample_idx.size:
            break

    if len(sample_y) < 2:
        return None

    ys = np.asarray(sample_y, dtype=np.int32)
    if np.unique(ys).size < 2:
        return None

    Xs = np.asarray(sample_X, dtype=np.float64)
    logger.info("Computing silhouette on sample_size=%s ...", f"{len(ys):,}")
    return float(silhouette_score(Xs, ys, metric="euclidean"))


def predict_and_write_assignments_streaming(
    model: MiniBatchKMeans,
    input_path: Path,
    output_path: Path,
    batch_size: int,
    normalize: bool,
    normalize_method: str,
) -> np.ndarray:
    """Predict labels in batches and write output Parquet incrementally (input columns + `cluster`)."""
    logger.info("Predicting labels + writing assignments streaming: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(input_path)
    out_schema = pf.schema_arrow.append(pa.field("cluster", pa.int32()))
    writer = pq.ParquetWriter(output_path, out_schema, compression="snappy")

    k = int(model.n_clusters)
    counts = np.zeros(k, dtype=np.int64)

    try:
        for rb in iter_profile_batches(input_path, batch_size=batch_size, columns=None):
            X = recordbatch_profiles_to_numpy(rb, profile_col="profile")
            if normalize and normalize_method != "none":
                X = normalize_batch(X, normalize_method)

            labels = model.predict(X).astype(np.int32)
            counts += np.bincount(labels, minlength=k)

            out_rb = rb.append_column("cluster", pa.array(labels, type=pa.int32()))
            writer.write_batch(out_rb)
    finally:
        writer.close()

    total = int(counts.sum())
    logger.info("  Cluster distribution (total=%s):", f"{total:,}")
    for c, n in enumerate(counts.tolist()):
        pct = (n / total * 100.0) if total > 0 else 0.0
        logger.info("    Cluster %d: %s profiles (%.1f%%)", c, f"{n:,}", pct)

    return counts


# =============================================================================
# Plotting + outputs
# =============================================================================


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

    _fig, ax = plt.subplots(figsize=(12, 6))
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


def save_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    """Write run metadata and summary metrics to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("  Saved metadata: %s", output_path)


# =============================================================================
# k-range evaluation
# =============================================================================


@dataclass(frozen=True)
class KEvalResult:
    k: int
    inertia: float
    silhouette_score_sample: float | None
    n_profiles: int
    normalized: bool
    normalize_method: str
    batch_size: int
    n_init: int
    random_state: int


def choose_best_k(results: list[KEvalResult]) -> int:
    """
    Choose best k. Primary criterion: max silhouette_score_sample when available.
    Fallback: min inertia.
    """
    with_sil = [r for r in results if r.silhouette_score_sample is not None]
    if with_sil:
        # Tie-breakers: higher silhouette, then higher k (to be deterministic).
        with_sil_sorted = sorted(with_sil, key=lambda r: (r.silhouette_score_sample, r.k), reverse=True)
        return int(with_sil_sorted[0].k)

    # If silhouette unavailable for all, pick minimum inertia (still deterministic).
    by_inertia = sorted(results, key=lambda r: (r.inertia, r.k))
    return int(by_inertia[0].k)


def make_silhouette_sample_idx(n_profiles: int, sample_size: int, seed: int) -> np.ndarray:
    """Deterministic global row index sample."""
    if sample_size <= 0 or n_profiles <= 0:
        return np.array([], dtype=np.int64)
    if sample_size >= n_profiles:
        return np.arange(n_profiles, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n_profiles, size=int(sample_size), replace=False)
    idx.sort()
    return idx.astype(np.int64)


# =============================================================================
# Main
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MiniBatch K-Means Clustering (k-range, PyArrow-batched)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True, help="Path to sampled_profiles.parquet")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory (run_dir/clustering)")

    # Orchestrator-required interface
    p.add_argument("--k-min", type=int, default=3, help="Minimum k (inclusive)")
    p.add_argument("--k-max", type=int, default=6, help="Maximum k (inclusive)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (model + sampling)")

    # Tunables
    p.add_argument("--batch-size", type=int, default=50_000, help="Batch size (default: 50k)")
    p.add_argument("--n-init", type=int, default=3, help="MiniBatchKMeans n_init (default: 3)")
    p.add_argument("--normalize", action="store_true", help="Normalize profiles per row")
    p.add_argument("--normalize-method", choices=["minmax", "zscore", "none"], default="minmax")
    p.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=5_000,
        help="Sample size for silhouette (default: 5000; set 0 to skip)",
    )

    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.k_min <= 1 or args.k_max <= 1:
        raise ValueError("k-min and k-max must be >= 2")
    if args.k_min > args.k_max:
        raise ValueError(f"Invalid k range: k-min ({args.k_min}) > k-max ({args.k_max})")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MINIBATCH K-MEANS CLUSTERING (K-RANGE, PYARROW-BATCHED)")
    logger.info("=" * 70)
    logger.info("Input: %s", args.input)
    logger.info("Output: %s", args.output_dir)
    logger.info("k range: %d..%d", args.k_min, args.k_max)
    logger.info("batch_size: %s", f"{args.batch_size:,}")
    logger.info("n_init: %d", args.n_init)
    logger.info("seed: %d", args.seed)

    eff_norm = bool(args.normalize and args.normalize_method != "none")
    logger.info("Normalize: %s (method=%s)", eff_norm, args.normalize_method if eff_norm else "none")

    n_profiles = parquet_num_rows(args.input)
    logger.info("Profiles (from parquet metadata): %s", f"{n_profiles:,}")

    sample_idx = make_silhouette_sample_idx(int(n_profiles), int(args.silhouette_sample_size), int(args.seed))
    if sample_idx.size > 0:
        logger.info("Silhouette sample size: %s", f"{sample_idx.size:,}")
    else:
        logger.info("Silhouette: disabled")

    results: list[KEvalResult] = []

    # Evaluate each k
    for k in range(int(args.k_min), int(args.k_max) + 1):
        logger.info("-" * 70)
        logger.info("EVALUATING k=%d", k)

        model = fit_minibatch_kmeans(
            input_path=args.input,
            k=int(k),
            batch_size=int(args.batch_size),
            n_init=int(args.n_init),
            random_state=int(args.seed),
            normalize=eff_norm,
            normalize_method=str(args.normalize_method),
        )

        sil = None
        if sample_idx.size > 0:
            sil = compute_silhouette_on_sample(
                model=model,
                input_path=args.input,
                batch_size=int(args.batch_size),
                normalize=eff_norm,
                normalize_method=str(args.normalize_method),
                sample_idx=sample_idx,
            )
            if sil is not None:
                logger.info("k=%d silhouette(sample)=%.3f", k, sil)
            else:
                logger.info("k=%d silhouette(sample)=None (insufficient clusters or sample)", k)

        res = KEvalResult(
            k=int(k),
            inertia=float(model.inertia_),
            silhouette_score_sample=float(sil) if sil is not None else None,
            n_profiles=int(n_profiles),
            normalized=eff_norm,
            normalize_method=str(args.normalize_method) if eff_norm else "none",
            batch_size=int(args.batch_size),
            n_init=int(args.n_init),
            random_state=int(args.seed),
        )
        results.append(res)

    # Choose best k
    best_k = choose_best_k(results)
    logger.info("=" * 70)
    logger.info("SELECTED k=%d", best_k)
    logger.info("=" * 70)

    # Save k evaluation summary
    k_eval_path = args.output_dir / "k_evaluation.json"
    k_eval_payload: dict[str, Any] = {
        "k_min": int(args.k_min),
        "k_max": int(args.k_max),
        "selected_k": int(best_k),
        "selection_rule": "max silhouette (sample) if available else min inertia",
        "results": [asdict(r) for r in results],
    }
    save_metadata(k_eval_payload, k_eval_path)

    # Refit best model (deterministic given same seed + same stream order)
    best_model = fit_minibatch_kmeans(
        input_path=args.input,
        k=int(best_k),
        batch_size=int(args.batch_size),
        n_init=int(args.n_init),
        random_state=int(args.seed),
        normalize=eff_norm,
        normalize_method=str(args.normalize_method),
    )

    # Write assignments
    assignments_path = args.output_dir / "cluster_assignments.parquet"
    counts = predict_and_write_assignments_streaming(
        model=best_model,
        input_path=args.input,
        output_path=assignments_path,
        batch_size=int(args.batch_size),
        normalize=eff_norm,
        normalize_method=str(args.normalize_method),
    )

    # Centroids + plot + metadata
    centroids = best_model.cluster_centers_
    plot_centroids(centroids, args.output_dir / "cluster_centroids.png")
    save_centroids_parquet(centroids, args.output_dir / "cluster_centroids.parquet")

    best_res = next(r for r in results if r.k == best_k)
    metadata = {
        "k_selected": int(best_k),
        "n_profiles": int(n_profiles),
        "n_timepoints": int(centroids.shape[1]),
        "normalized": bool(eff_norm),
        "normalize_method": str(args.normalize_method) if eff_norm else "none",
        "batch_size": int(args.batch_size),
        "n_init": int(args.n_init),
        "seed": int(args.seed),
        "algorithm": "MiniBatchKMeans",
        "inertia": float(best_model.inertia_),
        "silhouette_score_sample": best_res.silhouette_score_sample,
        "cluster_counts": {str(i): int(c) for i, c in enumerate(counts.tolist())},
        "artifacts": {
            "assignments": str(assignments_path),
            "centroids_parquet": str(args.output_dir / "cluster_centroids.parquet"),
            "centroids_plot": str(args.output_dir / "cluster_centroids.png"),
            "k_evaluation": str(k_eval_path),
        },
    }
    save_metadata(metadata, args.output_dir / "clustering_metadata.json")

    logger.info("=" * 70)
    logger.info("CLUSTERING COMPLETE")
    logger.info("=" * 70)
    logger.info("Selected k: %d", best_k)
    if best_res.silhouette_score_sample is not None:
        logger.info("Silhouette (sample): %.3f", float(best_res.silhouette_score_sample))
    logger.info("Output: %s", args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
