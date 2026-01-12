#!/usr/bin/env python3
"""
MiniBatch K-Means clustering for large load-profile datasets (PyArrow-batched).

Reads an input Parquet file in streaming batches via PyArrow, optionally normalizes
each profile row on-the-fly, fits MiniBatchKMeans with partial_fit, then predicts and
writes cluster assignments back to Parquet incrementally (selected columns + `cluster`).

Outputs (written to --output-dir):
- cluster_assignments.parquet
- cluster_centroids.parquet
- cluster_centroids.png
- clustering_metadata.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

try:
    import psutil  # optional
except Exception:
    psutil = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CGROUP_ROOT = Path("/sys/fs/cgroup")
_BASELINE_CGROUP_EVENTS: dict[str, int] = {}


# =============================================================================
# MEMORY TELEMETRY (unprivileged container-safe)
# =============================================================================
def _read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _read_int_file(path: Path) -> int | None:
    s = _read_text_file(path)
    if s is None:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _get_rss_bytes() -> int | None:
    try:
        if psutil is not None:
            return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:  # noqa: S110
        pass
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except Exception:
        return None
    return None


def _read_cgroup_memory_bytes() -> dict[str, int | None]:
    cur = _read_int_file(CGROUP_ROOT / "memory.current")
    peak = _read_int_file(CGROUP_ROOT / "memory.peak")
    maxv = _read_text_file(CGROUP_ROOT / "memory.max")
    limit = None
    if maxv is not None and maxv != "max":
        try:
            limit = int(maxv)
        except Exception:
            limit = None
    return {"current": cur, "peak": peak, "limit": limit}


def _read_cgroup_memory_events() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        txt = (CGROUP_ROOT / "memory.events").read_text(encoding="utf-8")
        for line in txt.splitlines():
            parts = line.split()
            if len(parts) == 2:
                out[parts[0]] = int(parts[1])
    except Exception:  # noqa: S110
        pass
    return out


def _mb(x: int | None) -> float | None:
    if x is None:
        return None
    return round(x / (1024.0 * 1024.0), 3)


def log_memory(stage: str, extra: dict[str, Any] | None = None) -> None:
    global _BASELINE_CGROUP_EVENTS
    if not _BASELINE_CGROUP_EVENTS:
        _BASELINE_CGROUP_EVENTS = _read_cgroup_memory_events()

    rss_b = _get_rss_bytes()
    cg = _read_cgroup_memory_bytes()
    ev = _read_cgroup_memory_events()
    keys = set(_BASELINE_CGROUP_EVENTS) | set(ev)
    delta = {k: ev.get(k, 0) - _BASELINE_CGROUP_EVENTS.get(k, 0) for k in keys}

    payload: dict[str, Any] = {
        "ts": round(time.time(), 3),
        "event": "mem",
        "stage": stage,
        "rss_mb": _mb(rss_b),
        "cgroup_current_mb": _mb(cg.get("current")),
        "cgroup_peak_mb": _mb(cg.get("peak")),
        "cgroup_limit_mb": _mb(cg.get("limit")),
        "cgroup_oom_kill_delta": delta.get("oom_kill", 0),
        "cgroup_events_delta": delta,
    }
    if extra:
        payload.update(extra)

    logger.info("[MEMORY] %s", json.dumps(payload, separators=(",", ":"), sort_keys=True))


# =============================================================================
# IO + batch utilities
# =============================================================================
def parquet_num_rows(path: Path) -> int:
    """Return Parquet row count from file metadata (no full scan)."""
    pf = pq.ParquetFile(path)
    md = pf.metadata
    if md is None:
        total = 0
        for i in range(pf.num_row_groups):
            rg = pf.metadata.row_group(i)  # type: ignore[union-attr]
            total += int(rg.num_rows)
        return total
    return int(md.num_rows)


def iter_profile_batches(path: Path, batch_size: int, columns: list[str] | None) -> Iterable[pa.RecordBatch]:
    """Yield record batches from a Parquet file."""
    pf = pq.ParquetFile(path)
    yield from pf.iter_batches(batch_size=batch_size, columns=columns)


def _list_array_to_numpy_matrix(arr: pa.Array) -> np.ndarray:
    """
    Convert Arrow list-like array (List/LargeList/FixedSizeList) to a 2D NumPy array
    WITHOUT materializing Python lists.

    Assumes lists are uniform length within the batch (true for 48-pt profiles).
    """
    if isinstance(arr, pa.FixedSizeListArray):
        n = len(arr)
        list_size = int(arr.type.list_size)
        values = arr.values
        v = values.to_numpy(zero_copy_only=False)
        if v.size != n * list_size:
            raise ValueError(f"Unexpected FixedSizeList flatten size={v.size} for n={n}, list_size={list_size}")
        X = v.reshape((n, list_size))
        return X

    if not (pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type)):
        raise TypeError(f"Expected List/LargeList/FixedSizeList; got {arr.type}")

    n = len(arr)
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)

    # ListArray / LargeListArray: use offsets + values
    # Offsets length = n+1; differences must be constant (48) for this dataset.
    offsets = arr.offsets.to_numpy(zero_copy_only=False)
    diffs = np.diff(offsets)
    if diffs.size == 0:
        return np.empty((0, 0), dtype=np.float64)

    first = int(diffs[0])
    if not np.all(diffs == first):
        raise ValueError("Non-uniform list lengths detected in profile column within a batch; cannot reshape safely.")
    if first <= 0:
        raise ValueError(f"Invalid list length inferred from offsets: {first}")

    values = arr.values
    v = values.to_numpy(zero_copy_only=False)
    expected = int(n * first)
    if v.size < expected:
        raise ValueError(f"Flattened values too short: {v.size} < expected {expected}")
    if v.size != expected:
        # This should not happen for uniform lists; fail loud to avoid silent corruption.
        raise ValueError(f"Flattened values size mismatch: {v.size} != expected {expected}")

    X = v.reshape((n, first))
    return X


def recordbatch_profiles_to_numpy(rb: pa.RecordBatch, profile_col: str = "profile") -> np.ndarray:
    """
    Convert RecordBatch `profile` list column into a 2D float64 NumPy array,
    avoiding Python object materialization.
    """
    idx = rb.schema.get_field_index(profile_col)
    if idx < 0:
        raise ValueError(f"RecordBatch missing required column '{profile_col}'")
    col = rb.column(idx)

    # RecordBatch columns are typically Array, not ChunkedArray; handle defensively.
    if isinstance(col, pa.ChunkedArray):
        if col.num_chunks != 1:
            # Concatenate chunks cheaply in Arrow-space (still bounded by batch size).
            col = pa.chunked_array(col.chunks).combine_chunks()
        col_arr = col.chunk(0)
    else:
        col_arr = col

    X = _list_array_to_numpy_matrix(col_arr).astype(np.float64, copy=False)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D profile array; got shape={X.shape}")

    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X


def parse_output_columns(spec: str) -> list[str] | None:
    """
    Parse --output-columns.

    Returns:
      - None if spec is empty (meaning "all columns").
      - Otherwise a de-duplicated list of column names, in order.
    """
    s = (spec or "").strip()
    if not s:
        return None

    cols: list[str] = []
    seen: set[str] = set()
    for part in s.split(","):
        c = part.strip()
        if not c:
            continue
        if c not in seen:
            cols.append(c)
            seen.add(c)
    return cols


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
# Clustering
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
    """Fit MiniBatchKMeans by streaming over `profile` batches and calling partial_fit()."""
    logger.info("Fitting MiniBatchKMeans (k=%d, batch_size=%s, n_init=%d)...", k, f"{batch_size:,}", n_init)
    log_memory("fit_start", {"k": int(k), "batch_size": int(batch_size), "n_init": int(n_init)})

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

        if bi == 1 or bi == n_batches or bi % 10 == 0:
            log_memory("fit_progress", {"batch_i": int(bi), "n_batches": int(n_batches), "seen": int(seen)})

        del X, rb
        gc.collect()

    logger.info("  Training complete. Inertia: %s", f"{float(model.inertia_):,.2f}")
    log_memory("fit_done", {"seen": int(seen), "inertia": float(model.inertia_)})
    return model


def make_silhouette_sample_idx(n_profiles: int, sample_size: int, seed: int) -> np.ndarray:
    """Deterministic global row index sample (sorted)."""
    if sample_size <= 0 or n_profiles <= 0:
        return np.array([], dtype=np.int64)
    if sample_size >= n_profiles:
        return np.arange(n_profiles, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n_profiles, size=int(sample_size), replace=False)
    idx.sort()
    return idx.astype(np.int64)


def predict_write_and_optional_silhouette(  # noqa: C901
    model: MiniBatchKMeans,
    input_path: Path,
    output_path: Path,
    batch_size: int,
    normalize: bool,
    normalize_method: str,
    read_columns: list[str] | None,
    write_columns: list[str] | None,
    silhouette_sample_idx: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    """
    Predict labels in batches, write output Parquet incrementally, and optionally compute
    silhouette on a sampled set of global row indices.

    Notes:
    - `read_columns` controls what we read from input (must include `profile`).
    - `write_columns` controls what we keep in output (may exclude `profile`).
      If None, we write all read columns (including `profile`).
    """
    logger.info("Predicting labels + writing assignments streaming: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_memory("predict_start", {"batch_size": int(batch_size)})

    k = int(model.n_clusters)
    counts = np.zeros(k, dtype=np.int64)

    use_sil = silhouette_sample_idx.size > 0
    sample_X: list[np.ndarray] = []
    sample_y: list[int] = []
    sample_pos = 0
    global_row = 0

    writer: pq.ParquetWriter | None = None

    try:
        for bi, rb in enumerate(iter_profile_batches(input_path, batch_size=batch_size, columns=read_columns), start=1):
            # Compute labels (requires profile)
            X = recordbatch_profiles_to_numpy(rb, profile_col="profile")
            if normalize and normalize_method != "none":
                X = normalize_batch(X, normalize_method)

            labels = model.predict(X).astype(np.int32, copy=False)
            counts += np.bincount(labels, minlength=k)

            # Silhouette sampling (uses labels on this pass; avoids a second scan)
            if use_sil:
                n = int(X.shape[0])

                while sample_pos < silhouette_sample_idx.size and int(silhouette_sample_idx[sample_pos]) < global_row:
                    sample_pos += 1
                start = sample_pos
                while (
                    sample_pos < silhouette_sample_idx.size and int(silhouette_sample_idx[sample_pos]) < global_row + n
                ):
                    sample_pos += 1

                if sample_pos > start:
                    idx_in_batch = silhouette_sample_idx[start:sample_pos] - global_row
                    # Copy rows into a compact array later; these are small (<=5000).
                    for j in idx_in_batch:
                        jj = int(j)
                        sample_X.append(X[jj].copy())
                        sample_y.append(int(labels[jj]))

            # Build output batch: select write columns (or all columns), then append cluster
            if write_columns is None:
                out_rb = rb
            else:
                indices: list[int] = []
                for name in write_columns:
                    i = rb.schema.get_field_index(name)
                    if i < 0:
                        raise ValueError(f"Input batch missing requested output column '{name}'")
                    indices.append(i)
                out_rb = rb.select(indices)

            out_rb = out_rb.append_column("cluster", pa.array(labels, type=pa.int32()))

            if writer is None:
                writer = pq.ParquetWriter(output_path, out_rb.schema, compression="snappy")

            writer.write_batch(out_rb)
            global_row += int(labels.shape[0])

            if bi == 1 or (bi % 10 == 0):
                log_memory("predict_progress", {"batch_i": int(bi), "rows_written": int(global_row)})

            del out_rb, labels, X, rb
            gc.collect()

    finally:
        if writer is not None:
            writer.close()

    total = int(counts.sum())
    logger.info("  Cluster distribution (total=%s):", f"{total:,}")
    for c, n in enumerate(counts.tolist()):
        pct = (n / total * 100.0) if total > 0 else 0.0
        logger.info("    Cluster %d: %s profiles (%.1f%%)", c, f"{n:,}", pct)

    sil: float | None = None
    if use_sil and len(sample_y) >= 2:
        ys = np.asarray(sample_y, dtype=np.int32)
        if np.unique(ys).size >= 2:
            Xs = np.asarray(sample_X, dtype=np.float64)
            logger.info("Computing silhouette on sample_size=%s ...", f"{len(ys):,}")
            log_memory("silhouette_start", {"sample_size": len(ys)})
            sil = float(silhouette_score(Xs, ys, metric="euclidean"))
            logger.info("  Silhouette score (sample): %.3f", sil)
            log_memory("silhouette_done", {"silhouette": float(sil)})
        else:
            logger.info("Silhouette: skipped (sample fell into a single cluster)")
    elif use_sil:
        logger.info("Silhouette: skipped (insufficient sample)")

    log_memory("predict_done", {"rows_written": int(total)})
    return counts, sil


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


def save_metadata(metadata: dict[str, object], output_path: Path) -> None:
    """Write run metadata and summary metrics to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("  Saved metadata: %s", output_path)


# =============================================================================
# Main
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Memory-Efficient K-Means Clustering (MiniBatch, PyArrow-batched)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True, help="Path to sampled_profiles.parquet")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--k", type=int, required=True, help="Number of clusters")

    p.add_argument("--normalize", action="store_true", help="Normalize profiles per row")
    p.add_argument("--normalize-method", choices=["minmax", "zscore", "none"], default="minmax")

    p.add_argument("--batch-size", type=int, default=50_000, help="Batch size (default: 50k)")
    p.add_argument("--n-init", type=int, default=3, help="MiniBatchKMeans n_init (default: 3)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (model + sampling)")

    p.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=5_000,
        help="Sample size for silhouette (default: 5000; set 0 to skip)",
    )

    p.add_argument(
        "--output-columns",
        type=str,
        default="",
        help=(
            "Comma-separated columns to carry through to cluster_assignments.parquet "
            "(default: all input columns). `cluster` is always added."
        ),
    )

    return p


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.k <= 1:
        raise ValueError("--k must be >= 2")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.n_init <= 0:
        raise ValueError("--n-init must be > 0")

    logger.info("=" * 70)
    logger.info("MINIBATCH K-MEANS CLUSTERING (PYARROW-BATCHED)")
    logger.info("=" * 70)
    logger.info("Input: %s", args.input)
    logger.info("Output: %s", args.output_dir)
    logger.info("k: %d", args.k)
    logger.info("batch_size: %s", f"{args.batch_size:,}")
    logger.info("n_init: %d", args.n_init)
    logger.info("seed: %d", args.seed)

    eff_norm = bool(args.normalize and args.normalize_method != "none")
    logger.info("Normalize: %s (method=%s)", eff_norm, args.normalize_method if eff_norm else "none")

    n_profiles = parquet_num_rows(args.input)
    logger.info("Profiles (from parquet metadata): %s", f"{n_profiles:,}")
    log_memory("start", {"n_profiles": int(n_profiles)})

    requested_out_cols = parse_output_columns(args.output_columns)

    if requested_out_cols is None:
        read_columns = None
        write_columns = None
        logger.info("Assignments output columns: ALL (default)")
    else:
        read_columns: list[str] = []
        seen = set()
        for c in [*requested_out_cols, "profile"]:
            if c not in seen:
                read_columns.append(c)
                seen.add(c)
        write_columns = requested_out_cols
        logger.info("Assignments output columns: %s (+cluster)", ",".join(write_columns))

    model = fit_minibatch_kmeans(
        input_path=args.input,
        k=int(args.k),
        batch_size=int(args.batch_size),
        n_init=int(args.n_init),
        random_state=int(args.seed),
        normalize=eff_norm,
        normalize_method=str(args.normalize_method),
    )
    centroids = model.cluster_centers_

    sample_idx = make_silhouette_sample_idx(int(n_profiles), int(args.silhouette_sample_size), int(args.seed))
    if sample_idx.size > 0:
        logger.info("Silhouette sample size: %s", f"{sample_idx.size:,}")
    else:
        logger.info("Silhouette: disabled")

    assignments_path = args.output_dir / "cluster_assignments.parquet"
    counts, sil_score = predict_write_and_optional_silhouette(
        model=model,
        input_path=args.input,
        output_path=assignments_path,
        batch_size=int(args.batch_size),
        normalize=eff_norm,
        normalize_method=str(args.normalize_method),
        read_columns=read_columns,
        write_columns=write_columns,
        silhouette_sample_idx=sample_idx,
    )

    centroids_plot_path = args.output_dir / "cluster_centroids.png"
    centroids_parquet_path = args.output_dir / "cluster_centroids.parquet"
    metadata_path = args.output_dir / "clustering_metadata.json"

    plot_centroids(centroids, centroids_plot_path)
    save_centroids_parquet(centroids, centroids_parquet_path)

    metadata: dict[str, object] = {
        "k": int(args.k),
        "n_profiles": int(n_profiles),
        "n_timepoints": int(centroids.shape[1]),
        "normalized": bool(eff_norm),
        "normalize_method": str(args.normalize_method) if eff_norm else "none",
        "batch_size": int(args.batch_size),
        "n_init": int(args.n_init),
        "seed": int(args.seed),
        "algorithm": "MiniBatchKMeans",
        "inertia": float(model.inertia_),
        "silhouette_score_sample": float(sil_score) if sil_score is not None else None,
        "cluster_counts": {str(i): int(c) for i, c in enumerate(counts.tolist())},
        "assignments_output_columns": write_columns if write_columns is not None else "ALL",
        "artifacts": {
            "assignments": str(assignments_path),
            "centroids_parquet": str(centroids_parquet_path),
            "centroids_plot": str(centroids_plot_path),
        },
    }
    save_metadata(metadata, metadata_path)

    logger.info("=" * 70)
    logger.info("CLUSTERING COMPLETE")
    logger.info("=" * 70)
    logger.info("Profiles: %s", f"{n_profiles:,}")
    logger.info("Clusters: %d", int(args.k))
    if sil_score is not None:
        logger.info("Silhouette (sample): %.3f", float(sil_score))
    logger.info("Output: %s", args.output_dir)
    log_memory("end")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
