#!/usr/bin/env python3
"""
Unified benchmark suite for comparing eager vs lazy Polars pipelines
and reporting S3 storage footprints for ComEd Zip4 data.

This module consolidates all previous scripts:
- eager_from_manifest.py
- lazy_from_manifest.py
- s3_size_report.py
- summary_memory builder
- correctness comparators

Usage Examples:

    python -m smart_meter_analysis.eager_vs_lazy_benchmarks eager 1000
    python -m smart_meter_analysis.eager_vs_lazy_benchmarks lazy 100
    python -m smart_meter_analysis.eager_vs_lazy_benchmarks summary
    python -m smart_meter_analysis.eager_vs_lazy_benchmarks s3-size 202101 202509
    python -m smart_meter_analysis.eager_vs_lazy_benchmarks full

All results are stored under:
    results/benchmarks/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import boto3
import polars as pl

# =============================================================================
# Paths
# =============================================================================

BASE = Path("results/benchmarks")
MANIFEST_DIR = BASE / "manifests"
PARQUET_DIR = BASE / "parquet"
PROFILES_DIR = BASE / "profiles"
SUMMARY_DIR = BASE / "summary"

for d in [BASE, MANIFEST_DIR, PARQUET_DIR, PROFILES_DIR, SUMMARY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Manifest helpers
# =============================================================================


def read_manifest(path: Path, bucket: str) -> list[str]:
    uris = []
    with open(path) as f:
        for line in f:
            key = line.strip()
            if not key or key.startswith("#"):
                continue
            if key.startswith("s3://"):
                uris.append(key)
            else:
                uris.append(f"s3://{bucket}/{key}")
    return uris


# =============================================================================
# Eager benchmark
# =============================================================================


def run_eager(manifest_name: str, bucket: str, day_mode: str = "billing") -> Path:
    """
    Run eager benchmark on a manifest (e.g. "aug202308_00100.txt").
    Returns path to generated parquet.
    """
    manifest = MANIFEST_DIR / manifest_name
    uris = read_manifest(manifest, bucket)

    out_path = PARQUET_DIR / f"eager_{manifest.stem}.parquet"

    schema_overrides = {
        "ZIP_CODE": pl.Utf8,
        "DELIVERY_SERVICE_CLASS": pl.Utf8,
        "DELIVERY_SERVICE_NAME": pl.Utf8,
        "ACCOUNT_IDENTIFIER": pl.Utf8,
    }

    frames = []
    for s3_uri in uris:
        try:
            df = pl.read_csv(s3_uri, schema_overrides=schema_overrides, ignore_errors=True)
            frames.append(df)
        except Exception as e:
            print(f"Skipping {s3_uri}: {e}")

    if not frames:
        raise RuntimeError("No valid CSVs loaded.")

    df_combined = pl.concat(frames, how="diagonal_relaxed")

    from smart_meter_analysis.transformation import transform_frame_eager

    df_final = transform_frame_eager(df_combined, day_mode=day_mode)

    df_final.write_parquet(out_path)
    print(f"[EAGER] wrote: {out_path}")
    return out_path


# =============================================================================
# Lazy benchmark
# =============================================================================


def run_lazy(manifest_name: str, bucket: str, day_mode: str = "billing") -> Path:
    """
    Run lazy benchmark on a manifest.
    Returns path to generated parquet.
    """
    manifest = MANIFEST_DIR / manifest_name
    uris = read_manifest(manifest, bucket)

    out_path = PARQUET_DIR / f"lazy_{manifest.stem}.parquet"

    # Build lazy frames and union
    lfs = []
    for s3_uri in uris:
        lf = pl.scan_csv(s3_uri)
        lfs.append(lf)

    if not lfs:
        raise RuntimeError("No valid CSVs loaded.")

    from smart_meter_analysis.transformation import transform_frame_lazy

    lf_all = pl.concat(lfs, how="diagonal")

    lf_final = transform_frame_lazy(lf_all, day_mode=day_mode)
    lf_final.sink_parquet(out_path)

    print(f"[LAZY] wrote: {out_path}")
    return out_path


# =============================================================================
# S3 Size Report
# =============================================================================


def s3_size_report(bucket: str, prefix_base: str, start: int, end: int, out_csv: Path) -> None:
    """
    Compute file counts + total size for each YYYYMM between start and end.
    """
    s3 = boto3.client("s3")

    rows = []
    for ym in range(start, end + 1):
        prefix = f"{prefix_base}/{ym}/"

        paginator = s3.get_paginator("list_objects_v2")
        total_bytes = 0
        count = 0

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                count += 1
                total_bytes += obj["Size"]

        gib = total_bytes / (1024**3)
        rows.append([f"{ym}", count, total_bytes, gib])
        print(f"{ym}: {count:,} files, {gib:.3f} GiB")

    total_count = sum(r[1] for r in rows)
    total_bytes = sum(r[2] for r in rows)
    total_gib = total_bytes / (1024**3)
    rows.append(["TOTAL", total_count, total_bytes, total_gib])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["month", "count", "total_bytes", "gib"])
        w.writerows(rows)

    print(f"Wrote S3 size report: {out_csv}")


# =============================================================================
# Summary Memory Stats
# =============================================================================


def load_mprof(path: Path) -> tuple[float, float, float]:
    """
    Parse mprof .dat file and return:
        duration_seconds, peak_mib, mean_mib
    """
    times = []
    mem = []

    with open(path) as f:
        for line in f:
            if not line.startswith("MEM"):
                continue
            _, mb, t = line.split()
            mem.append(float(mb))
            times.append(float(t))
    if not mem:
        return (0.0, 0.0, 0.0)

    duration = max(times)
    peak = max(mem)
    mean = sum(mem) / len(mem)
    return (duration, peak, mean)


def build_summary_csv(configs: list[tuple[str, int, str]], out_csv: Path):
    """
    configs: list of (run_type, size, filename)
    """
    rows = []
    for run_type, size, fname in configs:
        path = PROFILES_DIR / fname
        if not path.exists():
            print(f"Missing profile: {path}")
            continue
        duration, peak, mean = load_mprof(path)
        rows.append([run_type, size, duration, peak, mean, fname])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "size", "duration_s", "peak_mib", "mean_mib", "file"])
        w.writerows(rows)

    print(f"Wrote summary: {out_csv}")


# =============================================================================
# Correctness Check
# =============================================================================


def correctness_check(lazy_path: Path, eager_path: Path):
    """
    Compare row counts, 48-interval days, and kWh sanity.
    """
    df_lazy = pl.read_parquet(lazy_path)
    df_eager = pl.read_parquet(eager_path)

    print("lazy rows:", df_lazy.height, "eager rows:", df_eager.height)

    def per_day(df):
        return df.group_by(["account_identifier", "date"]).len().filter(pl.col("len") == 48).height

    print("48-row days — lazy:", per_day(df_lazy), "eager:", per_day(df_eager))
    print("lazy null kwh:", df_lazy.filter(pl.col("kwh").is_null()).height)
    print("eager null kwh:", df_eager.filter(pl.col("kwh").is_null()).height)
    print("lazy negative kwh:", df_lazy.filter(pl.col("kwh") < 0).height)
    print("eager negative kwh:", df_eager.filter(pl.col("kwh") < 0).height)


# =============================================================================
# Full benchmark pipeline
# =============================================================================


def run_full_pipeline(bucket: str):
    """
    Runs all benchmarks:
      100, 1000, 10000 eager
      100, 1000 lazy (we skip 10k lazy intentionally)
    Assumes all manifests already exist under results/benchmarks/manifests/.
    """
    sets = [
        ("eager", "aug202308_00100.txt"),
        ("eager", "aug202308_01000.txt"),
        ("eager", "aug202308_10000.txt"),
        ("lazy", "aug202308_00100.txt"),
        ("lazy", "aug202308_01000.txt"),
    ]

    for run_type, fname in sets:
        print(f"\n*** Running {run_type.upper()} on {fname}")
        if run_type == "eager":
            run_eager(fname, bucket)
        else:
            run_lazy(fname, bucket)

    print("\nFull benchmark completed.")


# =============================================================================
# CLI
# =============================================================================


def main():
    p = argparse.ArgumentParser(description="Eager vs Lazy Benchmark Suite")
    sub = p.add_subparsers(dest="cmd", required=True)

    # eager
    sp_e = sub.add_parser("eager", help="Run eager benchmark")
    sp_e.add_argument("manifest", help="Manifest filename in results/benchmarks/manifests/")
    sp_e.add_argument("--bucket", default="smart-meter-data-sb")
    sp_e.add_argument("--day-mode", default="billing")

    # lazy
    sp_l = sub.add_parser("lazy", help="Run lazy benchmark")
    sp_l.add_argument("manifest")
    sp_l.add_argument("--bucket", default="smart-meter-data-sb")
    sp_l.add_argument("--day-mode", default="billing")

    # s3-size
    sp_s = sub.add_parser("s3-size", help="Compute S3 storage by month")
    sp_s.add_argument("start", type=int)
    sp_s.add_argument("end", type=int)
    sp_s.add_argument("--bucket", default="smart-meter-data-sb")
    sp_s.add_argument("--prefix", default="sharepoint-files/Zip4")

    # summary
    sub.add_parser("summary", help="Rebuild summary CSV from existing profiles")

    # correctness test
    sp_c = sub.add_parser("correct", help="Compare eager vs lazy parquet")
    sp_c.add_argument("lazy_file")
    sp_c.add_argument("eager_file")

    # full
    sp_f = sub.add_parser("full", help="Run full benchmarks")
    sp_f.add_argument("--bucket", default="smart-meter-data-sb")

    args = p.parse_args()

    if args.cmd == "eager":
        run_eager(args.manifest, args.bucket, day_mode=args.day_mode)
    elif args.cmd == "lazy":
        run_lazy(args.manifest, args.bucket, day_mode=args.day_mode)
    elif args.cmd == "s3-size":
        out = SUMMARY_DIR / "s3_sizes_by_month.csv"
        s3_size_report(args.bucket, args.prefix, args.start, args.end, out)
    elif args.cmd == "summary":
        configs = [
            ("eager", 100, "mprof_eager_00100.dat"),
            ("eager", 1000, "mprof_eager_01000.dat"),
            ("eager", 10000, "mprof_eager_10000.dat"),
            ("lazy", 100, "mprof_lazy_00100.dat"),
            ("lazy", 1000, "mprof_lazy_01000.dat"),
        ]
        build_summary_csv(configs, SUMMARY_DIR / "summary_memory.csv")
    elif args.cmd == "correct":
        correctness_check(Path(args.lazy_file), Path(args.eager_file))
    elif args.cmd == "full":
        run_full_pipeline(args.bucket)
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
