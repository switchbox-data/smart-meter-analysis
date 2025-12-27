#!/usr/bin/env python3
"""ComEd Smart Meter Analysis Pipeline Orchestrator (Phase 3)

Directory structure (required):
data/runs/{run_name}/
├── raw/                         # Downloaded CSVs
├── processed/                   # Canonical interval parquet
├── clustering/                  # Stage 1 outputs + clustering outputs
├── stage2/                      # Stage 2 outputs
├── logs/                        # All logs
├── run_manifest.json            # Pipeline metadata
├── download_manifest.jsonl      # S3 download record
└── processing_manifest.jsonl    # CSV processing record
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from smart_meter_analysis.aws_loader import download_s3_batch, list_s3_files

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = Path("data/runs")
DEFAULT_CROSSWALK_PATH = Path("data/reference/2023_comed_zip4_census_crosswalk.txt")
DEFAULT_STATE_FIPS = "17"
DEFAULT_ACS_YEAR = 2023


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_path), encoding="utf-8"),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def _get_git_sha(repo_root: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return None
        sha = (r.stdout or "").strip()
        return sha or None
    except Exception:
        return None


def create_directories(run_dir: Path) -> dict[str, Path]:
    paths = {
        "run_dir": run_dir,
        "raw": run_dir / "raw",
        "processed": run_dir / "processed",
        "clustering": run_dir / "clustering",
        "stage2": run_dir / "stage2",
        "logs": run_dir / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _run_subprocess(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    logger.info("Command: %s", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed with return code {r.returncode}: {' '.join(cmd)}")


def download_from_s3(*, year_month: str, num_files: int, run_dir: Path) -> dict[str, Any]:
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "download_manifest.jsonl"

    # list_s3_files MUST return keys (not URIs).
    s3_keys = list_s3_files(year_month=year_month, max_files=num_files)

    logger.info("Downloading %d files for %s into %s", len(s3_keys), year_month, raw_dir)
    result = download_s3_batch(
        s3_keys=s3_keys,
        output_dir=raw_dir,
        manifest_path=manifest_path,
        max_files=num_files,
        fail_fast=True,
        max_errors=10,
        retries=3,
        backoff_factor=2.0,
    )
    logger.info(
        "Download complete: downloaded=%s failed=%s manifest=%s",
        result.get("downloaded"),
        result.get("failed"),
        result.get("manifest_path"),
    )
    return result


def process_csvs(*, run_dir: Path, year_month: str, day_mode: str = "calendar") -> Path:
    raw_dir = run_dir / "raw"
    processed_dir = run_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_file = processed_dir / f"comed_{year_month}.parquet"
    manifest_file = run_dir / "processing_manifest.jsonl"

    cmd = [
        sys.executable,
        "scripts/process_csvs_batched_optimized.py",
        "--input-dir",
        str(raw_dir),
        "--output",
        str(output_file),
        "--processing-manifest",
        str(manifest_file),
        "--day-mode",
        str(day_mode),
    ]

    logger.info("Ingesting CSVs into canonical interval parquet: %s", output_file)
    _run_subprocess(cmd)

    if not output_file.exists():
        raise FileNotFoundError(f"Expected processed parquet missing: {output_file}")

    return output_file


def prepare_clustering(
    *,
    run_dir: Path,
    processed_parquet: Path,
    sample_days: int,
    sample_households: int | None,
    seed: int,
) -> Path:
    clustering_dir = run_dir / "clustering"
    clustering_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "analysis/clustering/prepare_clustering_data_households.py",
        "--input",
        str(processed_parquet),
        "--output-dir",
        str(clustering_dir),
        "--sample-days",
        str(int(sample_days)),
        "--seed",
        str(int(seed)),
        "--streaming",
    ]

    if sample_households is not None:
        cmd.extend(["--sample-households", str(int(sample_households))])

    logger.info("Preparing Stage 1 clustering data in %s", clustering_dir)
    _run_subprocess(cmd)

    profiles = clustering_dir / "sampled_profiles.parquet"
    if not profiles.exists():
        raise FileNotFoundError(f"Expected clustering profiles missing: {profiles}")
    return profiles


def run_clustering(
    *,
    run_dir: Path,
    profiles_path: Path,
    k: int,
    clustering_seed: int,
) -> None:
    clustering_dir = run_dir / "clustering"
    clustering_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "analysis/clustering/euclidean_clustering_minibatch.py",
        "--input",
        str(profiles_path),
        "--output-dir",
        str(clustering_dir),
        "--k",
        str(int(k)),
        "--random-state",
        str(int(clustering_seed)),
        "--normalize",
        "--normalize-method",
        "minmax",
    ]

    logger.info("Running clustering (MiniBatchKMeans) k=%d output=%s", k, clustering_dir)
    _run_subprocess(cmd)

    expected = clustering_dir / "cluster_assignments.parquet"
    if not expected.exists():
        raise FileNotFoundError(f"Expected clustering output missing: {expected}")


def _default_stage2_census_cache(*, stage2_dir: Path, state_fips: str, acs_year: int) -> Path:
    # Option 1: cache lives under the run directory
    return stage2_dir / f"census_cache_{state_fips}_{acs_year}.parquet"


def run_stage2_logratio(
    *,
    run_dir: Path,
    crosswalk_path: Path,
    state_fips: str,
    acs_year: int,
    min_obs_per_bg: int,
    alpha: float,
    standardize: bool,
    fetch_census: bool,
    no_ols: bool,
    baseline_cluster: str | None,
    predictors_from: Path | None,
    census_cache_path: Path | None,
) -> None:
    stage2_dir = run_dir / "stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)

    clusters_path = run_dir / "clustering" / "cluster_assignments.parquet"
    if not clusters_path.exists():
        raise FileNotFoundError(f"Missing cluster assignments for Stage 2: {clusters_path}")

    if not crosswalk_path.exists():
        raise FileNotFoundError(f"Missing crosswalk for Stage 2: {crosswalk_path}")

    script_path = Path("analysis/clustering/stage2_logratio_regression.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Missing Stage 2 script: {script_path}")

    resolved_cache = census_cache_path or _default_stage2_census_cache(
        stage2_dir=stage2_dir,
        state_fips=state_fips,
        acs_year=acs_year,
    )

    cmd = [
        sys.executable,
        str(script_path),
        "--clusters",
        str(clusters_path),
        "--crosswalk",
        str(crosswalk_path),
        "--output-dir",
        str(stage2_dir),
        "--census-cache",
        str(resolved_cache),
        "--state-fips",
        str(state_fips),
        "--acs-year",
        str(int(acs_year)),
        "--min-obs-per-bg",
        str(int(min_obs_per_bg)),
        "--alpha",
        str(float(alpha)),
    ]

    if fetch_census:
        cmd.append("--fetch-census")

    if standardize:
        cmd.append("--standardize")

    if no_ols:
        cmd.append("--no-ols")

    if baseline_cluster is not None:
        cmd.extend(["--baseline-cluster", str(baseline_cluster)])

    if predictors_from is not None:
        cmd.extend(["--predictors-from", str(predictors_from)])

    logger.info("Running Stage 2 log-ratio regression output=%s", stage2_dir)
    _run_subprocess(cmd)


def write_run_manifest(*, run_dir: Path, args: argparse.Namespace) -> None:
    repo_root = Path().resolve()

    manifest: dict[str, Any] = {
        "run_name": args.run_name,
        "timestamp_utc": _utc_now_iso(),
        "args": vars(args),
        "git_sha": _get_git_sha(repo_root),
        "python_version": sys.version,
        "polars_version": pl.__version__,
        "seeds": {
            "sampling": args.seed,
            "clustering": args.clustering_seed,
        },
        "directory_structure": "data/runs/{run_name}/",
        "pipeline_version": "v1.0-restored-baseline",
        "cwd": str(Path.cwd()),
        "platform": {
            "os_name": os.name,
            "sys_platform": sys.platform,
        },
    }

    out = run_dir / "run_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote run manifest: %s", out)


def run_pipeline(args: argparse.Namespace) -> None:
    run_dir = DEFAULT_RUNS_DIR / args.run_name
    paths = create_directories(run_dir)

    _configure_logging(paths["logs"] / "pipeline.log")

    logger.info("Run directory: %s", run_dir)
    logger.info("Year-month: %s", args.year_month)

    if args.from_s3:
        download_from_s3(year_month=args.year_month, num_files=int(args.num_files), run_dir=run_dir)
    else:
        logger.info("Skipping S3 download (--from-s3 not set). Expecting CSVs in: %s", paths["raw"])

    processed_parquet = process_csvs(run_dir=run_dir, year_month=args.year_month, day_mode=args.day_mode)

    profiles_path = prepare_clustering(
        run_dir=run_dir,
        processed_parquet=processed_parquet,
        sample_days=int(args.sample_days),
        sample_households=(int(args.sample_households) if args.sample_households is not None else None),
        seed=int(args.seed),
    )

    run_clustering(
        run_dir=run_dir,
        profiles_path=profiles_path,
        k=int(args.k),
        clustering_seed=int(args.clustering_seed),
    )

    if bool(args.run_stage2):
        run_stage2_logratio(
            run_dir=run_dir,
            crosswalk_path=Path(args.stage2_crosswalk),
            state_fips=str(args.stage2_state_fips),
            acs_year=int(args.stage2_acs_year),
            min_obs_per_bg=int(args.stage2_min_obs_per_bg),
            alpha=float(args.stage2_alpha),
            standardize=bool(args.stage2_standardize),
            no_ols=bool(args.stage2_no_ols),
            fetch_census=bool(args.stage2_fetch_census),
            baseline_cluster=(str(args.stage2_baseline_cluster) if args.stage2_baseline_cluster is not None else None),
            predictors_from=(Path(args.stage2_predictors_from) if args.stage2_predictors_from is not None else None),
            census_cache_path=(Path(args.stage2_census_cache) if args.stage2_census_cache is not None else None),
        )

    write_run_manifest(run_dir=run_dir, args=args)

    logger.info("Pipeline completed successfully.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ComEd Smart Meter Analysis Pipeline Orchestrator (Phase 3)")

    p.add_argument("--run-name", required=True, help="Run name (directory under data/runs/)")
    p.add_argument("--year-month", required=True, help="Target month in YYYYMM format (e.g., 202307)")
    p.add_argument("--from-s3", action="store_true", help="Download CSVs from S3 into run_dir/raw/")

    p.add_argument("--num-files", type=int, default=10, help="Number of S3 files to download (default: 10)")

    p.add_argument("--sample-days", type=int, default=31, help="Days to sample for clustering (default: 31)")
    p.add_argument(
        "--sample-households",
        type=int,
        default=None,
        help="Households to sample (default: all). Provide an integer to limit.",
    )

    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    p.add_argument("--clustering-seed", type=int, default=42, help="Random seed for clustering (default: 42)")

    p.add_argument("--k", type=int, required=True, help="Number of clusters (single k per run)")

    p.add_argument("--day-mode", choices=["calendar", "billing"], default="calendar", help="Day attribution mode")

    p.add_argument("--run-stage2", action="store_true", help="Run Stage 2 log-ratio regression (optional)")

    # ----------------------------
    # Stage 2 options (user-facing)
    # ----------------------------
    p.add_argument(
        "--stage2-crosswalk",
        default=str(DEFAULT_CROSSWALK_PATH),
        help=f"ZIP+4 → block-group crosswalk path (default: {DEFAULT_CROSSWALK_PATH})",
    )
    p.add_argument(
        "--stage2-state-fips",
        default=DEFAULT_STATE_FIPS,
        help=f"State FIPS (default: {DEFAULT_STATE_FIPS})",
    )
    p.add_argument(
        "--stage2-acs-year",
        type=int,
        default=DEFAULT_ACS_YEAR,
        help=f"ACS year (default: {DEFAULT_ACS_YEAR})",
    )
    p.add_argument(
        "--stage2-fetch-census",
        action="store_true",
        help="Stage 2: force re-fetch Census data (ignore cache)",
    )
    p.add_argument(
        "--stage2-census-cache",
        default=None,
        help=(
            "Optional census cache parquet path. If omitted, defaults to "
            "data/runs/{run_name}/stage2/census_cache_{state_fips}_{acs_year}.parquet (Option 1)."
        ),
    )

    p.add_argument(
        "--stage2-min-obs-per-bg",
        type=int,
        default=50,
        help="Stage 2 minimum household-day observations per block group (default: 50)",
    )
    p.add_argument("--stage2-alpha", type=float, default=0.5, help="Stage 2 Laplace smoothing alpha (default: 0.5)")
    p.add_argument("--stage2-standardize", action="store_true", help="Stage 2: standardize predictors")
    p.add_argument("--stage2-no-ols", action="store_true", help="Stage 2: skip OLS robustness check")
    p.add_argument(
        "--stage2-baseline-cluster",
        default=None,
        help="Stage 2: optional baseline cluster label (default: most frequent cluster)",
    )
    p.add_argument(
        "--stage2-predictors-from",
        default=None,
        help="Stage 2: optional path to predictors list (one per line) to force exact predictors",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    if int(args.num_files) <= 0 and args.from_s3:
        raise ValueError("--num-files must be > 0 when using --from-s3")

    if int(args.k) <= 1:
        raise ValueError("--k must be >= 2")

    run_pipeline(args)


if __name__ == "__main__":
    main()
