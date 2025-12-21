#!/usr/bin/env python3
"""
Main pipeline script for monthly smart meter analysis.

This script parameterizes the pipeline to process any month by changing
a single --month parameter (1-12).

Usage:
    python scripts/run_pipeline.py --month 7 --input path/to/input.parquet
    python scripts/run_pipeline.py --month 1 --year 2023 --input path/to/input.parquet
    python scripts/run_pipeline.py --month 7 --config config/custom.yaml --input path/to/input.parquet

The script:
1. Loads configuration from config/monthly_run.yaml (or custom config)
2. Overrides month/year if provided via CLI
3. Runs the clustering pipeline with month-specific filtering
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.clustering.euclidean_clustering_minibatch import main as clustering_main
from analysis.clustering.prepare_clustering_data_households import prepare_clustering_data
from smart_meter_analysis.config import get_year_month_str, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineArgs:
    """Parsed CLI arguments for the monthly pipeline."""

    month: int
    year: int | None
    config: Path | None
    input: Path
    output_dir: Path | None
    skip_clustering: bool


def _parse_args(argv: list[str] | None = None) -> PipelineArgs:
    parser = argparse.ArgumentParser(
        description="Run smart meter analysis pipeline for a specific month",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--month",
        type=int,
        required=True,
        choices=range(1, 13),
        metavar="MONTH",
        help="Month to process (1-12, e.g., 7 for July)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year to process (default: from config file, typically 2023)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (default: config/monthly_run.yaml)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input parquet file path (processed interval data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for clustering results (default: data/clustering)",
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Only prepare clustering data, skip actual clustering",
    )

    ns = parser.parse_args(argv)
    return PipelineArgs(
        month=ns.month,
        year=ns.year,
        config=ns.config,
        input=ns.input,
        output_dir=ns.output_dir,
        skip_clustering=ns.skip_clustering,
    )


def _load_and_override_config(args: PipelineArgs) -> dict[str, Any]:
    try:
        config: dict[str, Any] = load_config(args.config)
    except FileNotFoundError as e:
        logger.error("Config file not found: %s", e)
        raise

    # Override month/year from CLI
    config["month"] = args.month
    if args.year is not None:
        config["year"] = args.year

    return config


def _resolve_output_dir(config: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override

    default_dir = config.get("output", {}).get("clustering_dir", "data/clustering")
    return Path(default_dir)


def _log_run_header(*, year: int, month: int, year_month_str: str, input_path: Path) -> None:
    logger.info("=" * 70)
    logger.info("MONTHLY PIPELINE EXECUTION")
    logger.info("=" * 70)
    logger.info("Year: %d", year)
    logger.info("Month: %d", month)
    logger.info("Year-Month: %s", year_month_str)
    logger.info("Input: %s", input_path)


def _prepare_data(*, config: dict[str, Any], input_path: Path, output_dir: Path, year: int, month: int) -> None:
    sampling_config = config.get("sampling", {})
    sample_households = sampling_config.get("sample_households")
    sample_days = sampling_config.get("sample_days", 20)
    day_strategy = sampling_config.get("day_strategy", "stratified")
    streaming = sampling_config.get("streaming", True)
    chunk_size = sampling_config.get("chunk_size", 5000)
    seed = sampling_config.get("seed", 42)

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 1: PREPARING CLUSTERING DATA")
    logger.info("=" * 70)

    stats = prepare_clustering_data(
        input_paths=[input_path],
        output_dir=output_dir,
        sample_households=sample_households,
        sample_days=sample_days,
        day_strategy=day_strategy,
        streaming=streaming,
        chunk_size=chunk_size,
        seed=seed,
        year=year,
        month=month,
    )

    logger.info("✓ Clustering data preparation complete")
    logger.info("  Profiles: %s", f"{stats['n_profiles']:,}")
    logger.info("  Households: %s", f"{stats['n_households']:,}")


def _run_clustering(*, config: dict[str, Any], output_dir: Path) -> int:
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: RUNNING CLUSTERING")
    logger.info("=" * 70)

    clustering_config = config.get("clustering", {})
    n_clusters = clustering_config.get("n_clusters", 4)
    batch_size = clustering_config.get("batch_size", 10000)
    n_init = clustering_config.get("n_init", 3)
    random_state = clustering_config.get("random_state", 42)
    normalize = clustering_config.get("normalize", True)
    normalize_method = clustering_config.get("normalize_method", "minmax")
    silhouette_sample_size = clustering_config.get("silhouette_sample_size", 5000)

    input_profiles = output_dir / "sampled_profiles.parquet"
    clustering_output_dir = output_dir / "results"
    clustering_output_dir.mkdir(parents=True, exist_ok=True)

    clustering_args: list[str] = [
        "--input",
        str(input_profiles),
        "--output-dir",
        str(clustering_output_dir),
        "--k",
        str(n_clusters),
        "--batch-size",
        str(batch_size),
        "--n-init",
        str(n_init),
        "--random-state",
        str(random_state),
        "--silhouette-sample-size",
        str(silhouette_sample_size),
    ]

    if normalize:
        clustering_args.extend(["--normalize", "--normalize-method", normalize_method])
    else:
        clustering_args.extend(["--normalize-method", "none"])

    old_argv = sys.argv
    try:
        sys.argv = ["euclidean_clustering_minibatch.py", *clustering_args]
        result = clustering_main()
        if result != 0:
            logger.error("Clustering failed")
            return int(result)
        logger.info("✓ Clustering complete")
        return 0
    finally:
        sys.argv = old_argv


def main(argv: list[str] | None = None) -> int:
    """Main entry point for monthly pipeline."""
    args = _parse_args(argv)

    try:
        config = _load_and_override_config(args)
    except FileNotFoundError:
        return 1

    year = int(config.get("year", 2023))
    month = int(config.get("month", args.month))
    year_month_str = get_year_month_str(config)

    _log_run_header(year=year, month=month, year_month_str=year_month_str, input_path=args.input)

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    output_dir = _resolve_output_dir(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    try:
        _prepare_data(config=config, input_path=args.input, output_dir=output_dir, year=year, month=month)
    except Exception as e:
        logger.error("Failed to prepare clustering data: %s", e, exc_info=True)
        return 1

    if args.skip_clustering:
        logger.info("Skipping clustering (--skip-clustering specified)")
    else:
        try:
            result = _run_clustering(config=config, output_dir=output_dir)
        except Exception as e:
            logger.error("Failed to run clustering: %s", e, exc_info=True)
            return 1
        if result != 0:
            return result

    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Output: %s", output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
