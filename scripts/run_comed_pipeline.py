#!/usr/bin/env python3
"""
ComEd Smart Meter Analysis Pipeline

Main entry point for the ComEd smart meter clustering analysis. This script
handles the complete workflow from raw S3 data to clustered load profiles.

================================================================================
PIPELINE OVERVIEW
================================================================================

Stage 1: Usage Pattern Clustering (this script)
    1. Download    - Fetch CSV files from S3
    2. Process     - Transform wide→long, add time features, write parquet
    3. Aggregate   - Create daily load profiles per ZIP+4
    4. Cluster     - DTW k-means to identify usage patterns
    5. Validate    - Check data quality at each step

Stage 2: Demographic Analysis (separate script: stage2_regression.py)
    - Join clusters with Census demographics
    - Run multinomial logistic regression
    - Identify demographic predictors of usage patterns

================================================================================
USAGE MODES
================================================================================

FULL PIPELINE (download from S3 and run everything):
    python run_comed_pipeline.py --from-s3 --num-files 1000

VALIDATE ONLY (check existing files):
    python run_comed_pipeline.py --validate-only

PROCESS ONLY (no clustering, useful for testing):
    python run_comed_pipeline.py --from-s3 --num-files 100 --skip-clustering

SPECIFIC STAGE VALIDATION:
    python run_comed_pipeline.py --validate-only --stage processed
    python run_comed_pipeline.py --validate-only --stage clustering

================================================================================
OUTPUT STRUCTURE
================================================================================

data/validation_runs/{run_name}/
├── samples/                    # Raw CSV files from S3
├── processed/
│   └── comed_{year_month}.parquet   # Interval-level data (long format)
├── clustering/
│   ├── sampled_profiles.parquet     # Daily load profiles for clustering
│   ├── zip4_demographics.parquet    # ZIP+4 list (demographics added in Stage 2)
│   └── results/
│       ├── cluster_assignments.parquet
│       ├── cluster_centroids.parquet
│       ├── clustering_metrics.json
│       └── centroid_plot.png

================================================================================
EXAMPLES
================================================================================

# Quick test with 100 files
python run_comed_pipeline.py --from-s3 --num-files 100 --sample-zips 100 --sample-days 10

# Standard analysis (1000 files, ~30min clustering)
python run_comed_pipeline.py --from-s3 --num-files 1000

# Large scale (will take hours)
python run_comed_pipeline.py --from-s3 --num-files 10000 --sample-zips 1000

# Just validate existing results
python run_comed_pipeline.py --validate-only --run-name 202308_1000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_PATHS = {
    "processed": Path("data/processed/comed_202308.parquet"),
    "enriched": Path("data/enriched_test/enriched.parquet"),
    "clustering_dir": Path("data/clustering"),
    "crosswalk": Path("data/reference/2023_comed_zip4_census_crosswalk.txt"),
}

DEFAULT_S3_CONFIG = {
    "bucket": "smart-meter-data-sb",
    "prefix": "sharepoint-files/Zip4/",
}

DEFAULT_CLUSTERING_CONFIG = {
    "sample_days": 20,
    "sample_zips": 500,
    "k_min": 3,
    "k_max": 5,  # Reduced from 6 - faster evaluation
    "day_strategy": "stratified",
    # DTW performance parameters
    "max_eval_samples": 2000,  # Subsample for k evaluation
    "eval_max_iter": 10,
    "eval_n_init": 3,
    "final_max_iter": 10,
    "final_n_init": 3,
}


# =============================================================================
# PIPELINE EXECUTOR CLASS
# =============================================================================


class ComedPipeline:
    """
    Orchestrates the ComEd smart meter analysis pipeline.

    This class manages the complete workflow from raw S3 data to clustered
    load profiles, with validation at each step.

    Attributes:
        base_dir: Project root directory
        run_name: Identifier for this pipeline run (e.g., "202308_1000")
        run_dir: Output directory for this run
        paths: Dictionary of file paths for this run
    """

    def __init__(self, base_dir: Path, run_name: str | None = None):
        """
        Initialize pipeline with project directory and optional run name.

        Args:
            base_dir: Root directory of the smart-meter-analysis project
            run_name: Identifier for this run. If provided, outputs go to
                     data/validation_runs/{run_name}/. If None, uses default paths.
        """
        self.base_dir = base_dir
        self.run_name = run_name
        self.results: dict[str, dict] = {}

        if run_name:
            self.run_dir = base_dir / "data" / "validation_runs" / run_name
            self.year_month = run_name.split("_")[0] if "_" in run_name else run_name
            self.paths = {
                "samples": self.run_dir / "samples",
                "processed": self.run_dir / "processed" / f"comed_{self.year_month}.parquet",
                "clustering_dir": self.run_dir / "clustering",
            }
        else:
            self.run_dir = None
            self.paths = DEFAULT_PATHS.copy()

    # =========================================================================
    # PIPELINE STEPS
    # =========================================================================

    def setup_directories(self) -> None:
        """Create directory structure for pipeline outputs."""
        if not self.run_dir:
            return

        for subdir in ["samples", "processed", "clustering/results"]:
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Created output directory: {self.run_dir}")

    def download_from_s3(
        self,
        year_month: str,
        num_files: int,
        bucket: str = DEFAULT_S3_CONFIG["bucket"],
        prefix: str = DEFAULT_S3_CONFIG["prefix"],
    ) -> bool:
        """
        Download CSV files from S3 for processing.

        Args:
            year_month: Target month in YYYYMM format (e.g., '202308')
            num_files: Number of CSV files to download
            bucket: S3 bucket name
            prefix: S3 key prefix for ComEd data

        Returns:
            True if download successful, False otherwise
        """
        try:
            import boto3
        except ImportError:
            logger.error("boto3 not installed. Run: pip install boto3")
            return False

        logger.info(f"Connecting to S3: s3://{bucket}/{prefix}{year_month}/")

        try:
            s3 = boto3.client("s3")
            full_prefix = f"{prefix}{year_month}/"

            # List files
            paginator = s3.get_paginator("list_objects_v2")
            csv_keys = []

            for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
                if "Contents" not in page:
                    continue
                for obj in page["Contents"]:
                    if obj["Key"].endswith(".csv"):
                        csv_keys.append(obj["Key"])
                        if len(csv_keys) >= num_files:
                            break
                if len(csv_keys) >= num_files:
                    break

            if not csv_keys:
                logger.error(f"No CSV files found in s3://{bucket}/{full_prefix}")
                return False

            logger.info(f"Downloading {len(csv_keys)} files to {self.paths['samples']}")

            for i, key in enumerate(csv_keys, 1):
                filename = Path(key).name
                local_path = self.paths["samples"] / filename
                s3.download_file(bucket, key, str(local_path))

                if i % 100 == 0 or i == len(csv_keys):
                    logger.info(f"  Downloaded {i}/{len(csv_keys)} files")

            logger.info(f"Download complete: {len(csv_keys)} files")
            return True

        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return False

    def process_raw_data(self, year_month: str) -> bool:
        """
        Process raw CSV files into analysis-ready parquet format.

        Transforms wide-format CSVs to long format with time features.
        Uses lazy evaluation for memory efficiency.

        Args:
            year_month: Month identifier for output file naming

        Returns:
            True if processing successful, False otherwise
        """
        csv_files = sorted(self.paths["samples"].glob("*.csv"))
        if not csv_files:
            logger.error(f"No CSV files found in {self.paths['samples']}")
            return False

        logger.info(f"Processing {len(csv_files)} CSV files")

        from smart_meter_analysis.aws_loader import (
            COMED_SCHEMA,
            add_time_columns_lazy,
            transform_wide_to_long_lazy,
        )

        lazy_frames = []
        for i, csv_path in enumerate(csv_files, 1):
            if i % 200 == 0 or i == len(csv_files):
                logger.info(f"  Scanned {i}/{len(csv_files)} files")

            try:
                lf = pl.scan_csv(str(csv_path), schema_overrides=COMED_SCHEMA, ignore_errors=True)
                lf = transform_wide_to_long_lazy(lf)
                lf = add_time_columns_lazy(lf, day_mode="calendar")
                lazy_frames.append(lf)
            except Exception as e:
                logger.warning(f"Failed to scan {csv_path.name}: {e}")

        if not lazy_frames:
            logger.error("No files successfully scanned")
            return False

        logger.info("Writing combined parquet file...")
        self.paths["processed"].parent.mkdir(parents=True, exist_ok=True)

        lf_combined = pl.concat(lazy_frames, how="diagonal_relaxed")
        lf_combined.sink_parquet(self.paths["processed"])

        row_count = pl.scan_parquet(self.paths["processed"]).select(pl.len()).collect()[0, 0]
        logger.info(f"Wrote {row_count:,} records to {self.paths['processed']}")

        return True

    def prepare_clustering_data(
        self,
        sample_days: int = DEFAULT_CLUSTERING_CONFIG["sample_days"],
        sample_zips: int = DEFAULT_CLUSTERING_CONFIG["sample_zips"],
        day_strategy: str = DEFAULT_CLUSTERING_CONFIG["day_strategy"],
    ) -> bool:
        """
        Prepare daily load profiles for clustering.

        Aggregates interval data to ZIP+4 level and creates daily profiles.
        Uses chunked processing to manage memory.

        Args:
            sample_days: Number of days to sample
            sample_zips: Number of ZIP+4 codes to sample
            day_strategy: 'stratified' (70/30 weekday/weekend) or 'random'

        Returns:
            True if preparation successful, False otherwise
        """
        import subprocess

        input_path = self.paths["processed"]
        output_dir = self.paths["clustering_dir"]

        if not input_path.exists():
            logger.error(f"Processed data not found: {input_path}")
            return False

        cmd = [
            sys.executable,
            str(self.base_dir / "analysis" / "clustering" / "prepare_clustering_data.py"),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--day-strategy",
            day_strategy,
            "--sample-days",
            str(sample_days),
            "--sample-zips",
            str(sample_zips),
        ]

        logger.info(f"Preparing clustering data (sampling {sample_zips} ZIP+4s x {sample_days} days)")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Clustering prep failed: {result.stderr}")
            return False

        logger.info("Clustering data prepared")
        return True

    def run_clustering(
        self,
        k_min: int = DEFAULT_CLUSTERING_CONFIG["k_min"],
        k_max: int = DEFAULT_CLUSTERING_CONFIG["k_max"],
        max_eval_samples: int = DEFAULT_CLUSTERING_CONFIG["max_eval_samples"],
        eval_max_iter: int = DEFAULT_CLUSTERING_CONFIG["eval_max_iter"],
        eval_n_init: int = DEFAULT_CLUSTERING_CONFIG["eval_n_init"],
        final_max_iter: int = DEFAULT_CLUSTERING_CONFIG["final_max_iter"],
        final_n_init: int = DEFAULT_CLUSTERING_CONFIG["final_n_init"],
    ) -> bool:
        """
        Run DTW k-means clustering on prepared profiles.

        Args:
            k_min: Minimum number of clusters to test
            k_max: Maximum number of clusters to test
            max_eval_samples: Max profiles for k evaluation (subsample)
            eval_max_iter: Max iterations for evaluation runs
            eval_n_init: Number of initializations for evaluation
            final_max_iter: Max iterations for final clustering
            final_n_init: Number of initializations for final clustering

        Returns:
            True if clustering successful, False otherwise
        """
        import subprocess

        profiles_path = self.paths["clustering_dir"] / "sampled_profiles.parquet"
        results_dir = self.paths["clustering_dir"] / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        if not profiles_path.exists():
            logger.error(f"Profiles not found: {profiles_path}")
            return False

        cmd = [
            sys.executable,
            str(self.base_dir / "analysis" / "clustering" / "dtw_clustering.py"),
            "--input",
            str(profiles_path),
            "--output-dir",
            str(results_dir),
            "--k-range",
            str(k_min),
            str(k_max),
            "--find-optimal-k",
            "--normalize",
            "--max-eval-samples",
            str(max_eval_samples),
            "--eval-max-iter",
            str(eval_max_iter),
            "--eval-n-init",
            str(eval_n_init),
            "--final-max-iter",
            str(final_max_iter),
            "--final-n-init",
            str(final_n_init),
        ]

        logger.info(f"Running DTW clustering (k={k_min}-{k_max}, eval on {max_eval_samples} samples)...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Clustering failed: {result.stderr}")
            if result.stdout:
                logger.error(f"stdout: {result.stdout}")
            return False

        logger.info("Clustering complete")
        return True

    def run_stage2_regression(
        self,
        crosswalk_path: Path | None = None,
        census_cache_path: Path | None = None,
    ) -> bool:
        """
        Run Stage 2: Multinomial regression to predict clusters from demographics.

        stage2_regression.py handles census fetching internally if cache doesn't exist.

        Args:
            crosswalk_path: Path to ZIP+4 crosswalk file
            census_cache_path: Path to cached census data

        Returns:
            True if regression successful, False otherwise
        """
        import subprocess

        clusters_path = self.paths["clustering_dir"] / "results" / "cluster_assignments.parquet"
        output_dir = self.paths["clustering_dir"] / "results" / "stage2"

        if not clusters_path.exists():
            logger.error(f"Cluster assignments not found: {clusters_path}")
            logger.error("Run Stage 1 clustering first")
            return False

        # Default paths
        if crosswalk_path is None:
            crosswalk_path = self.base_dir / "data" / "reference" / "2023_comed_zip4_census_crosswalk.txt"
        if census_cache_path is None:
            census_cache_path = self.base_dir / "data" / "reference" / "census_17_2023.parquet"

        if not crosswalk_path.exists():
            logger.error(f"Crosswalk not found: {crosswalk_path}")
            return False

        # stage2_regression.py handles census fetching internally if cache missing
        cmd = [
            sys.executable,
            str(self.base_dir / "analysis" / "clustering" / "multinomial_regression.py"),
            "--clusters",
            str(clusters_path),
            "--crosswalk",
            str(crosswalk_path),
            "--census-cache",
            str(census_cache_path),
            "--output-dir",
            str(output_dir),
        ]

        logger.info("Running Stage 2 regression...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Print output (report goes to stdout)
        if result.stdout:
            print(result.stdout)

        if result.returncode != 0:
            logger.error(f"Stage 2 regression failed: {result.stderr}")
            return False

        logger.info(f"Stage 2 complete: {output_dir}")
        return True

    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================

    def validate_processed_data(self) -> dict:
        """Validate processed interval-level data using lazy evaluation."""
        path = self.paths["processed"]

        if not path.exists():
            return self._fail("processed", f"File not found: {path}")

        logger.info(f"Validating processed data: {path}")

        errors = []
        warnings = []

        try:
            lf = pl.scan_parquet(path)
            schema = lf.collect_schema()
        except Exception as e:
            return self._fail("processed", f"Failed to read: {e}")

        # Check required columns
        required = ["zip_code", "account_identifier", "datetime", "kwh", "date", "hour"]
        missing = [c for c in required if c not in schema.names()]
        if missing:
            errors.append(f"Missing columns: {missing}")

        # Get stats using lazy evaluation (no full load)
        try:
            stats_df = lf.select([
                pl.len().alias("rows"),
                pl.col("zip_code").n_unique().alias("zip_codes"),
                pl.col("account_identifier").n_unique().alias("accounts"),
                pl.col("kwh").min().alias("kwh_min"),
                pl.col("kwh").null_count().alias("kwh_nulls"),
            ]).collect()

            stats_dict = stats_df.to_dicts()[0]

            # Check row count
            if stats_dict["rows"] == 0:
                errors.append("No data rows")

            # Check for nulls
            if stats_dict["kwh_nulls"] > 0:
                null_pct = stats_dict["kwh_nulls"] / stats_dict["rows"] * 100
                if null_pct > 5:
                    errors.append(f"kwh: {null_pct:.1f}% null")
                elif null_pct > 0:
                    warnings.append(f"kwh: {null_pct:.1f}% null")

            # Check kWh range
            if stats_dict["kwh_min"] is not None and stats_dict["kwh_min"] < 0:
                warnings.append(f"Negative kWh values: min={stats_dict['kwh_min']}")

            stats = {
                "rows": stats_dict["rows"],
                "zip_codes": stats_dict["zip_codes"],
                "accounts": stats_dict["accounts"],
                "file_size_mb": path.stat().st_size / 1024 / 1024,
            }
        except Exception as e:
            return self._fail("processed", f"Failed to compute stats: {e}")

        return self._result("processed", errors, warnings, stats)

    def validate_clustering_inputs(self) -> dict:
        """Validate clustering input files."""
        profiles_path = self.paths["clustering_dir"] / "sampled_profiles.parquet"

        if not profiles_path.exists():
            return self._fail("clustering_inputs", f"Profiles not found: {profiles_path}")

        logger.info(f"Validating clustering inputs: {profiles_path}")

        try:
            df = pl.read_parquet(profiles_path)
        except Exception as e:
            return self._fail("clustering_inputs", f"Failed to read: {e}")

        errors = []
        warnings = []

        # Check required columns
        required = ["zip_code", "date", "profile"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")

        # Check profile lengths
        if "profile" in df.columns:
            lengths = df.select(pl.col("profile").list.len()).unique()["profile"].to_list()
            if len(lengths) > 1:
                errors.append(f"Inconsistent profile lengths: {lengths}")
            elif lengths[0] != 48:
                errors.append(f"Expected 48-point profiles, got {lengths[0]}")

        stats = {
            "profiles": len(df),
            "zip_codes": df["zip_code"].n_unique() if "zip_code" in df.columns else 0,
            "dates": df["date"].n_unique() if "date" in df.columns else 0,
        }

        return self._result("clustering_inputs", errors, warnings, stats)

    def validate_clustering_outputs(self) -> dict:
        """Validate clustering output files."""
        results_dir = self.paths["clustering_dir"] / "results"
        assignments_path = results_dir / "cluster_assignments.parquet"

        if not assignments_path.exists():
            return self._skip("clustering_outputs", "No clustering results yet")

        logger.info(f"Validating clustering outputs: {results_dir}")

        try:
            assignments = pl.read_parquet(assignments_path)
        except Exception as e:
            return self._fail("clustering_outputs", f"Failed to read: {e}")

        errors = []
        warnings = []

        # Check required columns
        if "cluster" not in assignments.columns:
            errors.append("Missing 'cluster' column")

        # Check cluster distribution
        if "cluster" in assignments.columns:
            cluster_counts = assignments["cluster"].value_counts()

            # Check for empty clusters
            if cluster_counts["count"].min() == 0:
                warnings.append("Some clusters have no assignments")

        stats = {
            "n_assigned": len(assignments),
            "k": assignments["cluster"].n_unique() if "cluster" in assignments.columns else 0,
        }

        # Load metrics if available
        metrics_path = results_dir / "clustering_metrics.json"
        if metrics_path.exists():
            import json

            with open(metrics_path) as f:
                metrics = json.load(f)
                stats["silhouette"] = metrics.get("silhouette_score")
                stats["inertia"] = metrics.get("inertia")

        return self._result("clustering_outputs", errors, warnings, stats)

    # =========================================================================
    # ORCHESTRATION METHODS
    # =========================================================================

    def run_full_pipeline(
        self,
        year_month: str,
        num_files: int,
        sample_days: int = DEFAULT_CLUSTERING_CONFIG["sample_days"],
        sample_zips: int = DEFAULT_CLUSTERING_CONFIG["sample_zips"],
        k_min: int = DEFAULT_CLUSTERING_CONFIG["k_min"],
        k_max: int = DEFAULT_CLUSTERING_CONFIG["k_max"],
        max_eval_samples: int = DEFAULT_CLUSTERING_CONFIG["max_eval_samples"],
        eval_max_iter: int = DEFAULT_CLUSTERING_CONFIG["eval_max_iter"],
        skip_clustering: bool = False,
        run_stage2: bool = False,
    ) -> bool:
        """
        Execute the complete pipeline.

        Args:
            year_month: Target month (YYYYMM format)
            num_files: Number of S3 files to download
            sample_days: Days to sample for clustering
            sample_zips: ZIP codes to sample for clustering
            k_min: Minimum clusters to test
            k_max: Maximum clusters to test
            max_eval_samples: Max profiles for k evaluation subsample
            eval_max_iter: Max iterations for k evaluation
            skip_clustering: If True, stop after preparing data
            run_stage2: If True, run demographic regression after clustering

        Returns:
            True if all steps succeed, False otherwise
        """
        self._print_header("COMED PIPELINE EXECUTION")
        print(f"Year-Month: {year_month}")
        print(f"Files: {num_files}")
        print(f"Output: {self.run_dir}")
        print(
            f"Clustering: {'Skipped' if skip_clustering else f'k={k_min}-{k_max}, eval on {max_eval_samples} samples'}"
        )
        print(f"Stage 2: {'Yes' if run_stage2 else 'No'}")

        self.setup_directories()

        # Step 1: Download
        self._print_step("DOWNLOADING FROM S3")
        if not self.download_from_s3(year_month, num_files):
            return False

        # Step 2: Process
        self._print_step("PROCESSING RAW DATA")
        if not self.process_raw_data(year_month):
            return False

        # Step 3: Prepare clustering
        self._print_step("PREPARING CLUSTERING DATA")
        if not self.prepare_clustering_data(sample_days, sample_zips):
            return False

        # Step 4: Cluster (optional)
        if not skip_clustering:
            self._print_step("RUNNING DTW CLUSTERING")
            if not self.run_clustering(
                k_min=k_min,
                k_max=k_max,
                max_eval_samples=max_eval_samples,
                eval_max_iter=eval_max_iter,
            ):
                return False

        # Step 5: Stage 2 regression (optional)
        if run_stage2 and not skip_clustering:
            self._print_step("RUNNING STAGE 2: DEMOGRAPHIC REGRESSION")
            if not self.run_stage2_regression():
                logger.warning("Stage 2 regression failed, but Stage 1 completed successfully")

        logger.info("Pipeline execution complete")
        return True

    def validate_all(self) -> bool:
        """
        Run all validation checks.

        Returns:
            True if all critical validations pass, False otherwise
        """
        self._print_header("VALIDATION")

        self.results["processed"] = self.validate_processed_data()
        self.results["clustering_inputs"] = self.validate_clustering_inputs()
        self.results["clustering_outputs"] = self.validate_clustering_outputs()

        return self._print_summary()

    def validate_stage(self, stage: str) -> bool:
        """Validate a specific pipeline stage."""
        if stage == "processed":
            self.results["processed"] = self.validate_processed_data()
        elif stage == "clustering":
            self.results["clustering_inputs"] = self.validate_clustering_inputs()
            self.results["clustering_outputs"] = self.validate_clustering_outputs()
        else:
            logger.error(f"Unknown stage: {stage}")
            return False

        return self._print_summary()

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _print_header(self, title: str) -> None:
        print(f"\n{'=' * 70}")
        print(title)
        print(f"{'=' * 70}")

    def _print_step(self, title: str) -> None:
        print(f"\n{'─' * 70}")
        print(title)
        print(f"{'─' * 70}")

    def _result(self, stage: str, errors: list, warnings: list, stats: dict) -> dict:
        status = "PASS" if not errors else "FAIL"
        icon = "✅" if status == "PASS" else "❌"
        print(f"\n{icon} {stage.upper()}: {status}")

        for e in errors:
            print(f"   Error: {e}")
        for w in warnings:
            print(f"   ⚠️  {w}")

        return {"status": status, "errors": errors, "warnings": warnings, "stats": stats}

    def _fail(self, stage: str, message: str) -> dict:
        print(f"\n❌ {stage.upper()}: FAILED - {message}")
        return {"status": "FAIL", "errors": [message], "warnings": [], "stats": {}}

    def _skip(self, stage: str, message: str) -> dict:
        print(f"\n⏭️  {stage.upper()}: SKIPPED - {message}")
        return {"status": "SKIP", "errors": [], "warnings": [], "stats": {}}

    def _print_summary(self) -> bool:
        self._print_header("SUMMARY")

        all_passed = True
        for stage, result in self.results.items():
            status = result.get("status", "UNKNOWN")
            icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}.get(status, "❓")
            print(f"{icon} {stage}: {status}")

            if status == "FAIL":
                all_passed = False

        # Show clustering summary if available
        if "clustering_outputs" in self.results:
            stats = self.results["clustering_outputs"].get("stats", {})
            if stats.get("k"):
                print("\nClustering Results:")
                print(f"  • {stats.get('n_assigned', '?')} profiles → {stats.get('k')} clusters")
                if stats.get("silhouette"):
                    print(f"  • Silhouette score: {stats['silhouette']:.3f}")

        print()
        return all_passed


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="ComEd Smart Meter Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with 1000 files
  python run_comed_pipeline.py --from-s3 --num-files 1000

  # Quick test with 100 files, skip clustering
  python run_comed_pipeline.py --from-s3 --num-files 100 --skip-clustering

  # Validate existing results
  python run_comed_pipeline.py --validate-only --run-name 202308_1000

  # Custom clustering parameters
  python run_comed_pipeline.py --from-s3 --num-files 1000 \\
      --sample-zips 200 --sample-days 15 --k-range 3 5
        """,
    )

    # Mode selection
    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument(
        "--from-s3",
        action="store_true",
        help="Run full pipeline: download from S3, process, cluster, validate",
    )
    mode_group.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing files (no processing)",
    )
    mode_group.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Run pipeline but skip clustering step (useful for testing)",
    )
    mode_group.add_argument(
        "--run-stage2",
        action="store_true",
        help="Run Stage 2 regression after clustering (requires census data)",
    )

    # Data selection
    data_group = parser.add_argument_group("Data Selection")
    data_group.add_argument(
        "--year-month",
        default="202308",
        help="Target month in YYYYMM format (default: 202308)",
    )
    data_group.add_argument(
        "--num-files",
        type=int,
        default=1000,
        help="Number of S3 files to download (default: 1000)",
    )

    # Clustering parameters
    cluster_group = parser.add_argument_group("Clustering Parameters")
    cluster_group.add_argument(
        "--sample-zips",
        type=int,
        default=DEFAULT_CLUSTERING_CONFIG["sample_zips"],
        help=f"ZIP+4 codes to sample (default: {DEFAULT_CLUSTERING_CONFIG['sample_zips']})",
    )
    cluster_group.add_argument(
        "--sample-days",
        type=int,
        default=DEFAULT_CLUSTERING_CONFIG["sample_days"],
        help=f"Days to sample (default: {DEFAULT_CLUSTERING_CONFIG['sample_days']})",
    )
    cluster_group.add_argument(
        "--k-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[DEFAULT_CLUSTERING_CONFIG["k_min"], DEFAULT_CLUSTERING_CONFIG["k_max"]],
        help=f"Cluster range to test (default: {DEFAULT_CLUSTERING_CONFIG['k_min']} {DEFAULT_CLUSTERING_CONFIG['k_max']})",
    )
    cluster_group.add_argument(
        "--day-strategy",
        choices=["stratified", "random"],
        default=DEFAULT_CLUSTERING_CONFIG["day_strategy"],
        help="Day sampling strategy (default: stratified = 70%% weekday, 30%% weekend)",
    )

    # Performance tuning
    perf_group = parser.add_argument_group("Performance Tuning")
    perf_group.add_argument(
        "--max-eval-samples",
        type=int,
        default=DEFAULT_CLUSTERING_CONFIG["max_eval_samples"],
        help=f"Max profiles for k evaluation subsample (default: {DEFAULT_CLUSTERING_CONFIG['max_eval_samples']})",
    )
    perf_group.add_argument(
        "--eval-max-iter",
        type=int,
        default=DEFAULT_CLUSTERING_CONFIG["eval_max_iter"],
        help=f"Max iterations for k evaluation (default: {DEFAULT_CLUSTERING_CONFIG['eval_max_iter']})",
    )
    perf_group.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: k=3-4, 1000 eval samples, 5 iterations (for testing)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--run-name",
        help="Name for this run (default: {year_month}_{num_files})",
    )
    output_group.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Project root directory (default: current directory)",
    )
    output_group.add_argument(
        "--stage",
        choices=["processed", "clustering", "all"],
        default="all",
        help="Stage to validate (default: all)",
    )

    args = parser.parse_args()

    # Handle --fast mode
    if args.fast:
        args.k_range = [3, 4]
        args.max_eval_samples = 1000
        args.eval_max_iter = 5
        logger.info("Fast mode enabled: k=3-4, 1000 eval samples, 5 iterations")

    # Determine run name
    run_name = args.run_name or f"{args.year_month}_{args.num_files}" if args.from_s3 else args.run_name

    # Create pipeline
    pipeline = ComedPipeline(args.base_dir, run_name)

    # Execute based on mode
    if args.from_s3:
        success = pipeline.run_full_pipeline(
            year_month=args.year_month,
            num_files=args.num_files,
            sample_days=args.sample_days,
            sample_zips=args.sample_zips,
            k_min=args.k_range[0],
            k_max=args.k_range[1],
            max_eval_samples=args.max_eval_samples,
            eval_max_iter=args.eval_max_iter,
            skip_clustering=args.skip_clustering,
            run_stage2=args.run_stage2,
        )
        if success:
            pipeline.validate_all()
    elif args.validate_only:
        success = pipeline.validate_all() if args.stage == "all" else pipeline.validate_stage(args.stage)
    else:
        parser.print_help()
        print("\n⚠️  Specify --from-s3 to run pipeline or --validate-only to check existing files")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
