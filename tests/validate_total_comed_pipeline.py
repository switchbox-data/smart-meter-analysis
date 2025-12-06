#!/usr/bin/env python3
"""
ComEd Smart Meter Analysis Pipeline Validation

Validates data integrity and correctness across all stages of the ComEd smart meter
clustering analysis. Can operate on existing local files or pull fresh test data
from AWS S3 to perform end-to-end validation.

Validation Stages:
    1. Processed Data - Interval-level energy data after wide-to-long transformation
    2. Enriched Data - Energy data joined with Census demographics
    3. Clustering Inputs - Daily load profiles and ZIP+4 demographics
    4. Clustering Outputs - Cluster assignments, centroids, and evaluation metrics

Usage:
    # Validate existing local files
    python validate_total_comed_pipeline.py

    # Pull fresh data from S3 and run full validation
    python validate_total_comed_pipeline.py --from-s3 --num-files 1000

    # Validate specific stage only
    python validate_total_comed_pipeline.py --stage clustering
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Default paths for production data
DEFAULT_PATHS = {
    "processed": Path("data/processed/comed_202308.parquet"),
    "enriched": Path("data/enriched_test/enriched.parquet"),
    "clustering_dir": Path("data/clustering"),
    "crosswalk": Path("data/reference/2023_comed_zip4_census_crosswalk.txt"),
}


class PipelineValidator:
    """
    Validates data quality and integrity across the ComEd analysis pipeline.

    This validator performs comprehensive checks at each pipeline stage to ensure
    data transformations preserve integrity and outputs meet quality standards
    required for reliable clustering analysis.
    """

    def __init__(self, base_dir: Path, run_name: str | None = None):
        """
        Initialize validator with project base directory.

        Args:
            base_dir: Root directory of the smart-meter-analysis project
            run_name: Optional name for this validation run (used for S3 test data)
        """
        self.base_dir = base_dir
        self.run_name = run_name
        self.results: dict[str, dict] = {}

        # If run_name provided, use validation_runs directory structure
        if run_name:
            self.run_dir = base_dir / "data" / "validation_runs" / run_name
            # Extract year_month from run_name (e.g., "202308_1000" -> "202308")
            self.year_month = run_name.split("_")[0] if "_" in run_name else run_name
            self.paths = {
                "samples": self.run_dir / "samples",
                "processed": self.run_dir / "processed" / f"comed_{self.year_month}.parquet",
                "enriched": self.run_dir / "enriched" / "enriched.parquet",
                "clustering_dir": self.run_dir / "clustering",
            }
        else:
            self.run_dir = None
            self.paths = DEFAULT_PATHS.copy()

    def setup_run_directories(self) -> None:
        """Create directory structure for a validation run."""
        if not self.run_dir:
            return

        for subdir in ["samples", "processed", "enriched", "clustering/results"]:
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Created validation run directory: {self.run_dir}")

    def download_from_s3(
        self,
        year_month: str,
        num_files: int,
        bucket: str = "smart-meter-data-sb",
        prefix: str = "sharepoint-files/Zip4/",
    ) -> bool:
        """
        Download sample CSV files from S3 for validation testing.

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

        logger.info(f"Connecting to S3 bucket: {bucket}")

        try:
            s3 = boto3.client("s3")
            full_prefix = f"{prefix}{year_month}/"

            # List available files
            paginator = s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=full_prefix)

            csv_keys = []
            for page in pages:
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

            logger.info(f"Found {len(csv_keys)} files, downloading to {self.paths['samples']}")

            # Download files
            for i, key in enumerate(csv_keys, 1):
                filename = Path(key).name
                local_path = self.paths["samples"] / filename

                if i % 100 == 0 or i == len(csv_keys):
                    logger.info(f"  Downloaded {i}/{len(csv_keys)} files")

                s3.download_file(bucket, key, str(local_path))

            logger.info(f"Successfully downloaded {len(csv_keys)} files")
            return True

        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            logger.error("Verify AWS credentials are configured (aws configure or environment variables)")
            return False

    def run_processing_stage(self, year_month: str) -> bool:
        """
        Execute the data processing stage on downloaded samples.

        Uses lazy evaluation via Polars scan_csv and sink_parquet to process
        files without loading all data into memory simultaneously.

        Args:
            year_month: Month identifier for output file naming

        Returns:
            True if processing successful, False otherwise
        """
        csv_files = sorted(self.paths["samples"].glob("*.csv"))
        if not csv_files:
            logger.error(f"No CSV files found in {self.paths['samples']}")
            return False

        logger.info(f"Processing {len(csv_files)} CSV files using lazy evaluation")

        # Import the schema and transformation functions from aws_loader
        from smart_meter_analysis.aws_loader import (
            COMED_SCHEMA,
            add_time_columns_lazy,
            transform_wide_to_long_lazy,
        )

        # Build lazy frames for each CSV
        lazy_frames = []
        for i, csv_path in enumerate(csv_files, 1):
            if i % 200 == 0 or i == len(csv_files):
                logger.info(f"  Scanned {i}/{len(csv_files)} files")

            try:
                lf = pl.scan_csv(str(csv_path), schema_overrides=COMED_SCHEMA, ignore_errors=True)
                lf_long = transform_wide_to_long_lazy(lf)
                lf_time = add_time_columns_lazy(lf_long, day_mode="calendar")
                lazy_frames.append(lf_time)
            except Exception as e:
                logger.warning(f"Failed to scan {csv_path.name}: {e}")
                continue

        if not lazy_frames:
            logger.error("No files were successfully scanned")
            return False

        logger.info(f"Concatenating {len(lazy_frames)} lazy frames and writing to parquet")

        # Concatenate lazily and sink to parquet (memory-efficient)
        lf_combined = pl.concat(lazy_frames, how="diagonal_relaxed")

        # Ensure output directory exists
        self.paths["processed"].parent.mkdir(parents=True, exist_ok=True)

        # sink_parquet executes the lazy query and writes directly to disk
        lf_combined.sink_parquet(self.paths["processed"])

        # Read back row count for logging
        row_count = pl.scan_parquet(self.paths["processed"]).select(pl.len()).collect()[0, 0]
        logger.info(f"Wrote {row_count:,} records to {self.paths['processed']}")

        return True

    def run_enrichment_stage(self, year_month: str) -> bool:
        """
        Execute the census enrichment stage using lazy evaluation.

        Joins processed energy data with Census demographics via ZIP+4 crosswalk.
        Uses streaming joins to handle large datasets without excessive memory.

        Args:
            year_month: Month identifier for locating input file

        Returns:
            True if enrichment successful, False otherwise
        """
        from smart_meter_analysis.census import fetch_census_data

        crosswalk_path = self.base_dir / DEFAULT_PATHS["crosswalk"]
        if not crosswalk_path.exists():
            logger.error(f"Crosswalk file not found: {crosswalk_path}")
            return False

        # Ensure output directory exists
        self.paths["enriched"].parent.mkdir(parents=True, exist_ok=True)

        # Census cache location
        cache_dir = self.base_dir / "data" / "reference"
        cache_dir.mkdir(parents=True, exist_ok=True)
        census_cache = cache_dir / "census_17_2023.parquet"

        try:
            # Load or fetch census data
            if census_cache.exists():
                logger.info(f"Loading cached census data from {census_cache}")
                census_df = pl.read_parquet(census_cache)
            else:
                logger.info("Fetching census data from API")
                census_df = fetch_census_data(state_fips="17", acs_year=2023)
                census_df.write_parquet(census_cache)
                logger.info(f"Cached census data to {census_cache}")

            logger.info(f"Census data: {len(census_df):,} block groups")

            # Load crosswalk and create enriched mapping
            logger.info(f"Loading crosswalk from {crosswalk_path}")
            crosswalk = pl.read_csv(crosswalk_path, separator="\t", infer_schema_length=10000)
            logger.info(f"  Loaded {len(crosswalk):,} ZIP+4 mappings")

            # Create standardized join keys
            crosswalk = crosswalk.with_columns([
                (pl.col("Zip").cast(pl.Utf8).str.zfill(5) + "-" + pl.col("Zip4").cast(pl.Utf8).str.zfill(4)).alias(
                    "zip4"
                ),
                pl.col("CensusKey2023").cast(pl.Utf8).str.zfill(15).str.slice(0, 12).alias("block_group_geoid"),
            ]).select(["zip4", "block_group_geoid"])

            # Prepare census data for join
            census_df = census_df.with_columns(pl.col("GEOID").cast(pl.Utf8).str.zfill(12).alias("block_group_geoid"))
            census_cols = [
                c for c in census_df.columns if c not in ["GEOID", "NAME", "state", "county", "tract", "block group"]
            ]
            census_for_join = census_df.select(
                ["block_group_geoid"] + [c for c in census_cols if c != "block_group_geoid"]
            )

            # Join crosswalk with census
            logger.info("Creating enriched crosswalk")
            enriched_crosswalk = crosswalk.join(census_for_join, on="block_group_geoid", how="left")
            logger.info(f"  Enriched crosswalk: {len(enriched_crosswalk):,} rows")

            # Lazy join with energy data
            logger.info("Joining energy data with demographics (lazy)")
            energy_lf = pl.scan_parquet(self.paths["processed"])
            crosswalk_lf = enriched_crosswalk.lazy()

            enriched_lf = energy_lf.join(
                crosswalk_lf,
                left_on="zip_code",
                right_on="zip4",
                how="left",
            )

            # Sink to parquet
            logger.info(f"Writing enriched data to {self.paths['enriched']}")
            enriched_lf.sink_parquet(self.paths["enriched"])

            # Get row count for logging
            row_count = pl.scan_parquet(self.paths["enriched"]).select(pl.len()).collect()[0, 0]
            logger.info(f"Enrichment complete: {row_count:,} records")

            return True

        except Exception as e:
            logger.error(f"Enrichment failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_clustering_prep_stage(self) -> bool:
        """
        Execute the clustering data preparation stage.

        Aggregates interval data to daily ZIP+4 profiles for clustering.
        Works directly from processed data - demographic enrichment is
        handled separately in Stage 2.

        Returns:
            True if preparation successful, False otherwise
        """
        import subprocess

        clustering_dir = self.paths["clustering_dir"]
        clustering_dir.mkdir(parents=True, exist_ok=True)

        # Use processed data directly (not enriched)
        input_path = self.paths["processed"]

        if not input_path.exists():
            logger.error(f"Processed data not found: {input_path}")
            return False

        cmd = [
            sys.executable,
            str(self.base_dir / "analysis" / "clustering" / "prepare_clustering_data.py"),
            "--input",
            str(input_path),
            "--output-dir",
            str(clustering_dir),
            "--day-strategy",
            "stratified",
            "--sample-days",
            "20",
            "--sample-zips",
            "500",
        ]

        logger.info(f"Preparing clustering data from {input_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Clustering prep failed: {result.stderr}")
            # Print stdout too for debugging
            if result.stdout:
                logger.error(f"stdout: {result.stdout}")
            return False

        logger.info(f"Clustering data prepared: {clustering_dir}")
        return True

    def run_clustering_stage(self) -> bool:
        """
        Execute the DTW clustering stage.

        Performs k-means clustering with DTW distance on daily load profiles.

        Returns:
            True if clustering successful, False otherwise
        """
        import subprocess

        clustering_dir = self.paths["clustering_dir"]
        results_dir = clustering_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        profiles_path = clustering_dir / "sampled_profiles.parquet"
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
            "3",
            "6",
            "--find-optimal-k",
            "--normalize",
        ]

        logger.info("Running DTW clustering")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Clustering failed: {result.stderr}")
            return False

        logger.info(f"Clustering complete: {results_dir}")
        return True

    def validate_processed_data(self, path: Path | None = None) -> dict:
        """
        Validate processed energy data quality.

        Verifies the wide-to-long transformation produced valid interval-level
        data with expected schema, reasonable value ranges, and complete coverage.

        Args:
            path: Path to processed parquet file (uses default if not specified)

        Returns:
            Validation result dictionary with status, errors, warnings, and statistics
        """
        stage = "processed"
        print(f"\n{'=' * 70}")
        print("STAGE 1: PROCESSED DATA VALIDATION")
        print(f"{'=' * 70}")

        path = path or self.paths.get("processed") or DEFAULT_PATHS["processed"]

        if not path.exists():
            return self._fail(stage, f"File not found: {path}")

        print(f"File: {path}")

        errors = []
        warnings = []
        stats = {}

        try:
            df = pl.read_parquet(path)
            stats["rows"] = len(df)
            stats["columns"] = len(df.columns)
            print(f"Shape: {stats['rows']:,} rows × {stats['columns']} columns")

            # Schema validation
            required = ["zip_code", "account_identifier", "datetime", "kwh", "date", "hour"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                errors.append(f"Missing required columns: {missing}")
            else:
                print("✓ Required columns present")

            # Record counts
            stats["accounts"] = df["account_identifier"].n_unique()
            stats["zip_codes"] = df["zip_code"].n_unique()
            print(f"✓ Unique accounts: {stats['accounts']:,}")
            print(f"✓ Unique ZIP+4 codes: {stats['zip_codes']:,}")

            # Temporal coverage
            stats["min_date"] = str(df["date"].min())
            stats["max_date"] = str(df["date"].max())
            stats["unique_dates"] = df["date"].n_unique()
            print(f"✓ Date range: {stats['min_date']} to {stats['max_date']} ({stats['unique_dates']} days)")

            # Energy value validation
            kwh_min = df["kwh"].min()
            kwh_max = df["kwh"].max()
            kwh_mean = df["kwh"].mean()
            stats["kwh_min"] = float(kwh_min) if kwh_min is not None else None
            stats["kwh_max"] = float(kwh_max) if kwh_max is not None else None
            stats["kwh_mean"] = float(kwh_mean) if kwh_mean is not None else None

            if kwh_min is not None and kwh_min < 0:
                warnings.append(f"Negative kWh values detected: min={kwh_min:.4f}")
            print(f"✓ kWh range: {kwh_min:.4f} to {kwh_max:.2f} (mean: {kwh_mean:.4f})")

            # Null value check
            null_cols = []
            for col in required:
                if col in df.columns:
                    null_pct = df[col].null_count() / len(df) * 100
                    if null_pct > 0:
                        null_cols.append(f"{col}: {null_pct:.2f}%")

            if null_cols:
                warnings.append(f"Null values found: {', '.join(null_cols)}")
            else:
                print("✓ No null values in required columns")

        except Exception as e:
            return self._fail(stage, f"Error reading file: {e}")

        return self._result(stage, errors, warnings, stats)

    def validate_enriched_data(self, path: Path | None = None) -> dict:
        """
        Validate census-enriched energy data.

        Verifies geographic join success rate and presence of demographic variables
        required for the clustering analysis.

        Args:
            path: Path to enriched parquet file (uses default if not specified)

        Returns:
            Validation result dictionary with status, errors, warnings, and statistics
        """
        stage = "enriched"
        print(f"\n{'=' * 70}")
        print("STAGE 2: ENRICHED DATA VALIDATION")
        print(f"{'=' * 70}")

        path = path or self.paths.get("enriched") or DEFAULT_PATHS["enriched"]

        if not path.exists():
            return self._skip(stage, f"File not found: {path}")

        print(f"File: {path}")

        errors = []
        warnings = []
        stats = {}

        try:
            df = pl.read_parquet(path)
            stats["rows"] = len(df)
            stats["columns"] = len(df.columns)
            print(f"Shape: {stats['rows']:,} rows × {stats['columns']} columns")

            # Geographic enrichment validation
            if "block_group_geoid" not in df.columns:
                errors.append("Missing block_group_geoid column - census join may have failed")
            else:
                matched = df.filter(pl.col("block_group_geoid").is_not_null()).height
                match_rate = matched / len(df) * 100
                stats["geographic_match_rate"] = match_rate
                stats["block_groups"] = df["block_group_geoid"].n_unique()

                if match_rate < 90:
                    errors.append(f"Geographic match rate below 90%: {match_rate:.1f}%")
                elif match_rate < 95:
                    warnings.append(f"Geographic match rate below 95%: {match_rate:.1f}%")
                else:
                    print(f"✓ Geographic match rate: {match_rate:.1f}%")

                print(f"✓ Unique block groups: {stats['block_groups']:,}")

            # Demographic variable validation
            census_indicators = ["Total_Households", "Median_Household_Income", "Owner_Occupied"]
            found_census = [c for c in census_indicators if c in df.columns]

            if not found_census:
                errors.append("No census demographic variables found")
            else:
                excluded = {
                    "zip_code",
                    "account_identifier",
                    "datetime",
                    "kwh",
                    "date",
                    "hour",
                    "weekday",
                    "is_weekend",
                    "block_group_geoid",
                    "delivery_service_class",
                    "delivery_service_name",
                    "is_spring_forward_day",
                    "is_fall_back_day",
                    "is_dst_day",
                }
                census_cols = [c for c in df.columns if c not in excluded]
                stats["census_variables"] = len(census_cols)
                print(f"✓ Census variables: {stats['census_variables']}")

        except Exception as e:
            return self._fail(stage, f"Error reading file: {e}")

        return self._result(stage, errors, warnings, stats)

    def validate_clustering_inputs(self) -> dict:
        """
        Validate clustering input data structures.

        Ensures daily load profiles have the expected 48-point structure and
        demographic data provides complete coverage of profiled ZIP+4 codes.

        Returns:
            Validation result dictionary with status, errors, warnings, and statistics
        """
        stage = "clustering_inputs"
        print(f"\n{'=' * 70}")
        print("STAGE 3: CLUSTERING INPUTS VALIDATION")
        print(f"{'=' * 70}")

        clustering_dir = self.paths.get("clustering_dir") or DEFAULT_PATHS["clustering_dir"]
        profiles_path = clustering_dir / "sampled_profiles.parquet"
        demos_path = clustering_dir / "zip4_demographics.parquet"

        errors = []
        warnings = []
        stats = {}

        # Profile validation
        if not profiles_path.exists():
            return self._skip(stage, f"Profiles not found: {profiles_path}")

        print(f"Profiles: {profiles_path}")

        try:
            profiles = pl.read_parquet(profiles_path)
            stats["n_profiles"] = len(profiles)
            stats["n_zip_codes"] = profiles["zip_code"].n_unique()
            stats["n_dates"] = profiles["date"].n_unique()
            print(f"✓ Profiles: {stats['n_profiles']} ({stats['n_zip_codes']} ZIP codes × {stats['n_dates']} dates)")

            # Profile length validation (must be 48 for 30-minute intervals)
            profile_lengths = profiles.select(pl.col("profile").list.len()).unique()["profile"].to_list()
            stats["profile_lengths"] = profile_lengths

            if profile_lengths != [48]:
                if all(length in [47, 48] for length in profile_lengths):
                    warnings.append(f"Some profiles have 47 intervals (likely DST days): {profile_lengths}")
                else:
                    errors.append(f"Invalid profile lengths detected: {profile_lengths}")
            else:
                print("✓ All profiles have 48 timepoints")

            # Null profile check
            null_profiles = profiles.filter(pl.col("profile").is_null()).height
            if null_profiles > 0:
                errors.append(f"{null_profiles} null profiles detected")

        except Exception as e:
            return self._fail(stage, f"Error reading profiles: {e}")

        # Demographics validation
        if not demos_path.exists():
            warnings.append(f"Demographics file not found: {demos_path}")
        else:
            print(f"Demographics: {demos_path}")
            try:
                demos = pl.read_parquet(demos_path)
                stats["n_demo_zips"] = len(demos)
                stats["n_demo_vars"] = len(demos.columns) - 2  # Exclude zip_code and block_group_geoid
                print(f"✓ Demographics: {stats['n_demo_zips']} ZIP codes, {stats['n_demo_vars']} variables")

                # Coverage validation
                profile_zips = set(profiles["zip_code"].unique().to_list())
                demo_zips = set(demos["zip_code"].unique().to_list())
                missing = profile_zips - demo_zips

                if missing:
                    warnings.append(f"{len(missing)} profile ZIP codes missing demographics")
                else:
                    print("✓ Demographics cover all profile ZIP codes")

            except Exception as e:
                warnings.append(f"Error reading demographics: {e}")

        return self._result(stage, errors, warnings, stats)

    def validate_clustering_outputs(self) -> dict:
        """
        Validate clustering analysis outputs.

        Verifies cluster assignments, metadata, and visualizations were generated
        correctly and that cluster distribution is reasonable.

        Returns:
            Validation result dictionary with status, errors, warnings, and statistics
        """
        stage = "clustering_outputs"
        print(f"\n{'=' * 70}")
        print("STAGE 4: CLUSTERING OUTPUTS VALIDATION")
        print(f"{'=' * 70}")

        clustering_dir = self.paths.get("clustering_dir") or DEFAULT_PATHS["clustering_dir"]
        results_dir = clustering_dir / "results"

        if not results_dir.exists():
            return self._skip(stage, f"Results directory not found: {results_dir}")

        errors = []
        warnings = []
        stats = {}

        # Cluster assignments validation
        assignments_path = results_dir / "cluster_assignments.parquet"
        if not assignments_path.exists():
            errors.append("cluster_assignments.parquet not found")
        else:
            print(f"Assignments: {assignments_path}")
            try:
                assignments = pl.read_parquet(assignments_path)
                stats["n_assigned"] = len(assignments)

                if "cluster" not in assignments.columns:
                    errors.append("cluster column missing from assignments")
                else:
                    clusters = assignments["cluster"].unique().sort().to_list()
                    stats["clusters"] = clusters
                    stats["k"] = len(clusters)
                    print(f"✓ Clusters: {stats['k']} (labels: {clusters})")

                    # Distribution analysis
                    dist = assignments.group_by("cluster").agg(pl.len().alias("count")).sort("cluster")
                    print("✓ Cluster distribution:")
                    for row in dist.iter_rows(named=True):
                        pct = row["count"] / len(assignments) * 100
                        print(f"    Cluster {row['cluster']}: {row['count']} ({pct:.1f}%)")

                    # Flag highly imbalanced clusters
                    min_cluster_size = dist["count"].min()
                    if min_cluster_size < len(assignments) * 0.05:
                        warnings.append(f"Smallest cluster has only {min_cluster_size} profiles (<5%)")

            except Exception as e:
                errors.append(f"Error reading assignments: {e}")

        # Metadata validation
        metadata_path = results_dir / "clustering_metadata.json"
        if not metadata_path.exists():
            errors.append("clustering_metadata.json not found")
        else:
            try:
                with open(metadata_path) as f:
                    meta = json.load(f)
                stats["metadata"] = meta
                print(f"✓ Metadata: k={meta.get('k')}, inertia={meta.get('inertia', 0):.2f}")
            except Exception as e:
                errors.append(f"Error reading metadata: {e}")

        # K evaluation validation (optional - only present if --find-optimal-k was used)
        k_eval_path = results_dir / "k_evaluation.json"
        if k_eval_path.exists():
            try:
                with open(k_eval_path) as f:
                    k_eval = json.load(f)
                best_k_idx = k_eval["silhouette"].index(max(k_eval["silhouette"]))
                best_k = k_eval["k_values"][best_k_idx]
                best_sil = k_eval["silhouette"][best_k_idx]
                stats["best_k"] = best_k
                stats["best_silhouette"] = best_sil
                print(f"✓ K evaluation: best k={best_k} (silhouette={best_sil:.3f})")
            except Exception as e:
                warnings.append(f"Error reading k_evaluation: {e}")

        # Visualization validation
        viz_files = ["elbow_curve.png", "cluster_centroids.png", "cluster_samples.png"]
        missing_viz = [f for f in viz_files if not (results_dir / f).exists()]

        if missing_viz:
            warnings.append(f"Missing visualizations: {missing_viz}")
        else:
            print("✓ All visualizations generated")

        return self._result(stage, errors, warnings, stats)

    def run_full_pipeline(self, year_month: str, num_files: int) -> bool:
        """
        Execute and validate the complete Stage 1 pipeline from S3 through clustering.

        Stage 1 focuses on usage pattern clustering only. Demographic enrichment
        is deferred to Stage 2 (multinomial regression) to reduce memory requirements
        and maintain separation of concerns.

        Args:
            year_month: Target month in YYYYMM format
            num_files: Number of S3 files to download

        Returns:
            True if all stages complete successfully, False otherwise
        """
        print(f"\n{'=' * 70}")
        print("STAGE 1 PIPELINE EXECUTION")
        print(f"{'=' * 70}")
        print(f"Year-Month: {year_month}")
        print(f"Files: {num_files}")
        print(f"Output: {self.run_dir}")

        self.setup_run_directories()

        # Step 1: Download from S3
        print(f"\n{'─' * 70}")
        print("DOWNLOADING FROM S3")
        print(f"{'─' * 70}")
        if not self.download_from_s3(year_month, num_files):
            return False

        # Step 2: Process raw data
        print(f"\n{'─' * 70}")
        print("PROCESSING RAW DATA")
        print(f"{'─' * 70}")
        if not self.run_processing_stage(year_month):
            return False

        # Step 3: Prepare clustering data (from processed, not enriched)
        print(f"\n{'─' * 70}")
        print("PREPARING CLUSTERING DATA")
        print(f"{'─' * 70}")
        if not self.run_clustering_prep_stage():
            return False

        # Step 4: Run clustering
        print(f"\n{'─' * 70}")
        print("RUNNING DTW CLUSTERING")
        print(f"{'─' * 70}")
        if not self.run_clustering_stage():
            return False

        logger.info("Stage 1 pipeline execution complete")
        return True

    def validate_all(self) -> bool:
        """
        Run validation checks on all Stage 1 pipeline outputs.

        Note: Enrichment validation is skipped for Stage 1. Demographic
        enrichment occurs in Stage 2 (multinomial regression).

        Returns:
            True if all critical validations pass, False otherwise
        """
        self.results["processed"] = self.validate_processed_data()
        # Skip enriched validation for Stage 1 - demographics added in Stage 2
        self.results["clustering_inputs"] = self.validate_clustering_inputs()
        self.results["clustering_outputs"] = self.validate_clustering_outputs()

        return self._print_summary()

    def validate_stage(self, stage: str) -> bool:
        """
        Run validation for a specific pipeline stage.

        Args:
            stage: One of 'processed', 'enriched', 'clustering'

        Returns:
            True if validation passes, False otherwise
        """
        if stage == "processed":
            self.results["processed"] = self.validate_processed_data()
        elif stage == "enriched":
            self.results["enriched"] = self.validate_enriched_data()
        elif stage == "clustering":
            self.results["clustering_inputs"] = self.validate_clustering_inputs()
            self.results["clustering_outputs"] = self.validate_clustering_outputs()
        else:
            print(f"Unknown stage: {stage}")
            return False

        return self._print_summary()

    def _result(self, stage: str, errors: list, warnings: list, stats: dict) -> dict:
        """Format validation results for a stage."""
        status = "PASS" if not errors else "FAIL"
        if status == "PASS":
            print(f"\n✅ {stage.upper()}: PASSED")
        else:
            print(f"\n❌ {stage.upper()}: FAILED")
            for e in errors:
                print(f"   Error: {e}")

        if warnings:
            for w in warnings:
                print(f"   ⚠️  {w}")

        return {"status": status, "errors": errors, "warnings": warnings, "stats": stats}

    def _fail(self, stage: str, message: str) -> dict:
        """Create a failed validation result."""
        print(f"\n❌ {stage.upper()}: FAILED - {message}")
        return {"status": "FAIL", "errors": [message], "warnings": [], "stats": {}}

    def _skip(self, stage: str, message: str) -> dict:
        """Create a skipped validation result."""
        print(f"\n⏭️  {stage.upper()}: SKIPPED - {message}")
        return {"status": "SKIP", "errors": [], "warnings": [], "stats": {}}

    def _print_summary(self) -> bool:
        """Print validation summary and return overall success status."""
        print(f"\n{'=' * 70}")
        print("VALIDATION SUMMARY")
        print(f"{'=' * 70}")

        all_passed = True
        for stage, result in self.results.items():
            status = result.get("status", "UNKNOWN")
            icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}.get(status, "❓")
            print(f"{icon} {stage}: {status}")

            if status == "FAIL":
                all_passed = False
                for e in result.get("errors", []):
                    print(f"      Error: {e}")

        print()
        if all_passed:
            print("✓ All validations passed")

            # Report Stage 2 readiness
            if "clustering_outputs" in self.results:
                out_stats = self.results["clustering_outputs"].get("stats", {})
                if out_stats.get("k"):
                    print("\nStage 1 Analysis Complete:")
                    print(f"  • {out_stats.get('n_assigned', '?')} profiles clustered into {out_stats.get('k')} groups")
                    print("  • Ready for Stage 2: Multinomial logistic regression")
        else:
            print("⚠️  Some validations failed. Review errors above.")

        return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate ComEd smart meter analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate existing local files
    python validate_total_comed_pipeline.py

    # Download from S3 and run full pipeline validation
    python validate_total_comed_pipeline.py --from-s3 --num-files 1000

    # Validate specific stage only
    python validate_total_comed_pipeline.py --stage clustering
        """,
    )
    parser.add_argument(
        "--stage",
        choices=["processed", "enriched", "clustering", "all"],
        default="all",
        help="Pipeline stage to validate (default: all)",
    )
    parser.add_argument("--from-s3", action="store_true", help="Download fresh test data from S3 and run full pipeline")
    parser.add_argument("--num-files", type=int, default=1000, help="Number of S3 files to download (default: 1000)")
    parser.add_argument("--year-month", default="202308", help="Target month in YYYYMM format (default: 202308)")
    parser.add_argument(
        "--base-dir", type=Path, default=Path("."), help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--run-name", help="Name for this validation run (default: auto-generated from year-month and num-files)"
    )

    args = parser.parse_args()

    # Generate run name if pulling from S3
    run_name = None
    if args.from_s3:
        run_name = args.run_name or f"{args.year_month}_{args.num_files}"

    validator = PipelineValidator(args.base_dir, run_name)

    if args.from_s3:
        # Full pipeline: download, process, enrich, cluster, validate
        if not validator.run_full_pipeline(args.year_month, args.num_files):
            print("\n❌ Pipeline execution failed")
            sys.exit(1)

        # Validate all outputs
        success = validator.validate_all()
    elif args.stage == "all":
        success = validator.validate_all()
    else:
        success = validator.validate_stage(args.stage)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
