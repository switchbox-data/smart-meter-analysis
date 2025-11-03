#!/usr/bin/env python3
"""
End-to-end pipeline for ComEd smart meter analysis.

This script demonstrates how all the pieces fit together:
1. Download raw data from ComEd sources (or use local samples)
2. Transform to clean format
3. Enrich with Census data (crosswalk)
4. Run analysis

DATA SOURCES:
- 's3': Pull from S3 bucket (production data)
- 'local': Use sample files from data/samples/ (for testing)

Currently, step 3 is blocked on the Census crosswalk file.
The script will run steps 1-2 and fail gracefully at step 3 with a clear message.

Usage:
    # Test with local sample data
    python run_comed_pipeline.py --source local

    # Process from S3 with first 10 files
    python run_comed_pipeline.py --year-month 202308 --max-files 10 --source s3

    # Process full month from S3
    python run_comed_pipeline.py --year-month 202308 --source s3

    # Skip download if data already exists
    python run_comed_pipeline.py --year-month 202308 --skip-download --source s3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def step1_process_local_samples(skip_processing: bool) -> Path:
    """
    Step 1 (Local Mode): Process sample CSV files from data/samples/.

    This provides a fast, self-contained test loop without needing S3 access.
    Perfect for testing transformations and inspecting intermediate results.

    Returns: Path to output Parquet file
    """
    import polars as pl

    from smart_meter_analysis.transformation import add_time_columns, transform_wide_to_long

    sample_dir = Path("data/samples")
    output_path = Path("data/processed/comed_samples.parquet")

    if skip_processing and output_path.exists():
        logger.info(f"Skipping processing - using existing file: {output_path}")
        return output_path

    logger.info("=" * 80)
    logger.info("STEP 1: Process Local Sample Data")
    logger.info("=" * 80)

    if not sample_dir.exists():
        logger.error(f"Sample directory not found: {sample_dir}")
        logger.error("Download sample data first:")
        logger.error("  just download-samples 202308 5")
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    # Find all CSV files in samples directory
    csv_files = list(sample_dir.glob("*.csv"))

    if not csv_files:
        logger.error(f"No CSV files found in: {sample_dir}")
        logger.error("Download sample data first:")
        logger.error("  just download-samples 202308 5")
        raise FileNotFoundError(f"No CSV files in: {sample_dir}")

    logger.info(f"Found {len(csv_files)} sample files")

    # Process each file
    dfs = []
    for i, csv_path in enumerate(sorted(csv_files), 1):
        logger.info(f"Processing {i}/{len(csv_files)}: {csv_path.name}")
        try:
            # Read CSV
            df_raw = pl.read_csv(csv_path, ignore_errors=True, infer_schema_length=5000)

            # Transform
            df_long = transform_wide_to_long(df_raw)
            df_time = add_time_columns(df_long)

            dfs.append(df_time)
            logger.info(f"  Transformed to {df_time.height:,} rows")
        except Exception:
            logger.exception(f"Failed to process {csv_path}")
            continue

    if not dfs:
        raise ValueError("No files were successfully processed")

    # Combine and save
    combined = pl.concat(dfs, how="diagonal")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output_path)

    logger.info(f"✓ Step 1 complete: {output_path}")
    logger.info(f"  Total rows: {combined.height:,}")
    logger.info(f"  Unique accounts: {combined['account_identifier'].n_unique()}")
    logger.info(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")

    return output_path


def step1_download_transform(year_month: str, max_files: int | None, skip_download: bool) -> Path:
    """
    Step 1 (S3 Mode): Download and transform ComEd data from S3.

    Downloads raw CSV files from S3, transforms from wide to long format,
    adds time features, and writes to a single Parquet file.

    Uses lazy semantics - streams data through transformations without
    loading everything into memory.

    Returns: Path to output Parquet file
    """
    from smart_meter_analysis.step0_aws import process_month_batch

    output_path = Path(f"data/processed/comed_{year_month}.parquet")

    if skip_download and output_path.exists():
        logger.info(f"Skipping download - using existing file: {output_path}")
        return output_path

    logger.info("=" * 80)
    logger.info("STEP 1: Download and Transform ComEd Data from S3")
    logger.info("=" * 80)

    process_month_batch(
        year_month=year_month,
        output_path=output_path,
        max_files=max_files,
        sort_output=False,  # Don't sort - use group_by in analysis instead
    )

    logger.info(f"✓ Step 1 complete: {output_path}")
    return output_path


def step2_add_census_crosswalk(input_path: Path, crosswalk_path: Path | None) -> Path:
    """
    Step 2: Enrich with Census data using ZIP→Tract crosswalk.

    CURRENTLY BLOCKED: Waiting for crosswalk file.

    This step will:
    - Join smart meter data with Census geography (ZIP → Tract)
    - Add demographic features from Census ACS data
    - Filter to relevant tracts/demographics

    Returns: Path to enriched Parquet file
    """
    logger.info("=" * 80)
    logger.info("STEP 2: Add Census Crosswalk")
    logger.info("=" * 80)

    if crosswalk_path is None:
        crosswalk_path = Path("data/crosswalks/zip_to_tract.parquet")

    if not crosswalk_path.exists():
        logger.error(f"✗ Crosswalk file not found: {crosswalk_path}")
        logger.error("")
        logger.error("This step is currently blocked pending the Census crosswalk.")
        logger.error("Expected file: ZIP→Tract mapping with demographic data")
        logger.error("")
        logger.error("Once you have the crosswalk:")
        logger.error(f"  1. Place it at: {crosswalk_path}")
        logger.error("  2. Implement the join logic in step2_census.py")
        logger.error("  3. Re-run this pipeline")
        logger.error("")
        logger.error("Pipeline will stop here for now.")
        raise FileNotFoundError(f"Crosswalk not found: {crosswalk_path}")

    # TODO: Implement census crosswalk join
    # from smart_meter_analysis.step2_census import add_census_features
    # output_path = input_path.with_name(input_path.stem + "_with_census.parquet")
    # add_census_features(input_path, crosswalk_path, output_path)
    # return output_path

    raise NotImplementedError("Census crosswalk join not yet implemented")


def step3_analyze(input_path: Path) -> None:
    """
    Step 3: Run analysis and generate outputs.

    This step will:
    - Calculate usage patterns by demographics
    - Generate visualizations
    - Produce summary statistics
    - Export results for reporting
    """
    logger.info("=" * 80)
    logger.info("STEP 3: Run Analysis")
    logger.info("=" * 80)

    # TODO: Implement analysis
    # from smart_meter_analysis.analysis import run_analysis
    # run_analysis(input_path)

    raise NotImplementedError("Analysis not yet implemented")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end ComEd smart meter analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with local sample data (fast!)
  python run_comed_pipeline.py --source local

  # Test with 10 files from S3
  python run_comed_pipeline.py --year-month 202308 --max-files 10 --source s3

  # Process full month from S3
  python run_comed_pipeline.py --year-month 202308 --source s3

  # Skip download if data exists
  python run_comed_pipeline.py --year-month 202308 --skip-download --source s3

  # Enable debug logging
  python run_comed_pipeline.py --year-month 202308 --debug --source s3
        """,
    )

    parser.add_argument(
        "--source",
        choices=["s3", "local"],
        default="local",
        help="Data source: 's3' for production data, 'local' for sample files (default: local)",
    )
    parser.add_argument(
        "--year-month",
        required=False,
        help="Year-month to process (format: YYYYMM, e.g., 202308). Required for --source s3",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files to process (useful for testing with S3)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download/processing step if output file already exists",
    )
    parser.add_argument(
        "--crosswalk-path",
        type=Path,
        default=None,
        help="Path to Census crosswalk file (default: data/crosswalks/zip_to_tract.parquet)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop pipeline on first error (default: continue to show what works)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.source == "s3" and not args.year_month:
        parser.error("--year-month is required when --source is 's3'")

    # Update log level if debug requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 80)
    logger.info("ComEd Smart Meter Analysis Pipeline")
    logger.info("=" * 80)
    logger.info(f"Data Source: {args.source}")
    if args.source == "s3":
        logger.info(f"Year-Month: {args.year_month}")
        logger.info(f"Max Files: {args.max_files or 'All'}")
    logger.info(f"Skip Download: {args.skip_download}")
    logger.info("")

    try:
        # Step 1: Download/process data
        if args.source == "local":
            transformed_path = step1_process_local_samples(args.skip_download)
        else:  # s3
            transformed_path = step1_download_transform(
                args.year_month,
                args.max_files,
                args.skip_download,
            )
        logger.info("")

        # Step 2: Add census crosswalk
        try:
            enriched_path = step2_add_census_crosswalk(
                transformed_path,
                args.crosswalk_path,
            )
            logger.info("")

            # Step 3: Analyze
            step3_analyze(enriched_path)
            logger.info("")

        except (FileNotFoundError, NotImplementedError) as e:
            logger.warning(f"Step 2/3 blocked: {e}")
            if args.stop_on_error:
                raise
            logger.info("")
            logger.info("Pipeline completed Step 1 successfully!")
            logger.info("Steps 2-3 are blocked pending Census crosswalk.")
            logger.info("")
            logger.info("What you can do now:")
            logger.info("  1. Review the transformed data:")
            logger.info(f"     {transformed_path}")
            logger.info("  2. Inspect with polars:")
            logger.info(f"     just inspect-data {transformed_path}")
            logger.info("  3. Count rows:")
            logger.info(f"     just count-rows {transformed_path}")
            if args.source == "local":
                logger.info("  4. Try with real S3 data:")
                logger.info("     python run_comed_pipeline.py --year-month 202308 --max-files 10 --source s3")
            return 0

    except Exception:
        logger.exception("Pipeline failed")
        return 1

    logger.info("=" * 80)
    logger.info("✓ Pipeline completed successfully!")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
