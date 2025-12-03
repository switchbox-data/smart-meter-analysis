"""
Enrich ComEd smart meter data with census demographics at the interval level.

This program joins processed smart meter data with census demographics, creating
enriched parquet files for downstream analysis and aggregation.

This program:
1. Loads the ZIP+4 to Census Block crosswalk
2. Fetches/loads Census Block Group demographics via Census API (with local caching)
3. Joins crosswalk with census data to create geographic-demographic reference
4. Joins interval-level energy data with enriched reference on ZIP+4
5. Outputs enriched parquet with all original time features and census variables

Census data is automatically cached to avoid redundant API calls across runs.

Usage:
    # Initial run - fetch census data from API
    python scripts/add_census_data_comed.py 202308 \
        --crosswalk data/reference/2023_comed_zip4_census_crosswalk.txt \
        --fetch-census

    # Subsequent runs - use cached census data
    python scripts/add_census_data_comed.py 202309 \
        --crosswalk data/reference/zip4_census_crosswalk.txt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from smart_meter_analysis.census import fetch_census_data

logger = logging.getLogger(__name__)


def load_and_enrich_crosswalk(
    crosswalk_path: Path,
    census_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Load ZIP+4 crosswalk and join with census demographics.

    Args:
        crosswalk_path: Path to tab-delimited ZIP+4 crosswalk file
        census_df: Census demographics at Block Group level (12-digit GEOID)

    Returns:
        Enriched crosswalk mapping ZIP+4 to Block Group with demographics
    """
    logger.info(f"Loading crosswalk from {crosswalk_path}")

    crosswalk = pl.read_csv(
        crosswalk_path,
        separator="\t",
        infer_schema_length=10000,
    )

    logger.info(f"  Loaded {len(crosswalk):,} ZIP+4 mappings")

    # Standardize crosswalk columns
    crosswalk = crosswalk.with_columns([
        # Create ZIP+4 with hyphen to match ComEd format (e.g., "60002-1102")
        (pl.col("Zip").cast(pl.Utf8).str.zfill(5) + "-" + pl.col("Zip4").cast(pl.Utf8).str.zfill(4)).alias("zip4"),
        # Extract Block Group GEOID from Census Block (first 12 digits of 15-digit CensusKey2023)
        pl.col("CensusKey2023").cast(pl.Utf8).str.zfill(15).str.slice(0, 12).alias("block_group_geoid"),
    ])

    logger.info(f"  {crosswalk['zip4'].n_unique():,} unique ZIP+4 codes")
    logger.info(f"  {crosswalk['block_group_geoid'].n_unique():,} unique block groups")

    # Standardize census GEOID column
    census_df = census_df.with_columns(pl.col("GEOID").cast(pl.Utf8).str.zfill(12).alias("block_group_geoid"))

    # Join crosswalk with census demographics
    logger.info("Joining crosswalk with census demographics...")
    enriched = crosswalk.join(census_df, on="block_group_geoid", how="left")

    # Check demographic coverage
    matched = enriched.filter(pl.col("Total_Households").is_not_null())
    match_rate = (len(matched) / len(enriched)) * 100
    logger.info(f"  Demographic match rate: {match_rate:.1f}%")

    # Select final columns: zip4, block_group_geoid, and all census variables
    census_cols = [c for c in census_df.columns if c not in ["GEOID", "NAME", "block_group_geoid"]]
    cols_to_keep = ["zip4", "block_group_geoid", *census_cols]

    enriched = enriched.select([c for c in cols_to_keep if c in enriched.columns])

    logger.info(f"  Enriched crosswalk: {len(enriched):,} rows x {len(enriched.columns)} columns")

    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich ComEd interval data with census demographics")

    # Required arguments
    parser.add_argument(
        "year_month",
        help="Year-month to process (e.g., 202308)",
    )
    parser.add_argument(
        "--crosswalk",
        type=Path,
        required=True,
        help="Path to ZIP+4 crosswalk file (tab-separated)",
    )

    # Census data options
    census_group = parser.add_mutually_exclusive_group()
    census_group.add_argument(
        "--fetch-census",
        action="store_true",
        help="Fetch census data from API and cache locally",
    )
    census_group.add_argument(
        "--census-cache",
        type=Path,
        help="Path to cached census parquet file",
    )

    # Census API parameters
    parser.add_argument(
        "--state-fips",
        default="17",
        help="State FIPS code (default: 17 for Illinois)",
    )
    parser.add_argument(
        "--acs-year",
        type=int,
        default=2023,
        help="ACS year (default: 2023)",
    )

    # Input/output paths
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing processed energy parquet files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: data/analysis/{year_month}/enriched.parquet)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/reference"),
        help="Directory for census data cache",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Processing {args.year_month}")

    # Set output path
    if args.output is None:
        output_dir = Path("data/analysis") / args.year_month
        output_path = output_dir / "enriched.parquet"
    else:
        output_path = args.output
        output_dir = output_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Load or fetch census data
    # ========================================================================

    default_cache = args.cache_dir / f"census_{args.state_fips}_{args.acs_year}.parquet"

    if args.fetch_census:
        logger.info(f"Fetching census data from API (state={args.state_fips}, year={args.acs_year})")
        census_df = fetch_census_data(
            state_fips=args.state_fips,
            acs_year=args.acs_year,
        )
        census_df.write_parquet(default_cache)
        logger.info(f"Cached census data to {default_cache}")

    elif args.census_cache:
        logger.info(f"Loading census data from {args.census_cache}")
        census_df = pl.read_parquet(args.census_cache)

    elif default_cache.exists():
        logger.info(f"Loading cached census data from {default_cache}")
        census_df = pl.read_parquet(default_cache)

    else:
        logger.info(f"No cache found, fetching from API (state={args.state_fips}, year={args.acs_year})")
        census_df = fetch_census_data(
            state_fips=args.state_fips,
            acs_year=args.acs_year,
        )
        census_df.write_parquet(default_cache)
        logger.info(f"Cached census data to {default_cache}")

    logger.info(f"Census data: {len(census_df):,} block groups, {len(census_df.columns)} columns")

    # ========================================================================
    # Load crosswalk and join with census
    # ========================================================================

    enriched_crosswalk = load_and_enrich_crosswalk(args.crosswalk, census_df)

    demographic_cols = [c for c in enriched_crosswalk.columns if c not in ["zip4", "block_group_geoid"]]
    logger.info(f"Demographic variables: {len(demographic_cols)}")

    # ========================================================================
    # Load processed energy data
    # ========================================================================

    input_path = args.input_dir / f"comed_{args.year_month}.parquet"
    if not input_path.exists():
        logger.error(f"Processed data file not found: {input_path}")
        logger.error("Run run_comed_pipeline.py first to process raw S3 data")
        raise FileNotFoundError(f"Missing: {input_path}")

    logger.info(f"Loading energy data from {input_path}")
    energy_df = pl.read_parquet(input_path)
    logger.info(f"  {len(energy_df):,} interval records")
    logger.info(f"  {energy_df['zip_code'].n_unique():,} unique ZIP+4 codes")
    logger.info(f"  {energy_df['account_identifier'].n_unique():,} unique accounts")

    # ========================================================================
    # Join energy data with enriched crosswalk
    # ========================================================================

    logger.info("Joining energy data with enriched crosswalk...")
    enriched = energy_df.join(enriched_crosswalk, left_on="zip_code", right_on="zip4", how="left")

    # Calculate match statistics
    matched = enriched.filter(pl.col("block_group_geoid").is_not_null())
    match_rate = (len(matched) / len(enriched)) * 100
    logger.info(f"  Geographic match rate: {match_rate:.1f}% ({len(matched):,} / {len(enriched):,} records)")

    if match_rate < 95:
        unmatched_zips = enriched.filter(pl.col("block_group_geoid").is_null()).select("zip_code").unique()
        logger.warning(f"  {len(unmatched_zips):,} ZIP+4 codes without block group match")

    logger.info(f"Enriched data: {len(enriched):,} rows x {len(enriched.columns)} columns")

    # ========================================================================
    # Save enriched interval-level data
    # ========================================================================

    logger.info(f"Writing enriched data to {output_path}")
    enriched.write_parquet(output_path)
    logger.info(f"  Saved: {enriched.shape[0]:,} rows x {enriched.shape[1]} columns")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 80)
    print(f"ENRICHMENT COMPLETE: {args.year_month}")
    print("=" * 80)

    print("\nInput:")
    print(f"  Energy data: {input_path}")
    print(f"  Crosswalk: {args.crosswalk}")
    print(f"  Census data: {args.census_cache if args.census_cache else default_cache}")

    print("\nEnrichment statistics:")
    print(f"  Geographic match rate: {match_rate:.1f}%")
    print(f"  Interval records: {len(enriched):,}")
    print(f"  Block groups: {enriched['block_group_geoid'].n_unique():,}")
    print(f"  Demographic variables: {len(demographic_cols)}")

    print("\nOutput:")
    print(f"  {output_path}")
    print("  Interval-level data with time features and demographics preserved")

    print("\n" + "=" * 80)

    logger.info(f"✅ Enrichment complete for {args.year_month}")


if __name__ == "__main__":
    main()
