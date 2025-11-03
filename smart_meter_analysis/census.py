# smart_meter_analysis/census.py
"""
Census data fetcher for demographic analysis.
Fetches ACS 5-year and Decennial Census data at Block Group level.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cenpy as cen  # type: ignore[import-untyped]
import polars as pl

logger = logging.getLogger(__name__)

# Census uses special codes for missing/suppressed data instead of nulls
CENSUS_MISSING_CODES = [-666666666, -999999999, -888888888, -555555555, -222222222]


def build_geoid(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build 12-digit Block Group GEOID from components.

    GEOID format: State(2) + County(3) + Tract(6) + BlockGroup(1) = 12 digits
    """
    return df.with_columns(
        pl.concat_str([
            pl.col("state").cast(pl.Utf8).str.zfill(2),
            pl.col("county").cast(pl.Utf8).str.zfill(3),
            pl.col("tract").cast(pl.Utf8).str.zfill(6),
            pl.col("block group").cast(pl.Utf8).str.zfill(1),
        ]).alias("GEOID")
    )


def safe_percent(numer: pl.Expr, denom: pl.Expr) -> pl.Expr:
    """Calculate percentage safely, handling division by zero."""
    return (
        pl.when(denom.is_not_null() & numer.is_not_null() & (denom > 0)).then((numer / denom) * 100.0).otherwise(None)
    )


def clean_census_values(df: pl.DataFrame) -> pl.DataFrame:
    """Replace Census missing codes with null values."""
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    return df.with_columns([
        pl.when(pl.col(col).is_in(CENSUS_MISSING_CODES)).then(None).otherwise(pl.col(col)).alias(col)
        for col in numeric_cols
    ])


def fetch_acs_data(state_fips: str = "17", year: int = 2023) -> pl.DataFrame:
    """
    Fetch ACS 5-Year data at Block Group level.

    Args:
        state_fips: State FIPS code (default: '17' for Illinois)
        year: ACS year (default: 2023)

    Returns:
        DataFrame with ACS demographics by block group
    """
    logger.info(f"Fetching ACS {year} data for state {state_fips}")

    # Connect to ACS API
    conn_acs = cen.remote.APIConnection(f"ACSDP5Y{year}")

    # Variable mappings
    acs_vars = {
        "DP02_0001E": "Total_Households",
        "DP02_0016E": "Avg_Household_Size",
        "DP02_0017E": "Avg_Family_Size",
        "DP02_0060E": "Less_9th_Grade",
        "DP02_0062E": "High_School_Dip",
        "DP02_0063E": "Some_College",
        "DP02_0064E": "Associates_Deg",
        "DP02_0065E": "Bachelors_Deg",
        "DP02_0066E": "Grad_Professional_Deg",
        "DP03_0052E": "Income_Less_10k",
        "DP03_0053E": "Income_10k_to_15k",
        "DP03_0054E": "Income_15k_to_25k",
        "DP03_0055E": "Income_25k_to_35k",
        "DP03_0056E": "Income_35k_to_50k",
        "DP03_0057E": "Income_50k_to_75k",
        "DP03_0058E": "Income_75k_to_100k",
        "DP03_0059E": "Income_100k_to_150k",
        "DP03_0060E": "Income_150k_to_200k",
        "DP03_0061E": "Income_200k_Plus",
        "DP03_0062E": "Median_Household_Income",
        "DP04_0001E": "Total_Housing_Units",
        "DP04_0002E": "Occupied_Housing_Units",
        "DP04_0003E": "Vacant_Housing_Units",
        "DP04_0017E": "Built_2020_After",
        "DP04_0018E": "Built_2010_2019",
        "DP04_0019E": "Built_2000_2009",
        "DP04_0020E": "Built_1990_1999",
        "DP04_0021E": "Built_1980_1989",
        "DP04_0022E": "Built_1970_1979",
        "DP04_0023E": "Built_1960_1969",
        "DP04_0024E": "Built_1950_1959",
        "DP04_0025E": "Built_1940_1949",
        "DP04_0026E": "Built_Before_1940",
        "DP04_0028E": "Rooms_1",
        "DP04_0029E": "Rooms_2",
        "DP04_0030E": "Rooms_3",
        "DP04_0031E": "Rooms_4",
        "DP04_0032E": "Rooms_5",
        "DP04_0033E": "Rooms_6",
        "DP04_0034E": "Rooms_7",
        "DP04_0035E": "Rooms_8",
        "DP04_0036E": "Rooms_9_Plus",
        "DP04_0046E": "Owner_Occupied",
        "DP04_0047E": "Renter_Occupied",
        "DP04_0063E": "Heat_Utility_Gas",
        "DP04_0065E": "Heat_Electric",
        "DP05_0006E": "Age_5_to_9",
        "DP05_0007E": "Age_10_to_14",
        "DP05_0008E": "Age_15_to_19",
        "DP05_0009E": "Age_20_to_24",
        "DP05_0010E": "Age_25_to_34",
        "DP05_0011E": "Age_35_to_44",
        "DP05_0012E": "Age_45_to_54",
        "DP05_0013E": "Age_55_to_59",
        "DP05_0014E": "Age_60_to_64",
        "DP05_0015E": "Age_65_to_74",
        "DP05_0016E": "Age_75_to_84",
        "DP05_0017E": "Age_85_Plus",
        "DP04_0081E": "Value_Less_50k",
        "DP04_0082E": "Value_50k_to_100k",
        "DP04_0083E": "Value_100k_to_150k",
        "DP04_0084E": "Value_150k_to_200k",
        "DP04_0085E": "Value_200k_to_300k",
        "DP04_0086E": "Value_300k_to_500k",
        "DP04_0087E": "Value_500k_to_1M",
        "DP04_0088E": "Value_1M_Plus",
    }

    # Query ACS data at BLOCK GROUP level
    acs_df = conn_acs.query(
        cols=["NAME", *list(acs_vars.keys())], geo_unit="block group", geo_filter={"state": state_fips}
    )

    logger.info(f"Retrieved {len(acs_df)} block groups from ACS")

    # Convert to Polars
    acs_df = pl.from_pandas(acs_df)
    acs_df = build_geoid(acs_df)
    acs_df = acs_df.rename({k: v for k, v in acs_vars.items() if k in acs_df.columns})

    return acs_df


def fetch_decennial_data(state_fips: str = "17", year: int = 2020) -> pl.DataFrame:
    """
    Fetch Decennial Census urban/rural data at Block Group level.

    Args:
        state_fips: State FIPS code (default: '17' for Illinois)
        year: Census year (default: 2020)

    Returns:
        DataFrame with urban/rural classification by block group
    """
    logger.info(f"Fetching Decennial {year} data for state {state_fips}")

    conn_dhc = cen.remote.APIConnection(f"DECENNIALDHC{year}")

    dhc_vars = {"H2_002N": "Urban_Housing_Units", "H2_003N": "Rural_Housing_Units"}

    # Query at BLOCK GROUP level
    dhc_df = conn_dhc.query(
        cols=["NAME", *list(dhc_vars.keys())], geo_unit="block group", geo_filter={"state": state_fips}
    )

    logger.info(f"Retrieved {len(dhc_df)} block groups from Decennial Census")

    # Convert to Polars
    dhc_df = pl.from_pandas(dhc_df)
    dhc_df = build_geoid(dhc_df)
    dhc_df = dhc_df.rename({k: v for k, v in dhc_vars.items() if k in dhc_df.columns})

    # Calculate urban/rural metrics
    dhc_df = dhc_df.with_columns([
        pl.col("Urban_Housing_Units").cast(pl.Float64),
        pl.col("Rural_Housing_Units").cast(pl.Float64),
    ]).with_columns([
        safe_percent(
            pl.col("Urban_Housing_Units"), pl.col("Urban_Housing_Units") + pl.col("Rural_Housing_Units")
        ).alias("Urban_Percent"),
        # Classify based on majority
        pl.when((pl.col("Urban_Housing_Units") + pl.col("Rural_Housing_Units")) == 0)
        .then(pl.lit("No Housing"))
        .when(pl.col("Urban_Housing_Units") > pl.col("Rural_Housing_Units"))
        .then(pl.lit("Urban"))
        .otherwise(pl.lit("Rural"))
        .alias("Urban_Rural_Classification"),
    ])

    return dhc_df


def fetch_census_data(
    state_fips: str = "17",
    acs_year: int = 2023,
    decennial_year: int = 2020,
    output_path: Path | str | None = None,
) -> pl.DataFrame:
    """
    Fetch and combine ACS and Decennial Census data at Block Group level.

    Args:
        state_fips: State FIPS code (default: '17' for Illinois)
        acs_year: ACS year (default: 2023)
        decennial_year: Decennial Census year (default: 2020)
        output_path: Optional path to save combined data as Parquet

    Returns:
        Combined DataFrame with all census variables by block group
    """
    # Fetch both datasets
    acs_df = fetch_acs_data(state_fips, acs_year)
    dhc_df = fetch_decennial_data(state_fips, decennial_year)

    # Merge
    census_df = acs_df.join(
        dhc_df.select(["GEOID", "Urban_Percent", "Urban_Rural_Classification"]), on="GEOID", how="left"
    )

    logger.info(f"Combined dataset: {census_df.height} block groups")

    # Convert string numerics to float
    numeric_cols = [col for col in census_df.columns if col not in ["NAME", "GEOID", "Urban_Rural_Classification"]]

    census_df = census_df.with_columns([
        pl.col(col).cast(pl.Float64, strict=False) for col in numeric_cols if census_df[col].dtype == pl.Utf8
    ])

    # Clean missing codes
    census_df = clean_census_values(census_df)

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        census_df.write_parquet(output_path)
        logger.info(f"Saved census data to {output_path}")

    return census_df


def validate_census_data(df: pl.DataFrame) -> dict:
    """
    Validate census data quality.

    Returns:
        Dict with validation metrics
    """
    return {
        "total_block_groups": df.height,
        "unique_geoids": df["GEOID"].n_unique(),
        "columns": len(df.columns),
        "null_counts": {col: int(df[col].null_count()) for col in df.columns if df[col].null_count() > 0},
        "geoid_length_correct": all(df["GEOID"].str.len_chars() == 12),
    }
