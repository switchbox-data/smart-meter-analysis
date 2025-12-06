# smart_meter_analysis/census.py
"""
Census data fetcher for demographic analysis.
Fetches ACS 5-year (detailed tables) and Decennial Census data at Block Group level.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cenpy as cen  # type: ignore[import-untyped]
import polars as pl
import requests

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
    Fetch ACS 5-Year *detailed table* data at Block Group level from the
    official Census API.

    Args:
        state_fips: State FIPS code (default: '17' for Illinois)
        year: ACS 5-year endpoint year (default: 2023)

    Returns:
        DataFrame with ACS demographics by block group.
    """
    logger.info(f"Fetching ACS {year} detailed-table data for state {state_fips}")

    acs_vars: dict[str, str] = {
        # Population Demographics
        "B01002_001E": "Median_Age",
        "B01003_001E": "Total_Population",
        "B02001_002E": "White_Alone_Population",
        "B02001_003E": "Black_Alone_Population",
        "B03001_003E": "Hispanic_or_Latino",
        "B15003_001E": "Educational_Attainment_25_Plus",
        "B17001_001E": "Poverty_Status",
        "B01001_002E": "Male_Total",
        "B01001_003E": "Male_Under_5",
        "B01001_004E": "Male_5_to_9",
        "B01001_005E": "Male_10_to_14",
        "B01001_006E": "Male_15_to_17",
        "B01001_007E": "Male_18_to_19",
        "B01001_008E": "Male_20",
        "B01001_009E": "Male_21",
        "B01001_010E": "Male_22_to_24",
        "B01001_011E": "Male_25_to_29",
        "B01001_012E": "Male_30_to_34",
        "B01001_013E": "Male_35_to_39",
        "B01001_014E": "Male_40_to_44",
        "B01001_015E": "Male_45_to_49",
        "B01001_016E": "Male_50_to_54",
        "B01001_017E": "Male_55_to_59",
        "B01001_018E": "Male_60_to_61",
        "B01001_019E": "Male_62_to_64",
        "B01001_020E": "Male_65_to_66",
        "B01001_021E": "Male_67_to_69",
        "B01001_022E": "Male_70_to_74",
        "B01001_023E": "Male_75_to_79",
        "B01001_024E": "Male_80_to_84",
        "B01001_025E": "Male_85_plus",
        "B01001_026E": "Female_Total",
        "B01001_027E": "Female_Under_5",
        "B01001_028E": "Female_5_to_9",
        "B01001_029E": "Female_10_to_14",
        "B01001_030E": "Female_15_to_17",
        "B01001_031E": "Female_18_to_19",
        "B01001_032E": "Female_20",
        "B01001_033E": "Female_21",
        "B01001_034E": "Female_22_to_24",
        "B01001_035E": "Female_25_to_29",
        "B01001_036E": "Female_30_to_34",
        "B01001_037E": "Female_35_to_39",
        "B01001_038E": "Female_40_to_44",
        "B01001_039E": "Female_45_to_49",
        "B01001_040E": "Female_50_to_54",
        "B01001_041E": "Female_55_to_59",
        "B01001_042E": "Female_60_to_61",
        "B01001_043E": "Female_62_to_64",
        "B01001_044E": "Female_65_to_66",
        "B01001_045E": "Female_67_to_69",
        "B01001_046E": "Female_70_to_74",
        "B01001_047E": "Female_75_to_79",
        "B01001_048E": "Female_80_to_84",
        "B01001_049E": "Female_85_plus",
        # Households & income
        "B11001_001E": "Total_Households",
        "B25010_001E": "Average_Household_Size",
        "B11016_002E": "Average_Family_Size",
        "B19013_001E": "Median_Household_Income",
        "B19001_002E": "HH_Income_Less_10k",
        "B19001_003E": "HH_Income_10k_to_14.999k",
        "B19001_004E": "HH_Income_15k_to_19.999k",
        "B19001_005E": "HH_Income_20k_to_24.999k",
        "B19001_006E": "HH_Income_25k_to_29.999k",
        "B19001_007E": "HH_Income_30k_to_34.999k",
        "B19001_008E": "HH_Income_35k_to_39.999k",
        "B19001_009E": "HH_Income_40k_to_44.999k",
        "B19001_010E": "HH_Income_45k_to_49.999k",
        "B19001_011E": "HH_Income_50k_to_59.999k",
        "B19001_012E": "HH_Income_60k_to_74.999k",
        "B19001_013E": "HH_Income_75k_to_99.999k",
        "B19001_014E": "HH_Income_100k_to_124.999k",
        "B19001_015E": "HH_Income_125k_to_149.999k",
        "B19001_016E": "HH_Income_150k_to_199.999k",
        "B19001_017E": "HH_Income_More_200k",
        # Housing units & tenure
        "B25001_001E": "Total_Housing_Units",
        "B25003_001E": "Occupied_Housing_Units",
        "B25003_002E": "Owner_Occupied",
        "B25003_003E": "Renter_Occupied",
        # Property value
        "B25075_002E": "Value_less_$10k",
        "B25075_003E": "Value_$10k_to_$14.999k",
        "B25075_004E": "Value_$15k_to_$19.999k",
        "B25075_005E": "Value_$20k_to_$24.999k",
        "B25075_006E": "Value_$25k_to_$29.999k",
        "B25075_007E": "Value_$30k_to_$34.999k",
        "B25075_008E": "Value_$35k_to_$39.999k",
        "B25075_009E": "Value_$40k_to_$49.999k",
        "B25075_010E": "Value_$50k_to_$59.999k",
        "B25075_011E": "Value_$60k_to_$69.999k",
        "B25075_012E": "Value_$70_to_$79.999k",
        "B25075_013E": "Value_$80_to_$89.999k",
        "B25075_014E": "Value_$90k_to_$99.999k",
        "B25075_015E": "Value_$100k_to_$124.999k",
        "B25075_016E": "Value_$125k_to_$149.999k",
        "B25075_017E": "Value_$150k_to_$174.999k",
        "B25075_018E": "Value_$175k_to_$199.999k",
        "B25075_019E": "Value_$200k_to_$249.999k",
        "B25075_020E": "Value_$250k_to_$299.999k",
        "B25075_021E": "Value_$300k_to_$399.999k",
        "B25075_022E": "Value_$400k_to_$499.999k",
        "B25075_023E": "Value_$500k_to_$749.999k",
        "B25075_024E": "Value_$750k_to_$999.999k",
        "B25075_025E": "Value_$1m_to_$1.49m",
        "B25075_026E": "Value_$1.5m_to_$1.9m",
        "B25075_027E": "Value_more_$2m",
        # Year structure built (B25034)
        "B25034_002E": "Built_2020_After",
        "B25034_003E": "Built_2010_2019",
        "B25034_004E": "Built_2000_2009",
        "B25034_005E": "Built_1990_1999",
        "B25034_006E": "Built_1980_1989",
        "B25034_007E": "Built_1970_1979",
        "B25034_008E": "Built_1960_1969",
        "B25034_009E": "Built_1950_1959",
        "B25034_010E": "Built_1940_1949",
        "B25034_011E": "Built_1939_Earlier",
        # Heating fuel
        "B25040_002E": "Heat_Utility_Gas",
        "B25040_004E": "Heat_Electric",
    }

    # ------------------------------------------------------------------
    # Call Census API directly (acs/acs5)
    # ------------------------------------------------------------------
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"

    params: dict[str, str] = {
        "get": ",".join(["NAME", *acs_vars.keys()]),
        "for": "block group:*",
        "in": f"state:{state_fips} county:*",
    }

    api_key = os.getenv("CENSUS_API_KEY")
    if api_key:
        params["key"] = api_key

    resp = requests.get(base_url, params=params, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    if not data or len(data) < 2:
        raise RuntimeError("ACS API returned no data")

    header = data[0]
    rows = data[1:]

    # Build a Polars DataFrame directly from the raw rows.
    acs_df = pl.DataFrame(rows, schema=header)

    logger.info(f"Retrieved {acs_df.height} block groups from ACS (detailed tables)")

    # GEOID + friendly names
    acs_df = build_geoid(acs_df)

    # Rename B-table columns to friendlier names
    rename_map = {var: name for var, name in acs_vars.items() if var in acs_df.columns}
    acs_df = acs_df.rename(rename_map)

    # Cast numeric columns where possible (everything except identifiers)
    non_numeric = {"NAME", "GEOID", "state", "county", "tract", "block group"}
    numeric_cols = [c for c in acs_df.columns if c not in non_numeric]

    acs_df = acs_df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols])

    # Clean special missing codes
    acs_df = clean_census_values(acs_df)

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

    # DHC 2020 = /data/2020/dec/dhc
    conn_dhc = cen.remote.APIConnection(f"DECENNIALDHC{year}")

    # H2 = URBAN AND RURAL table in the DHC detailed tables
    dhc_vars = {
        "H2_002N": "Urban_Housing_Units",  # !!Total:!!Urban
        "H2_003N": "Rural_Housing_Units",  # !!Total:!!Rural
    }

    dhc_df = conn_dhc.query(
        cols=["NAME", *list(dhc_vars.keys())],
        geo_unit="block group:*",
        geo_filter={"state": state_fips, "county": "*", "tract": "*"},
    )

    logger.info(f"Retrieved {len(dhc_df)} block groups from Decennial Census (DHC H2 urban/rural)")

    # Convert to Polars
    dhc_df = pl.from_pandas(dhc_df)

    # Build standard 12-digit GEOID from state / county / tract / block group
    dhc_df = build_geoid(dhc_df)

    # Rename to friendlier column names if present
    dhc_df = dhc_df.rename({k: v for k, v in dhc_vars.items() if k in dhc_df.columns})

    # Ensure numeric types
    dhc_df = dhc_df.with_columns([
        pl.col("Urban_Housing_Units").cast(pl.Float64, strict=False),
        pl.col("Rural_Housing_Units").cast(pl.Float64, strict=False),
    ])

    # Calculate urban share and a simple classification
    dhc_df = dhc_df.with_columns([
        safe_percent(
            pl.col("Urban_Housing_Units"),
            pl.col("Urban_Housing_Units") + pl.col("Rural_Housing_Units"),
        ).alias("Urban_Percent"),
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
        acs_year: ACS year (default: 2023, 5-year ACS detailed tables)
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
        dhc_df.select(["GEOID", "Urban_Percent", "Urban_Rural_Classification"]),
        on="GEOID",
        how="left",
    )

    logger.info(f"Combined dataset: {census_df.height} block groups")

    # Convert string numerics to float (just in case anything slipped through)
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
