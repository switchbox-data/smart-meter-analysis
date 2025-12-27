# smart_meter_analysis/census.py
"""Census data fetcher for demographic analysis.

Fetches:
- ACS 5-year (detailed tables) data at Block Group level via the official Census API
- Decennial Census 2020 DHC H2 table at Block Group level via cenpy (urban/rural housing units)

Design:
- VARIABLE_SPECS defines engineered features (percentages, sums, logs).
- This module fetches raw ACS codes required by VARIABLE_SPECS, computes engineered features,
  then joins Decennial-derived urban_percent.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import cenpy as cen  # type: ignore[import-untyped]
import polars as pl
import requests

from smart_meter_analysis.census_specs import VARIABLE_SPECS

logger = logging.getLogger(__name__)

# Census uses special codes for missing/suppressed data instead of nulls
CENSUS_MISSING_CODES = [-666666666, -999999999, -888888888, -555555555, -222222222]

# Minimal raw-code extractor for ACS detailed-table variables (e.g., B25040_002E)
# Note: Decennial H2_* variables are handled in fetch_decennial_data (cenpy), not via ACS API.
_VAR_RE = re.compile(r"\b[BCH]\d{5}_\d{3}E\b")


def chunk_list(items: list[Any], size: int) -> list[list[Any]]:
    """Split a list into chunks of specified size."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_geoid(df: pl.DataFrame) -> pl.DataFrame:
    """Build 12-digit Block Group GEOID from components.

    GEOID format: State(2) + County(3) + Tract(6) + BlockGroup(1) = 12 digits
    """
    return df.with_columns(
        pl.concat_str([
            pl.col("state").cast(pl.Utf8).str.zfill(2),
            pl.col("county").cast(pl.Utf8).str.zfill(3),
            pl.col("tract").cast(pl.Utf8).str.zfill(6),
            pl.col("block group").cast(pl.Utf8).str.zfill(1),
        ]).alias("GEOID"),
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


# -----------------------------------------------------------------------------
# Spec-driven helpers
# -----------------------------------------------------------------------------
def extract_raw_codes(expr: str | None) -> set[str]:
    """Extract ACS raw variable codes (e.g., B25040_002E) from a spec string.

    Supports expressions like:
      "B25070_007E + B25070_008E"
    """
    if not expr:
        return set()
    return set(_VAR_RE.findall(expr))


def gather_acs_codes(variable_specs: list[dict[str, Any]]) -> list[str]:
    """Gather the full set of ACS variable codes required to compute VARIABLE_SPECS.

    We parse:
    - spec["numerator"]
    - spec["denominator"]
    - spec["variable"]
    """
    codes: set[str] = set()
    for spec in variable_specs:
        codes |= extract_raw_codes(spec.get("numerator"))
        codes |= extract_raw_codes(spec.get("denominator"))
        codes |= extract_raw_codes(spec.get("variable"))
    return sorted(codes)


def expr_to_polars(expr: str) -> pl.Expr:
    """Convert a simple linear expression into a Polars expression.

    Supported grammar:
      - "A"
      - "A + B + C"  (only "+" supported; keep minimal for robustness)

    Notes:
      - Inputs are expected to be ACS codes present in the DataFrame.
      - Each term is cast to Float64 to avoid integer division issues.

    """
    parts = [p.strip() for p in expr.split("+")]
    out = pl.lit(0.0)
    for p in parts:
        if not p:
            continue
        out = out + pl.col(p).cast(pl.Float64, strict=False)
    return out


def is_acs_only_spec(spec: dict[str, Any]) -> bool:
    """Check if a spec is ACS-only (excludes Decennial H2_*N variables).

    Returns True if the spec does not reference any H2_*N variables in its
    numerator or denominator expressions.
    """
    numer = spec.get("numerator", "")
    denom = spec.get("denominator", "")

    # Check if numerator or denominator contain H2_*N variables
    has_h2 = "H2_" in str(numer) or "H2_" in str(denom)

    return not has_h2


def build_feature_columns(variable_specs: list[dict[str, Any]]) -> list[pl.Expr]:
    """Build Polars expressions for engineered features defined by VARIABLE_SPECS.

    - If "variable" exists: use it directly (single ACS code)
    - If denominator exists: compute percent = numerator / denominator * 100 (safe_percent).
    - If denominator is None: use numerator directly.
    - transformation:
        - "none": no transformation
        - "log": apply log(value + 1) to support zeros (null if value < 0)
    """
    cols: list[pl.Expr] = []

    for spec in variable_specs:
        name = spec["name"]
        variable = spec.get("variable")  # Single variable (no ratio)
        numer = spec.get("numerator")
        denom = spec.get("denominator")
        transform = spec.get("transformation", "none")

        # Determine the base value expression
        if variable:
            # Single variable (e.g., median_household_income)
            value = expr_to_polars(variable)
        elif numer:
            # Ratio (e.g., unemployment_rate, pct_owner_occupied)
            numer_expr = expr_to_polars(numer)
            if denom:
                denom_expr = expr_to_polars(denom)
                value = safe_percent(numer_expr, denom_expr)
            else:
                value = numer_expr
        else:
            raise ValueError(f"Spec '{name}' must have either 'variable' or 'numerator'")

        # Apply transformation
        if transform == "log":
            value = pl.when(value.is_not_null() & (value >= 0)).then((value + 1.0).log()).otherwise(None)
        elif transform == "none":
            pass
        else:
            raise ValueError(f"Unknown transformation '{transform}' for spec '{name}'")

        cols.append(value.alias(name))

    return cols


# -----------------------------------------------------------------------------
# Fetchers
# -----------------------------------------------------------------------------
def fetch_acs_data(
    state_fips: str = "17",
    year: int = 2023,
    county_fips: str | None = None,
    keep_raw_debug_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Fetch ACS 5-Year *detailed table* raw variables needed for VARIABLE_SPECS at Block Group level,
    then compute engineered features.

    Args:
        state_fips: State FIPS code (default '17' for Illinois)
        year: ACS 5-year endpoint year (default 2023)
        county_fips: Optional County FIPS to limit the pull for quick testing (e.g., '031' for Cook).
                     If None, pulls all counties in the state.
        keep_raw_debug_cols: Optional list of raw ACS variable column names to retain in output
                             for debugging (e.g., ['B17001_001E', 'B17001_002E']).
                             Only columns that exist in the DataFrame will be kept.

    Returns:
        DataFrame with:
          - GEOID, NAME
          - engineered ACS features defined by VARIABLE_SPECS (ACS-only specs)
          - optionally, raw ACS variables specified in keep_raw_debug_cols

    """
    suffix = f", county {county_fips}" if county_fips else ""
    logger.info(f"Fetching ACS {year} detailed-table data for state {state_fips}{suffix}")

    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    # Filter to ACS-only specs (exclude Decennial H2_*N variables) before gathering codes
    acs_specs = [s for s in VARIABLE_SPECS if is_acs_only_spec(s)]
    raw_codes = gather_acs_codes(acs_specs)

    # Chunk API calls to avoid URL length limits (chunk size 45 variables per request)
    chunk_size = 45
    code_chunks = chunk_list(raw_codes, chunk_size)
    logger.info(f"Fetching {len(raw_codes)} variables in {len(code_chunks)} chunks (chunk size: {chunk_size})")

    in_clause = f"state:{state_fips} county:{county_fips}" if county_fips else f"state:{state_fips} county:*"
    api_key = os.getenv("CENSUS_API_KEY")

    chunk_dfs: list[pl.DataFrame] = []
    join_keys = ["state", "county", "tract", "block group", "GEOID"]

    for i, code_chunk in enumerate(code_chunks):
        logger.info(f"  Chunk {i + 1}/{len(code_chunks)}: {len(code_chunk)} variables")

        # First chunk includes NAME, subsequent chunks don't need it
        get_vars = ["NAME", *code_chunk] if i == 0 else code_chunk

        params: dict[str, str] = {
            "get": ",".join(get_vars),
            "for": "block group:*",
            "in": in_clause,
        }

        if api_key:
            params["key"] = api_key

        try:
            resp = requests.get(base_url, params=params, timeout=120)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # resp exists even if raise_for_status() failed
            status_code = resp.status_code
            response_body = resp.text[:500] if resp.text else str(e)
            raise RuntimeError(
                f"ACS API chunk {i + 1}/{len(code_chunks)} failed with HTTP {status_code}.\n"
                f"Response body (first 500 chars): {response_body}",
            ) from e
        except Exception as e:
            raise RuntimeError(f"ACS API chunk {i + 1}/{len(code_chunks)} failed with error: {e}") from e

        data = resp.json()
        if not data or len(data) < 2:
            raise RuntimeError(f"ACS API chunk {i + 1}/{len(code_chunks)} returned no data")

        header = data[0]
        rows = data[1:]

        # Debug: Check if requested variables are in the response
        missing_in_response = [v for v in code_chunk if v not in header]
        if missing_in_response:
            logger.warning(
                f"  Chunk {i + 1}: Variables requested but not in API response: {missing_in_response[:5]}"
                + (f" (and {len(missing_in_response) - 5} more)" if len(missing_in_response) > 5 else ""),
            )

        chunk_df = pl.DataFrame(rows, schema=header, orient="row")

        # Build GEOID for this chunk
        chunk_df = build_geoid(chunk_df)

        # Drop NAME from chunks after the first (we only need it once)
        if i > 0 and "NAME" in chunk_df.columns:
            chunk_df = chunk_df.drop("NAME")

        chunk_dfs.append(chunk_df)

    # Join all chunks on geographic identifiers
    logger.info(f"Joining {len(chunk_dfs)} chunks...")
    df = chunk_dfs[0]
    for chunk_df in chunk_dfs[1:]:
        df = df.join(chunk_df, on=join_keys, how="inner")

    logger.info(f"Retrieved {df.height} block groups from ACS raw pull")

    # Debug: Check if required variables are present
    required_vars = ["B17001_001E", "B17001_002E", "B22001_001E", "B22001_002E", "B03001_001E", "B03001_003E"]
    missing_vars = [v for v in required_vars if v not in df.columns]
    if missing_vars:
        logger.warning(f"Required variables not found in API response: {missing_vars}")
    else:
        logger.info("All required variables present in raw data")

    # Cast numeric columns where possible (everything except identifiers)
    non_numeric = {"NAME", "GEOID", "state", "county", "tract", "block group"}
    numeric_cols = [c for c in df.columns if c not in non_numeric]
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols])

    # Clean special missing codes
    df = clean_census_values(df)

    # Compute engineered ACS features (using same acs_specs filtered earlier)
    df = df.with_columns(build_feature_columns(acs_specs))

    # Keep only identifiers + ACS-engineered features (exclude Decennial-derived specs)
    keep = ["GEOID", "NAME"] + [s["name"] for s in acs_specs]

    # Optionally add debug raw columns if requested
    if keep_raw_debug_cols is not None:
        # Only add columns that actually exist in the DataFrame
        existing_debug_cols = [c for c in keep_raw_debug_cols if c in df.columns]
        if existing_debug_cols:
            keep.extend(existing_debug_cols)
            logger.info(f"Retaining {len(existing_debug_cols)} raw debug columns: {existing_debug_cols}")
        missing_debug_cols = [c for c in keep_raw_debug_cols if c not in df.columns]
        if missing_debug_cols:
            logger.warning(f"Debug columns not found in DataFrame: {missing_debug_cols}")

    df = df.select([c for c in keep if c in df.columns])

    return df


def fetch_decennial_data(state_fips: str = "17", year: int = 2020) -> pl.DataFrame:
    """Fetch Decennial Census DHC H2 (Urban and Rural) at Block Group level, then compute urban_percent.

    Args:
        state_fips: State FIPS code (default '17' for Illinois)
        year: Census year (default 2020)

    Returns:
        DataFrame with:
          - GEOID
          - Urban_Housing_Units
          - Rural_Housing_Units
          - urban_percent

    """
    logger.info(f"Fetching Decennial {year} DHC H2 data for state {state_fips}")

    # DHC 2020 = DECENNIALDHC2020
    conn_dhc = cen.remote.APIConnection(f"DECENNIALDHC{year}")

    dhc_vars = {
        "H2_002N": "Urban_Housing_Units",  # Total: Urban
        "H2_003N": "Rural_Housing_Units",  # Total: Rural
    }

    dhc_df = conn_dhc.query(
        cols=["NAME", *list(dhc_vars.keys())],
        geo_unit="block group:*",
        geo_filter={"state": state_fips, "county": "*", "tract": "*"},
    )

    logger.info(f"Retrieved {len(dhc_df)} block groups from Decennial Census (DHC H2)")

    # Convert to Polars
    df = pl.from_pandas(dhc_df)

    # Build standard 12-digit GEOID from state / county / tract / block group
    df = build_geoid(df)

    # Rename to friendlier column names
    df = df.rename({k: v for k, v in dhc_vars.items() if k in df.columns})

    # Ensure numeric types
    df = df.with_columns([
        pl.col("Urban_Housing_Units").cast(pl.Float64, strict=False),
        pl.col("Rural_Housing_Units").cast(pl.Float64, strict=False),
    ])

    # Compute urban_percent
    df = df.with_columns([
        safe_percent(
            pl.col("Urban_Housing_Units"),
            pl.col("Urban_Housing_Units") + pl.col("Rural_Housing_Units"),
        ).alias("urban_percent"),
    ])

    return df.select(["GEOID", "Urban_Housing_Units", "Rural_Housing_Units", "urban_percent"])


def fetch_census_data(
    state_fips: str = "17",
    acs_year: int = 2023,
    decennial_year: int = 2020,
    county_fips: str | None = None,
    output_path: Path | str | None = None,
    keep_raw_debug_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Fetch and combine ACS (engineered features) and Decennial (urban_percent) at Block Group level.

    Args:
        state_fips: State FIPS code (default: '17' for Illinois)
        acs_year: ACS year (default: 2023)
        decennial_year: Decennial year (default: 2020)
        county_fips: Optional county filter applied ONLY to ACS pull for fast testing.
                     Decennial is still statewide and joined by GEOID (safe).
        output_path: Optional path to save combined data as Parquet.
        keep_raw_debug_cols: Optional list of raw ACS variable column names to retain in output
                            for debugging (e.g., ['B17001_001E', 'B17001_002E']).
                            Only columns that exist in the DataFrame will be kept.

    Returns:
        Combined DataFrame with all engineered ACS features + urban_percent by block group.
        Optionally includes raw ACS variables specified in keep_raw_debug_cols.

    """
    acs_df = fetch_acs_data(
        state_fips=state_fips,
        year=acs_year,
        county_fips=county_fips,
        keep_raw_debug_cols=keep_raw_debug_cols,
    )
    dhc_df = fetch_decennial_data(state_fips=state_fips, year=decennial_year)

    census_df = acs_df.join(
        dhc_df.select(["GEOID", "urban_percent"]),
        on="GEOID",
        how="left",
    )

    logger.info(f"Combined dataset: {census_df.height} block groups (ACS filtered={bool(county_fips)})")

    # Final clean missing codes (defensive)
    census_df = clean_census_values(census_df)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        census_df.write_parquet(output_path)
        logger.info(f"Saved census data to {output_path}")

    return census_df


def validate_census_data(df: pl.DataFrame) -> dict[str, Any]:
    """Validate census data quality.

    Returns:
        Dict with validation metrics.

    """
    # Percent columns are conventionally named with "pct_" or explicitly "urban_percent"
    pct_cols = [c for c in df.columns if c.startswith("pct_")] + (
        ["urban_percent"] if "urban_percent" in df.columns else []
    )

    # Out-of-bounds checks for percent columns
    pct_oob: dict[str, dict[str, int]] = {}
    for c in pct_cols:
        stats = df.select([
            (pl.col(c) < 0).sum().alias("lt0"),
            (pl.col(c) > 100).sum().alias("gt100"),
        ]).to_dicts()[0]
        if stats["lt0"] or stats["gt100"]:
            pct_oob[c] = {"lt0": int(stats["lt0"]), "gt100": int(stats["gt100"])}

    return {
        "total_block_groups": df.height,
        "unique_geoids": int(df["GEOID"].n_unique()) if "GEOID" in df.columns else None,
        "columns": len(df.columns),
        "null_counts": {col: int(df[col].null_count()) for col in df.columns if df[col].null_count() > 0},
        "geoid_length_correct": bool((df["GEOID"].str.len_chars() == 12).all()) if "GEOID" in df.columns else None,
        "pct_out_of_bounds": pct_oob,
    }
