import pytest

pytest.skip("smart_meter_analysis.census module removed during repo restructure", allow_module_level=True)

"""Tests for Census data module"""

import polars as pl

from smart_meter_analysis.census import (
    build_geoid,
    fetch_census_data,
    validate_census_data,
)


def test_build_geoid():
    """Test GEOID construction for block groups"""
    df = pl.DataFrame({
        "state": ["17"],
        "county": ["031"],
        "tract": ["839102"],
        "block group": ["1"],
    })

    result = build_geoid(df)
    assert "GEOID" in result.columns
    assert result["GEOID"][0] == "170318391021"
    assert len(result["GEOID"][0]) == 12


@pytest.mark.slow
def test_fetch_census_data_small():
    """Test fetching census data (limited to Cook County for speed)"""
    # This will be slow, so we'll skip in normal runs
    pytest.skip("Skipping slow Census API test")

    df = fetch_census_data(state_fips="17")

    # Basic checks
    assert df.height > 0
    assert "GEOID" in df.columns
    assert "Median_Household_Income" in df.columns
    assert "Urban_Rural_Classification" in df.columns

    # Validate GEOIDs are 12 digits
    assert all(df["GEOID"].str.len_chars() == 12)


def test_validate_census_data():
    """Test census data validation"""
    # Create sample data
    df = pl.DataFrame({
        "GEOID": ["170318391021", "170318391022"],
        "NAME": ["Block Group 1", "Block Group 2"],
        "Median_Household_Income": [75000.0, None],
        "Total_Households": [250, 180],
    })

    validation = validate_census_data(df)

    assert validation["total_block_groups"] == 2
    assert validation["unique_geoids"] == 2
    assert validation["geoid_length_correct"] is True
    assert "Median_Household_Income" in validation["null_counts"]


def test_imports():
    """Test that census module can be imported"""
    from smart_meter_analysis import census

    assert census is not None
