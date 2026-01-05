# smart_meter_analysis/census_specs.py
"""Census variable registry + stable Stage 2 predictor list.

- VARIABLE_SPECS is the spec-driven registry used by census.py to decide which ACS/Decennial
  variables to request and how to engineer features.
- STAGE2_PREDICTORS_47 is the *stable* final predictor list used by Stage 2 regression.
"""

from __future__ import annotations

from typing import Any

# -----------------------------------------------------------------------------
# Stable Stage 2 predictor list (47 predictors)
# -----------------------------------------------------------------------------
STAGE2_PREDICTORS_47: list[str] = [
    "avg_family_size",
    "avg_household_size",
    "median_age",
    "median_household_income",
    "old_building_pct",
    "pct_asian_alone",
    "pct_black_alone",
    "pct_female",
    "pct_heat_electric",
    "pct_heat_utility_gas",
    "pct_home_value_150k_to_299k",
    "pct_home_value_300k_plus",
    "pct_home_value_under_150k",
    "pct_housing_built_1980_1999",
    "pct_housing_built_2000_plus",
    "pct_in_civilian_labor_force",
    "pct_income_25k_to_75k",
    "pct_income_75k_plus",
    "pct_income_under_25k",
    "pct_not_in_labor_force",
    "pct_owner_cost_burden_30_plus_mortgage",
    "pct_owner_cost_burden_50_plus_mortgage",
    "pct_owner_occupied",
    "pct_owner_overcrowded_2plus_per_room",
    "pct_population_18_to_24",
    "pct_population_25_to_44",
    "pct_population_45_to_64",
    "pct_population_5_to_17",
    "pct_population_65_plus",
    "pct_population_under_5",
    "pct_rent_burden_30_plus",
    "pct_rent_burden_50_plus",
    "pct_renter_occupied",
    "pct_renter_overcrowded_2plus_per_room",
    "pct_single_parent_households",
    "pct_structure_mobile_home",
    "pct_structure_multifamily_10_plus",
    "pct_structure_multifamily_20_plus",
    "pct_structure_multifamily_2_to_4",
    "pct_structure_multifamily_5_to_19",
    "pct_structure_single_family_attached",
    "pct_structure_single_family_detached",
    "pct_two_or_more_races",
    "pct_vacant_housing_units",
    "pct_white_alone",
    "unemployment_rate",
    "urban_percent",
]

# -----------------------------------------------------------------------------
# Complete spec registry with ACS variable codes
# -----------------------------------------------------------------------------
VARIABLE_SPECS: list[dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # ECONOMIC (8 variables)
    # -------------------------------------------------------------------------
    {
        "name": "median_household_income",
        "category": "economic",
        "source": "acs",
        "variable": "B19013_001E",
        "transformation": "none",
    },
    {
        "name": "unemployment_rate",
        "category": "economic",
        "source": "acs",
        "numerator": "B23025_005E",  # Unemployed
        "denominator": "B23025_003E",  # In civilian labor force
        "transformation": "none",
    },
    {
        "name": "pct_in_civilian_labor_force",
        "category": "economic",
        "source": "acs",
        "numerator": "B23025_003E",  # In civilian labor force
        "denominator": "B23025_002E",  # Population 16+ in labor force
        "transformation": "none",
    },
    {
        "name": "pct_not_in_labor_force",
        "category": "economic",
        "source": "acs",
        "numerator": "B23025_007E",  # Not in labor force
        "denominator": "B23025_001E",  # Population 16+
        "transformation": "none",
    },
    {
        "name": "pct_income_under_25k",
        "category": "economic",
        "source": "acs",
        "numerator": "B19001_002E + B19001_003E + B19001_004E + B19001_005E",
        "denominator": "B19001_001E",
        "transformation": "none",
    },
    {
        "name": "pct_income_25k_to_75k",
        "category": "economic",
        "source": "acs",
        "numerator": "B19001_006E + B19001_007E + B19001_008E + B19001_009E + B19001_010E + B19001_011E + B19001_012E",
        "denominator": "B19001_001E",
        "transformation": "none",
    },
    {
        "name": "pct_income_75k_plus",
        "category": "economic",
        "source": "acs",
        "numerator": "B19001_013E + B19001_014E + B19001_015E + B19001_016E + B19001_017E",
        "denominator": "B19001_001E",
        "transformation": "none",
    },
    # -------------------------------------------------------------------------
    # HOUSING (25 variables)
    # -------------------------------------------------------------------------
    {
        "name": "pct_owner_occupied",
        "category": "housing",
        "source": "acs",
        "numerator": "B25003_002E",  # Owner occupied
        "denominator": "B25003_001E",  # Total occupied
        "transformation": "none",
    },
    {
        "name": "pct_renter_occupied",
        "category": "housing",
        "source": "acs",
        "numerator": "B25003_003E",  # Renter occupied
        "denominator": "B25003_001E",  # Total occupied
        "transformation": "none",
    },
    {
        "name": "pct_vacant_housing_units",
        "category": "housing",
        "source": "acs",
        "numerator": "B25002_003E",  # Vacant
        "denominator": "B25002_001E",  # Total housing units
        "transformation": "none",
    },
    {
        "name": "pct_heat_utility_gas",
        "category": "housing",
        "source": "acs",
        "numerator": "B25040_002E",  # Utility gas
        "denominator": "B25040_001E",  # Total occupied units
        "transformation": "none",
    },
    {
        "name": "pct_heat_electric",
        "category": "housing",
        "source": "acs",
        "numerator": "B25040_005E",  # Electricity
        "denominator": "B25040_001E",  # Total occupied units
        "transformation": "none",
    },
    {
        "name": "pct_housing_built_2000_plus",
        "category": "housing",
        "source": "acs",
        "numerator": "B25034_002E + B25034_003E + B25034_004E + B25034_005E",  # 2000-2009, 2010-2013, 2014-2017, 2018-2020+
        "denominator": "B25034_001E",
        "transformation": "none",
    },
    {
        "name": "pct_housing_built_1980_1999",
        "category": "housing",
        "source": "acs",
        "numerator": "B25034_006E + B25034_007E",  # 1990-1999, 1980-1989
        "denominator": "B25034_001E",
        "transformation": "none",
    },
    {
        "name": "old_building_pct",
        "category": "housing",
        "source": "acs",
        "numerator": "B25034_008E + B25034_009E + B25034_010E + B25034_011E",  # Pre-1980
        "denominator": "B25034_001E",
        "transformation": "none",
    },
    {
        "name": "pct_structure_single_family_detached",
        "category": "housing",
        "source": "acs",
        "numerator": "B25024_002E",  # 1-unit detached
        "denominator": "B25024_001E",
        "transformation": "none",
    },
    {
        "name": "pct_structure_single_family_attached",
        "category": "housing",
        "source": "acs",
        "numerator": "B25024_003E",  # 1-unit attached
        "denominator": "B25024_001E",
        "transformation": "none",
    },
    {
        "name": "pct_structure_multifamily_2_to_4",
        "category": "housing",
        "source": "acs",
        "numerator": "B25024_004E + B25024_005E",  # 2 units + 3-4 units
        "denominator": "B25024_001E",
        "transformation": "none",
    },
    {
        "name": "pct_structure_multifamily_5_to_19",
        "category": "housing",
        "source": "acs",
        "numerator": "B25024_006E + B25024_007E",  # 5-9 units + 10-19 units
        "denominator": "B25024_001E",
        "transformation": "none",
    },
    {
        "name": "pct_structure_multifamily_20_plus",
        "category": "housing",
        "source": "acs",
        "numerator": "B25024_008E + B25024_009E",  # 20-49 units + 50+ units
        "denominator": "B25024_001E",
        "transformation": "none",
    },
    {
        "name": "pct_structure_multifamily_10_plus",
        "category": "housing",
        "source": "acs",
        "numerator": "B25024_007E + B25024_008E + B25024_009E",  # 10-19, 20-49, 50+
        "denominator": "B25024_001E",
        "transformation": "none",
    },
    {
        "name": "pct_structure_mobile_home",
        "category": "housing",
        "source": "acs",
        "numerator": "B25024_010E",  # Mobile home
        "denominator": "B25024_001E",
        "transformation": "none",
    },
    {
        "name": "pct_home_value_under_150k",
        "category": "housing",
        "source": "acs",
        "numerator": "B25075_002E + B25075_003E + B25075_004E + B25075_005E + B25075_006E + B25075_007E + B25075_008E + B25075_009E + B25075_010E + B25075_011E + B25075_012E + B25075_013E + B25075_014E",
        "denominator": "B25075_001E",
        "transformation": "none",
    },
    {
        "name": "pct_home_value_150k_to_299k",
        "category": "housing",
        "source": "acs",
        "numerator": "B25075_015E + B25075_016E + B25075_017E + B25075_018E + B25075_019E",
        "denominator": "B25075_001E",
        "transformation": "none",
    },
    {
        "name": "pct_home_value_300k_plus",
        "category": "housing",
        "source": "acs",
        "numerator": "B25075_020E + B25075_021E + B25075_022E + B25075_023E + B25075_024E + B25075_025E + B25075_026E + B25075_027E",
        "denominator": "B25075_001E",
        "transformation": "none",
    },
    {
        "name": "pct_rent_burden_30_plus",
        "category": "housing",
        "source": "acs",
        "numerator": "B25070_007E + B25070_008E + B25070_009E + B25070_010E",  # 30-34.9%, 35-39.9%, 40-49.9%, 50%+
        "denominator": "B25070_001E",
        "transformation": "none",
    },
    {
        "name": "pct_rent_burden_50_plus",
        "category": "housing",
        "source": "acs",
        "numerator": "B25070_010E",  # 50%+
        "denominator": "B25070_001E",
        "transformation": "none",
    },
    {
        "name": "pct_owner_cost_burden_30_plus_mortgage",
        "category": "housing",
        "source": "acs",
        "numerator": "B25091_008E + B25091_009E + B25091_010E + B25091_011E",  # With mortgage: 30-34.9%, 35-39.9%, 40-49.9%, 50%+
        "denominator": "B25091_002E",  # With mortgage total
        "transformation": "none",
    },
    {
        "name": "pct_owner_cost_burden_50_plus_mortgage",
        "category": "housing",
        "source": "acs",
        "numerator": "B25091_011E",  # With mortgage: 50%+
        "denominator": "B25091_002E",
        "transformation": "none",
    },
    {
        "name": "pct_owner_overcrowded_2plus_per_room",
        "category": "housing",
        "source": "acs",
        "numerator": "B25014_007E",  # Owner: 2+ persons per room
        "denominator": "B25014_002E",  # Owner total
        "transformation": "none",
    },
    {
        "name": "pct_renter_overcrowded_2plus_per_room",
        "category": "housing",
        "source": "acs",
        "numerator": "B25014_013E",  # Renter: 2+ persons per room
        "denominator": "B25014_008E",  # Renter total
        "transformation": "none",
    },
    # -------------------------------------------------------------------------
    # HOUSEHOLD (3 variables)
    # -------------------------------------------------------------------------
    {
        "name": "avg_household_size",
        "category": "household",
        "source": "acs",
        "variable": "B25010_001E",
        "transformation": "none",
    },
    {
        "name": "avg_family_size",
        "category": "household",
        "source": "acs",
        "variable": "B25010_002E",
        "transformation": "none",
    },
    {
        "name": "pct_single_parent_households",
        "category": "household",
        "source": "acs",
        "numerator": "B11001_006E + B11001_007E",  # Male householder + Female householder, no spouse
        "denominator": "B11001_001E",  # Total households
        "transformation": "none",
    },
    # -------------------------------------------------------------------------
    # DEMOGRAPHIC (10 variables)
    # -------------------------------------------------------------------------
    {
        "name": "median_age",
        "category": "demographic",
        "source": "acs",
        "variable": "B01002_001E",
        "transformation": "none",
    },
    {
        "name": "pct_female",
        "category": "demographic",
        "source": "acs",
        "numerator": "B01001_026E",  # Female total
        "denominator": "B01001_001E",  # Total population
        "transformation": "none",
    },
    {
        "name": "pct_white_alone",
        "category": "demographic",
        "source": "acs",
        "numerator": "B03002_003E",  # White alone, not Hispanic/Latino
        "denominator": "B03002_001E",
        "transformation": "none",
    },
    {
        "name": "pct_black_alone",
        "category": "demographic",
        "source": "acs",
        "numerator": "B03002_004E",  # Black alone, not Hispanic/Latino
        "denominator": "B03002_001E",
        "transformation": "none",
    },
    {
        "name": "pct_asian_alone",
        "category": "demographic",
        "source": "acs",
        "numerator": "B03002_006E",  # Asian alone, not Hispanic/Latino
        "denominator": "B03002_001E",
        "transformation": "none",
    },
    {
        "name": "pct_two_or_more_races",
        "category": "demographic",
        "source": "acs",
        "numerator": "B03002_009E",  # Two or more races, not Hispanic/Latino
        "denominator": "B03002_001E",
        "transformation": "none",
    },
    {
        "name": "pct_population_under_5",
        "category": "demographic",
        "source": "acs",
        "numerator": "B01001_003E + B01001_027E",  # Male + Female under 5
        "denominator": "B01001_001E",
        "transformation": "none",
    },
    {
        "name": "pct_population_5_to_17",
        "category": "demographic",
        "source": "acs",
        "numerator": "B01001_004E + B01001_005E + B01001_006E + B01001_028E + B01001_029E + B01001_030E",
        "denominator": "B01001_001E",
        "transformation": "none",
    },
    {
        "name": "pct_population_18_to_24",
        "category": "demographic",
        "source": "acs",
        "numerator": "B01001_007E + B01001_008E + B01001_009E + B01001_010E + B01001_031E + B01001_032E + B01001_033E + B01001_034E",
        "denominator": "B01001_001E",
        "transformation": "none",
    },
    {
        "name": "pct_population_25_to_44",
        "category": "demographic",
        "source": "acs",
        "numerator": "B01001_011E + B01001_012E + B01001_013E + B01001_014E + B01001_035E + B01001_036E + B01001_037E + B01001_038E",
        "denominator": "B01001_001E",
        "transformation": "none",
    },
    {
        "name": "pct_population_45_to_64",
        "category": "demographic",
        "source": "acs",
        "numerator": "B01001_015E + B01001_016E + B01001_017E + B01001_018E + B01001_019E + B01001_039E + B01001_040E + B01001_041E + B01001_042E + B01001_043E",
        "denominator": "B01001_001E",
        "transformation": "none",
    },
    {
        "name": "pct_population_65_plus",
        "category": "demographic",
        "source": "acs",
        "numerator": "B01001_020E + B01001_021E + B01001_022E + B01001_023E + B01001_024E + B01001_025E + B01001_044E + B01001_045E + B01001_046E + B01001_047E + B01001_048E + B01001_049E",
        "denominator": "B01001_001E",
        "transformation": "none",
    },
    # -------------------------------------------------------------------------
    # SPATIAL (1 variable - from Decennial, not ACS)
    # -------------------------------------------------------------------------
    {
        "name": "urban_percent",
        "category": "spatial",
        "source": "decennial",
        "numerator": "H2_002N",  # Urban housing units
        "denominator": "H2_002N + H2_003N",  # Total (urban + rural)
        "transformation": "none",
    },
]

# Optional convenience: expose the expected count
EXPECTED_STAGE2_PREDICTOR_COUNT = len(STAGE2_PREDICTORS_47)
