# smart_meter_analysis/census_specs.py
"""
Census variable registry + stable Stage 2 predictor list.

- VARIABLE_SPECS is the spec-driven registry used by census.py to decide which ACS/Decennial
  variables to request and how to engineer features.
- STAGE2_PREDICTORS_47 is the *stable* final predictor list used by Stage 2 regression.

Note: Some predictors in STAGE2_PREDICTORS_47 may be engineered composites (e.g., multifamily_10_plus)
and therefore may not correspond 1:1 to a single ACS code.
"""

from __future__ import annotations

from typing import Any

# -----------------------------------------------------------------------------
# Stable Stage 2 predictor list (exactly what your run log shows as "Using 47 predictors")
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
# Spec registry used by census.py for fetching/engineering.
# -----------------------------------------------------------------------------

VARIABLE_SPECS: list[dict[str, Any]] = [
    # --- Spatial ---
    {"name": "urban_percent", "category": "spatial", "source": "acs"},
    # --- Economic ---
    {"name": "median_household_income", "category": "economic", "source": "acs"},
    {"name": "unemployment_rate", "category": "economic", "source": "acs"},
    {"name": "pct_in_civilian_labor_force", "category": "economic", "source": "acs"},
    {"name": "pct_not_in_labor_force", "category": "economic", "source": "acs"},
    {"name": "pct_income_under_25k", "category": "economic", "source": "acs"},
    {"name": "pct_income_25k_to_75k", "category": "economic", "source": "acs"},
    {"name": "pct_income_75k_plus", "category": "economic", "source": "acs"},
    # --- Housing ---
    {"name": "pct_owner_occupied", "category": "housing", "source": "acs"},
    {"name": "pct_renter_occupied", "category": "housing", "source": "acs"},
    {"name": "pct_heat_utility_gas", "category": "housing", "source": "acs"},
    {"name": "pct_heat_electric", "category": "housing", "source": "acs"},
    {"name": "pct_housing_built_2000_plus", "category": "housing", "source": "acs"},
    {"name": "pct_housing_built_1980_1999", "category": "housing", "source": "acs"},
    {"name": "old_building_pct", "category": "housing", "source": "acs"},
    {"name": "pct_structure_single_family_detached", "category": "housing", "source": "acs"},
    {"name": "pct_structure_single_family_attached", "category": "housing", "source": "acs"},
    {"name": "pct_structure_multifamily_2_to_4", "category": "housing", "source": "acs"},
    {"name": "pct_structure_multifamily_5_to_19", "category": "housing", "source": "acs"},
    {"name": "pct_structure_multifamily_20_plus", "category": "housing", "source": "acs"},
    {"name": "pct_structure_multifamily_10_plus", "category": "housing", "source": "acs"},
    {"name": "pct_structure_mobile_home", "category": "housing", "source": "acs"},
    {"name": "pct_vacant_housing_units", "category": "housing", "source": "acs"},
    {"name": "pct_home_value_under_150k", "category": "housing", "source": "acs"},
    {"name": "pct_home_value_150k_to_299k", "category": "housing", "source": "acs"},
    {"name": "pct_home_value_300k_plus", "category": "housing", "source": "acs"},
    {"name": "pct_rent_burden_30_plus", "category": "housing", "source": "acs"},
    {"name": "pct_rent_burden_50_plus", "category": "housing", "source": "acs"},
    {"name": "pct_owner_cost_burden_30_plus_mortgage", "category": "housing", "source": "acs"},
    {"name": "pct_owner_cost_burden_50_plus_mortgage", "category": "housing", "source": "acs"},
    {"name": "pct_owner_overcrowded_2plus_per_room", "category": "housing", "source": "acs"},
    {"name": "pct_renter_overcrowded_2plus_per_room", "category": "housing", "source": "acs"},
    # --- Household ---
    {"name": "avg_household_size", "category": "household", "source": "acs"},
    {"name": "avg_family_size", "category": "household", "source": "acs"},
    {"name": "pct_single_parent_households", "category": "household", "source": "acs"},
    # --- Demographic ---
    {"name": "median_age", "category": "demographic", "source": "acs"},
    {"name": "pct_white_alone", "category": "demographic", "source": "acs"},
    {"name": "pct_black_alone", "category": "demographic", "source": "acs"},
    {"name": "pct_asian_alone", "category": "demographic", "source": "acs"},
    {"name": "pct_two_or_more_races", "category": "demographic", "source": "acs"},
    {"name": "pct_population_under_5", "category": "demographic", "source": "acs"},
    {"name": "pct_population_5_to_17", "category": "demographic", "source": "acs"},
    {"name": "pct_population_18_to_24", "category": "demographic", "source": "acs"},
    {"name": "pct_population_25_to_44", "category": "demographic", "source": "acs"},
    {"name": "pct_population_45_to_64", "category": "demographic", "source": "acs"},
    {"name": "pct_population_65_plus", "category": "demographic", "source": "acs"},
    {"name": "pct_female", "category": "demographic", "source": "acs"},
]

# Optional convenience: expose the expected count (useful for asserts/logging)
EXPECTED_STAGE2_PREDICTOR_COUNT = len(STAGE2_PREDICTORS_47)
