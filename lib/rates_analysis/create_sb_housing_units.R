library(tidyverse)
library(scales)
library(viridis)
library(ggplot2)
library(patchwork)
library(lubridate)
library(arrow)


########################################################
# Constants and Conversions
########################################################
# Gas
# https://www.rienergy.com/site/-/media/rie-jss-app/home/ways-to-save/rates-and-shopping/service-rates/residential-rates/Rates-5-01-25.ashx?sc_lang=en&hash=EB0DA36E10360398E10E6365C5AD7FBB
convert_gas_therm_to_mmbtu <- 1
convert_gas_therm_to_ccf <- 1 / 1.028
convert_gas_ccf_to_therm <- 1.028
convert_gas_therm_to_kwh <- 29.3

# Delivered Fuels
convert_gal_fuel_oil_to_kwh <- 40.2778
convert_propane_gal_to_kwh <- 27


########################################################
# Misc helper Functions
########################################################
print_all_column_names <- function(table) {
  print(colnames(table))
}

########################################################
# Preferred Labels
########################################################
add_baseline_heating_type <- function(housing_units) {
  housing_units <- housing_units |>
    mutate(
      baseline_heating_type = case_when(
        `in.hvac_heating_type_and_fuel` == "Electricity ASHP" ~ "Heat Pump",
        `in.hvac_heating_type_and_fuel` ==
          "Electricity Baseboard" ~ "Electric Resistance",
        `in.hvac_heating_type_and_fuel` ==
          "Electricity Electric Boiler" ~ "Electric Resistance",
        `in.hvac_heating_type_and_fuel` ==
          "Electricity Electric Furnace" ~ "Electric Resistance",
        `in.hvac_heating_type_and_fuel` ==
          "Electricity Electric Wall Furnace" ~ "Electric Resistance",
        `in.hvac_heating_type_and_fuel` == "Electricity MSHP" ~ "Heat Pump",
        `in.hvac_heating_type_and_fuel` ==
          "Electricity Shared Heating" ~ "Electric Resistance",
        `in.hvac_heating_type_and_fuel` == "Fuel Oil Fuel Boiler" ~ "Fuel Oil",
        `in.hvac_heating_type_and_fuel` == "Fuel Oil Fuel Furnace" ~ "Fuel Oil",
        `in.hvac_heating_type_and_fuel` ==
          "Fuel Oil Fuel Wall/Floor Furnace" ~ "Fuel Oil",
        `in.hvac_heating_type_and_fuel` ==
          "Fuel Oil Shared Heating" ~ "Fuel Oil",
        `in.hvac_heating_type_and_fuel` ==
          "Natural Gas Fuel Boiler" ~ "Natural Gas",
        `in.hvac_heating_type_and_fuel` ==
          "Natural Gas Fuel Furnace" ~ "Natural Gas",
        `in.hvac_heating_type_and_fuel` ==
          "Natural Gas Fuel Wall/Floor Furnace" ~ "Natural Gas",
        `in.hvac_heating_type_and_fuel` ==
          "Natural Gas Shared Heating" ~ "Natural Gas",
        `in.hvac_heating_type_and_fuel` == "None" ~ "Other/None",
        `in.hvac_heating_type_and_fuel` ==
          "Other Fuel Fuel Boiler" ~ "Other/None",
        `in.hvac_heating_type_and_fuel` ==
          "Other Fuel Fuel Furnace" ~ "Other/None",
        `in.hvac_heating_type_and_fuel` ==
          "Other Fuel Fuel Wall/Floor Furnace" ~ "Other/None",
        `in.hvac_heating_type_and_fuel` ==
          "Other Fuel Shared Heating" ~ "Other/None",
        `in.hvac_heating_type_and_fuel` == "Propane Fuel Boiler" ~ "Propane",
        `in.hvac_heating_type_and_fuel` == "Propane Fuel Furnace" ~ "Propane",
        `in.hvac_heating_type_and_fuel` ==
          "Propane Fuel Wall/Floor Furnace" ~ "Propane",
        `in.hvac_heating_type_and_fuel` == "Propane Shared Heating" ~ "Propane",
        TRUE ~ "Other/None"
      )
    ) |>
    select(-`in.hvac_heating_type_and_fuel`)
  return(housing_units)
}

add_baseline_cooling_type <- function(housing_units) {
  housing_units <- housing_units |>
    mutate(
      baseline_cooling_type = case_when(
        `in.hvac_cooling_type` == "Central AC" ~ "Central AC",
        `in.hvac_cooling_type` == "Room AC" ~ "Room AC",
        `in.hvac_cooling_type` == "Ducted Heat Pump" ~ "Heat Pump",
        `in.hvac_cooling_type` == "Non-ducted Heat Pump" ~ "Heat Pump",
        `in.hvac_cooling_type` == "None" ~ "None",
        TRUE ~ "Other Cooling"
      )
    ) |>
    select(-`in.hvac_cooling_type`)
  return(housing_units)
}

add_hvac_appliances_shell <- function(housing_units) {
  housing_units <- housing_units |>
    mutate(
      hvac = case_when(
        upgrade == 0 ~ baseline_heating_type,
        upgrade == 1 ~ "hp_low",
        upgrade == 2 ~ "hp_high",
        upgrade == 3 ~ "hp_best",
        upgrade == 4 ~ "hp_low",
        upgrade == 5 ~ "hp_geo",
        upgrade == 6 ~ "hp_low",
        upgrade == 7 ~ "hp_high",
        upgrade == 8 ~ "hp_best",
        upgrade == 9 ~ "hp_high",
        upgrade == 10 ~ "hp_geo",
        upgrade == 11 ~ "hp_low",
        upgrade == 12 ~ "hp_high",
        upgrade == 13 ~ "hp_best",
        upgrade == 14 ~ "hp_low",
        upgrade == 15 ~ "hp_geo",
        upgrade == 16 ~ baseline_heating_type,
        TRUE ~ "missed_hvac"
      )
    ) |>
    mutate(
      hvac_backup = case_when(
        upgrade == 0 ~ baseline_heating_type,
        upgrade == 1 ~ "electric_resistance",
        upgrade == 2 ~ "electric_resistance",
        upgrade == 3 ~ "electric_resistance",
        upgrade == 4 ~ baseline_heating_type,
        upgrade == 5 ~ "none",
        upgrade == 6 ~ "electric_resistance",
        upgrade == 7 ~ "electric_resistance",
        upgrade == 8 ~ "electric_resistance",
        upgrade == 9 ~ baseline_heating_type,
        upgrade == 10 ~ "none",
        upgrade == 11 ~ "electric_resistance",
        upgrade == 12 ~ "electric_resistance",
        upgrade == 13 ~ "electric_resistance",
        upgrade == 14 ~ baseline_heating_type,
        upgrade == 15 ~ "none",
        upgrade == 16 ~ baseline_heating_type,
        TRUE ~ "missed_hvac_backup"
      )
    ) |>
    mutate(
      shell = case_when(
        upgrade == 0 ~ "baseline",
        upgrade == 1 ~ "baseline",
        upgrade == 2 ~ "baseline",
        upgrade == 3 ~ "baseline",
        upgrade == 4 ~ "baseline",
        upgrade == 5 ~ "baseline",
        upgrade == 6 ~ "light_touch",
        upgrade == 7 ~ "light_touch",
        upgrade == 8 ~ "light_touch",
        upgrade == 9 ~ "light_touch",
        upgrade == 10 ~ "light_touch",
        upgrade == 11 ~ "light_touch",
        upgrade == 12 ~ "light_touch",
        upgrade == 13 ~ "light_touch",
        upgrade == 14 ~ "light_touch",
        upgrade == 15 ~ "light_touch",
        upgrade == 16 ~ "light_touch",
        TRUE ~ "missed_shell"
      )
    ) |>
    mutate(
      appliances = case_when(
        upgrade == 0 ~ "baseline",
        upgrade == 1 ~ "baseline",
        upgrade == 2 ~ "baseline",
        upgrade == 3 ~ "baseline",
        upgrade == 4 ~ "baseline",
        upgrade == 5 ~ "baseline",
        upgrade == 6 ~ "baseline",
        upgrade == 7 ~ "baseline",
        upgrade == 8 ~ "baseline",
        upgrade == 9 ~ "baseline",
        upgrade == 10 ~ "baseline",
        upgrade == 11 ~ "all_electric",
        upgrade == 12 ~ "all_electric",
        upgrade == 13 ~ "all_electric",
        upgrade == 14 ~ "all_electric",
        upgrade == 15 ~ "all_electric",
        upgrade == 16 ~ "baseline",
        TRUE ~ "missed_appliances"
      )
    )
  return(housing_units)
}


########################################################
# Preferred Groupings (building type, etc)
########################################################

# Building Type
update_building_type_group <- function(housing_units) {
  housing_units <- housing_units |>
    mutate(
      building_type_group = case_when(
        `in.geometry_building_type_acs` %in%
          c(
            "Single-Family Detached",
            "Mobile Home",
            "Single-Family Attached"
          ) ~ "Single-Family",
        `in.geometry_building_type_acs` %in%
          c("2 Unit", "3 or 4 Unit") ~ "2-4 Units",
        `in.geometry_building_type_acs` %in%
          c(
            "5 to 9 Unit",
            "10 to 19 Unit",
            "20 to 49 Unit",
            "50 or more Unit"
          ) ~ "5+ Units",
        TRUE ~ "Other"
      )
    ) |>
    select(-`in.geometry_building_type_acs`)

  return(housing_units)
}

# Occupants
change_occupants_to_number <- function(housing_units) {
  housing_units <- housing_units |>
    mutate(
      occupants = case_when(
        in.occupants == "10+" ~ 10,
        .default = as.numeric(in.occupants)
      )
    ) |>
    select(-in.occupants)
  return(housing_units)
}

add_occupants_group <- function(housing_units) {
  housing_units <- housing_units |>
    mutate(
      occupants_group = case_when(
        occupants == 0 ~ "Vacant",
        occupants == 1 ~ "Single",
        occupants == 2 ~ "Couple",
        occupants == 3 ~ "3-4 Occupants",
        occupants == 4 ~ "3-4 Occupants",
        occupants == 5 ~ "5+ Occupants",
        occupants == 6 ~ "5+ Occupants",
        occupants == 7 ~ "5+ Occupants",
        occupants == 8 ~ "5+ Occupants",
        occupants == 9 ~ "5+ Occupants",
        occupants == 10 ~ "5+ Occupants",
        TRUE ~ "Other"
      )
    )
  return(housing_units)
}


########################################################
# Income and LMI Discounts
########################################################
inflate_income_to_2024 <- function(housing_units, from_year = 2018) {
  # load inflation factors

  inflation_adj_factors <- readRDS(
    "/workspaces/reports2/data/fred/inflation_factors.rds"
  )

  inflation_factor <- inflation_adj_factors$inflation_factor[
    inflation_adj_factors$year == from_year
  ]

  # inflation adjustment based on the Employment Cost Index
  housing_units <- housing_units |>
    mutate(
      in.representative_income = in.representative_income * inflation_factor
    )
  return(housing_units)
}


group_income_by_smi <- function(
  housing_units,
  table_name,
  url_smi_thresholds,
  smi_tiers = c(0.6, 0.8)
) {
  library(googlesheets4)
  googlesheets4::gs4_deauth()
  smi_thresholds <- googlesheets4::read_sheet(
    url_smi_thresholds,
    sheet = table_name
  )
  smi_thresholds <- smi_thresholds |>
    select(-source, -note)

  housing_units <- housing_units |>
    left_join(
      smi_thresholds,
      by = join_by(
        `occupants` == occupants_min
      )
    ) |>

    # calculate percent of SMI
    mutate(percent_of_smi = `in.representative_income` / smi) |>
    select(-smi) |>

    # group into SMI tiers
    mutate(
      smi_tier = case_when(
        percent_of_smi < 0.01 ~ "No Income",
        percent_of_smi >= 0.01 & percent_of_smi < smi_tiers[1] ~ "Low Income",
        percent_of_smi >= smi_tiers[1] &
          percent_of_smi < smi_tiers[2] ~ "Moderate Income",
        percent_of_smi >= smi_tiers[2] ~ "Not LMI",
        TRUE ~ "Not LMI"
      )
    )
  return(housing_units)
}

group_income_by_dollars <- function(
  housing_units,
  dollar_tiers = c(1000, 55374, 92290)
) {
  housing_units <- housing_units |>
    mutate(
      dollar_tier = case_when(
        `in.representative_income` < dollar_tiers[1] ~ "No Income",
        `in.representative_income` >= dollar_tiers[1] &
          `in.representative_income` < dollar_tiers[2] ~ "Low Income",
        `in.representative_income` >= dollar_tiers[2] &
          `in.representative_income` < dollar_tiers[3] ~ "Moderate Income",
        `in.representative_income` >= dollar_tiers[3] ~ "Not LMI",
        TRUE ~ "Not LMI"
      )
    )
  return(housing_units)
}


add_liheap_eligibility <- function(
  housing_units,
  electric_lmi_thresholds,
  gas_lmi_thresholds
) {
  housing_units <- housing_units |>
    mutate(lmi = (discount_rate_elec > 0 | `discount_rate_gas` > 0))
  return(housing_units)
}

add_lmi_discount <- function(
  housing_units,
  electric_lmi_thresholds,
  gas_lmi_thresholds
) {
  housing_units_with_electric <- housing_units |>
    left_join(
      electric_lmi_thresholds,
      by = join_by(
        electric_utility,
        `in.representative_income` >= income_threshold_lower,
        `in.representative_income` < income_threshold_upper,
        `occupants` == occupants_min
      ),
      suffix = c("", "_electric")
    ) |>
    # Create electric_discount_rate column, use 0 if no match
    mutate(discount_rate_elec = coalesce(discount_rate, 0)) |>
    # Remove the intermediate discount_rate column from the join
    select(
      -discount_rate,
      -customer_class,
      -income_threshold_lower,
      -income_threshold_upper
    )

  housing_units_with_both <- housing_units_with_electric |>
    left_join(
      gas_lmi_thresholds,
      by = join_by(
        electric_utility, # Both threshold tables use electric_utility as the key
        `in.representative_income` >= income_threshold_lower,
        `in.representative_income` <= income_threshold_upper,
        `occupants` == occupants_min
      ),
      suffix = c("", "_gas")
    ) |>
    # Create gas_discount_rate column (note: user requested "gas_discount-rate" with hyphen)
    mutate(`discount_rate_gas` = coalesce(discount_rate, 0)) |>
    # Remove the intermediate discount_rate column from the join
    select(
      -discount_rate,
      -customer_class,
      -income_threshold_lower,
      -income_threshold_upper
    )

  # Finally, create the lmi column
  housing_units_final <- housing_units_with_both |>
    mutate(
      lmi = (discount_rate_elec > 0 | `discount_rate_gas` > 0)
    )

  return(housing_units_final)
}


########################################################
# Utility Assignment
########################################################
assign_utilities <- function(
  housing_units,
  path_to_bldg_utility_crosswalk = NULL
) {
  #' Assign electricity utility to housing units
  #'
  #' @param housing_units A dataframe containing housing units
  #' @param path_to_bldg_utility_crosswalk A path to a CSV file containing a mapping of building IDs to electricity utilities
  #' @return A dataframe containing housing units with electricity utilities assigned
  #' @examples
  #' assign_electricity_utility(housing_units, path_to_bldg_utility_crosswalk = "/workspaces/reports2/data/resstock/2024_resstock_amy2018_release_2/rs_2024_bldg_utility_crosswalk.csv")

  if (is.null(path_to_bldg_utility_crosswalk)) {
    path_to_bldg_utility_crosswalk <- "/workspaces/reports2/data/resstock/2024_release2_tmy3/metadata/rs2024_bldg_utility_crosswalk.feather"
  }

  if (!file.exists(path_to_bldg_utility_crosswalk)) {
    #make_empty_utility_crosswalk(path_to_rs2024_metadata)
    stop(
      "Utility crosswalk file does not exist. Please run /data/resstock/just/make_empty_utility_crosswalk_2024() to create it."
    )
  }

  bldg_utility_mapping <- read_feather(path_to_bldg_utility_crosswalk)

  housing_units <- housing_units |>
    left_join(
      bldg_utility_mapping,
      by = c("bldg_id", "in.state")
    )
  return(housing_units)
}
